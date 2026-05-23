import argparse
import io
import json
import os
import random
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def split_patches(img: Image.Image, patch_size: int):
    img = img.convert("RGB").resize((1500, 1500), Image.BICUBIC)
    cell = 1500 // patch_size
    patches = []
    for r in range(patch_size):
        for c in range(patch_size):
            patch = img.crop((c * cell, r * cell, (c + 1) * cell, (r + 1) * cell))
            patch = patch.resize((300, 300), Image.BICUBIC)
            patches.append(patch)
    return patches


def infer_pair_id_phase(member_name: str):
    normalized = member_name.replace("\\", "/")
    if "/images/" not in normalized or not normalized.lower().endswith(".png"):
        return None, None

    stem = Path(normalized).stem
    if stem.endswith("_pre_disaster"):
        return stem[: -len("_pre_disaster")], "pre"
    if stem.endswith("_post_disaster"):
        return stem[: -len("_post_disaster")], "post"
    return None, None


def infer_archive_role(path: Path) -> str:
    lower = path.name.lower()
    if "hold" in lower or "tier3" in lower:
        return "hold"
    if "test" in lower:
        return "test"
    if "train" in lower:
        return "train"
    return path.stem


def collect_archive_catalog(archive_paths: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    rows = {}
    archive_summaries = []

    for archive_path in archive_paths:
        if not archive_path.exists():
            continue

        role = infer_archive_role(archive_path)
        image_count = 0
        pair_counter = Counter()

        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                pair_id, phase = infer_pair_id_phase(member.name)
                if pair_id is None:
                    continue

                image_count += 1
                pair_counter[phase] += 1

                if pair_id not in rows:
                    rows[pair_id] = {
                        "pair_id": pair_id,
                        "disaster": pair_id.split("_", 1)[0],
                        "archive_role": role,
                        "archive_path": str(archive_path),
                        "pre_member": "",
                        "post_member": "",
                    }
                else:
                    existing = rows[pair_id]
                    if existing["archive_path"] != str(archive_path):
                        raise ValueError(
                            f"Duplicate pair_id across archives is ambiguous: {pair_id} "
                            f"({existing['archive_path']} vs {archive_path})"
                        )

                target_key = f"{phase}_member"
                if rows[pair_id][target_key]:
                    raise ValueError(f"Duplicate {phase} image for pair_id={pair_id} inside {archive_path}")
                rows[pair_id][target_key] = member.name

        archive_summaries.append(
            {
                "archive_path": str(archive_path),
                "archive_role": role,
                "image_entries": int(image_count),
                "pre_entries": int(pair_counter["pre"]),
                "post_entries": int(pair_counter["post"]),
            }
        )

    catalog = pd.DataFrame(rows.values()).sort_values("pair_id").reset_index(drop=True)
    if catalog.empty:
        return catalog, archive_summaries

    complete_mask = catalog["pre_member"].map(bool) & catalog["post_member"].map(bool)
    incomplete = catalog.loc[~complete_mask, ["pair_id", "archive_role", "pre_member", "post_member"]]
    if not incomplete.empty:
        raise ValueError(
            "Found xBD pairs without both pre/post images. Examples:\n"
            + incomplete.head(10).to_string(index=False)
        )

    return catalog.loc[complete_mask].reset_index(drop=True), archive_summaries


def build_airloc_style_partition(num_items: int, seed: int) -> list[int]:
    partition = [0] * num_items
    for idx in range(int(num_items * 0.7), int(num_items * 0.85)):
        partition[idx] = 1
    for idx in range(int(num_items * 0.85), num_items):
        partition[idx] = 2

    rng = random.Random(seed)
    rng.shuffle(partition)
    return partition


def normalize_pair_key(value: str) -> str:
    stem = Path(str(value)).stem
    if stem.endswith("_pre_disaster"):
        return stem[: -len("_pre_disaster")]
    if stem.endswith("_post_disaster"):
        return stem[: -len("_post_disaster")]
    return stem


def resolve_partition_from_csv(catalog: pd.DataFrame, split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv).copy()
    key_column = None
    for candidate in ["pair_id", "image_id", "filename", "sample_id"]:
        if candidate in df.columns:
            key_column = candidate
            break
    if key_column is None:
        raise ValueError(f"Cannot infer key column from split csv: {split_csv}")

    partition_column = None
    for candidate in ["partition", "split"]:
        if candidate in df.columns:
            partition_column = candidate
            break
    if partition_column is None:
        raise ValueError(f"Cannot infer partition column from split csv: {split_csv}")

    df["pair_id"] = df[key_column].map(normalize_pair_key)
    df = df.drop_duplicates(subset=["pair_id"], keep="first").copy()

    merged = catalog.merge(df[["pair_id", partition_column]], on="pair_id", how="left")
    if merged[partition_column].isna().any():
        missing = merged.loc[merged[partition_column].isna(), "pair_id"].head(10).tolist()
        raise ValueError(f"Split csv does not cover all xBD pairs. Missing examples: {missing}")

    merged["partition"] = merged[partition_column]
    return merged.drop(columns=[partition_column])


def assign_paper_test800(catalog: pd.DataFrame, seed: int, expected_test_pairs: int) -> pd.DataFrame:
    test_catalog = catalog.loc[catalog["archive_role"].astype(str).str.lower() == "test"].copy()
    if len(test_catalog) < expected_test_pairs:
        raise ValueError(
            f"paper-test800 requires at least {expected_test_pairs} test pairs, found {len(test_catalog)}."
        )

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for _, group in test_catalog.groupby("disaster", sort=True):
        quota = int(np.floor(len(group) * expected_test_pairs / len(test_catalog)))
        if quota <= 0:
            continue
        group_indices = list(group.index)
        rng.shuffle(group_indices)
        selected_indices.extend(group_indices[:quota])

    remaining_needed = expected_test_pairs - len(selected_indices)
    if remaining_needed > 0:
        selected_set = set(selected_indices)
        remaining = [idx for idx in test_catalog.index if idx not in selected_set]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[:remaining_needed])

    selected_indices = sorted(selected_indices[:expected_test_pairs])
    assigned = catalog.copy()
    assigned["partition"] = "unused"
    assigned.loc[selected_indices, "partition"] = "test"
    return assigned


def assign_partition(
    catalog: pd.DataFrame,
    split_mode: str,
    split_csv: str,
    seed: int,
    expected_test_pairs: int,
) -> pd.DataFrame:
    if split_mode == "explicit":
        if not split_csv:
            raise ValueError("split_mode=explicit requires --split-csv to avoid inventing an xBD benchmark subset.")
        return resolve_partition_from_csv(catalog, Path(split_csv))

    if split_mode == "paper-test800":
        return assign_paper_test800(catalog, seed, expected_test_pairs=expected_test_pairs)

    assigned = catalog.copy()
    assigned["partition"] = build_airloc_style_partition(len(assigned), seed)
    return assigned


def build_manifest(
    catalog: pd.DataFrame,
    archive_summaries: list[dict],
    args,
    output_dir: Path,
    blockers: list[str],
    paper_faithful: bool,
) -> dict:
    partition_counts = {}
    if "partition" in catalog.columns:
        for key, value in catalog["partition"].value_counts(dropna=False).sort_index().items():
            partition_counts[str(key)] = int(value)

    return {
        "dataset": "xbd",
        "raw_archives": archive_summaries,
        "output_dir": str(output_dir),
        "pair_count": int(len(catalog)),
        "partition_counts": partition_counts,
        "split_mode": args.split_mode,
        "split_csv": args.split_csv or None,
        "selection_note": getattr(args, "selection_note", ""),
        "seed": int(args.seed),
        "patch_size": int(args.patch_size),
        "expected_total_pairs": int(args.expected_total_pairs),
        "expected_test_pairs": int(args.expected_test_pairs),
        "paper_faithful": bool(paper_faithful),
        "blockers": blockers,
    }


def build_embeddings(rows: pd.DataFrame, patch_size: int, model_name: str, device: str):
    from transformers import CLIPVisionModelWithProjection

    transform_overhead = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988)),
        ]
    )

    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()

    pre_embeddings: dict[str, np.ndarray] = {}
    post_embeddings: dict[str, np.ndarray] = {}
    wanted: dict[str, dict[str, tuple[str, str]]] = {}
    for row in rows.itertuples(index=False):
        key = f"img_{row.eval_idx}"
        wanted.setdefault(str(row.archive_path), {})[str(row.pre_member)] = (key, "pre")
        wanted.setdefault(str(row.archive_path), {})[str(row.post_member)] = (key, "post")

    progress = tqdm(total=len(rows) * 2, desc="xbd images")
    try:
        for archive_path, members in wanted.items():
            found = 0
            found_names: set[str] = set()
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf:
                    target = members.get(member.name)
                    if target is None:
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        raise FileNotFoundError(f"Cannot extract {member.name} from {archive_path}")
                    image = Image.open(io.BytesIO(extracted.read()))
                    patches = split_patches(image, patch_size)
                    batch = torch.stack([transform_overhead(patch) for patch in patches], dim=0).to(device)
                    with torch.no_grad():
                        embeds = model(batch).image_embeds.detach().cpu().numpy()
                    key, phase = target
                    if phase == "pre":
                        pre_embeddings[key] = embeds
                    else:
                        post_embeddings[key] = embeds
                    found += 1
                    found_names.add(member.name)
                    progress.update(1)
                    if found == len(members):
                        break
            if found != len(members):
                missing = sorted(set(members) - found_names)
                raise FileNotFoundError(
                    f"Archive scan did not find all requested xBD members in {archive_path}; "
                    f"found {found}, expected {len(members)}. Missing examples: {missing[:5]}"
                )
    finally:
        progress.close()

    return pre_embeddings, post_embeddings


def main():
    parser = argparse.ArgumentParser(description="Prepare xBD benchmark arrays in GeoExplorer-compatible format.")
    parser.add_argument("--train-tar", default="data/xbd/raw/archives/train_images_labels_targets.tar.gz")
    parser.add_argument("--hold-tar", default="")
    parser.add_argument("--test-tar", default="data/xbd/raw/archives/test_images_labels_targets.tar.gz")
    parser.add_argument("--split-mode", choices=["explicit", "airloc-generated", "paper-test800"], default="paper-test800")
    parser.add_argument("--split-csv", default="")
    parser.add_argument("--output-dir", default="data/xbd/processed")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--model-name", default="MVRL/Sat2Cap")
    parser.add_argument("--device", default=os.getenv("GEOEXPLORER_DEVICE", "cuda:0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-total-pairs", type=int, default=5333)
    parser.add_argument("--expected-test-pairs", type=int, default=800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    archive_paths = [Path(args.train_tar).resolve(), Path(args.hold_tar).resolve(), Path(args.test_tar).resolve()]
    archive_paths = [Path(path).resolve() for path in [args.train_tar, args.hold_tar, args.test_tar] if path]
    existing_archives = [path for path in archive_paths if path.exists()]
    catalog, archive_summaries = collect_archive_catalog(existing_archives)

    blockers = []
    if args.split_mode != "paper-test800" and args.train_tar and not Path(args.train_tar).resolve().exists():
        blockers.append(f"Missing train archive: {Path(args.train_tar).resolve()}")
    if args.split_mode != "paper-test800" and args.hold_tar and not Path(args.hold_tar).resolve().exists():
        blockers.append(
            "Missing hold/tier3 archive. With only train+test, xBD cannot match the full xView2-based paper setup."
        )
    if not Path(args.test_tar).resolve().exists():
        blockers.append(f"Missing test archive: {Path(args.test_tar).resolve()}")

    assigned = catalog.copy()
    if not catalog.empty:
        try:
            assigned = assign_partition(catalog, args.split_mode, args.split_csv, args.seed, args.expected_test_pairs)
        except Exception as exc:
            blockers.append(str(exc))

    if args.split_mode != "paper-test800" and len(catalog) != args.expected_total_pairs:
        blockers.append(
            f"xBD pair count mismatch: found {len(catalog)}, expected {args.expected_total_pairs} for the full paper setup."
        )

    if "partition" in assigned.columns:
        partition_series = assigned["partition"].astype(str)
        test_rows = assigned.loc[partition_series.isin(["2", "test"])].copy()
    else:
        test_rows = assigned.iloc[0:0].copy()
    if not test_rows.empty and len(test_rows) != args.expected_test_pairs:
        blockers.append(
            f"xBD test subset mismatch: found {len(test_rows)}, expected {args.expected_test_pairs} for the paper benchmark."
        )

    if args.split_mode == "paper-test800":
        args.selection_note = (
            "Deterministic stratified 800-pair subset from the official xBD test archive. "
            "The original GeoExplorer/GOMAA 800-pair split file was not available locally; "
            "use this as a paper-style reproduction protocol, not a claim of identical split."
        )
    else:
        args.selection_note = "Explicit split from split_csv." if args.split_mode == "explicit" else "Generated AirLoc-style partition."
    paper_faithful = not blockers and args.split_mode == "explicit"
    manifest = build_manifest(assigned, archive_summaries, args, output_dir, blockers, paper_faithful)

    manifest_path = output_dir / "xbd_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {manifest_path}")

    if assigned.empty:
        raise ValueError("No usable xBD image pairs were found in the provided archives.")

    index_csv = output_dir / "xbd_index.csv"
    assigned.to_csv(index_csv, index=False, encoding="utf-8")
    print(f"saved {index_csv}")

    if args.audit_only:
        return

    if blockers:
        raise ValueError("xBD preprocessing blocked:\n- " + "\n- ".join(blockers))

    test_rows = test_rows.reset_index(drop=True).copy()
    if args.limit > 0:
        test_rows = test_rows.iloc[: args.limit].copy()
    test_rows.insert(0, "eval_idx", np.arange(len(test_rows), dtype=int))

    pre_embeddings, post_embeddings = build_embeddings(test_rows, args.patch_size, args.model_name, device)
    np.save(output_dir / "xbd_pre_grid_5.npy", pre_embeddings)
    print(f"saved {output_dir / 'xbd_pre_grid_5.npy'}")
    np.save(output_dir / "xbd_post_grid_5.npy", post_embeddings)
    print(f"saved {output_dir / 'xbd_post_grid_5.npy'}")


if __name__ == "__main__":
    main()
