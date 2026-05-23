from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection


SERIES = "ultra_long_eval_20260521"
EXPERIMENT = "anchor0624_ultralong_grid_stress"

REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
MONITORING = EXP_ROOT / "monitoring"
OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260521_ultra_long_grid_stress_v3_grid25"
STATUS_PATH = MONITORING / "ultra_long_status_latest.json"
LOG_DIR = MONITORING / "logs"
EVALUATOR = MONITORING / "paper_geo_evaluator.py"

COMPARE_ROOT = Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle")
GOMAA_ROOT = COMPARE_ROOT / "gomaa_geo_official"
NVIDIA_COMPAT_LIB = REMOTE_ROOT / "env/nvidia_535_288/usr/lib/x86_64-linux-gnu"

ANCHOR_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_ablation_anchor0624_20260515/"
    / "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01/"
    / "g1_p1_e1_v1_seed321_t480k/geoexplorer.pt"
)
ANCHOR_LLM = REMOTE_ROOT / "results/checkpoint/env_modeling_fullrerun_20260407_111046/state_action.ckpt"
GOMAA_CKPT = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_ppo_seed42_t480k/formal_ppo.pt"
GOMAA_LLM = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_pretrain_seed42_e50/formal_falcon.ckpt"
PRISTINE_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_dualseed480k_20260427/"
    / "masa_plus_mmgag_arena_pristine_a040_a0405_sine0405_dualseed480k/"
    / "wave2_seed321_4gpu/pristine_seed321_t480k/geoexplorer.pt"
)
PRISTINE_LLM = ANCHOR_LLM

TASK_BANK_SEED = 20260521
MODEL_NAME = os.getenv("ULTRA_LONG_EMBED_MODEL", "MVRL/Sat2Cap")
EMBED_BATCH_SIZE = int(os.getenv("ULTRA_LONG_EMBED_BATCH_SIZE", "64"))

MASA_ZIP_CANDIDATES = [
    REMOTE_ROOT / "data/masa/Massachusetts Buildings Dataset_datasets.zip",
    REPO_ROOT / "data/masa/Massachusetts Buildings Dataset_datasets.zip",
]
MASA_METADATA_CANDIDATES = [
    REMOTE_ROOT / "data/masa/metadata.csv",
    REPO_ROOT / "data/masa/metadata.csv",
]
MASA_OUTPUT_DIR = REMOTE_ROOT / "data/masa"

PROTOCOLS = {
    7: {"distances": [4, 6, 8, 10, 12], "budget": 18},
    8: {"distances": [10, 11, 12, 13, 14], "budget": 24},
    10: {"distances": [14, 15, 16, 17, 18], "budget": 32},
    25: {"distances": [12, 16, 20, 24, 28, 32, 36, 40, 44, 48], "budget": 60},
}

METHODS = [
    {
        "key": "gomaa",
        "method": "gomaa",
        "label": "GOMAA-Geo",
        "display_label": "GOMAA-Geo",
        "repo_dir": str(GOMAA_ROOT),
        "checkpoint": str(GOMAA_CKPT),
        "llm_checkpoint": str(GOMAA_LLM),
    },
    {
        "key": "pristine",
        "method": "geoexplorer",
        "label": "GeoExplorer-pristine",
        "display_label": "GeoExplorer",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(PRISTINE_CKPT),
        "llm_checkpoint": str(PRISTINE_LLM),
    },
    {
        "key": "anchor0624",
        "method": "geoexplorer",
        "label": "GeoExplorer-anchor0624",
        "display_label": "This work",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(ANCHOR_CKPT),
        "llm_checkpoint": str(ANCHOR_LLM),
    },
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(phase: str, **extra) -> None:
    MONITORING.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "phase": phase,
        "output_root": str(OUTPUT_ROOT),
        "task_bank_seed": TASK_BANK_SEED,
    }
    payload.update(extra)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {label}: {[str(path) for path in candidates]}")


def load_split_ids(metadata_csv: Path) -> dict[str, list[str]]:
    import csv as csv_module

    split_to_ids: dict[str, list[str]] = defaultdict(list)
    with metadata_csv.open("r", encoding="utf-8") as handle:
        reader = csv_module.DictReader(handle)
        for row in reader:
            split = row["split"].strip()
            image_id = row["image_id"].strip()
            split_to_ids[split].append(image_id)
    return split_to_ids


def build_path_index(zip_infos) -> dict[str, str]:
    by_suffix = {}
    for info in zip_infos:
        name = info.filename
        if not name.lower().endswith(".png"):
            continue
        parts = name.split("/")
        if len(parts) < 2:
            continue
        by_suffix[f"{parts[-2]}/{parts[-1]}"] = name
    return by_suffix


def split_patches(image: Image.Image, patch_size: int) -> list[Image.Image]:
    image = image.convert("RGB").resize((1500, 1500), Image.BICUBIC)
    cell = 1500 // patch_size
    patches = []
    for row in range(patch_size):
        for col in range(patch_size):
            box = (col * cell, row * cell, (col + 1) * cell, (row + 1) * cell)
            patches.append(image.crop(box))
    return patches


def embedding_ready(path: Path, patch_size: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = np.load(path, allow_pickle=True)
        bank = payload[()]
        if not bank:
            return False
        first = next(iter(bank.values()))
        return tuple(first.shape) == (patch_size * patch_size, 512)
    except Exception:
        return False


def encode_patches(model, transform, patches: list[Image.Image], device: torch.device) -> np.ndarray:
    chunks = []
    for start in range(0, len(patches), EMBED_BATCH_SIZE):
        batch_patches = patches[start : start + EMBED_BATCH_SIZE]
        batch = torch.stack([transform(patch) for patch in batch_patches], dim=0).to(device)
        with torch.no_grad():
            chunks.append(model(batch).image_embeds.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def prepare_masa_test_grid(patch_size: int, force_rebuild: bool = False) -> Path:
    out_path = MASA_OUTPUT_DIR / f"sat_test_grid_{patch_size}.npy"
    if not force_rebuild and embedding_ready(out_path, patch_size):
        write_status("embedding_reused", patch_size=patch_size, output_path=str(out_path))
        return out_path

    masa_zip = first_existing(MASA_ZIP_CANDIDATES, "MASA zip")
    metadata_csv = first_existing(MASA_METADATA_CANDIDATES, "MASA metadata")
    split_to_ids = load_split_ids(metadata_csv)
    image_ids = split_to_ids.get("test", [])
    if not image_ids:
        raise RuntimeError("MASA metadata has no test split.")

    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988)),
        ]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embeddings = {}
    with zipfile.ZipFile(masa_zip, "r") as outer_zip:
        inner_name = None
        for name in outer_zip.namelist():
            if name.lower().endswith("_png_datasets.zip"):
                inner_name = name
                break
        if inner_name is None:
            raise RuntimeError("Cannot find *_png_datasets.zip inside MASA zip.")

        inner_bytes = outer_zip.read(inner_name)
        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_zip:
            path_index = build_path_index(inner_zip.infolist())
            for idx, image_id in enumerate(image_ids):
                write_status(
                    "embedding",
                    patch_size=patch_size,
                    image_index=idx,
                    total_images=len(image_ids),
                    output_path=str(out_path),
                )
                suffix = f"test/{image_id}.png"
                internal_path = path_index.get(suffix)
                if internal_path is None:
                    raise FileNotFoundError(f"Missing {suffix} in nested MASA png zip.")
                image = Image.open(io.BytesIO(inner_zip.read(internal_path)))
                patches = split_patches(image, patch_size)
                embeddings[f"img_{idx}"] = encode_patches(model, transform, patches, device)

    MASA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    if not embedding_ready(out_path, patch_size):
        raise RuntimeError(f"Generated embedding failed validation: {out_path}")
    return out_path


def parse_grids(mode: str, grids_text: str) -> list[int]:
    if grids_text.strip():
        grids = [int(item.strip()) for item in grids_text.split(",") if item.strip()]
    elif mode == "full":
        grids = [8, 10]
    else:
        grids = [8]
    unsupported = [grid for grid in grids if grid not in PROTOCOLS]
    if unsupported:
        raise ValueError(f"Unsupported grid(s): {unsupported}; supported={sorted(PROTOCOLS)}")
    return grids


def eval_output_path(mode: str, grid: int, method_key: str) -> Path:
    return OUTPUT_ROOT / "raw" / mode / f"grid{grid}" / method_key / f"masa_aerial_grid{grid}.json"


def output_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return "success_ratio" in payload and "per_distance" in payload
    except Exception:
        return False


def build_eval_command(
    mode: str,
    method: dict,
    grid: int,
    test_path: Path,
    repeats_per_dist: int,
    max_images: int,
) -> list[str]:
    protocol = PROTOCOLS[grid]
    out_path = eval_output_path(mode, grid, method["key"])
    cmd = [
        "/usr/bin/python3",
        "-u",
        str(EVALUATOR),
        "--method",
        method["method"],
        "--method-label",
        method["label"],
        "--repo-dir",
        method["repo_dir"],
        "--checkpoint",
        method["checkpoint"],
        "--llm-checkpoint",
        method["llm_checkpoint"],
        "--benchmark",
        f"masa_aerial_ultralong_grid{grid}",
        "--paper-table",
        "Ultra-long grid stress",
        "--dataset",
        "masa",
        "--goal-mode",
        "aerial",
        "--fixed-goal-mode",
        "none",
        "--test-path",
        str(test_path),
        "--device",
        "cuda:0",
        "--patch-size",
        str(grid),
        "--budget",
        str(protocol["budget"]),
        "--distances",
        ",".join(str(item) for item in protocol["distances"]),
        "--repeats-per-dist",
        str(repeats_per_dist),
        "--seed",
        str(TASK_BANK_SEED),
        "--max-images",
        str(max_images),
        "--output-path",
        str(out_path),
    ]
    return cmd


def launch_eval(job: dict, gpu: int) -> dict:
    out_path = Path(job["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"eval__{job['mode']}__grid{job['grid']}__{job['method_key']}.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n[{now_iso()}] launching {job['key']} on GPU{gpu}\n".encode("utf-8"))
    env = os.environ.copy()
    pythonpath = (
        "/root/geoexplorer/env/geoexplorer_site:"
        "/root/geoexplorer:"
        "/root/geoexplorer/GeoExplorer:"
        "/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle"
    )
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if NVIDIA_COMPAT_LIB.exists():
        env["LD_LIBRARY_PATH"] = str(NVIDIA_COMPAT_LIB)
    process = subprocess.Popen(
        job["command"],
        cwd=str(MONITORING),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"process": process, "log_handle": log_handle, "gpu": gpu, "log_path": str(log_path)}


def evaluate_jobs(jobs: list[dict], eval_gpus: list[int]) -> None:
    states = {
        job["key"]: {
            "status": "pending",
            "grid": job["grid"],
            "method_key": job["method_key"],
            "method": job["method_label"],
            "output_path": job["output_path"],
        }
        for job in jobs
    }
    active: dict[str, dict] = {}
    while True:
        finished = []
        for key, item in active.items():
            code = item["process"].poll()
            if code is None:
                continue
            try:
                item["log_handle"].close()
            except Exception:
                pass
            ok = code == 0 and output_ready(Path(states[key]["output_path"]))
            states[key].update({"status": "completed" if ok else "failed", "returncode": int(code), "ended_at": now_iso()})
            finished.append(key)
        for key in finished:
            active.pop(key, None)

        counts = {gpu: 0 for gpu in eval_gpus}
        for item in active.values():
            counts[int(item["gpu"])] += 1

        for job in jobs:
            key = job["key"]
            state = states[key]
            if state["status"] == "pending" and output_ready(Path(job["output_path"])):
                state["status"] = "completed"
                state["resume_reused"] = True
                continue
            if state["status"] != "pending":
                continue
            available = [gpu for gpu in eval_gpus if counts[gpu] < 1]
            if not available:
                continue
            gpu = available[0]
            launched = launch_eval(job, gpu)
            active[key] = launched
            counts[gpu] += 1
            state.update({"status": "running", "gpu": gpu, "pid": int(launched["process"].pid), "started_at": now_iso()})

        write_status("evaluating", active_eval_processes=len(active), eval_jobs=states)
        if any(state["status"] == "failed" for state in states.values()):
            for item in active.values():
                try:
                    item["process"].terminate()
                    item["log_handle"].close()
                except Exception:
                    pass
            raise RuntimeError("One or more ultra-long evaluation jobs failed.")
        if all(state["status"] == "completed" for state in states.values()):
            return
        time.sleep(20)


def metric_from_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_distance = payload.get("per_distance", [])
    per_map = {int(row["distance"]): float(row["success_ratio"]) for row in per_distance}
    sg_map = {int(row["distance"]): float(row.get("sg_mean", math.nan)) for row in per_distance}
    return {
        "payload": payload,
        "sr": float(payload["success_ratio"]),
        "sg": float(payload["sg_mean"]),
        "num_images": int(payload.get("num_images", 0)),
        "num_tasks": int(payload.get("num_tasks", 0)),
        "per_distance_sr": per_map,
        "per_distance_sg": sg_map,
    }


def display_name(method_key: str, label: str) -> str:
    if method_key == "anchor0624":
        return "本文方法"
    if method_key == "pristine":
        return "GeoExplorer"
    return label


def aggregate(mode: str, grids: list[int], jobs: list[dict]) -> None:
    rows = []
    per_rows = []
    all_distances = sorted({dist for grid in grids for dist in PROTOCOLS[grid]["distances"]})
    for job in jobs:
        metric = metric_from_payload(Path(job["output_path"]))
        row = {
            "mode": mode,
            "grid": f"{job['grid']}x{job['grid']}",
            "budget": PROTOCOLS[job["grid"]]["budget"],
            "distances": ",".join(str(item) for item in PROTOCOLS[job["grid"]]["distances"]),
            "method_key": job["method_key"],
            "method": display_name(job["method_key"], job["method_label"]),
            "raw_method_label": job["method_label"],
            "success_ratio": metric["sr"],
            "sg_mean": metric["sg"],
            "num_images": metric["num_images"],
            "num_tasks": metric["num_tasks"],
            "checkpoint": job["checkpoint"],
        }
        for dist in all_distances:
            row[f"d{dist}"] = metric["per_distance_sr"].get(dist, math.nan)
        rows.append(row)
        for dist in PROTOCOLS[job["grid"]]["distances"]:
            per_rows.append(
                {
                    "mode": mode,
                    "grid": f"{job['grid']}x{job['grid']}",
                    "budget": PROTOCOLS[job["grid"]]["budget"],
                    "method_key": job["method_key"],
                    "method": row["method"],
                    "distance": dist,
                    "success_ratio": metric["per_distance_sr"].get(dist, math.nan),
                    "sg_mean": metric["per_distance_sg"].get(dist, math.nan),
                    "output_path": job["output_path"],
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUTPUT_ROOT / f"ultra_long_{mode}_summary.csv"
    fieldnames = [
        "mode",
        "grid",
        "budget",
        "distances",
        "method",
        "method_key",
        "raw_method_label",
        "success_ratio",
        "sg_mean",
        *[f"d{dist}" for dist in all_distances],
        "num_images",
        "num_tasks",
        "checkpoint",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    per_csv = OUTPUT_ROOT / f"ultra_long_{mode}_per_distance.csv"
    with per_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames_per = ["mode", "grid", "budget", "method", "method_key", "distance", "success_ratio", "sg_mean", "output_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames_per)
        writer.writeheader()
        writer.writerows(per_rows)

    aggregate_payload = {
        "created_at": now_iso(),
        "mode": mode,
        "protocols": {str(grid): PROTOCOLS[grid] for grid in grids},
        "task_bank_seed": TASK_BANK_SEED,
        "rows": rows,
        "per_distance": per_rows,
        "jobs": jobs,
    }
    aggregate_json = OUTPUT_ROOT / f"ultra_long_{mode}_aggregate.json"
    aggregate_json.write_text(json.dumps(aggregate_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = OUTPUT_ROOT / f"ultra_long_{mode}_summary_zh.md"
    lines = [
        "# 超长距离网格压力测试结果",
        "",
        f"- 生成时间：{aggregate_payload['created_at']}",
        f"- 模式：{mode}",
        "- 数据集：MASA aerial test split，仅做俯视图探索。",
        "- 说明：该实验为 evaluation-only 压力测试，不是原论文标准协议。",
        f"- 任务种子：{TASK_BANK_SEED}",
        "",
    ]
    for grid in grids:
        grid_rows = [row for row in rows if row["grid"] == f"{grid}x{grid}"]
        protocol = PROTOCOLS[grid]
        dist_cols = protocol["distances"]
        lines.extend(
            [
                f"## {grid}x{grid} 网格",
                "",
                f"- 距离桶：{','.join(str(item) for item in dist_cols)}",
                f"- 搜索预算：B={protocol['budget']}",
                "",
                "| 方法 | SR | SG | " + " | ".join(f"D={dist}" for dist in dist_cols) + " |",
                "| --- | ---: | ---: | " + " | ".join("---:" for _ in dist_cols) + " |",
            ]
        )
        for row in grid_rows:
            values = [row["method"], f"{row['success_ratio']:.4f}", f"{row['sg_mean']:.4f}"]
            values.extend("" if math.isnan(float(row.get(f"d{dist}", math.nan))) else f"{float(row[f'd{dist}']):.4f}" for dist in dist_cols)
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(
        [
            "## 写作建议",
            "",
            "该表可表述为：在扩大搜索网格和起始距离后，模型需要在更长路径上持续保持目标方向判断。若本文方法在高距离桶保持更高 SR 或更低 SG，可作为中远距离优势的补充证据。",
            "",
        ]
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")


def build_jobs(mode: str, grids: list[int], test_paths: dict[int, Path], repeats_per_dist: int, max_images: int) -> list[dict]:
    jobs = []
    for grid in grids:
        for method in METHODS:
            out_path = eval_output_path(mode, grid, method["key"])
            job = {
                "key": f"{mode}::grid{grid}::{method['key']}",
                "mode": mode,
                "grid": grid,
                "method_key": method["key"],
                "method_label": method["label"],
                "display_label": method["display_label"],
                "checkpoint": method["checkpoint"],
                "output_path": str(out_path),
                "command": build_eval_command(mode, method, grid, test_paths[grid], repeats_per_dist, max_images),
            }
            jobs.append(job)
    return jobs


def main_impl() -> int:
    parser = argparse.ArgumentParser(description="Run ultra-long grid stress evaluation.")
    parser.add_argument("--mode", choices=["smoke", "formal", "full"], default=os.getenv("ULTRA_LONG_MODE", "smoke"))
    parser.add_argument("--grids", default=os.getenv("ULTRA_LONG_GRIDS", ""))
    parser.add_argument("--eval-gpus", default=os.getenv("ULTRA_LONG_EVAL_GPUS", "0,1,2"))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--repeats-per-dist", type=int, default=0)
    args = parser.parse_args()

    grids = parse_grids(args.mode, args.grids)
    eval_gpus = [int(item.strip()) for item in args.eval_gpus.split(",") if item.strip()]
    if not eval_gpus:
        eval_gpus = [0]
    repeats_per_dist = args.repeats_per_dist or (1 if args.mode == "smoke" else 20)
    max_images = args.max_images if args.max_images >= 0 else (2 if args.mode == "smoke" else 0)

    write_status(
        "starting",
        mode=args.mode,
        grids=grids,
        eval_gpus=eval_gpus,
        repeats_per_dist=repeats_per_dist,
        max_images=max_images,
    )

    for method in METHODS:
        if not Path(method["checkpoint"]).exists():
            raise FileNotFoundError(f"Missing checkpoint for {method['key']}: {method['checkpoint']}")
        if not Path(method["llm_checkpoint"]).exists():
            raise FileNotFoundError(f"Missing LLM checkpoint for {method['key']}: {method['llm_checkpoint']}")

    test_paths = {}
    for grid in grids:
        test_paths[grid] = prepare_masa_test_grid(grid, force_rebuild=args.force_rebuild)

    jobs = build_jobs(args.mode, grids, test_paths, repeats_per_dist, max_images)
    evaluate_jobs(jobs, eval_gpus)
    aggregate(args.mode, grids, jobs)
    write_status(
        "completed",
        mode=args.mode,
        grids=grids,
        repeats_per_dist=repeats_per_dist,
        max_images=max_images,
        output_root=str(OUTPUT_ROOT),
    )
    return 0


def main() -> int:
    try:
        return main_impl()
    except Exception as exc:
        write_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
