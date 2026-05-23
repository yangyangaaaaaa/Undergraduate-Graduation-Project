import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection


def maybe_login(token=None):
    try:
        from huggingface_hub import login
    except Exception:
        return

    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        login(hf_token)


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


def build_sample_index(raw_root: Path, captions_csv: Path) -> pd.DataFrame:
    ground_dir = raw_root / "ground_images_jpg"
    sat_dir = raw_root / "sat_images_patches"
    coords_df = pd.read_csv(raw_root / "coordinates.csv").copy()
    captions_df = pd.read_csv(captions_csv).copy()

    coords_df["sample_id"] = coords_df["filename"].map(lambda value: Path(str(value)).stem)
    captions_df["sample_id"] = captions_df["filename"].map(lambda value: Path(str(value)).stem)

    ground_index = {path.stem: path for path in ground_dir.glob("*.jpg")}
    sat_index = {path.stem: path for path in sat_dir.iterdir() if path.is_dir()}

    merged = captions_df.merge(coords_df, on="sample_id", how="inner", suffixes=("_caption", "_coord"))
    merged["ground_path"] = merged["sample_id"].map(lambda key: str(ground_index.get(key, "")))
    merged["sat_dir"] = merged["sample_id"].map(lambda key: str(sat_index.get(key, "")))
    merged["stitched_image"] = merged["sample_id"].map(
        lambda key: str((sat_index[key] / "stitched_image.jpg")) if key in sat_index else ""
    )

    merged = merged[
        merged["ground_path"].map(bool)
        & merged["sat_dir"].map(bool)
        & merged["stitched_image"].map(lambda value: Path(value).exists())
    ].copy()
    merged = merged.reset_index(drop=True)
    merged.insert(0, "idx", np.arange(len(merged), dtype=int))
    return merged


def build_sat_embeddings(samples: pd.DataFrame, patch_size: int, model_name: str, device: str):
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988)),
        ]
    )

    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()

    sat_embeddings = {}
    iterator = tqdm(samples.itertuples(index=False), total=len(samples), desc="mmgag-sat")
    for row in iterator:
        stitched = Image.open(row.stitched_image)
        patches = split_patches(stitched, patch_size)
        batch = torch.stack([transform(patch) for patch in patches], dim=0).to(device)
        with torch.no_grad():
            image_embeds = model(batch).image_embeds.detach().cpu().numpy()
        sat_embeddings[f"img_{row.idx}"] = image_embeds
    return sat_embeddings


def build_ground_embeddings(samples: pd.DataFrame, model_name: str, device: str):
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    ground_embeddings = {}
    iterator = tqdm(samples.itertuples(index=False), total=len(samples), desc="mmgag-ground")
    for row in iterator:
        image = Image.open(row.ground_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            image_embeds = model(**inputs).image_embeds.squeeze(0).detach().cpu().numpy()
        ground_embeddings[f"img_{row.idx}"] = np.array(image_embeds)
    return ground_embeddings


def build_text_embeddings(samples: pd.DataFrame, model_name: str, device: str, batch_size: int):
    model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    captions = samples["Caption"].astype(str).tolist()
    rows = []
    for start in tqdm(range(0, len(captions), batch_size), desc="mmgag-text"):
        batch = captions[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt", max_length=77, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            text_embeds = model(**inputs).text_embeds.detach().cpu().numpy()
        rows.append(text_embeds)
    return np.concatenate(rows, axis=0)


def write_alias(path: Path, payload):
    np.save(path, payload)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MM-GAG embeddings in GeoExplorer-compatible format.")
    parser.add_argument("--raw-root", default="data/mm_gag/raw/gomaa_geo_mm_gag")
    parser.add_argument("--captions-csv", default="data/mm_gag/raw/mvrl_aux/coords_captions.csv")
    parser.add_argument("--output-dir", default="data/mm_gag/processed")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--sat-model-name", default="MVRL/Sat2Cap")
    parser.add_argument("--ground-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=os.getenv("GEOEXPLORER_DEVICE", "cuda:0"))
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    raw_root = Path(args.raw_root).resolve()
    captions_csv = Path(args.captions_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_login(args.hf_token or None)
    samples = build_sample_index(raw_root, captions_csv)
    if args.limit > 0:
        samples = samples.iloc[: args.limit].copy()

    index_csv = output_dir / "mmgag_index.csv"
    samples.to_csv(index_csv, index=False, encoding="utf-8")
    print(f"saved {index_csv}")

    sat_embeddings = build_sat_embeddings(samples, args.patch_size, args.sat_model_name, device)
    ground_embeddings = build_ground_embeddings(samples, args.ground_model_name, device)
    text_embeddings = build_text_embeddings(samples, args.text_model_name, device, args.text_batch_size)

    write_alias(output_dir / "mmgag_sat_patches.npy", sat_embeddings)
    write_alias(output_dir / f"mmgag_sat_grid_{args.patch_size}.npy", sat_embeddings)
    write_alias(output_dir / "mmgag_ground_embeds.npy", ground_embeddings)
    write_alias(output_dir / "papr_my_ground_embeds.npy", ground_embeddings)
    write_alias(output_dir / "mmgag_text_embeds.npy", text_embeddings)
    write_alias(output_dir / "papr_my_text_embeds.npy", text_embeddings)

    manifest = {
        "sample_count": int(len(samples)),
        "device": device,
        "patch_size": int(args.patch_size),
        "sat_model_name": args.sat_model_name,
        "ground_model_name": args.ground_model_name,
        "text_model_name": args.text_model_name,
        "raw_root": str(raw_root),
        "captions_csv": str(captions_csv),
        "outputs": {
            "index_csv": str(index_csv),
            "sat_embeddings": str(output_dir / "mmgag_sat_patches.npy"),
            "ground_embeddings": str(output_dir / "mmgag_ground_embeds.npy"),
            "text_embeddings": str(output_dir / "mmgag_text_embeds.npy"),
        },
    }
    (output_dir / "mmgag_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_dir / 'mmgag_manifest.json'}")


if __name__ == "__main__":
    main()
