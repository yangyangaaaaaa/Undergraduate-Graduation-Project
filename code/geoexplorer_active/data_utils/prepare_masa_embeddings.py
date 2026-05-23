import argparse
import csv
import io
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection


def load_split_ids(metadata_csv: Path):
    split_to_ids = defaultdict(list)
    with metadata_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            image_id = row["image_id"].strip()
            split_to_ids[split].append(image_id)
    return split_to_ids


def build_path_index(zip_infos):
    by_suffix = {}
    for info in zip_infos:
        name = info.filename
        if not name.lower().endswith(".png"):
            continue
        parts = name.split("/")
        if len(parts) < 2:
            continue
        split = parts[-2]
        image_name = parts[-1]
        key = f"{split}/{image_name}"
        by_suffix[key] = name
    return by_suffix


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--masa-zip",
        default="data/masa/Massachusetts Buildings Dataset_datasets.zip",
    )
    parser.add_argument("--metadata-csv", default="data/masa/metadata.csv")
    parser.add_argument("--output-dir", default="data/masa")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--model-name", default="MVRL/Sat2Cap")
    parser.add_argument("--device", default=os.getenv("GEOEXPLORER_DEVICE", "cuda:0"))
    args = parser.parse_args()

    masa_zip = Path(args.masa_zip)
    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_to_ids = load_split_ids(metadata_csv)

    transform_overhead = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988)),
        ]
    )

    model = CLIPVisionModelWithProjection.from_pretrained(args.model_name).to(args.device)
    model.eval()

    with zipfile.ZipFile(masa_zip, "r") as outer_zip:
        inner_name = None
        for name in outer_zip.namelist():
            if name.lower().endswith("_png_datasets.zip"):
                inner_name = name
                break
        if inner_name is None:
            raise RuntimeError("Cannot find *_png_datasets.zip in MASA outer zip.")

        inner_bytes = outer_zip.read(inner_name)
        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_zip:
            path_index = build_path_index(inner_zip.infolist())

            for split in ["train", "val", "test"]:
                image_ids = split_to_ids.get(split, [])
                if not image_ids:
                    continue

                embeddings = {}
                iterator = tqdm(enumerate(image_ids), total=len(image_ids), desc=f"masa-{split}")
                for idx, image_id in iterator:
                    suffix = f"{split}/{image_id}.png"
                    internal_path = path_index.get(suffix)
                    if internal_path is None:
                        raise FileNotFoundError(f"Missing {suffix} in nested png zip.")

                    image_bytes = inner_zip.read(internal_path)
                    image = Image.open(io.BytesIO(image_bytes))
                    patches = split_patches(image, args.patch_size)
                    batch = torch.stack([transform_overhead(p) for p in patches], dim=0).to(args.device)
                    with torch.no_grad():
                        image_embeds = model(batch).image_embeds.detach().cpu().numpy()
                    embeddings[f"img_{idx}"] = image_embeds

                out_path = output_dir / f"sat_{split}_grid_{args.patch_size}.npy"
                np.save(out_path, embeddings)
                print(f"saved {split} -> {out_path} ({len(embeddings)} items)")


if __name__ == "__main__":
    main()
