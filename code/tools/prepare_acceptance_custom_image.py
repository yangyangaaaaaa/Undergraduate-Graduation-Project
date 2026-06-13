from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def split_and_save_patches(image: Image.Image, patch_dir: Path, patch_size: int) -> list[Image.Image]:
    image = image.convert("RGB").resize((1500, 1500), Image.BICUBIC)
    patch_dir.mkdir(parents=True, exist_ok=True)
    cell = 1500 // patch_size
    patches: list[Image.Image] = []
    for row in range(patch_size):
        for col in range(patch_size):
            patch = image.crop((col * cell, row * cell, (col + 1) * cell, (row + 1) * cell))
            patch = patch.resize((300, 300), Image.BICUBIC)
            patch.save(patch_dir / f"patch_{len(patches)}.jpg", quality=95)
            patches.append(patch)
    return patches


def build_embeddings(
    patches: list[Image.Image],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988)),
        ]
    )
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.stack([transform(patch) for patch in patches[start : start + batch_size]], dim=0).to(device)
            rows.append(model(batch).image_embeds.detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a single custom image for GeoExplorer acceptance inference.")
    parser.add_argument("--image", required=True, help="Custom aerial image path on the server.")
    parser.add_argument("--output-dir", required=True, help="Directory for temporary patches and embedding bank.")
    parser.add_argument("--asset-cache-dir", required=True, help="VIS_ROOT asset cache directory for visualization.")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--device", default=os.getenv("ACCEPTANCE_EMBED_DEVICE", os.getenv("GEOEXPLORER_DEVICE", "cuda:0")))
    parser.add_argument("--model-name", default=os.getenv("ACCEPTANCE_SAT_MODEL", "MVRL/Sat2Cap"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("ACCEPTANCE_EMBED_BATCH_SIZE", "8")))
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"custom image not found: {image_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = output_dir / "patches" / "img_0"
    asset_dir = Path(args.asset_cache_dir).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)

    normalized = Image.open(image_path).convert("RGB").resize((1500, 1500), Image.BICUBIC)
    normalized_path = output_dir / "custom_image_1500.png"
    normalized.save(normalized_path)
    normalized.save(asset_dir / "img_000.png")

    patches = split_and_save_patches(normalized, patch_dir, args.patch_size)
    device = resolve_device(args.device)
    embeds = build_embeddings(patches, args.model_name, device, args.batch_size)

    embedding_path = output_dir / "custom_sat_patches.npy"
    np.save(embedding_path, {"img_0": embeds})

    manifest = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_image": str(image_path),
        "normalized_image": str(normalized_path),
        "asset_image": str(asset_dir / "img_000.png"),
        "embedding_path": str(embedding_path),
        "patch_dir": str(patch_dir),
        "patch_size": int(args.patch_size),
        "patch_count": len(patches),
        "embedding_shape": list(embeds.shape),
        "model_name": args.model_name,
        "device": device,
    }
    manifest_path = output_dir / "custom_image_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
