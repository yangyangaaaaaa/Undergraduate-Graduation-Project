import os
from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection


def get_ground_embeddings(
    data_path="../data/swissview/SwissViewMonuments/ground_view/",
    save_path="../data/swissview/swissviewmonuments_grd.npy",
    device=None,
    model_name="openai/clip-vit-base-patch32",
):
    data_path = os.getenv("GEOEXPLORER_GROUND_VIEW_DIR", data_path)
    save_path = os.getenv("GEOEXPLORER_GROUND_EMBEDS_SAVE_PATH", save_path)
    model_name = os.getenv("GEOEXPLORER_GROUND_MODEL", model_name)
    device = device or os.getenv("GEOEXPLORER_EMBED_DEVICE") or os.getenv("GEOEXPLORER_DEVICE", "cuda:0")
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    grd_embeddings = {}
    image_list = sorted(os.listdir(data_path))
    print(image_list)

    for i in range(len(image_list)):
        img = Image.open(data_path / image_list[i])
        inputs = processor(images=img, return_tensors="pt", padding=True)
        inputs.to(device)
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds.squeeze(0).detach().cpu().numpy()
        grd_embeddings[f"img_{i}"] = np.array(image_embeds)

    np.save(save_path, grd_embeddings)


if __name__ == "__main__":
    fire.Fire(get_ground_embeddings)
