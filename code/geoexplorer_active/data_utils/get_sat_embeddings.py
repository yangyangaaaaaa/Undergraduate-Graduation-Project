import glob
import os

import fire
import numpy as np
import torch
from PIL import Image
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

class SatPatches(Dataset):
    def __init__(self, path, patch_size=5):
        self.path = path
        self.patch_size=patch_size
        self.transform_overhead = transforms.Compose([
            transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988))
        ])

    def __len__(self):
        return self.patch_size**2

    def __getitem__(self, idx):
        img = Image.open(f"{self.path}/patch_{idx}.jpg")
        transformed_img = self.transform_overhead(img)
        return transformed_img


def maybe_login(token=None):
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        login(hf_token)


def get_sat_embeddings(
    data_path="../data/swissview/swissview100_patches/patches/*",
    patch_size=5,
    save_path="../data/swissview/swissview100_sat_patches.npy",
    device=None,
    num_workers=4,
    model_name="MVRL/Sat2Cap",
    hf_token=None,
):
    data_path = os.getenv("GEOEXPLORER_SAT_PATCH_GLOB", data_path)
    save_path = os.getenv("GEOEXPLORER_SAT_SAVE_PATH", save_path)
    model_name = os.getenv("GEOEXPLORER_SAT_MODEL", model_name)
    device = device or os.getenv("GEOEXPLORER_EMBED_DEVICE") or os.getenv("GEOEXPLORER_DEVICE", "cuda:0")
    maybe_login(hf_token)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    model = model.to(device)
    sat_embeddings = {}
    file_list = sorted(glob.glob(data_path))
    for i in range(len(file_list)):
        dataset = SatPatches(
            file_list[i], patch_size=patch_size)
        predloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        preds = []
        for idx, batch in tqdm(enumerate(predloader)):
            preds.append(model(batch.to(device)).image_embeds.squeeze(0).detach().cpu().numpy())
        sat_embeddings[f"img_{i}"] = np.array(preds)
    np.save(save_path, sat_embeddings)

if __name__ == "__main__":
    fire.Fire(get_sat_embeddings)
