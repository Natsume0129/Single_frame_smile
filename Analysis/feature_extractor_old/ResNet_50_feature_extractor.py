# extract_features_resnet50.py
# Output features: (N x 2048)

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


ROOT_DIR = r"E:\Single_frame_smile\data\Isornot"
DAT_FILE = r"E:\Single_frame_smile\data\Isornot\matched.dat"
OUT_FILE = r"E:\Single_frame_smile\data\Isornot\features\resnet50_matched_features.npz"

BATCH_SIZE = 32
NUM_WORKERS = 0
USE_AMP = True


class DatDataset(Dataset):
    def __init__(self, dat_file: str, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.items: List[Tuple[str, int]] = []
        with open(dat_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rel_path, y = ln.split()
                self.items.append((rel_path, int(y)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel_path, y = self.items[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, rel_path, y


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_resnet50_feature_extractor(device: torch.device) -> torch.nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()  # output: (B, 2048)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device)


@torch.no_grad()
def main():
    Path(os.path.dirname(OUT_FILE)).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tf = build_eval_transform()
    ds = DatDataset(DAT_FILE, ROOT_DIR, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_resnet50_feature_extractor(device)
    use_amp = USE_AMP and (device.type == "cuda")

    all_paths: List[str] = []
    all_feats: List[np.ndarray] = []

    for x, rel_paths, _y in loader:
        x = x.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                feats = model(x)  # (B, 2048)
        else:
            feats = model(x)

        feats = feats.float().cpu().numpy().astype(np.float32)
        all_feats.append(feats)
        all_paths.extend(list(rel_paths))

    feats_mat = np.concatenate(all_feats, axis=0)
    paths_arr = np.array(all_paths)

    np.savez_compressed(OUT_FILE, paths=paths_arr, feats=feats_mat)
    print("Saved:", OUT_FILE)
    print("N =", len(paths_arr), "D =", feats_mat.shape[1])


if __name__ == "__main__":
    main()
