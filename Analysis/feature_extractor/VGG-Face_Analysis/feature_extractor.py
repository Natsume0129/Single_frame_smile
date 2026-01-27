'''
Docstring for Analysis.feature_extractor.VGG-Face_Analysis.feature_extractor
这里面用的内容是：VGG-Face 特征提取器，用于从图像中提取人脸特征向量。
特征提取器使用的模型是 VGGFace_conv，并加载预训练的权重文件vggface
'''


# feature_extractor.py
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

import face_comp_torch as FCmodel 


# -------------------------
# Dataset: read all images in a folder (keep order by filename)
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp"), transform=None):
        self.img_dir = Path(img_dir)
        self.paths = sorted([p for p in self.img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in: {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # return tensor + filename (for alignment)
        return img, p.name


# -------------------------
# Feature extractor: VGGFace_conv + load state_dict
# output: [B, 512*7*7] = [B, 25088]
# -------------------------
class VGGFaceConvExtractor(nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        self.backbone = FCmodel.VGGFace_conv()  # matches keys like features.conv_1_1.*, fc.* in your weights
        state_dict = torch.load(weight_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)              # [B, 25088] (flattened in forward)
        x = F.normalize(x, dim=1)         # optional but usually helpful
        return x


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to vggface.pth (state_dict)")
    parser.add_argument("--img_dir", required=True, help="folder containing face images")
    parser.add_argument("--save", required=True, help="output .pt path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows default: 0 is safest
    parser.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # Use the same preprocessing as the original project
    transform = transforms.Compose([
        transforms.Resize((FCmodel.IMG_HEIGHT_VGG16, FCmodel.IMG_WIDTH_VGG16)),
        transforms.ToTensor(),
        transforms.Normalize(
            (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),  # RGB
            (1.0, 1.0, 1.0)
        )
    ])

    ds = ImageFolderDataset(args.img_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,              # IMPORTANT: keep order
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = VGGFaceConvExtractor(args.weights).to(device)
    model.eval()

    all_feats: List[torch.Tensor] = []
    all_names: List[str] = []

    with torch.no_grad():
        for imgs, names in dl:
            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs)                 # [B, 25088]
            all_feats.append(feats.cpu())
            all_names.extend(list(names))

    feats_mat = torch.cat(all_feats, dim=0)     # [T, 25088]

    os.makedirs(str(Path(args.save).parent), exist_ok=True)
    torch.save({"names": all_names, "feats": feats_mat}, args.save)

    print("Done.")
    print("Num images:", len(all_names))
    print("Feature shape:", tuple(feats_mat.shape))
    print("Saved to:", args.save)


if __name__ == "__main__":
    main()
