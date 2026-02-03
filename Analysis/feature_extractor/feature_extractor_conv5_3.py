# feature_extractor_conv5_3.py
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import face_comp_torch as FCmodel
import re

def extract_last_int(path: Path) -> int:
    # 从文件名（不含扩展名）提取最后一个整数，作为帧号
    stem = path.stem
    m = re.search(r"(\d+)(?!.*\d)", stem)
    if not m:
        raise RuntimeError(f"Cannot parse frame index from filename: {path.name}")
    return int(m.group(1))

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: str, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp")):
        self.img_dir = Path(img_dir)
        paths = [p for p in self.img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if len(paths) == 0:
            raise RuntimeError(f"No images found in: {img_dir}")

        # 关键：按“帧号数值”排序，而不是字典序
        self.paths = sorted(paths, key=extract_last_int)

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, p.name


def build_preprocess():
    return transforms.Compose([
        transforms.Resize((FCmodel.IMG_HEIGHT_VGG16, FCmodel.IMG_WIDTH_VGG16)),
        transforms.ToTensor(),
        transforms.Normalize(
            (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
            (1.0, 1.0, 1.0)
        )
    ])


class VGGFaceConv53Extractor(nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        self.backbone = FCmodel.VGGFace_conv()
        sd = torch.load(weight_path, map_location="cpu")
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)

        # vggface.pth 会多出 fc.*，这里忽略是正常的
        if len(unexpected) > 0:
            print(f"[INFO] Ignored unexpected keys (example): {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")
        if len(missing) > 0:
            # 如果缺 features.* 才需要警惕
            print(f"[WARN] Missing keys (example): {missing[:6]}{'...' if len(missing)>6 else ''}")

    def forward(self, x: torch.Tensor):
        # 输出 maxp_5_3 后的特征图: [B,512,7,7]
        for k, layer in self.backbone.features.items():
            x = layer(x)
            if k == "maxp_5_3":
                break
        conv_map = x
        gap512 = conv_map.mean(dim=(2, 3))  # [B,512]
        return conv_map, gap512


def resolve_save_paths(save_arg: str):
    """
    --save 可以传：
      1) 目录：E:\...\features\
         -> 输出 features\conv5_3.pt 和 features\gap512.pt
      2) 前缀（带文件名但不带后缀或不以 .pt 结尾）：E:\...\features\sample1
         -> 输出 sample1_conv5_3.pt 和 sample1_gap512.pt
      3) 直接给一个 .pt 文件：E:\...\features\sample1.pt
         -> 输出 sample1_conv5_3.pt 和 sample1_gap512.pt
    """
    p = Path(save_arg)

    # 情况 1：传的是目录
    if p.exists() and p.is_dir():
        out_dir = p
        prefix = "vggface"
    else:
        # 如果看起来像目录（以 \ 结尾）但还不存在
        if str(save_arg).endswith(("\\", "/")):
            out_dir = p
            prefix = "vggface"
        else:
            out_dir = p.parent if p.suffix else p.parent
            # prefix 取 stem（有后缀就 stem；没后缀就 name）
            prefix = p.stem if p.suffix else p.name
            if prefix == "":
                prefix = "vggface"

    out_dir.mkdir(parents=True, exist_ok=True)
    conv_path = out_dir / f"{prefix}_conv5_3.pt"
    gap_path  = out_dir / f"{prefix}_gap512.pt"
    return conv_path, gap_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="vggface_conv.pth (recommended) or vggface.pth")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--save", required=True, help="output dir or prefix path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    transform = build_preprocess()
    ds = ImageFolderDataset(args.img_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VGGFaceConv53Extractor(args.weights).to(device)
    model.eval()

    all_names: List[str] = []
    all_conv: List[torch.Tensor] = []
    all_gap: List[torch.Tensor] = []

    with torch.no_grad():
        for imgs, names in dl:
            imgs = imgs.to(device, non_blocking=True)
            conv_map, gap512 = model(imgs)
            all_conv.append(conv_map.cpu())
            all_gap.append(gap512.cpu())
            all_names.extend(list(names))

    conv_mat = torch.cat(all_conv, dim=0)  # [T,512,7,7]
    gap_mat = torch.cat(all_gap, dim=0)    # [T,512]

    conv_path, gap_path = resolve_save_paths(args.save)

    torch.save({"names": all_names, "conv5_3": conv_mat}, str(conv_path))
    torch.save({"names": all_names, "gap512": gap_mat}, str(gap_path))

    print("Done.")
    print("Num images:", len(all_names))
    print("conv5_3 shape:", tuple(conv_mat.shape), "->", str(conv_path))
    print("gap512 shape:", tuple(gap_mat.shape), "->", str(gap_path))


if __name__ == "__main__":
    main()
