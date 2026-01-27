import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import face_comp_torch as FCmodel  # 你现有文件：包含 VGGFace_conv 与 IMG_HEIGHT/IMG_WIDTH


# -------------------------
# Dataset
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, img_dir: str, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp")):
        self.img_dir = Path(img_dir)
        self.paths = sorted([p for p in self.img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in: {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, p.name


# -------------------------
# Full VGGFace (conv + fc6/fc7/fc8) that matches keys: fc.fc6.*, fc.fc7.*, fc.fc8.*
# -------------------------
class VGGFaceFull(nn.Module):
    def __init__(self, fc8_out: int):
        super().__init__()
        self.features = FCmodel.VGGFace_conv().features  # 复用同一套 conv 定义（keys: features.conv_*)
        self.fc = nn.ModuleDict({
            "fc6": nn.Linear(512 * 7 * 7, 4096),
            "fc7": nn.Linear(4096, 4096),
            "fc8": nn.Linear(4096, fc8_out),
        })

    def forward(self, x):
        # conv blocks
        for _, layer in self.features.items():
            x = layer(x)
        x = x.view(x.size(0), -1)  # [B, 25088]
        x = F.relu(self.fc["fc6"](x))
        x = F.relu(self.fc["fc7"](x))
        x = self.fc["fc8"](x)
        return x

    def forward_fc7(self, x):
        for _, layer in self.features.items():
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc["fc6"](x))
        x = F.relu(self.fc["fc7"](x))  # [B, 4096]
        return x


def infer_fc8_out_from_state_dict(sd: dict) -> int:
    w = sd["fc.fc8.weight"]  # shape: [out_features, 4096]
    return int(w.shape[0])


def build_preprocess():
    # 与你老师工程一致的 normalize :contentReference[oaicite:1]{index=1}
    return transforms.Compose([
        transforms.Resize((FCmodel.IMG_HEIGHT_VGG16, FCmodel.IMG_WIDTH_VGG16)),
        transforms.ToTensor(),
        transforms.Normalize(
            (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
            (1.0, 1.0, 1.0)
        )
    ])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="vggface.pth (contains conv + fc6/7/8)")
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--save", required=True, help="output .pt path")
    parser.add_argument("--mode", choices=["conv", "fc7"], default="fc7",
                        help="conv: 25088-dim; fc7: 4096-dim")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)  # windows: 0 safest
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    sd = torch.load(args.weights, map_location="cpu")
    fc8_out = infer_fc8_out_from_state_dict(sd)
    print("Inferred fc8 out_features =", fc8_out)

    model_full = VGGFaceFull(fc8_out=fc8_out)

    # 关键：把 state_dict 的 key 对齐到我们定义的模块名
    # state_dict keys: features.conv_*, fc.fc6.*, fc.fc7.*, fc.fc8.*
    # model keys:      features.conv_*, fc.fc6.*, fc.fc7.*, fc.fc8.*  (一致)
    model_full.load_state_dict(sd, strict=True)
    model_full.to(device).eval()

    transform = build_preprocess()
    ds = ImageFolderDataset(args.img_dir, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    all_feats: List[torch.Tensor] = []
    all_names: List[str] = []

    with torch.no_grad():
        for imgs, names in dl:
            imgs = imgs.to(device, non_blocking=True)
            if args.mode == "conv":
                # 直接用 conv 输出（25088）
                x = imgs
                for _, layer in model_full.features.items():
                    x = layer(x)
                x = x.view(x.size(0), -1)
                feats = F.normalize(x, dim=1)
            else:
                # fc7 输出（4096）
                feats = model_full.forward_fc7(imgs)
                feats = F.normalize(feats, dim=1)

            all_feats.append(feats.cpu())
            all_names.extend(list(names))

    feats_mat = torch.cat(all_feats, dim=0)
    os.makedirs(str(Path(args.save).parent), exist_ok=True)
    torch.save({"names": all_names, "feats": feats_mat}, args.save)

    print("Done.")
    print("Num images:", len(all_names))
    print("Feature shape:", tuple(feats_mat.shape))
    print("Saved to:", args.save)


if __name__ == "__main__":
    main()
