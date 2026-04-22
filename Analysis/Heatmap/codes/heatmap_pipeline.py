from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import sys


ANALYSIS_FEATURE_DIR = Path(__file__).resolve().parents[2] / "feature_extractor"
if str(ANALYSIS_FEATURE_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_FEATURE_DIR))

import face_comp_torch as FCmodel  # type: ignore  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class HeatmapConfig:
    polite_source_dir: Path
    truesmile_source_dir: Path
    ambiguous_source_dir: Path
    vggface_model_path: Path
    output_dir: Path
    heatmap_alpha: float
    interpolation_method: str
    colormap: str = "turbo"
    device: str = "cuda:0"


def parse_source_dat(path: Path) -> HeatmapConfig:
    kv: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"Invalid line in source.dat: {line}")
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

    required = {
        "polite_source_dir",
        "truesmile_source_dir",
        "ambiguous_source_dir",
        "vggface_model_path",
        "output_dir",
        "heatmap_alpha",
        "interpolation_method",
    }
    missing = sorted(required - set(kv))
    if missing:
        raise RuntimeError(f"Missing keys in source.dat: {missing}")

    return HeatmapConfig(
        polite_source_dir=Path(kv["polite_source_dir"]),
        truesmile_source_dir=Path(kv["truesmile_source_dir"]),
        ambiguous_source_dir=Path(kv["ambiguous_source_dir"]),
        vggface_model_path=Path(kv["vggface_model_path"]),
        output_dir=Path(kv["output_dir"]),
        heatmap_alpha=float(kv["heatmap_alpha"]),
        interpolation_method=kv["interpolation_method"],
    )


def extract_last_int(path: Path) -> int:
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    if not m:
        raise RuntimeError(f"Cannot parse frame index from filename: {path.name}")
    return int(m.group(1))


def list_sorted_images(img_dir: Path) -> list[Path]:
    paths = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise RuntimeError(f"No images found in: {img_dir}")
    return sorted(paths, key=extract_last_int)


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((FCmodel.IMG_HEIGHT_VGG16, FCmodel.IMG_WIDTH_VGG16)),
            transforms.ToTensor(),
            transforms.Normalize(
                (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
                (1.0, 1.0, 1.0),
            ),
        ]
    )


class VGGFaceConv53Heatmap(nn.Module):
    def __init__(self, weight_path: Path):
        super().__init__()
        self.backbone = FCmodel.VGGFace_conv()
        sd = torch.load(str(weight_path), map_location="cpu")
        self.backbone.load_state_dict(sd, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep consistency with the existing conv5_3 extractor in this repo:
        # return the 512x7x7 activation after maxp_5_3.
        for name, layer in self.backbone.features.items():
            x = layer(x)
            if name == "maxp_5_3":
                break
        return x


def normalize_heatmap(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def resize_heatmap(arr: np.ndarray, width: int, height: int, interpolation_method: str) -> np.ndarray:
    inter_map = {
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "bicubic": cv2.INTER_CUBIC,
    }
    if interpolation_method not in inter_map:
        raise RuntimeError(f"Unsupported interpolation method: {interpolation_method}")
    return cv2.resize(arr, (width, height), interpolation=inter_map[interpolation_method])


def apply_colormap(arr01: np.ndarray, colormap: str) -> np.ndarray:
    color_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
    }
    if colormap not in color_map:
        raise RuntimeError(f"Unsupported colormap: {colormap}")
    img_u8 = np.uint8(np.clip(arr01 * 255.0, 0, 255))
    heat_bgr = cv2.applyColorMap(img_u8, color_map[colormap])
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(original_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base = original_rgb.astype(np.float32)
    heat = heat_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat
    return np.uint8(np.clip(out, 0, 255))


def save_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def process_sequence(
    model: VGGFaceConv53Heatmap,
    preprocess: transforms.Compose,
    img_dir: Path,
    output_dir: Path,
    alpha: float,
    interpolation_method: str,
    colormap: str,
    device: torch.device,
) -> dict:
    images = list_sorted_images(img_dir)
    summary = {"source_dir": str(img_dir), "num_frames": len(images), "files": []}

    model.eval()
    with torch.no_grad():
        for img_path in images:
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            fmap = model(img_tensor)[0].cpu().numpy()  # [512, 7, 7]
            heat_2d = fmap.mean(axis=0)  # channel-wise mean
            heat_norm = normalize_heatmap(heat_2d)

            orig_rgb = np.array(img_pil, dtype=np.uint8)
            h, w = orig_rgb.shape[:2]
            heat_up = resize_heatmap(heat_norm, w, h, interpolation_method)
            heat_rgb = apply_colormap(heat_up, colormap)
            overlay_rgb = overlay_heatmap(orig_rgb, heat_rgb, alpha)

            stem = img_path.stem
            out_original = output_dir / f"{stem}_original.png"
            out_heatmap = output_dir / f"{stem}_heatmap.png"
            out_overlay = output_dir / f"{stem}_overlay.png"
            out_npy = output_dir / f"{stem}_heatmap.npy"

            save_image(out_original, orig_rgb)
            save_image(out_heatmap, heat_rgb)
            save_image(out_overlay, overlay_rgb)
            np.save(out_npy, heat_up.astype(np.float32))

            summary["files"].append(
                {
                    "source_file": img_path.name,
                    "original": str(out_original),
                    "heatmap": str(out_heatmap),
                    "overlay": str(out_overlay),
                    "heatmap_npy": str(out_npy),
                }
            )
    return summary


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    source_dat = base_dir / "source.dat"
    cfg = parse_source_dat(source_dat)

    device = torch.device(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    preprocess = build_preprocess()
    model = VGGFaceConv53Heatmap(cfg.vggface_model_path).to(device)

    sequence_map = {
        "polite": cfg.polite_source_dir,
        "truesmile": cfg.truesmile_source_dir,
        "ambiguous": cfg.ambiguous_source_dir,
    }

    all_summary: dict[str, dict] = {
        "model_path": str(cfg.vggface_model_path),
        "output_dir": str(cfg.output_dir),
        "target_layer": "conv5_3_by_project_convention_maxp_5_3",
        "alpha": cfg.heatmap_alpha,
        "interpolation_method": cfg.interpolation_method,
        "colormap": cfg.colormap,
        "device": str(device),
        "sequences": {},
    }

    for class_name, src_dir in sequence_map.items():
        out_dir = cfg.output_dir / class_name
        summary = process_sequence(
            model=model,
            preprocess=preprocess,
            img_dir=src_dir,
            output_dir=out_dir,
            alpha=cfg.heatmap_alpha,
            interpolation_method=cfg.interpolation_method,
            colormap=cfg.colormap,
            device=device,
        )
        all_summary["sequences"][class_name] = summary
        print(f"[HEATMAP] {class_name}: processed {summary['num_frames']} frames")

    summary_path = cfg.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HEATMAP] Done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
