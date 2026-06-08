from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

import sys


ANALYSIS_FEATURE_DIR = Path(__file__).resolve().parents[2] / "feature_extractor"
if str(ANALYSIS_FEATURE_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_FEATURE_DIR))

import face_comp_torch as FCmodel  # type: ignore  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LAYER_NAMES = ("maxp_5_3", "relu_5_3", "relu_4_3")
AGG_NAMES = ("agg_A", "agg_B", "agg_C", "agg_D")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
TARGET_RATIO = 4 / 3
PADDING = 6
BG = (255, 255, 255)
THRESHOLDS = (0.5, 0.75)
COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class HeatmapConfig:
    polite_source_dir: Path
    truesmile_source_dir: Path
    ambiguous_source_dir: Path
    vggface_model_path: Path
    heatmap_alpha: float
    interpolation_method: str
    output_dir: Path
    colormap: str = "turbo"
    device: str = "cuda:0"


def parse_source_dat(path: Path, output_override: Path | None = None) -> HeatmapConfig:
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
        "heatmap_alpha",
        "interpolation_method",
    }
    missing = sorted(required - set(kv))
    if missing:
        raise RuntimeError(f"Missing keys in source.dat: {missing}")

    out_dir = output_override if output_override is not None else Path(kv.get("output_dir", r"E:\Single_frame_smile\Analysis\Heatmap\output"))
    return HeatmapConfig(
        polite_source_dir=Path(kv["polite_source_dir"]),
        truesmile_source_dir=Path(kv["truesmile_source_dir"]),
        ambiguous_source_dir=Path(kv["ambiguous_source_dir"]),
        vggface_model_path=Path(kv["vggface_model_path"]),
        heatmap_alpha=float(kv["heatmap_alpha"]),
        interpolation_method=kv["interpolation_method"],
        output_dir=Path(out_dir),
    )


def parse_source_dat(path: Path, output_override: Path | None = None) -> HeatmapConfig:
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
        "heatmap_alpha",
        "interpolation_method",
    }
    missing = sorted(required - set(kv))
    if missing:
        raise RuntimeError(f"Missing keys in source.dat: {missing}")

    out_dir = output_override if output_override is not None else Path(kv.get("output_dir", r"E:\Single_frame_smile\Analysis\Heatmap\output"))
    return HeatmapConfig(
        polite_source_dir=Path(kv["polite_source_dir"]),
        truesmile_source_dir=Path(kv["truesmile_source_dir"]),
        ambiguous_source_dir=Path(kv["ambiguous_source_dir"]),
        vggface_model_path=Path(kv["vggface_model_path"]),
        heatmap_alpha=float(kv["heatmap_alpha"]),
        interpolation_method=kv["interpolation_method"],
        output_dir=Path(out_dir),
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


class VGGFaceMultiLayerExtractor(nn.Module):
    def __init__(self, weight_path: Path):
        super().__init__()
        self.backbone = FCmodel.VGGFace_conv()
        sd = torch.load(str(weight_path), map_location="cpu")
        self.backbone.load_state_dict(sd, strict=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name, layer in self.backbone.features.items():
            x = layer(x)
            if name in LAYER_NAMES:
                outputs[name] = x
        return outputs


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
    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
    }
    if colormap not in cmap_map:
        raise RuntimeError(f"Unsupported colormap: {colormap}")
    img_u8 = np.uint8(np.clip(arr01 * 255.0, 0, 255))
    heat_bgr = cv2.applyColorMap(img_u8, cmap_map[colormap])
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(original_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base = original_rgb.astype(np.float32)
    heat = heat_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat
    return np.uint8(np.clip(out, 0, 255))


def save_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def aggregate_map(delta: np.ndarray, agg_name: str) -> np.ndarray:
    abs_delta = np.abs(delta)
    if agg_name == "agg_A":
        return abs_delta.mean(axis=0).astype(np.float32)
    if agg_name == "agg_B":
        return np.linalg.norm(delta, axis=0).astype(np.float32)
    if agg_name == "agg_C":
        return abs_delta.sum(axis=0).astype(np.float32)
    if agg_name == "agg_D":
        denom = np.linalg.norm(delta * 0 + delta + 0, axis=0)  # placeholder to keep shape path simple
        raise RuntimeError("agg_D requires both current and start features; use aggregate_map_relative()")
    raise RuntimeError(f"Unknown aggregation: {agg_name}")


def aggregate_map_relative(delta: np.ndarray, start: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(delta, axis=0)
    den = np.linalg.norm(start, axis=0) + 1e-8
    return (num / den).astype(np.float32)


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
    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
    }
    if colormap not in cmap_map:
        raise RuntimeError(f"Unsupported colormap: {colormap}")
    img_u8 = np.uint8(np.clip(arr01 * 255.0, 0, 255))
    heat_bgr = cv2.applyColorMap(img_u8, cmap_map[colormap])
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(original_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base = original_rgb.astype(np.float32)
    heat = heat_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat
    return np.uint8(np.clip(out, 0, 255))


def save_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def choose_grid(n: int) -> tuple[int, int]:
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        ratio = cols / rows
        diff = abs(ratio - TARGET_RATIO)
        score = (diff, rows * cols)
        if best is None or score < best[0]:
            best = (score, cols, rows)
    assert best is not None
    return best[1], best[2]


def build_montage(images: list[Path], out_path: Path) -> None:
    if not images:
        return
    pil_images = [Image.open(p).convert("RGB") for p in images]
    w = max(img.width for img in pil_images)
    h = max(img.height for img in pil_images)
    cols, rows = choose_grid(len(pil_images))
    canvas_w = cols * w + (cols + 1) * PADDING
    canvas_h = rows * h + (rows + 1) * PADDING
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    for idx, img in enumerate(pil_images):
        r = idx // cols
        c = idx % cols
        x = PADDING + c * (w + PADDING)
        y = PADDING + r * (h + PADDING)
        if img.size != (w, h):
            img = ImageOps.pad(img, (w, h), color=BG)
        canvas.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


class HeatmapMultiPipeline:
    def __init__(self, config: HeatmapConfig):
        self.cfg = config
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def process_sequence(self, model: VGGFaceMultiLayerExtractor, preprocess: transforms.Compose, class_name: str, img_dir: Path, device: torch.device) -> dict:
        images = list_sorted_images(img_dir)
        img_pils = [Image.open(p).convert("RGB") for p in images]
        img_tensors = [preprocess(img).unsqueeze(0).to(device) for img in img_pils]

        with torch.no_grad():
            start_feats = model(img_tensors[0])

        layer_outputs: dict[str, dict[str, dict]] = {layer: {agg: {} for agg in AGG_NAMES} for layer in LAYER_NAMES}
        summary_files: list[dict] = []

        for frame_idx, (img_path, img_pil, img_tensor) in enumerate(zip(images, img_pils, img_tensors)):
            with torch.no_grad():
                curr_feats = model(img_tensor)
            orig_rgb = np.array(img_pil, dtype=np.uint8)
            h, w = orig_rgb.shape[:2]
            stem = img_path.stem

            for layer_name in LAYER_NAMES:
                start_map = start_feats[layer_name][0].cpu().numpy()
                curr_map = curr_feats[layer_name][0].cpu().numpy()
                delta = curr_map - start_map

                for agg_name in AGG_NAMES:
                    if agg_name == "agg_D":
                        heat_raw = aggregate_map_relative(delta, start_map)
                    else:
                        heat_raw = aggregate_map(delta, agg_name)

                    heat_norm = normalize_heatmap(heat_raw)
                    heat_up = resize_heatmap(heat_norm, w, h, self.cfg.interpolation_method)
                    heat_rgb = apply_colormap(heat_up, self.cfg.colormap)
                    overlay_rgb = overlay_heatmap(orig_rgb, heat_rgb, self.cfg.heatmap_alpha)

                    out_dir = self.cfg.output_dir / layer_name / agg_name / class_name
                    out_original = out_dir / f"{stem}_original.png"
                    out_heatmap = out_dir / f"{stem}_heatmap.png"
                    out_overlay = out_dir / f"{stem}_overlay.png"
                    out_heatmap_raw = out_dir / f"{stem}_heatmap_raw.npy"
                    out_heatmap_norm = out_dir / f"{stem}_heatmap.npy"

                    save_image(out_original, orig_rgb)
                    save_image(out_heatmap, heat_rgb)
                    save_image(out_overlay, overlay_rgb)
                    np.save(out_heatmap_raw, heat_raw.astype(np.float32))
                    np.save(out_heatmap_norm, heat_up.astype(np.float32))

                    layer_outputs[layer_name][agg_name][frame_idx] = {
                        "source_file": img_path.name,
                        "original": str(out_original),
                        "heatmap": str(out_heatmap),
                        "overlay": str(out_overlay),
                        "heatmap_raw_npy": str(out_heatmap_raw),
                        "heatmap_npy": str(out_heatmap_norm),
                        "raw_small_shape": list(heat_raw.shape),
                    }

            summary_files.append({"source_file": img_path.name, "frame_index": frame_idx})

        # sequence-level outputs
        for layer_name in LAYER_NAMES:
            for agg_name in AGG_NAMES:
                out_dir = self.cfg.output_dir / layer_name / agg_name / class_name
                ordered = [layer_outputs[layer_name][agg_name][i] for i in sorted(layer_outputs[layer_name][agg_name])]
                heatmap_paths = [Path(item["heatmap"]) for item in ordered]
                overlay_paths = [Path(item["overlay"]) for item in ordered]
                build_montage(heatmap_paths, out_dir / "heatmap_still.png")
                build_montage(overlay_paths, out_dir / "overlay_still.png")

                raw_maps = [np.load(item["heatmap_raw_npy"], allow_pickle=False) for item in ordered]
                point_matrix = np.stack([m.reshape(-1) for m in raw_maps], axis=1)  # [H*W, T]
                fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
                im = ax.imshow(point_matrix, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(f"pointwise_timeseries_heatmap ({layer_name}, {agg_name}, {class_name})")
                ax.set_xlabel("Time index")
                ax.set_ylabel("Spatial position index")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(out_dir / "pointwise_timeseries_heatmap.png")
                plt.close(fig)

                for threshold in THRESHOLDS:
                    area = []
                    total = []
                    for item in ordered:
                        arr = np.load(item["heatmap_npy"], allow_pickle=False)
                        mask = arr > threshold
                        area.append(int(mask.sum()))
                        total.append(float(arr[mask].sum()))

                    x = list(range(len(ordered)))
                    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
                    ax.plot(x, area, linewidth=2.0, color=COLORS[class_name])
                    ax.set_title(f"Area of heatmap > {threshold} over time ({layer_name}, {agg_name}, {class_name})")
                    ax.set_xlabel("Frame index")
                    ax.set_ylabel("Area (pixel count)")
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"heatmap_area_over_threshold_{str(threshold).replace('.', '')}_over_time.png")
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
                    ax.plot(x, total, linewidth=2.0, color=COLORS[class_name])
                    ax.set_title(f"Sum of heatmap values > {threshold} over time ({layer_name}, {agg_name}, {class_name})")
                    ax.set_xlabel("Frame index")
                    ax.set_ylabel("Sum of heat values")
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"heatmap_sum_over_threshold_{str(threshold).replace('.', '')}_over_time.png")
                    plt.close(fig)

        return {
            "source_dir": str(img_dir),
            "num_frames": len(images),
            "files": summary_files,
        }

    def run(self) -> None:
        device = torch.device(self.cfg.device if (self.cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        preprocess = build_preprocess()
        model = VGGFaceMultiLayerExtractor(self.cfg.vggface_model_path).to(device)

        sequence_map = {
            "polite": self.cfg.polite_source_dir,
            "truesmile": self.cfg.truesmile_source_dir,
            "ambiguous": self.cfg.ambiguous_source_dir,
        }

        summary = {
            "model_path": str(self.cfg.vggface_model_path),
            "output_dir": str(self.cfg.output_dir),
            "layers": list(LAYER_NAMES),
            "aggregations": list(AGG_NAMES),
            "alpha": self.cfg.heatmap_alpha,
            "interpolation_method": self.cfg.interpolation_method,
            "colormap": self.cfg.colormap,
            "device": str(device),
            "sequences": {},
        }

        for class_name, src_dir in sequence_map.items():
            seq_summary = self.process_sequence(model, preprocess, class_name, src_dir, device)
            summary["sequences"][class_name] = seq_summary
            print(f"[HEATMAP_V2] {class_name}: processed {seq_summary['num_frames']} frames")

        self.save_json(self.cfg.output_dir / "summary_v2.json", summary)
        print(f"[HEATMAP_V2] Done. Summary saved to: {self.cfg.output_dir / 'summary_v2.json'}")


def main() -> None:
    source_dat = Path(__file__).resolve().parents[1] / "source.dat"
    cfg = parse_source_dat(source_dat, output_override=Path(r"E:\Matsuda_data\heapmap"))
    pipeline = HeatmapMultiPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
