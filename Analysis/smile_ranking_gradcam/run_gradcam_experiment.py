from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MODEL_ORDER = ("true", "bitter", "polite")
EXPERIMENT_ORDER = ("native_anchor", "fixed_pair_control")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    weights: Path
    scale_dir: Path
    ranking_items: Path


@dataclass(frozen=True)
class SampledFrame:
    frame_id: str
    input_label: str
    sequence_id: str
    sample_index: int
    source_frame_index: int
    normalized_position: float
    frame_path: Path


@dataclass(frozen=True)
class Anchor:
    level: int
    image_path: Path
    source_image_file: str
    source_frame_path: str


@dataclass(frozen=True)
class NativeAssignment:
    model_name: str
    frame_id: str
    score_0_9: float
    raw_score_unclamped: float
    initial_lower_anchor_level: int | None
    lower_anchor_level: int | None
    fallback_steps: int | None
    p_a_stronger_than_initial_b: float | None
    p_a_stronger_than_b: float | None
    stream_consistency: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare true/bitter/polite SmileComp ranking classifiers with Grad-CAM. "
            "The native experiment uses each model's immediately lower scale anchor; "
            "the fixed-pair control uses the input type's native anchor for all models."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "experiment_config.json",
    )
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--samples_per_sequence", type=int)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_ORDER,
        default=list(EXPERIMENT_ORDER),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max_sequences_per_label",
        type=int,
        help="Smoke-test limit. Omit for the full selected-sequence experiment.",
    )
    parser.add_argument(
        "--max_frames_per_sequence",
        type=int,
        help="Smoke-test limit applied after uniform sampling.",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"sequence_root", "output_root", "model_code_dir", "models", "input_labels"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    return config


def load_model_configs(config: dict[str, Any]) -> dict[str, ModelConfig]:
    models: dict[str, ModelConfig] = {}
    for name in MODEL_ORDER:
        raw = config["models"].get(name)
        if raw is None:
            raise ValueError(f"Missing model config: {name}")
        model_config = ModelConfig(
            name=name,
            weights=Path(raw["weights"]),
            scale_dir=Path(raw["scale_dir"]),
            ranking_items=Path(raw["ranking_items"]),
        )
        for path in (model_config.weights, model_config.scale_dir, model_config.ranking_items):
            if not path.exists():
                raise FileNotFoundError(path)
        models[name] = model_config
    return models


def natural_key(path: Path) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def list_frame_paths(sequence_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_key)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_sequence_ids(ranking_items_path: Path) -> list[str]:
    rows = read_csv_rows(ranking_items_path)
    ids = {str(row["sequence_id"]).strip() for row in rows}
    return sorted(ids, key=lambda value: int(value) if value.isdigit() else value)


def uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 0:
        raise ValueError("samples_per_sequence must be positive")
    count = min(length, count)
    indices = np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()
    return list(dict.fromkeys(indices))


def build_sampled_frames(
    sequence_root: Path,
    input_labels: list[str],
    model_configs: dict[str, ModelConfig],
    samples_per_sequence: int,
    max_sequences_per_label: int | None,
    max_frames_per_sequence: int | None,
) -> list[SampledFrame]:
    sampled: list[SampledFrame] = []
    for input_label in input_labels:
        if input_label not in model_configs:
            raise ValueError(f"No corresponding ranking dataset for input label: {input_label}")
        sequence_ids = selected_sequence_ids(model_configs[input_label].ranking_items)
        if max_sequences_per_label is not None:
            sequence_ids = sequence_ids[: max_sequences_per_label]
        for sequence_id in sequence_ids:
            sequence_dir = sequence_root / input_label / sequence_id
            if not sequence_dir.is_dir():
                raise FileNotFoundError(sequence_dir)
            frame_paths = list_frame_paths(sequence_dir)
            if not frame_paths:
                raise RuntimeError(f"No frames found: {sequence_dir}")
            indices = uniform_indices(len(frame_paths), samples_per_sequence)
            if max_frames_per_sequence is not None:
                indices = indices[: max_frames_per_sequence]
            denominator = max(1, len(frame_paths) - 1)
            for sample_index, source_frame_index in enumerate(indices):
                frame_path = frame_paths[source_frame_index]
                frame_id = (
                    f"{input_label}_seq{int(sequence_id):03d}_"
                    f"s{sample_index:02d}_f{source_frame_index:04d}"
                )
                sampled.append(
                    SampledFrame(
                        frame_id=frame_id,
                        input_label=input_label,
                        sequence_id=sequence_id,
                        sample_index=sample_index,
                        source_frame_index=source_frame_index,
                        normalized_position=source_frame_index / denominator,
                        frame_path=frame_path,
                    )
                )
    return sampled


def load_anchors(model_config: ModelConfig) -> list[Anchor]:
    rows = read_csv_rows(model_config.scale_dir / "scale10_items.csv")
    anchors = []
    for row in rows:
        image_path = model_config.scale_dir / row["reference_image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        anchors.append(
            Anchor(
                level=int(row["scale_level"]),
                image_path=image_path,
                source_image_file=row.get("source_image_file", ""),
                source_frame_path=row.get("source_frame_path", ""),
            )
        )
    anchors.sort(key=lambda item: item.level)
    levels = [anchor.level for anchor in anchors]
    if levels != list(range(10)):
        raise ValueError(f"Expected scale levels 0..9 in {model_config.scale_dir}, found {levels}")
    return anchors


def build_preprocess(fcmodel: Any) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((fcmodel.IMG_HEIGHT_VGG16, fcmodel.IMG_WIDTH_VGG16)),
            transforms.ToTensor(),
            transforms.Normalize(
                (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
                (1.0, 1.0, 1.0),
            ),
        ]
    )


def load_image_tensor(path: Path, preprocess: transforms.Compose, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = preprocess(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def load_resized_rgb(path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def build_model(fcmodel: Any, model_config: ModelConfig, device: torch.device) -> torch.nn.Module:
    model = fcmodel.face_comp_siamese_model_vgg16based_for_predict(
        str(model_config.weights), on_device=device
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_flat_features(model: torch.nn.Module, tensors: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.extractor(tensors).detach()


def score_against_anchors(
    model: torch.nn.Module,
    frame_tensor: torch.Tensor,
    anchor_tensors: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    frame_feature = extract_flat_features(model, frame_tensor)
    anchor_features = extract_flat_features(model, anchor_tensors)
    repeated_frame = frame_feature.repeat(anchor_features.shape[0], 1)
    with torch.no_grad():
        logits_org = model.comp_layer(torch.cat((repeated_frame, anchor_features), dim=1))
        logits_swp = model.comp_layer(torch.cat((anchor_features, repeated_frame), dim=1))
        probs_org = F.softmax(logits_org, dim=1)
        probs_swp = F.softmax(logits_swp, dim=1)
        p_a_stronger = (probs_org[:, 0] + probs_swp[:, 1]) * 0.5
        consistency = (
            torch.minimum(probs_org[:, 0], probs_swp[:, 1])
            + torch.minimum(probs_org[:, 1], probs_swp[:, 0])
        )
    probabilities = p_a_stronger.detach().cpu().numpy()
    consistencies = consistency.detach().cpu().numpy()
    raw_score = float(probabilities.sum() - 0.5)
    score = float(np.clip(raw_score, 0.0, 9.0))
    return probabilities, consistencies, raw_score, score


def immediately_lower_anchor_level(score: float) -> int | None:
    if score <= 0.0:
        return None
    # Largest integer that is strictly below the continuous score.
    level = math.ceil(score - 1e-8) - 1
    return max(0, min(8, level))


def select_supported_lower_anchor(
    score: float,
    p_a_stronger_by_level: np.ndarray,
) -> tuple[int | None, int | None, int | None]:
    """Select the nearest numerically lower anchor that the model actually beats.

    The aggregate 0-9 score does not guarantee a consistent local pairwise
    decision. Start at the mathematically adjacent lower level and step down
    until P(A>B) is strictly greater than 0.5. Returning no selected level is
    preferable to calling an anchor lower when the comparator disagrees.
    """
    initial_level = immediately_lower_anchor_level(score)
    if initial_level is None:
        return None, None, None
    for candidate_level in range(initial_level, -1, -1):
        if float(p_a_stronger_by_level[candidate_level]) > 0.5:
            return initial_level, candidate_level, initial_level - candidate_level
    return initial_level, None, None


def forward_feature_map(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    value = image_tensor
    with torch.no_grad():
        for layer in model.extractor.features.values():
            value = layer(value)
    return value.detach().requires_grad_(True)


def normalize_for_display(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum - minimum <= 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - minimum) / (maximum - minimum)


def gradcam_for_pair(
    model: torch.nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return logits and raw 7x7 standard Grad-CAMs for both classes and inputs.

    Class 0 means A is stronger than B; class 1 means A is weaker than B. Both
    fixed class targets are saved because ReLU Grad-CAM can legitimately be
    empty for one input branch when that branch supplies only negative evidence
    for a class. The targets are never changed to the predicted class.
    """
    feature_a = forward_feature_map(model, image_a)
    feature_b = forward_feature_map(model, image_b)
    logits = model.comp_layer(
        torch.cat((feature_a.flatten(start_dim=1), feature_b.flatten(start_dim=1)), dim=1)
    )

    def make_cam(feature: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feature).sum(dim=1))[0]
        return cam.detach().cpu().numpy().astype(np.float32)

    gradients_zero = torch.autograd.grad(
        logits[0, 0], (feature_a, feature_b), retain_graph=True
    )
    gradients_one = torch.autograd.grad(logits[0, 1], (feature_a, feature_b))
    cams = {
        "a_class0": make_cam(feature_a, gradients_zero[0]),
        "b_class0": make_cam(feature_b, gradients_zero[1]),
        "a_class1": make_cam(feature_a, gradients_one[0]),
        "b_class1": make_cam(feature_b, gradients_one[1]),
    }
    return logits.detach().cpu().numpy()[0], cams


def direct_pair_metrics(
    model: torch.nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        logits_org, logits_swp = model(image_a, image_b)
        probs_org = F.softmax(logits_org, dim=1)
        probs_swp = F.softmax(logits_swp, dim=1)
        p_a_stronger = float(((probs_org[:, 0] + probs_swp[:, 1]) * 0.5).item())
        consistency = float(
            (
                torch.minimum(probs_org[:, 0], probs_swp[:, 1])
                + torch.minimum(probs_org[:, 1], probs_swp[:, 0])
            ).item()
        )
    return p_a_stronger, consistency


def resize_cam(cam_7x7: np.ndarray) -> np.ndarray:
    return cv2.resize(cam_7x7, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def colorize_cam(cam_normalized: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(cam_normalized * 255.0, 0, 255)), cv2.COLORMAP_TURBO
    )
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_cam(original_rgb: np.ndarray, cam_normalized: np.ndarray, alpha: float) -> np.ndarray:
    if float(cam_normalized.max()) <= 0.0:
        return original_rgb.copy()
    heatmap = colorize_cam(cam_normalized).astype(np.float32)
    original = original_rgb.astype(np.float32)
    return np.uint8(np.clip((1.0 - alpha) * original + alpha * heatmap, 0, 255))


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def make_comparison_image(
    original_a: np.ndarray,
    original_b: np.ndarray,
    overlay_a_class0: np.ndarray,
    overlay_b_class0: np.ndarray,
    overlay_a_class1: np.ndarray,
    overlay_b_class1: np.ndarray,
    metadata_line_1: str,
    metadata_line_2: str,
) -> Image.Image:
    panel_size = 224
    header_height = 24
    footer_height = 44
    canvas = Image.new("RGB", (panel_size * 2, header_height + panel_size * 3 + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 6), "A: input frame", fill="black", font=font)
    draw.text((panel_size + 6, 6), "B: immediately lower anchor", fill="black", font=font)
    canvas.paste(Image.fromarray(original_a), (0, header_height))
    canvas.paste(Image.fromarray(original_b), (panel_size, header_height))
    canvas.paste(Image.fromarray(overlay_a_class0), (0, header_height + panel_size))
    canvas.paste(Image.fromarray(overlay_b_class0), (panel_size, header_height + panel_size))
    canvas.paste(Image.fromarray(overlay_a_class1), (0, header_height + panel_size * 2))
    canvas.paste(Image.fromarray(overlay_b_class1), (panel_size, header_height + panel_size * 2))
    for y, label in (
        (header_height + panel_size, "class 0: A > B"),
        (header_height + panel_size * 2, "class 1: A < B"),
    ):
        draw.rectangle((0, y, 102, y + 16), fill="white")
        draw.text((3, y + 2), label, fill="black", font=font)
    footer_y = header_height + panel_size * 3
    draw.text((6, footer_y + 5), metadata_line_1, fill="black", font=font)
    draw.text((6, footer_y + 22), metadata_line_2, fill="black", font=font)
    return canvas


def case_output_paths(
    run_dir: Path,
    experiment: str,
    test_model: str,
    frame: SampledFrame,
    anchor_level: int,
) -> dict[str, Path]:
    directory = run_dir / experiment / f"{test_model}_model" / frame.input_label / f"seq{int(frame.sequence_id):03d}"
    stem = f"{frame.frame_id}_b{anchor_level:02d}"
    return {
        "cam_data": directory / f"{stem}_cams.npz",
        "a_overlay_class0": directory / f"{stem}_a_class0_overlay.png",
        "b_overlay_class0": directory / f"{stem}_b_class0_overlay.png",
        "a_overlay_class1": directory / f"{stem}_a_class1_overlay.png",
        "b_overlay_class1": directory / f"{stem}_b_class1_overlay.png",
        "comparison": directory / f"{stem}_comparison.png",
    }


def generate_case(
    run_dir: Path,
    experiment: str,
    test_model_name: str,
    anchor_source_model: str,
    frame: SampledFrame,
    assignment: NativeAssignment,
    anchor: Anchor,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    overlay_alpha: float,
) -> dict[str, Any]:
    tensor_a = load_image_tensor(frame.frame_path, preprocess, device)
    tensor_b = load_image_tensor(anchor.image_path, preprocess, device)
    p_a_stronger, consistency = direct_pair_metrics(model, tensor_a, tensor_b)
    logits, cams_7x7 = gradcam_for_pair(model, tensor_a, tensor_b)
    cams_224_normalized = {
        name: normalize_for_display(resize_cam(cam)) for name, cam in cams_7x7.items()
    }
    original_a = load_resized_rgb(frame.frame_path)
    original_b = load_resized_rgb(anchor.image_path)
    overlays = {
        "a_class0": overlay_cam(original_a, cams_224_normalized["a_class0"], overlay_alpha),
        "b_class0": overlay_cam(original_b, cams_224_normalized["b_class0"], overlay_alpha),
        "a_class1": overlay_cam(original_a, cams_224_normalized["a_class1"], overlay_alpha),
        "b_class1": overlay_cam(original_b, cams_224_normalized["b_class1"], overlay_alpha),
    }

    paths = case_output_paths(
        run_dir, experiment, test_model_name, frame, anchor.level
    )
    paths["cam_data"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["cam_data"],
        a_class0_cam_7x7_raw=cams_7x7["a_class0"],
        b_class0_cam_7x7_raw=cams_7x7["b_class0"],
        a_class1_cam_7x7_raw=cams_7x7["a_class1"],
        b_class1_cam_7x7_raw=cams_7x7["b_class1"],
        a_class0_cam_224_normalized=cams_224_normalized["a_class0"],
        b_class0_cam_224_normalized=cams_224_normalized["b_class0"],
        a_class1_cam_224_normalized=cams_224_normalized["a_class1"],
        b_class1_cam_224_normalized=cams_224_normalized["b_class1"],
    )
    for name, overlay in overlays.items():
        save_rgb(paths[f"{name.split('_')[0]}_overlay_{name.split('_')[1]}"], overlay)
    comparison = make_comparison_image(
        original_a,
        original_b,
        overlays["a_class0"],
        overlays["b_class0"],
        overlays["a_class1"],
        overlays["b_class1"],
        (
            f"experiment={experiment} model={test_model_name} input={frame.input_label} "
            f"seq={frame.sequence_id} sample={frame.sample_index}"
        ),
        (
            f"score={assignment.score_0_9:.3f} B=L{anchor.level} "
            f"P(A>B)={p_a_stronger:.3f} consistency={consistency:.3f}"
        ),
    )
    comparison.save(paths["comparison"])

    return {
        "case_id": (
            f"{experiment}__{test_model_name}__{frame.frame_id}__"
            f"anchor-{anchor_source_model}-L{anchor.level}"
        ),
        "experiment": experiment,
        "test_model": test_model_name,
        "anchor_source_model": anchor_source_model,
        "input_label": frame.input_label,
        "sequence_id": frame.sequence_id,
        "frame_id": frame.frame_id,
        "sample_index": frame.sample_index,
        "source_frame_index": frame.source_frame_index,
        "normalized_position": f"{frame.normalized_position:.8f}",
        "a_frame_path": str(frame.frame_path),
        "a_score_under_anchor_source_model": f"{assignment.score_0_9:.8f}",
        "a_initial_lower_anchor_level": (
            "" if assignment.initial_lower_anchor_level is None else assignment.initial_lower_anchor_level
        ),
        "b_anchor_level": anchor.level,
        "anchor_fallback_steps": "" if assignment.fallback_steps is None else assignment.fallback_steps,
        "b_anchor_path": str(anchor.image_path),
        "b_anchor_source_image_file": anchor.source_image_file,
        "b_anchor_source_frame_path": anchor.source_frame_path,
        "class_targets": "0:A_stronger_than_B;1:A_weaker_than_B",
        "class0_logit": f"{float(logits[0]):.8f}",
        "class1_logit": f"{float(logits[1]):.8f}",
        "p_a_stronger_than_b": f"{p_a_stronger:.8f}",
        "stream_consistency": f"{consistency:.8f}",
        "pair_relation_supported": int(p_a_stronger > 0.5),
        "fixed_pair_consensus_a_gt_b": "",
        "a_class0_cam_positive_sum": f"{float(cams_7x7['a_class0'].sum()):.10f}",
        "b_class0_cam_positive_sum": f"{float(cams_7x7['b_class0'].sum()):.10f}",
        "a_class1_cam_positive_sum": f"{float(cams_7x7['a_class1'].sum()):.10f}",
        "b_class1_cam_positive_sum": f"{float(cams_7x7['b_class1'].sum()):.10f}",
        "cam_data_path": str(paths["cam_data"]),
        "a_class0_overlay_path": str(paths["a_overlay_class0"]),
        "b_class0_overlay_path": str(paths["b_overlay_class0"]),
        "a_class1_overlay_path": str(paths["a_overlay_class1"]),
        "b_class1_overlay_path": str(paths["b_overlay_class1"]),
        "comparison_path": str(paths["comparison"]),
    }


RESULT_FIELDS = [
    "case_id",
    "experiment",
    "test_model",
    "anchor_source_model",
    "input_label",
    "sequence_id",
    "frame_id",
    "sample_index",
    "source_frame_index",
    "normalized_position",
    "a_frame_path",
    "a_score_under_anchor_source_model",
    "a_initial_lower_anchor_level",
    "b_anchor_level",
    "anchor_fallback_steps",
    "b_anchor_path",
    "b_anchor_source_image_file",
    "b_anchor_source_frame_path",
    "class_targets",
    "class0_logit",
    "class1_logit",
    "p_a_stronger_than_b",
    "stream_consistency",
    "pair_relation_supported",
    "fixed_pair_consensus_a_gt_b",
    "a_class0_cam_positive_sum",
    "b_class0_cam_positive_sum",
    "a_class1_cam_positive_sum",
    "b_class1_cam_positive_sum",
    "cam_data_path",
    "a_class0_overlay_path",
    "b_class0_overlay_path",
    "a_class1_overlay_path",
    "b_class1_overlay_path",
    "comparison_path",
]


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    sequence_root = Path(config["sequence_root"])
    output_root = Path(config["output_root"])
    model_code_dir = Path(config["model_code_dir"])
    input_labels = [str(value) for value in config["input_labels"]]
    samples_per_sequence = args.samples_per_sequence or int(config.get("samples_per_sequence", 10))
    model_configs = load_model_configs(config)

    if not sequence_root.is_dir():
        raise FileNotFoundError(sequence_root)
    if not model_code_dir.is_dir():
        raise FileNotFoundError(model_code_dir)
    run_dir = output_root / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use a new --run_name to preserve prior results."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(model_code_dir))
    fcmodel = importlib.import_module("face_comp_torch")
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    preprocess = build_preprocess(fcmodel)
    anchors_by_model = {
        name: load_anchors(model_config) for name, model_config in model_configs.items()
    }
    sampled_frames = build_sampled_frames(
        sequence_root=sequence_root,
        input_labels=input_labels,
        model_configs=model_configs,
        samples_per_sequence=samples_per_sequence,
        max_sequences_per_label=args.max_sequences_per_label,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    frame_by_id = {frame.frame_id: frame for frame in sampled_frames}
    sampled_rows = [
        {
            "frame_id": frame.frame_id,
            "input_label": frame.input_label,
            "sequence_id": frame.sequence_id,
            "sample_index": frame.sample_index,
            "source_frame_index": frame.source_frame_index,
            "normalized_position": f"{frame.normalized_position:.8f}",
            "frame_path": str(frame.frame_path),
        }
        for frame in sampled_frames
    ]
    write_csv(
        run_dir / "manifests" / "sampled_frames.csv",
        [
            "frame_id",
            "input_label",
            "sequence_id",
            "sample_index",
            "source_frame_index",
            "normalized_position",
            "frame_path",
        ],
        sampled_rows,
    )

    assignments: dict[tuple[str, str], NativeAssignment] = {}
    assignment_rows: list[dict[str, Any]] = []
    print(f"[SETUP] device={device} sampled_frames={len(sampled_frames)} run_dir={run_dir}")

    # Pass 1: each model assigns every A frame a continuous score and native lower anchor.
    for model_name in MODEL_ORDER:
        model_config = model_configs[model_name]
        print(f"[SCORE] loading {model_name} model")
        model = build_model(fcmodel, model_config, device)
        anchor_tensors = torch.cat(
            [
                load_image_tensor(anchor.image_path, preprocess, device)
                for anchor in anchors_by_model[model_name]
            ],
            dim=0,
        )
        for index, frame in enumerate(sampled_frames, start=1):
            frame_tensor = load_image_tensor(frame.frame_path, preprocess, device)
            probabilities, consistencies, raw_score, score = score_against_anchors(
                model, frame_tensor, anchor_tensors
            )
            # Re-evaluate candidate anchors through the exact single-pair path
            # used later for CAM generation. Batched feature extraction can
            # differ by tiny floating-point amounts, especially when A is the
            # same source image as an anchor; those numerical differences must
            # not turn an equal pair into A>B.
            selection_probabilities = probabilities.copy()
            selection_consistencies = consistencies.copy()
            initial_level = immediately_lower_anchor_level(score)
            if initial_level is not None:
                for candidate_level in range(initial_level, -1, -1):
                    direct_probability, direct_consistency = direct_pair_metrics(
                        model,
                        frame_tensor,
                        anchor_tensors[candidate_level : candidate_level + 1],
                    )
                    selection_probabilities[candidate_level] = direct_probability
                    selection_consistencies[candidate_level] = direct_consistency
            initial_level, lower_level, fallback_steps = select_supported_lower_anchor(
                score, selection_probabilities
            )
            assignment = NativeAssignment(
                model_name=model_name,
                frame_id=frame.frame_id,
                score_0_9=score,
                raw_score_unclamped=raw_score,
                initial_lower_anchor_level=initial_level,
                lower_anchor_level=lower_level,
                fallback_steps=fallback_steps,
                p_a_stronger_than_initial_b=(
                    float(selection_probabilities[initial_level])
                    if initial_level is not None
                    else None
                ),
                p_a_stronger_than_b=(
                    float(selection_probabilities[lower_level]) if lower_level is not None else None
                ),
                stream_consistency=(
                    float(selection_consistencies[lower_level]) if lower_level is not None else None
                ),
            )
            assignments[(model_name, frame.frame_id)] = assignment
            assignment_rows.append(
                {
                    "model_name": model_name,
                    "frame_id": frame.frame_id,
                    "input_label": frame.input_label,
                    "sequence_id": frame.sequence_id,
                    "sample_index": frame.sample_index,
                    "score_0_9": f"{score:.8f}",
                    "raw_score_unclamped": f"{raw_score:.8f}",
                    "initial_lower_anchor_level": "" if initial_level is None else initial_level,
                    "p_a_stronger_than_initial_b": (
                        ""
                        if initial_level is None
                        else f"{selection_probabilities[initial_level]:.8f}"
                    ),
                    "selected_lower_anchor_level": "" if lower_level is None else lower_level,
                    "fallback_steps": "" if fallback_steps is None else fallback_steps,
                    "p_a_stronger_than_native_b": (
                        ""
                        if lower_level is None
                        else f"{selection_probabilities[lower_level]:.8f}"
                    ),
                    "native_pair_stream_consistency": (
                        ""
                        if lower_level is None
                        else f"{selection_consistencies[lower_level]:.8f}"
                    ),
                }
            )
            if index % 50 == 0 or index == len(sampled_frames):
                print(f"[SCORE] {model_name}: {index}/{len(sampled_frames)}")
        del model, anchor_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(
        run_dir / "manifests" / "frame_scores_and_native_anchors.csv",
        [
            "model_name",
            "frame_id",
            "input_label",
            "sequence_id",
            "sample_index",
            "score_0_9",
            "raw_score_unclamped",
            "initial_lower_anchor_level",
            "p_a_stronger_than_initial_b",
            "selected_lower_anchor_level",
            "fallback_steps",
            "p_a_stronger_than_native_b",
            "native_pair_stream_consistency",
        ],
        assignment_rows,
    )

    result_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    # Pass 2: load one test model at a time and generate both experiment families.
    for test_model_name in MODEL_ORDER:
        print(f"[CAM] loading {test_model_name} model")
        model = build_model(fcmodel, model_configs[test_model_name], device)

        if "native_anchor" in args.experiments:
            for index, frame in enumerate(sampled_frames, start=1):
                assignment = assignments[(test_model_name, frame.frame_id)]
                if assignment.lower_anchor_level is None:
                    skipped_rows.append(
                        {
                            "experiment": "native_anchor",
                            "test_model": test_model_name,
                            "frame_id": frame.frame_id,
                            "reason": "no_lower_anchor_satisfies_p_a_gt_b_over_0.5",
                        }
                    )
                    continue
                anchor = anchors_by_model[test_model_name][assignment.lower_anchor_level]
                result_rows.append(
                    generate_case(
                        run_dir=run_dir,
                        experiment="native_anchor",
                        test_model_name=test_model_name,
                        anchor_source_model=test_model_name,
                        frame=frame,
                        assignment=assignment,
                        anchor=anchor,
                        model=model,
                        preprocess=preprocess,
                        device=device,
                        overlay_alpha=args.overlay_alpha,
                    )
                )
                if index % 25 == 0 or index == len(sampled_frames):
                    print(f"[CAM] native {test_model_name}: {index}/{len(sampled_frames)}")

        if "fixed_pair_control" in args.experiments:
            for index, frame in enumerate(sampled_frames, start=1):
                anchor_source_model = frame.input_label
                assignment = assignments[(anchor_source_model, frame.frame_id)]
                if assignment.lower_anchor_level is None:
                    skipped_rows.append(
                        {
                            "experiment": "fixed_pair_control",
                            "test_model": test_model_name,
                            "frame_id": frame.frame_id,
                            "reason": "source_model_has_no_lower_anchor_satisfying_p_a_gt_b_over_0.5",
                        }
                    )
                    continue
                anchor = anchors_by_model[anchor_source_model][assignment.lower_anchor_level]
                result_rows.append(
                    generate_case(
                        run_dir=run_dir,
                        experiment="fixed_pair_control",
                        test_model_name=test_model_name,
                        anchor_source_model=anchor_source_model,
                        frame=frame,
                        assignment=assignment,
                        anchor=anchor,
                        model=model,
                        preprocess=preprocess,
                        device=device,
                        overlay_alpha=args.overlay_alpha,
                    )
                )
                if index % 25 == 0 or index == len(sampled_frames):
                    print(f"[CAM] fixed {test_model_name}: {index}/{len(sampled_frames)}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixed_groups: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        if row["experiment"] == "fixed_pair_control":
            fixed_groups.setdefault(str(row["frame_id"]), []).append(row)
    fixed_consensus_by_frame = {
        frame_id: (
            len(group) == len(MODEL_ORDER)
            and all(int(row["pair_relation_supported"]) == 1 for row in group)
        )
        for frame_id, group in fixed_groups.items()
    }
    for row in result_rows:
        if row["experiment"] == "fixed_pair_control":
            row["fixed_pair_consensus_a_gt_b"] = int(
                fixed_consensus_by_frame[str(row["frame_id"])]
            )

    results_path = run_dir / "manifests" / "gradcam_results.csv"
    write_csv(results_path, RESULT_FIELDS, result_rows)
    write_csv(
        run_dir / "manifests" / "skipped_cases.csv",
        ["experiment", "test_model", "frame_id", "reason"],
        skipped_rows,
    )

    supported_count = sum(int(row["pair_relation_supported"]) for row in result_rows)
    native_rows = [row for row in result_rows if row["experiment"] == "native_anchor"]
    fixed_rows = [row for row in result_rows if row["experiment"] == "fixed_pair_control"]
    fixed_consensus_frames = sum(fixed_consensus_by_frame.values())
    summary = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "device": str(device),
        "samples_per_sequence": samples_per_sequence,
        "input_labels": input_labels,
        "models": list(MODEL_ORDER),
        "experiments": args.experiments,
        "num_sampled_frames": len(sampled_frames),
        "sampled_sequences": {
            label: sorted(
                {frame.sequence_id for frame in sampled_frames if frame.input_label == label},
                key=lambda value: int(value),
            )
            for label in input_labels
        },
        "num_gradcam_cases": len(result_rows),
        "num_pair_relation_supported": supported_count,
        "num_pair_relation_not_supported": len(result_rows) - supported_count,
        "num_native_cases": len(native_rows),
        "num_native_pair_relation_not_supported": sum(
            int(row["pair_relation_supported"]) == 0 for row in native_rows
        ),
        "num_fixed_pair_cases": len(fixed_rows),
        "num_fixed_pair_consensus_frames": fixed_consensus_frames,
        "num_fixed_pair_non_consensus_frames": len(fixed_consensus_by_frame) - fixed_consensus_frames,
        "num_skipped_cases": len(skipped_rows),
        "class_targets": {
            "class_0": "first image A is stronger than second image B",
            "class_1": "first image A is weaker than second image B",
        },
        "native_anchor_definition": (
            "Start at the largest integer scale level strictly below score_0_9, then step "
            "down until the source model gives P(A stronger than B) > 0.5; skip if none exists."
        ),
        "fixed_pair_definition": (
            "For each input frame, B is selected by the ranking model matching the input label; "
            "the identical A/B pair is then evaluated by all three test models."
        ),
        "results_csv": str(results_path),
    }
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] cases={len(result_rows)} skipped={len(skipped_rows)} run_dir={run_dir}")


if __name__ == "__main__":
    main()
