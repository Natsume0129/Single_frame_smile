"""Frozen participant-specific keep-ROI masks estimated from training images only.

The mask is shared by all expression types and every frame for a participant.
No per-frame landmark or hidden-region pixel information enters masked inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "output/monotonic_accepted_training_20261101/accepted_manifest.json"
OUT = ROOT / "output/regional_ranking_extension_20261103/masks"
REGIONS = ("periocular", "nose", "mouth_perioral", "upper_face", "lower_face")
RGB_MEAN = np.array([129.1863, 104.7624, 93.5940], dtype=np.float32)
SIZE = (224, 224)
LABELS = ("polite", "bitter", "true")
PROTOCOL = "train_only_fixed_participant_keep_roi_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_masks(person: str) -> dict[str, np.ndarray]:
    """Load fixed boolean masks; True means retain the original RGB pixel."""
    path = OUT / person / "masks.npz"
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].astype(bool) for name in (*REGIONS, "full")}
    for name, mask in result.items():
        if mask.shape != SIZE or not mask.any():
            raise ValueError(f"Invalid fixed mask: {person}/{name}")
    return result


def mask_image(image: Image.Image | np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep ROI; return float32 RGB [0,255] with exact VGG mean elsewhere.

    Do not convert this result through uint8/PIL for model preprocessing: retain
    float32 so hidden pixels become exactly zero after RGB-mean subtraction.
    For visual PNG previews only, round/clip to uint8.
    """
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if rgb.shape != (*SIZE, 3) or mask.shape != SIZE:
        raise ValueError(f"Expected 224x224 RGB and mask; got {rgb.shape}, {mask.shape}")
    return np.where(mask.astype(bool)[..., None], rgb.astype(np.float32), RGB_MEAN)


def _samples(rows: list[dict], person: str, per_label: int = 10) -> list[dict]:
    selected = []
    seen = set()
    for label in LABELS:
        clips = sorted((x for x in rows if x["person"] == person and x["label"] == label
                        and x["split"] == "train" and x["sample_frames"]), key=lambda x: x["id"])
        if not clips:
            raise ValueError(f"No selected training images: {person}/{label}")
        # Distinct clips when possible; cycling supports small train partitions.
        indices = np.rint(np.linspace(0, len(clips) - 1, min(per_label, len(clips)))).astype(int)
        for j, idx in enumerate(indices):
            clip = clips[int(idx)]
            frames = sorted(clip["sample_frames"], key=lambda f: f["source_frame"])
            phase = (j % 3) / 2
            frame = frames[round(phase * (len(frames) - 1))]
            path = Path(frame["path"])
            digest = sha(path)
            if frame.get("sha256") and frame["sha256"] != digest:
                raise ValueError(f"Training image SHA mismatch: {path}")
            if digest in seen:
                continue
            seen.add(digest)
            selected.append(dict(person=person, label=label, clip_id=clip["id"],
                                 split="train", source_group=clip.get("split_group_id") or clip.get("group_id", clip["id"]),
                                 source_frame=frame["source_frame"], path=str(path), sha256=digest,
                                 selection_phase=phase))
    return selected


def _draw_samples(person: str, selected: list[dict], masks: dict[str, np.ndarray], target: Path):
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 18)
    small = ImageFont.truetype(str(font_path), 15)
    labels = ["Original", "Eyes / periocular", "Nose / alar", "Mouth / corners", "Upper face", "Lower face"]
    selected_rows = []
    for label in LABELS:
        candidates = [s for s in selected if s["label"] == label]
        selected_rows.append(min(candidates, key=lambda s: abs(s["selection_phase"] - .5)))
    cell, gap, left, header = 224, 10, 92, 76
    sheet = Image.new("RGB", (left + 6 * (cell + gap) + gap, header + 3 * (cell + 40) + 42), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{person} | Fixed TRAIN-only masks | Retain indicated region; constant RGB mean outside", fill="black", font=font)
    for col, label in enumerate(labels):
        draw.text((left + col * (cell + gap), 47), label, fill="black", font=small)
    previews = []
    for row, record in enumerate(selected_rows):
        rgb = np.asarray(Image.open(record["path"]).convert("RGB"))
        y = header + row * (cell + 40)
        draw.text((8, y + 100), record["label"].capitalize(), fill="black", font=small)
        for col, name in enumerate(("full", *REGIONS)):
            rendered = np.clip(np.rint(mask_image(rgb, masks[name])), 0, 255).astype(np.uint8)
            sheet.paste(Image.fromarray(rendered), (left + col * (cell + gap), y))
        draw.text((left, y + cell + 5), f"Train clip {record['clip_id']} | source frame {record['source_frame']}", fill="#555555", font=small)
        previews.append(record)
    draw.text((12, sheet.height - 30), "Identical spatial masks for every frame and all three expression types within this participant. No hidden-pixel-derived fill.", fill="#555555", font=small)
    sheet.save(target / "sample_montage.png")
    (target / "preview_sources.json").write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(device: str = "cuda", persons: list[str] | None = None):
    """Prepare frozen masks and provenance; reject replacing an incompatible freeze."""
    import face_alignment
    import torch

    sys.path.insert(0, str(ROOT / "Analysis/smile_ranking_gradcam"))
    from build_region_mask_samples import build_natural_masks, morphology

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_hash = sha(MANIFEST)
    persons = persons or sorted({x["person"] for x in rows})
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    aligner = None
    report = []
    relevant = {"periocular": np.r_[17:27, 36:48], "nose": np.arange(27, 36),
                "mouth_perioral": np.arange(48, 68)}
    critical = {"periocular": [36, 39, 42, 45], "nose": [31, 35], "mouth_perioral": [48, 54]}
    for person in persons:
        dest = OUT / person
        dest.mkdir(exist_ok=True)
        provenance_path = dest / "provenance.json"
        if provenance_path.exists():
            prior = json.loads(provenance_path.read_text(encoding="utf-8"))
            if prior["manifest_sha256"] != manifest_hash or prior["protocol"] != PROTOCOL:
                raise ValueError(f"Refusing to overwrite frozen masks: {person}")
            load_masks(person)
            report.append(prior)
            print(f"{person}: existing frozen masks", flush=True)
            continue
        selected = _samples(rows, person)
        if len(selected) < 15:
            raise ValueError(f"Need at least 15 training-only geometry samples: {person}")
        landmark_dir = dest / "training_landmarks"
        landmark_dir.mkdir(exist_ok=True)
        landmarks = []
        for i, record in enumerate(selected):
            image = np.asarray(Image.open(record["path"]).convert("RGB"))
            if image.shape != (*SIZE, 3):
                raise ValueError(f"Unexpected crop shape: {record['path']}")
            cached = landmark_dir / (record["sha256"] + ".npy")
            if cached.exists():
                points = np.load(cached, allow_pickle=False)
            else:
                if aligner is None:
                    aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device,
                                                           flip_input=False, compile=False)
                predictions = aligner.get_landmarks_from_image(image)
                if predictions is None or len(predictions) != 1:
                    raise ValueError(f"Expected exactly one landmark face: {record['path']}")
                points = np.array(predictions[0], dtype=np.float32)
                np.save(cached, points)
            if points.shape != (68, 2) or not np.isfinite(points).all():
                raise ValueError(f"Invalid landmarks: {record['path']}")
            landmarks.append(points)
            record["landmarks_file"] = str(cached)
            print(f"{person}: train geometry {i+1}/{len(selected)}", flush=True)
        points = np.stack(landmarks)
        median = np.median(points, axis=0)
        face, natural, split_y = build_natural_masks(SIZE, median)
        masks, expansions, coverage, critical_coverage = {}, {}, {}, {}
        # Two fixed pixels of minimum margin, at most twelve, chosen solely from
        # the 95th percentile training landmark distance outside the median ROI.
        # This covers typical crop movement without making the mask follow smiles.
        for region, indices in relevant.items():
            coords = np.rint(points[:, indices, :]).astype(int).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0], 0, SIZE[1] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, SIZE[0] - 1)
            distance = distance_transform_edt(~natural[region])
            corner_coords = np.rint(points[:, critical[region], :]).astype(int).reshape(-1, 2)
            corner_coords[:, 0] = np.clip(corner_coords[:, 0], 0, SIZE[1] - 1)
            corner_coords[:, 1] = np.clip(corner_coords[:, 1], 0, SIZE[0] - 1)
            broad_margin = np.ceil(np.quantile(distance[coords[:, 1], coords[:, 0]], .95)) + 1
            corner_margin = np.ceil(distance[corner_coords[:, 1], corner_coords[:, 0]].max()) + 2
            radius = int(np.clip(max(broad_margin, corner_margin), 2, 12))
            masks[region] = morphology(natural[region], radius, "dilate")
            expansions[region] = radius
            coverage[region] = float(masks[region][coords[:, 1], coords[:, 0]].mean())
            critical_coverage[region] = float(masks[region][corner_coords[:, 1], corner_coords[:, 0]].mean())
            if critical_coverage[region] < 1:
                raise ValueError(f"Training corners/alar points exceed modest fixed mask expansion: {person}/{region}")
        # Large masks remain a disjoint partition of the median anatomical face.
        masks["upper_face"] = natural["upper_face"]
        masks["lower_face"] = natural["lower_face"]
        masks["full"] = np.ones(SIZE, dtype=bool)
        assert not (masks["upper_face"] & masks["lower_face"]).any()
        assert np.array_equal(masks["upper_face"] | masks["lower_face"], face)
        for region, mask in masks.items():
            if not mask.any() or (region != "full" and mask.all()):
                raise ValueError(f"Degenerate ROI: {person}/{region}")
        np.save(dest / "median_landmarks.npy", median)
        np.savez_compressed(dest / "masks.npz", **masks, anatomical_face=face)
        for region, mask in masks.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(dest / f"{region}_mask.png")
        provenance = dict(protocol=PROTOCOL, person=person, manifest=str(MANIFEST), manifest_sha256=manifest_hash,
                          train_only=True, samples=selected, sample_count=len(selected),
                          selection="Up to 10 evenly spaced distinct train clips/type; selected early/mid/late frame in a repeating cycle; image SHA deduplicated",
                          geometry="Coordinatewise median of 68-point 2DFAN landmarks in existing 224x224 RGB crop coordinates",
                          base_geometry=str(ROOT / "Analysis/smile_ranking_gradcam/build_region_mask_samples.py"),
                          base_geometry_sha256=sha(ROOT / "Analysis/smile_ranking_gradcam/build_region_mask_samples.py"),
                          semantics="True pixels retained; outside replaced by fixed VGG RGB mean; masks do not depend on input frame",
                          outside_rgb_mean=RGB_MEAN.tolist(), inference_output="float32 RGB; exact mean rather than PNG-rounded mean",
                          expansion_rule="Fine ROI fixed dilation = max(ceil(95th percentile train anatomical-point distance)+1, ceil(max train eye/mouth-corner or alar distance)+2), clipped to [2,12] pixels; distances measured outside median natural ROI",
                          expansion_pixels=expansions, sampled_anatomical_point_coverage=coverage,
                          sampled_corner_alar_coverage=critical_coverage,
                          fine_roi_overlap="Fixed fine masks may overlap after training-only spatial expansion; analyzed separately",
                          upper_lower_split_y=split_y, upper_lower="Disjoint complementary masks within median anatomical face; split at median nose-tip height",
                          mask_pixels={k:int(v.sum()) for k,v in masks.items()}, masks_sha256=sha(dest / "masks.npz"),
                          limitations=["Fixed masks can miss uncommon head/crop shifts; training-sample coverage is not held-out coverage",
                                       "Spatial ROI and face partition are operational definitions, not perfect anatomical segmentation"])
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        _draw_samples(person, selected, masks, dest)
        # One localized semantic check: visible pixels exact; hidden constant.
        test_image = np.asarray(Image.open(selected[0]["path"]).convert("RGB"))
        for mask in masks.values():
            test = mask_image(test_image, mask)
            assert np.array_equal(test[mask], test_image[mask].astype(np.float32))
            if (~mask).any():
                assert np.array_equal(test[~mask], np.broadcast_to(RGB_MEAN, test[~mask].shape))
        report.append(provenance)
        print(f"{person}: frozen masks ready; anatomical training coverage {coverage}", flush=True)
    summary = dict(protocol=PROTOCOL, manifest_sha256=manifest_hash, participants=[dict(person=p["person"],
                   sample_count=p["sample_count"], expansion_pixels=p["expansion_pixels"],
                   sampled_anatomical_point_coverage=p["sampled_anatomical_point_coverage"],
                   sampled_corner_alar_coverage=p["sampled_corner_alar_coverage"],
                   masks_sha256=p["masks_sha256"]) for p in report])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--persons", nargs="*")
    args = parser.parse_args()
    prepare(args.device, args.persons)
