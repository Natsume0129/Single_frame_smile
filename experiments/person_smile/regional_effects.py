"""Conclusion 2: area-matched remove-ROI systems, retrained with rebuilt scales.

These are relative absolute score changes, not fixed-model causal importance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline import ROOT, LABELS, digest, read_csv, run, save_json, write_csv

FINE_ROIS = ("periocular", "mouth_perioral", "nose", "cheeks")


def summarize_effects(frame_effects):
    """Frames -> sequences -> equal-weight input classes -> four-region shares."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(frame_effects)
    seq = df.groupby(["model", "input_label", "sequence_id", "roi"], as_index=False).agg(
        mean_absolute_change=("absolute_change", "mean"),
        mean_signed_change=("signed_change", "mean"),
    )
    cells = seq.groupby(["model", "input_label", "roi"], as_index=False).mean(numeric_only=True)
    profile = cells.groupby(["model", "roi"], as_index=False).mean(numeric_only=True)
    profile["share_percent"] = profile.groupby("model").mean_absolute_change.transform(
        lambda values: values / values.sum() * 100 if values.sum() > 1e-12 else values * np.nan)
    metrics = {}
    for model in LABELS:
        shares = profile[profile.model == model].set_index("roi").share_percent.reindex(FINE_ROIS)
        valid = bool(shares.notna().all())
        metrics[f"{model}_cheeks_share_percent"] = float(shares["cheeks"]) if valid else None
        metrics[f"{model}_cheeks_minus_next_largest_share"] = (
            float(shares["cheeks"] - shares.drop("cheeks").max()) if valid else None)
        metrics[f"{model}_share_tv_from_uniform"] = (
            float(np.abs(shares - 25).sum() / 200) if valid else None)
    p, b = (metrics[f"{m}_share_tv_from_uniform"] for m in LABELS)
    metrics["polite_minus_bitter_share_imbalance"] = p - b if p is not None and b is not None else None
    return seq, profile, metrics


def run_regional_effects(cfg, person, frames, cache, fc, transform, device):
    import numpy as np
    import torch
    from PIL import Image
    from region_ablation_common import MaskCondition, apply_condition
    from roi_pattern_common import score_feature_batch

    root = person / "regional_systems"
    # Freeze mask size from training candidates only, never from evaluation data.
    fractions = []
    for label in LABELS:
        data = person / "ranking" / label
        for item in read_csv(data / "scale_candidates.csv"):
            _, _, face, masks = cache[str(data / "images" / item["image_file"])]
            fractions.extend(float(masks[r].sum() / face.sum()) for r in FINE_ROIS)
    area_fraction = float(np.median(fractions))
    save_json(root / "protocol.json", {
        "version": "four_conclusions_v2",
        "area_fraction": area_fraction, "area_source": "pooled train candidates only",
        "regions": FINE_ROIS,
        "control": "existing unmasked model trained with identical split, seed and hyperparameters",
        "effect": "control native-scale score minus masked/retrained/rebuilt-native-scale score",
        "interpretation": "whole-system response, not causal anatomical importance",
    })

    def score_system(model, anchor_paths, condition=None):
        with torch.inference_mode():
            anchor_tensors = [transform(Image.open(p).convert("RGB")) for p in anchor_paths]
            anchors = model.extractor(torch.stack(anchor_tensors).to(device))
            results = {}
            for start in range(0, len(frames), cfg["batch_size"]):
                batch = frames[start:start + cfg["batch_size"]]
                images = []
                for row in batch:
                    rgb, points, _, _ = cache[row["path"]]
                    if condition is not None:
                        rgb, _ = apply_condition(rgb, points, condition, area_fraction)
                    images.append(transform(Image.fromarray(rgb)))
                features = model.extractor(torch.stack(images).to(device))
                scores = score_feature_batch(model, features, anchors)["score_0_9"]
                results.update({r["path"]: float(v) for r, v in zip(batch, scores)})
            return results

    effects = []
    for label in LABELS:
        source = person / "ranking" / label
        baseline = fc.face_comp_siamese_model_vgg16based_for_predict(
            str(source / "model/model.pth"), on_device=device).to(device).eval()
        control = score_system(baseline, [source / "scale10" / f"level_{n:02d}.png" for n in range(10)])
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for roi in FINE_ROIS:
            condition = MaskCondition(f"area_matched_{roi}", "area_matched", roi)
            data = root / label / roi
            complete = data / "complete.json"
            fingerprint = {
                "version": "four_conclusions_v2", "config": cfg,
                "area_fraction": area_fraction, "roi": roi,
                "initial_weights": digest(cfg["initial_weights"]),
                "split_hashes": {s: digest(source / f"{s}.dat") for s in ("train", "val", "test", "all")},
                "candidate_hashes": {r["image_file"]: digest(source / "images" / r["image_file"])
                                     for r in read_csv(source / "ranking_items.csv")},
            }
            if complete.exists() and json.loads(complete.read_text(encoding="utf-8")) != fingerprint:
                raise ValueError(f"Regional training inputs changed: {data}; use a new experiment root.")
            if not complete.exists():
                (data / "images").mkdir(parents=True, exist_ok=True)
                import shutil
                for name in ("train.dat", "val.dat", "test.dat", "all.dat", "scale_candidates.csv"):
                    shutil.copy2(source / name, data / name)
                qc = []
                for item in read_csv(source / "ranking_items.csv"):
                    rgb, points, _, _ = cache[str(source / "images" / item["image_file"])]
                    masked, info = apply_condition(rgb, points, condition, area_fraction)
                    Image.fromarray(masked).save(data / "images" / item["image_file"])
                    qc.append({"image_file": item["image_file"], "mask_pixels": info["mask_pixels"],
                               "face_pixels": info["face_pixels"], "split": item["split"]})
                write_csv(data / "mask_qc.csv", qc)
                run([sys.executable, ROOT / "annotation/train_smile_rank_with_holdout.py",
                     "--smilecomp_dir", cfg["smilecomp_dir"], "--weights", cfg["initial_weights"],
                     "--dataset_file", data / "all.dat", "--dataset_dir", data / "images",
                     "--train_file", data / "train.dat", "--val_file", data / "val.dat",
                     "--holdout_file", data / "test.dat", "--use_existing_splits",
                     "--dst_weights", data / "model/model.pth", "--dst_history", data / "model/history.png",
                     "--dst_metrics", data / "model/metrics.json", "--epochs", cfg["epochs"],
                     "--batch_size", cfg["batch_size"], "--learning_rate", cfg["learning_rate"],
                     "--seed", cfg["seed"]], data / "train.log")
                run([sys.executable, ROOT / "annotation/build_true_smile_scale_from_model.py",
                     "--smilecomp_dir", cfg["smilecomp_dir"], "--ranking_items", data / "scale_candidates.csv",
                     "--images_dir", data / "images", "--weights", data / "model/model.pth",
                     "--manual_pairs", data / "train.dat", "--output_dir", data / "scale10"],
                    data / "scale.log")
                save_json(complete, fingerprint)
            model = fc.face_comp_siamese_model_vgg16based_for_predict(
                str(data / "model/model.pth"), on_device=device).to(device).eval()
            masked_scores = score_system(
                model, [data / "scale10" / f"level_{n:02d}.png" for n in range(10)], condition)
            for row in frames:
                change = control[row["path"]] - masked_scores[row["path"]]
                effects.append({**row, "model": label, "roi": roi,
                                "control_score": control[row["path"]],
                                "masked_system_score": masked_scores[row["path"]],
                                "signed_change": change, "absolute_change": abs(change)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(root / "frame_effects.csv", effects)
    seq, profile, metrics = summarize_effects(effects)
    seq.to_csv(root / "sequence_effects.csv", index=False)
    profile.to_csv(root / "relative_effect_profiles.csv", index=False)
    save_json(root / "summary.json", metrics)
    return profile, metrics
