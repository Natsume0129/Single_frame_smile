"""Held-out analysis using existing CAM, anatomical masks and scoring definitions."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from pipeline import ROOT, LABELS, read_csv, write_csv, save_json, uniform_indices, frame_number
from analysis_contract import ANALYSIS_DIR, ANALYSIS_VERSION, paired_upper_lower, summarize_cam
from regional_effects import run_regional_effects


def analyze(cfg, person):
    import cv2
    import face_alignment
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from scipy.stats import spearmanr

    sys.path.insert(0, str(ROOT / "Analysis/smile_ranking_gradcam"))
    sys.path.insert(0, str(ROOT / "Analysis/feature_extractor"))
    sys.path.insert(0, cfg["smilecomp_dir"])
    import face_comp_torch as fc
    from feature_extractor_fc7 import VGGFaceFull, infer_fc8_out_from_state_dict
    from run_gradcam_experiment import build_preprocess, gradcam_for_pair, resize_cam, overlay_cam
    from roi_pattern_common import RoiCondition, build_keep_roi, score_feature_batch
    from build_region_mask_samples import build_natural_masks

    output = person / ANALYSIS_DIR
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly set device=cpu in a new configuration.")
    transform = build_preprocess(fc)
    sequence_rows = [r for r in read_csv(person / "sequences.csv") if r["split"] == "test"]
    frames = []
    for seq in sequence_rows:
        paths = sorted(Path(seq["sequence_dir"]).glob("*.png"), key=frame_number)
        for idx in uniform_indices(len(paths), cfg["analysis_frames_per_sequence"]):
            frames.append({"sequence_id": seq["sequence_id"], "input_label": seq["label"],
                           "frame_index": frame_number(paths[idx]), "path": str(paths[idx])})
    write_csv(output / "analysis_frames.csv", frames)
    # Balanced, shared train-only references: identical anchors across both models.
    # This is a two-class adaptation, not the historical true-anchor experiment.
    anchors = [person / "ranking" / label / "scale10" / f"level_{level:02d}.png"
               for label in LABELS for level in (0, 2, 4, 6, 9)]
    write_csv(output / "shared_anchors.csv",
              [{"path": str(p), "source": "training_sequence_scale"} for p in anchors])
    aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                          device=str(device), flip_input=False, compile=False)
    cache = {}
    candidates = [
        str(person / "ranking" / label / "images" / row["image_file"])
        for label in LABELS
        for row in read_csv(person / "ranking" / label / "ranking_items.csv")]
    for path in dict.fromkeys([f["path"] for f in frames] + [str(p) for p in anchors] + candidates):
        rgb = np.asarray(Image.open(path).convert("RGB").resize((224, 224)))
        predictions = aligner.get_landmarks_from_image(rgb)
        if predictions is None or len(predictions) != 1:
            save_json(output / "failure.json", {"path": path, "reason": "Expected exactly one landmark face"})
            raise RuntimeError(f"Landmark QC failed: {path}")
        points = np.asarray(predictions[0], dtype=np.float32)
        face, masks, _ = build_natural_masks(rgb.shape[:2], points)
        cache[path] = (rgb, points, face, masks)
    del aligner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    regional_profile, regional_metrics = run_regional_effects(
        cfg, person, frames, cache, fc, transform, device)
    rois = ("upper_face", "lower_face", "periocular", "cheeks", "mouth_perioral", "nose")
    conditions = [RoiCondition("all", "all", "unprocessed")] + [
        RoiCondition(f"{fill}_{roi}", roi, fill)
        for fill in ("blur", "local_mean") for roi in rois]
    scores, cam_rows, cam_similarity = [], [], {}
    batch_size = cfg["batch_size"]

    def tensors(paths, condition):
        images = []
        for path in paths:
            rgb, points, _, _ = cache[str(path)]
            processed, _ = build_keep_roi(rgb, points, condition)
            images.append(transform(Image.fromarray(processed)))
        return torch.stack(images).to(device)

    for label in LABELS:
        model = fc.face_comp_siamese_model_vgg16based_for_predict(
            str(person / "ranking" / label / "model/model.pth"), on_device=device).to(device).eval()
        for condition in conditions:
            with torch.inference_mode():
                anchor_features = model.extractor(tensors(anchors, condition))
                for start in range(0, len(frames), batch_size):
                    batch = frames[start:start + batch_size]
                    features = model.extractor(tensors([f["path"] for f in batch], condition))
                    result = score_feature_batch(model, features, anchor_features)
                    for i, frame in enumerate(batch):
                        scores.append({**frame, "model": label, "condition": condition.condition_id,
                                       "roi": condition.roi, "fill": condition.fill_method,
                                       "signed_response": float(result["signed_response"][i]),
                                       "score_0_9": float(result["score_0_9"][i])})
        # Fixed shared B for both models. No model-dependent anchor selection.
        # Mid-level reference matching input type; disagreement is retained.
        for frame in frames:
            b = person / "ranking" / frame["input_label"] / "scale10/level_04.png"
            a_tensor = transform(Image.fromarray(cache[frame["path"]][0])).unsqueeze(0).to(device)
            b_tensor = transform(Image.open(b).convert("RGB")).unsqueeze(0).to(device)
            logits, cams = gradcam_for_pair(model, a_tensor, b_tensor)
            combined = resize_cam(cams["a_class0"] + cams["a_class1"])
            total = float(combined.sum())
            rgb, _, face, masks = cache[frame["path"]]
            for roi in rois:
                mask = masks[roi]
                mass = float(combined[mask].sum()) / total if total > 0 else float("nan")
                cam_rows.append({**frame, "model": label, "roi": roi, "reference": str(b),
                                 "p_a_gt_b_unsymmetrized": float(torch.softmax(torch.tensor(logits), 0)[0]),
                                 "density": mass / float(mask.mean()) if total > 0 else float("nan"),
                                 "class0_empty": bool(cams["a_class0"].sum() == 0),
                                 "combined_empty": total == 0})
            token = f'{frame["sequence_id"]}_{frame["frame_index"]}'
            cam_dir = output / "cams" / label
            cam_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cam_dir / f"{token}.npz", **cams)
            display = combined / combined.max() if total > 0 else combined
            Image.fromarray(overlay_cam(rgb, display, .45)).save(cam_dir / f"{token}.png")
            cam_similarity.setdefault(token, {})[label] = combined.flatten()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output / "frame_roi_scores.csv", scores)
    write_csv(output / "frame_cam_regions.csv", cam_rows)
    df = pd.DataFrame(scores)
    base = df[df.condition == "all"].set_index(["model", "path"])
    roi_rows = []
    for (model, seq, roi, fill), group in df[df.condition != "all"].groupby(
            ["model", "sequence_id", "roi", "fill"], sort=False):
        reference = base.loc[[(model, p) for p in group.path], "signed_response"].to_numpy()
        response = group.signed_response.to_numpy()
        rho = float(spearmanr(reference, response).statistic) if (
            np.ptp(reference) > 0 and np.ptp(response) > 0) else float("nan")
        roi_rows.append({"model": model, "sequence_id": seq, "input_label": group.input_label.iloc[0],
                         "roi": roi, "fill": fill, "rank_preservation": rho,
                         "mean_signed_deviation": float(np.mean(response - reference)),
                         "mean_absolute_deviation": float(np.mean(np.abs(response - reference)))})
    write_csv(output / "sequence_roi_summary.csv", roi_rows)
    roi_df = pd.DataFrame(roi_rows)
    cam_df = pd.DataFrame(cam_rows)
    cam_seq = cam_df.groupby(["model", "input_label", "sequence_id", "roi"]).density.mean().reset_index()
    cam_seq.to_csv(output / "sequence_cam_summary.csv", index=False)
    roi_cells = roi_df.groupby(["model", "input_label", "roi", "fill"])[
        ["rank_preservation", "mean_signed_deviation", "mean_absolute_deviation"]].mean().reset_index()
    roi_cells.to_csv(output / "roi_cells.csv", index=False)
    upper_lower, upper_lower_cells, keep_metrics = paired_upper_lower(roi_rows)
    upper_lower.to_csv(output / "upper_lower_paired_sequences.csv", index=False)
    upper_lower_cells.to_csv(output / "upper_lower_cells.csv", index=False)
    similarities = []
    frame_lookup = {f'{f["sequence_id"]}_{f["frame_index"]}': f for f in frames}
    for token, maps in cam_similarity.items():
        a, b = maps["polite"], maps["bitter"]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        similarities.append({**frame_lookup[token], "frame": token,
                             "cosine": float(a @ b / denominator) if denominator else float("nan")})
    write_csv(output / "cam_model_similarity.csv", similarities)
    cam_sequence_similarity, cam_region_differences, cam_metrics = summarize_cam(similarities, cam_rows)
    cam_sequence_similarity.to_csv(output / "cam_sequence_similarity.csv", index=False)
    cam_region_differences.to_csv(output / "cam_region_model_differences.csv", index=False)
    sensitivity = []
    for (model, seq, roi), group in roi_df.groupby(["model", "sequence_id", "roi"]):
        lookup = group.set_index("fill")
        sensitivity.append({"model": model, "sequence_id": seq, "roi": roi,
                            "blur_rho": float(lookup.loc["blur", "rank_preservation"]),
                            "local_mean_rho": float(lookup.loc["local_mean", "rank_preservation"]),
                            "same_deviation_sign": bool(np.sign(lookup.loc["blur", "mean_signed_deviation"]) ==
                                                        np.sign(lookup.loc["local_mean", "mean_signed_deviation"]))})
    write_csv(output / "fill_sensitivity.csv", sensitivity)
    # Historical geometry uses frozen, L2-normalized fc7, NOT the trained comparison heads.
    state = torch.load(cfg["fc7_weights"], map_location="cpu", weights_only=True)
    feature_model = VGGFaceFull(infer_fc8_out_from_state_dict(state))
    feature_model.load_state_dict(state)
    del state
    feature_model.to(device).eval()
    direction_rows, directions, trajectories = [], {}, []
    for seq in sequence_rows:
        paths = sorted(Path(seq["sequence_dir"]).glob("*.png"), key=frame_number)
        features = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start+batch_size]]).to(device)
                features.append(torch.nn.functional.normalize(feature_model.forward_fc7(images), dim=1).cpu().numpy())
        f = np.concatenate(features)
        delta = f - f[:5].mean(axis=0)
        # Objective maximum-displacement reference; not a manually identified smile peak.
        peak = int(np.linalg.norm(delta, axis=1).argmax())
        vector = delta[peak]
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            directions[seq["sequence_id"]] = (seq["label"], vector / norm)
        direction_rows.append({"sequence_id": seq["sequence_id"], "input_label": seq["label"],
                               "max_displacement": norm, "reference_frame": frame_number(paths[peak]),
                               "reference_definition": "max_fc7_displacement_not_manual_peak"})
        for idx in uniform_indices(len(paths), 20):
            projection = float(delta[idx] @ vector / norm) if norm > 1e-8 else float("nan")
            residual = float(np.linalg.norm(delta[idx] - projection * vector / norm)) if norm > 1e-8 else float("nan")
            trajectories.append({"sequence_id": seq["sequence_id"], "input_label": seq["label"],
                                 "frame_index": frame_number(paths[idx]), "progress": idx / (len(paths)-1),
                                 "magnitude": float(np.linalg.norm(delta[idx])),
                                 "projection": projection, "off_axis": residual})
    write_csv(output / "direction_sequences.csv", direction_rows)
    write_csv(output / "direction_trajectories.csv", trajectories)
    direction_pairs = [{"sequence_a": a, "sequence_b": b,
                        "relation": "within" if x[0] == y[0] else "between",
                        "class_pair": "-".join(sorted((x[0], y[0]))), "cosine": float(x[1] @ y[1])}
                       for (a, x), (b, y) in itertools.combinations(directions.items(), 2)]
    if not direction_pairs:
        raise ValueError("Insufficient nonzero directions.")
    write_csv(output / "direction_pairs.csv", direction_pairs)
    fig, grid = plt.subplots(2, 2, figsize=(14, 10))
    axes = grid.ravel()
    direction_df = pd.DataFrame(direction_pairs)
    direction_df.groupby("class_pair").cosine.mean().plot.bar(ax=axes[0])
    axes[0].set_title("1. Feature-space direction")
    axes[0].set_ylabel("Mean direction cosine")
    regional_profile.pivot(index="roi", columns="model", values="share_percent").plot.bar(ax=axes[1])
    axes[1].set_title("2. Relative absolute change after masked retraining")
    axes[1].set_ylabel("Share (%) — not causal importance")
    for fill in ("blur", "local_mean"):
        values = roi_cells[roi_cells.fill == fill].groupby("roi").rank_preservation.mean().reindex(rois)
        axes[2].plot(range(len(rois)), values, "o-", label=fill)
    axes[2].set_xticks(range(len(rois)), rois, rotation=45, ha="right")
    axes[2].set_title("3. Keep-ROI ranking preservation")
    axes[2].set_ylabel("Mean within-sequence Spearman rho")
    axes[2].legend()
    cam_sequence_similarity.groupby("input_label").mean_cosine.mean().plot.bar(ax=axes[3])
    axes[3].set_title("4. Same-pair CAM similarity between models")
    axes[3].set_ylabel("Mean cosine (sequence aggregated)")
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=160)
    plt.close(fig)
    within = direction_df[direction_df.relation == "within"].groupby("class_pair").cosine.mean().mean()
    between = direction_df[direction_df.relation == "between"].cosine.mean()
    metrics = {
        "person": person.name, "test_sequence_count": len(sequence_rows),
        "analysis_version": ANALYSIS_VERSION,
        "within_minus_between_direction_cosine": float(within - between),
        **regional_metrics, **keep_metrics, **cam_metrics,
        "inference": "descriptive; no frame-level significance tests",
        "geometry": "frozen fc7, baseline=first5, reference=max displacement",
        "roi_anchors": "shared 5 polite + 5 bitter train-only scale images",
        "cam_reference": "fixed input-type level04 shared between models",
    }
    save_json(output / "summary.json", metrics)
    report = [
        f"# {person.name}：Polite / Bitter 分析", "",
        "分析仅使用预先划分的 test 序列；模型选择使用 val，anchors 仅来自 train。",
        "每个模型的人工 pair holdout 指标见下表。结果为描述性复现，不自动宣布旧结论成立。", "",
        "| 模型 | Holdout 正确率 | Pair 数 |", "|---|---:|---:|"]
    for label in LABELS:
        m = json.loads((person / "ranking" / label / "model/metrics.json").read_text())
        report.append(f'| {label} | {m["holdout_accuracy"]:.3f} | {m["num_holdout_pairs"]} |')
    report += ["", "## 1. 特征空间方向", "",
               f"类内−类间方向余弦：{metrics['within_minus_between_direction_cosine']:.4f}。正值表示类内方向更一致。",
               "这是 polite/bitter 的方向检验；旧 true/polite/ambiguous 结果不能直接当作它已成立的证据。",
               "", "## 2. 区域相对评分变化分布", "",
               f"Polite 脸颊−其余最大占比(pp)：{regional_metrics['polite_cheeks_minus_next_largest_share']}。",
               f"Polite−Bitter 不均衡度：{regional_metrics['polite_minus_bitter_share_imbalance']}。",
               "前者为正对应脸颊占比最高，后者为正对应 bitter 更均衡；零总变化时占比不可定义，不记成均衡。",
               "四细区域使用面积匹配遮罩、分别重训并重建 native scale。按序列汇总，再对两类输入等权。",
               "数值是整套系统的相对绝对评分变化，不是模型直接权重或器官因果贡献。",
               "", "## 3. 下半脸与上半脸的排序保持", "",
               f"Blur 配对差：{keep_metrics['lower_minus_upper_rank_preservation']:.4f}；"
               f"local-mean 配对差：{keep_metrics['local_mean_lower_minus_upper_rank_preservation']:.4f}。",
               "差值在同一序列内配对，正值对应下半脸更能保持排序；两种填充均报告。",
               "", "## 4. Grad-CAM 整体相似与局部差异", "",
               f"序列汇总、输入类别等权的模型间 CAM 余弦：{cam_metrics['cam_model_cosine']:.4f}。",
               "局部区域差异见 cam_region_model_differences.csv；空 CAM 不当作零相似或完全相似。",
               "", "![Summary](summary.png)", "", "```json", json.dumps(metrics, ensure_ascii=False, indent=2), "```", "",
               "解释边界：",
               "- 方向分析使用冻结 fc7；比较头重训不会改变这个空间。最大位移帧不等于人工峰值，起始 5 帧也不保证中性。",
               "- ROI 是区域细节保留实验，不是单块区域必要性或因果贡献；区域外仍有低频信息。",
               "- ROI 使用两类训练 anchors 各 5 张，共同送给两个模型；这是旧 true-anchor 协议的两类适配。",
               "- Grad-CAM 使用输入类别的固定 level04 B，不筛选 A>B；不同模型收到完全相同 A/B，保留不认同关系和空 CAM。",
               "- 单元汇总先按序列；方向 pair 不是独立统计样本。跨人报告每人等权，不能用帧数充当样本量。",
               "- 查看 fill_sensitivity.csv，不能只选择支持预期的填充结果。",
               "- 人工 tie 与 discard 不用于二分类训练，其数量记录在 annotation_audit.json。",
               "", "完整数值：roi_cells.csv、upper_lower_paired_sequences.csv、sequence_cam_summary.csv、",
               "cam_region_model_differences.csv、direction_pairs.csv、direction_trajectories.csv。",
               "第2项详细结果：[区域评分变化](../../regional_systems/relative_effect_profiles.csv)。"]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Analysis complete: {output / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    analyze(json.loads(args.config.read_text(encoding="utf-8-sig")), args.person_root)
