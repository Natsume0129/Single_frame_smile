"""Shared output contract for the four requested research questions."""
ANALYSIS_VERSION = "four_conclusions_v2"
ANALYSIS_DIR = f"analysis/{ANALYSIS_VERSION}"

SUMMARY_COLUMNS = {
    "within_minus_between_direction_cosine": "1 类内−类间方向余弦",
    "polite_cheeks_minus_next_largest_share": "2 Polite 脸颊−其他最大占比(pp)",
    "polite_minus_bitter_share_imbalance": "2 Polite−Bitter 不均衡度",
    "lower_minus_upper_rank_preservation": "3 下半脸−上半脸(blur)",
    "local_mean_lower_minus_upper_rank_preservation": "3 下半脸−上半脸(mean)",
    "cam_model_cosine": "4 同一A/B的CAM余弦",
}


def paired_upper_lower(roi_rows):
    """Pair upper/lower within each sequence, then balance input classes/models."""
    import pandas as pd
    df = pd.DataFrame(roi_rows)
    pairs = df[df.roi.isin(["upper_face", "lower_face"])].pivot(
        index=["model", "input_label", "sequence_id", "fill"],
        columns="roi", values="rank_preservation").reset_index()
    pairs["lower_minus_upper"] = pairs["lower_face"] - pairs["upper_face"]
    cells = pairs.groupby(["model", "input_label", "fill"], as_index=False).agg(
        mean_lower_minus_upper=("lower_minus_upper", "mean"),
        valid_sequences=("lower_minus_upper", "count"),
    )
    metrics = {
        ("lower_minus_upper_rank_preservation" if fill == "blur"
         else "local_mean_lower_minus_upper_rank_preservation"):
        float(cells[cells.fill == fill].mean_lower_minus_upper.mean())
        for fill in ("blur", "local_mean")
    }
    return pairs, cells, metrics


def summarize_cam(similarities, region_rows):
    import pandas as pd
    frame = pd.DataFrame(similarities)
    seq = frame.groupby(["input_label", "sequence_id"], as_index=False).agg(
        mean_cosine=("cosine", "mean"), valid_frames=("cosine", "count"),
        sampled_frames=("cosine", "size"),
    )
    # A sequence with empty CAMs is missing evidence, not perfect similarity.
    cosine = float(seq.groupby("input_label").mean_cosine.mean().mean())
    regions = pd.DataFrame(region_rows).groupby(
        ["model", "input_label", "sequence_id", "roi"], as_index=False).density.mean()
    delta = regions.pivot(index=["input_label", "sequence_id", "roi"],
                          columns="model", values="density").reset_index()
    delta["polite_minus_bitter_density"] = delta["polite"] - delta["bitter"]
    return seq, delta, {"cam_model_cosine": cosine}
