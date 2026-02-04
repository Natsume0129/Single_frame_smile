#!/usr/bin/env python3
# plot_base_to_peak_outputs.py
#
# Input: multiple output .pt files produced by extract_base_to_peak.py
# Output:
#   1) m(t) curves overlay   -> m_curves.png
#   2) s(t) curves overlay   -> s_curves.png
#   3) endpoint magnitude bar-> endpoint_mag.png + endpoint_mag.csv
#
# Usage (PowerShell example):
#   python plot_base_to_peak_outputs.py `
#     --pts "E:\out\polite.pt" "E:\out\wry.pt" "E:\out\true.pt" `
#     --labels polite wry true `
#     --out-dir "E:\out\plots" `
#     --x frames
#
# Notes:
# - Works for both gap512 (D=512) and fc7 (D=4096), because it reads m/s/delta from the output pt.
# - --x frames uses frames_found on x-axis; --x index uses 0..L-1.
'''
Docstring for Analysis.vggface_feature_analysis.plot_base_to_peak_outputs
python plot_base_to_peak_outputs.py 
  --pts "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\ambiguous_2318-2324_rvm_base_to_peak.pt" "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\polite_0912-0921_rvm_base_to_peak.pt" "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\true_1544-1551_rvm_base_to_peak.pt"
  --labels polite wry true 
  --out-dir "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak"
  --x frames

'''

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt


def load_one(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu")

    # required keys
    for k in ["m", "s", "delta", "frames_found", "meta"]:
        if k not in obj:
            raise KeyError(f"{pt_path} missing key '{k}'. keys={list(obj.keys())}")

    m = obj["m"].detach().cpu().float()           # [L]
    s = obj["s"].detach().cpu().float()           # [L-1]
    delta = obj["delta"].detach().cpu().float()   # [L,D]
    frames_found = obj["frames_found"]            # list[int]
    meta = obj["meta"]

    if m.ndim != 1 or s.ndim != 1 or delta.ndim != 2:
        raise ValueError(f"{pt_path} has unexpected shapes: m={tuple(m.shape)}, s={tuple(s.shape)}, delta={tuple(delta.shape)}")

    # endpoint magnitude = ||delta[-1]||
    endpoint_mag = torch.linalg.norm(delta[-1]).item()

    return {
        "pt": str(pt_path),
        "m": m,
        "s": s,
        "frames": frames_found,
        "endpoint_mag": endpoint_mag,
        "meta": meta,
    }


def make_x(mode: str, frames: List[int], length: int) -> List[int]:
    if mode == "index":
        return list(range(length))
    # mode == "frames"
    if frames and len(frames) == length:
        return frames
    # fallback: still plot by index if frames list missing/mismatched
    return list(range(length))


def save_endpoint_csv(rows: List[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", nargs="+", required=True, help="output .pt files from extract_base_to_peak.py")
    ap.add_argument("--labels", nargs="+", default=None, help="labels for legend; same length as --pts. If omitted, use filename stem.")
    ap.add_argument("--out-dir", required=True, help="directory to save plots")
    ap.add_argument("--x", choices=["frames", "index"], default="frames", help="x-axis: frames_found or 0..L-1")
    ap.add_argument("--title-prefix", default="", help="optional prefix for plot titles")
    args = ap.parse_args()

    pt_paths = [Path(p) for p in args.pts]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.labels is None:
        labels = [p.stem for p in pt_paths]
    else:
        if len(args.labels) != len(pt_paths):
            raise ValueError(f"--labels length must match --pts. got labels={len(args.labels)} pts={len(pt_paths)}")
        labels = args.labels

    data = [load_one(p) for p in pt_paths]

    # --------
    # 1) m(t) curves
    # --------
    plt.figure()
    for d, lab in zip(data, labels):
        x = make_x(args.x, d["frames"], len(d["m"]))
        plt.plot(x, d["m"].numpy(), label=lab)
    plt.xlabel("frame" if args.x == "frames" else "index")
    plt.ylabel("m(t) = ||f(t) - baseline||")
    title = "m(t) curves"
    if args.title_prefix:
        title = f"{args.title_prefix} | {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_m = out_dir / "m_curves.png"
    plt.savefig(out_m, dpi=200)
    plt.close()

    # --------
    # 2) s(t) curves
    # --------
    plt.figure()
    for d, lab in zip(data, labels):
        # s length is L-1
        if args.x == "frames" and d["frames"] and len(d["frames"]) >= 2:
            x = d["frames"][1:]  # align with v(t)=f(t)-f(t-1)
            if len(x) != len(d["s"]):
                x = list(range(len(d["s"])))
        else:
            x = list(range(len(d["s"])))
        plt.plot(x, d["s"].numpy(), label=lab)
    plt.xlabel("frame" if args.x == "frames" else "index")
    plt.ylabel("s(t) = ||f(t+1) - f(t)||")
    title = "s(t) curves"
    if args.title_prefix:
        title = f"{args.title_prefix} | {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_s = out_dir / "s_curves.png"
    plt.savefig(out_s, dpi=200)
    plt.close()

    # --------
    # 3) endpoint magnitude bar
    # --------
    mags = [d["endpoint_mag"] for d in data]
    plt.figure()
    plt.bar(list(range(len(mags))), mags, tick_label=labels)
    plt.xlabel("segment")
    plt.ylabel("endpoint_mag = ||delta[-1]||")
    title = "Endpoint magnitude"
    if args.title_prefix:
        title = f"{args.title_prefix} | {title}"
    plt.title(title)
    plt.tight_layout()
    out_bar = out_dir / "endpoint_mag.png"
    plt.savefig(out_bar, dpi=200)
    plt.close()

    # also save a small CSV
    rows = []
    for lab, d in zip(labels, data):
        meta = d["meta"]
        rows.append({
            "label": lab,
            "pt": d["pt"],
            "feat_key": meta.get("feat_key", ""),
            "D": meta.get("D", ""),
            "tstart": meta.get("tstart", ""),
            "tpeak": meta.get("tpeak", ""),
            "len_found": meta.get("len_found", ""),
            "missing_frames": meta.get("missing_frames", ""),
            "endpoint_mag": d["endpoint_mag"],
        })
    out_csv = out_dir / "endpoint_mag.csv"
    save_endpoint_csv(rows, out_csv)

    print("[OK] saved:")
    print(" ", out_m)
    print(" ", out_s)
    print(" ", out_bar)
    print(" ", out_csv)


if __name__ == "__main__":
    main()
