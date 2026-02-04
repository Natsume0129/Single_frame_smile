#!/usr/bin/env python3
# extract_base_to_peak.py
#
# Input:  one feature .pt + tstart + tpeak
# Output: a .pt containing:
#   - b     : baseline vector [D] (mean of first k frames in segment)
#   - seg   : [L, D] raw segment features
#   - delta : [L, D] where delta(t)=f(t)-b
#   - m     : [L]    where m(t)=||delta(t)||
#   - v     : [L-1,D] where v(t)=f(t+1)-f(t)
#   - s     : [L-1]  where s(t)=||v(t)||
#
# Works for:
#   gap512 pt: {"names": [...], "gap512": Tensor[T,512]}
#   fc7   pt: {"names": [...], "feats" : Tensor[T,4096]}
'''
Docstring for Analysis.vggface_feature_analysis.extract_base_to_peak
python extract_base_to_peak.py 
  --pt "E:\Matsuda_data\vgg-face_analysis\classic_segments\extracted_features_rvm\true_1544-1551\vggface_conv5_3_rvm_gap512.pt"
  --tstart 70 --tpeak 147
  --out "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\true_1544-1551_rvm_base_to_peak.pt"
  --index-mode frame 
  --parse-mode first 
  --baseline-k 3

'''


import argparse
import re
from pathlib import Path
from typing import Optional, Dict

import torch


RE_FIRST_INT = re.compile(r"^(\d+)")
RE_LAST_INT = re.compile(r"(\d+)(?!.*\d)")


def infer_feat_key(obj: dict, feat_key: Optional[str]) -> str:
    if feat_key:
        if feat_key not in obj:
            raise KeyError(f"--feat-key '{feat_key}' not found. keys={list(obj.keys())}")
        return feat_key
    if "gap512" in obj:
        return "gap512"
    if "feats" in obj:
        return "feats"
    raise KeyError(f"Cannot infer feature key. keys={list(obj.keys())} (expected gap512 or feats)")


def parse_frame_index(name: str, parse_mode: str) -> Optional[int]:
    stem = Path(name).stem
    if parse_mode == "first":
        m = RE_FIRST_INT.match(stem)
        return int(m.group(1)) if m else None
    if parse_mode == "last":
        m = RE_LAST_INT.search(stem)
        return int(m.group(1)) if m else None
    raise ValueError("--parse-mode must be 'first' or 'last'")


def frame_to_row_map(names, index_mode: str, parse_mode: str) -> Dict[int, int]:
    if index_mode == "row":
        return {i: i for i in range(len(names))}
    m: Dict[int, int] = {}
    parsed = 0
    for i, n in enumerate(names):
        fi = parse_frame_index(n, parse_mode)
        if fi is None:
            continue
        parsed += 1
        # keep first occurrence if duplicates exist
        if fi not in m:
            m[fi] = i
    if parsed == 0 or len(m) == 0:
        raise RuntimeError("Could not parse frame indices from names. Use --index-mode row, or change --parse-mode.")
    return m


def select_segment(feats: torch.Tensor, names, tstart: int, tpeak: int, fmap: Dict[int, int], index_mode: str):
    if tpeak < tstart:
        raise ValueError(f"tpeak < tstart: {tstart}..{tpeak}")

    if index_mode == "row":
        if tstart < 0 or tpeak >= feats.shape[0]:
            raise IndexError(f"row index out of range: 0..{feats.shape[0]-1}, got {tstart}..{tpeak}")
        rows = list(range(tstart, tpeak + 1))
        frames_found = rows[:]
    else:
        rows = []
        frames_found = []
        for fr in range(tstart, tpeak + 1):
            if fr in fmap:
                rows.append(fmap[fr])
                frames_found.append(fr)
        if len(rows) < 2:
            raise RuntimeError(f"Segment has <2 frames after lookup. requested={tstart}..{tpeak}, found={len(rows)}")

    seg = feats[rows, :]
    name_list = [names[r] for r in rows]
    return seg, frames_found, name_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="input feature .pt (gap512 or fc7)")
    ap.add_argument("--tstart", type=int, required=True, help="start (frame number or row index)")
    ap.add_argument("--tpeak", type=int, required=True, help="peak  (frame number or row index)")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--baseline-k", type=int, default=3, help="baseline = mean of first k frames in segment")
    ap.add_argument("--feat-key", default=None, help="override feature key (gap512 or feats). usually omit")
    ap.add_argument("--index-mode", choices=["frame", "row"], default="frame",
                    help="frame: parse frame numbers from names; row: treat tstart/tpeak as row indices")
    ap.add_argument("--parse-mode", choices=["first", "last"], default="first",
                    help="when index-mode=frame: first for '15.png'/'15_1.png'; last for '..._15.png'")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0 (compute only; saved on cpu)")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    obj = torch.load(pt_path, map_location="cpu")

    if "names" not in obj:
        raise KeyError(f"{pt_path} has no 'names'. keys={list(obj.keys())}")
    names = obj["names"]

    feat_key = infer_feat_key(obj, args.feat_key)
    feats = obj[feat_key]
    if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
        raise RuntimeError(f"{feat_key} must be Tensor[T,D]. got type={type(feats)} shape={getattr(feats,'shape',None)}")

    T, D = feats.shape
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    feats = feats.to(device)

    fmap = frame_to_row_map(names, args.index_mode, args.parse_mode)

    seg, frames_found, names_found = select_segment(
        feats, names, args.tstart, args.tpeak, fmap, args.index_mode
    )

    L = seg.shape[0]
    k = max(1, min(args.baseline_k, L))
    b = seg[:k].mean(dim=0)
    delta = seg - b
    m = torch.linalg.norm(delta, dim=1)
    v = seg[1:] - seg[:-1]
    s = torch.linalg.norm(v, dim=1)

    missing = 0
    if args.index_mode == "frame":
        missing = (args.tpeak - args.tstart + 1) - L

    out_payload = {
        "meta": {
            "src_pt": str(pt_path),
            "feat_key": feat_key,
            "D": int(D),
            "T_total": int(T),
            "tstart": int(args.tstart),
            "tpeak": int(args.tpeak),
            "index_mode": args.index_mode,
            "parse_mode": args.parse_mode,
            "baseline_k": int(k),
            "len_found": int(L),
            "missing_frames": int(missing),
        },
        "frames_found": frames_found,
        "names_found": names_found,
        "b": b.detach().cpu(),
        "seg": seg.detach().cpu(),
        "delta": delta.detach().cpu(),
        "m": m.detach().cpu(),
        "v": v.detach().cpu(),
        "s": s.detach().cpu(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out_path)

    print("[OK] saved:", out_path)
    print("  feat_key:", feat_key, "| D:", D, "| segment_len:", L, "| missing_frames:", missing)


if __name__ == "__main__":
    main()
