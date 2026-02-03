#
'''
python check_pt_order.py --pt "E:\Matsuda_data\vgg-face_analysis\classic_segments\extracted_features_rvm\ambiguous 1443-1448\vggface_fc7_rvm.pt" --show 10 --check_step 1

'''


import re
import argparse
from pathlib import Path
import torch


def extract_last_int(stem: str):
    """
    尝试从文件名（不含扩展名）中提取“最后一个整数”
    例如: 20251029_15-44-15-51_0_6_176 -> 176
    """
    m = re.search(r"(\d+)(?!.*\d)", stem)  # 最后一个整数
    return int(m.group(1)) if m else None


def find_first_bad_pair(names):
    for i in range(len(names) - 1):
        if names[i] > names[i + 1]:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="path to .pt file (fc7.pt or gap512.pt etc.)")
    ap.add_argument("--show", type=int, default=10, help="how many sample names to print")
    ap.add_argument("--check_step", type=int, default=1, help="expected step between frame indices (optional)")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    obj = torch.load(pt_path, map_location="cpu")

    if "names" not in obj:
        raise KeyError(f"{pt_path} has no key 'names'. keys={list(obj.keys())}")

    names = obj["names"]
    if not isinstance(names, list) or not names:
        raise ValueError("names is empty or not a list.")

    print(f"[OK] loaded: {pt_path}")
    print(f"Total frames: {len(names)}")
    print("Sample names:")
    for s in names[:args.show]:
        print(" ", s)

    # 1) 字符串单调性检查
    bad = find_first_bad_pair(names)
    if bad is None:
        print("[PASS] names are non-decreasing (string order).")
    else:
        print("[FAIL] names are NOT sorted (string order).")
        print("  idx:", bad)
        print("  names[i]   =", names[bad])
        print("  names[i+1] =", names[bad + 1])
        return

    # 2) 从文件名提取“最后一个整数”做帧号检查
    indices = []
    missing = 0
    for n in names:
        stem = Path(n).stem
        idx = extract_last_int(stem)
        if idx is None:
            missing += 1
        indices.append(idx)

    if missing == len(names):
        print("[SKIP] cannot parse any integer index from filenames.")
        return
    if missing > 0:
        print(f"[WARN] {missing}/{len(names)} filenames had no parseable integer index (using None).")

    # 过滤掉 None
    idx_pairs = [(i, indices[i]) for i in range(len(indices)) if indices[i] is not None]
    idx_only = [x for _, x in idx_pairs]

    # 单调性
    bad2 = None
    for k in range(len(idx_only) - 1):
        if idx_only[k] > idx_only[k + 1]:
            bad2 = k
            break

    if bad2 is None:
        print("[PASS] parsed indices are non-decreasing.")
    else:
        i0, v0 = idx_pairs[bad2]
        i1, v1 = idx_pairs[bad2 + 1]
        print("[FAIL] parsed indices are NOT non-decreasing.")
        print(f"  at names[{i0}] -> {v0}")
        print(f"  next names[{i1}] -> {v1}")
        return

    # 步长合理性（可选）
    if args.check_step is not None:
        diffs = [idx_only[i + 1] - idx_only[i] for i in range(len(idx_only) - 1)]
        # 允许重复（diff=0）偶尔出现；重点找负数或特别大的跳变
        neg = sum(d < 0 for d in diffs)
        big = sum(d > max(5, args.check_step * 5) for d in diffs)
        print(f"Diff stats (min/median/max): {min(diffs)}/{sorted(diffs)[len(diffs)//2]}/{max(diffs)}")
        if neg == 0 and big == 0:
            print("[PASS] index diffs look reasonable.")
        else:
            print(f"[WARN] unusual diffs: neg={neg}, very_large={big}")
            print("  (This may be normal if frames are dropped or filenames are not strict frame indices.)")


if __name__ == "__main__":
    main()
