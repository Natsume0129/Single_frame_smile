from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_EXTRACTOR_DIR = SCRIPT_DIR.parent / "feature_extractor"
if str(FEATURE_EXTRACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTOR_DIR))

from feature_extractor_fc7 import (  # type: ignore
    VGGFaceFull,
    build_preprocess,
    infer_fc8_out_from_state_dict,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class PairImageDataset(Dataset):
    def __init__(self, entries: list[dict], transform):
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        p = Path(entry["image_path"])
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, idx


def load_manifest(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "date",
            "segment_folder",
            "segment_start",
            "segment_end",
            "start_image",
            "end_image",
            "start_ts",
            "end_ts",
            "folder_path",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Manifest missing fields: {sorted(missing)}")

        for row in reader:
            folder = Path(row["folder_path"])
            start_img = row["start_image"]
            end_img = row["end_image"]
            start_path = folder / start_img
            end_path = folder / end_img

            if not folder.is_dir():
                raise RuntimeError(f"Folder does not exist: {folder}")
            if start_path.suffix.lower() not in IMAGE_EXTS or end_path.suffix.lower() not in IMAGE_EXTS:
                raise RuntimeError(f"Unsupported image extension in folder: {folder}")
            if not start_path.exists():
                raise RuntimeError(f"Start image not found: {start_path}")
            if not end_path.exists():
                raise RuntimeError(f"End image not found: {end_path}")

            row["segment_start"] = int(row["segment_start"])
            row["segment_end"] = int(row["segment_end"])
            row["start_ts"] = int(row["start_ts"])
            row["end_ts"] = int(row["end_ts"])
            row["start_path"] = str(start_path)
            row["end_path"] = str(end_path)
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows found in manifest: {manifest_path}")
    return rows


def extract_fc7_for_entries(
    entries: list[dict],
    model: VGGFaceFull,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    transform = build_preprocess()
    ds = PairImageDataset(entries, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feat_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for imgs, idxs in dl:
            imgs = imgs.to(device, non_blocking=True)
            feats = model.forward_fc7(imgs)
            feats = F.normalize(feats, dim=1)
            feat_chunks.append(feats.cpu())
            index_chunks.append(idxs)

    all_feats = torch.cat(feat_chunks, dim=0)
    all_idxs = torch.cat(index_chunks, dim=0)
    order = torch.argsort(all_idxs)
    return all_feats[order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fc7 features for key-frame pairs and compute diff vector (vp-vs)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\key_frames_manifest.csv"),
        help="CSV manifest built from key_frames pairs.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"E:\Single_frame_smile\data\models\vggface.pth"),
        help="VGGFace full weights containing conv+fc6+fc7+fc8.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\feature_vectors"),
        help="Output directory.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print("Device:", device)

    rows = load_manifest(args.manifest)
    print("Manifest rows:", len(rows))

    sd = torch.load(args.weights, map_location="cpu")
    fc8_out = infer_fc8_out_from_state_dict(sd)
    model = VGGFaceFull(fc8_out=fc8_out)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    entries: list[dict] = []
    for i, row in enumerate(rows):
        entries.append(
            {
                "pair_index": i,
                "role": "vs",
                "image_path": row["start_path"],
            }
        )
        entries.append(
            {
                "pair_index": i,
                "role": "vp",
                "image_path": row["end_path"],
            }
        )

    all_feats = extract_fc7_for_entries(
        entries=entries,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if all_feats.shape[0] != len(rows) * 2:
        raise RuntimeError("Unexpected number of extracted features.")

    vs = all_feats[0::2]
    vp = all_feats[1::2]
    diff = vp - vs

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.out_dir / "fc7_pair_diff.pt"
    out_csv = args.out_dir / "fc7_pair_diff_manifest.csv"
    out_meta = args.out_dir / "fc7_pair_diff_meta.json"

    payload = {
        "date": [row["date"] for row in rows],
        "segment_folder": [row["segment_folder"] for row in rows],
        "segment_start": [row["segment_start"] for row in rows],
        "segment_end": [row["segment_end"] for row in rows],
        "start_image": [row["start_image"] for row in rows],
        "end_image": [row["end_image"] for row in rows],
        "start_ts": [row["start_ts"] for row in rows],
        "end_ts": [row["end_ts"] for row in rows],
        "vs": vs,
        "vp": vp,
        "diff": diff,
    }
    torch.save(payload, out_pt)

    csv_fields = [
        "pair_index",
        "date",
        "segment_folder",
        "segment_start",
        "segment_end",
        "start_image",
        "end_image",
        "start_ts",
        "end_ts",
        "pair_span",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {
                    "pair_index": i,
                    "date": row["date"],
                    "segment_folder": row["segment_folder"],
                    "segment_start": row["segment_start"],
                    "segment_end": row["segment_end"],
                    "start_image": row["start_image"],
                    "end_image": row["end_image"],
                    "start_ts": row["start_ts"],
                    "end_ts": row["end_ts"],
                    "pair_span": int(row["end_ts"]) - int(row["start_ts"]),
                }
            )

    meta = {
        "manifest": str(args.manifest),
        "weights": str(args.weights),
        "num_pairs": len(rows),
        "feature_dim": int(diff.shape[1]),
        "output_pt": str(out_pt),
        "output_csv": str(out_csv),
        "normalize": "L2 on fc7 before diff",
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("pairs:", len(rows))
    print("vs shape:", tuple(vs.shape))
    print("vp shape:", tuple(vp.shape))
    print("diff shape:", tuple(diff.shape))
    print("Saved:", out_pt)
    print("Saved:", out_csv)
    print("Saved:", out_meta)


if __name__ == "__main__":
    main()
