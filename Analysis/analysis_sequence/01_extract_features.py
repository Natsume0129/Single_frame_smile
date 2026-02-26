from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
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

from common.base import PipelineConfig, SequenceTaskBase


class SequenceImageDataset(Dataset):
    def __init__(self, frame_paths: list[Path], transform):
        self.frame_paths = frame_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int):
        p = self.frame_paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


class FeatureExtractionTask(SequenceTaskBase):
    def run(self) -> None:
        device = torch.device(
            self.cfg.device if (self.cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        state_dict = torch.load(self.cfg.weights, map_location="cpu")
        model = VGGFaceFull(fc8_out=infer_fc8_out_from_state_dict(state_dict))
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        transform = build_preprocess()

        for seq in self.discover_sequences():
            frame_paths = self.list_sorted_frames(seq.sequence_path)
            ds = SequenceImageDataset(frame_paths=frame_paths, transform=transform)
            dl = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            feat_chunks: list[torch.Tensor] = []
            idx_chunks: list[torch.Tensor] = []
            with torch.no_grad():
                for imgs, idxs in dl:
                    imgs = imgs.to(device, non_blocking=True)
                    feats = model.forward_fc7(imgs)
                    feats = torch.nn.functional.normalize(feats, dim=1)
                    feat_chunks.append(feats.cpu())
                    idx_chunks.append(idxs)

            feats = torch.cat(feat_chunks, dim=0)
            order = torch.argsort(torch.cat(idx_chunks, dim=0))
            feats = feats[order].numpy().astype(np.float32)

            out_dir = self.metrics_seq_dir("sequence_features", seq)
            self.save_npy(out_dir / "sequence_features.npy", feats)
            with (out_dir / "frame_names.json").open("w", encoding="utf-8") as f:
                json.dump([p.name for p in frame_paths], f, ensure_ascii=False, indent=2)

            print(f"[STEP1] {seq.class_name}/{seq.sequence_id}: shape={tuple(feats.shape)}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 1: extract VGG-Face fc7 features per sequence.")
    args = parser.parse_args()
    task = FeatureExtractionTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

