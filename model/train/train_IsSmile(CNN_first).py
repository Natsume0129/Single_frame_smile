# train_IsSmile_optimized.py
# Windows-safe, RTX 5090-friendly PyTorch training script (Scheme A: WeightedRandomSampler + online aug)
# - Fixes Windows DataLoader spawn issue (main guard)
# - Default num_workers=0 (safe on Windows); you can raise it after it's stable
# - Adds AMP (mixed precision) for speed on modern GPUs
# - Uses dat paths relative to ROOT_DIR (your layout: images/...png label)
# - Saves best checkpoint by Val F1, prints Train/Val each epoch, runs final Test

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score, recall_score


# =========================
# Config (edit here)
# =========================
@dataclass
class CFG:
    # Your dataset root; dat paths are like "images/train/xxx.png"
    ROOT_DIR: str = r"E:\Single_frame_smile\data\Isornot"

    # dat files
    TRAIN_DAT: str = r"E:\Single_frame_smile\data\Isornot\train.dat"
    VAL_DAT: str   = r"E:\Single_frame_smile\data\Isornot\20251019before.dat"
    TEST_DAT: str  = r"E:\Single_frame_smile\data\Isornot\20251019after.dat"

    # training
    BATCH_SIZE: int = 32
    EPOCHS: int = 25
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    SEED: int = 42

    # Windows-safe default. If you later add main guard (we already do), you can try 2~8.
    NUM_WORKERS: int = 0

    # speed
    USE_AMP: bool = True
    CUDNN_BENCHMARK: bool = True

    # output
    OUT_DIR: str = r"E:\Single_frame_smile\model\train\outputs"
    CKPT_NAME: str = "best_model.pth"

    # threshold
    THRESH: float = 0.5


cfg = CFG()


# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# =========================
# Dataset
# =========================
class DatDataset(Dataset):
    """
    dat line format:
      images/train/20250926_0_0_10026.png 0
    file on disk:
      ROOT_DIR\images\train\20250926_0_0_10026.png
    """
    def __init__(self, dat_file: str, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.items: List[Tuple[str, int]] = []
        with open(dat_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                p, y = ln.split()
                self.items.append((p, int(y)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel_path, label = self.items[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # Fail fast with clear error if any path is wrong
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# =========================
# Transforms
# =========================
def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_tf, eval_tf


# =========================
# Model
# =========================
def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)  # single logit
    return model.to(device)


# =========================
# Metrics
# =========================
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, thresh: float):
    model.eval()
    all_y, all_pred = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > thresh).long().cpu().numpy()

        total_loss += loss.item() * x.size(0)
        all_pred.extend(preds.tolist())
        all_y.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, zero_division=0)
    rec = recall_score(all_y, all_pred, zero_division=0)
    return avg_loss, acc, f1, rec


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                scaler: torch.cuda.amp.GradScaler,
                use_amp: bool,
                thresh: float):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    all_y, all_pred = [], []
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > thresh).long().cpu().numpy()
            all_pred.extend(preds.tolist())
            all_y.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, zero_division=0)
    rec = recall_score(all_y, all_pred, zero_division=0)
    return avg_loss, acc, f1, rec


# =========================
# Main
# =========================
def main():
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUT_DIR)

    if cfg.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_tf, eval_tf = build_transforms()

    train_ds = DatDataset(cfg.TRAIN_DAT, cfg.ROOT_DIR, transform=train_tf)
    val_ds   = DatDataset(cfg.VAL_DAT,   cfg.ROOT_DIR, transform=eval_tf)
    test_ds  = DatDataset(cfg.TEST_DAT,  cfg.ROOT_DIR, transform=eval_tf)

    # Scheme A: WeightedRandomSampler to balance classes in training batches
    train_labels = [y for _, y in train_ds.items]
    n0 = train_labels.count(0)
    n1 = train_labels.count(1)
    if n0 == 0 or n1 == 0:
        raise RuntimeError(f"Train set must contain both classes. Got n0={n0}, n1={n1}")

    w0 = 1.0 / n0
    w1 = 1.0 / n1
    weights = [w1 if y == 1 else w0 for y in train_labels]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    # Windows-safe: keep num_workers=0 unless you really need more
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.USE_AMP and device.type == "cuda"))

    ckpt_path = os.path.join(cfg.OUT_DIR, cfg.CKPT_NAME)

    print(f"Train size: {len(train_ds)}  (0={n0}, 1={n1})")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")
    print(f"AMP: {cfg.USE_AMP and device.type == 'cuda'} | num_workers: {cfg.NUM_WORKERS}")
    print(f"Saving best checkpoint to: {ckpt_path}\n")

    best_val_f1 = -1.0

    for epoch in range(1, cfg.EPOCHS + 1):
        tr = train_epoch(model, train_loader, optimizer, device, scaler, cfg.USE_AMP, cfg.THRESH)
        va = eval_epoch(model, val_loader, device, cfg.THRESH)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train loss {tr[0]:.4f} | acc {tr[1]:.3f} | f1 {tr[2]:.3f} | rec {tr[3]:.3f} || "
            f"Val loss {va[0]:.4f} | acc {va[1]:.3f} | f1 {va[2]:.3f} | rec {va[3]:.3f}"
        )

        if va[2] > best_val_f1:
            best_val_f1 = va[2]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "cfg": vars(cfg),
                },
                ckpt_path
            )

    print(f"\nTraining finished. Best Val F1 = {best_val_f1:.4f}")

    # Final test using best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te = eval_epoch(model, test_loader, device, cfg.THRESH)
    print(f"[Test] loss {te[0]:.4f} | acc {te[1]:.3f} | f1 {te[2]:.3f} | rec {te[3]:.3f}")


if __name__ == "__main__":
    # Required for Windows multiprocessing safety; also safe even with num_workers=0
    main()
