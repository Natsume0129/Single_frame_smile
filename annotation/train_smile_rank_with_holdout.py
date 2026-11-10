from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_SMILECOMP_DIR = Path(
    r"E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi"
)
DEFAULT_VGGFACE_CONV = DEFAULT_SMILECOMP_DIR / "models" / "vggface_conv.pth"


@dataclass(frozen=True)
class PairRow:
    image1: str
    image2: str
    label: int


class PairDataset(Dataset):
    def __init__(self, rows: list[PairRow], image_dir: Path, transform=None):
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.image_dir / row.image1) as image:
            image1 = image.convert("RGB")
        with Image.open(self.image_dir / row.image2) as image:
            image2 = image.convert("RGB")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        label = 0 if row.label > 0 else 1
        return image1, image2, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SmileComp ranking model with fixed seed and a held-out manual pair set."
    )
    parser.add_argument("--smilecomp_dir", type=Path, default=DEFAULT_SMILECOMP_DIR)
    parser.add_argument("--dataset_file", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_VGGFACE_CONV)
    parser.add_argument("--dst_weights", type=Path, required=True)
    parser.add_argument("--dst_history", type=Path, required=True)
    parser.add_argument("--dst_metrics", type=Path, required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--val_file", type=Path, required=True)
    parser.add_argument("--holdout_file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--holdout_ratio", type=float, default=0.20)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--wo_consist_loss", action="store_true")
    parser.add_argument("--use_existing_splits", action="store_true",
                        help="Read preassigned train/val/holdout files instead of randomly splitting pairs.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def load_pairs(path: Path, image_dir: Path) -> list[PairRow]:
    rows: list[PairRow] = []
    bad_labels: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if not raw or len(raw) < 3:
                continue
            image1 = raw[0].strip()
            image2 = raw[1].strip()
            label_text = raw[2].strip()
            if label_text not in {"1", "-1"}:
                bad_labels[label_text] += 1
                continue
            for image_name in (image1, image2):
                if not (image_dir / image_name).is_file():
                    raise FileNotFoundError(image_dir / image_name)
            rows.append(PairRow(image1=image1, image2=image2, label=int(label_text)))
    if not rows:
        raise ValueError(f"No trainable pair rows found in {path}")
    if bad_labels:
        print(f"[SPLIT] skipped non-trainable labels: {dict(bad_labels)}")
    return rows


def stratified_split(
    rows: list[PairRow],
    seed: int,
    holdout_ratio: float,
    val_ratio: float,
) -> tuple[list[PairRow], list[PairRow], list[PairRow]]:
    rng = random.Random(seed)
    by_label: dict[int, list[PairRow]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)

    train_rows: list[PairRow] = []
    val_rows: list[PairRow] = []
    holdout_rows: list[PairRow] = []
    for label, label_rows in sorted(by_label.items()):
        shuffled = label_rows[:]
        rng.shuffle(shuffled)
        holdout_count = max(1, round(len(shuffled) * holdout_ratio))
        remaining = shuffled[holdout_count:]
        val_count = max(1, round(len(remaining) * val_ratio))
        holdout_rows.extend(shuffled[:holdout_count])
        val_rows.extend(remaining[:val_count])
        train_rows.extend(remaining[val_count:])

    for split_rows in (train_rows, val_rows, holdout_rows):
        rng.shuffle(split_rows)
    return train_rows, val_rows, holdout_rows


def write_dat(path: Path, rows: list[PairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row.image1},{row.image2},{row.label}" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_counts(rows: list[PairRow]) -> dict[str, int]:
    counts = Counter(row.label for row in rows)
    return {str(label): counts[label] for label in sorted(counts)}


def build_transforms(fcmodel):
    train_transform = transforms.Compose(
        [
            transforms.Resize((fcmodel.IMG_HEIGHT_VGG16, fcmodel.IMG_WIDTH_VGG16)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
                (1.0, 1.0, 1.0),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((fcmodel.IMG_HEIGHT_VGG16, fcmodel.IMG_WIDTH_VGG16)),
            transforms.ToTensor(),
            transforms.Normalize(
                (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
                (1.0, 1.0, 1.0),
            ),
        ]
    )
    return train_transform, eval_transform


def stream_consistency(out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    softmax = nn.Softmax(dim=1)
    estimated = torch.cat((softmax(out1), softmax(out2[:, [1, 0]])), 1)
    return torch.mean(torch.min(estimated[:, 0], estimated[:, 2]) + torch.min(estimated[:, 1], estimated[:, 3]))


def run_epoch(model, loader, criterion, optimizer, device, use_consistency_loss: bool, train: bool) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_consistency = 0.0
    total_samples = 0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for image1, image2, label in loader:
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            if train:
                optimizer.zero_grad()

            out1, out2 = model(image1, image2)
            consistency = stream_consistency(out1, out2)
            loss = criterion(out1, label) + criterion(out2, 1 - label)
            if use_consistency_loss:
                loss = loss + (1 - consistency)

            if train:
                loss.backward()
                optimizer.step()

            _, pred = torch.max(out1, 1)
            batch_size = label.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int(pred.eq(label).sum().item())
            total_consistency += float(consistency.item()) * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "consistency": total_consistency / total_samples,
    }


def plot_history(path: Path, history: list[dict[str, float]], holdout_metrics: dict[str, float]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots()
    ax.plot([0, max(epochs)], [1.0, 1.0], color="black", linestyle="dashed", label="1.0")
    for key in (
        "train_loss",
        "train_accuracy",
        "train_consistency",
        "val_loss",
        "val_accuracy",
        "val_consistency",
    ):
        ax.plot(epochs, [row[key] for row in history], label=key)
    ax.axhline(holdout_metrics["accuracy"], color="purple", linestyle=":", label="holdout_accuracy")
    ax.set_ylim(0.0, 2.0)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.smilecomp_dir.is_dir():
        raise FileNotFoundError(args.smilecomp_dir)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    sys.path.insert(0, str(args.smilecomp_dir))
    import face_comp_torch as fcmodel  # noqa: PLC0415

    all_rows = load_pairs(args.dataset_file, args.dataset_dir)
    if args.use_existing_splits:
        train_rows, val_rows, holdout_rows = [
            load_pairs(path, args.dataset_dir)
            for path in (args.train_file, args.val_file, args.holdout_file)
        ]
        sets = [{tuple(sorted((r.image1, r.image2))) for r in rows}
                for rows in (train_rows, val_rows, holdout_rows)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Pair overlap in existing splits.")
        image_sets = [{name for r in rows for name in (r.image1, r.image2)}
                      for rows in (train_rows, val_rows, holdout_rows)]
        if any(image_sets[i] & image_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Image overlap in existing splits.")
        if Counter(train_rows + val_rows + holdout_rows) != Counter(all_rows):
            raise ValueError("Existing splits do not match dataset_file.")
    else:
        train_rows, val_rows, holdout_rows = stratified_split(
            all_rows, args.seed, args.holdout_ratio, args.val_ratio,
        )
        write_dat(args.train_file, train_rows)
        write_dat(args.val_file, val_rows)
        write_dat(args.holdout_file, holdout_rows)

    train_transform, eval_transform = build_transforms(fcmodel)
    train_dataset = PairDataset(train_rows, args.dataset_dir, train_transform)
    val_dataset = PairDataset(val_rows, args.dataset_dir, eval_transform)
    holdout_dataset = PairDataset(holdout_rows, args.dataset_dir, eval_transform)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    holdout_loader = DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"program works on {device}")
    print(
        f"full: {len(all_rows)} -> train: {len(train_rows)}, "
        f"val: {len(val_rows)}, holdout: {len(holdout_rows)}"
    )
    print(f"label counts train={label_counts(train_rows)} val={label_counts(val_rows)} holdout={label_counts(holdout_rows)}")

    model = fcmodel.face_comp_siamese_model_vgg16based_for_train(weights=str(args.weights)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    use_consistency_loss = not args.wo_consist_loss

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, use_consistency_loss, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, use_consistency_loss, train=False)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_consistency": train_metrics["consistency"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_consistency": val_metrics["consistency"],
        }
        history.append(history_row)
        print(
            f"epoch: {epoch}, train - loss:{train_metrics['loss']}, "
            f"ACC:{100 * train_metrics['accuracy']:.6f}%, consistency: {train_metrics['consistency']}"
        )
        print(
            f"            val - loss:{val_metrics['loss']}, "
            f"ACC:{100 * val_metrics['accuracy']:.6f}%, consistency: {val_metrics['consistency']}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state was captured.")
    model.load_state_dict(best_state)
    args.dst_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.dst_weights)

    holdout_metrics = run_epoch(
        model,
        holdout_loader,
        criterion,
        optimizer,
        device,
        use_consistency_loss,
        train=False,
    )
    plot_history(args.dst_history, history, holdout_metrics)

    metrics = {
        "dataset_file": str(args.dataset_file),
        "dataset_dir": str(args.dataset_dir),
        "weights": str(args.weights),
        "dst_weights": str(args.dst_weights),
        "dst_history": str(args.dst_history),
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "holdout_file": str(args.holdout_file),
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "use_consistency_loss": use_consistency_loss,
        "device": str(device),
        "num_all_pairs": len(all_rows),
        "num_train_pairs": len(train_rows),
        "num_val_pairs": len(val_rows),
        "num_holdout_pairs": len(holdout_rows),
        "label_counts_train": label_counts(train_rows),
        "label_counts_val": label_counts(val_rows),
        "label_counts_holdout": label_counts(holdout_rows),
        "best_epoch_by_val_loss": best_epoch,
        "best_val_loss": best_val_loss,
        "holdout_loss": holdout_metrics["loss"],
        "holdout_accuracy": holdout_metrics["accuracy"],
        "holdout_consistency": holdout_metrics["consistency"],
        "history": history,
    }
    args.dst_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.dst_metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
