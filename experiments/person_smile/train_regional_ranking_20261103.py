"""Train frozen-backbone regional rankers on the existing clip-grouped pairs.

This is a NEW matched-control protocol: six fixed augmentation views replace the
old trainer's continuously random rotations. The full-face control uses precisely
the same protocol as the five regional inputs. No historical output is modified.

Feature cache (torch.load): image_files[N], features float32[N,6,25088],
split_by_image{name:split}, native_pixel_hashes{name:hash}, and fingerprint.
View 0 is the unaugmented input used for training-only scale construction;
validation/test have view 0 repeated, and are never augmented during evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
# Must be set before the first CUDA BLAS operation in this training process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "output/monotonic_accepted_training_20261101"
OUT = ROOT / "output/regional_ranking_extension_20261103"
PERSONS = ("JIANG", "JIE", "TAKEDA", "XU", "YAMAMOTO")
LABELS = ("polite", "bitter", "true")
REGIONS = ("full", "periocular", "nose", "mouth_perioral", "upper_face", "lower_face")
MEAN_RGB = (129.1863, 104.7624, 93.5940)
FEATURE_DIM = 512 * 7 * 7
VIEW_NAMES = ("original", "mirror", "rotate_minus5", "rotate_plus5",
              "mirror_rotate_minus5", "mirror_rotate_plus5")
VERSION = "regional_fixed_six_views_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_torch(path, map_location="cpu"):
    # These are trusted local experiment artifacts, with fingerprints checked below.
    return torch.load(path, map_location=map_location, weights_only=False)


def make_head():
    """Exact comp_layer architecture from face_comp_torch.siamese_vgg16based."""
    return nn.Sequential(nn.Linear(FEATURE_DIM * 2, 512), nn.ReLU(), nn.Dropout(.25),
                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.25),
                         nn.Linear(256, 256), nn.ReLU(), nn.Dropout(.25),
                         nn.Linear(256, 2))


def load_trained_head(model_dir, device="cpu"):
    """Return (evaluation-mode comparison head, checkpoint metadata)."""
    checkpoint = load_torch(Path(model_dir) / "head.pth")
    meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    if sha(meta["backbone_path"]) != meta["backbone_sha256"]:
        raise ValueError("Frozen backbone hash changed")
    head = make_head()
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval(), meta


def load_feature_cache(person, label, region, output_root=OUT):
    return load_torch(Path(output_root) / "cache" / person / label / (region + ".pt"))


def native_image_tensor(image_path, mask):
    """Exact native inference preprocessing shared with regional scoring.

    Resize to 224, apply static region mask with floating VGG mean RGB outside,
    and subtract VGG mean after division by 255. Returns CHW normalized tensor
    and the masked float32 RGB image, useful for pixel-duplicate auditing.
    """
    from regional_masks_20261103 import mask_image
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
    masked = np.asarray(mask_image(rgb, mask), dtype=np.float32)
    if masked.shape != (224, 224, 3) or not np.isfinite(masked).all():
        raise ValueError(f"Invalid masked image: {image_path}")
    tensor = torch.from_numpy(masked.copy()).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(MEAN_RGB, dtype=torch.float32)[:, None, None]) / 255.0
    return tensor.contiguous(), masked


def six_views(native_tensor):
    """Rotate normalized float inputs with zero fill = exact VGG mean RGB."""
    mirrored = TF.hflip(native_tensor)
    rotate = lambda value, angle: TF.rotate(value, angle, interpolation=InterpolationMode.BILINEAR,
                                           expand=False, fill=0.0)
    return (native_tensor, mirrored, rotate(native_tensor, -5), rotate(native_tensor, 5),
            rotate(mirrored, -5), rotate(mirrored, 5))


def parse_pairs(path):
    result = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line in csv.reader(stream):
            if not line:
                continue
            if len(line) != 3 or line[2].strip() not in ("-1", "1"):
                raise ValueError(f"Unexpected pair row in {path}: {line}")
            result.append((line[0].strip(), line[1].strip(), 0 if int(line[2]) > 0 else 1))
    if not result:
        raise ValueError(f"Empty pair set: {path}")
    return result


def source_inputs(person, label):
    if person not in PERSONS or label not in LABELS:
        raise ValueError((person, label))
    latest = read_json(UPSTREAM / "models" / person / label / "latest.json")
    directory = Path(latest["directory"])
    cfg = read_json(directory / "config.json")
    clips = read_json(directory / "accepted_clips.json")
    pairs = {split: parse_pairs(directory / (split + ".dat")) for split in ("train", "val", "test")}
    frame_info = {}
    group_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for clip in clips:
        if clip["person"] != person or clip["label"] != label:
            raise ValueError("Source model directory contains another person/category")
        group_splits[clip["split_group_id"]].add(clip["split"])
        source_splits[clip["source_sha256"]].add(clip["split"])
        for frame in clip["sample_frames"]:
            name = clip["id"] + "_" + str(frame["source_frame"]) + ".png"
            frame_info[name] = dict(split=clip["split"], sha256=frame["sha256"], clip_id=clip["id"],
                                    source_group=clip["group_id"], component=clip["split_group_id"],
                                    source_video=clip["source_video"], source_frame=frame["source_frame"])
    if any(len(value) != 1 for value in list(group_splits.values()) + list(source_splits.values())):
        raise ValueError("Upstream source groups cross splits")
    used = set()
    for split, rows in pairs.items():
        for first, second, target in rows:
            for name in (first, second):
                if frame_info[name]["split"] != split:
                    raise ValueError(f"Pair/image split mismatch: {name}")
                used.add(name)
            if frame_info[first]["clip_id"] != frame_info[second]["clip_id"]:
                raise ValueError("Expected current within-clip temporal pairs")
    names = sorted(used)
    for name in names:
        if sha(directory / "images" / name) != frame_info[name]["sha256"]:
            raise ValueError(f"Source image hash changed: {name}")
    return directory, cfg, pairs, names, frame_info


def prepare_features(person, label, region, device="cuda:0", output_root=OUT, feature_batch=32):
    """Prepare/reuse immutable features and copied original pair metadata."""
    from regional_masks_20261103 import load_masks
    if region not in REGIONS:
        raise ValueError(region)
    output_root = Path(output_root)
    destination = output_root / "models" / person / label / region
    source, cfg, pairs, names, frame_info = source_inputs(person, label)
    masks = load_masks(person)
    mask = np.asarray(masks[region])
    if mask.shape != (224, 224) or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError(f"Invalid {person}/{region} mask")
    if np.count_nonzero(mask > .5) < 16:
        raise ValueError(f"Empty/tiny {person}/{region} mask")
    backbone = Path(cfg["initial_weights"])
    fingerprint = dict(version=VERSION, person=person, label=label, region=region,
                       source_directory=str(source), source_pairs={s: sha(source / (s + ".dat"))
                       for s in ("train", "val", "test", "all")},
                       source_images={n: frame_info[n]["sha256"] for n in names},
                       backbone_path=str(backbone), backbone_sha256=sha(backbone),
                       mask_sha256=array_sha(mask), mask_pixels=int(np.count_nonzero(mask > .5)),
                       input_processing="PIL resize bilinear 224; static mask; exact float mean fill; divide255 and center",
                       mean_rgb=list(MEAN_RGB), views=list(VIEW_NAMES), train_augmentation="six fixed views",
                       inference_view=0, source_manifest_sha256=sha(source / "accepted_clips.json"))
    cache_path = output_root / "cache" / person / label / (region + ".pt")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "input_fingerprint.json").exists():
        if read_json(destination / "input_fingerprint.json") != fingerprint:
            raise ValueError(f"Inputs changed; use a new output root: {destination}")
    else:
        save_json(destination / "input_fingerprint.json", fingerprint)
    for name in ("train.dat", "val.dat", "test.dat", "all.dat", "accepted_clips.json", "config.json"):
        target = destination / name
        if target.exists():
            if sha(target) != sha(source / name):
                raise ValueError(f"Copied metadata changed: {target}")
        else:
            shutil.copy2(source / name, target)
    save_json(destination / "images.json", {n: {**frame_info[n], "path": str(source / "images" / n)} for n in names})
    candidates = [dict(image_file=n, person=person, label=label, region=region, split="train",
                       source_video=frame_info[n]["source_video"], source_frame_index=frame_info[n]["source_frame"],
                       sequence_id=frame_info[n]["clip_id"], source_frame_path=str(source / "images" / n))
                  for n in names if frame_info[n]["split"] == "train"]
    with (destination / "scale_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    if cache_path.exists():
        cache = load_torch(cache_path)
        if cache["fingerprint"] != fingerprint or cache["image_files"] != names:
            raise ValueError(f"Feature cache mismatch: {cache_path}")
        return cache, destination, pairs, cfg
    sys.path.insert(0, cfg["smilecomp_dir"])
    import face_comp_torch as fc
    extractor = fc.VGGFace_conv()
    extractor.load_state_dict(load_torch(backbone))
    extractor = extractor.to(device).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    features = torch.empty((len(names), 6, FEATURE_DIM), dtype=torch.float32)
    pending, positions = [], []
    native_hashes, hash_splits = {}, defaultdict(set)

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            value = extractor(torch.stack(pending).to(device)).cpu()
        for (image_i, view_i), feature in zip(positions, value):
            features[image_i, view_i].copy_(feature)
        pending.clear()
        positions.clear()

    for index, name in enumerate(names):
        native, masked = native_image_tensor(source / "images" / name, mask)
        pixel_hash = array_sha(masked)
        native_hashes[name] = pixel_hash
        hash_splits[pixel_hash].add(frame_info[name]["split"])
        views = six_views(native) if frame_info[name]["split"] == "train" else (native,)
        for view_index, view in enumerate(views):
            pending.append(view)
            positions.append((index, view_index))
            if len(pending) >= feature_batch:
                flush()
    flush()
    collisions = {h: sorted(s) for h, s in hash_splits.items() if len(s) > 1}
    audit = dict(images=len(names), by_split=dict(Counter(frame_info[n]["split"] for n in names)),
                 source_groups_cross_split=0, masked_duplicate_hashes_cross_split=collisions,
                 mask_pixels=int(np.count_nonzero(mask > .5)))
    save_json(destination / "input_audit.json", audit)
    if collisions:
        raise ValueError(f"Masked native-image duplicates cross splits: {destination}")
    for index, name in enumerate(names):
        if frame_info[name]["split"] != "train":
            features[index, 1:] = features[index, 0]
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite frozen features")
    cache = dict(image_files=names, features=features, native_pixel_hashes=native_hashes,
                 split_by_image={n: frame_info[n]["split"] for n in names}, fingerprint=fingerprint)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{person}/{label}/{region}: cached {len(names)} images", flush=True)
    return cache, destination, pairs, cfg


def comparison_loss(head, first, second, targets):
    original = head(torch.cat((first, second), dim=1))
    swapped = head(torch.cat((second, first), dim=1))
    p = original.softmax(dim=1)
    q = swapped[:, [1, 0]].softmax(dim=1)
    consistency = torch.minimum(p, q).sum(dim=1).mean()
    loss = nn.functional.cross_entropy(original, targets) + nn.functional.cross_entropy(swapped, 1 - targets) + 1 - consistency
    return loss, original, consistency


def run_epoch(head, features, rows, optimizer, device, epoch, generator, batch_size=32):
    training = optimizer is not None
    head.train(training)
    order = torch.randperm(len(rows), generator=generator) if training else torch.arange(len(rows))
    n, loss_sum, correct, consistency_sum = 0, 0.0, 0, 0.0
    with torch.set_grad_enabled(training):
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            batch = rows[indices]
            # Per-image independent fixed-view schedules, identical in every ROI/control.
            first_view = (epoch + indices * 3 + 1) % 6 if training else torch.zeros_like(indices)
            second_view = (epoch * 5 + indices * 7 + 2) % 6 if training else torch.zeros_like(indices)
            first = features[batch[:, 0], first_view].to(device)
            second = features[batch[:, 1], second_view].to(device)
            targets = batch[:, 2].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, logits, consistency = comparison_loss(head, first, second, targets)
            if training:
                loss.backward()
                optimizer.step()
            count = len(indices)
            loss_sum += float(loss.detach()) * count
            consistency_sum += float(consistency.detach()) * count
            correct += int(logits.argmax(dim=1).eq(targets).sum())
            n += count
    return dict(pairs=n, loss=loss_sum / n, accuracy=correct / n, correct=correct,
                consistency=consistency_sum / n)


def train_one(person, label, region, device="cuda:0", output_root=OUT, epochs=100, feature_batch=32):
    if not 1 <= epochs <= 100:
        raise ValueError("epochs must be in [1,100]")
    started = time.time()
    cache, destination, pairs, cfg = prepare_features(person, label, region, device, output_root, feature_batch)
    seed = int(cfg["seed"])
    protocol = dict(version=VERSION, seed=seed, epochs=epochs, early_stopping=False,
                    batch_size=32, learning_rate=5e-5, optimizer="Adam",
                    selection="minimum sample-weighted validation loss; test once after selection",
                    architecture="frozen VGGFace conv + original Siamese comparison head",
                    loss="CE(original,target)+CE(swapped,1-target)+1-stream_consistency",
                    train_views=list(VIEW_NAMES), train_view_schedule="left=(epoch+pair_index*3+1)%6; right=(epoch*5+pair_index*7+2)%6",
                    val_test_view=0, matched_full_face_control=True,
                    protocol_change="Fixed six-view training, not the old random-continuous-rotation training",
                    input_fingerprint=cache["fingerprint"])
    if (destination / "complete.json").exists():
        if read_json(destination / "protocol.json") != protocol:
            raise ValueError(f"Completed model has a different protocol: {destination}")
        return read_json(destination / "metrics.json")
    save_json(destination / "protocol.json", protocol)
    seed_everything(seed)
    # Instantiate original constructor to retain its initialization ordering.
    sys.path.insert(0, cfg["smilecomp_dir"])
    import face_comp_torch as fc
    original = fc.face_comp_siamese_model_vgg16based_for_train(weights=None)
    head = original.comp_layer.to(device)
    del original
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-5)
    name_to_index = {name: index for index, name in enumerate(cache["image_files"])}
    rows = {s: torch.tensor([(name_to_index[a], name_to_index[b], y) for a, b, y in p], dtype=torch.long)
            for s, p in pairs.items()}
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_state = float("inf"), -1, None
    history = []
    for epoch in range(epochs):
        training = run_epoch(head, cache["features"], rows["train"], optimizer, device, epoch, generator)
        validation = run_epoch(head, cache["features"], rows["val"], None, device, epoch, generator)
        history.append(dict(epoch=epoch + 1, train=training, val=validation))
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            save_json(destination / "history.json", history)
            save_json(destination / "progress.json", dict(state="training", epoch=epoch + 1, epochs=epochs,
                      best_epoch=best_epoch, best_val_loss=best_loss, elapsed_seconds=time.time() - started))
            print(f"{person}/{label}/{region} {epoch + 1}/{epochs}: train={training['accuracy']:.3f} val={validation['accuracy']:.3f} val_loss={validation['loss']:.5f}", flush=True)
    head.load_state_dict(best_state)
    test = run_epoch(head, cache["features"], rows["test"], None, device, 0, generator)
    fingerprint = cache["fingerprint"]
    checkpoint = dict(state_dict=best_state, format="comparison_head_only_v1", feature_dim=FEATURE_DIM,
                      person=person, label=label, region=region,
                      backbone_path=fingerprint["backbone_path"], backbone_sha256=fingerprint["backbone_sha256"],
                      mask_sha256=fingerprint["mask_sha256"], best_epoch=best_epoch, protocol=protocol)
    temporary = destination / "head.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(destination / "head.pth")
    metrics = dict(person=person, label=label, region=region, best_epoch=best_epoch, best_val_loss=best_loss,
                   train_pairs=len(pairs["train"]), val_pairs=len(pairs["val"]), test_pairs=len(pairs["test"]),
                   best_epoch_metrics=history[best_epoch - 1], test=test,
                   elapsed_seconds=time.time() - started, head_path=str(destination / "head.pth"),
                   head_sha256=sha(destination / "head.pth"), feature_cache=str(Path(output_root) / "cache" / person / label / (region + ".pt")))
    save_json(destination / "history.json", history)
    save_json(destination / "metrics.json", metrics)
    save_json(destination / "complete.json", dict(head_sha256=metrics["head_sha256"], protocol_sha256=sha(destination / "protocol.json")))
    save_json(destination / "progress.json", dict(state="complete", epochs=epochs, best_epoch=best_epoch))
    del head, optimizer, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", nargs="+", choices=PERSONS, default=list(PERSONS))
    parser.add_argument("--label", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--region", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--feature-batch", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for person in args.person:
        for label in args.label:
            for region in args.region:
                if args.prepare_only:
                    prepare_features(person, label, region, args.device, args.output_root, args.feature_batch)
                else:
                    train_one(person, label, region, args.device, args.output_root, args.epochs, args.feature_batch)


if __name__ == "__main__":
    main()
