"""Build train-only regional scales and score identical reviewed frames.

Each participant/region input is extracted once per native image batch and shared
by its three category comparison heads. Original gallery data is never modified.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F

from regional_masks_20261103 import load_masks
from train_regional_ranking_20261103 import (
    FEATURE_DIM, LABELS, OUT, PERSONS, REGIONS, ROOT, array_sha,
    load_feature_cache, load_torch, load_trained_head, native_image_tensor,
    read_json, save_json, sha,
)

GALLERY = ROOT / "output/three_model_clip_gallery_20261102"
SCALES = OUT / "scales"
CURVES = OUT / "curves"
VERSION = "fixed_regional_native_scale_curve_v1"


def object_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class FactoredHead:
    """Algebraic factorization of the first concatenated linear layer only."""

    def __init__(self, head: nn.Sequential):
        head.eval()
        if not isinstance(head[0], nn.Linear) or head[0].in_features != 2 * FEATURE_DIM:
            raise ValueError("Expected original comparison-head first linear layer")
        self.head = head
        self.first = head[0]
        self.remaining = nn.Sequential(*list(head.children())[1:]).eval()
        self.device = self.first.weight.device

    @torch.inference_mode()
    def project(self, features: torch.Tensor):
        values = features.to(self.device)
        return (F.linear(values, self.first.weight[:, :FEATURE_DIM]),
                F.linear(values, self.first.weight[:, FEATURE_DIM:]))

    @torch.inference_mode()
    def paired_probability(self, left, right):
        """Elementwise symmetric P(first image stronger), projected pairs."""
        al, ar = left
        bl, br = right
        forward = self.remaining(al + br + self.first.bias).softmax(dim=-1)
        reverse = self.remaining(bl + ar + self.first.bias).softmax(dim=-1)
        return (forward[:, 0] + reverse[:, 1]) * .5

    @torch.inference_mode()
    def probabilities(self, queries, references, block=32):
        """All query/reference symmetric probabilities, returned on CPU."""
        al, ar = queries
        bl, br = references
        chunks = []
        for start in range(0, len(al), block):
            a, b = al[start:start + block], ar[start:start + block]
            shape = (len(a), len(bl))
            forward = self.remaining((a[:, None] + br[None, :] + self.first.bias).reshape(-1, a.shape[-1])).softmax(dim=-1)
            reverse = self.remaining((bl[None, :] + b[:, None] + self.first.bias).reshape(-1, a.shape[-1])).softmax(dim=-1)
            chunks.append(((forward[:, 0] + reverse[:, 1]) * .5).reshape(shape).cpu())
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def verify(self, features: torch.Tensor):
        """Numerically compare actual native features with original concatenation."""
        if len(features) < 2:
            raise ValueError("Need actual feature pairs for factorization verification")
        indices = np.rint(np.linspace(0, len(features) - 1, min(20, len(features)))).astype(int)
        first = features[indices].to(self.device)
        second = first.flip(0)
        direct_f = self.head(torch.cat((first, second), dim=1)).softmax(dim=-1)
        direct_r = self.head(torch.cat((second, first), dim=1)).softmax(dim=-1)
        expected = (direct_f[:, 0] + direct_r[:, 1]) * .5
        actual = self.paired_probability(self.project(first), self.project(second))
        difference = float((expected - actual).abs().max())
        if not np.isfinite(difference) or difference >= .001:
            raise ValueError(f"Factorization differs from direct head: max |delta P|={difference}")
        return dict(actual_native_feature_pairs=len(first), max_absolute_probability_difference=difference,
                    required_maximum=.001, passed=True,
                    method="first Linear W=[WL,WR]; WL*fA+WR*fB+b; remaining layers unchanged in eval mode")


def model_path(person, label, region):
    return OUT / "models" / person / label / region


def require_model(person, label, region):
    directory = model_path(person, label, region)
    if not (directory / "complete.json").is_file() or not (directory / "head.pth").is_file():
        raise FileNotFoundError(f"Trained regional model is not complete: {directory}")
    completion = read_json(directory / "complete.json")
    if sha(directory / "head.pth") != completion["head_sha256"]:
        raise ValueError(f"Checkpoint differs from completed training: {directory}")
    return directory, completion


def render_scale(directory: Path, person: str, label: str, region: str):
    tile, gap, margin = 150, 9, 16
    image = Image.new("RGB", (margin * 2 + tile * 10 + gap * 9, 238), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    draw.text((margin, 12), f"{person} / {label} / {region} | Training-only model ranking: 0 weakest -> 9 strongest", fill="#183047", font=font)
    for level in range(10):
        x = margin + level * (tile + gap)
        with Image.open(directory / f"level_{level:02d}.png") as source:
            image.paste(source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), (x, 48))
        draw.text((x + 47, 208), f"Level {level}", fill="#183047", font=small)
    image.save(directory / "scale10_montage.png")


def build_scale(person, label, region, device="cuda:0"):
    started = time.time()
    model_dir, completion = require_model(person, label, region)
    destination = SCALES / person / label / region
    destination.mkdir(parents=True, exist_ok=True)
    mask = load_masks(person)[region]
    cache = load_feature_cache(person, label, region)
    input_fp = cache["fingerprint"]
    if input_fp != read_json(model_dir / "input_fingerprint.json") or input_fp["mask_sha256"] != array_sha(mask):
        raise ValueError(f"Model/cache/mask identity mismatch: {model_dir}")
    candidates = [i for i, name in enumerate(cache["image_files"]) if cache["split_by_image"][name] == "train"]
    names = [cache["image_files"][i] for i in candidates]
    if len(names) < 10:
        raise ValueError(f"Need ten distinct training candidate images: {model_dir}")
    fingerprint = dict(version=VERSION, model_sha256=completion["head_sha256"], mask_sha256=array_sha(mask),
                       cache_input_sha256=object_sha(input_fp), training_candidates=names,
                       candidate_selection="All native-view images participating in training pairs only",
                       ordering="all unordered pairs; symmetric probability; hard wins + mean probability tie-break",
                       selection="round(k*(N-1)/9), k=0..9 over weak-to-strong order",
                       feature_view=0)
    done = destination / "complete.json"
    if done.exists():
        prior = read_json(done)
        if prior["fingerprint"] != fingerprint:
            raise ValueError(f"Refusing to mix stale regional scale: {destination}")
        if sha(destination / "anchor_features.pt") != prior["anchor_features_sha256"]:
            raise ValueError(f"Anchor features changed: {destination}")
        return prior
    features = cache["features"][candidates, 0].contiguous()
    del cache
    head, meta = load_trained_head(model_dir, device)
    if (meta["person"], meta["label"], meta["region"]) != (person, label, region):
        raise ValueError("Loaded wrong participant/type/region head")
    factored = FactoredHead(head)
    audit = factored.verify(features)
    projected = factored.project(features)
    size = len(names)
    hard_wins = np.zeros(size, dtype=np.int32)
    score_sum = np.zeros(size, dtype=np.float64)
    probability = np.full((size, size), np.nan, dtype=np.float32)
    with torch.inference_mode():
        # The i<j convention exactly matches the earlier all-pairs voting builder,
        # including its deterministic treatment of a probability exactly 0.5.
        for start in range(0, size, 32):
            block = factored.probabilities((projected[0][start:start+32], projected[1][start:start+32]), projected).numpy()
            for local, i in enumerate(range(start, min(start + 32, size))):
                if i + 1 >= size:
                    continue
                js = np.arange(i + 1, size)
                values = block[local, js].astype(np.float64)
                probability[i, js] = values
                probability[js, i] = 1 - values
                score_sum[i] += values.sum()
                score_sum[js] += 1 - values
                hard_wins[i] += int((values >= .5).sum())
                hard_wins[js] += (values < .5).astype(np.int32)
    mean_probability = score_sum / (size - 1)
    vote = hard_wins + mean_probability
    ranking = sorted(range(size), key=lambda i: (vote[i], mean_probability[i], names[i]))
    positions = [round(k * (size - 1) / 9) for k in range(10)]
    chosen = [ranking[pos] for pos in positions]
    source_info = read_json(model_dir / "images.json")
    ranked_rows = []
    for rank, i in enumerate(ranking):
        info = source_info[names[i]]
        if info["split"] != "train":
            raise ValueError("Non-training image entered the regional scale")
        ranked_rows.append(dict(rank_weak_to_strong=rank, image_file=names[i],
                                vote_score=float(vote[i]), hard_wins=int(hard_wins[i]),
                                mean_stronger_probability=float(mean_probability[i]),
                                source_image_path=info["path"], source_sha256=info["sha256"],
                                source_clip_id=info["clip_id"], source_frame=info["source_frame"], split="train"))
    items = []
    for level, (position, i) in enumerate(zip(positions, chosen)):
        info = source_info[names[i]]
        if sha(Path(info["path"])) != info["sha256"]:
            raise ValueError(f"Anchor source image changed: {info['path']}")
        _, masked = native_image_tensor(info["path"], mask)
        path = destination / f"level_{level:02d}.png"
        Image.fromarray(np.clip(np.rint(masked), 0, 255).astype(np.uint8)).save(path)
        items.append(dict(scale_level=level, rank_position_weak_to_strong=position,
                          reference_image=path.name, reference_image_sha256=sha(path),
                          source_image_file=names[i], source_image_path=info["path"], source_sha256=info["sha256"],
                          source_video=info["source_video"], source_clip_id=info["clip_id"], source_frame=info["source_frame"],
                          person=person, label=label, region=region, split="train",
                          vote_score=float(vote[i]), hard_wins=int(hard_wins[i]),
                          mean_stronger_probability=float(mean_probability[i])))
    anchor_payload = dict(features=features[chosen].clone(), image_files=[names[i] for i in chosen],
                          fingerprint=fingerprint, feature_view=0,
                          note="Native float32-mask features; displayed PNG is rounded and must not replace these inference anchors")
    temporary = destination / "anchor_features.tmp"
    torch.save(anchor_payload, temporary)
    temporary.replace(destination / "anchor_features.pt")
    write_csv(destination / "scale10_ranked_all.csv", ranked_rows)
    write_csv(destination / "scale10_items.csv", items)
    save_json(destination / "scale10_items.json", items)
    np.save(destination / "strong_probability_matrix.npy", probability)
    save_json(destination / "factorization_audit.json", audit)
    render_scale(destination, person, label, region)
    result = dict(fingerprint=fingerprint, person=person, label=label, region=region,
                  training_candidate_images=size, levels=10, unordered_pairs=size * (size-1)//2,
                  anchor_features_sha256=sha(destination / "anchor_features.pt"),
                  scale_items_sha256=sha(destination / "scale10_items.json"),
                  factorization_audit=audit, elapsed_seconds=time.time()-started)
    save_json(done, result)
    del features, head, factored, projected
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Scale {person}/{label}/{region}: {size} training images, 10 levels", flush=True)
    return result


def scale_summary():
    rows = []
    for path in sorted(SCALES.glob("*/*/*/complete.json")):
        result = read_json(path)
        rows.append(dict(person=result["person"], label=result["label"], region=result["region"],
                         directory=str(path.parent), candidates=result["training_candidate_images"],
                         model_sha256=result["fingerprint"]["model_sha256"],
                         factorization_max_difference=result["factorization_audit"]["max_absolute_probability_difference"]))
    save_json(SCALES / "manifest.json", rows)
    save_json(SCALES / "summary.json", dict(completed_scales=len(rows), expected_scales=90, levels_each=10))


def smooth(values, times):
    values = np.asarray(values, dtype=float)
    return [float(np.median(values[np.abs(times - value) <= .05])) for value in times]


def source_identity(row, gallery_hash):
    frames = []
    for frame in row["frames"]:
        path = Path(frame["face_path"])
        frames.append(dict(source_frame=frame["source_frame"], source_seconds=frame["source_seconds"],
                           seconds=frame["seconds"], path=str(path), sha256=sha(path)))
    return dict(gallery_manifest_sha256=gallery_hash, source_video_sha256=row["source_sha256"],
                onset_frame=row["onset_frame"], peak_frame=row["peak_frame"], frames_sha256=object_sha(frames))


def score_region(person, region, rows, identities, device="cuda:0", feature_batch=32):
    """Extract native masked features once and reuse across all three heads."""
    mask = load_masks(person)[region]
    models, specs, heads = {}, {}, {}
    cfg = None
    for label in LABELS:
        directory, done = require_model(person, label, region)
        scale_dir = SCALES / person / label / region
        if not (scale_dir / "complete.json").is_file():
            raise FileNotFoundError(f"Build the regional scale first: {scale_dir}")
        complete = read_json(scale_dir / "complete.json")
        if complete["fingerprint"]["model_sha256"] != done["head_sha256"] or complete["fingerprint"]["mask_sha256"] != array_sha(mask):
            raise ValueError("Scale/model/mask mismatch")
        if sha(scale_dir / "anchor_features.pt") != complete["anchor_features_sha256"]:
            raise ValueError("Scale anchors changed")
        head, meta = load_trained_head(directory, device)
        if (meta["person"], meta["label"], meta["region"]) != (person, label, region):
            raise ValueError("Wrong comparison head identity")
        heads[label] = FactoredHead(head)
        anchors = load_torch(scale_dir / "anchor_features.pt")
        if anchors["features"].shape != (10, FEATURE_DIM) or anchors["fingerprint"] != complete["fingerprint"]:
            raise ValueError("Unexpected native scale features")
        models[label] = heads[label].project(anchors["features"])
        specs[label] = dict(head_sha256=done["head_sha256"], model=str(directory / "head.pth"),
                            mask_sha256=array_sha(mask), scale_sha256=complete["anchor_features_sha256"],
                            scale_directory=str(scale_dir), backbone_sha256=meta["backbone_sha256"])
        cfg = read_json(directory / "config.json")
    if len({v["backbone_sha256"] for v in specs.values()}) != 1:
        raise ValueError("Cannot share extraction across different frozen backbones")
    fingerprint_common = dict(version=VERSION, person=person, region=region, models=specs,
                              score="clamp(sum over ten anchors of symmetric P(query stronger)-0.5,0,9)",
                              smoothing="centered +/-0.05 second median on exact retained frame times",
                              frame_selection="Unmodified human-reviewed gallery onset through peak, inclusive")
    pending = []
    for row in rows:
        destination = CURVES / person / row["label"] / row["id"]
        cache_file = destination / ("region_" + region + ".json")
        fingerprint = dict(**fingerprint_common, source=identities[row["id"]])
        if cache_file.exists():
            old = read_json(cache_file)
            if old["fingerprint"] != fingerprint or len(old["points"]) != len(row["frames"]):
                raise ValueError(f"Stale regional curve: {cache_file}")
        else:
            pending.append((row, cache_file, fingerprint))
    if not pending:
        print(f"Curves {person}/{region}: existing {len(rows)} clips", flush=True)
        return
    sys.path.insert(0, cfg["smilecomp_dir"])
    import face_comp_torch as fc
    extractor = fc.VGGFace_conv()
    backbone = Path(cfg["initial_weights"])
    if sha(backbone) != next(iter(specs.values()))["backbone_sha256"]:
        raise ValueError("Backbone configuration changed")
    extractor.load_state_dict(load_torch(backbone))
    extractor = extractor.to(device).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    # Batch across clip boundaries for efficient extraction of short clips.
    outputs = {row["id"]: {label: [None] * len(row["frames"]) for label in LABELS} for row, _, _ in pending}
    tensors, positions = [], []
    completed = 0
    remaining = {row["id"]: len(row["frames"]) for row, _, _ in pending}
    lookup = {row["id"]: (row, dest, fp) for row, dest, fp in pending}
    validation_done = False

    def flush():
        nonlocal completed, validation_done
        if not tensors:
            return
        with torch.inference_mode():
            features = extractor(torch.stack(tensors).to(device))
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not torch.isfinite(features).all():
                raise ValueError("Invalid native masked frame features")
            if not validation_done:
                audit = {label: heads[label].verify(features) for label in LABELS}
                save_json(CURVES / person / (region + "_factorization_audit.json"), audit)
                validation_done = True
            for label in LABELS:
                probability = heads[label].probabilities(heads[label].project(features), models[label])
                scores = torch.clamp(probability.sum(dim=1) - .5, 0, 9).numpy()
                for (clip_id, frame_index), value in zip(positions, scores):
                    outputs[clip_id][label][frame_index] = float(value)
        newly_complete = []
        for clip_id, frame_index in positions:
            remaining[clip_id] -= 1
            if remaining[clip_id] == 0:
                newly_complete.append(clip_id)
        for clip_id in newly_complete:
            row, path, fingerprint = lookup[clip_id]
            times = np.array([f["seconds"] for f in row["frames"]])
            smoothed = {label: smooth(outputs[clip_id][label], times) for label in LABELS}
            points = []
            for i, frame in enumerate(row["frames"]):
                point = {k: frame[k] for k in ("source_frame", "source_seconds", "seconds")}
                for label in LABELS:
                    point[region + "__" + label] = outputs[clip_id][label][i]
                    point[region + "__" + label + "_smooth"] = smoothed[label][i]
                if not all(np.isfinite(v) for v in point.values()):
                    raise ValueError("Nonfinite curve point")
                points.append(point)
            save_json(path, dict(id=clip_id, region=region, fingerprint=fingerprint, points=points))
            del outputs[clip_id]
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(f"Curves {person}/{region}: {completed}/{len(pending)} clips", flush=True)
                save_json(CURVES / person / (region + "_progress.json"), dict(completed_clips=completed, pending_at_start=len(pending), total_person_clips=len(rows)))
        tensors.clear()
        positions.clear()

    for row, path, fingerprint in pending:
        for i, frame in enumerate(row["frames"]):
            tensor, _ = native_image_tensor(frame["face_path"], mask)
            tensors.append(tensor)
            positions.append((row["id"], i))
            if len(tensors) >= feature_batch:
                flush()
    flush()
    if completed != len(pending):
        raise ValueError("Some regional clips were not completed")
    del extractor, heads, models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def combine_curves(rows):
    combined = 0
    for row in rows:
        directory = CURVES / row["person"] / row["label"] / row["id"]
        parts = [directory / ("region_" + region + ".json") for region in REGIONS]
        if not all(path.exists() for path in parts):
            continue
        records = [read_json(path) for path in parts]
        points = [{k: frame[k] for k in ("source_frame", "source_seconds", "seconds")} for frame in row["frames"]]
        source_identity_hashes = {object_sha(value["fingerprint"]["source"]) for value in records}
        if len(source_identity_hashes) != 1:
            raise ValueError(f"Regional frame identities disagree: {row['id']}")
        models = {}
        for region, record in zip(REGIONS, records):
            if record["region"] != region or len(record["points"]) != len(points):
                raise ValueError("Regional curve length mismatch")
            for target, source in zip(points, record["points"]):
                if any(source[key] != target[key] for key in ("source_frame", "source_seconds", "seconds")):
                    raise ValueError("Regional curves have different frame coordinates")
                target.update(source)
            models.update({region + "__" + label: record["fingerprint"]["models"][label] for label in LABELS})
        fingerprint = {region: object_sha(record["fingerprint"]) for region, record in zip(REGIONS, records)}
        expected_keys = {region + "__" + label + suffix for region in REGIONS for label in LABELS for suffix in ("", "_smooth")}
        if any(set(point) != expected_keys | {"source_frame", "source_seconds", "seconds"} for point in points):
            raise ValueError("Combined curve schema missing a model/region")
        payload = {k: row[k] for k in ("id", "person", "label", "onset_frame", "peak_frame", "onset_seconds", "peak_seconds", "interval_seconds")}
        payload.update(schema_version=1, frame_count=len(points), points=points, models=models,
                       fingerprint=fingerprint, regions=list(REGIONS), model_types=list(LABELS),
                       original_gallery_manifest=str(GALLERY / "manifest.json"))
        save_json(directory / "data.json", payload)
        write_csv(directory / "curve.csv", points)
        # JSONP-style local script asset supports file:// viewers without fetch.
        script = 'window.showRegionalClipData(' + json.dumps(row["id"]) + ',' + json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c') + ');'
        temporary = directory / "data.js.tmp"
        temporary.write_text(script, encoding="utf-8")
        temporary.replace(directory / "data.js")
        combined += 1
    cards = []
    for path in sorted(CURVES.glob("*/*/*/data.json")):
        data = read_json(path)
        cards.append(dict(id=data["id"], person=data["person"], label=data["label"], frame_count=data["frame_count"],
                          data_json=str(path), data_js=str(path.with_suffix(".js")), curve_csv=str(path.parent / "curve.csv")))
    save_json(CURVES / "manifest.json", cards)
    save_json(CURVES / "summary.json", dict(completed_clips=len(cards), expected_clips=1215,
              curve_count=len(cards)*18, scored_tracked_frames=sum(x["frame_count"] for x in cards),
              regions=list(REGIONS), model_types=list(LABELS), by_person=dict(Counter(x["person"] for x in cards))))
    print(f"Combined {combined} clips in current selection; {len(cards)}/1215 total", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", action="store_true")
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--person", nargs="+", choices=PERSONS, default=list(PERSONS))
    parser.add_argument("--label", nargs="+", choices=LABELS, default=list(LABELS), help="Scale building only; curves always use all three heads")
    parser.add_argument("--region", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-batch", type=int, default=32)
    args = parser.parse_args()
    if not (args.scales or args.curves or args.combine_only):
        parser.error("Select --scales, --curves and/or --combine-only")
    torch.set_num_threads(4)
    if args.scales:
        for person in args.person:
            for label in args.label:
                for region in args.region:
                    build_scale(person, label, region, args.device)
                    scale_summary()
    if args.curves or args.combine_only:
        manifest_path = GALLERY / "manifest.json"
        rows = read_json(manifest_path)
        if len(rows) != 1215 or any(row["person"] not in PERSONS for row in rows):
            raise ValueError("Unexpected reviewed gallery scope")
        selected = [row for row in rows if row["person"] in args.person]
        if args.curves:
            gallery_hash = sha(manifest_path)
            for person in args.person:
                subset = [row for row in selected if row["person"] == person]
                identities = {row["id"]: source_identity(row, gallery_hash) for row in subset}
                for region in args.region:
                    score_region(person, region, subset, identities, args.device, args.feature_batch)
                combine_curves(subset)
        else:
            combine_curves(selected)


if __name__ == "__main__":
    main()
