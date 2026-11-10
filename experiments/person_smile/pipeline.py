"""Two-stage, per-person polite/bitter experiment. No background replacement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from analysis_contract import ANALYSIS_DIR, ANALYSIS_VERSION, SUMMARY_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("polite", "bitter")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_number(path):
    return int(re.search(r"(\d+)$", Path(path).stem).group(1))


def uniform_indices(n, count):
    count = min(n, count)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]


def run(command, log, cwd=ROOT):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    print("Running:", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    with Path(log).open("w", encoding="utf-8") as f:
        subprocess.run([str(x) for x in command], cwd=cwd, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def split_clips(clips, seed):
    if len(clips) < 6:
        raise ValueError("Each person/type needs >=6 eligible clips: >=2 per train/val/test split.")
    clips = sorted(clips)
    random.Random(seed).shuffle(clips)
    n_test = max(2, round(len(clips) * .2))
    n_val = max(2, round(len(clips) * .1))
    return {c: ("test" if i < n_test else "val" if i < n_test + n_val else "train")
            for i, c in enumerate(clips)}


def assign_sequence_splits(manifest, cfg):
    labels = tuple(cfg.get("labels", LABELS))
    pattern = cfg.get("source_group_suffix_regex")
    if pattern:
        groups = {}
        for row in manifest:
            group = re.sub(pattern, "", Path(row["source_video"]).stem)
            row["source_group"] = group
            groups.setdefault(group, []).append(row)
        assignments = split_clips(list(groups), cfg["seed"])
        for group, rows in groups.items():
            for row in rows:
                row["split"] = assignments[group]
        for label in labels:
            for split in ("train", "val", "test"):
                if sum(r["label"] == label and r["split"] == split for r in manifest) < 2:
                    raise ValueError(f"Too few {label}/{split} sequences after source grouping.")
    else:
        for label in labels:
            assignments = split_clips([r["sequence_id"] for r in manifest if r["label"] == label], cfg["seed"])
            for row in manifest:
                if row["label"] == label:
                    row["split"] = assignments[row["sequence_id"]]


def create_annotation(path, pairs):
    template = Path(__file__).with_name("annotate.html").read_text(encoding="utf-8")
    payload = json.dumps(pairs, ensure_ascii=True).replace("<", "\\u003c")
    path.write_text(template.replace("/*PAIR_DATA*/", payload), encoding="utf-8")


def track_clip(cfg, clip, clip_id, out):
    tracking = out / "tracking" / clip_id
    if (tracking / "complete.json").exists():
        return
    if tracking.exists() and any(tracking.iterdir()):
        raise RuntimeError(f"Incomplete tracking output: {tracking}. Inspect logs before resuming.")
    run([cfg["tracking_python"], cfg["tracking_script"], "-m", clip,
         "-o", tracking, "-n", clip_id, "-r", "224",
         "--thr-orientation", str(cfg["orientation_threshold"])],
        out / "logs" / f"tracking_{clip_id}.log", Path(cfg["tracking_script"]).parent)
    save_json(tracking / "complete.json", {"source_sha256": digest(clip)})


def prepare_person(cfg, person, out):
    """Tracks are never combined silently: multi-track clips need an explicit choice."""
    from PIL import Image
    labels = tuple(cfg.get("labels", LABELS))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.json").exists():
        old = json.loads((out / "prepared.json").read_text(encoding="utf-8"))
        if any(not Path(r["path"]).is_file() or digest(r["path"]) != r["sha256"]
               for r in old["source_manifest"]):
            raise ValueError("Prepared source videos changed; use a fresh output_root.")
        print(f"{person.name}: already prepared; preserving annotation dataset.")
        return
    selection_path = out / "track_selection.json"
    selections = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    exclusions_path = out / "exclude_clips.json"
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8")) if exclusions_path.exists() else {}
    if cfg.get("tracking_workers", 1) > 1:
        jobs = []
        for label in labels:
            source = person / cfg.get("label_folders", {}).get(label, label)
            for clip in sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_EXTS):
                clip_id = f"{label}_{digest(clip)[:16]}"
                if clip_id not in exclusions:
                    jobs.append((clip, clip_id))
        if len({j[1] for j in jobs}) != len(jobs):
            raise ValueError("Duplicate source clips.")
        with ThreadPoolExecutor(max_workers=cfg["tracking_workers"]) as pool:
            futures = [pool.submit(track_clip, cfg, clip, clip_id, out) for clip, clip_id in jobs]
            for future in futures:
                future.result()
    manifest, issues, source_records = [], [], []
    for label in labels:
        source = person / cfg.get("label_folders", {}).get(label, label)
        clips = sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not clips:
            raise ValueError(f"No video clips in {source}")
        for clip in clips:
            clip_hash = digest(clip)
            clip_id = f"{label}_{clip_hash[:16]}"
            if any(r["clip_id"] == clip_id for r in source_records):
                raise ValueError(f"Duplicate source clip: {clip}")
            source_records.append({"clip_id": clip_id, "path": str(clip), "sha256": clip_hash})
            if clip_id in exclusions:
                issues.append({"clip_id": clip_id, "reason": "manual_exclusion",
                               "detail": exclusions[clip_id]})
                continue
            tracking = out / "tracking" / clip_id
            track_clip(cfg, clip, clip_id, out)
            detected = tracking / "DetectedFaces" / clip_id
            tracks = sorted(p for p in detected.iterdir() if p.is_dir() and (p / "0").is_dir())
            tracks = [p for p in tracks if list((p / "0").glob("*.png"))]
            choice = selections.get(clip_id)
            if choice is not None:
                tracks = [p for p in tracks if p.name == str(choice)]
            if len(tracks) != 1:
                issues.append({"clip_id": clip_id, "reason": "choose_one_track",
                               "tracks": [p.name for p in tracks]})
                continue
            frames = sorted((tracks[0] / "0").glob("*.png"), key=frame_number)
            if len(frames) < cfg["min_frames"]:
                issues.append({"clip_id": clip_id, "reason": "too_few_front_frames", "count": len(frames)})
                continue
            sequence_dir = out / "sequences" / label / clip_id
            sequence_dir.mkdir(parents=True, exist_ok=True)
            for frame in frames:
                with Image.open(frame) as image:
                    image.verify()
                with Image.open(frame) as image:
                    if image.size != (224, 224):
                        raise ValueError(f"Unexpected crop size: {frame}")
                shutil.copy2(frame, sequence_dir / frame.name)
            sys.path.insert(0, str(ROOT / "annotation"))
            from build_sequence_stillimages import build_stillimage
            build_stillimage(frames, out / "review" / f"{clip_id}.png")
            manifest.append({"label": label, "sequence_id": clip_id, "source_video": str(clip),
                             "sequence_dir": str(sequence_dir), "frame_count": len(frames),
                             "track_id": tracks[0].name, "filled_count": 0})
    save_json(out / "preparation_qc.json", {"issues": issues, "sources": source_records})
    if any(i["reason"] == "choose_one_track" for i in issues):
        raise RuntimeError(f"Resolve ambiguous/missing tracks in {out / 'preparation_qc.json'}; "
                           "write clip_id -> track ID in track_selection.json, then rerun.")
    assign_sequence_splits(manifest, cfg)
    # Exact decoded-image duplicates across splits would invalidate independence.
    hashes = {}
    for row in manifest:
        for path in Path(row["sequence_dir"]).glob("*.png"):
            with Image.open(path) as image:
                h = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if h in hashes and hashes[h] != row["split"]:
                raise ValueError(f"Exact image duplicate across splits at {path}; resolve duplicate clips.")
            hashes[h] = row["split"]
    write_csv(out / "sequences.csv", manifest)
    for label in labels:
        data = out / "ranking" / label
        (data / "images").mkdir(parents=True, exist_ok=True)
        items = []
        for row in manifest:
            if row["label"] != label:
                continue
            frames = sorted(Path(row["sequence_dir"]).glob("*.png"), key=frame_number)
            for ordinal, idx in enumerate(uniform_indices(len(frames), cfg["candidates_per_sequence"])):
                name = f'{row["sequence_id"]}_{frame_number(frames[idx]):08d}.png'
                shutil.copy2(frames[idx], data / "images" / name)
                items.append({"image_file": name, "sequence_id": row["sequence_id"],
                              "provisional_level": ordinal, "source_frame_index": frame_number(frames[idx]),
                              "source_frame_name": frames[idx].name, "source_frame_path": str(frames[idx]),
                              "split": row["split"]})
        write_csv(data / "ranking_items.csv", items)
        write_csv(data / "scale_candidates.csv", [r for r in items if r["split"] == "train"])
        pairs = []
        rng = random.Random(cfg["seed"])
        for split in ("train", "val", "test"):
            subset = [r for r in items if r["split"] == split]
            candidates = [(a["image_file"], b["image_file"]) for a, b in itertools.combinations(subset, 2)
                          if a["sequence_id"] != b["sequence_id"]]
            rng.shuffle(candidates)
            for a, b in candidates[:cfg["pairs_per_split"][split]]:
                if rng.random() < .5:
                    a, b = b, a
                pairs.append({"pair_id": f"{label}_{len(pairs):05d}", "image1": a,
                              "image2": b, "split": split, "label": ""})
        write_csv(data / "pairs.csv", pairs)
        create_annotation(data / "annotate.html", pairs)
        save_json(data / "task_summary.json", {
            "candidate_images": len(items),
            "pair_counts": {s: sum(p["split"] == s for p in pairs) for s in ("train", "val", "test")},
            "sampling": "uniform temporal candidates; random cross-sequence pairs; no automatic labels",
            "provisional_level": "temporal position only, not smile strength",
        })
    save_json(out / "prepared.json", {"status": "awaiting_manual_annotation", "config": cfg,
                                     "source_manifest": source_records})


def import_annotations(data):
    expected = read_csv(data / "pairs.csv")
    actual = read_csv(data / "annotations.csv")
    by_id = {r["pair_id"]: r for r in actual}
    if len(by_id) != len(actual) or set(by_id) != {r["pair_id"] for r in expected}:
        raise ValueError(f"{data}: missing, duplicate or unknown pair IDs.")
    splits = {s: [] for s in ("train", "val", "test")}
    counts = {"tie": 0, "discard": 0}
    for row in expected:
        supplied = by_id[row["pair_id"]]
        if any(supplied[k] != row[k] for k in ("image1", "image2")):
            raise ValueError(f"Changed pair images: {row['pair_id']}")
        label = supplied["label"]
        if label not in {"1", "-1", "0", "discard"}:
            raise ValueError(f"Unfinished/invalid annotation: {row['pair_id']}")
        if label in {"0", "discard"}:
            counts["tie" if label == "0" else "discard"] += 1
        else:
            splits[row["split"]].append(f'{row["image1"]},{row["image2"]},{label}')
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"{data}: no decisive labels in {split}.")
        (data / f"{split}.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data / "all.dat").write_text("\n".join(sum(splits.values(), [])) + "\n", encoding="utf-8")
    save_json(data / "annotation_audit.json", {**counts, **{k: len(v) for k, v in splits.items()},
                                              "annotation_sha256": digest(data / "annotations.csv")})


def train_analyze(cfg, out):
    if not (out / "prepared.json").exists():
        raise ValueError(f"Run prepare first: {out}")
    # Validate BOTH annotation sets before starting either model.
    for label in LABELS:
        import_annotations(out / "ranking" / label)
    for label in LABELS:
        data = out / "ranking" / label
        model_dir = data / "model"
        fingerprint = {"annotations": digest(data / "annotations.csv"), "config": cfg,
                       "initial_weights": digest(cfg["initial_weights"])}
        done = model_dir / "complete.json"
        if done.exists():
            if json.loads(done.read_text()) != fingerprint:
                raise ValueError("Training inputs changed; use a new experiment output root.")
            continue
        run([sys.executable, ROOT / "annotation/train_smile_rank_with_holdout.py",
             "--smilecomp_dir", cfg["smilecomp_dir"], "--weights", cfg["initial_weights"],
             "--dataset_file", data / "all.dat", "--dataset_dir", data / "images",
             "--train_file", data / "train.dat", "--val_file", data / "val.dat",
             "--holdout_file", data / "test.dat", "--use_existing_splits",
             "--dst_weights", model_dir / "model.pth", "--dst_history", model_dir / "history.png",
             "--dst_metrics", model_dir / "metrics.json", "--epochs", cfg["epochs"],
             "--batch_size", cfg["batch_size"], "--learning_rate", cfg["learning_rate"],
             "--seed", cfg["seed"]], out / "logs" / f"train_{label}.log")
        run([sys.executable, ROOT / "annotation/build_true_smile_scale_from_model.py",
             "--smilecomp_dir", cfg["smilecomp_dir"], "--ranking_items", data / "scale_candidates.csv",
             "--images_dir", data / "images", "--weights", model_dir / "model.pth",
             "--manual_pairs", data / "train.dat", "--output_dir", data / "scale10"],
            out / "logs" / f"scale_{label}.log")
        save_json(done, fingerprint)
    run([sys.executable, Path(__file__).with_name("analyze.py"), "--person-root", out,
         "--config", out.parent / "experiment_config.json"], out / "logs" / "analysis.log")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "prepare", "run"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--person", help="Optional person folder name; otherwise all persons.")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    labels = tuple(cfg.get("labels", LABELS))
    if not labels or len(set(labels)) != len(labels) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", x) for x in labels):
        raise ValueError("labels must contain distinct simple lowercase category names")
    if args.stage == "run" and labels != LABELS:
        raise ValueError("The run stage supports polite/bitter analysis only; custom categories are prepare-only.")
    source, output = Path(cfg["input_root"]).resolve(), Path(cfg["output_root"]).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output roots must be separate, non-nested directories.")
    for key in ("tracking_python", "tracking_script", "initial_weights", "fc7_weights"):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(f"{key}: {cfg[key]}")
    if args.stage == "check":
        print(json.dumps({"tools_exist": True, "input_exists": source.is_dir(),
                          "new_data_required": not source.is_dir()}, indent=2))
        return
    people = sorted(p for p in source.iterdir() if p.is_dir() and (not args.person or p.name == args.person))
    if not people:
        raise ValueError("No matching person folders.")
    for person in people:
        for label in labels:
            label_dir = person / cfg.get("label_folders", {}).get(label, label)
            if not label_dir.is_dir():
                raise FileNotFoundError(label_dir)
    frozen = output / "experiment_config.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != cfg:
        raise ValueError("Configuration changed for existing output. Choose a fresh output_root.")
    save_json(frozen, cfg)
    for person in people:
        if args.stage == "prepare":
            prepare_person(cfg, person, output / person.name)
        else:
            train_analyze(cfg, output / person.name)
    if args.stage == "prepare":
        print("Prepared. STOP: complete each ranking/*/annotate.html and save annotations.csv beside it.")
    else:
        reports = [{"person": p.name, "report": str(output / p.name / ANALYSIS_DIR / "report.md")}
                   for p in people]
        save_json(output / "reports.json", reports)
        summaries = [json.loads(p.read_text(encoding="utf-8"))
                     for p in sorted(output.glob(f"*/{ANALYSIS_DIR}/summary.json"))]
        summaries = [s for s in summaries if s.get("analysis_version") == ANALYSIS_VERSION]
        fields = tuple(SUMMARY_COLUMNS)
        write_csv(output / "person_comparison.csv",
                  [{"person": s["person"], **{k: s[k] for k in fields}} for s in summaries])
        lines = ["# 跨人物复现汇总", "",
                 "每人一行、两套主模型及八套区域辅助模型；不把帧或序列对当作独立人物。",
                 "仅汇总四项分析。区域差异是遮罩重训系统效应，不是直接权重；不自动判断显著性。",
                 "第1项差值为正表示类内方向更一致；第2项两差值为正分别表示 polite 脸颊占比最高、bitter 更均衡。",
                 "第3项正值表示下半脸更能保持排序；第4项直接报告余弦，不人为设定复现成功阈值。", "",
                 "| 人物 | " + " | ".join(SUMMARY_COLUMNS.values()) + " |",
                 "|---|" + "---:|" * len(fields)]
        for s in summaries:
            lines.append("| " + " | ".join([s["person"]] + [str(s[k]) for k in fields]) + " |")
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
