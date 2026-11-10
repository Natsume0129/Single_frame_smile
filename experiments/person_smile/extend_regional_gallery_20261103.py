"""Validate all regional curves, then explicitly publish an extended gallery.

Default invocation is read-only readiness checking. --apply is allowed only when
all 1,215 regional payloads align with the original reviewed frames. A frozen
backup precedes any edit to the original gallery/template. The original three
curve files, videos, stills, annotations and models are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
GALLERY = ROOT / "output/three_model_clip_gallery_20261102"
REGIONAL = ROOT / "output/regional_ranking_extension_20261103"
TEMPLATE = Path(__file__).with_name("three_model_clip_gallery.html")
CONTROLS = Path(__file__).with_name("regional_gallery_controls.js")
REGIONS = ("full", "periocular", "nose", "mouth_perioral", "upper_face", "lower_face")
LABELS = ("polite", "bitter", "true")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf8")
    temporary.replace(path)


def parse_callback(path, name):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    prefix = "window." + name + "("
    if not text.startswith(prefix) or not text.endswith(");"):
        raise ValueError(f"Unexpected callback wrapper: {path}")
    values = json.loads("[" + text[len(prefix):-2] + "]")
    if len(values) != 2:
        raise ValueError(f"Unexpected callback arguments: {path}")
    return values


def validate(gallery=GALLERY, regional=REGIONAL):
    cards = read(gallery / "gallery_manifest.json")
    originals = {row["id"]: row for row in read(gallery / "manifest.json")}
    if len(cards) != 1215 or len(originals) != 1215 or len({r["id"] for r in cards}) != 1215:
        raise ValueError("Expected the frozen 1,215-clip gallery")
    missing = []
    verified_frames = 0
    updated = []
    for row in cards:
        folder = regional / "curves" / row["person"] / row["label"] / row["id"]
        js, csv = folder / "data.js", folder / "curve.csv"
        if not js.is_file() or not csv.is_file():
            missing.append(row["id"])
            continue
        callback_id, payload = parse_callback(js, "showRegionalClipData")
        if callback_id != row["id"] or payload.get("id", callback_id) != row["id"]:
            raise ValueError(f"Curve identity mismatch: {js}")
        if payload.get("person", row["person"]) != row["person"] or payload.get("label", row["label"]) != row["label"]:
            raise ValueError(f"Curve model/clip identity mismatch: {js}")
        original = originals[row["id"]]
        points, expected = payload["points"], original["frames"]
        if len(points) != len(expected) or len(points) != row["tracked_frames"]:
            raise ValueError(f"Curve frame count mismatch: {js}")
        for point, frame in zip(points, expected):
            if point["source_frame"] != frame["source_frame"] or abs(point["seconds"] - frame["seconds"]) > 1e-6:
                raise ValueError(f"Curve time/frame mismatch: {js}")
            for region in REGIONS:
                for label in LABELS:
                    for suffix in ("", "_smooth"):
                        value = point.get(region + "__" + label + suffix)
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or not -1e-6 <= value <= 9 + 1e-6:
                            raise ValueError(f"Missing/nonfinite regional score: {js}")
        verified_frames += len(points)
        updated.append({**row, "regional_curve_js": os.path.relpath(js, gallery).replace("\\", "/"),
                        "regional_curve_csv": os.path.relpath(csv, gallery).replace("\\", "/")})
    return updated, dict(expected_clips=len(cards), ready_clips=len(updated), missing_clips=len(missing),
                         missing_ids=missing, verified_tracked_frames=verified_frames, regional_curves_per_clip=18,
                         original_curves_per_clip=3)


def extended_template(base):
    if "regional_gallery_controls.js" in base:
        raise ValueError("Use the frozen original template rather than injecting twice")
    if "__DATA__" not in base or "</html>" not in base:
        raise ValueError("Original gallery template markers missing")
    base = base.replace("Onset-to-Peak Clips — Three Personal Smile Models",
                        "Onset-to-Peak Clips — Full-Face and Regional Smile Models")
    # Avoid an obsolete original curve request showing an error on a newer clip.
    old = "script.onerror=()=>{$('message').textContent='Could not load curve data. Keep the gallery folder structure intact.';script.remove();};"
    new = "script.onerror=()=>{if(current===r.id)$('message').textContent='Could not load curve data. Keep the gallery folder structure intact.';script.remove();};"
    if old not in base:
        raise ValueError("Original lazy-loader changed; review before patching")
    base = base.replace(old, new)
    return base.replace("</html>", '<script src="regional_gallery_controls.js"></script></html>')


def apply(gallery=GALLERY, regional=REGIONAL):
    cards, report = validate(gallery, regional)
    if report["missing_clips"]:
        raise ValueError(f"Not publishing: {report['missing_clips']} regional clips are not ready")
    backup = regional / "gallery_backup"
    backup.mkdir(parents=True, exist_ok=True)
    originals = {"index.html": gallery / "index.html", "gallery_manifest.json": gallery / "gallery_manifest.json",
                 "three_model_clip_gallery.html": TEMPLATE}
    manifest_path = backup / "backup_manifest.json"
    if manifest_path.exists():
        saved = read(manifest_path)
        for name, record in saved.items():
            if sha(backup / name) != record["sha256"]:
                raise ValueError(f"Frozen backup changed: {name}")
    else:
        saved = {}
        for name, path in originals.items():
            if (backup / name).exists():
                raise ValueError(f"Incomplete prior backup requires inspection: {backup}")
            shutil.copy2(path, backup / name)
            saved[name] = dict(source=str(path), sha256=sha(path))
        write_atomic(manifest_path, json.dumps(saved, indent=2))
    template = extended_template((backup / "three_model_clip_gallery.html").read_text(encoding="utf8"))
    encoded = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__DATA__", encoded)
    write_atomic(gallery / "regional_gallery_controls.js", CONTROLS.read_text(encoding="utf8"))
    write_atomic(gallery / "gallery_manifest.json", json.dumps(cards, indent=2, ensure_ascii=False))
    write_atomic(TEMPLATE, template)
    write_atomic(gallery / "index.html", page)
    # A direct new entry point uses the original directory as its base, preserving
    # every existing relative video/still/curve URL without copying the media.
    new_page = page.replace('<meta charset="utf-8">', '<meta charset="utf-8"><base href="' + gallery.as_uri() + '/">', 1)
    write_atomic(regional / "gallery.html", new_page)
    report.update(gallery=str(gallery / "index.html"), standalone=str(regional / "gallery.html"),
                  backup=str(backup), helper_sha256=sha(CONTROLS), applied=True)
    write_atomic(regional / "gallery_extension.json", json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Publish only after every regional curve passes validation")
    args = parser.parse_args()
    if args.apply:
        report = apply()
    else:
        _, report = validate()
        report["missing_ids"] = report["missing_ids"][:10]
        report["applied"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
