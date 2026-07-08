from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from itertools import combinations, product
from pathlib import Path


DEFAULT_INPUT = Path(r"E:\Dataset\smile_ranking_true\ranking_items.csv")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Dataset\smile_ranking_true")
SEED = 20260707


def numeric_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 0, value


def load_items(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def item_by_sequence_level(items: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (str(item["sequence_id"]), int(item["provisional_level"])): item
        for item in items
    }


def make_candidates(items: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    by_key = item_by_sequence_level(items)
    sequence_ids = sorted({str(item["sequence_id"]) for item in items}, key=numeric_key)
    levels = sorted({int(item["provisional_level"]) for item in items})
    candidates: dict[str, list[dict[str, object]]] = {
        "same_level": [],
        "adjacent_level": [],
        "gap2_level": [],
    }

    for level in levels:
        for seq1, seq2 in combinations(sequence_ids, 2):
            item1 = by_key[(seq1, level)]
            item2 = by_key[(seq2, level)]
            candidates["same_level"].append(
                {
                    "image1": item1["image_file"],
                    "image2": item2["image_file"],
                    "initial_label": "unknown",
                    "category": "same_level",
                    "sequence1_id": seq1,
                    "sequence2_id": seq2,
                    "level1": level,
                    "level2": level,
                    "expected_relation": "same_provisional_level",
                }
            )

    for gap, category in ((1, "adjacent_level"), (2, "gap2_level")):
        for level1 in levels:
            level2 = level1 + gap
            if level2 not in levels:
                continue
            for seq1, seq2 in product(sequence_ids, sequence_ids):
                if seq1 == seq2:
                    continue
                weaker = by_key[(seq1, level1)]
                stronger = by_key[(seq2, level2)]
                candidates[category].append(
                    {
                        "image1": weaker["image_file"],
                        "image2": stronger["image_file"],
                        "initial_label": "-1",
                        "category": category,
                        "sequence1_id": seq1,
                        "sequence2_id": seq2,
                        "level1": level1,
                        "level2": level2,
                        "expected_relation": "image2_stronger_by_level_prior",
                    }
                )
    return candidates


def balanced_targets() -> dict[str, dict[int, int]]:
    same = {level: 6 for level in range(10)}
    adjacent = {level: 9 for level in range(8)}
    adjacent[8] = 8
    gap2 = {level: 8 if level < 4 else 7 for level in range(8)}
    return {
        "same_level": same,
        "adjacent_level": adjacent,
        "gap2_level": gap2,
    }


def select_low_reuse(
    candidates: list[dict[str, object]],
    count: int,
    degrees: Counter[str],
    used_pairs: set[tuple[str, str]],
    rng: random.Random,
) -> list[dict[str, object]]:
    remaining = candidates[:]
    rng.shuffle(remaining)
    selected: list[dict[str, object]] = []
    while len(selected) < count:
        ranked = sorted(
            (
                row
                for row in remaining
                if tuple(sorted((str(row["image1"]), str(row["image2"])))) not in used_pairs
            ),
            key=lambda row: (
                degrees[str(row["image1"])] + degrees[str(row["image2"])],
                max(degrees[str(row["image1"])], degrees[str(row["image2"])]),
                str(row["image1"]),
                str(row["image2"]),
            ),
        )
        if not ranked:
            break
        row = ranked[0]
        selected.append(row)
        remaining.remove(row)
        pair_key = tuple(sorted((str(row["image1"]), str(row["image2"]))))
        used_pairs.add(pair_key)
        degrees[str(row["image1"])] += 1
        degrees[str(row["image2"])] += 1
    if len(selected) != count:
        raise RuntimeError(f"Could only select {len(selected)} pairs out of requested {count}.")
    return selected


def orient_known_pairs(rows: list[dict[str, object]], rng: random.Random) -> None:
    known_rows = [row for row in rows if row["initial_label"] != "unknown"]
    target_flip = round(len(known_rows) * 0.35)
    for row in rng.sample(known_rows, target_flip):
        row["image1"], row["image2"] = row["image2"], row["image1"]
        row["sequence1_id"], row["sequence2_id"] = row["sequence2_id"], row["sequence1_id"]
        row["level1"], row["level2"] = row["level2"], row["level1"]
        row["initial_label"] = "1"
        row["expected_relation"] = "image1_stronger_by_level_prior"


def build_sample(items: list[dict[str, str]]) -> list[dict[str, object]]:
    rng = random.Random(SEED)
    candidates = make_candidates(items)
    targets = balanced_targets()
    degrees: Counter[str] = Counter()
    used_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []

    for category in ("same_level", "adjacent_level", "gap2_level"):
        for level, count in targets[category].items():
            level_candidates = [
                row for row in candidates[category] if int(row["level1"]) == level
            ]
            rows.extend(select_low_reuse(level_candidates, count, degrees, used_pairs, rng))

    orient_known_pairs(rows, rng)
    return rows


def write_outputs(output_root: Path, rows: list[dict[str, object]]) -> None:
    dat_path = output_root / "manual_review_cross_sequence_sample200.dat"
    with dat_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row["image1"], row["image2"], row["initial_label"]])

    meta_path = output_root / "manual_review_cross_sequence_sample200_meta.csv"
    fields = [
        "image1",
        "image2",
        "initial_label",
        "category",
        "sequence1_id",
        "sequence2_id",
        "level1",
        "level2",
        "expected_relation",
    ]
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    labels = Counter(str(row["initial_label"]) for row in rows)
    categories = Counter(str(row["category"]) for row in rows)
    degrees: Counter[str] = Counter()
    for row in rows:
        degrees[str(row["image1"])] += 1
        degrees[str(row["image2"])] += 1
    print(f"[CROSS] pairs: {len(rows)}")
    print(f"[CROSS] labels: {dict(labels)}")
    print(f"[CROSS] categories: {dict(categories)}")
    print(f"[CROSS] unique_images: {len(degrees)}")
    print(f"[CROSS] max_image_reuse: {max(degrees.values()) if degrees else 0}")
    print(f"[CROSS] degree_hist: {dict(sorted(Counter(degrees.values()).items()))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample-like cross-sequence review pair file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_sample(load_items(args.input))
    write_outputs(args.output_root, rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
