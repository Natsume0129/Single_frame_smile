from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}
SEARCH_PERCENTS = np.arange(0.0, 101.0, 1.0, dtype=np.float64)


@dataclass
class CompetitionFrameConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    nearest_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_competition_frames")
    relative_threshold: float = 0.10
    nonlocal_gap_percent: float = 10.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CompetitionFrameConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            nearest_output_root=Path(args.nearest_output_root),
            output_root=Path(args.output_root),
            relative_threshold=float(args.relative_threshold),
            nonlocal_gap_percent=float(args.nonlocal_gap_percent),
        )


class CompetitionFrameReport:
    def __init__(self, cfg: CompetitionFrameConfig):
        self.cfg = cfg
        self.asset_root = cfg.output_root / "assets" / "frames"
        self.asset_root.mkdir(parents=True, exist_ok=True)
        self._baseline_cache: dict[tuple[str, str], np.ndarray] = {}
        self._target_cache: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def point_at_percent(curve: np.ndarray, percent: float) -> np.ndarray:
        pos = (percent / 100.0) * (curve.shape[0] - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            return curve[lo]
        alpha = pos - lo
        return (1.0 - alpha) * curve[lo] + alpha * curve[hi]

    @staticmethod
    def frame_index(percent: float, n_frames: int = 20) -> int:
        idx = int((percent / 100.0) * (n_frames - 1) + 0.5)
        return max(0, min(n_frames - 1, idx))

    def prototype_path(self, method: str, class_name: str) -> Path:
        return (
            self.cfg.projection_output_root
            / method
            / "prototypes"
            / f"prototype_{class_name}_{method}.npy"
        )

    def normalized_sequence_path(self, class_name: str, sequence_id: str) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "normalized"
            / class_name
            / str(sequence_id)
            / "normalized_sequence.npy"
        )

    def normalized_frames_dir(self, class_name: str, sequence_id: str) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "normalized_frames"
            / class_name
            / str(sequence_id)
        )

    def load_baseline_samples(self, method: str, baseline_class: str) -> np.ndarray:
        key = (method, baseline_class)
        if key not in self._baseline_cache:
            curve = np.load(self.prototype_path(method, baseline_class), allow_pickle=False).astype(np.float64)
            self._baseline_cache[key] = np.vstack(
                [self.point_at_percent(curve, p) for p in SEARCH_PERCENTS]
            )
        return self._baseline_cache[key]

    def load_target_curve(self, target_class: str, sequence_id: str) -> np.ndarray:
        key = (target_class, sequence_id)
        if key not in self._target_cache:
            self._target_cache[key] = np.load(
                self.normalized_sequence_path(target_class, sequence_id),
                allow_pickle=False,
            ).astype(np.float64)
        return self._target_cache[key]

    def load_methodb_meta(self) -> dict:
        return json.loads(
            (
                self.cfg.projection_output_root
                / "methodB"
                / "prototypes"
                / "projection_meta_methodB.json"
            ).read_text(encoding="utf-8")
        )

    def copy_frame(self, source: Path, subdir: str) -> str:
        if not source.is_file():
            raise FileNotFoundError(f"Missing frame: {source}")
        destination = self.asset_root / subdir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        return destination.relative_to(self.cfg.output_root).as_posix()

    def target_asset(self, target_class: str, sequence_id: str, stage_percent: float) -> tuple[int, str]:
        idx = self.frame_index(stage_percent)
        source = self.normalized_frames_dir(target_class, sequence_id) / f"{idx:03d}.png"
        asset = self.copy_frame(source, f"target/{target_class}_{sequence_id}")
        return idx, asset

    def baseline_asset(
        self,
        method: str,
        baseline_class: str,
        progress_percent: float,
        methodb_meta: dict,
    ) -> tuple[int | None, str | None, str | None]:
        if method != "methodB":
            return None, None, "methodA baseline is a median prototype and has no real frame."
        baseline_seq = str(methodb_meta[baseline_class]["sequence_id"])
        idx = self.frame_index(progress_percent)
        source = self.normalized_frames_dir(baseline_class, baseline_seq) / f"{idx:03d}.png"
        asset = self.copy_frame(source, f"baseline/{method}_{baseline_class}_{baseline_seq}")
        return idx, asset, None

    def second_nonlocal_index(self, distances: np.ndarray, best_idx: int) -> int:
        for idx in np.argsort(distances):
            idx = int(idx)
            if abs(float(idx) - float(best_idx)) > self.cfg.nonlocal_gap_percent:
                return idx
        raise RuntimeError("Could not find a nonlocal second candidate.")

    def build_cases(self) -> tuple[list[dict], dict]:
        rows = self.read_csv(
            self.cfg.nearest_output_root / "csv" / "nearest6_nearest_baseline_curve_all.csv"
        )
        methodb_meta = self.load_methodb_meta()
        cases: list[dict] = []

        for row in rows:
            method = row["method"]
            baseline_class = row["baseline_class"]
            target_class = row["target_class"]
            sequence_id = row["sequence_id"]
            stage = float(row["target_stage_percent"])

            baseline_samples = self.load_baseline_samples(method, baseline_class)
            target_curve = self.load_target_curve(target_class, sequence_id)
            target_point = self.point_at_percent(target_curve, stage)
            distances = np.linalg.norm(baseline_samples - target_point, axis=1)
            best_idx = int(np.argmin(distances))
            second_idx = self.second_nonlocal_index(distances, best_idx)
            best_distance = float(distances[best_idx])
            second_distance = float(distances[second_idx])
            relative_gap = (
                (second_distance - best_distance) / best_distance
                if best_distance > 1e-12
                else float("inf")
            )

            if relative_gap > self.cfg.relative_threshold:
                continue

            target_idx, target_src = self.target_asset(target_class, sequence_id, stage)
            best_frame_idx, best_src, best_note = self.baseline_asset(
                method,
                baseline_class,
                float(best_idx),
                methodb_meta,
            )
            second_frame_idx, second_src, second_note = self.baseline_asset(
                method,
                baseline_class,
                float(second_idx),
                methodb_meta,
            )
            cases.append(
                {
                    "method": method,
                    "baseline_class": baseline_class,
                    "target_class": target_class,
                    "sequence_id": sequence_id,
                    "rank": row["rank"],
                    "target_stage_percent": stage,
                    "target_frame_index": target_idx,
                    "target_frame_src": target_src,
                    "best_progress": float(best_idx),
                    "best_distance": best_distance,
                    "best_frame_index": best_frame_idx,
                    "best_frame_src": best_src,
                    "best_note": best_note,
                    "second_progress": float(second_idx),
                    "second_distance": second_distance,
                    "second_frame_index": second_frame_idx,
                    "second_frame_src": second_src,
                    "second_note": second_note,
                    "relative_gap": float(relative_gap),
                    "absolute_gap": float(second_distance - best_distance),
                    "label": (
                        f"{method} | baseline={baseline_class} | {target_class} "
                        f"seq={sequence_id}, rank={row['rank']} | stage={stage:.0f}% | "
                        f"best={best_idx}% second={second_idx}% gap={relative_gap * 100:.1f}%"
                    ),
                }
            )

        summary = {
            "total_nearest6_points": len(rows),
            "competitive_cases": len(cases),
            "relative_threshold_percent": self.cfg.relative_threshold * 100.0,
            "nonlocal_gap_percent": self.cfg.nonlocal_gap_percent,
            "by_method": dict(Counter(c["method"] for c in cases)),
            "by_method_baseline": [
                {"method": k[0], "baseline_class": k[1], "count": v}
                for k, v in sorted(Counter((c["method"], c["baseline_class"]) for c in cases).items())
            ],
            "by_target_class": dict(Counter(c["target_class"] for c in cases)),
        }
        return cases, summary

    @staticmethod
    def json_data(data: object) -> str:
        return json.dumps(data, ensure_ascii=False, allow_nan=False)

    def render_html(self, cases: list[dict], summary: dict) -> str:
        payload = self.json_data(
            {
                "cases": cases,
                "summary": summary,
                "methods": METHODS,
                "baselineClasses": BASELINE_CLASSES,
                "classNames": CLASS_NAMES,
                "colors": COLORS,
            }
        )
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Nearest Candidate Competition Frames</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.48}}
main{{max-width:1320px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 8px}}
p{{max-width:1050px}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
.metric{{background:white;border:1px solid #ddd;padding:12px}}
.metric b{{font-size:22px;display:block}}
.controls{{position:sticky;top:0;z-index:10;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0;margin:10px 0 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
select,input{{font-size:13px}}
label{{font-size:13px;display:inline-flex;align-items:center;gap:6px}}
.note{{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}}
.count{{font-weight:700}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}}
.card{{background:white;border:1px solid #ddd;padding:10px}}
.card h3{{font-size:14px;margin:0 0 8px;line-height:1.35}}
.images{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}}
.imgbox{{border:1px solid #ddd;background:#f4f4f4;padding:6px}}
.imgbox h4{{font-size:12px;margin:0 0 5px}}
.imgbox img{{width:100%;height:180px;object-fit:contain;background:#eee;display:block}}
.placeholder{{height:180px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px;color:#555;background:#eee;padding:8px}}
.meta{{font-size:12px;color:#444;margin-top:6px}}
.hidden{{display:none}}
</style>
</head>
<body>
<main>
<h1>Nearest Candidate Competition Frames</h1>
<p>This report shows every nearest-baseline point where the best baseline candidate and the second nonlocal baseline candidate are close enough to compete.</p>
<div class="note">Competition definition used here: <b>(second_nonlocal_distance - best_distance) / best_distance <= 10%</b>. The second candidate must be more than 10% baseline-progress away from the best candidate, so adjacent 1% grid points from the same local valley are not counted as competition.</div>
<div class="note">methodB baselines have real medoid frames. methodA baselines are median prototypes, so baseline images cannot be shown for methodA without fabricating frames.</div>
<div class="summary" id="summary"></div>
<div class="controls">
<label>method <select id="methodSelect"><option value="all">all</option></select></label>
<label>baseline <select id="baselineSelect"><option value="all">all</option></select></label>
<label>target <select id="targetSelect"><option value="all">all</option></select></label>
<label><input type="checkbox" id="methodBOnly"> methodB only: real baseline frames</label>
<label>sort <select id="sortSelect"><option value="gap">smallest relative gap</option><option value="stage">target stage</option><option value="target">target class</option></select></label>
<span class="count" id="visibleCount"></span>
</div>
<section class="gallery" id="gallery"></section>
</main>
<script>
const DATA = {payload};
const methodSelect = document.getElementById('methodSelect');
const baselineSelect = document.getElementById('baselineSelect');
const targetSelect = document.getElementById('targetSelect');
const sortSelect = document.getElementById('sortSelect');
const methodBOnly = document.getElementById('methodBOnly');
const gallery = document.getElementById('gallery');

DATA.methods.forEach(v => methodSelect.add(new Option(v, v)));
DATA.baselineClasses.forEach(v => baselineSelect.add(new Option(v, v)));
DATA.classNames.forEach(v => targetSelect.add(new Option(v, v)));

function fmt(v, d=3) {{ return Number(v).toFixed(d); }}
function frameLabel(idx) {{ return idx === null ? 'N/A' : String(idx).padStart(3,'0'); }}

function showSummary() {{
  const s = DATA.summary;
  const items = [
    ['nearest6 points scanned', s.total_nearest6_points],
    ['competitive cases', s.competitive_cases],
    ['relative threshold', `${{fmt(s.relative_threshold_percent,1)}}%`],
    ['nonlocal separation', `>${{fmt(s.nonlocal_gap_percent,0)}}%`],
  ];
  document.getElementById('summary').innerHTML = items.map(([k,v]) => `<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`).join('');
}}

function filteredCases() {{
  let cases = DATA.cases.slice();
  if (methodBOnly.checked) cases = cases.filter(c => c.method === 'methodB');
  if (methodSelect.value !== 'all') cases = cases.filter(c => c.method === methodSelect.value);
  if (baselineSelect.value !== 'all') cases = cases.filter(c => c.baseline_class === baselineSelect.value);
  if (targetSelect.value !== 'all') cases = cases.filter(c => c.target_class === targetSelect.value);
  if (sortSelect.value === 'gap') cases.sort((a,b) => a.relative_gap - b.relative_gap);
  if (sortSelect.value === 'stage') cases.sort((a,b) => a.target_stage_percent - b.target_stage_percent || a.relative_gap - b.relative_gap);
  if (sortSelect.value === 'target') cases.sort((a,b) => a.target_class.localeCompare(b.target_class) || a.relative_gap - b.relative_gap);
  return cases;
}}

function imgOrPlaceholder(src, note) {{
  if (src) return `<img loading="lazy" src="${{src}}" alt="frame">`;
  return `<div class="placeholder">${{note || 'No frame available'}}</div>`;
}}

function card(c) {{
  return `<article class="card">
    <h3>${{c.label}}</h3>
    <div class="images">
      <div class="imgbox">
        <h4>Target frame ${{frameLabel(c.target_frame_index)}}<br>${{c.target_class}} seq=${{c.sequence_id}}, stage=${{fmt(c.target_stage_percent,0)}}%</h4>
        ${{imgOrPlaceholder(c.target_frame_src)}}
      </div>
      <div class="imgbox">
        <h4>Best baseline frame ${{frameLabel(c.best_frame_index)}}<br>progress=${{fmt(c.best_progress,0)}}%, d=${{fmt(c.best_distance,4)}}</h4>
        ${{imgOrPlaceholder(c.best_frame_src, c.best_note)}}
      </div>
      <div class="imgbox">
        <h4>Second candidate frame ${{frameLabel(c.second_frame_index)}}<br>progress=${{fmt(c.second_progress,0)}}%, d=${{fmt(c.second_distance,4)}}</h4>
        ${{imgOrPlaceholder(c.second_frame_src, c.second_note)}}
      </div>
    </div>
    <div class="meta">absolute gap=${{fmt(c.absolute_gap,6)}}; relative gap=${{fmt(c.relative_gap * 100,2)}}%; method=${{c.method}}; baseline=${{c.baseline_class}}</div>
  </article>`;
}}

function render() {{
  const cases = filteredCases();
  document.getElementById('visibleCount').textContent = `${{cases.length}} visible cases`;
  gallery.innerHTML = cases.map(card).join('');
}}

[methodSelect, baselineSelect, targetSelect, sortSelect, methodBOnly].forEach(el => el.addEventListener('change', render));
showSummary();
render();
</script>
</body>
</html>
"""

    def run(self) -> Path:
        cases, summary = self.build_cases()
        output = self.cfg.output_root / "nearest_candidate_competition_frames.html"
        output.write_text(self.render_html(cases, summary), encoding="utf-8")
        return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate frame gallery for competing nearest-baseline candidates.")
    parser.add_argument("--analysis_input_root", default=str(CompetitionFrameConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(CompetitionFrameConfig.projection_output_root))
    parser.add_argument("--nearest_output_root", default=str(CompetitionFrameConfig.nearest_output_root))
    parser.add_argument("--output_root", default=str(CompetitionFrameConfig.output_root))
    parser.add_argument("--relative_threshold", type=float, default=CompetitionFrameConfig.relative_threshold)
    parser.add_argument("--nonlocal_gap_percent", type=float, default=CompetitionFrameConfig.nonlocal_gap_percent)
    return parser


def main() -> None:
    cfg = CompetitionFrameConfig.from_args(build_arg_parser().parse_args())
    output = CompetitionFrameReport(cfg).run()
    print(f"competition_frame_html: {output}")


if __name__ == "__main__":
    main()
