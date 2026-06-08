from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class FramePairConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    nearest_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_frame_pair_interactive")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FramePairConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            nearest_output_root=Path(args.nearest_output_root),
            output_root=Path(args.output_root),
        )


class NearestFramePairReport:
    def __init__(self, cfg: FramePairConfig):
        self.cfg = cfg
        self.asset_root = cfg.output_root / "assets" / "frames"
        self.asset_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def frame_index(percent: float, n_frames: int = 20) -> int:
        idx = int((percent / 100.0) * (n_frames - 1) + 0.5)
        return max(0, min(n_frames - 1, idx))

    def normalized_frames_dir(self, class_name: str, sequence_id: str) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "normalized_frames"
            / class_name
            / str(sequence_id)
        )

    def load_methodb_meta(self) -> dict:
        path = self.cfg.projection_output_root / "methodB" / "prototypes" / "projection_meta_methodB.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def copy_frame(self, source: Path, asset_subdir: str) -> str:
        if not source.is_file():
            raise FileNotFoundError(f"Missing normalized frame image: {source}")
        destination = self.asset_root / asset_subdir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        return destination.relative_to(self.cfg.output_root).as_posix()

    def target_frame_asset(self, class_name: str, sequence_id: str, stage_percent: float) -> tuple[int, str]:
        idx = self.frame_index(stage_percent)
        source = self.normalized_frames_dir(class_name, sequence_id) / f"{idx:03d}.png"
        asset = self.copy_frame(source, f"target/{class_name}_{sequence_id}")
        return idx, asset

    def baseline_frame_asset(
        self,
        method: str,
        baseline_class: str,
        nearest_progress_percent: float,
        methodb_meta: dict,
    ) -> tuple[int | None, str | None, str | None]:
        if method != "methodB":
            return None, None, "methodA baseline is a median prototype and has no real source frame."
        baseline_seq = str(methodb_meta[baseline_class]["sequence_id"])
        idx = self.frame_index(nearest_progress_percent)
        source = self.normalized_frames_dir(baseline_class, baseline_seq) / f"{idx:03d}.png"
        asset = self.copy_frame(source, f"baseline/{method}_{baseline_class}_{baseline_seq}")
        return idx, asset, None

    def build_points(self) -> list[dict]:
        rows = self.read_csv(
            self.cfg.nearest_output_root / "csv" / "nearest6_nearest_baseline_curve_all.csv"
        )
        methodb_meta = self.load_methodb_meta()
        points: list[dict] = []

        for row in rows:
            stage = float(row["target_stage_percent"])
            nearest_progress = float(row["nearest_baseline_progress_percent"])
            target_idx, target_asset = self.target_frame_asset(row["target_class"], row["sequence_id"], stage)
            baseline_idx, baseline_asset, baseline_note = self.baseline_frame_asset(
                row["method"],
                row["baseline_class"],
                nearest_progress,
                methodb_meta,
            )
            label = (
                f"{row['method']} | baseline={row['baseline_class']} | "
                f"{row['target_class']} seq={row['sequence_id']}, rank={row['rank']}"
            )
            points.append(
                {
                    "method": row["method"],
                    "baseline_class": row["baseline_class"],
                    "target_class": row["target_class"],
                    "sequence_id": row["sequence_id"],
                    "rank": row["rank"],
                    "label": label,
                    "target_stage_percent": stage,
                    "nearest_baseline_progress_percent": nearest_progress,
                    "nearest_distance": float(row["nearest_distance"]),
                    "target_frame_index": target_idx,
                    "target_frame_src": target_asset,
                    "baseline_frame_index": baseline_idx,
                    "baseline_frame_src": baseline_asset,
                    "baseline_note": baseline_note,
                }
            )
        return points

    @staticmethod
    def json_data(data: object) -> str:
        return json.dumps(data, ensure_ascii=False)

    def render_html(self, points: list[dict]) -> str:
        data_json = self.json_data(
            {
                "points": points,
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
<title>Nearest Baseline Frame Pair Viewer</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.5}}
main{{max-width:1220px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 8px}}
h2{{font-size:20px;margin:24px 0 10px}}
p{{max-width:980px}}
.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:white;border:1px solid #ddd;padding:12px;margin:14px 0}}
.controls label{{font-size:13px;display:inline-flex;gap:6px;align-items:center}}
select{{font-size:13px;padding:5px}}
.class-filter{{border:1px solid #bbb;padding:6px 9px;background:#fff;cursor:pointer;user-select:none}}
.chart-card{{background:white;border:1px solid #ddd;padding:12px;margin:14px 0}}
svg{{width:100%;height:auto;display:block;background:white}}
.axis{{stroke:#222;stroke-width:1.2}}
.grid-line{{stroke:#ddd;stroke-width:1}}
.tick{{font-size:12px;fill:#555}}
.line{{fill:none;stroke-width:1.8;opacity:.38;vector-effect:non-scaling-stroke}}
.line:hover{{opacity:.95;stroke-width:3.2}}
.point{{stroke:white;stroke-width:1.3;cursor:pointer;opacity:.85}}
.point:hover{{stroke:#111;stroke-width:2.2;opacity:1}}
.point.selected{{stroke:#111;stroke-width:3.2}}
.viewer{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}}
.frame-panel{{background:white;border:1px solid #ddd;padding:12px}}
.frame-panel h3{{font-size:16px;margin:0 0 8px}}
.frame-panel img{{width:100%;max-height:480px;object-fit:contain;background:#eee;display:block;border:1px solid #ddd}}
.placeholder{{min-height:260px;display:flex;align-items:center;justify-content:center;background:#f2f2f2;border:1px solid #ddd;color:#555;text-align:center;padding:16px}}
.meta{{background:#f7f7f7;border:1px solid #ddd;padding:10px;font-size:13px;margin-top:10px}}
.note{{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}}
.tooltip{{position:fixed;display:none;pointer-events:none;background:#222;color:white;padding:5px 7px;border-radius:3px;font-size:12px;z-index:20}}
</style>
</head>
<body>
<main>
<h1>Nearest Baseline Frame Pair Viewer</h1>
<p>Click a point on the curve to compare the target normalized frame with the nearest baseline normalized frame. The image frame is the nearest available 20-point normalized sample, so it is a visual approximation of the interpolated feature point.</p>
<div class="note">Important: methodA baselines are median prototypes and have no real source frame. Full target/baseline image comparison is available for methodB baselines.</div>

<div class="controls">
<label>method <select id="methodSelect"></select></label>
<label>baseline <select id="baselineSelect"></select></label>
<label>chart <select id="chartSelect">
<option value="progress">nearest progress over target stage</option>
<option value="distance">nearest distance over target stage</option>
<option value="newcurve">new curve: progress vs distance</option>
</select></label>
<label class="class-filter"><input type="checkbox" data-class-filter="polite" checked> polite</label>
<label class="class-filter"><input type="checkbox" data-class-filter="truesmile" checked> truesmile</label>
<label class="class-filter"><input type="checkbox" data-class-filter="ambiguous" checked> ambiguous</label>
</div>

<section class="chart-card">
<div id="chart"></div>
<div class="meta" id="selectedMeta">Click a curve point to inspect the frame pair.</div>
</section>

<section class="viewer">
<div class="frame-panel">
<h3>Target frame</h3>
<div id="targetFrame"></div>
</div>
<div class="frame-panel">
<h3>Nearest baseline frame</h3>
<div id="baselineFrame"></div>
</div>
</section>
<div class="tooltip" id="tooltip"></div>
</main>
<script>
const DATA = {data_json};

const methodSelect = document.getElementById('methodSelect');
const baselineSelect = document.getElementById('baselineSelect');
const chartSelect = document.getElementById('chartSelect');
const chartRoot = document.getElementById('chart');
const selectedMeta = document.getElementById('selectedMeta');
const targetFrame = document.getElementById('targetFrame');
const baselineFrame = document.getElementById('baselineFrame');
const tooltip = document.getElementById('tooltip');

DATA.methods.forEach(m => methodSelect.add(new Option(m, m)));
DATA.baselineClasses.forEach(b => baselineSelect.add(new Option(b, b)));
methodSelect.value = 'methodB';
baselineSelect.value = 'truesmile';

function selectedClasses() {{
  return new Set(Array.from(document.querySelectorAll('[data-class-filter]:checked')).map(x => x.dataset.classFilter));
}}

function filteredPoints() {{
  const classes = selectedClasses();
  return DATA.points.filter(p =>
    p.method === methodSelect.value &&
    p.baseline_class === baselineSelect.value &&
    classes.has(p.target_class)
  );
}}

function groupByCurve(points) {{
  const groups = new Map();
  for (const p of points) {{
    const key = `${{p.target_class}}|${{p.sequence_id}}|${{p.rank}}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }}
  return Array.from(groups.values()).map(g => g.sort((a,b) => a.target_stage_percent - b.target_stage_percent));
}}

function chartXY(p) {{
  const type = chartSelect.value;
  if (type === 'progress') return {{x:p.target_stage_percent, y:p.nearest_baseline_progress_percent}};
  if (type === 'distance') return {{x:p.target_stage_percent, y:p.nearest_distance}};
  return {{x:p.nearest_baseline_progress_percent, y:p.nearest_distance}};
}}

function renderChart() {{
  const groups = groupByCurve(filteredPoints());
  const width = 1040, height = 560;
  const left = 70, right = 24, top = 36, bottom = 58;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const all = groups.flat();
  const coords = all.map(chartXY);
  let xMin = 0, xMax = 100;
  let yMin = 0, yMax = 100;
  if (chartSelect.value !== 'progress') {{
    yMax = Math.max(0.01, ...coords.map(c => c.y)) * 1.1;
  }}
  const sx = x => left + (x - xMin) / (xMax - xMin) * plotW;
  const sy = y => top + plotH - (y - yMin) / (yMax - yMin) * plotH;
  const xTicks = [0,20,40,60,80,100];
  const yTicks = chartSelect.value === 'progress' ? [0,20,40,60,80,100] : [0, yMax*.25, yMax*.5, yMax*.75, yMax];
  const path = pts => pts.map((p,i) => {{
    const c = chartXY(p);
    return `${{i ? 'L' : 'M'}}${{sx(c.x).toFixed(2)}},${{sy(c.y).toFixed(2)}}`;
  }}).join(' ');
  let svg = `<svg viewBox="0 0 ${{width}} ${{height}}">`;
  svg += `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white"/>`;
  for (const t of xTicks) {{
    svg += `<line x1="${{sx(t)}}" y1="${{top}}" x2="${{sx(t)}}" y2="${{top+plotH}}" class="grid-line"/>`;
    svg += `<text x="${{sx(t)}}" y="${{top+plotH+23}}" class="tick" text-anchor="middle">${{t}}</text>`;
  }}
  for (const t of yTicks) {{
    svg += `<line x1="${{left}}" y1="${{sy(t)}}" x2="${{left+plotW}}" y2="${{sy(t)}}" class="grid-line"/>`;
    svg += `<text x="${{left-10}}" y="${{sy(t)+4}}" class="tick" text-anchor="end">${{chartSelect.value === 'progress' ? t.toFixed(0) : t.toFixed(2)}}</text>`;
  }}
  svg += `<line x1="${{left}}" y1="${{top+plotH}}" x2="${{left+plotW}}" y2="${{top+plotH}}" class="axis"/>`;
  svg += `<line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+plotH}}" class="axis"/>`;
  for (const group of groups) {{
    const color = DATA.colors[group[0].target_class];
    svg += `<path class="line" d="${{path(group)}}" stroke="${{color}}"/>`;
    for (const p of group) {{
      const c = chartXY(p);
      const payload = encodeURIComponent(JSON.stringify(p));
      svg += `<circle class="point" cx="${{sx(c.x)}}" cy="${{sy(c.y)}}" r="4.5" fill="${{color}}" data-point="${{payload}}" data-tip="${{p.label}} | target ${{p.target_stage_percent}}%, nearest baseline ${{p.nearest_baseline_progress_percent}}%"/>`;
    }}
  }}
  const xLabel = chartSelect.value === 'newcurve' ? 'nearest baseline progress (%)' : 'target stage (%)';
  const yLabel = chartSelect.value === 'progress' ? 'nearest baseline progress (%)' : 'nearest distance (L2)';
  svg += `<text x="${{left+plotW/2}}" y="${{height-14}}" class="tick" text-anchor="middle">${{xLabel}}</text>`;
  svg += `<text x="18" y="${{top+plotH/2}}" class="tick" text-anchor="middle" transform="rotate(-90 18 ${{top+plotH/2}})">${{yLabel}}</text>`;
  svg += `</svg>`;
  chartRoot.innerHTML = svg;
  attachEvents();
}}

function attachEvents() {{
  document.querySelectorAll('.point').forEach(el => {{
    el.addEventListener('click', () => {{
      document.querySelectorAll('.point.selected').forEach(p => p.classList.remove('selected'));
      el.classList.add('selected');
      showPoint(JSON.parse(decodeURIComponent(el.dataset.point)));
    }});
    el.addEventListener('mousemove', e => {{
      tooltip.textContent = el.dataset.tip;
      tooltip.style.display = 'block';
      tooltip.style.left = `${{e.clientX + 12}}px`;
      tooltip.style.top = `${{e.clientY + 12}}px`;
    }});
    el.addEventListener('mouseleave', () => tooltip.style.display = 'none');
  }});
}}

function showPoint(p) {{
  selectedMeta.innerHTML = `
    <b>${{p.label}}</b><br>
    target stage: ${{p.target_stage_percent}}%, target frame index: ${{String(p.target_frame_index).padStart(3,'0')}}<br>
    nearest baseline progress: ${{p.nearest_baseline_progress_percent}}%, baseline frame index: ${{p.baseline_frame_index === null ? 'N/A' : String(p.baseline_frame_index).padStart(3,'0')}}<br>
    nearest distance: ${{p.nearest_distance.toFixed(6)}}
  `;
  targetFrame.innerHTML = `
    <img src="${{p.target_frame_src}}" alt="target frame">
    <div class="meta">Target: ${{p.target_class}} seq=${{p.sequence_id}}, rank=${{p.rank}}, stage=${{p.target_stage_percent}}%</div>
  `;
  if (p.baseline_frame_src) {{
    baselineFrame.innerHTML = `
      <img src="${{p.baseline_frame_src}}" alt="baseline frame">
      <div class="meta">Baseline: ${{p.method}} ${{p.baseline_class}}, nearest progress=${{p.nearest_baseline_progress_percent}}%</div>
    `;
  }} else {{
    baselineFrame.innerHTML = `<div class="placeholder">${{p.baseline_note}}</div>`;
  }}
}}

[methodSelect, baselineSelect, chartSelect].forEach(el => el.addEventListener('change', renderChart));
document.querySelectorAll('[data-class-filter]').forEach(el => el.addEventListener('change', renderChart));
renderChart();
</script>
</body>
</html>
"""

    def run(self) -> Path:
        points = self.build_points()
        output = self.cfg.output_root / "nearest_frame_pair_viewer.html"
        output.write_text(self.render_html(points), encoding="utf-8")
        return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an interactive frame-pair viewer for nearest-baseline points.")
    parser.add_argument("--analysis_input_root", default=str(FramePairConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(FramePairConfig.projection_output_root))
    parser.add_argument("--nearest_output_root", default=str(FramePairConfig.nearest_output_root))
    parser.add_argument("--output_root", default=str(FramePairConfig.output_root))
    return parser


def main() -> None:
    cfg = FramePairConfig.from_args(build_arg_parser().parse_args())
    output = NearestFramePairReport(cfg).run()
    print(f"frame_pair_html: {output}")


if __name__ == "__main__":
    main()
