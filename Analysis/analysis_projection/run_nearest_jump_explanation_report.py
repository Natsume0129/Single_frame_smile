from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
STAGES = tuple(range(5, 101, 5))
SEARCH_PERCENTS = np.arange(0.0, 101.0, 1.0, dtype=np.float64)


@dataclass
class JumpReportConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    nearest_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_jump_explanation")
    max_cases: int = 40

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "JumpReportConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            nearest_output_root=Path(args.nearest_output_root),
            output_root=Path(args.output_root),
            max_cases=int(args.max_cases),
        )


class NearestJumpExplanationReport:
    def __init__(self, cfg: JumpReportConfig):
        self.cfg = cfg
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)

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
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

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

    def load_baseline_curve(self, method: str, baseline_class: str) -> np.ndarray:
        return np.load(self.prototype_path(method, baseline_class), allow_pickle=False).astype(np.float64)

    def load_target_curve(self, source_type: str, method: str, target_class: str, sequence_id: str) -> np.ndarray:
        if source_type == "prototype":
            return np.load(self.prototype_path(method, target_class), allow_pickle=False).astype(np.float64)
        return np.load(
            self.normalized_sequence_path(target_class, sequence_id),
            allow_pickle=False,
        ).astype(np.float64)

    def load_output_rows(self) -> list[dict[str, str]]:
        csv_dir = self.cfg.nearest_output_root / "csv"
        rows: list[dict[str, str]] = []
        for source_type, filename in (
            ("nearest6", "nearest6_nearest_baseline_curve_all.csv"),
            ("prototype", "prototype_nearest_baseline_curve_all.csv"),
        ):
            for row in self.read_csv(csv_dir / filename):
                row = dict(row)
                row["source_table"] = source_type
                rows.append(row)
        return rows

    @staticmethod
    def group_rows(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], list[dict[str, str]]]:
        grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            key = (
                row["source_table"],
                row["method"],
                row["baseline_class"],
                row["target_class"],
                row["sequence_id"],
                row["rank"],
            )
            grouped[key].append(row)
        return grouped

    def scan_jump_events(self, rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
        grouped = self.group_rows(rows)
        events: list[dict] = []
        groups_with_jumps = set()

        for key, group_rows in grouped.items():
            sorted_rows = sorted(group_rows, key=lambda r: float(r["target_stage_percent"]))
            for prev, cur in zip(sorted_rows, sorted_rows[1:]):
                p0 = float(prev["nearest_baseline_progress_percent"])
                p1 = float(cur["nearest_baseline_progress_percent"])
                delta = p1 - p0
                if delta < 0.0:
                    groups_with_jumps.add(key)
                    events.append(
                        {
                            "source_type": key[0],
                            "method": key[1],
                            "baseline_class": key[2],
                            "target_class": key[3],
                            "sequence_id": key[4],
                            "rank": key[5],
                            "stage_from": float(prev["target_stage_percent"]),
                            "stage_to": float(cur["target_stage_percent"]),
                            "progress_from": p0,
                            "progress_to": p1,
                            "delta_progress": delta,
                            "distance_from": float(prev["nearest_distance"]),
                            "distance_to": float(cur["nearest_distance"]),
                        }
                    )

        threshold_summary = []
        for threshold in (-1, -5, -10, -20, -40):
            selected = [e for e in events if e["delta_progress"] <= threshold]
            threshold_summary.append(
                {
                    "threshold": threshold,
                    "event_count": len(selected),
                    "group_count": len(
                        {
                            (
                                e["source_type"],
                                e["method"],
                                e["baseline_class"],
                                e["target_class"],
                                e["sequence_id"],
                                e["rank"],
                            )
                            for e in selected
                        }
                    ),
                    "by_source_target": [
                        {"source_type": key[0], "target_class": key[1], "count": value}
                        for key, value in sorted(Counter((e["source_type"], e["target_class"]) for e in selected).items())
                    ],
                }
            )

        high_to_low = [
            e for e in events if e["progress_from"] >= 70.0 and e["progress_to"] <= 20.0
        ]
        summary = {
            "total_groups": len(grouped),
            "groups_with_backward_jumps": len(groups_with_jumps),
            "total_backward_events": len(events),
            "threshold_summary": threshold_summary,
            "high_to_low_event_count": len(high_to_low),
        }
        return events, summary

    def baseline_samples(self, curve: np.ndarray) -> np.ndarray:
        return np.vstack([self.point_at_percent(curve, p) for p in SEARCH_PERCENTS])

    def mapping_for_curve(self, baseline_curve: np.ndarray, target_curve: np.ndarray, reanchor: bool = False) -> list[dict]:
        b = baseline_curve - baseline_curve[0] if reanchor else baseline_curve
        t = target_curve - target_curve[0] if reanchor else target_curve
        samples = self.baseline_samples(b)
        rows: list[dict] = []
        previous_point: np.ndarray | None = None

        for stage in STAGES:
            target_point = self.point_at_percent(t, stage)
            distances = np.linalg.norm(samples - target_point, axis=1)
            order = np.argsort(distances)
            best = int(order[0])
            second_far = next(int(i) for i in order[1:] if abs(int(i) - best) > 10)
            step_length = 0.0 if previous_point is None else float(np.linalg.norm(target_point - previous_point))
            rows.append(
                {
                    "stage": stage,
                    "progress": float(best),
                    "distance": float(distances[best]),
                    "second_far_progress": float(second_far),
                    "second_far_distance": float(distances[second_far]),
                    "margin": float(distances[second_far] - distances[best]),
                    "target_norm": float(np.linalg.norm(target_point)),
                    "target_step_length": step_length,
                }
            )
            previous_point = target_point
        return rows

    def distance_profile(self, baseline_curve: np.ndarray, target_curve: np.ndarray, stage: float) -> list[dict]:
        samples = self.baseline_samples(baseline_curve)
        target_point = self.point_at_percent(target_curve, stage)
        distances = np.linalg.norm(samples - target_point, axis=1)
        return [
            {"progress": float(progress), "distance": float(distance)}
            for progress, distance in zip(SEARCH_PERCENTS, distances)
        ]

    def candidate_table(
        self,
        baseline_curve: np.ndarray,
        target_curve: np.ndarray,
        stage: float,
        n: int = 12,
    ) -> list[dict]:
        samples = self.baseline_samples(baseline_curve)
        target_point = self.point_at_percent(target_curve, stage)
        distances = np.linalg.norm(samples - target_point, axis=1)
        order = np.argsort(distances)[:n]
        return [
            {"rank": i + 1, "progress": int(idx), "distance": float(distances[idx])}
            for i, idx in enumerate(order)
        ]

    def case_detail(self, event: dict) -> dict:
        baseline_curve = self.load_baseline_curve(event["method"], event["baseline_class"])
        target_curve = self.load_target_curve(
            event["source_type"],
            event["method"],
            event["target_class"],
            event["sequence_id"],
        )
        mapping_raw = self.mapping_for_curve(baseline_curve, target_curve, reanchor=False)
        mapping_reanchor = self.mapping_for_curve(baseline_curve, target_curve, reanchor=True)
        stage_from = event["stage_from"]
        stage_to = event["stage_to"]
        progress_from = int(event["progress_from"])
        progress_to = int(event["progress_to"])
        b_from = self.point_at_percent(baseline_curve, progress_from)
        b_to = self.point_at_percent(baseline_curve, progress_to)
        t_from = self.point_at_percent(target_curve, stage_from)
        t_to = self.point_at_percent(target_curve, stage_to)

        signed_boundary = []
        for stage in STAGES:
            z = self.point_at_percent(target_curve, stage)
            signed = float(np.linalg.norm(z - b_to) ** 2 - np.linalg.norm(z - b_from) ** 2)
            signed_boundary.append({"stage": stage, "signed": signed})

        geometry = {
            "baseline_start_norm": float(np.linalg.norm(baseline_curve[0])),
            "target_start_norm": float(np.linalg.norm(target_curve[0])),
            "start_offset_norm": float(np.linalg.norm(target_curve[0] - baseline_curve[0])),
            "baseline_endpoint_norm": float(np.linalg.norm(baseline_curve[-1] - baseline_curve[0])),
            "candidate_distance": float(np.linalg.norm(b_from - b_to)),
            "target_step_from_to": float(np.linalg.norm(t_to - t_from)),
            "target_norm_from": float(np.linalg.norm(t_from - target_curve[0])),
            "target_norm_to": float(np.linalg.norm(t_to - target_curve[0])),
            "candidate_cosine_from_origin": self.cosine(b_from, b_to),
            "candidate_cosine_from_baseline_start": self.cosine(
                b_from - baseline_curve[0],
                b_to - baseline_curve[0],
            ),
        }

        return {
            "event": event,
            "label": self.case_label(event),
            "mapping_raw": mapping_raw,
            "mapping_reanchor": mapping_reanchor,
            "profile_from": self.distance_profile(baseline_curve, target_curve, stage_from),
            "profile_to": self.distance_profile(baseline_curve, target_curve, stage_to),
            "candidates_from": self.candidate_table(baseline_curve, target_curve, stage_from),
            "candidates_to": self.candidate_table(baseline_curve, target_curve, stage_to),
            "signed_boundary": signed_boundary,
            "geometry": geometry,
        }

    @staticmethod
    def case_label(event: dict) -> str:
        rank = f", rank={event['rank']}" if event["rank"] else ""
        return (
            f"{event['source_type']} | {event['method']} | baseline={event['baseline_class']} | "
            f"target={event['target_class']} | seq={event['sequence_id']}{rank} | "
            f"{event['stage_from']:.0f}->{event['stage_to']:.0f}%: "
            f"{event['progress_from']:.0f}->{event['progress_to']:.0f}%"
        )

    def selected_cases(self, events: list[dict]) -> list[dict]:
        top = sorted(events, key=lambda e: e["delta_progress"])[: self.cfg.max_cases]
        high_to_low = [
            e for e in events if e["progress_from"] >= 70.0 and e["progress_to"] <= 20.0
        ]
        keyed: dict[tuple, dict] = {}
        for event in high_to_low + top:
            key = (
                event["source_type"],
                event["method"],
                event["baseline_class"],
                event["target_class"],
                event["sequence_id"],
                event["rank"],
                event["stage_from"],
                event["stage_to"],
            )
            keyed[key] = event
        return sorted(keyed.values(), key=lambda e: e["delta_progress"])

    @staticmethod
    def to_json(data: object) -> str:
        def clean(value: object) -> object:
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, tuple):
                return [clean(v) for v in value]
            if isinstance(value, (float, np.floating)):
                number = float(value)
                return number if math.isfinite(number) else None
            if isinstance(value, (int, np.integer)):
                return int(value)
            return value

        return json.dumps(clean(data), ensure_ascii=False, allow_nan=False)

    def render_html(self, payload: dict) -> str:
        data_json = self.to_json(payload)
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Nearest Progress Jump Explanation</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.55}}
main{{max-width:1180px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 8px}}
h2{{font-size:21px;margin:30px 0 10px;border-bottom:1px solid #ddd;padding-bottom:6px}}
h3{{font-size:16px;margin:0 0 6px}}
p{{max-width:960px}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0 18px}}
.metric{{background:white;border:1px solid #ddd;padding:12px}}
.metric b{{font-size:22px;display:block}}
.panel{{background:white;border:1px solid #ddd;padding:14px 16px;margin:16px 0}}
.case-controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}}
select{{font-size:13px;padding:6px;max-width:100%}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.wide{{grid-column:1/-1}}
.chart{{border:1px solid #ddd;background:#fff;padding:10px}}
.chart svg{{width:100%;height:auto;display:block}}
.axis{{stroke:#222;stroke-width:1.2}}
.grid-line{{stroke:#ddd;stroke-width:1}}
.tick{{font-size:11px;fill:#555}}
.title{{font-size:14px;font-weight:700;text-anchor:middle}}
.line{{fill:none;stroke-width:2.2;vector-effect:non-scaling-stroke}}
.line.secondary{{stroke-dasharray:6 5}}
.line.thin{{stroke-width:1.7}}
.jump{{stroke:#d62728;stroke-width:5;opacity:.9;vector-effect:non-scaling-stroke}}
.marker{{stroke:white;stroke-width:1.5}}
.tooltip{{position:fixed;pointer-events:none;background:#222;color:white;padding:5px 7px;border-radius:3px;font-size:12px;display:none;z-index:20}}
table{{border-collapse:collapse;font-size:13px;width:100%;margin-top:8px}}
th,td{{border:1px solid #ddd;padding:6px 7px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f3f3f3}}
.note{{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}}
.formula{{font-family:Consolas,monospace;background:#f6f6f6;border-left:4px solid #777;padding:10px 12px}}
.small{{font-size:13px;color:#555}}
.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.fact{{background:#f8f8f8;border:1px solid #ddd;padding:8px;font-size:13px}}
</style>
</head>
<body>
<main>
<h1>Nearest Progress Jump Explanation</h1>
<p>This standalone report explains why nearest-baseline progress can jump backward even when the facial expression changes naturally. The key point is that nearest progress is an unconstrained geometric argmin, not time itself.</p>

<div class="summary" id="summary"></div>

<section class="panel">
<h2>Core Mechanism</h2>
<div class="formula">
d_t(p) = || C_target(t) - C_baseline(p) ||_2<br>
p*(t) = argmin_p d_t(p)
</div>
<p>If a target point is close to the boundary between two baseline candidates, a small movement in feature space can switch the winner from a late baseline stage to an early baseline stage. This is a nearest-assignment jump, not direct evidence that the smile itself moved backward in time.</p>
<div class="note">This report does not use a time-window constraint. Baseline progress is still searched globally over 0% to 100%.</div>
</section>

<section class="panel">
<h2>Case Inspector</h2>
<div class="case-controls">
<label for="caseSelect">Jump case:</label>
<select id="caseSelect"></select>
</div>
<div id="caseFacts" class="facts"></div>
</section>

<section class="grid">
<div class="chart wide" id="progressChart"></div>
<div class="chart" id="profileChart"></div>
<div class="chart" id="marginChart"></div>
<div class="chart" id="normChart"></div>
<div class="chart" id="boundaryChart"></div>
</section>

<section class="panel">
<h2>Nearest Candidate Tables</h2>
<div class="grid">
<div>
<h3 id="fromTableTitle">Before jump</h3>
<div id="fromTable"></div>
</div>
<div>
<h3 id="toTableTitle">After jump</h3>
<div id="toTable"></div>
</div>
</div>
</section>

<section class="panel">
<h2>Interpretation</h2>
<ul>
<li>Baseline subtraction reduces static identity, but it does not turn fc7 into a pure expression coordinate.</li>
<li>The start point of a target sequence can still differ from the prototype start point, so residual offset affects all L2 distances.</li>
<li>True-smile sequences can have peak-and-relaxation behavior; later normalized stages are not guaranteed to be stronger than earlier stages.</li>
<li>A curved high-dimensional baseline can have multiple local candidate regions. When their distances are close, the argmin can switch abruptly.</li>
</ul>
</section>

<div class="tooltip" id="tooltip"></div>
</main>
<script>
const DATA = {data_json};

const COLORS = {{
  raw: '#1f77b4',
  reanchor: '#9467bd',
  before: '#2ca02c',
  after: '#d62728',
  margin: '#ff7f0e',
  norm: '#1f77b4',
  boundary: '#555'
}};

function fmt(value, digits = 3) {{
  if (value === null || Number.isNaN(value)) return 'nan';
  return Number(value).toFixed(digits);
}}

function showSummary() {{
  const s = DATA.summary;
  const items = [
    ['Total curves', s.total_groups],
    ['Curves with backward jumps', s.groups_with_backward_jumps],
    ['Backward jump events', s.total_backward_events],
    ['High-to-low events', s.high_to_low_event_count],
  ];
  document.getElementById('summary').innerHTML = items.map(([k,v]) =>
    `<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`
  ).join('');
}}

function setupSelect() {{
  const sel = document.getElementById('caseSelect');
  DATA.cases.forEach((c, i) => {{
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = c.label;
    sel.appendChild(opt);
  }});
  sel.addEventListener('change', () => renderCase(Number(sel.value)));
}}

function chart(containerId, title, series, options = {{}}) {{
  const width = 980, height = 430;
  const left = 58, right = 26, top = 42, bottom = 54;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const allX = series.flatMap(s => s.points.map(p => p.x));
  const allY = series.flatMap(s => s.points.map(p => p.y));
  let xMin = options.xMin ?? Math.min(...allX);
  let xMax = options.xMax ?? Math.max(...allX);
  let yMin = options.yMin ?? Math.min(...allY);
  let yMax = options.yMax ?? Math.max(...allY);
  if (xMax <= xMin) xMax = xMin + 1;
  if (yMax <= yMin) yMax = yMin + 1;
  const yPad = (yMax - yMin) * 0.08;
  yMin = options.yMin ?? (yMin - yPad);
  yMax = options.yMax ?? (yMax + yPad);
  const sx = x => left + (x - xMin) / (xMax - xMin) * plotW;
  const sy = y => top + plotH - (y - yMin) / (yMax - yMin) * plotH;
  const xTicks = options.xTicks ?? [0,20,40,60,80,100];
  const yTicks = options.yTicks ?? Array.from({{length:5}}, (_,i) => yMin + (yMax-yMin)*i/4);
  const path = pts => pts.map((p,i) => `${{i?'L':'M'}}${{sx(p.x).toFixed(2)}},${{sy(p.y).toFixed(2)}}`).join(' ');
  let html = `<svg viewBox="0 0 ${{width}} ${{height}}" aria-label="${{title}}">`;
  html += `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white"/>`;
  html += `<text x="${{width/2}}" y="24" class="title">${{title}}</text>`;
  xTicks.forEach(t => {{
    html += `<line x1="${{sx(t)}}" y1="${{top}}" x2="${{sx(t)}}" y2="${{top+plotH}}" class="grid-line"/>`;
    html += `<text x="${{sx(t)}}" y="${{top+plotH+22}}" class="tick" text-anchor="middle">${{fmt(t,0)}}</text>`;
  }});
  yTicks.forEach(t => {{
    html += `<line x1="${{left}}" y1="${{sy(t)}}" x2="${{left+plotW}}" y2="${{sy(t)}}" class="grid-line"/>`;
    html += `<text x="${{left-9}}" y="${{sy(t)+4}}" class="tick" text-anchor="end">${{fmt(t, options.yDigits ?? 2)}}</text>`;
  }});
  html += `<line x1="${{left}}" y1="${{top+plotH}}" x2="${{left+plotW}}" y2="${{top+plotH}}" class="axis"/>`;
  html += `<line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+plotH}}" class="axis"/>`;
  if (options.zeroLine && yMin < 0 && yMax > 0) {{
    html += `<line x1="${{left}}" y1="${{sy(0)}}" x2="${{left+plotW}}" y2="${{sy(0)}}" stroke="#777" stroke-dasharray="4 4"/>`;
  }}
  series.forEach(s => {{
    html += `<path class="line ${{s.className || ''}}" d="${{path(s.points)}}" stroke="${{s.color}}" stroke-dasharray="${{s.dash || ''}}"/>`;
    if (s.markers) {{
      s.points.forEach(p => {{
        html += `<circle class="marker" cx="${{sx(p.x)}}" cy="${{sy(p.y)}}" r="${{p.r || 3.5}}" fill="${{s.color}}" data-tip="${{p.tip || ''}}"/>`;
      }});
    }}
  }});
  if (options.jumpSegment) {{
    const j = options.jumpSegment;
    html += `<line class="jump" x1="${{sx(j.x1)}}" y1="${{sy(j.y1)}}" x2="${{sx(j.x2)}}" y2="${{sy(j.y2)}}"/>`;
  }}
  html += `<text x="${{left + plotW/2}}" y="${{height-14}}" class="tick" text-anchor="middle">${{options.xLabel || ''}}</text>`;
  html += `</svg>`;
  document.getElementById(containerId).innerHTML = `<h3>${{title}}</h3>${{html}}<div class="small">${{options.caption || ''}}</div>`;
  attachTooltip(document.getElementById(containerId));
}}

function attachTooltip(root) {{
  const tooltip = document.getElementById('tooltip');
  root.querySelectorAll('[data-tip]').forEach(el => {{
    el.addEventListener('mousemove', e => {{
      const tip = el.getAttribute('data-tip');
      if (!tip) return;
      tooltip.textContent = tip;
      tooltip.style.display = 'block';
      tooltip.style.left = `${{e.clientX + 12}}px`;
      tooltip.style.top = `${{e.clientY + 12}}px`;
    }});
    el.addEventListener('mouseleave', () => tooltip.style.display = 'none');
  }});
}}

function table(rows) {{
  return `<table><tr><th>rank</th><th>baseline progress</th><th>distance</th></tr>` +
    rows.map(r => `<tr><td>${{r.rank}}</td><td>${{r.progress}}%</td><td>${{fmt(r.distance,6)}}</td></tr>`).join('') +
    `</table>`;
}}

function renderCase(index) {{
  const c = DATA.cases[index];
  const e = c.event;
  const g = c.geometry;
  document.getElementById('caseFacts').innerHTML = [
    ['jump', `${{fmt(e.stage_from,0)}}→${{fmt(e.stage_to,0)}}% target stage; ${{fmt(e.progress_from,0)}}→${{fmt(e.progress_to,0)}}% nearest progress`],
    ['distance change', `${{fmt(e.distance_from,6)}} → ${{fmt(e.distance_to,6)}}`],
    ['target step length', fmt(g.target_step_from_to,6)],
    ['target norm from start', `${{fmt(g.target_norm_from,6)}} → ${{fmt(g.target_norm_to,6)}}`],
    ['start offset ||T0-B0||', fmt(g.start_offset_norm,6)],
    ['baseline endpoint norm', fmt(g.baseline_endpoint_norm,6)],
  ].map(([k,v]) => `<div class="fact"><b>${{k}}</b><br>${{v}}</div>`).join('');

  const rawPts = c.mapping_raw.map(r => ({{x:r.stage, y:r.progress, tip:`stage ${{r.stage}}%, progress ${{r.progress}}%, distance ${{fmt(r.distance,4)}}`}}));
  const rePts = c.mapping_reanchor.map(r => ({{x:r.stage, y:r.progress, tip:`re-anchor stage ${{r.stage}}%, progress ${{r.progress}}%`}}));
  const fromRaw = c.mapping_raw.find(r => r.stage === e.stage_from);
  const toRaw = c.mapping_raw.find(r => r.stage === e.stage_to);
  chart('progressChart', 'Nearest baseline progress over target stage', [
    {{points: rawPts, color: COLORS.raw, markers: true}},
    {{points: rePts, color: COLORS.reanchor, dash: '6 5', markers: false}},
  ], {{
    xMin:0, xMax:100, yMin:0, yMax:100, xLabel:'target stage (%)',
    jumpSegment: {{x1:e.stage_from, y1:fromRaw.progress, x2:e.stage_to, y2:toRaw.progress}},
    caption:'Blue = raw nearest mapping. Purple dashed = diagnostic re-anchor mapping after subtracting each curve start point. Red segment marks the selected backward jump.'
  }});

  chart('profileChart', 'Distance to every baseline progress before/after jump', [
    {{points: c.profile_from.map(r => ({{x:r.progress, y:r.distance, tip:`before: B(${{r.progress}}%) distance ${{fmt(r.distance,4)}}`}})), color: COLORS.before, markers: false}},
    {{points: c.profile_to.map(r => ({{x:r.progress, y:r.distance, tip:`after: B(${{r.progress}}%) distance ${{fmt(r.distance,4)}}`}})), color: COLORS.after, markers: false}},
  ], {{
    xMin:0, xMax:100, yMin:0, xLabel:'baseline progress (%)',
    caption:`Green = target stage ${{fmt(e.stage_from,0)}}%. Red = target stage ${{fmt(e.stage_to,0)}}%. Near-ties across distant baseline regions create jump risk.`
  }});

  chart('marginChart', 'Instability margin: second nonlocal candidate minus best', [
    {{points: c.mapping_raw.map(r => ({{x:r.stage, y:r.margin, tip:`stage ${{r.stage}}%, margin ${{fmt(r.margin,4)}}`}})), color: COLORS.margin, markers: true}},
  ], {{
    xMin:0, xMax:100, yMin:0, xLabel:'target stage (%)',
    caption:'Small margin means the nearest baseline assignment is unstable because another distant baseline region is almost equally close.'
  }});

  chart('normChart', 'Target norm and step length diagnostics', [
    {{points: c.mapping_raw.map(r => ({{x:r.stage, y:r.target_norm, tip:`stage ${{r.stage}}%, target norm ${{fmt(r.target_norm,4)}}`}})), color: COLORS.norm, markers: true}},
    {{points: c.mapping_raw.map(r => ({{x:r.stage, y:r.target_step_length, tip:`stage ${{r.stage}}%, step length ${{fmt(r.target_step_length,4)}}`}})), color: '#8c564b', dash: '5 4', markers: true}},
  ], {{
    xMin:0, xMax:100, yMin:0, xLabel:'target stage (%)',
    caption:'Large step length or decreasing target norm indicates that the target trajectory itself is not simply moving monotonically outward.'
  }});

  chart('boundaryChart', 'Signed boundary between the two jump candidates', [
    {{points: c.signed_boundary.map(r => ({{x:r.stage, y:r.signed, tip:`stage ${{r.stage}}%, signed d² ${{fmt(r.signed,5)}}`}})), color: COLORS.boundary, markers: true}},
  ], {{
    xMin:0, xMax:100, xLabel:'target stage (%)', zeroLine:true, yDigits:3,
    caption:'This compares squared distance to the after-jump candidate versus the before-jump candidate. Crossing zero means the nearest winner switches.'
  }});

  document.getElementById('fromTableTitle').textContent = `Candidates at target stage ${{fmt(e.stage_from,0)}}%`;
  document.getElementById('toTableTitle').textContent = `Candidates at target stage ${{fmt(e.stage_to,0)}}%`;
  document.getElementById('fromTable').innerHTML = table(c.candidates_from);
  document.getElementById('toTable').innerHTML = table(c.candidates_to);
}}

showSummary();
setupSelect();
renderCase(0);
</script>
</body>
</html>
"""

    def build_payload(self) -> dict:
        rows = self.load_output_rows()
        events, summary = self.scan_jump_events(rows)
        cases = [self.case_detail(event) for event in self.selected_cases(events)]
        return {"summary": summary, "cases": cases}

    def run(self) -> Path:
        payload = self.build_payload()
        output_path = self.cfg.output_root / "nearest_progress_jump_explanation.html"
        output_path.write_text(self.render_html(payload), encoding="utf-8")
        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an interactive explanation report for nearest progress jumps.")
    parser.add_argument("--analysis_input_root", default=str(JumpReportConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(JumpReportConfig.projection_output_root))
    parser.add_argument("--nearest_output_root", default=str(JumpReportConfig.nearest_output_root))
    parser.add_argument("--output_root", default=str(JumpReportConfig.output_root))
    parser.add_argument("--max_cases", type=int, default=JumpReportConfig.max_cases)
    return parser


def main() -> None:
    cfg = JumpReportConfig.from_args(build_arg_parser().parse_args())
    output_path = NearestJumpExplanationReport(cfg).run()
    print(f"jump_explanation_html: {output_path}")


if __name__ == "__main__":
    main()
