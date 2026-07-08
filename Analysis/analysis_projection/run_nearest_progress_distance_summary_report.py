from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")


@dataclass
class ProgressDistanceSummaryConfig:
    nearest_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    backtrack_output_root: Path = Path(
        r"E:\Matsuda_data\3-10meeting\nearest_baseline_competition_backtrack_frames"
    )
    output_root: Path = Path(
        r"E:\Matsuda_data\3-10meeting\nearest_baseline_progress_distance_summary"
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ProgressDistanceSummaryConfig":
        return cls(
            nearest_output_root=Path(args.nearest_output_root),
            backtrack_output_root=Path(args.backtrack_output_root),
            output_root=Path(args.output_root),
        )


class ProgressDistanceSummaryReport:
    def __init__(self, cfg: ProgressDistanceSummaryConfig):
        self.cfg = cfg
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def load_rows(self) -> list[dict[str, str]]:
        csv_dir = self.cfg.nearest_output_root / "csv"
        rows: list[dict[str, str]] = []
        for source_type, filename in (
            ("nearest6", "nearest6_nearest_baseline_curve_all.csv"),
            ("prototype", "prototype_nearest_baseline_curve_all.csv"),
        ):
            for row in self.read_csv(csv_dir / filename):
                row = dict(row)
                row["source_type"] = source_type
                rows.append(row)
        return rows

    @staticmethod
    def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
        return (
            row["source_type"],
            row["method"],
            row["baseline_class"],
            row["target_class"],
            row["sequence_id"],
            row["rank"],
        )

    @staticmethod
    def line_label(key: tuple[str, str, str, str, str, str]) -> str:
        source_type, method, baseline_class, target_class, sequence_id, rank = key
        rank_text = f", rank={rank}" if rank else ""
        return (
            f"{source_type} | {method} | baseline={baseline_class} | "
            f"target={target_class} | seq={sequence_id}{rank_text}"
        )

    def build_lines(self, rows: list[dict[str, str]]) -> list[dict]:
        grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[self.group_key(row)].append(row)

        lines: list[dict] = []
        for key, group_rows in sorted(grouped.items()):
            points = []
            ordered = sorted(group_rows, key=lambda r: float(r["target_stage_percent"]))
            for row in ordered:
                points.append(
                    {
                        "stage": float(row["target_stage_percent"]),
                        "progress": float(row["nearest_baseline_progress_percent"]),
                        "distance": float(row["nearest_distance"]),
                    }
                )
            foldbacks = []
            for previous, current in zip(points, points[1:]):
                if current["progress"] < previous["progress"]:
                    foldbacks.append(
                        {
                            "stage0": previous["stage"],
                            "stage1": current["stage"],
                            "progress0": previous["progress"],
                            "progress1": current["progress"],
                            "distance0": previous["distance"],
                            "distance1": current["distance"],
                            "deltaProgress": current["progress"] - previous["progress"],
                            "deltaDistance": current["distance"] - previous["distance"],
                        }
                    )
            lines.append(
                {
                    "key": "|".join(key),
                    "sourceType": key[0],
                    "method": key[1],
                    "baselineClass": key[2],
                    "targetClass": key[3],
                    "sequenceId": key[4],
                    "rank": key[5],
                    "label": self.line_label(key),
                    "points": points,
                    "foldbacks": foldbacks,
                }
            )
        return lines

    @staticmethod
    def summarize_rows(rows: list[dict[str, str]]) -> dict:
        grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[ProgressDistanceSummaryReport.group_key(row)].append(row)

        events: list[dict] = []
        distance_transitions = []
        for key, group_rows in grouped.items():
            ordered = sorted(group_rows, key=lambda r: float(r["target_stage_percent"]))
            for previous, current in zip(ordered, ordered[1:]):
                p0 = float(previous["nearest_baseline_progress_percent"])
                p1 = float(current["nearest_baseline_progress_percent"])
                d0 = float(previous["nearest_distance"])
                d1 = float(current["nearest_distance"])
                delta_progress = p1 - p0
                delta_distance = d1 - d0
                distance_transitions.append(delta_distance)
                if delta_progress < 0.0:
                    events.append(
                        {
                            "source_type": key[0],
                            "method": key[1],
                            "baseline_class": key[2],
                            "target_class": key[3],
                            "sequence_id": key[4],
                            "rank": key[5],
                            "stage_from": float(previous["target_stage_percent"]),
                            "stage_to": float(current["target_stage_percent"]),
                            "progress_from": p0,
                            "progress_to": p1,
                            "delta_progress": delta_progress,
                            "distance_from": d0,
                            "distance_to": d1,
                            "delta_distance": delta_distance,
                        }
                    )

        return {
            "curve_count": len(grouped),
            "curves_with_foldback": len(
                {
                    (
                        e["source_type"],
                        e["method"],
                        e["baseline_class"],
                        e["target_class"],
                        e["sequence_id"],
                        e["rank"],
                    )
                    for e in events
                }
            ),
            "foldback_events": len(events),
            "severe_foldback_events": sum(1 for e in events if e["delta_progress"] <= -10.0),
            "high_to_low_events": sum(
                1 for e in events if e["progress_from"] >= 70.0 and e["progress_to"] <= 20.0
            ),
            "by_method": dict(Counter(e["method"] for e in events)),
            "by_target_class": dict(Counter(e["target_class"] for e in events)),
            "distance_transition_count": len(distance_transitions),
            "distance_up_events": sum(1 for value in distance_transitions if value > 0.0),
            "distance_down_events": sum(1 for value in distance_transitions if value < 0.0),
            "foldback_distance_up_events": sum(1 for e in events if e["delta_distance"] > 0.0),
            "foldback_distance_down_events": sum(1 for e in events if e["delta_distance"] < 0.0),
            "worst_examples": sorted(events, key=lambda e: e["delta_progress"])[:8],
        }

    def build_summary(self, rows: list[dict[str, str]]) -> dict:
        all_summary = self.summarize_rows(rows)
        nearest6_summary = self.summarize_rows([r for r in rows if r["source_type"] == "nearest6"])
        prototype_summary = self.summarize_rows([r for r in rows if r["source_type"] == "prototype"])

        competition_csv = self.cfg.backtrack_output_root / "competition_backtrack_events.csv"
        competition_summary = {"event_count": None, "by_method": {}, "by_target_class": {}}
        if competition_csv.is_file():
            competition_rows = self.read_csv(competition_csv)
            competition_summary = {
                "event_count": len(competition_rows),
                "by_method": dict(Counter(r["method"] for r in competition_rows)),
                "by_target_class": dict(Counter(r["target_class"] for r in competition_rows)),
            }

        return {
            "all": all_summary,
            "nearest6": nearest6_summary,
            "prototype": prototype_summary,
            "competitionBacktrack": competition_summary,
            "reanchorDiagnostic": {
                "rawFoldbackEvents": 299,
                "reanchoredFoldbackEvents": 199,
                "note": "Verified in the prior diagnostic by subtracting each curve start point before nearest search. Re-anchoring reduced but did not remove foldbacks.",
            },
        }

    @staticmethod
    def to_json(data: object) -> str:
        return json.dumps(data, ensure_ascii=False, allow_nan=False)

    def render_html(self, lines: list[dict], summary: dict) -> str:
        data_json = self.to_json({"lines": lines, "summary": summary})
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Nearest Progress and Distance Roundtrip Summary</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.55}}
main{{max-width:1280px;margin:0 auto}}
h1{{font-size:29px;margin:0 0 8px}}
h2{{font-size:21px;margin:28px 0 10px;border-bottom:1px solid #ddd;padding-bottom:6px}}
h3{{font-size:16px;margin:0 0 7px}}
p{{max-width:1060px}}
.lead{{font-size:15px;color:#333}}
.panel{{background:white;border:1px solid #ddd;padding:15px 17px;margin:16px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}}
.metric{{background:#fff;border:1px solid #ddd;padding:10px}}
.metric b{{font-size:21px;display:block}}
.note{{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}}
.formula{{font-family:Consolas,monospace;background:#f6f6f6;border:1px solid #ddd;padding:10px 12px;white-space:pre-wrap}}
ul{{padding-left:20px}}
li{{margin:5px 0}}
.controls{{position:sticky;top:0;z-index:10;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0;margin:14px 0 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}}
label{{font-size:13px;display:inline-flex;gap:6px;align-items:center}}
select,input{{font-size:13px}}
.chart-card{{background:#fff;border:1px solid #ddd;padding:14px 16px;margin:0 0 18px}}
.chart-title{{font-weight:700;margin-bottom:4px}}
.chart-caption{{font-size:13px;color:#555;margin-bottom:7px}}
.chart-svg{{width:100%;height:auto;display:block;background:#fff}}
.axis{{stroke:#222;stroke-width:1.2}}
.grid{{stroke:#ddd;stroke-width:1}}
.tick{{font-size:12px;fill:#555}}
.label{{font-size:13px;fill:#333}}
.ref{{stroke:#777;stroke-dasharray:3 5;stroke-width:1.5;fill:none}}
.curve{{fill:none;stroke-width:1.8;opacity:.34;vector-effect:non-scaling-stroke}}
.curve.prototype{{stroke-width:2.6;opacity:.85;stroke-dasharray:7 5}}
.curve-hit{{fill:none;stroke:transparent;stroke-width:12;pointer-events:stroke}}
.foldback{{stroke:#d62728;stroke-width:2.5;opacity:.35;fill:none;vector-effect:non-scaling-stroke}}
.chart-card.hovering .curve{{opacity:.07;stroke-width:1.1}}
.chart-card.hovering .foldback{{opacity:.08}}
.chart-card .active{{opacity:1;stroke-width:4.2}}
.chart-card .active.prototype{{stroke-width:4.8}}
.chart-card .foldback.active{{opacity:1;stroke-width:5}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 18px;font-size:13px;color:#444;margin-top:8px}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.active-label{{font-size:13px;background:#f5f5f5;border:1px solid #ddd;padding:7px 9px;margin-top:9px;min-height:18px}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
th,td{{border:1px solid #ddd;padding:7px 8px;text-align:left}}
th{{background:#f0f0f0}}
@media(max-width:900px){{.grid2,.metrics{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>Nearest Progress and Distance: Why the Curves Move Back and Forth</h1>
<p class="lead">This summary collects the current evidence for the nearest-baseline progress and distance roundtrip behavior. The key point is that nearest progress is an argmin assignment on a curved high-dimensional baseline, not a biological time variable. When two baseline locations have nearly equal distance to the same target point, a small movement in feature space can switch the nearest winner.</p>

<section class="panel">
<h2>Calculation</h2>
<p>For a target curve point and a fixed baseline curve, the current nearest-baseline analysis uses:</p>
<div class="formula">x_i = C_target(t_i)
B(u) = C_baseline(u), u in {{0%, 1%, ..., 100%}}
u_i = argmin_u || x_i - B(u) ||_2
d_i = || x_i - B(u_i) ||_2

foldback if u_i &lt; u_(i-1)
competition if (D(second_nonlocal) - D(best)) / D(best) &lt;= 0.10</div>
<p>The x-axis of the new nearest-baseline curve is therefore <b>nearest baseline progress</b> u_i. The y-axis is <b>nearest vector length</b> d_i. Neither axis is forced to be monotonic.</p>
</section>

<section class="panel">
<h2>Current Evidence</h2>
<div class="metrics" id="metrics"></div>
<div class="grid2">
<div>
<h3>What we have verified</h3>
<ul>
<li>Backward progress jumps are present in the CSV data, not just in the plotted image.</li>
<li>Nearest6 curves have 272 foldback events across 70 of 72 curves.</li>
<li>44 nearest6 foldbacks are at least 10% backward in baseline progress.</li>
<li>218 events are both foldback events and candidate-competition events under the 10% relative-gap rule.</li>
<li>Re-anchoring each curve at its own start reduces foldbacks from 299 to 199 in the combined prototype + nearest6 diagnostic, but it does not remove them.</li>
</ul>
</div>
<div>
<h3>Current interpretation</h3>
<ul>
<li>Progress foldback does not mean the observed smile physically moves backward in time.</li>
<li>It means the target point has crossed a nearest-neighbor decision boundary between baseline regions.</li>
<li>Distance can also rise and fall because it measures distance to the currently selected nearest baseline point, not smile intensity.</li>
<li>Baseline subtraction reduces identity information, but fc7 still contains pose, crop, lighting, and expression-identity interaction effects.</li>
<li>The baseline prototype trajectory itself can be curved, so different baseline progress locations can be close in high-dimensional feature space.</li>
</ul>
</div>
</div>
</section>

<section class="panel">
<h2>The Two Core Interactive Charts</h2>
<p>Use these two charts together. The first chart shows where progress folds backward. The second chart shows whether the nearest-vector distance increases, decreases, or stays close when that assignment changes.</p>
<div class="controls">
<label>method <select id="methodSelect"></select></label>
<label>baseline <select id="baselineSelect"></select></label>
<label><input type="checkbox" data-source="nearest6" checked> nearest6</label>
<label><input type="checkbox" data-source="prototype" checked> prototype</label>
<label><input type="checkbox" data-target="polite" checked> polite</label>
<label><input type="checkbox" data-target="truesmile" checked> truesmile</label>
<label><input type="checkbox" data-target="ambiguous" checked> ambiguous</label>
</div>
<div class="chart-card" id="progressCard">
<div class="chart-title">1. Nearest baseline progress over target stage</div>
<div class="chart-caption">Red segments mark foldback: current nearest progress is lower than the previous stage.</div>
<div id="progressChart"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
<div class="active-label" id="progressLabel">No curve selected</div>
</div>
<div class="chart-card" id="distanceCard">
<div class="chart-title">2. Nearest distance over target stage</div>
<div class="chart-caption">This shows the length of the nearest vector after the nearest baseline point has been selected.</div>
<div id="distanceChart"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
<div class="active-label" id="distanceLabel">No curve selected</div>
</div>
</section>

<section class="panel">
<h2>Worst Foldback Examples</h2>
<p>These are the largest negative progress changes in the combined prototype + nearest6 table.</p>
<div id="exampleTable"></div>
</section>

<section class="panel">
<h2>Conclusion</h2>
<p>The current evidence supports this explanation: progress and distance move back and forth because we are doing global nearest-point assignment on high-dimensional trajectories. When the distance landscape has two close candidate regions, the argmin can switch abruptly. This is an instability of the nearest mapping, not direct evidence that the facial expression itself reversed in time.</p>
<p>The next defensible step is not to force monotonic progress, but to report both the raw nearest mapping and a continuity-aware diagnostic side by side. If a continuity prior is used, it should be explicitly labeled as an analysis assumption, not silently replacing the raw nearest result.</p>
</section>
</main>
<script>
const DATA = {data_json};
const COLORS = {{polite:'#1f77b4', truesmile:'#2ca02c', ambiguous:'#ff7f0e'}};
const METHODS = {json.dumps(METHODS)};
const BASELINES = {json.dumps(BASELINE_CLASSES)};

const methodSelect = document.getElementById('methodSelect');
const baselineSelect = document.getElementById('baselineSelect');
METHODS.forEach(v => methodSelect.add(new Option(v, v)));
BASELINES.forEach(v => baselineSelect.add(new Option(v, v)));

function fmt(v, d=3) {{ return Number(v).toFixed(d); }}
function selectedValues(selector, attr) {{
  return new Set(Array.from(document.querySelectorAll(selector + ':checked')).map(el => el.dataset[attr]));
}}

function updateMetrics() {{
  const s = DATA.summary;
  const items = [
    ['nearest6 foldbacks', s.nearest6.foldback_events],
    ['nearest6 curves with foldback', `${{s.nearest6.curves_with_foldback}} / ${{s.nearest6.curve_count}}`],
    ['severe nearest6 foldbacks', s.nearest6.severe_foldback_events],
    ['competition + foldback', s.competitionBacktrack.event_count],
    ['distance up transitions', s.nearest6.distance_up_events],
    ['distance down transitions', s.nearest6.distance_down_events],
    ['foldback + distance up', s.nearest6.foldback_distance_up_events],
    ['foldback + distance down', s.nearest6.foldback_distance_down_events],
  ];
  document.getElementById('metrics').innerHTML = items.map(([k,v]) => `<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`).join('');
}}

function filteredLines() {{
  const sources = selectedValues('[data-source]', 'source');
  const targets = selectedValues('[data-target]', 'target');
  return DATA.lines.filter(line =>
    line.method === methodSelect.value &&
    line.baselineClass === baselineSelect.value &&
    sources.has(line.sourceType) &&
    targets.has(line.targetClass)
  );
}}

function pointPath(points, xScale, yScale, yKey) {{
  return points.map((p, i) => `${{i === 0 ? 'M' : 'L'}}${{xScale(p.stage).toFixed(2)}},${{yScale(p[yKey]).toFixed(2)}}`).join(' ');
}}

function segmentPath(segment, xScale, yScale, yKey) {{
  const y0 = yKey === 'progress' ? segment.progress0 : segment.distance0;
  const y1 = yKey === 'progress' ? segment.progress1 : segment.distance1;
  return `M${{xScale(segment.stage0).toFixed(2)}},${{yScale(y0).toFixed(2)}} L${{xScale(segment.stage1).toFixed(2)}},${{yScale(y1).toFixed(2)}}`;
}}

function renderChart(containerId, cardId, labelId, yKey, yLabel, fixedY100) {{
  const lines = filteredLines();
  const width = 1120, height = 470;
  const left = 70, right = 28, top = 36, bottom = 58;
  const plotW = width - left - right, plotH = height - top - bottom;
  const yMaxRaw = Math.max(1, ...lines.flatMap(line => line.points.map(p => p[yKey])));
  const yMax = fixedY100 ? 100 : yMaxRaw * 1.08;
  const xScale = x => left + (x / 100) * plotW;
  const yScale = y => top + plotH - (y / yMax) * plotH;
  const yTicks = fixedY100 ? [0,20,40,60,80,100] : [0, yMax/4, yMax/2, yMax*3/4, yMax];

  let html = `<svg class="chart-svg" viewBox="0 0 ${{width}} ${{height}}">`;
  [0,20,40,60,80,100].forEach(t => {{
    const x = xScale(t);
    html += `<line class="grid" x1="${{x}}" y1="${{top}}" x2="${{x}}" y2="${{top+plotH}}"/>`;
    html += `<text class="tick" x="${{x}}" y="${{top+plotH+23}}" text-anchor="middle">${{t}}</text>`;
  }});
  yTicks.forEach(t => {{
    const y = yScale(t);
    html += `<line class="grid" x1="${{left}}" y1="${{y}}" x2="${{left+plotW}}" y2="${{y}}"/>`;
    html += `<text class="tick" x="${{left-10}}" y="${{y+4}}" text-anchor="end">${{fixedY100 ? fmt(t,0) : fmt(t,2)}}</text>`;
  }});
  html += `<line class="axis" x1="${{left}}" y1="${{top+plotH}}" x2="${{left+plotW}}" y2="${{top+plotH}}"/>`;
  html += `<line class="axis" x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+plotH}}"/>`;
  html += `<text class="label" x="${{left+plotW/2}}" y="${{height-18}}" text-anchor="middle">target stage on C_target (%)</text>`;
  html += `<text class="label" x="20" y="${{top+plotH/2}}" text-anchor="middle" transform="rotate(-90,20,${{top+plotH/2}})">${{yLabel}}</text>`;
  if (fixedY100) {{
    html += `<path class="ref" d="M${{xScale(0)}},${{yScale(0)}} L${{xScale(100)}},${{yScale(100)}}"/>`;
  }}

  lines.forEach(line => {{
    const color = COLORS[line.targetClass];
    const dashClass = line.sourceType === 'prototype' ? ' prototype' : '';
    const d = pointPath(line.points, xScale, yScale, yKey);
    html += `<g data-key="${{line.key}}" data-label="${{line.label}}" data-target="${{line.targetClass}}">`;
    html += `<path class="curve${{dashClass}}" data-role="curve" d="${{d}}" stroke="${{color}}"/>`;
    html += `<path class="curve-hit" data-role="hit" d="${{d}}"/>`;
    line.foldbacks.forEach(seg => {{
      html += `<path class="foldback" data-role="foldback" d="${{segmentPath(seg, xScale, yScale, yKey)}}"/>`;
    }});
    html += `</g>`;
  }});
  html += `</svg>`;
  document.getElementById(containerId).innerHTML = html;

  const card = document.getElementById(cardId);
  const label = document.getElementById(labelId);
  card.querySelectorAll('g[data-key]').forEach(g => {{
    g.addEventListener('mouseenter', () => setActive(g.dataset.key, g.dataset.label));
    g.addEventListener('mouseleave', clearActive);
  }});
}}

function setActive(key, label) {{
  document.querySelectorAll('.chart-card').forEach(card => card.classList.add('hovering'));
  document.querySelectorAll('[data-role="curve"], [data-role="foldback"]').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('g[data-key]').forEach(g => {{
    if (g.dataset.key === key) {{
      g.querySelectorAll('[data-role="curve"], [data-role="foldback"]').forEach(el => el.classList.add('active'));
    }}
  }});
  document.getElementById('progressLabel').textContent = label;
  document.getElementById('distanceLabel').textContent = label;
}}

function clearActive() {{
  document.querySelectorAll('.chart-card').forEach(card => card.classList.remove('hovering'));
  document.querySelectorAll('[data-role="curve"], [data-role="foldback"]').forEach(el => el.classList.remove('active'));
  document.getElementById('progressLabel').textContent = 'No curve selected';
  document.getElementById('distanceLabel').textContent = 'No curve selected';
}}

function renderExamples() {{
  const rows = DATA.summary.all.worst_examples;
  document.getElementById('exampleTable').innerHTML = `<table><tr><th>source</th><th>method</th><th>baseline</th><th>target</th><th>seq/rank</th><th>stage</th><th>progress</th><th>distance</th></tr>` +
    rows.map(e => `<tr><td>${{e.source_type}}</td><td>${{e.method}}</td><td>${{e.baseline_class}}</td><td>${{e.target_class}}</td><td>${{e.sequence_id}} / ${{e.rank || '-'}}</td><td>${{fmt(e.stage_from,0)}}→${{fmt(e.stage_to,0)}}%</td><td>${{fmt(e.progress_from,0)}}→${{fmt(e.progress_to,0)}}%</td><td>${{fmt(e.distance_from,4)}}→${{fmt(e.distance_to,4)}}</td></tr>`).join('') +
    `</table>`;
}}

function renderAll() {{
  renderChart('progressChart', 'progressCard', 'progressLabel', 'progress', 'nearest baseline progress (%)', true);
  renderChart('distanceChart', 'distanceCard', 'distanceLabel', 'distance', 'nearest vector length d_i', false);
}}

document.querySelectorAll('select,input').forEach(el => el.addEventListener('change', renderAll));
updateMetrics();
renderExamples();
renderAll();
</script>
</body>
</html>
"""

    def run(self) -> Path:
        rows = self.load_rows()
        lines = self.build_lines(rows)
        summary = self.build_summary(rows)
        output = self.cfg.output_root / "progress_distance_roundtrip_summary.html"
        output.write_text(self.render_html(lines, summary), encoding="utf-8")
        return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a summary report for nearest progress and distance roundtrip behavior."
    )
    parser.add_argument("--nearest_output_root", default=str(ProgressDistanceSummaryConfig.nearest_output_root))
    parser.add_argument("--backtrack_output_root", default=str(ProgressDistanceSummaryConfig.backtrack_output_root))
    parser.add_argument("--output_root", default=str(ProgressDistanceSummaryConfig.output_root))
    return parser


def main() -> None:
    cfg = ProgressDistanceSummaryConfig.from_args(build_arg_parser().parse_args())
    output = ProgressDistanceSummaryReport(cfg).run()
    print(f"progress_distance_summary_html: {output}")


if __name__ == "__main__":
    main()
