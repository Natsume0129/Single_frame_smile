from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from run_nearest_competition_frame_report import CompetitionFrameConfig, CompetitionFrameReport
from run_nearest_progress_distance_summary_report import (
    BASELINE_CLASSES,
    CLASS_NAMES,
    METHODS,
    ProgressDistanceSummaryConfig,
    ProgressDistanceSummaryReport,
)


@dataclass
class NewCurveCompetitionSummaryConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    nearest_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    backtrack_output_root: Path = Path(
        r"E:\Matsuda_data\3-10meeting\nearest_baseline_competition_backtrack_frames"
    )
    output_root: Path = Path(
        r"E:\Matsuda_data\3-10meeting\nearest_baseline_new_curve_competition_summary"
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "NewCurveCompetitionSummaryConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            nearest_output_root=Path(args.nearest_output_root),
            backtrack_output_root=Path(args.backtrack_output_root),
            output_root=Path(args.output_root),
        )


class NewCurveCompetitionSummaryReport:
    def __init__(self, cfg: NewCurveCompetitionSummaryConfig):
        self.cfg = cfg
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def to_json(data: object) -> str:
        def clean(value: object) -> object:
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, tuple):
                return [clean(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        return json.dumps(clean(data), ensure_ascii=False, allow_nan=False)

    @staticmethod
    def line_key(row: dict[str, str]) -> str:
        return "|".join(ProgressDistanceSummaryReport.group_key(row))

    def frame_report(self) -> CompetitionFrameReport:
        return CompetitionFrameReport(
            CompetitionFrameConfig(
                analysis_input_root=self.cfg.analysis_input_root,
                projection_output_root=self.cfg.projection_output_root,
                nearest_output_root=self.cfg.nearest_output_root,
                output_root=self.cfg.output_root,
            )
        )

    @staticmethod
    def second_nonlocal_index(distances: np.ndarray, best_idx: int) -> int:
        for idx in np.argsort(distances):
            idx = int(idx)
            if abs(float(idx) - float(best_idx)) > 10.0:
                return idx
        raise RuntimeError("Could not find a nonlocal second candidate.")

    def load_target_curve(
        self,
        helper: CompetitionFrameReport,
        source_type: str,
        method: str,
        target_class: str,
        sequence_id: str,
    ) -> np.ndarray:
        if source_type == "prototype":
            return np.load(helper.prototype_path(method, target_class), allow_pickle=False).astype(np.float64)
        return helper.load_target_curve(target_class, sequence_id)

    def copy_asset(self, helper: CompetitionFrameReport, source: Path, subdir: str) -> str:
        destination = helper.asset_root / subdir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        return destination.relative_to(self.cfg.output_root).as_posix()

    def target_asset(
        self,
        helper: CompetitionFrameReport,
        source_type: str,
        method: str,
        target_class: str,
        sequence_id: str,
        stage_percent: float,
        methodb_meta: dict,
    ) -> tuple[int | None, str | None, str | None]:
        if source_type == "nearest6":
            idx, src = helper.target_asset(target_class, sequence_id, stage_percent)
            return idx, src, None
        if method != "methodB":
            return None, None, "methodA target prototype is a median trajectory and has no real frame."
        medoid_sequence_id = str(methodb_meta[target_class]["sequence_id"])
        idx = helper.frame_index(stage_percent)
        source = helper.normalized_frames_dir(target_class, medoid_sequence_id) / f"{idx:03d}.png"
        src = self.copy_asset(helper, source, f"target/prototype_{method}_{target_class}_{medoid_sequence_id}")
        return idx, src, None

    def build_inspector_details(self, rows: list[dict[str, str]]) -> dict[str, dict]:
        helper = self.frame_report()
        methodb_meta = helper.load_methodb_meta()
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[self.line_key(row)].append(row)

        details: dict[str, dict] = {}
        for key, group_rows in grouped.items():
            ordered = sorted(group_rows, key=lambda r: float(r["target_stage_percent"]))
            first = ordered[0]
            method = first["method"]
            baseline_class = first["baseline_class"]
            target_class = first["target_class"]
            source_type = first["source_type"]
            sequence_id = first["sequence_id"]
            rank = first["rank"]
            baseline_samples = helper.load_baseline_samples(method, baseline_class)
            target_curve = self.load_target_curve(helper, source_type, method, target_class, sequence_id)

            stages: list[dict] = []
            previous_progress: float | None = None
            for row in ordered:
                stage = float(row["target_stage_percent"])
                target_point = helper.point_at_percent(target_curve, stage)
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
                return_amount = (
                    previous_progress - float(best_idx)
                    if previous_progress is not None
                    else 0.0
                )
                is_return = previous_progress is not None and return_amount >= 10.0
                is_competition = relative_gap <= 0.10

                target_idx, target_src, target_note = self.target_asset(
                    helper,
                    source_type,
                    method,
                    target_class,
                    sequence_id,
                    stage,
                    methodb_meta,
                )
                nearest_idx, nearest_src, nearest_note = helper.baseline_asset(
                    method,
                    baseline_class,
                    float(best_idx),
                    methodb_meta,
                )
                second_frame_idx, second_src, second_note = helper.baseline_asset(
                    method,
                    baseline_class,
                    float(second_idx),
                    methodb_meta,
                )

                stages.append(
                    {
                        "stage": stage,
                        "targetFrameIndex": target_idx,
                        "targetFrameSrc": target_src,
                        "targetNote": target_note,
                        "nearestProgress": float(best_idx),
                        "nearestDistance": best_distance,
                        "nearestFrameIndex": nearest_idx,
                        "nearestFrameSrc": nearest_src,
                        "nearestNote": nearest_note,
                        "secondProgress": float(second_idx),
                        "secondDistance": second_distance,
                        "secondFrameIndex": second_frame_idx,
                        "secondFrameSrc": second_src,
                        "secondNote": second_note,
                        "relativeGap": float(relative_gap),
                        "isCompetition": is_competition,
                        "isReturn": is_return,
                        "returnAmount": float(return_amount),
                        "showCompetitionExtra": bool(is_return and is_competition),
                    }
                )
                previous_progress = float(best_idx)

            details[key] = {
                "sourceType": source_type,
                "method": method,
                "baselineClass": baseline_class,
                "targetClass": target_class,
                "sequenceId": sequence_id,
                "rank": rank,
                "stageCount": len(stages),
                "returnCount": sum(1 for stage in stages if stage["isReturn"]),
                "returnCompetitionCount": sum(1 for stage in stages if stage["showCompetitionExtra"]),
                "stages": stages,
            }
        return details

    @staticmethod
    def foldbacks(points: list[dict], threshold: float = 10.0) -> list[dict]:
        events = []
        for previous, current in zip(points, points[1:]):
            delta = float(current["progress"]) - float(previous["progress"])
            if delta <= -threshold:
                events.append(
                    {
                        "stage0": previous["stage"],
                        "stage1": current["stage"],
                        "progress0": previous["progress"],
                        "progress1": current["progress"],
                        "deltaProgress": delta,
                    }
                )
        return events

    def build_corrected_lines(self, lines: list[dict], inspector_details: dict[str, dict]) -> tuple[list[dict], dict]:
        corrected_lines = []
        replacement_count = 0
        raw_severe_foldbacks = 0
        corrected_severe_foldbacks = 0
        by_method_baseline: dict[str, int] = defaultdict(int)

        for line in lines:
            detail = inspector_details[line["key"]]
            corrected_points = []
            replacements = []
            previous_progress: float | None = None
            for stage in detail["stages"]:
                raw_progress = float(stage["nearestProgress"])
                raw_distance = float(stage["nearestDistance"])
                progress = raw_progress
                distance = raw_distance
                was_replaced = False
                if (
                    previous_progress is not None
                    and previous_progress - raw_progress >= 10.0
                    and stage["isCompetition"]
                ):
                    progress = float(stage["secondProgress"])
                    distance = float(stage["secondDistance"])
                    was_replaced = True
                    replacement_count += 1
                    by_method_baseline[f"{line['method']}|{line['baselineClass']}"] += 1
                    replacements.append(
                        {
                            "stage": float(stage["stage"]),
                            "rawProgress": raw_progress,
                            "replacementProgress": progress,
                            "rawDistance": raw_distance,
                            "replacementDistance": distance,
                            "relativeGap": float(stage["relativeGap"]),
                        }
                    )

                corrected_points.append(
                    {
                        "stage": float(stage["stage"]),
                        "progress": progress,
                        "distance": distance,
                        "rawProgress": raw_progress,
                        "rawDistance": raw_distance,
                        "corrected": was_replaced,
                    }
                )
                previous_progress = progress

            raw_foldbacks = self.foldbacks(line["points"])
            corrected_foldbacks = self.foldbacks(corrected_points)
            raw_severe_foldbacks += len(raw_foldbacks)
            corrected_severe_foldbacks += len(corrected_foldbacks)
            corrected_line = dict(line)
            corrected_line["points"] = corrected_points
            corrected_line["foldbacks"] = corrected_foldbacks
            corrected_line["rawSevereFoldbackCount"] = len(raw_foldbacks)
            corrected_line["correctedSevereFoldbackCount"] = len(corrected_foldbacks)
            corrected_line["replacementCount"] = len(replacements)
            corrected_line["replacements"] = replacements
            corrected_lines.append(corrected_line)

        summary = {
            "replacementCount": replacement_count,
            "rawSevereFoldbackCount": raw_severe_foldbacks,
            "correctedSevereFoldbackCount": corrected_severe_foldbacks,
            "byMethodBaseline": dict(by_method_baseline),
            "rule": "If raw best progress returns by >=10% from the previous corrected progress and relative gap <=10%, replace that point with the second nonlocal candidate.",
        }
        return corrected_lines, summary

    def build_payload(self) -> dict:
        summary_report = ProgressDistanceSummaryReport(
            ProgressDistanceSummaryConfig(
                nearest_output_root=self.cfg.nearest_output_root,
                backtrack_output_root=self.cfg.backtrack_output_root,
                output_root=self.cfg.output_root,
            )
        )
        rows = summary_report.load_rows()
        lines = summary_report.build_lines(rows)
        curve_summary = summary_report.build_summary(rows)
        inspector_details = self.build_inspector_details(rows)
        corrected_lines, corrected_summary = self.build_corrected_lines(lines, inspector_details)
        return {
            "lines": lines,
            "correctedLines": corrected_lines,
            "curveSummary": curve_summary,
            "correctedSummary": corrected_summary,
            "inspectorDetails": inspector_details,
            "methods": METHODS,
            "baselineClasses": BASELINE_CLASSES,
            "classNames": CLASS_NAMES,
        }

    def render_html(self, payload: dict) -> str:
        data_json = self.to_json(payload)
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Nearest-Baseline New Curves and Selected Curve Inspector</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.55}}
main{{max-width:1340px;margin:0 auto}}
h1{{font-size:29px;margin:0 0 8px}}
h2{{font-size:21px;margin:30px 0 10px;border-bottom:1px solid #ddd;padding-bottom:6px}}
h3{{font-size:16px;margin:0 0 7px}}
p{{max-width:1060px}}
.lead{{font-size:15px;color:#333}}
.panel{{background:white;border:1px solid #ddd;padding:15px 17px;margin:16px 0}}
.note{{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}}
.formula{{font-family:Consolas,monospace;background:#f6f6f6;border:1px solid #ddd;padding:10px 12px;white-space:pre-wrap}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}}
.metric{{background:#fff;border:1px solid #ddd;padding:10px}}
.metric b{{font-size:21px;display:block}}
.controls{{position:sticky;top:0;z-index:10;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0;margin:14px 0 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}}
label{{font-size:13px;display:inline-flex;gap:6px;align-items:center}}
input{{font-size:13px}}
.chart-card{{background:#fff;border:1px solid #ddd;padding:14px 16px;margin:0 0 22px}}
.chart-title{{font-weight:700;margin-bottom:4px}}
.chart-caption{{font-size:13px;color:#555;margin-bottom:7px}}
.chart-svg{{width:100%;height:auto;display:block;background:#fff}}
.axis{{stroke:#222;stroke-width:1.2}}
.grid{{stroke:#ddd;stroke-width:1}}
.tick{{font-size:12px;fill:#555}}
.label{{font-size:13px;fill:#333}}
.curve{{fill:none;stroke-width:2.3;opacity:.62;vector-effect:non-scaling-stroke;pointer-events:stroke;cursor:pointer}}
.curve.prototype{{stroke-width:3.1;opacity:.95;stroke-dasharray:7 5}}
.curve-hit{{fill:none;stroke:#000;stroke-opacity:.001;stroke-width:14;pointer-events:stroke;cursor:pointer}}
.chart-card.hovering .curve{{opacity:.07;stroke-width:1.1}}
.chart-card.hovering .curve.active{{opacity:1;stroke-width:4.2}}
.chart-card.hovering .curve.prototype.active{{opacity:1;stroke-width:4.8}}
.curve-hit{{cursor:pointer}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 18px;font-size:13px;color:#444;margin-top:8px}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.active-label{{font-size:13px;background:#f5f5f5;border:1px solid #ddd;padding:7px 9px;margin-top:9px;min-height:18px}}
.stage-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:12px;margin-top:12px}}
.stage-card{{background:#fff;border:1px solid #ddd;padding:10px}}
.stage-card.return{{border-color:#b88700;background:#fffdf6}}
.stage-card.competition{{border-color:#9a4d00}}
.stage-card h3{{font-size:14px;line-height:1.35;margin:0 0 8px}}
.stage-images{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.imgbox{{border:1px solid #ddd;background:#f4f4f4;padding:6px;min-width:0}}
.imgbox h4{{font-size:12px;margin:0 0 5px;line-height:1.3}}
.imgbox img{{width:100%;height:150px;object-fit:contain;background:#eee;display:block}}
.placeholder{{height:150px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px;color:#555;background:#eee;padding:8px}}
.competition-extra{{margin-top:8px;border-top:1px solid #ddd;padding-top:8px}}
.competition-extra .stage-images{{grid-template-columns:1fr 1fr 1fr}}
.badge{{display:inline-block;border:1px solid #aaa;background:#f6f6f6;padding:2px 6px;margin-left:5px;font-size:11px;color:#333}}
.badge.return{{border-color:#b88700;background:#fff7df}}
.badge.competition{{border-color:#9a4d00;background:#ffe8c8}}
.inspector-facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}}
.fact{{background:#f7f7f7;border:1px solid #ddd;padding:8px;font-size:13px}}
.selected-chart{{border:1px solid #ddd;background:#fff;padding:10px;margin:12px 0}}
.selected-chart .chart-svg{{max-height:430px}}
.meta{{font-size:12px;color:#444;margin-top:7px}}
ul{{padding-left:20px}}
li{{margin:5px 0}}
@media(max-width:900px){{.metrics,.inspector-facts{{grid-template-columns:1fr 1fr}}.stage-grid{{grid-template-columns:1fr}}.competition-extra .stage-images{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>Nearest-Baseline New Curves and Selected Curve Inspector</h1>
<p class="lead">This page focuses on the actual new-curve plots: x = nearest baseline progress, y = nearest vector length. Click any curve to inspect every target stage and its nearest baseline point. Return points are marked when nearest progress moves backward by at least 10% from the previous stage.</p>

<section class="panel">
<h2>Correct New-Curve Definition</h2>
<div class="formula">x_i = C_target(t_i)
B(u) = C_baseline(u), u in {{0%, 1%, ..., 100%}}
u_i = argmin_u || x_i - B(u) ||_2
d_i = || x_i - B(u_i) ||_2

new-curve point = (u_i, d_i)</div>
<p>The four charts below are the same type as <b>Nearest-baseline new curve</b> in the previous interactive report. Lines connect target stages in time order, but the horizontal coordinate is nearest baseline progress, not target time.</p>
<div class="note">MethodA baseline is a median prototype, so there is no real baseline frame to inspect. MethodB baseline is a medoid real sequence, so MethodB can be checked with actual frame examples.</div>
</section>

<section class="panel">
<h2>Current Numbers</h2>
<div class="metrics" id="metrics"></div>
<ul>
<li>Competition means the best baseline candidate and a nonlocal second candidate differ by no more than 10% relative distance.</li>
<li>When such competition occurs, a small target movement can switch the nearest winner to another baseline progress region.</li>
<li>Similar-looking new curves should therefore be treated as a hypothesis, not proof. Click a curve and inspect the stage-by-stage target-nearest image pairs below.</li>
</ul>
</section>

<section class="panel">
<h2>Four New-Curve Charts</h2>
<div class="controls">
<label><input type="checkbox" data-source="nearest6" checked> nearest6</label>
<label><input type="checkbox" data-source="prototype" checked> prototype</label>
<label><input type="checkbox" data-target="polite" checked> polite</label>
<label><input type="checkbox" data-target="truesmile" checked> truesmile</label>
<label><input type="checkbox" data-target="ambiguous" checked> ambiguous</label>
</div>
<div id="charts"></div>
</section>

<section class="panel">
<h2>Selected Curve Inspector</h2>
<p>Select one curve from the dropdown below. This section will list every target stage, its nearest baseline progress, and the target-nearest image pair. If a stage is a return point and also has candidate competition, the competing second candidate is shown as an extra image.</p>
<div class="controls">
<label>curve <select id="curveSelect"><option value="">Select a curve</option></select></label>
</div>
<div id="inspector"><div class="note">No curve selected yet.</div></div>
</section>

<section class="panel">
<h2>Working Conclusion</h2>
<p>The curve similarity is real as a feature-space observation, but it is not enough by itself. The strongest current explanation for the back-and-forth behavior is nearest-candidate competition in fc7 feature space. MethodB is the most interpretable branch because both baseline candidates correspond to real frames; MethodA can still be inspected numerically, but its prototype baseline has no real frame.</p>
</section>

<section class="panel">
<h2>What If Return + Competition Uses the Second Candidate?</h2>
<p>This added module keeps the raw analysis above unchanged. It only asks what the four new-curve charts would look like if a return point with candidate competition is replaced by the second nonlocal candidate.</p>
<div class="formula">for target stages in temporal order:
    raw point = (u_i, d_i)
    if previous_corrected_progress - u_i &gt;= 10%
       and (d_second - d_i) / d_i &lt;= 10%:
           corrected point = (u_second, d_second)
    else:
           corrected point = raw point</div>
<div class="note">This is a diagnostic correction, not a replacement for the raw nearest-baseline result. It adds a continuity assumption after competition has already been detected.</div>
<div class="metrics" id="correctedMetrics"></div>
<div id="correctedCharts"></div>
</section>
</main>
<script>
const DATA = {data_json};
const COLORS = {{polite:'#1f77b4', truesmile:'#2ca02c', ambiguous:'#ff7f0e'}};
const CHARTS = [
  {{method:'methodB', baseline:'truesmile', title:'Nearest-baseline new curve | methodB | baseline=truesmile'}},
  {{method:'methodB', baseline:'polite', title:'Nearest-baseline new curve | methodB | baseline=polite'}},
  {{method:'methodA', baseline:'truesmile', title:'Nearest-baseline new curve | methodA | baseline=truesmile'}},
  {{method:'methodA', baseline:'polite', title:'Nearest-baseline new curve | methodA | baseline=polite'}},
];

function fmt(v, d=3) {{
  const n = Number(v);
  if (v === null || v === undefined || !Number.isFinite(n)) return 'N/A';
  return n.toFixed(d);
}}
function selectedValues(selector, attr) {{
  return new Set(Array.from(document.querySelectorAll(selector + ':checked')).map(el => el.dataset[attr]));
}}
function frameLabel(idx) {{ return idx === null ? 'N/A' : String(idx).padStart(3,'0'); }}

function showMetrics() {{
  const nearest6 = DATA.curveSummary.nearest6;
  const competition = DATA.curveSummary.competitionBacktrack;
  const items = [
    ['nearest6 curves', nearest6.curve_count],
    ['nearest6 foldback events', nearest6.foldback_events],
    ['competition + foldback', competition.event_count],
    ['inspector curves', Object.keys(DATA.inspectorDetails).length],
    ['methodB comp+foldback', competition.by_method.methodB || 0],
    ['methodA comp+foldback', competition.by_method.methodA || 0],
    ['competition threshold', '<=10%'],
    ['return threshold', '>=10%'],
  ];
  document.getElementById('metrics').innerHTML = items.map(([k,v]) => `<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`).join('');
}}

function filteredLines(method, baseline) {{
  const sources = selectedValues('[data-source]', 'source');
  const targets = selectedValues('[data-target]', 'target');
  return DATA.lines.filter(line =>
    line.method === method &&
    line.baselineClass === baseline &&
    sources.has(line.sourceType) &&
    targets.has(line.targetClass)
  );
}}

function filteredCorrectedLines(method, baseline) {{
  const sources = selectedValues('[data-source]', 'source');
  const targets = selectedValues('[data-target]', 'target');
  return DATA.correctedLines.filter(line =>
    line.method === method &&
    line.baselineClass === baseline &&
    sources.has(line.sourceType) &&
    targets.has(line.targetClass)
  );
}}

function pathFromPoints(points, xScale, yScale) {{
  return points.map((p, i) => `${{i === 0 ? 'M' : 'L'}}${{xScale(p.progress).toFixed(2)}},${{yScale(p.distance).toFixed(2)}}`).join(' ');
}}
function renderNewCurveChart(spec, index, lineOverride=null, extraCaption='') {{
  const lines = lineOverride || filteredLines(spec.method, spec.baseline);
  const width = 1120, height = 500;
  const left = 70, right = 28, top = 36, bottom = 58;
  const plotW = width - left - right, plotH = height - top - bottom;
  const yMax = Math.max(1, ...lines.flatMap(line => line.points.map(p => p.distance))) * 1.08;
  const xScale = x => left + (x / 100) * plotW;
  const yScale = y => top + plotH - (y / yMax) * plotH;
  const yTicks = [0, yMax/4, yMax/2, yMax*3/4, yMax];
  let svg = `<svg class="chart-svg" viewBox="0 0 ${{width}} ${{height}}">`;
  [0,20,40,60,80,100].forEach(t => {{
    const x = xScale(t);
    svg += `<line class="grid" x1="${{x}}" y1="${{top}}" x2="${{x}}" y2="${{top+plotH}}"/>`;
    svg += `<text class="tick" x="${{x}}" y="${{top+plotH+23}}" text-anchor="middle">${{t}}</text>`;
  }});
  yTicks.forEach(t => {{
    const y = yScale(t);
    svg += `<line class="grid" x1="${{left}}" y1="${{y}}" x2="${{left+plotW}}" y2="${{y}}"/>`;
    svg += `<text class="tick" x="${{left-10}}" y="${{y+4}}" text-anchor="end">${{fmt(t,2)}}</text>`;
  }});
  svg += `<line class="axis" x1="${{left}}" y1="${{top+plotH}}" x2="${{left+plotW}}" y2="${{top+plotH}}"/>`;
  svg += `<line class="axis" x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+plotH}}"/>`;
  svg += `<text class="label" x="${{left+plotW/2}}" y="${{height-18}}" text-anchor="middle">nearest baseline progress (%)</text>`;
  svg += `<text class="label" x="20" y="${{top+plotH/2}}" text-anchor="middle" transform="rotate(-90,20,${{top+plotH/2}})">nearest vector length ||target - nearest||_2</text>`;

  lines.forEach(line => {{
    const color = COLORS[line.targetClass];
    const proto = line.sourceType === 'prototype' ? ' prototype' : '';
    const d = pathFromPoints(line.points, xScale, yScale);
    svg += `<g data-key="${{line.key}}" data-label="${{line.label}}" data-target="${{line.targetClass}}">`;
    svg += `<path class="curve${{proto}}" data-role="curve" data-key="${{line.key}}" data-label="${{line.label}}" d="${{d}}" stroke="${{color}}"/>`;
    svg += `<path class="curve-hit" data-role="hit" data-key="${{line.key}}" data-label="${{line.label}}" d="${{d}}"/>`;
    svg += `</g>`;
  }});
  svg += `</svg>`;

  return `<div class="chart-card" id="chartCard${{index}}">
    <div class="chart-title">${{spec.title}}</div>
    <div class="chart-caption">Hover any curve to highlight it. This chart keeps the same clean style as the previous nearest-baseline interactive report.${{extraCaption}}</div>
    <div>${{svg}}</div>
    <div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
    <div class="active-label">No curve selected</div>
  </div>`;
}}

function setActive(key, label) {{
  document.querySelectorAll('.chart-card').forEach(card => card.classList.add('hovering'));
  document.querySelectorAll('[data-role="curve"]').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('g[data-key]').forEach(g => {{
    if (g.dataset.key === key) {{
      g.parentNode.appendChild(g);
      g.querySelectorAll('[data-role="curve"]').forEach(el => el.classList.add('active'));
    }}
  }});
  document.querySelectorAll('.active-label').forEach(el => el.textContent = label);
}}
function clearActive() {{
  document.querySelectorAll('.chart-card').forEach(card => card.classList.remove('hovering'));
  document.querySelectorAll('[data-role="curve"]').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.active-label').forEach(el => el.textContent = 'No curve selected');
}}
function attachChartHover() {{
  document.querySelectorAll('g[data-key]').forEach(g => {{
    g.addEventListener('mouseover', () => setActive(g.dataset.key, g.dataset.label));
    g.addEventListener('mouseout', event => {{
      if (!g.contains(event.relatedTarget)) clearActive();
    }});
    g.addEventListener('click', event => {{
      event.preventDefault();
      event.stopPropagation();
      selectCurve(g.dataset.key, g.dataset.label);
    }});
  }});
  document.querySelectorAll('[data-role="curve"], [data-role="hit"]').forEach(path => {{
    path.addEventListener('click', event => {{
      event.preventDefault();
      event.stopPropagation();
      selectCurve(path.dataset.key, path.dataset.label);
    }});
  }});
  document.querySelectorAll('.chart-card').forEach(card => {{
    card.addEventListener('mouseleave', clearActive);
  }});
}}

function renderCharts() {{
  document.getElementById('charts').innerHTML = CHARTS.map((spec, index) => renderNewCurveChart(spec, index)).join('');
  updateCurveSelect();
}}

function showCorrectedMetrics() {{
  const s = DATA.correctedSummary;
  const items = [
    ['points replaced', s.replacementCount],
    ['raw return>=10% events', s.rawSevereFoldbackCount],
    ['after replacement return>=10%', s.correctedSevereFoldbackCount],
    ['methodA/truesmile', s.byMethodBaseline['methodA|truesmile'] || 0],
    ['methodA/polite', s.byMethodBaseline['methodA|polite'] || 0],
    ['methodB/truesmile', s.byMethodBaseline['methodB|truesmile'] || 0],
    ['methodB/polite', s.byMethodBaseline['methodB|polite'] || 0],
    ['replacement rule', 'return+competition'],
  ];
  document.getElementById('correctedMetrics').innerHTML = items.map(([k,v]) => `<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`).join('');
}}

function renderCorrectedCharts() {{
  const html = CHARTS.map((spec, index) => {{
    const lines = filteredCorrectedLines(spec.method, spec.baseline);
    const title = spec.title.replace('Nearest-baseline new curve', 'Corrected nearest-baseline new curve');
    const replacementCount = lines.reduce((total, line) => total + (line.replacementCount || 0), 0);
    return renderNewCurveChart(
      {{method: spec.method, baseline: spec.baseline, title}},
      `corrected${{index}}`,
      lines,
      ` Corrected points in currently visible lines: ${{replacementCount}}.`
    );
  }}).join('');
  document.getElementById('correctedCharts').innerHTML = html;
  showCorrectedMetrics();
}}

function renderAllCharts() {{
  renderCharts();
  renderCorrectedCharts();
  attachChartHover();
}}

function img(src) {{ return `<img loading="lazy" src="${{src}}" alt="frame">`; }}
function imgOrPlaceholder(src, note) {{
  if (src) return img(src);
  return `<div class="placeholder">${{note || 'No real frame available'}}</div>`;
}}

function badges(stage) {{
  const parts = [];
  if (stage.isReturn) parts.push(`<span class="badge return">return >=10%</span>`);
  if (stage.showCompetitionExtra) parts.push(`<span class="badge competition">competition</span>`);
  if (stage.isReturn && !stage.isCompetition) parts.push(`<span class="badge">return only</span>`);
  return parts.join('');
}}

function stageCard(stage) {{
  const cardClass = `stage-card${{stage.isReturn ? ' return' : ''}}${{stage.showCompetitionExtra ? ' competition' : ''}}`;
  const extra = stage.showCompetitionExtra ? `<div class="competition-extra">
    <div class="meta">Competition candidate: second nonlocal progress=${{fmt(stage.secondProgress,0)}}%, second distance=${{fmt(stage.secondDistance,4)}}, relative gap=${{fmt(stage.relativeGap * 100,2)}}%.</div>
    <div class="stage-images">
      <div class="imgbox"><h4>Current target<br>stage=${{fmt(stage.stage,0)}}%, frame ${{frameLabel(stage.targetFrameIndex)}}</h4>${{imgOrPlaceholder(stage.targetFrameSrc, stage.targetNote)}}</div>
      <div class="imgbox"><h4>Current nearest<br>progress=${{fmt(stage.nearestProgress,0)}}%, frame ${{frameLabel(stage.nearestFrameIndex)}}</h4>${{imgOrPlaceholder(stage.nearestFrameSrc, stage.nearestNote)}}</div>
      <div class="imgbox"><h4>Second candidate<br>progress=${{fmt(stage.secondProgress,0)}}%, frame ${{frameLabel(stage.secondFrameIndex)}}</h4>${{imgOrPlaceholder(stage.secondFrameSrc, stage.secondNote)}}</div>
    </div>
  </div>` : '';
  return `<article class="${{cardClass}}">
    <h3>target stage ${{fmt(stage.stage,0)}}% ${{badges(stage)}}</h3>
    <div class="stage-images">
      <div class="imgbox"><h4>Target frame ${{frameLabel(stage.targetFrameIndex)}}</h4>${{imgOrPlaceholder(stage.targetFrameSrc, stage.targetNote)}}</div>
      <div class="imgbox"><h4>Nearest baseline<br>progress=${{fmt(stage.nearestProgress,0)}}%, frame ${{frameLabel(stage.nearestFrameIndex)}}</h4>${{imgOrPlaceholder(stage.nearestFrameSrc, stage.nearestNote)}}</div>
    </div>
    <div class="meta">nearest distance=${{fmt(stage.nearestDistance,5)}}; second progress=${{fmt(stage.secondProgress,0)}}%; second distance=${{fmt(stage.secondDistance,5)}}; relative gap=${{fmt(stage.relativeGap * 100,2)}}%; return amount=${{fmt(stage.returnAmount,0)}}%</div>
    ${{extra}}
  </article>`;
}}

function selectedCurveChart(key) {{
  const line = DATA.lines.find(item => item.key === key);
  if (!line) return '<div class="note">No curve data found for selected key.</div>';
  const width = 980, height = 390;
  const left = 68, right = 24, top = 34, bottom = 54;
  const plotW = width - left - right, plotH = height - top - bottom;
  const yMax = Math.max(1, ...line.points.map(p => p.distance)) * 1.10;
  const xScale = x => left + (x / 100) * plotW;
  const yScale = y => top + plotH - (y / yMax) * plotH;
  const d = pathFromPoints(line.points, xScale, yScale);
  const yTicks = [0, yMax/4, yMax/2, yMax*3/4, yMax];
  const color = COLORS[line.targetClass];
  let svg = `<svg class="chart-svg" viewBox="0 0 ${{width}} ${{height}}">`;
  [0,20,40,60,80,100].forEach(t => {{
    const x = xScale(t);
    svg += `<line class="grid" x1="${{x}}" y1="${{top}}" x2="${{x}}" y2="${{top+plotH}}"/>`;
    svg += `<text class="tick" x="${{x}}" y="${{top+plotH+22}}" text-anchor="middle">${{t}}</text>`;
  }});
  yTicks.forEach(t => {{
    const y = yScale(t);
    svg += `<line class="grid" x1="${{left}}" y1="${{y}}" x2="${{left+plotW}}" y2="${{y}}"/>`;
    svg += `<text class="tick" x="${{left-10}}" y="${{y+4}}" text-anchor="end">${{fmt(t,2)}}</text>`;
  }});
  svg += `<line class="axis" x1="${{left}}" y1="${{top+plotH}}" x2="${{left+plotW}}" y2="${{top+plotH}}"/>`;
  svg += `<line class="axis" x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{top+plotH}}"/>`;
  svg += `<text class="label" x="${{left+plotW/2}}" y="${{height-16}}" text-anchor="middle">nearest baseline progress (%)</text>`;
  svg += `<text class="label" x="20" y="${{top+plotH/2}}" text-anchor="middle" transform="rotate(-90,20,${{top+plotH/2}})">nearest vector length ||target - nearest||_2</text>`;
  svg += `<path class="curve${{line.sourceType === 'prototype' ? ' prototype' : ''}} active" d="${{d}}" stroke="${{color}}"/>`;
  line.points.forEach(p => {{
    svg += `<circle cx="${{xScale(p.progress).toFixed(2)}}" cy="${{yScale(p.distance).toFixed(2)}}" r="3.4" fill="${{color}}"><title>stage ${{fmt(p.stage,0)}}%, progress ${{fmt(p.progress,0)}}%, distance ${{fmt(p.distance,4)}}</title></circle>`;
  }});
  svg += `</svg>`;
  return `<div class="selected-chart">
    <div class="chart-title">Selected new curve</div>
    <div class="chart-caption">x = nearest baseline progress; y = nearest vector length. Points are target stages.</div>
    ${{svg}}
  </div>`;
}}

function selectCurve(key, label) {{
  const detail = DATA.inspectorDetails[key];
  if (!detail) {{
    document.getElementById('inspector').innerHTML = `<div class="note">No inspector data found for ${{label}}.</div>`;
    return;
  }}
  document.getElementById('inspector').innerHTML = `
    <h3>${{label}}</h3>
    <div class="inspector-facts">
      <div class="fact"><b>method</b><br>${{detail.method}}</div>
      <div class="fact"><b>baseline</b><br>${{detail.baselineClass}}</div>
      <div class="fact"><b>target</b><br>${{detail.targetClass}}</div>
      <div class="fact"><b>source</b><br>${{detail.sourceType}}, seq=${{detail.sequenceId}}, rank=${{detail.rank || '-'}}</div>
      <div class="fact"><b>stages</b><br>${{detail.stageCount}}</div>
      <div class="fact"><b>return points</b><br>${{detail.returnCount}}</div>
      <div class="fact"><b>return + competition</b><br>${{detail.returnCompetitionCount}}</div>
      <div class="fact"><b>rule</b><br>return >=10%, competition gap <=10%</div>
    </div>
    ${{selectedCurveChart(key)}}
    <div class="stage-grid">${{detail.stages.map(stageCard).join('')}}</div>
  `;
}}

function updateCurveSelect() {{
  const select = document.getElementById('curveSelect');
  const previous = select.value;
  const available = [];
  CHARTS.forEach(spec => {{
    filteredLines(spec.method, spec.baseline).forEach(line => {{
      available.push({{key: line.key, label: line.label}});
    }});
  }});
  const unique = [];
  const seen = new Set();
  available.forEach(item => {{
    if (!seen.has(item.key)) {{
      unique.push(item);
      seen.add(item.key);
    }}
  }});
  select.innerHTML = '<option value="">Select a curve</option>' +
    unique.map(item => `<option value="${{item.key}}">${{item.label}}</option>`).join('');
  if (seen.has(previous)) {{
    select.value = previous;
  }} else {{
    select.value = '';
    document.getElementById('inspector').innerHTML = '<div class="note">No curve selected yet.</div>';
  }}
}}

document.querySelectorAll('input').forEach(el => el.addEventListener('change', renderAllCharts));
document.getElementById('curveSelect').addEventListener('change', event => {{
  const key = event.target.value;
  if (!key) {{
    document.getElementById('inspector').innerHTML = '<div class="note">No curve selected yet.</div>';
    return;
  }}
  const line = DATA.lines.find(item => item.key === key);
  selectCurve(key, line ? line.label : key);
}});
showMetrics();
renderAllCharts();
</script>
</body>
</html>
"""

    def run(self) -> Path:
        payload = self.build_payload()
        output = self.cfg.output_root / "new_curve_competition_summary.html"
        output.write_text(self.render_html(payload), encoding="utf-8")
        return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate new-curve summary with selected-curve stage inspector."
    )
    parser.add_argument("--analysis_input_root", default=str(NewCurveCompetitionSummaryConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(NewCurveCompetitionSummaryConfig.projection_output_root))
    parser.add_argument("--nearest_output_root", default=str(NewCurveCompetitionSummaryConfig.nearest_output_root))
    parser.add_argument("--backtrack_output_root", default=str(NewCurveCompetitionSummaryConfig.backtrack_output_root))
    parser.add_argument("--output_root", default=str(NewCurveCompetitionSummaryConfig.output_root))
    return parser


def main() -> None:
    cfg = NewCurveCompetitionSummaryConfig.from_args(build_arg_parser().parse_args())
    output = NewCurveCompetitionSummaryReport(cfg).run()
    print(f"new_curve_competition_summary_html: {output}")


if __name__ == "__main__":
    main()
