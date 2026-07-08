from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_nearest_competition_frame_report import (
    BASELINE_CLASSES,
    CLASS_NAMES,
    METHODS,
    CompetitionFrameConfig,
    CompetitionFrameReport,
)


class CompetitionBacktrackReport(CompetitionFrameReport):
    def row_metrics(self, row: dict[str, str]) -> dict:
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
        return {
            **row,
            "best_progress": float(best_idx),
            "second_progress": float(second_idx),
            "best_distance": best_distance,
            "second_distance": second_distance,
            "relative_gap": float(relative_gap),
            "absolute_gap": float(second_distance - best_distance),
        }

    def build_events(self) -> tuple[list[dict], dict]:
        rows = self.read_csv(
            self.cfg.nearest_output_root / "csv" / "nearest6_nearest_baseline_curve_all.csv"
        )
        methodb_meta = self.load_methodb_meta()
        grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
        for row in rows:
            metric_row = self.row_metrics(row)
            key = (
                metric_row["method"],
                metric_row["baseline_class"],
                metric_row["target_class"],
                metric_row["sequence_id"],
                metric_row["rank"],
            )
            grouped.setdefault(key, []).append(metric_row)

        events: list[dict] = []
        for group_rows in grouped.values():
            ordered = sorted(group_rows, key=lambda r: float(r["target_stage_percent"]))
            for previous, current in zip(ordered, ordered[1:]):
                if current["best_progress"] >= previous["best_progress"]:
                    continue
                if current["relative_gap"] > self.cfg.relative_threshold:
                    continue

                method = current["method"]
                baseline_class = current["baseline_class"]
                target_class = current["target_class"]
                sequence_id = current["sequence_id"]
                prev_stage = float(previous["target_stage_percent"])
                curr_stage = float(current["target_stage_percent"])

                prev_target_idx, prev_target_src = self.target_asset(
                    target_class, sequence_id, prev_stage
                )
                curr_target_idx, curr_target_src = self.target_asset(
                    target_class, sequence_id, curr_stage
                )
                prev_best_idx, prev_best_src, prev_best_note = self.baseline_asset(
                    method, baseline_class, previous["best_progress"], methodb_meta
                )
                curr_best_idx, curr_best_src, curr_best_note = self.baseline_asset(
                    method, baseline_class, current["best_progress"], methodb_meta
                )
                second_idx, second_src, second_note = self.baseline_asset(
                    method, baseline_class, current["second_progress"], methodb_meta
                )

                backtrack = previous["best_progress"] - current["best_progress"]
                events.append(
                    {
                        "method": method,
                        "baseline_class": baseline_class,
                        "target_class": target_class,
                        "sequence_id": sequence_id,
                        "rank": current["rank"],
                        "previous_stage_percent": prev_stage,
                        "current_stage_percent": curr_stage,
                        "previous_target_frame_index": prev_target_idx,
                        "previous_target_frame_src": prev_target_src,
                        "current_target_frame_index": curr_target_idx,
                        "current_target_frame_src": curr_target_src,
                        "previous_best_progress": previous["best_progress"],
                        "current_best_progress": current["best_progress"],
                        "second_progress": current["second_progress"],
                        "backtrack_percent": backtrack,
                        "previous_best_distance": previous["best_distance"],
                        "current_best_distance": current["best_distance"],
                        "second_distance": current["second_distance"],
                        "relative_gap": current["relative_gap"],
                        "absolute_gap": current["absolute_gap"],
                        "previous_best_frame_index": prev_best_idx,
                        "previous_best_frame_src": prev_best_src,
                        "previous_best_note": prev_best_note,
                        "current_best_frame_index": curr_best_idx,
                        "current_best_frame_src": curr_best_src,
                        "current_best_note": curr_best_note,
                        "second_frame_index": second_idx,
                        "second_frame_src": second_src,
                        "second_note": second_note,
                        "label": (
                            f"{method} | baseline={baseline_class} | {target_class} "
                            f"seq={sequence_id}, rank={current['rank']} | "
                            f"stage={prev_stage:.0f}->{curr_stage:.0f}% | "
                            f"progress={previous['best_progress']:.0f}->{current['best_progress']:.0f}% | "
                            f"second={current['second_progress']:.0f}% | gap={current['relative_gap'] * 100:.2f}%"
                        ),
                    }
                )

        events.sort(key=lambda e: (e["relative_gap"], -e["backtrack_percent"]))
        summary = {
            "total_nearest6_points": len(rows),
            "backtrack_competition_events": len(events),
            "relative_threshold_percent": self.cfg.relative_threshold * 100.0,
            "nonlocal_gap_percent": self.cfg.nonlocal_gap_percent,
            "by_method": dict(Counter(e["method"] for e in events)),
            "by_method_baseline": [
                {"method": k[0], "baseline_class": k[1], "count": v}
                for k, v in sorted(Counter((e["method"], e["baseline_class"]) for e in events).items())
            ],
            "by_target_class": dict(Counter(e["target_class"] for e in events)),
        }
        return events, summary

    def write_csv(self, events: list[dict]) -> Path:
        output = self.cfg.output_root / "competition_backtrack_events.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        text_fields = [
            "method",
            "baseline_class",
            "target_class",
            "sequence_id",
            "rank",
            "previous_stage_percent",
            "current_stage_percent",
            "previous_best_progress",
            "current_best_progress",
            "second_progress",
            "backtrack_percent",
            "previous_best_distance",
            "current_best_distance",
            "second_distance",
            "relative_gap",
            "absolute_gap",
            "label",
        ]
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=text_fields)
            writer.writeheader()
            for event in events:
                writer.writerow({field: event[field] for field in text_fields})
        return output

    def render_html(self, events: list[dict], summary: dict) -> str:
        payload = self.json_data(
            {
                "events": events,
                "summary": summary,
                "methods": METHODS,
                "baselineClasses": BASELINE_CLASSES,
                "classNames": CLASS_NAMES,
            }
        )
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Competition With Backtracking Frames</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.48}}
main{{max-width:1500px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 8px}}
p{{max-width:1120px}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
.metric{{background:white;border:1px solid #ddd;padding:12px}}
.metric b{{font-size:22px;display:block}}
.note{{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}}
.controls{{position:sticky;top:0;z-index:10;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0;margin:10px 0 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
select,input{{font-size:13px}}
label{{font-size:13px;display:inline-flex;align-items:center;gap:6px}}
.count{{font-weight:700}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));gap:14px}}
.card{{background:white;border:1px solid #ddd;padding:10px}}
.card h3{{font-size:14px;margin:0 0 8px;line-height:1.35}}
.images{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}
.imgbox{{border:1px solid #ddd;background:#f4f4f4;padding:6px;min-width:0}}
.imgbox h4{{font-size:12px;margin:0 0 5px;line-height:1.3}}
.imgbox img{{width:100%;height:150px;object-fit:contain;background:#eee;display:block}}
.placeholder{{height:150px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px;color:#555;background:#eee;padding:8px}}
.meta{{font-size:12px;color:#444;margin-top:6px}}
@media (max-width:900px){{.summary{{grid-template-columns:1fr 1fr}}.gallery{{grid-template-columns:1fr}}.images{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<main>
<h1>Competition With Backtracking Frames</h1>
<p>This report lists only events where the nearest baseline progress moves backward from one target stage to the next, and the foldback point also has a competing nonlocal nearest candidate.</p>
<div class="note">Event definition: current_best_progress &lt; previous_best_progress, and (second_nonlocal_distance - current_best_distance) / current_best_distance &lt;= 10%. The second candidate must be more than 10% baseline-progress away from the current best candidate.</div>
<div class="note">methodB baseline frames are real medoid frames. methodA baseline images are unavailable because methodA baselines are median prototypes.</div>
<div class="summary" id="summary"></div>
<div class="controls">
<label>method <select id="methodSelect"><option value="all">all</option></select></label>
<label>baseline <select id="baselineSelect"><option value="all">all</option></select></label>
<label>target <select id="targetSelect"><option value="all">all</option></select></label>
<label><input type="checkbox" id="methodBOnly"> methodB only: real baseline frames</label>
<label>sort <select id="sortSelect"><option value="gap">smallest relative gap</option><option value="backtrack">largest backtrack</option><option value="stage">current stage</option><option value="target">target class</option></select></label>
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
    ['events kept', s.backtrack_competition_events],
    ['relative threshold', `${{fmt(s.relative_threshold_percent,1)}}%`],
    ['nonlocal separation', `>${{fmt(s.nonlocal_gap_percent,0)}}%`],
  ];
  document.getElementById('summary').innerHTML = items.map(([k,v]) => `<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`).join('');
}}

function filteredEvents() {{
  let events = DATA.events.slice();
  if (methodBOnly.checked) events = events.filter(e => e.method === 'methodB');
  if (methodSelect.value !== 'all') events = events.filter(e => e.method === methodSelect.value);
  if (baselineSelect.value !== 'all') events = events.filter(e => e.baseline_class === baselineSelect.value);
  if (targetSelect.value !== 'all') events = events.filter(e => e.target_class === targetSelect.value);
  if (sortSelect.value === 'gap') events.sort((a,b) => a.relative_gap - b.relative_gap);
  if (sortSelect.value === 'backtrack') events.sort((a,b) => b.backtrack_percent - a.backtrack_percent || a.relative_gap - b.relative_gap);
  if (sortSelect.value === 'stage') events.sort((a,b) => a.current_stage_percent - b.current_stage_percent || a.relative_gap - b.relative_gap);
  if (sortSelect.value === 'target') events.sort((a,b) => a.target_class.localeCompare(b.target_class) || a.relative_gap - b.relative_gap);
  return events;
}}

function imgOrPlaceholder(src, note) {{
  if (src) return `<img loading="lazy" src="${{src}}" alt="frame">`;
  return `<div class="placeholder">${{note || 'No frame available'}}</div>`;
}}

function card(e) {{
  return `<article class="card">
    <h3>${{e.label}}</h3>
    <div class="images">
      <div class="imgbox">
        <h4>Previous target frame ${{frameLabel(e.previous_target_frame_index)}}<br>stage=${{fmt(e.previous_stage_percent,0)}}%</h4>
        ${{imgOrPlaceholder(e.previous_target_frame_src)}}
      </div>
      <div class="imgbox">
        <h4>Current target frame ${{frameLabel(e.current_target_frame_index)}}<br>stage=${{fmt(e.current_stage_percent,0)}}%</h4>
        ${{imgOrPlaceholder(e.current_target_frame_src)}}
      </div>
      <div class="imgbox">
        <h4>Previous best baseline ${{frameLabel(e.previous_best_frame_index)}}<br>progress=${{fmt(e.previous_best_progress,0)}}%, d=${{fmt(e.previous_best_distance,4)}}</h4>
        ${{imgOrPlaceholder(e.previous_best_frame_src, e.previous_best_note)}}
      </div>
      <div class="imgbox">
        <h4>Current best baseline ${{frameLabel(e.current_best_frame_index)}}<br>progress=${{fmt(e.current_best_progress,0)}}%, d=${{fmt(e.current_best_distance,4)}}</h4>
        ${{imgOrPlaceholder(e.current_best_frame_src, e.current_best_note)}}
      </div>
      <div class="imgbox">
        <h4>Current second candidate ${{frameLabel(e.second_frame_index)}}<br>progress=${{fmt(e.second_progress,0)}}%, d=${{fmt(e.second_distance,4)}}</h4>
        ${{imgOrPlaceholder(e.second_frame_src, e.second_note)}}
      </div>
    </div>
    <div class="meta">backtrack=${{fmt(e.backtrack_percent,0)}}%; absolute gap=${{fmt(e.absolute_gap,6)}}; relative gap=${{fmt(e.relative_gap * 100,2)}}%; method=${{e.method}}; baseline=${{e.baseline_class}}; target=${{e.target_class}}</div>
  </article>`;
}}

function render() {{
  const events = filteredEvents();
  document.getElementById('visibleCount').textContent = `${{events.length}} visible events`;
  gallery.innerHTML = events.map(card).join('');
}}

[methodSelect, baselineSelect, targetSelect, sortSelect, methodBOnly].forEach(el => el.addEventListener('change', render));
showSummary();
render();
</script>
</body>
</html>
"""

    def run(self) -> tuple[Path, Path]:
        events, summary = self.build_events()
        csv_output = self.write_csv(events)
        html_output = self.cfg.output_root / "competition_backtrack_frames.html"
        html_output.write_text(self.render_html(events, summary), encoding="utf-8")
        return html_output, csv_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List nearest-baseline competition events that also backtrack.")
    parser.add_argument("--analysis_input_root", default=str(CompetitionFrameConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(CompetitionFrameConfig.projection_output_root))
    parser.add_argument("--nearest_output_root", default=str(CompetitionFrameConfig.nearest_output_root))
    parser.add_argument(
        "--output_root",
        default=r"E:\Matsuda_data\3-10meeting\nearest_baseline_competition_backtrack_frames",
    )
    parser.add_argument("--relative_threshold", type=float, default=CompetitionFrameConfig.relative_threshold)
    parser.add_argument("--nonlocal_gap_percent", type=float, default=CompetitionFrameConfig.nonlocal_gap_percent)
    return parser


def main() -> None:
    cfg = CompetitionFrameConfig.from_args(build_arg_parser().parse_args())
    html_output, csv_output = CompetitionBacktrackReport(cfg).run()
    print(f"competition_backtrack_html: {html_output}")
    print(f"competition_backtrack_csv: {csv_output}")


if __name__ == "__main__":
    main()
