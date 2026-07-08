"""Build interactive time-ranking plots for selected sequences.

Ranking scores come from the existing frame-level SmileComp + true-smile scale
CSV. The initial non-growth region can be compressed to at most six frames.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_SCORE_CSV = Path(r"E:\Dataset\smileranking_plot\frame_smile_ranking_scores.csv")
DEFAULT_STILLIMAGE_ROOT = Path(r"E:\Dataset\stillimages")
DEFAULT_OUTPUT_DIR = Path(r"E:\Dataset\t_ranking_selected_cases")
DEFAULT_CASES = OrderedDict(
    [
        ("bitter", ["8", "11", "12", "13", "15", "17", "23"]),
        ("true", ["0", "2", "5", "10"]),
        ("polite", ["8", "14", "17", "18", "21"]),
    ]
)
CLASS_COLORS = {
    "bitter": "#d73b32",
    "true": "#1f77b4",
    "polite": "#2f9b4f",
}
CLASS_SHADE_COLORS = {
    "bitter": ["#f36c5b", "#e45747", "#cf3d34", "#b72d2b", "#9d2024", "#82181f", "#661019"],
    "true": ["#8ec7ee", "#66aee0", "#4595d0", "#2b79b8", "#1f5f99", "#174a7b"],
    "polite": ["#8bd17c", "#69bf63", "#4ca955", "#368f48", "#26763c", "#195d30"],
}
PLOT_DEFS = [
    {"id": "T_T", "title": "T_T plot: selected true-smile sequences", "labels": ["true"]},
    {"id": "T_B", "title": "T_B plot: selected bitter-smile sequences", "labels": ["bitter"]},
    {"id": "T_P", "title": "T_P plot: selected polite-smile sequences", "labels": ["polite"]},
    {"id": "ALL", "title": "ALL plot: selected sequences", "labels": ["true", "bitter", "polite"]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one interactive HTML containing T_T, T_B, T_P, and ALL ranking plots."
    )
    parser.add_argument("--score_csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--stillimage_root", type=Path, default=DEFAULT_STILLIMAGE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cases",
        default="",
        help="Optional format: bitter:8,11;true:0,2,5,10;polite:8,14",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_initial_static_frames", type=int, default=6)
    parser.add_argument("--growth_delta", type=float, default=0.05)
    parser.add_argument("--still_width", type=int, default=900)
    parser.add_argument("--curve_thumbnail_stride_frames", type=int, default=6)
    parser.add_argument("--curve_thumbnail_source_size", type=int, default=86)
    parser.add_argument("--curve_thumbnail_display_size", type=int, default=48)
    parser.add_argument("--jpeg_quality", type=int, default=84)
    return parser.parse_args()


def parse_cases(case_spec: str) -> OrderedDict[str, list[str]]:
    if not case_spec.strip():
        return OrderedDict((label, list(ids)) for label, ids in DEFAULT_CASES.items())

    parsed: OrderedDict[str, list[str]] = OrderedDict()
    for block in case_spec.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Invalid case block: {block!r}")
        label, raw_ids = block.split(":", 1)
        ids = [value.strip() for value in raw_ids.split(",") if value.strip()]
        if not label.strip() or not ids:
            raise ValueError(f"Invalid case block: {block!r}")
        parsed[label.strip()] = ids
    return parsed


def read_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def encode_stillimage(path: Path, max_width: int, jpeg_quality: int) -> str:
    with Image.open(path) as img:
        image = img.convert("RGB")
        if image.width > max_width:
            new_height = round(image.height * (max_width / image.width))
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def encode_frame_thumbnail(path: Path, size: int, jpeg_quality: int) -> str:
    with Image.open(path) as img:
        thumb = ImageOps.fit(img.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def first_growth_index(scores: list[float], growth_delta: float) -> int | None:
    if not scores:
        return None
    start = scores[0]
    threshold = start + growth_delta
    for index, value in enumerate(scores[1:], start=1):
        if value > threshold:
            return index
    return None


def adjusted_frame_positions(
    frame_indices: list[int],
    scores: list[float],
    max_initial_static_frames: int,
    growth_delta: float,
) -> tuple[list[float], dict]:
    growth_index = first_growth_index(scores, growth_delta)
    compressed = growth_index is not None and growth_index > max_initial_static_frames

    if not compressed:
        return [float(frame) for frame in frame_indices], {
            "compressed": False,
            "growthStartFrame": growth_index,
            "compressionShiftFrames": 0.0,
            "compressionScale": 1.0,
        }

    scale = max_initial_static_frames / growth_index
    adjusted = []
    for frame in frame_indices:
        if frame <= growth_index:
            adjusted.append(frame * scale)
        else:
            adjusted.append(max_initial_static_frames + (frame - growth_index))

    return adjusted, {
        "compressed": True,
        "growthStartFrame": growth_index,
        "compressionShiftFrames": float(growth_index - max_initial_static_frames),
        "compressionScale": float(scale),
    }


def sequence_color(label: str, color_index: int) -> str:
    palette = CLASS_SHADE_COLORS.get(label, [CLASS_COLORS.get(label, "#333333")])
    return palette[color_index % len(palette)]


def build_sequences(
    score_rows: list[dict[str, str]],
    cases: OrderedDict[str, list[str]],
    stillimage_root: Path,
    fps: float,
    max_initial_static_frames: int,
    growth_delta: float,
    still_width: int,
    curve_thumbnail_stride_frames: int,
    curve_thumbnail_source_size: int,
    jpeg_quality: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if curve_thumbnail_stride_frames <= 0:
        raise ValueError("--curve_thumbnail_stride_frames must be > 0")

    grouped: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    wanted = {(label, sequence_id) for label, ids in cases.items() for sequence_id in ids}
    for row in score_rows:
        key = (row["label"], row["sequence_id"])
        if key in wanted:
            grouped.setdefault(key, []).append(row)

    sequences = []
    manifest_rows = []
    coordinate_rows = []
    label_counts: dict[str, int] = {}

    for label, ids in cases.items():
        for sequence_id in ids:
            key = (label, sequence_id)
            rows = grouped.get(key, [])
            sequence_dir = rows[0]["sequence_dir"] if rows else ""
            stillimage_path = stillimage_root / label / f"{sequence_id}.png"
            manifest = {
                "label": label,
                "sequence_id": sequence_id,
                "status": "ok",
                "reason": "",
                "score_rows": len(rows),
                "sequence_dir": sequence_dir,
                "stillimage_path": str(stillimage_path),
            }
            if not rows:
                manifest["status"] = "missing_scores"
                manifest["reason"] = "no frame ranking rows found"
                manifest_rows.append(manifest)
                continue
            if not stillimage_path.is_file():
                manifest["status"] = "missing_stillimage"
                manifest["reason"] = "still image does not exist"
                manifest_rows.append(manifest)
                continue

            rows = sorted(rows, key=lambda row: int(row["plot_frame_index_from_onset"]))
            frame_indices = [int(row["plot_frame_index_from_onset"]) for row in rows]
            scores = [float(row["score_0_9"]) for row in rows]
            adjusted_frames, compression = adjusted_frame_positions(
                frame_indices,
                scores,
                max_initial_static_frames,
                growth_delta,
            )

            color_index = label_counts.get(label, 0)
            label_counts[label] = color_index + 1
            class_color = CLASS_COLORS.get(label, "#333333")
            shade_color = sequence_color(label, color_index)

            points = []
            thumbnails = []
            for row, original_frame, adjusted_frame, score in zip(
                rows, frame_indices, adjusted_frames, scores
            ):
                point = {
                    "originalFrame": original_frame,
                    "adjustedFrame": adjusted_frame,
                    "seconds": adjusted_frame / fps,
                    "score": score,
                    "frameFile": row["frame_file"],
                    "framePath": row["frame_path"],
                    "nearestLevel": int(row["nearest_level"]),
                }
                points.append(point)
                if original_frame % curve_thumbnail_stride_frames == 0:
                    frame_path = Path(row["frame_path"])
                    thumbnails.append(
                        {
                            "originalFrame": original_frame,
                            "adjustedFrame": adjusted_frame,
                            "seconds": adjusted_frame / fps,
                            "score": score,
                            "frameFile": row["frame_file"],
                            "src": encode_frame_thumbnail(
                                frame_path,
                                curve_thumbnail_source_size,
                                jpeg_quality,
                            ),
                        }
                    )
                coordinate_rows.append(
                    {
                        "label": label,
                        "sequence_id": sequence_id,
                        "original_frame_index": original_frame,
                        "adjusted_frame_index": f"{adjusted_frame:.8f}",
                        "adjusted_time_seconds": f"{adjusted_frame / fps:.8f}",
                        "score_0_9": f"{score:.8f}",
                        "nearest_level": int(row["nearest_level"]),
                        "frame_file": row["frame_file"],
                        "frame_path": row["frame_path"],
                        "compression_applied": int(compression["compressed"]),
                        "growth_start_frame": "" if compression["growthStartFrame"] is None else compression["growthStartFrame"],
                        "compression_shift_frames": f"{compression['compressionShiftFrames']:.8f}",
                        "compression_scale": f"{compression['compressionScale']:.8f}",
                    }
                )

            sequence = {
                "key": f"{label}/{sequence_id}",
                "label": label,
                "sequenceId": sequence_id,
                "frameCount": len(points),
                "originalFrameCount": len(points),
                "maxOriginalFrame": max(frame_indices),
                "maxAdjustedFrame": max(adjusted_frames),
                "fps": fps,
                "classColor": class_color,
                "shadeColor": shade_color,
                "colorByPlot": {
                    "T_T": shade_color,
                    "T_B": shade_color,
                    "T_P": shade_color,
                    "ALL": class_color,
                },
                "points": points,
                "thumbnails": thumbnails,
                "compression": compression,
                "scoreSummary": {
                    "min": min(scores),
                    "max": max(scores),
                    "start": scores[0],
                    "end": scores[-1],
                },
                "stillImage": {
                    "path": str(stillimage_path),
                    "src": encode_stillimage(stillimage_path, still_width, jpeg_quality),
                },
            }
            sequences.append(sequence)
            manifest["compressed"] = int(compression["compressed"])
            manifest["growth_start_frame"] = (
                "" if compression["growthStartFrame"] is None else compression["growthStartFrame"]
            )
            manifest["compression_shift_frames"] = f"{compression['compressionShiftFrames']:.8f}"
            manifest_rows.append(manifest)

    return sequences, manifest_rows, coordinate_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_plots(sequences: list[dict]) -> list[dict]:
    plots = []
    for plot_def in PLOT_DEFS:
        plot_sequences = [seq for seq in sequences if seq["label"] in plot_def["labels"]]
        if not plot_sequences:
            x_max = 10.0
        else:
            x_max = max(seq["maxAdjustedFrame"] for seq in plot_sequences)
        plots.append(
            {
                "id": plot_def["id"],
                "title": plot_def["title"],
                "labels": plot_def["labels"],
                "sequenceKeys": [seq["key"] for seq in plot_sequences],
                "domain": {
                    "xMin": 0.0,
                    "xMax": max(10.0, x_max),
                    "yMin": 0.0,
                    "yMax": 9.0,
                },
            }
        )
    return plots


def html_document(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title = html.escape(payload["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --text: #1f2933;
      --muted: #66717f;
      --grid: #d8dde3;
      --axis: #4b5563;
      --panel: #ffffff;
      --page: #f3f5f7;
      --border: #d6dce3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .layout {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(980px, 1fr) 360px;
    }}
    .plots {{
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(720px, 1fr);
      gap: 14px;
      align-content: start;
    }}
    .plot-card {{
      background: #fff;
      border: 1px solid var(--border);
      min-height: 390px;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    .title {{
      font-size: 17px;
      font-weight: 700;
      fill: var(--text);
    }}
    .subtitle, .tick-label {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .grid-line {{
      stroke: var(--grid);
      stroke-width: 1;
    }}
    .axis-line {{
      stroke: var(--axis);
      stroke-width: 1.2;
    }}
    .curve {{
      fill: none;
      stroke-linejoin: round;
      stroke-linecap: round;
      stroke-width: 1.8;
      opacity: 0.76;
      transition: opacity 120ms ease, stroke-width 120ms ease;
    }}
    .curve.is-muted {{
      opacity: 0.10;
      stroke-width: 1.1;
    }}
    .curve.is-active {{
      opacity: 1;
      stroke-width: 4.2;
    }}
    .hit-line {{
      fill: none;
      stroke: #fff;
      stroke-opacity: 0.001;
      stroke-width: 16;
      stroke-linejoin: round;
      stroke-linecap: round;
      cursor: pointer;
      pointer-events: stroke;
    }}
    .endpoint {{
      stroke: #fff;
      stroke-width: 1.2;
      transition: opacity 120ms ease;
    }}
    .endpoint.is-muted {{
      opacity: 0.12;
    }}
    .thumb-frame {{
      fill: #fff;
      stroke-width: 2;
      filter: drop-shadow(0 3px 6px rgba(15, 23, 42, 0.24));
    }}
    .thumb-label {{
      fill: #27313c;
      font-size: 9px;
      font-weight: 700;
      paint-order: stroke;
      stroke: white;
      stroke-width: 3;
    }}
    .side {{
      background: var(--panel);
      border-left: 1px solid var(--border);
      padding: 16px 14px;
      overflow: auto;
      max-height: 100vh;
    }}
    .meta {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
      margin-bottom: 12px;
    }}
    .meta h1 {{
      margin: 0 0 8px;
      font-size: 17px;
      line-height: 1.25;
    }}
    .meta p {{
      margin: 3px 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .control-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      color: var(--text);
      font-size: 13px;
    }}
    .control-row input {{
      width: 16px;
      height: 16px;
      margin: 0;
    }}
    .readout {{
      min-height: 92px;
      border: 1px solid var(--border);
      padding: 10px;
      margin-bottom: 12px;
      background: #fbfcfd;
      font-size: 13px;
      line-height: 1.45;
    }}
    .still-wrap {{
      border: 1px solid var(--border);
      background: #fbfcfd;
      padding: 8px;
      margin-bottom: 12px;
      min-height: 86px;
    }}
    .still-wrap img {{
      width: 100%;
      display: block;
    }}
    .still-placeholder {{
      color: var(--muted);
      font-size: 12px;
      padding: 18px 4px;
    }}
    .legend-title {{
      margin: 14px 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      width: 100%;
      border: 0;
      border-radius: 4px;
      padding: 7px 6px;
      background: transparent;
      color: var(--text);
      font: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .legend-item:hover, .legend-item.is-active {{
      background: #eef3f7;
    }}
    .swatch {{
      width: 22px;
      height: 3px;
      border-radius: 3px;
      flex: 0 0 22px;
    }}
    .legend-main {{
      flex: 1;
      font-size: 13px;
    }}
    .legend-sub {{
      color: var(--muted);
      font-size: 11px;
    }}
    @media (max-width: 1300px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .side {{ max-height: none; border-left: 0; border-top: 1px solid var(--border); }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <main class="plots" id="plots"></main>
    <aside class="side">
      <div class="meta">
        <h1>T-ranking Plots</h1>
        <p>Ranking: SmileComp model scored against the 10-level true-smile scale.</p>
        <p>X axis: adjusted frame index; FPS: {payload["fps"]:.1f}.</p>
        <p>Initial non-growth longer than {payload["maxInitialStaticFrames"]} frames is compressed.</p>
        <p>Highlighted curves show one frame thumbnail every {payload["curveThumbnailStrideFrames"]} frames.</p>
        <label class="control-row"><input id="showFrameThumbnails" type="checkbox" checked><span>Frame thumbnails</span></label>
      </div>
      <div id="readout" class="readout">No curve selected.</div>
      <div id="still" class="still-wrap"><div class="still-placeholder">Select a curve to show its still image.</div></div>
      <div class="legend-title">Selected Sequences</div>
      <div id="legend"></div>
    </aside>
  </div>
  <script id="plot-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('plot-data').textContent);
    const plotsEl = document.getElementById('plots');
    const legendEl = document.getElementById('legend');
    const readout = document.getElementById('readout');
    const still = document.getElementById('still');
    const showFrameThumbnails = document.getElementById('showFrameThumbnails');
    const NS = 'http://www.w3.org/2000/svg';
    const W = 860;
    const H = 390;
    const margin = {{ left: 58, top: 56, right: 18, bottom: 56 }};
    const plotW = W - margin.left - margin.right;
    const plotH = H - margin.top - margin.bottom;
    const thumbSize = data.curveThumbnailDisplaySize;
    const yMin = 0;
    const yMax = 9;
    const sequenceByKey = new Map(data.sequences.map(seq => [seq.key, seq]));
    const plotById = new Map(data.plots.map(plot => [plot.id, plot]));
    const curveNodes = new Map();
    const endpointNodes = new Map();
    const thumbnailLayers = new Map();
    let activeKey = null;

    function createSvgEl(name, attrs = {{}}, parent) {{
      const node = document.createElementNS(NS, name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      parent.appendChild(node);
      return node;
    }}

    function fmt(value) {{
      return Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(1);
    }}

    function xScale(value, domain) {{
      return margin.left + ((value - domain.xMin) / (domain.xMax - domain.xMin)) * plotW;
    }}

    function yScale(value) {{
      return margin.top + ((yMax - value) / (yMax - yMin)) * plotH;
    }}

    function clamp(value, low, high) {{
      return Math.max(low, Math.min(high, value));
    }}

    function polyline(seq, domain) {{
      return seq.points.map(p => `${{xScale(p.adjustedFrame, domain)}},${{yScale(p.score)}}`).join(' ');
    }}

    function tickValues(min, max, count) {{
      const values = [];
      for (let i = 0; i <= count; i += 1) values.push(min + (i / count) * (max - min));
      return values;
    }}

    function appendCurveNode(key, node) {{
      if (!curveNodes.has(key)) curveNodes.set(key, []);
      curveNodes.get(key).push(node);
    }}

    function appendEndpointNode(key, node) {{
      if (!endpointNodes.has(key)) endpointNodes.set(key, []);
      endpointNodes.get(key).push(node);
    }}

    function drawPlot(plot) {{
      const card = document.createElement('section');
      card.className = 'plot-card';
      const svg = document.createElementNS(NS, 'svg');
      svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
      svg.setAttribute('aria-label', plot.title);
      card.appendChild(svg);
      plotsEl.appendChild(card);

      createSvgEl('rect', {{ x: 0, y: 0, width: W, height: H, fill: '#fff' }}, svg);
      createSvgEl('text', {{ x: margin.left, y: 28, class: 'title' }}, svg).textContent = plot.title;
      createSvgEl('text', {{ x: margin.left, y: 45, class: 'subtitle' }}, svg).textContent = 'Y: smile ranking 0-9; X: adjusted frame index at 30fps';

      const grid = createSvgEl('g', {{}}, svg);
      tickValues(plot.domain.xMin, plot.domain.xMax, 6).forEach(value => {{
        const x = xScale(value, plot.domain);
        createSvgEl('line', {{ x1: x, y1: margin.top, x2: x, y2: margin.top + plotH, class: 'grid-line' }}, grid);
        createSvgEl('text', {{ x, y: margin.top + plotH + 22, 'text-anchor': 'middle', class: 'tick-label' }}, grid).textContent = fmt(value);
      }});
      tickValues(yMin, yMax, 9).forEach(value => {{
        const y = yScale(value);
        createSvgEl('line', {{ x1: margin.left, y1: y, x2: margin.left + plotW, y2: y, class: 'grid-line' }}, grid);
        createSvgEl('text', {{ x: margin.left - 10, y: y + 4, 'text-anchor': 'end', class: 'tick-label' }}, grid).textContent = fmt(value);
      }});

      createSvgEl('line', {{ x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH, class: 'axis-line' }}, svg);
      createSvgEl('line', {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH, class: 'axis-line' }}, svg);
      createSvgEl('text', {{ x: margin.left + plotW / 2, y: H - 18, 'text-anchor': 'middle', class: 'axis-label' }}, svg).textContent = 'adjusted frame index';
      createSvgEl('text', {{ x: 18, y: margin.top + plotH / 2, 'text-anchor': 'middle', class: 'axis-label', transform: `rotate(-90 18 ${{margin.top + plotH / 2}})` }}, svg).textContent = 'smile ranking';

      const curveLayer = createSvgEl('g', {{}}, svg);
      const endpointLayer = createSvgEl('g', {{}}, svg);
      const hitLayer = createSvgEl('g', {{}}, svg);
      const thumbnailLayer = createSvgEl('g', {{ 'pointer-events': 'none' }}, svg);
      thumbnailLayers.set(plot.id, thumbnailLayer);

      plot.sequenceKeys.forEach(key => {{
        const seq = sequenceByKey.get(key);
        const points = polyline(seq, plot.domain);
        const color = seq.colorByPlot[plot.id] || seq.classColor;
        const curve = createSvgEl('polyline', {{
          points,
          class: 'curve',
          stroke: color,
          'data-key': key,
          'data-plot': plot.id
        }}, curveLayer);
        appendCurveNode(key, curve);

        const first = seq.points[0];
        const last = seq.points[seq.points.length - 1];
        appendEndpointNode(key, createSvgEl('circle', {{ cx: xScale(first.adjustedFrame, plot.domain), cy: yScale(first.score), r: 3.8, fill: color, class: 'endpoint' }}, endpointLayer));
        appendEndpointNode(key, createSvgEl('rect', {{ x: xScale(last.adjustedFrame, plot.domain) - 4, y: yScale(last.score) - 4, width: 8, height: 8, fill: color, class: 'endpoint' }}, endpointLayer));

        const hit = createSvgEl('polyline', {{
          points,
          class: 'hit-line',
          'data-key': key
        }}, hitLayer);
        hit.addEventListener('mouseenter', () => focus(key));
      }});
      svg.addEventListener('click', event => {{
        if (!isCurveTarget(event.target)) clearFocus();
      }});
    }}

    function isCurveTarget(target) {{
      return target.classList && (
        target.classList.contains('hit-line') ||
        target.classList.contains('curve') ||
        target.classList.contains('endpoint')
      );
    }}

    function compressionText(seq) {{
      if (!seq.compression.compressed) return 'opening unchanged';
      return `opening compressed: growth frame ${{seq.compression.growthStartFrame}} -> frame ${{data.maxInitialStaticFrames}}`;
    }}

    function thumbnailStatusText(seq) {{
      if (showFrameThumbnails.checked) return `${{seq.thumbnails.length}} curve thumbnails`;
      return `${{seq.thumbnails.length}} curve thumbnails hidden`;
    }}

    function updateReadout(seq) {{
      const s = seq.scoreSummary;
      readout.innerHTML = `<strong>${{seq.key}}</strong><br>${{seq.frameCount}} frames; adjusted length ${{seq.maxAdjustedFrame.toFixed(1)}} frames (${{(seq.maxAdjustedFrame / data.fps).toFixed(2)}}s)<br>score: start ${{s.start.toFixed(2)}}, max ${{s.max.toFixed(2)}}, end ${{s.end.toFixed(2)}}<br>${{compressionText(seq)}}<br>${{thumbnailStatusText(seq)}}`;
    }}

    function clearThumbnails() {{
      thumbnailLayers.forEach(layer => layer.replaceChildren());
    }}

    function renderThumbnails(seq) {{
      clearThumbnails();
      data.plots.forEach(plot => {{
        if (!plot.sequenceKeys.includes(seq.key)) return;
        const layer = thumbnailLayers.get(plot.id);
        if (!layer) return;
        const color = seq.colorByPlot[plot.id] || seq.classColor;
        seq.thumbnails.forEach(t => {{
          const cx = xScale(t.adjustedFrame, plot.domain);
          const cy = yScale(t.score);
          const x = clamp(cx - thumbSize / 2, margin.left, margin.left + plotW - thumbSize);
          const y = clamp(cy - thumbSize / 2, margin.top, margin.top + plotH - thumbSize);
          createSvgEl('rect', {{
            x: x - 2,
            y: y - 2,
            width: thumbSize + 4,
            height: thumbSize + 4,
            rx: 3,
            class: 'thumb-frame',
            stroke: color
          }}, layer);
          createSvgEl('image', {{
            x,
            y,
            width: thumbSize,
            height: thumbSize,
            href: t.src,
            'pointer-events': 'none'
          }}, layer);
          createSvgEl('text', {{
            x: x + 4,
            y: y + thumbSize - 5,
            class: 'thumb-label'
          }}, layer).textContent = `f${{t.originalFrame}}`;
        }});
      }});
    }}

    function focus(key) {{
      activeKey = key;
      const seq = sequenceByKey.get(key);
      if (!seq) return;
      data.sequences.forEach(item => {{
        const isActive = item.key === key;
        (curveNodes.get(item.key) || []).forEach(node => {{
          node.classList.toggle('is-active', isActive);
          node.classList.toggle('is-muted', !isActive);
        }});
        (endpointNodes.get(item.key) || []).forEach(node => node.classList.toggle('is-muted', !isActive));
        const button = document.querySelector(`[data-legend-key="${{CSS.escape(item.key)}}"]`);
        if (button) button.classList.toggle('is-active', isActive);
      }});
      if (showFrameThumbnails.checked) renderThumbnails(seq);
      else clearThumbnails();
      updateReadout(seq);
      still.innerHTML = `<img alt="${{seq.key}} still image" src="${{seq.stillImage.src}}"><div class="legend-sub" style="margin-top:6px">${{seq.stillImage.path}}</div>`;
    }}

    function clearFocus() {{
      activeKey = null;
      clearThumbnails();
      data.sequences.forEach(item => {{
        (curveNodes.get(item.key) || []).forEach(node => node.classList.remove('is-active', 'is-muted'));
        (endpointNodes.get(item.key) || []).forEach(node => node.classList.remove('is-muted'));
        const button = document.querySelector(`[data-legend-key="${{CSS.escape(item.key)}}"]`);
        if (button) button.classList.remove('is-active');
      }});
      readout.textContent = 'No curve selected.';
      still.innerHTML = '<div class="still-placeholder">Select a curve to show its still image.</div>';
    }}

    data.plots.forEach(drawPlot);

    showFrameThumbnails.addEventListener('change', () => {{
      if (!activeKey) return;
      const seq = sequenceByKey.get(activeKey);
      if (!seq) return;
      if (showFrameThumbnails.checked) renderThumbnails(seq);
      else clearThumbnails();
      updateReadout(seq);
    }});

    const groups = new Map();
    data.sequences.forEach(seq => {{
      if (!groups.has(seq.label)) groups.set(seq.label, []);
      groups.get(seq.label).push(seq);
    }});
    groups.forEach((seqs, label) => {{
      const groupTitle = document.createElement('div');
      groupTitle.className = 'legend-title';
      groupTitle.textContent = `${{label}} (${{seqs.length}})`;
      legendEl.appendChild(groupTitle);
      seqs.forEach(seq => {{
        const button = document.createElement('button');
        button.className = 'legend-item';
        button.dataset.legendKey = seq.key;
        button.innerHTML = `<span class="swatch" style="background:${{seq.classColor}}"></span><span class="legend-main">${{seq.key}}<br><span class="legend-sub">${{seq.frameCount}} frames; ${{compressionText(seq)}}</span></span>`;
        button.addEventListener('mouseenter', () => focus(seq.key));
        button.addEventListener('focus', () => focus(seq.key));
        button.addEventListener('click', () => activeKey === seq.key ? clearFocus() : focus(seq.key));
        legendEl.appendChild(button);
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    cases = parse_cases(args.cases)
    score_rows = read_scores(args.score_csv)
    sequences, manifest_rows, coordinate_rows = build_sequences(
        score_rows,
        cases,
        args.stillimage_root,
        args.fps,
        args.max_initial_static_frames,
        args.growth_delta,
        args.still_width,
        args.curve_thumbnail_stride_frames,
        args.curve_thumbnail_source_size,
        args.jpeg_quality,
    )
    if not sequences:
        raise RuntimeError("No valid selected sequences were found.")

    plots = build_plots(sequences)
    payload = {
        "title": "Selected sequence t-ranking plots",
        "scoreCsv": str(args.score_csv),
        "stillimageRoot": str(args.stillimage_root),
        "fps": args.fps,
        "maxInitialStaticFrames": args.max_initial_static_frames,
        "growthDelta": args.growth_delta,
        "curveThumbnailStrideFrames": args.curve_thumbnail_stride_frames,
        "curveThumbnailDisplaySize": args.curve_thumbnail_display_size,
        "sequences": sequences,
        "plots": plots,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "selected_cases_t_ranking_interactive.html"
    html_path.write_text(html_document(payload), encoding="utf-8")

    manifest_fields = [
        "label",
        "sequence_id",
        "status",
        "reason",
        "score_rows",
        "sequence_dir",
        "stillimage_path",
        "compressed",
        "growth_start_frame",
        "compression_shift_frames",
    ]
    write_csv(args.output_dir / "selected_cases_t_ranking_manifest.csv", manifest_fields, manifest_rows)
    coordinate_fields = [
        "label",
        "sequence_id",
        "original_frame_index",
        "adjusted_frame_index",
        "adjusted_time_seconds",
        "score_0_9",
        "nearest_level",
        "frame_file",
        "frame_path",
        "compression_applied",
        "growth_start_frame",
        "compression_shift_frames",
        "compression_scale",
    ]
    write_csv(args.output_dir / "selected_cases_t_ranking_adjusted_scores.csv", coordinate_fields, coordinate_rows)

    summary = {
        "output_html": str(html_path),
        "score_csv": str(args.score_csv),
        "stillimage_root": str(args.stillimage_root),
        "fps": args.fps,
        "max_initial_static_frames": args.max_initial_static_frames,
        "growth_delta": args.growth_delta,
        "curve_thumbnail_stride_frames": args.curve_thumbnail_stride_frames,
        "curve_thumbnail_source_size": args.curve_thumbnail_source_size,
        "curve_thumbnail_display_size": args.curve_thumbnail_display_size,
        "sequence_count": len(sequences),
        "frame_count": len(coordinate_rows),
        "curve_thumbnail_count": sum(len(seq["thumbnails"]) for seq in sequences),
        "compressed_sequence_count": sum(1 for seq in sequences if seq["compression"]["compressed"]),
        "plots": [{k: plot[k] for k in ("id", "title", "sequenceKeys", "domain")} for plot in plots],
        "manifest_csv": str(args.output_dir / "selected_cases_t_ranking_manifest.csv"),
        "adjusted_scores_csv": str(args.output_dir / "selected_cases_t_ranking_adjusted_scores.csv"),
        "method": (
            "Scores are read from frame_smile_ranking_scores.csv, previously computed as "
            "score_0_9 = clamp(sum_k P(frame stronger than true-smile scale anchor k) - 0.5, 0, 9). "
            "For opening compression, first growth is the first frame whose score exceeds "
            "frame-0 score by growth_delta. If that frame is later than max_initial_static_frames, "
            "frames from 0 to the growth frame are linearly compressed into 0..max_initial_static_frames."
        ),
    }
    summary_path = args.output_dir / "selected_cases_t_ranking_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
