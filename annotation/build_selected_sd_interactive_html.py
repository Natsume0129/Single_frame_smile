"""Build an interactive HTML s-d plot from selected-case coordinate CSV."""

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


DEFAULT_OUTPUT_DIR = Path(r"E:\Dataset\sd_plot_selected_cases")
CLASS_COLORS = {
    "bitter": ["#f36c5b", "#e45747", "#cf3d34", "#b72d2b", "#9d2024", "#82181f", "#661019"],
    "true": ["#8ec7ee", "#66aee0", "#4595d0", "#2b79b8", "#1f5f99", "#174a7b"],
    "polite": ["#8bd17c", "#69bf63", "#4ca955", "#368f48", "#26763c", "#195d30"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained interactive HTML plot for selected s-d cases."
    )
    parser.add_argument(
        "--coordinates_csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "selected_cases_sd_coordinates.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "selected_cases_sd_summary.json",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "selected_cases_sd_interactive.html",
    )
    parser.add_argument("--thumbnail_size", type=int, default=92)
    parser.add_argument("--jpeg_quality", type=int, default=82)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def encode_thumbnail(path: Path, size: int, jpeg_quality: int) -> str:
    with Image.open(path) as img:
        thumb = ImageOps.fit(img.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_sequences(rows: list[dict[str, str]], thumbnail_size: int, jpeg_quality: int) -> list[dict]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault((row["label"], row["sequence_id"]), []).append(row)

    label_counts: dict[str, int] = {}
    sequences = []
    for (label, sequence_id), seq_rows in grouped.items():
        color_index = label_counts.get(label, 0)
        label_counts[label] = color_index + 1
        palette = CLASS_COLORS.get(label, ["#333333"])
        color = palette[color_index % len(palette)]

        points = []
        thumbnails = []
        for row in seq_rows:
            frame_index = int(row["frame_index_from_onset"])
            s = float(row["s"])
            d = float(row["d"])
            frame_file = row["frame_file"]
            frame_path = row["frame_path"]
            points.append(
                {
                    "frame": frame_index,
                    "s": s,
                    "d": d,
                    "file": frame_file,
                    "path": frame_path,
                }
            )
            if row["is_thumbnail"] == "1":
                image_path = Path(frame_path)
                thumbnails.append(
                    {
                        "frame": frame_index,
                        "s": s,
                        "d": d,
                        "file": frame_file,
                        "src": encode_thumbnail(image_path, thumbnail_size, jpeg_quality),
                    }
                )

        sequences.append(
            {
                "key": f"{label}/{sequence_id}",
                "label": label,
                "sequenceId": sequence_id,
                "color": color,
                "frameCount": len(points),
                "points": points,
                "thumbnails": thumbnails,
            }
        )
    return sequences


def compute_domain(sequences: list[dict]) -> dict[str, float]:
    xs = [point["s"] for seq in sequences for point in seq["points"]]
    ys = [point["d"] for seq in sequences for point in seq["points"]]
    if not xs or not ys:
        raise RuntimeError("No coordinates found.")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.08, 0.2)
    ypad = max((ymax - ymin) * 0.08, 0.2)
    return {
        "xMin": xmin - xpad,
        "xMax": xmax + xpad,
        "yMin": max(0.0, ymin - ypad),
        "yMax": ymax + ypad,
    }


def html_document(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    escaped_title = html.escape(payload["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --text: #1f2933;
      --muted: #66717f;
      --grid: #d8dde3;
      --axis: #4b5563;
      --panel: #ffffff;
      --page: #f5f7f9;
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
      grid-template-columns: minmax(900px, 1fr) 280px;
      gap: 0;
    }}
    .plot-wrap {{
      padding: 18px 16px 18px 20px;
      overflow: auto;
    }}
    .plot-shell {{
      min-width: 1180px;
      background: #fff;
      border: 1px solid var(--border);
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    .title {{
      font-size: 18px;
      font-weight: 700;
      fill: var(--text);
    }}
    .subtitle, .axis-label, .tick text {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .axis-label {{
      font-size: 13px;
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
    .zero-line {{
      stroke: #777;
      stroke-width: 1.2;
      stroke-dasharray: 5 5;
      opacity: 0.6;
    }}
    .curve {{
      fill: none;
      stroke-linejoin: round;
      stroke-linecap: round;
      stroke-width: 1.7;
      opacity: 0.72;
      transition: opacity 120ms ease, stroke-width 120ms ease;
    }}
    .curve.is-muted {{
      opacity: 0.10;
      stroke-width: 1.2;
    }}
    .curve.is-active {{
      opacity: 1;
      stroke-width: 4.2;
    }}
    .hit-line {{
      fill: none;
      stroke: #fff;
      stroke-opacity: 0.001;
      stroke-width: 18;
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
      filter: drop-shadow(0 4px 7px rgba(15, 23, 42, 0.22));
    }}
    .thumb-label {{
      font-size: 10px;
      fill: #27313c;
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
      font-size: 16px;
      line-height: 1.25;
    }}
    .meta p {{
      margin: 3px 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .active-readout {{
      min-height: 56px;
      border: 1px solid var(--border);
      padding: 10px;
      margin-bottom: 12px;
      background: #fbfcfd;
      font-size: 13px;
      line-height: 1.45;
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
  </style>
</head>
<body>
  <div class="layout">
    <main class="plot-wrap">
      <div class="plot-shell">
        <svg id="plot" viewBox="0 0 1480 980" aria-label="Interactive s-d plot"></svg>
      </div>
    </main>
    <aside class="side">
      <div class="meta">
        <h1>Selected s-d Plot</h1>
        <p>Baseline axis: {html.escape(payload["baseline"])}</p>
        <p>Sequences: {payload["sequenceCount"]}; frames: {payload["frameCount"]}</p>
        <p>Thumbnails are embedded every {payload["thumbnailStride"]} frames.</p>
      </div>
      <div id="activeReadout" class="active-readout">No curve selected.</div>
      <div class="legend-title">Curves</div>
      <div id="legend"></div>
    </aside>
  </div>
  <script id="plot-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('plot-data').textContent);
    const svg = document.getElementById('plot');
    const legend = document.getElementById('legend');
    const readout = document.getElementById('activeReadout');
    const NS = 'http://www.w3.org/2000/svg';
    const W = 1480;
    const H = 980;
    const margin = {{ left: 88, top: 74, right: 34, bottom: 78 }};
    const plotW = W - margin.left - margin.right;
    const plotH = H - margin.top - margin.bottom;
    const xMin = data.domain.xMin;
    const xMax = data.domain.xMax;
    const yMin = data.domain.yMin;
    const yMax = data.domain.yMax;
    const thumbSize = 66;

    function el(name, attrs = {{}}, parent = svg) {{
      const node = document.createElementNS(NS, name);
      for (const [key, value] of Object.entries(attrs)) {{
        node.setAttribute(key, value);
      }}
      parent.appendChild(node);
      return node;
    }}

    function fmt(value) {{
      return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
    }}

    function xScale(value) {{
      return margin.left + ((value - xMin) / (xMax - xMin)) * plotW;
    }}

    function yScale(value) {{
      return margin.top + ((yMax - value) / (yMax - yMin)) * plotH;
    }}

    function clamp(value, low, high) {{
      return Math.max(low, Math.min(high, value));
    }}

    function polyline(points) {{
      return points.map(p => `${{xScale(p.s)}},${{yScale(p.d)}}`).join(' ');
    }}

    function tickValues(min, max, count) {{
      const values = [];
      for (let i = 0; i <= count; i += 1) {{
        values.push(min + (i / count) * (max - min));
      }}
      return values;
    }}

    el('rect', {{ x: 0, y: 0, width: W, height: H, fill: '#fff' }});
    el('text', {{ x: margin.left, y: 32, class: 'title' }}).textContent = data.title;
    el('text', {{ x: margin.left, y: 52, class: 'subtitle' }}).textContent = data.subtitle;

    const grid = el('g');
    tickValues(xMin, xMax, 8).forEach(value => {{
      const x = xScale(value);
      el('line', {{ x1: x, y1: margin.top, x2: x, y2: margin.top + plotH, class: 'grid-line' }}, grid);
      el('text', {{ x, y: margin.top + plotH + 24, 'text-anchor': 'middle', class: 'subtitle' }}, grid).textContent = fmt(value);
    }});
    tickValues(yMin, yMax, 7).forEach(value => {{
      const y = yScale(value);
      el('line', {{ x1: margin.left, y1: y, x2: margin.left + plotW, y2: y, class: 'grid-line' }}, grid);
      el('text', {{ x: margin.left - 12, y: y + 4, 'text-anchor': 'end', class: 'subtitle' }}, grid).textContent = fmt(value);
    }});

    if (yMin <= 0 && yMax >= 0) {{
      const y0 = yScale(0);
      el('line', {{ x1: margin.left, y1: y0, x2: margin.left + plotW, y2: y0, class: 'zero-line' }});
    }}
    el('line', {{ x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH, class: 'axis-line' }});
    el('line', {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH, class: 'axis-line' }});
    el('text', {{ x: margin.left + plotW / 2, y: H - 24, 'text-anchor': 'middle', class: 'axis-label' }}).textContent = 's: progress along true/5 first-to-last feature axis';
    const yLabel = el('text', {{ x: 24, y: margin.top + plotH / 2, 'text-anchor': 'middle', class: 'axis-label', transform: `rotate(-90 24 ${{margin.top + plotH / 2}})` }});
    yLabel.textContent = 'd: deviation from the true/5 feature axis';

    const curveLayer = el('g');
    const endpointLayer = el('g');
    const hitLayer = el('g');
    const thumbLayer = el('g');
    const curveNodes = new Map();
    const endpointNodes = new Map();
    let activeKey = null;

    data.sequences.forEach((seq, index) => {{
      const points = polyline(seq.points);
      const curve = el('polyline', {{
        points,
        class: 'curve',
        stroke: seq.color,
        'data-key': seq.key
      }}, curveLayer);
      curveNodes.set(seq.key, curve);

      const endpoints = [];
      const first = seq.points[0];
      const last = seq.points[seq.points.length - 1];
      endpoints.push(el('circle', {{ cx: xScale(first.s), cy: yScale(first.d), r: 4, fill: seq.color, class: 'endpoint' }}, endpointLayer));
      endpoints.push(el('rect', {{ x: xScale(last.s) - 4, y: yScale(last.d) - 4, width: 8, height: 8, fill: seq.color, class: 'endpoint' }}, endpointLayer));
      endpointNodes.set(seq.key, endpoints);

      const hit = el('polyline', {{
        points,
        class: 'hit-line',
        'data-key': seq.key
      }}, hitLayer);
      hit.addEventListener('mouseenter', () => focus(seq.key));
    }});

    function renderThumbnails(seq) {{
      thumbLayer.replaceChildren();
      seq.thumbnails.forEach(t => {{
        const cx = xScale(t.s);
        const cy = yScale(t.d);
        const x = clamp(cx - thumbSize / 2, margin.left, margin.left + plotW - thumbSize);
        const y = clamp(cy - thumbSize / 2, margin.top, margin.top + plotH - thumbSize);
        el('rect', {{ x: x - 2, y: y - 2, width: thumbSize + 4, height: thumbSize + 4, rx: 3, class: 'thumb-frame', stroke: seq.color }}, thumbLayer);
        el('image', {{ x, y, width: thumbSize, height: thumbSize, href: t.src, 'pointer-events': 'none' }}, thumbLayer);
        el('text', {{ x: x + 4, y: y + thumbSize - 5, class: 'thumb-label' }}, thumbLayer).textContent = `f${{t.frame}}`;
      }});
    }}

    function focus(key) {{
      activeKey = key;
      const seq = data.sequences.find(item => item.key === key);
      if (!seq) return;
      data.sequences.forEach(item => {{
        const curve = curveNodes.get(item.key);
        const endpoints = endpointNodes.get(item.key) || [];
        const isActive = item.key === key;
        curve.classList.toggle('is-active', isActive);
        curve.classList.toggle('is-muted', !isActive);
        endpoints.forEach(node => node.classList.toggle('is-muted', !isActive));
        const button = document.querySelector(`[data-legend-key="${{CSS.escape(item.key)}}"]`);
        if (button) button.classList.toggle('is-active', isActive);
      }});
      renderThumbnails(seq);
      readout.innerHTML = `<strong>${{seq.key}}</strong><br>${{seq.frameCount}} frames; ${{seq.thumbnails.length}} thumbnails<br><span style="color:${{seq.color}}">highlighted</span>`;
    }}

    function clearFocus() {{
      activeKey = null;
      thumbLayer.replaceChildren();
      data.sequences.forEach(item => {{
        const curve = curveNodes.get(item.key);
        curve.classList.remove('is-active', 'is-muted');
        (endpointNodes.get(item.key) || []).forEach(node => node.classList.remove('is-muted'));
        const button = document.querySelector(`[data-legend-key="${{CSS.escape(item.key)}}"]`);
        if (button) button.classList.remove('is-active');
      }});
      readout.textContent = 'No curve selected.';
    }}

    svg.addEventListener('mouseleave', clearFocus);
    svg.addEventListener('click', event => {{
      if (event.target.tagName.toLowerCase() === 'rect' && !event.target.classList.contains('thumb-frame')) {{
        clearFocus();
      }}
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
      legend.appendChild(groupTitle);
      seqs.forEach(seq => {{
        const button = document.createElement('button');
        button.className = 'legend-item';
        button.dataset.legendKey = seq.key;
        button.innerHTML = `<span class="swatch" style="background:${{seq.color}}"></span><span class="legend-main">${{seq.key}}<br><span class="legend-sub">${{seq.frameCount}} frames</span></span>`;
        button.addEventListener('mouseenter', () => focus(seq.key));
        button.addEventListener('focus', () => focus(seq.key));
        button.addEventListener('click', () => activeKey === seq.key ? clearFocus() : focus(seq.key));
        legend.appendChild(button);
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = read_rows(args.coordinates_csv)
    sequences = build_sequences(rows, args.thumbnail_size, args.jpeg_quality)

    summary = {}
    if args.summary_json.is_file():
        with args.summary_json.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    payload = {
        "title": "Selected sequence s-d plot using SmileComp feature vectors",
        "subtitle": (
            f"Baseline axis: {summary.get('baseline_label', 'true')}/"
            f"{summary.get('baseline_id', '5')}; thumbnail stride: "
            f"{summary.get('thumbnail_stride', 8)} frames"
        ),
        "baseline": f"{summary.get('baseline_label', 'true')}/{summary.get('baseline_id', '5')}",
        "thumbnailStride": summary.get("thumbnail_stride", 8),
        "sequenceCount": len(sequences),
        "frameCount": sum(seq["frameCount"] for seq in sequences),
        "domain": compute_domain(sequences),
        "sequences": sequences,
    }

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_document(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_html": str(args.output_html),
                "sequence_count": payload["sequenceCount"],
                "frame_count": payload["frameCount"],
                "thumbnail_count": sum(len(seq["thumbnails"]) for seq in sequences),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
