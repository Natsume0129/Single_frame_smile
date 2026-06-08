from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class InteractiveConfig:
    input_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve_interactive")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "InteractiveConfig":
        return cls(input_root=Path(args.input_root), output_root=Path(args.output_root))


class InteractiveNearestBaselineReport:
    def __init__(self, cfg: InteractiveConfig):
        self.cfg = cfg
        self.csv_dir = cfg.input_root / "csv"
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_csv(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def as_float(row: dict, key: str) -> float:
        return float(row[key])

    @staticmethod
    def sorted_points(rows: Iterable[dict]) -> list[dict]:
        return sorted(rows, key=lambda r: float(r["target_stage_percent"]))

    @staticmethod
    def safe_id(raw: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in raw)

    def load_rows(self) -> tuple[list[dict], list[dict]]:
        nearest6_rows = self.read_csv(self.csv_dir / "nearest6_nearest_baseline_curve_all.csv")
        prototype_rows = self.read_csv(self.csv_dir / "prototype_nearest_baseline_curve_all.csv")
        return nearest6_rows, prototype_rows

    def chart_data(
        self,
        nearest6_rows: list[dict],
        prototype_rows: list[dict],
        method: str,
        baseline_class: str,
        x_key: str,
        y_key: str,
    ) -> list[dict]:
        lines: list[dict] = []
        selected_nearest = [
            r for r in nearest6_rows if r["method"] == method and r["baseline_class"] == baseline_class
        ]
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in selected_nearest:
            grouped[(row["target_class"], row["rank"], row["sequence_id"])].append(row)

        for (target_class, rank, sequence_id), rows in sorted(
            grouped.items(), key=lambda item: (CLASS_NAMES.index(item[0][0]), int(item[0][1]))
        ):
            points = self.sorted_points(rows)
            label = (
                f"{method} | baseline={baseline_class} | {target_class} "
                f"nearest6 seq={sequence_id}, rank={rank}"
            )
            lines.append(
                {
                    "kind": "nearest6",
                    "target_class": target_class,
                    "label": label,
                    "points": [(self.as_float(r, x_key), self.as_float(r, y_key)) for r in points],
                }
            )

        selected_proto = [
            r for r in prototype_rows if r["method"] == method and r["baseline_class"] == baseline_class
        ]
        for target_class in CLASS_NAMES:
            rows = [r for r in selected_proto if r["target_class"] == target_class]
            if not rows:
                continue
            points = self.sorted_points(rows)
            label = f"{method} | baseline={baseline_class} | {target_class} prototype"
            lines.append(
                {
                    "kind": "prototype",
                    "target_class": target_class,
                    "label": label,
                    "points": [(self.as_float(r, x_key), self.as_float(r, y_key)) for r in points],
                }
            )

        return lines

    @staticmethod
    def path_from_points(points: list[tuple[float, float]], scale_x, scale_y) -> str:
        commands = []
        for i, (x, y) in enumerate(points):
            cmd = "M" if i == 0 else "L"
            commands.append(f"{cmd}{scale_x(x):.2f},{scale_y(y):.2f}")
        return " ".join(commands)

    @staticmethod
    def tick_values(max_value: float, fixed_100: bool) -> list[float]:
        if fixed_100:
            return [0, 20, 40, 60, 80, 100]
        if max_value <= 0:
            return [0, 1]
        step = max_value / 4.0
        return [round(step * i, 3) for i in range(5)]

    def render_chart(
        self,
        chart_id: str,
        title: str,
        lines: list[dict],
        x_label: str,
        y_label: str,
        x_fixed_100: bool,
        y_fixed_100: bool,
        include_diagonal: bool = False,
    ) -> str:
        width = 980
        height = 560
        left = 76
        right = 28
        top = 50
        bottom = 76
        plot_w = width - left - right
        plot_h = height - top - bottom

        all_x = [x for line in lines for x, _ in line["points"]]
        all_y = [y for line in lines for _, y in line["points"]]
        x_min, x_max = (0.0, 100.0) if x_fixed_100 else (0.0, max(all_x or [1.0]) * 1.06)
        y_min, y_max = (0.0, 100.0) if y_fixed_100 else (0.0, max(all_y or [1.0]) * 1.10)
        if y_max <= y_min:
            y_max = y_min + 1.0
        if x_max <= x_min:
            x_max = x_min + 1.0

        def sx(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value: float) -> float:
            return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

        x_ticks = self.tick_values(x_max, x_fixed_100)
        y_ticks = self.tick_values(y_max, y_fixed_100)

        parts = [
            f'<section class="chart-card" id="{html.escape(chart_id)}">',
            f"<h3>{html.escape(title)}</h3>",
            '<div class="chart-help">Hover any curve to highlight it. The label below the chart updates to the selected curve.</div>',
            f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" class="svg-bg"/>',
            f'<text x="{width / 2:.1f}" y="24" class="svg-title">{html.escape(title)}</text>',
        ]

        for tick in x_ticks:
            x = sx(float(tick))
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid-line"/>')
            parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" class="tick-label" text-anchor="middle">{tick:g}</text>')
        for tick in y_ticks:
            y = sy(float(tick))
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid-line"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick-label" text-anchor="end">{tick:g}</text>')

        parts.extend(
            [
                f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis-line"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis-line"/>',
                f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" class="axis-label" text-anchor="middle">{html.escape(x_label)}</text>',
                f'<text x="22" y="{top + plot_h / 2:.1f}" class="axis-label rotated" text-anchor="middle">{html.escape(y_label)}</text>',
            ]
        )

        if include_diagonal:
            d = self.path_from_points([(0.0, 0.0), (100.0, 100.0)], sx, sy)
            parts.append(f'<path d="{d}" class="reference-line"/>')

        for index, line in enumerate(lines):
            color = COLORS[line["target_class"]]
            dash = "6 5" if line["kind"] == "prototype" else ""
            width_class = "prototype" if line["kind"] == "prototype" else "nearest6"
            d = self.path_from_points(line["points"], sx, sy)
            curve_id = f"{chart_id}_curve_{index}"
            label = html.escape(line["label"], quote=True)
            target_class = html.escape(line["target_class"], quote=True)
            parts.extend(
                [
                    f'<g class="curve {width_class}" data-label="{label}" data-target-class="{target_class}" data-color="{color}" id="{html.escape(curve_id)}">',
                    f'<path class="curve-visible" d="{d}" stroke="{color}" stroke-dasharray="{dash}"/>',
                    f'<path class="curve-hit" d="{d}"/>',
                    "</g>",
                ]
            )

        parts.extend(
            [
                "</svg>",
                '<div class="legend">',
                '<span><i style="background:#1f77b4"></i>polite</span>',
                '<span><i style="background:#2ca02c"></i>truesmile</span>',
                '<span><i style="background:#ff7f0e"></i>ambiguous</span>',
                '<span><b class="legend-line"></b>nearest6 sequence</span>',
                '<span><b class="legend-dash"></b>prototype</span>',
                "</div>",
                '<div class="active-label">No curve selected</div>',
                "</section>",
            ]
        )
        return "\n".join(parts)

    def render_html(self, nearest6_rows: list[dict], prototype_rows: list[dict]) -> str:
        chart_specs = [
            {
                "suffix": "new_curve",
                "title": "Nearest-baseline new curve",
                "x_key": "nearest_baseline_progress_percent",
                "y_key": "nearest_distance",
                "x_label": "nearest baseline progress (%)",
                "y_label": "nearest vector length (L2)",
                "x_fixed_100": True,
                "y_fixed_100": False,
                "include_diagonal": False,
            },
            {
                "suffix": "progress",
                "title": "Nearest baseline progress over target stage",
                "x_key": "target_stage_percent",
                "y_key": "nearest_baseline_progress_percent",
                "x_label": "target stage on C_2 (%)",
                "y_label": "nearest baseline progress (%)",
                "x_fixed_100": True,
                "y_fixed_100": True,
                "include_diagonal": True,
            },
            {
                "suffix": "distance",
                "title": "Nearest distance over target stage",
                "x_key": "target_stage_percent",
                "y_key": "nearest_distance",
                "x_label": "target stage on C_2 (%)",
                "y_label": "nearest vector length (L2)",
                "x_fixed_100": True,
                "y_fixed_100": False,
                "include_diagonal": False,
            },
        ]

        body_parts = []
        for method in METHODS:
            for baseline_class in BASELINE_CLASSES:
                body_parts.append(f"<h2>{html.escape(method)} / baseline={html.escape(baseline_class)}</h2>")
                for spec in chart_specs:
                    chart_id = self.safe_id(f"{method}_{baseline_class}_{spec['suffix']}")
                    lines = self.chart_data(
                        nearest6_rows=nearest6_rows,
                        prototype_rows=prototype_rows,
                        method=method,
                        baseline_class=baseline_class,
                        x_key=spec["x_key"],
                        y_key=spec["y_key"],
                    )
                    body_parts.append(
                        self.render_chart(
                            chart_id=chart_id,
                            title=f"{spec['title']} | {method} | baseline={baseline_class}",
                            lines=lines,
                            x_label=spec["x_label"],
                            y_label=spec["y_label"],
                            x_fixed_100=spec["x_fixed_100"],
                            y_fixed_100=spec["y_fixed_100"],
                            include_diagonal=spec["include_diagonal"],
                        )
                    )

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive nearest-baseline curves</title>
<style>
body{{font-family:Arial,sans-serif;line-height:1.5;margin:28px;color:#222;background:#fafafa}}
main{{max-width:1180px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 8px}}
h2{{font-size:21px;margin:34px 0 14px;border-bottom:1px solid #ddd;padding-bottom:6px}}
h3{{font-size:17px;margin:0 0 4px}}
p{{max-width:920px}}
.filter-panel{{position:sticky;top:0;z-index:5;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0 12px;margin:12px 0 18px}}
.filter-title{{font-size:13px;color:#555;margin-bottom:8px}}
.filter-controls{{display:flex;flex-wrap:wrap;gap:10px}}
.filter-button{{display:inline-flex;align-items:center;gap:7px;border:1px solid #bbb;background:white;padding:7px 11px;font-size:13px;cursor:pointer;user-select:none}}
.filter-button input{{accent-color:#333}}
.filter-button[data-class="polite"]{{border-color:#1f77b4}}
.filter-button[data-class="truesmile"]{{border-color:#2ca02c}}
.filter-button[data-class="ambiguous"]{{border-color:#ff7f0e}}
.chart-card{{background:white;border:1px solid #ddd;margin:0 0 22px;padding:14px 16px 12px}}
.chart-help{{font-size:13px;color:#555;margin-bottom:8px}}
.chart-svg{{width:100%;height:auto;display:block;background:white}}
.svg-bg{{fill:white}}
.svg-title{{font-size:16px;font-weight:700;text-anchor:middle;fill:#222}}
.axis-line{{stroke:#222;stroke-width:1.2}}
.grid-line{{stroke:#ddd;stroke-width:1}}
.tick-label{{font-size:12px;fill:#555}}
.axis-label{{font-size:13px;fill:#333}}
.rotated{{transform:rotate(-90deg);transform-origin:22px center}}
.reference-line{{fill:none;stroke:#777;stroke-width:1.8;stroke-dasharray:2 5}}
.curve-visible{{fill:none;stroke-width:1.8;opacity:.36;vector-effect:non-scaling-stroke;transition:opacity .12s,stroke-width .12s}}
.curve.prototype .curve-visible{{stroke-width:2.4;opacity:.82}}
.curve-hit{{fill:none;stroke:transparent;stroke-width:13;pointer-events:stroke}}
.chart-card.is-hovering .curve-visible{{opacity:.08;stroke-width:1.2}}
.chart-card.is-hovering .curve.is-active .curve-visible{{opacity:1;stroke-width:4.2}}
.chart-card.is-hovering .curve.prototype.is-active .curve-visible{{stroke-width:4.8}}
.curve.is-filtered-out{{display:none}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 18px;margin-top:8px;font-size:13px;color:#444}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.legend i{{display:inline-block;width:12px;height:12px;border-radius:50%}}
.legend-line{{display:inline-block;width:26px;border-top:2px solid #555}}
.legend-dash{{display:inline-block;width:26px;border-top:2px dashed #555}}
.active-label{{font-size:13px;color:#222;background:#f5f5f5;border:1px solid #ddd;padding:7px 9px;margin-top:9px;min-height:18px}}
</style>
</head>
<body>
<main>
<h1>Interactive nearest-baseline curves</h1>
<p>This standalone HTML is based on the nearest-baseline CSV outputs. It is meant to solve the readability problem in nearest-6 plots: hover any curve to dim the others and highlight the selected sequence or prototype.</p>
<p>New-curve charts use x = nearest baseline progress and y = nearest-vector length. Progress charts use x = target stage and y = nearest baseline progress. Distance charts use x = target stage and y = nearest-vector length.</p>
<div class="filter-panel">
<div class="filter-title">Visible target classes</div>
<div class="filter-controls" aria-label="Visible target classes">
<label class="filter-button" data-class="polite"><input type="checkbox" data-class-filter="polite" checked> polite</label>
<label class="filter-button" data-class="truesmile"><input type="checkbox" data-class-filter="truesmile" checked> truesmile</label>
<label class="filter-button" data-class="ambiguous"><input type="checkbox" data-class-filter="ambiguous" checked> ambiguous</label>
</div>
</div>
{''.join(body_parts)}
</main>
<script>
function selectedTargetClasses() {{
  return new Set(Array.from(document.querySelectorAll('[data-class-filter]:checked')).map((input) => input.dataset.classFilter));
}}

function clearActiveCurves() {{
  document.querySelectorAll('.chart-card').forEach((card) => {{
    card.classList.remove('is-hovering');
    card.querySelectorAll('.curve').forEach((curve) => curve.classList.remove('is-active'));
    const labelBox = card.querySelector('.active-label');
    if (labelBox) {{
      labelBox.textContent = 'No curve selected';
    }}
  }});
}}

function applyClassFilters() {{
  const selected = selectedTargetClasses();
  document.querySelectorAll('.curve').forEach((curve) => {{
    curve.classList.toggle('is-filtered-out', !selected.has(curve.dataset.targetClass));
  }});
  clearActiveCurves();
}}

document.querySelectorAll('[data-class-filter]').forEach((input) => {{
  input.addEventListener('change', applyClassFilters);
}});

document.querySelectorAll('.chart-card').forEach((card) => {{
  const labelBox = card.querySelector('.active-label');
  card.querySelectorAll('.curve').forEach((curve) => {{
    curve.addEventListener('mouseenter', () => {{
      if (curve.classList.contains('is-filtered-out')) {{
        return;
      }}
      card.classList.add('is-hovering');
      card.querySelectorAll('.curve').forEach((other) => other.classList.remove('is-active'));
      curve.classList.add('is-active');
      curve.parentNode.appendChild(curve);
      labelBox.textContent = curve.dataset.label;
    }});
    curve.addEventListener('mouseleave', () => {{
      card.classList.remove('is-hovering');
      curve.classList.remove('is-active');
      labelBox.textContent = 'No curve selected';
    }});
  }});
}});

applyClassFilters();
</script>
</body>
</html>
"""

    def run(self) -> Path:
        nearest6_rows, prototype_rows = self.load_rows()
        html_text = self.render_html(nearest6_rows, prototype_rows)
        output_path = self.cfg.output_root / "interactive_nearest6_curves.html"
        output_path.write_text(html_text, encoding="utf-8")
        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate interactive SVG nearest-baseline nearest6 report.")
    parser.add_argument("--input_root", default=str(InteractiveConfig.input_root))
    parser.add_argument("--output_root", default=str(InteractiveConfig.output_root))
    return parser


def main() -> None:
    cfg = InteractiveConfig.from_args(build_arg_parser().parse_args())
    output_path = InteractiveNearestBaselineReport(cfg).run()
    print(f"interactive_html: {output_path}")


if __name__ == "__main__":
    main()
