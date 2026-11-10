"""Create an offline English report from the completed tensor-sequence experiment.

Rendering is read-only with respect to all training data, selections and models.
Plots are saved as standalone SVGs and embedded in the HTML as well.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/tensor_sequence_classification_20261109"
LABELS = ("polite", "bitter", "true")
LABEL_NAMES = {"polite": "Polite", "bitter": "Bitter", "true": "True smile"}
MODEL_NAMES = {"baseline": "Previous clip classifier", "raw73": "Raw 73-feature control", "fixed": "Fixed length", "native": "Native length"}
PALETTE = {"baseline": "#708091", "raw73": "#8466a1", "fixed": "#176e99", "native": "#b2592e"}


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return "—" if value is None else f"{100 * float(value):.1f}%"


def pp(value):
    return "—" if value is None else f"{100 * float(value):+.1f} pp"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def matrix(rows, model):
    cm = np.zeros((3, 3), dtype=int)
    for row in rows:
        cm[LABELS.index(row["label"]), LABELS.index(row[model])] += 1
    return cm


def metrics(rows, model):
    cm = matrix(rows, model)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3, float), where=support > 0)
    f1 = np.divide(2 * cm.diagonal(), support + predicted, out=np.zeros(3, float), where=(support + predicted) > 0)
    n = len(rows)
    return dict(n=n, correct=int(cm.trace()), accuracy=float(cm.trace() / n) if n else None,
                balanced_accuracy=float(recall[support > 0].mean()) if n else None,
                macro_f1=float(f1.mean()) if n else None)


def paired(rows, reference, candidate):
    a = np.array([r[reference] == r["label"] for r in rows], dtype=bool)
    b = np.array([r[candidate] == r["label"] for r in rows], dtype=bool)
    return dict(both_correct=int((a & b).sum()), gained=int((~a & b).sum()), lost=int((a & ~b).sum()),
                both_wrong=int((~a & ~b).sum()), delta=float(b.mean() - a.mean()) if len(rows) else None)


def save_figure(fig, target):
    fig.savefig(target, format="svg", bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def confusion_figure(rows, target):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), constrained_layout=True)
    for ax, key in zip(axes.flat, MODEL_NAMES):
        cm = matrix(rows, key)
        totals = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        for i in range(3):
            for j in range(3):
                value = f"{cm[i, j]}\n{frac[i, j]:.0%}" if totals[i, 0] else "—"
                ax.text(j, i, value, ha="center", va="center", fontsize=11,
                        color="white" if frac[i, j] > .58 else "#183047")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=[LABEL_NAMES[x] for x in LABELS],
               yticklabels=[LABEL_NAMES[x] for x in LABELS], xlabel="Predicted category",
               title=MODEL_NAMES[key])
        ax.tick_params(length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Annotated clip category")
    save_figure(fig, target)


def participants_figure(rows, target):
    people = sorted({r["person"] for r in rows})
    fig, ax = plt.subplots(figsize=(10.2, 4), constrained_layout=True)
    xp = np.arange(len(people))
    for i, model in enumerate(MODEL_NAMES):
        yy = [metrics([r for r in rows if r["person"] == p], model)["accuracy"] for p in people]
        ax.scatter(xp + (i - 1.5) * .14, yy, label=MODEL_NAMES[model], color=PALETTE[model], s=58,
                   marker=("o", "D", "s", "^")[i], zorder=3)
    ax.set(xticks=xp, xticklabels=[p + f"\n(n={sum(r['person'] == p for r in rows)})" for p in people],
           yticks=np.linspace(0, 1, 6), yticklabels=[f"{t:.0%}" for t in np.linspace(0, 1, 6)],
           ylim=(-.03, 1.03), ylabel="Test clip accuracy")
    ax.grid(axis="y", color="#dce5ed", zorder=0)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False, fontsize=9)
    save_figure(fig, target)


def input_figure(example, target):
    """Visualize real, pre-standardization native and fixed matrices, time on x."""
    channels = example["columns"]
    if len(channels) != 18:
        raise ValueError("Expected exactly 18 ranking-response columns for the real example")
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.4), squeeze=False,
                             gridspec_kw={"height_ratios": [18, 2.3]}, constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bcc3ca")
    for j, role in enumerate(("native", "fixed")):
        source = example[role]
        values = np.asarray(source["x"], dtype=float)
        times = np.asarray(source["times"], dtype=float)
        if values.ndim != 2 or values.shape != (len(times), 18) or len(times) < 2 or not np.isfinite(times).all():
            raise ValueError(f"Invalid {role} example dimensions/timestamps")
        valid = np.asarray(source.get("valid", source.get("mask", np.ones(len(times), bool))), dtype=bool)
        if valid.shape != (len(times),) or not np.isfinite(values[valid]).all():
            raise ValueError(f"Invalid {role} example values/mask")
        if role == "native" and not valid.all():
            raise ValueError("Untracked/missing source frames cannot appear as native observations")
        values = np.ma.array(values, mask=np.broadcast_to(~valid[:, None], values.shape))
        ax = axes[0, j]
        im = ax.imshow(values.T, vmin=0, vmax=9, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set(yticks=range(18), yticklabels=[str(c).replace('__',' / ') for c in channels] if j == 0 else [],
               title=f"{role.capitalize()} length: {len(times)} steps × 18 responses")
        ax.tick_params(axis="y", length=0, labelsize=7)
        for k in range(3, 18, 3):
            ax.axhline(k - .5, color="white", lw=.7, alpha=.8)
        xticks = sorted({0, max(0, (len(values) - 1)//2), len(values) - 1})
        ax.set(xticks=xticks, xticklabels=[str(i+1) for i in xticks], xlabel="Sequence step")
        ax2 = axes[1, j]
        elapsed = times - times[0]
        ax2.scatter(elapsed[valid], np.zeros(valid.sum()), marker="|", s=145, color=PALETTE[role])
        if (~valid).any():
            ax2.scatter(elapsed[~valid], np.zeros((~valid).sum()), marker="x", s=30, c="#89959c")
        ax2.set(yticks=[], ylim=(-1, 1), xlim=(-.015 * example['duration'], 1.015 * example['duration']),
                xlabel="Time from onset (s); one tick per supported step")
        ax2.spines[['left','top','right']].set_visible(False)
        ax2.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=.8, ticks=[0, 3, 6, 9], label="Raw ranking response (0–9)")
    save_figure(fig, target)


STYLE = """
:root{--ink:#183044;--muted:#546b7c;--line:#d6e1e9;--blue:#16658d;--warm:#a7512e}*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}header,main,footer{max-width:1210px;margin:auto;padding:18px 26px}header{padding-top:30px;padding-bottom:0}h1{font-size:clamp(27px,3.7vw,42px);line-height:1.12;margin:9px 0 14px}h2{margin:0 0 8px;font-size:22px}h3{font-size:16px;margin:13px 0 5px}p{margin:7px 0}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--muted)}a{color:var(--blue)}nav{display:flex;gap:13px;flex-wrap:wrap;margin:16px 0 4px;font-size:13px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:19px 22px;margin:14px 0}.lead{background:#eaf3f7;border-left:5px solid var(--blue)}.muted,figcaption,.small{font-size:13px;color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.stat{padding:13px 15px;background:white;border:1px solid var(--line);border-radius:8px}.stat b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.stat span{font-size:12px;color:var(--muted)}figure{margin:14px 0}.plot svg{display:block;max-width:100%;width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px 9px;border-bottom:1px solid #e0e7ec}th{background:#edf3f7;white-space:nowrap}td.num{font-variant-numeric:tabular-nums}tr.primary{background:#eaf5f8}td small{display:block;color:var(--muted)}.scroll{overflow-x:auto}.checks{display:grid;grid-template-columns:1fr 1fr;gap:12px}.checks div{border-left:3px solid #ccdbe4;padding:5px 12px}.badge{font-size:11px;background:#deeff4;color:#125875;padding:2px 6px;border-radius:4px;white-space:nowrap}summary{cursor:pointer;color:var(--blue);font-weight:600}details{margin-top:11px}.tools{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}select{font:inherit;padding:5px 7px;border:1px solid #bdcbd4;background:white;color:var(--ink);border-radius:4px}code{font:12px/1.5 Consolas,monospace;overflow-wrap:anywhere}footer{padding-top:0;color:var(--muted);font-size:12px}@media(max-width:800px){header,main,footer{padding-left:13px;padding-right:13px}.stats,.checks{grid-template-columns:repeat(2,1fr)}.card{padding:15px}.scroll table{min-width:600px}}@media print{nav,.tools{display:none}body{background:white}.card{break-inside:avoid}}
"""


def embed(path):
    svg = path.read_text(encoding="utf-8")
    return '<div class="plot">' + svg[svg.index('<svg'):].strip() + '</div>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell + '</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def show_model(value):
    if value is None:
        return '—'
    if isinstance(value, dict):
        for key in ("model_id", "name", "id", "model", "family"):
            if key in value and isinstance(value[key], (str, int, float)):
                return str(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def compact_ci(value):
    if value is None:
        return '—'
    if isinstance(value, dict):
        value = (value.get('low', value.get('lower')), value.get('high', value.get('upper')))
    if len(value) != 2 or any(x is None for x in value):
        return '—'
    return f"[{float(value[0]):+.1f}, {float(value[1]):+.1f}] pp"


def check_data(data):
    primary = data['primary_arm']
    if primary not in ('fixed', 'native'):
        raise ValueError('Expected frozen validation decision: fixed or native')
    arms = data['arms']
    sources = {'baseline':data['baseline'], 'raw73':data['raw73_control'], 'fixed':arms['fixed'], 'native':arms['native']}
    rows = []
    for record in data['test_predictions']:
        row = {k:record[k] for k in ('id','person','label','component')}
        row.update({k:record[k+'_prediction'] for k in MODEL_NAMES})
        if any(row[k] not in LABELS for k in ('label', *MODEL_NAMES)):
            raise ValueError(f"Unknown annotated/predicted category: {record['id']}")
        rows.append(row)
    if not rows or len(rows) != len({r['id'] for r in rows}):
        raise ValueError('Need one matched prediction per unique held-out clip ID')
    for model, source in sources.items():
        actual = metrics(rows, model)
        declared = source['test']
        for key in ('n','correct','accuracy','balanced_accuracy','macro_f1'):
            if key in declared and not np.isclose(actual[key], declared[key], atol=1e-7):
                raise ValueError(f"{model}: declared test {key} does not match its predictions")
        if 'confusion_matrix' in declared and not np.array_equal(matrix(rows, model), declared['confusion_matrix']):
            raise ValueError(f"{model}: confusion/label order mismatch")
    for key in ('fixed', 'native'):
        values = paired(rows, 'baseline', key)
        audit = data['paired'][key]
        if (audit['corrected'], audit['new_errors']) != (values['gained'], values['lost']):
            raise ValueError(f"{key}: disagreement with provided paired counts")
    return rows, sources


def render(data, output, source_filename):
    rows, sources = check_data(data)
    primary = data['primary_arm']
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    example = data['input_example']
    figure_paths = dict(input=figures/'sequence_input_example.svg', confusion=figures/'test_confusions.svg',
                        participants=figures/'test_by_participant.svg')
    input_figure(example, figure_paths['input'])
    confusion_figure(rows, figure_paths['confusion'])
    participants_figure(rows, figure_paths['participants'])
    def figure(name, caption):
        path = figure_paths[name]
        return '<figure>'+embed(path)+'<figcaption>'+esc(caption)+' · <a href="'+path.relative_to(output).as_posix()+'" download>Download SVG</a></figcaption></figure>'
    splitrows = []
    for split, label in (('train','Train'), ('val','Validation'), ('test','Test')):
        value = data['split_counts'][split]
        if isinstance(value, dict):
            count = value.get('clips', value.get('n', value.get('count')))
            groups = value.get('groups', value.get('components'))
            frames = value.get('frames')
        else:
            count, groups, frames = value, None, None
        splitrows.append([label, esc(count) if count is not None else '—', esc(groups) if groups is not None else '—',
                          f'{frames:,}' if frames is not None else '—'])
    resultrows = []
    for model in MODEL_NAMES:
        source = sources[model]
        vals = source.get('val') or {}
        results = metrics(rows, model)
        method = esc(show_model(source.get('selected_model', source.get('name', MODEL_NAMES[model]))))
        descriptor = esc(MODEL_NAMES[model])
        if model == primary:
            descriptor += ' <span class="badge">Validation selection</span>'
        resultrows.append([descriptor, method, pct(vals.get('accuracy')), f"{vals['macro_f1']:.3f}" if vals.get('macro_f1') is not None else '—',
                           f"{results['correct']}/{results['n']}", pct(results['accuracy']),
                           pct(results['balanced_accuracy']), f"{results['macro_f1']:.3f}"])
    family_rows = []
    for family, title in (('kernel','Functional embedding + RBF-SVM'),('bigru','Bidirectional GRU'),('tcn','Masked dilated TCN')):
        values = [esc(title)]
        for arm in ('fixed','native'):
            result = data['family_comparison'][arm][family]
            val, test = result['val'],result['test']
            values += [esc(show_model(result['selected_model'])), pct(val['accuracy'])+f"<small>macro-F1 {val['macro_f1']:.3f}</small>",
                       pct(test['accuracy'])+f"<small>{int(test['correct'])}/{int(test['n'])}; macro-F1 {test['macro_f1']:.3f}</small>"]
        family_rows.append(values)
    pairrows = []
    for reference, candidate in [('baseline','fixed'), ('baseline','native'), ('fixed','native')]:
        values = paired(rows, reference, candidate)
        ci = data['paired'][candidate].get('ci95_pp') if reference == 'baseline' else data.get('fixed_vs_native',{}).get('ci95_pp')
        pairrows.append([esc(MODEL_NAMES[reference])+' → '+esc(MODEL_NAMES[candidate]), str(values['both_correct']),
                         str(values['gained']), str(values['lost']), str(values['both_wrong']), pp(values['delta']), compact_ci(ci)])
    people = sorted({r['person'] for r in rows})
    by_person = []
    for person in people:
        subset = [r for r in rows if r['person'] == person]
        entry = [esc(person), str(len(subset))]
        for model in MODEL_NAMES:
            values = metrics(subset, model)
            entry.append(f"{values['correct']}/{values['n']} · {pct(values['accuracy'])}")
        change = paired(subset, 'baseline', primary)
        entry.extend([f"+{change['gained']} / −{change['lost']}", pp(change['delta'])])
        by_person.append(entry)
    best = metrics(rows,primary)
    old = metrics(rows,'baseline')
    delta = paired(rows,'baseline',primary)
    direction = "more" if best['correct'] > old['correct'] else "fewer" if best['correct'] < old['correct'] else None
    difference = (f"{abs(best['correct'] - old['correct'])} {direction} correct" if direction else "the same number correct")
    inputs = figure('input', 'One column is one sequence step. The two response matrices use the same 0–9 color scale. The tick plots underneath use elapsed seconds; unequal spacing is visible for native observations. Gray marks, when present, denote unsupported fixed-grid locations, not zero responses.')
    matrices = figure('confusion', 'Counts and percentages of each annotated category (row). Diagonal entries are correct. All four panels use the exact same test clips and row denominators.')
    participants = figure('participants', 'One prediction per held-out clip for each model. The people were represented during training; this is not a test of generalization to unseen participants.')
    native_n = len(example['native']['times']); fixed_n = len(example['fixed']['times'])
    length = data['length_audit']
    gap_note = (f"Across {length['native_frames']:,} tracked observations, native lengths ranged from "
                f"{length['native_length_min']} to {length['native_length_max']} (median {length['native_length_median']:g}). "
                f"{length['clips_with_gaps']} clips had gaps exceeding 0.15 s; {length['masked_fixed_slots']} fixed-grid slots were masked.")
    rowsjson=json.dumps(rows, ensure_ascii=False, separators=(',',':')).replace('<','\\u003c')
    selectperson=''.join(f'<option>{esc(p)}</option>' for p in people)
    protocol_html=esc(json.dumps(data['protocol'],ensure_ascii=False,indent=2))
    lead = f"Validation selected the <strong>{esc(MODEL_NAMES[primary].lower())}</strong> arm. On the matched held-out clips it classified <strong>{best['correct']}/{best['n']} ({pct(best['accuracy'])})</strong> correctly, compared with {old['correct']}/{old['n']} ({pct(old['accuracy'])}) for the previous classifier: {difference} ({pp(delta['delta'])})."
    output_html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Clip classification from smile-response sequences</title><style>{STYLE}</style></head><body>
<header><div class="eyebrow">Clip-level temporal classification · fixed versus native sequence length</div><h1>Classifying smiles from onset to peak</h1><p>Does retaining the full sequence of regional ranking responses improve one prediction per clip? Fixed-grid and native-length representations are evaluated on the same source-clip partition.</p><nav><a href="#inputs">Inputs</a><a href="#results">Validation and test</a><a href="#pairs">Paired changes</a><a href="#people">Participants</a><a href="#confusions">Confusions</a><a href="#limits">Interpretation</a></nav></header>
<main><section class="card lead"><h2>Primary comparison</h2><p>{lead}</p><p class="small">The primary sequence arm was selected by validation accuracy, then validation macro-F1. Test accuracy did not choose the arm. Both arms are reported as originally evaluated.</p></section>
<div class="stats"><div class="stat"><b>{len(rows)}</b><span>Matched held-out clips</span></div><div class="stat"><b>{len({x['component'] for x in rows})}</b><span>Held-out source components</span></div><div class="stat"><b>{delta['gained']} / {delta['lost']}</b><span>Prior errors corrected / new errors</span></div><div class="stat"><b>{pp(delta['delta'])}</b><span>Primary minus previous accuracy</span></div></div>
<section class="card" id="inputs"><h2>What goes into each model?</h2><p>For one clip, read the 18 raw ranking responses at each tracked frame between its reviewed onset and peak. The channels are six face inputs (full, periocular, nose, mouth and corners, upper face, lower face) × three person-matched ranking models (polite, bitter, true).</p><div class="checks"><div><h3>Fixed length</h3><p>Interpolate the observed response channels using their recorded timestamps onto 20 positions spanning the reviewed interval. Interior grid positions across tracking gaps longer than 0.15 s are unsupported; they are masked or excluded rather than treated as observations.</p></div><div><h3>Native length</h3><p>Keep every observed tracked frame in order. Clips can have different numbers and spacings of steps; batching must not turn padding into facial evidence.</p></div></div><p>Both arms have access to normalized time and actual time increments, plus global duration and endpoint response changes. Inputs are normalized using training data only. The frozen 73-feature comparator used smoothed curve summaries; the raw 73-feature control helps distinguish that choice from the sequence representation.</p>
<p>{esc(gap_note)}</p><h3>One real input, represented both ways</h3><p class="muted">{esc(example['person'])} / {esc(LABEL_NAMES.get(example['label'],example['label']))} · clip <code>{esc(example['id'])}</code> · onset→peak {float(example['duration']):.3f} s · {native_n} native observations, {fixed_n} supplied fixed-grid positions.</p>{inputs}</section>
<section class="card" id="results"><h2>Validation selection and held-out clip accuracy</h2><p>The sequence candidates included bidirectional GRU, masked temporal convolution and functional-summary RBF-SVM approaches. Each arm's classifier and the primary choice between the two sequence arms were selected using validation only. The previous classifier and raw 73-feature model are separate comparators. “—” means not supplied or not applicable.</p>{table(['Representation','Selected classifier / control','Val ACC','Val macro-F1','Test correct','Test ACC','Test balanced ACC','Test macro-F1'],resultrows)}<h3>Matched comparisons within each model family</h3><p class="small">Each entry is that family’s validation-selected configuration for its representation. All test metrics were computed after the global selection freeze. A higher test result in this secondary table does not replace the prespecified primary model.</p>{table(['Family','Fixed configuration','Fixed val ACC','Fixed test ACC','Native configuration','Native val ACC','Native test ACC'],family_rows)}<h3>Frozen partitions</h3>{table(['Split','Source clips','Source components','Tracked frames'],splitrows)}<p class="small">ACC is per-clip classification accuracy, not single-frame ranking accuracy. A clip’s frames are not counted as independent test samples.</p></section>
<section class="card" id="pairs"><h2>Which errors changed?</h2><p>Read each row left to right. “Corrected” means the first model was wrong and the second was correct on that same clip. “New errors” means the reverse. The net difference is (corrected − new errors) divided by {len(rows)}.</p>{table(['Reference → candidate','Both correct','Corrected','New errors','Both wrong','ACC difference','Paired 95% interval'],pairrows)}<p class="small">Paired 95% intervals are supplied by the experiment: 2,000 bootstrap resamples of held-out source components, keeping clips within each component together.</p>
<details><summary>Inspect individual matched test predictions</summary><div class="tools"><label>Participant <select id="person"><option value="">All</option>{selectperson}</select></label><label>Category <select id="category"><option value="">All</option><option value="polite">Polite</option><option value="bitter">Bitter</option><option value="true">True smile</option></select></label><label>Previous → primary <select id="outcome"><option value="change">Corrected or new error</option><option value="gained">Corrected</option><option value="lost">New error</option><option value="all">All clips</option></select></label></div><p id="shown" class="small"></p><div class="scroll"><table><thead><tr><th>Clip ID</th><th>Participant</th><th>Annotated</th><th>Previous</th><th>Raw 73</th><th>Fixed</th><th>Native</th><th>Previous → primary</th></tr></thead><tbody id="cases"></tbody></table></div></details></section>
<section class="card" id="people"><h2>By participant</h2>{participants}{table(['Participant','Test clips','Previous','Raw 73','Fixed','Native','Primary +correct / −correct','Primary Δ ACC'],by_person)}</section>
<section class="card" id="confusions"><h2>Which categories were confused?</h2>{matrices}</section>
<section class="card" id="limits"><h2>What the comparison establishes</h2><p>These are labeled clips from participants with their own ranking models. The inputs use reviewed onset and peak, including the end of the rise. The result describes offline classification of a complete reviewed interval; it does not establish online early recognition or independent affect labels for individual frames.</p><p>The test cohort had already been inspected in previous experiments. The comparison is paired and exploratory, not a fresh confirmatory holdout. The 18 channels are separate models’ ordinal responses, not calibrated smile-category probabilities. Small participant-level differences may represent only one or two clips.</p><details><summary>Recorded experiment protocol</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">{protocol_html}</pre></details><p class="small"><a href="{esc(source_filename)}">Machine-readable results</a> · <a href="../three_model_clip_gallery_20261102/index.html">Reviewed clip gallery</a></p></section></main>
<footer>Static research report · one prediction per clip · original data, reviewed boundaries and fitted models are not changed by this page.</footer>
<script>const records={rowsjson}, primary={json.dumps(primary)};const $=id=>document.getElementById(id);function render(){{const body=$('cases');body.textContent='';let n=0;for(const r of records){{const prior=r.baseline===r.label,now=r[primary]===r.label;const value=!prior&&now?'gained':prior&&!now?'lost':'same';if(($('person').value&&$('person').value!==r.person)||($('category').value&&$('category').value!==r.label))continue;const filter=$('outcome').value;if((filter==='change'&&value==='same')||(['gained','lost'].includes(filter)&&value!==filter))continue;const tr=document.createElement('tr');for(const text of [r.id,r.person,r.label,r.baseline,r.raw73,r.fixed,r.native,value==='gained'?'Corrected':value==='lost'?'New error':prior?'Both correct':'Both wrong']){{const td=document.createElement('td');td.textContent=text;tr.appendChild(td)}}body.appendChild(tr);n++}}$('shown').textContent=n+' matching clip'+(n===1?'':'s')}}for(const key of ['person','category','outcome'])$(key).onchange=render;render();</script></body></html>'''
    (output / 'index.html').write_text(output_html,encoding='utf-8')
    return dict(primary=primary, test_clips=len(rows), models={m:metrics(rows,m) for m in MODEL_NAMES},
                primary_paired=delta, figures=[str(p) for p in figure_paths.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=OUT/'report_data.json')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Completed report data is required; not rendering placeholders: {args.input}')
    destination=args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_name=Path(args.input).resolve().relative_to(destination).as_posix() if Path(args.input).resolve().is_relative_to(destination) else Path(args.input).resolve().as_uri()
    summary=render(read(args.input),destination,source_name)
    print(json.dumps(dict(page=str(destination/'index.html'),**summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
