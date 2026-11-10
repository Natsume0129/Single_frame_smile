"""Render four offline scientific reports; never train or select a classifier.

Input: report_data.json with protocol and reports keyed by general_3d,
personal_3d, general_18d, personal_18d. Each report supplies phase_results;
each phase result supplies person (null for pooled models), phase 0..9,
selected_model, train/validation/test summaries and test cells. A cell has
person, type, n_frames, n_clips, correct; null correct denotes unevaluated.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/single_frame_phase_classification_20261105"
KEYS = ("general_3d", "personal_3d", "general_18d", "personal_18d")
TITLES = {
    "general_3d": "General models · 3D frame responses",
    "personal_3d": "Personal models · 3D frame responses",
    "general_18d": "General models · 18D frame responses",
    "personal_18d": "Personal models · 18D frame responses",
}

STYLE = r"""
:root{color-scheme:light;--ink:#153047;--muted:#586b7a;--line:#d9e3eb;--blue:#155e97;--paper:#fff;--wash:#f2f6fa}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
a{color:var(--blue)}header,main,footer{max-width:1540px;margin:auto;padding:23px 28px}header{padding-bottom:7px}
nav{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:23px}nav a{background:#fff;border:1px solid var(--line);padding:7px 12px;border-radius:6px;text-decoration:none}nav a[aria-current=page]{background:var(--ink);color:#fff;border-color:var(--ink)}
h1{font-size:29px;line-height:1.25;margin:12px 0}h2{font-size:20px;margin:0 0 8px}h3{font-size:16px;margin:0 0 6px}p{margin:8px 0}.muted,.caption{color:var(--muted)}.caption{font-size:13px}.eyebrow{letter-spacing:.08em;font-size:12px;text-transform:uppercase;color:var(--muted)}
.card{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:20px;margin-bottom:18px}.protocol{border-left:4px solid #7196b4}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:15px 0}.stat{padding:15px 18px;background:white;border:1px solid var(--line);border-radius:9px}.stat strong{display:block;font-size:24px;font-variant-numeric:tabular-nums}.stat span{color:var(--muted);font-size:13px}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:end}.controls label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:4px}select,button{font:inherit;background:white;color:var(--ink);border:1px solid #b8cad8;border-radius:5px;padding:8px 10px}button{cursor:pointer}button:hover{border-color:var(--blue)}.controls button{font-size:13px}.scroll{overflow-x:auto}.heat{border-collapse:separate;border-spacing:4px;min-width:1060px;width:100%;table-layout:fixed}.heat th{font-size:12px;font-weight:500;color:var(--muted);padding:5px 3px;text-align:center}.heat th:first-child{width:160px;text-align:left}.heat tbody th{font-weight:600;color:var(--ink)}.heat td{padding:0;height:65px}.heat button{width:100%;height:65px;border:1px solid transparent;border-radius:5px;padding:5px 2px;color:inherit;background:transparent}.heat button strong{display:block;font-size:16px;font-variant-numeric:tabular-nums}.heat button small{display:block;font-size:10px;white-space:nowrap}.heat .none{background:repeating-linear-gradient(135deg,#f0f3f5,#f0f3f5 5px,#e7ecef 5px,#e7ecef 6px);color:#5e6e7a}.heat button:focus,.heat button.active{outline:3px solid #d29b16;outline-offset:-3px}.heat th em{font-style:normal;font-size:10px;display:block}.legend{display:flex;gap:9px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}.gradient{width:140px;height:12px;background:linear-gradient(to right,#edf4fb,#175b92);border-radius:3px}.detail{background:#f5f9fc;border:1px solid #c8dbe8;border-radius:7px;padding:12px;margin-top:12px;min-height:52px;font-size:13px}
.metrics{border-collapse:collapse;width:100%;min-width:1240px;font-size:12px}.metrics th{text-align:left;background:#eef4f9;color:#3a5369;white-space:nowrap}.metrics td,.metrics th{padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}.metrics tbody tr:hover{background:#f5f9fc}.metrics .num{font-variant-numeric:tabular-nums;white-space:nowrap}.metrics small{display:block;color:var(--muted)}.badge{display:inline-block;border-radius:4px;padding:2px 7px;background:#e8f2ec;color:#286444;font-size:11px}.badge.pending{background:#f6f0df;color:#806016}.notice{padding:10px 12px;border-radius:5px;background:#f7f0df;color:#6d541f;font-size:13px}.index-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.index-grid .card{margin:0}.index-grid a.open{display:inline-block;margin-top:12px;border:1px solid var(--blue);padding:7px 12px;text-decoration:none;border-radius:5px}code{font:12px/1.5 Consolas,monospace;overflow-wrap:anywhere}.definition{display:grid;grid-template-columns:165px 1fr;gap:5px 16px;font-size:13px}.definition dt{font-weight:600}.definition dd{margin:0;color:var(--muted)}footer{font-size:12px;color:var(--muted);padding-top:0}.empty{padding:24px;text-align:center;color:var(--muted)}
@media(max-width:850px){header,main,footer{padding-left:14px;padding-right:14px}h1{font-size:24px}.summary{grid-template-columns:repeat(2,1fr)}.index-grid{grid-template-columns:1fr}.card{padding:15px}.definition{grid-template-columns:1fr;gap:2px}.definition dd{margin-bottom:8px}}@media print{body{background:white}.controls,nav,button.export{display:none}.card{break-inside:avoid}.scroll{overflow:visible}.heat{min-width:0}.heat button small{font-size:8px}.metrics{min-width:0;font-size:9px}}
"""

SCRIPT = r"""
const DATA=__DATA__;
const TYPES=['polite','bitter','true'];
const TYPE_NAME={polite:'Polite',bitter:'Bitter',true:'True smile'};
const PERSONS=[...new Set(DATA.phase_results.flatMap(r=>(r.cells||[]).map(c=>c.person).concat(r.person?[r.person]:[])))].sort();
const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const number=x=>finite(x)?x.toLocaleString('en-US'):'—';
const pct=x=>finite(x)?(100*x).toFixed(1)+'%':'—';
const fmtPhase=i=>(i+1).toString().padStart(2,'0');
const range=i=>`${(i/10).toFixed(1)}–${((i+1)/10).toFixed(1)}`;
const cells=DATA.phase_results.flatMap(r=>(r.cells||[]).map(c=>({...c,phase:r.phase,result_person:r.person,selected_model:r.selected_model,status:r.status})));
let activeCell=null;
function pool(rows){
 const usable=rows.filter(c=>finite(c.correct)&&finite(c.n_frames)&&c.n_frames>0);
 const available=rows.reduce((n,c)=>n+(finite(c.n_frames)?c.n_frames:0),0);
 const scored=usable.reduce((n,c)=>n+c.n_frames,0),correct=usable.reduce((n,c)=>n+c.correct,0);
 const clipCells=usable.filter(c=>finite(c.n_clips)&&c.n_clips>0&&(finite(c.clip_accuracy_sum)||finite(c.clip_equal_accuracy)));
 const clipCount=clipCells.reduce((n,c)=>n+c.n_clips,0),clipSum=clipCells.reduce((n,c)=>n+(finite(c.clip_accuracy_sum)?c.clip_accuracy_sum:c.clip_equal_accuracy*c.n_clips),0);
 return {available,scored,correct,accuracy:scored?correct/scored:null,clip_accuracy:clipCount?clipSum/clipCount:null,n_clips:usable.reduce((n,c)=>n+(finite(c.n_clips)?c.n_clips:0),0),clip_count:clipCount,unscored:available-scored,models:[...new Set(rows.map(c=>c.selected_model).filter(Boolean))]};
}
function currentCells(){return cells.filter(c=>(!$('person').value||c.person===$('person').value)&&(!$('type').value||c.type===$('type').value));}
function currentResults(){return DATA.phase_results.filter(r=>!$('person').value||!r.person||r.person===$('person').value);}
function paint(value){
 if(!finite(value))return 'background:#edf1f4;color:#61717b';
 const low=[237,244,251],high=[23,91,146];
 const rgb=low.map((a,i)=>Math.round(a+(high[i]-a)*Math.max(0,Math.min(1,value))));
 return `background:rgb(${rgb});color:${value>.58?'white':'#18354b'}`;
}
function heatmap(target,byPerson){
 const source=currentCells(),types=$('type').value?[$('type').value]:TYPES;
 const people=$('person').value?[$('person').value]:PERSONS;
 const rowDefs=byPerson?people.flatMap(p=>types.map(t=>({person:p,type:t,label:p+' / '+TYPE_NAME[t]}))):types.map(t=>({person:null,type:t,label:TYPE_NAME[t]}));
 const bins=Array.from({length:10},(_,i)=>i);
 const vals=rowDefs.flatMap(r=>bins.map(b=>pool(source.filter(c=>c.type===r.type&&c.phase===b&&(!r.person||c.person===r.person)))));
 const maxN=Math.max(1,...vals.map(v=>v.scored));
 let index=0;
 const rows=rowDefs.map(r=>'<tr><th scope="row">'+esc(r.label)+'</th>'+bins.map(b=>{
  const value=vals[index++],mode=$('metric').value;
  const accuracy=mode==='frame'?value.accuracy:value.clip_accuracy;
  const shade=mode==='support'?(value.scored?value.scored/maxN:null):accuracy;
  const key=JSON.stringify({person:r.person,type:r.type,phase:b});
  const label=mode==='support'?number(value.scored):pct(accuracy);
  const detail=value.scored?`n=${number(value.scored)} · clips=${number(value.n_clips)}`:(value.available?'Not evaluated':'No samples');
  const title=value.scored?`Clip-equal ACC ${pct(value.clip_accuracy)}; ordinary frame ACC ${pct(value.accuracy)}; ${value.correct}/${value.scored} frames correct; ${value.n_clips} clips`:(value.available?`${value.available} available frames were not evaluated`:'No support in this cell');
  return `<td style="${paint(shade)}" class="${value.scored?'':'none'}"><button data-cell="${esc(key)}" title="${esc(title)}" class="${activeCell===key?'active':''}"><strong>${label}</strong><small>${detail}</small>${value.unscored?'<small>'+number(value.unscored)+' unscored</small>':''}</button></td>`;
 }).join('')+'</tr>').join('');
 $(target).innerHTML='<table class="heat"><thead><tr><th>'+(byPerson?'Participant / clip label':'Annotated clip label')+'</th>'+bins.map(i=>`<th scope="col">Bin ${fmtPhase(i)}<em>${range(i)}</em></th>`).join('')+'</tr></thead><tbody>'+rows+'</tbody></table>';
 $(target).querySelectorAll('[data-cell]').forEach(button=>button.onclick=()=>{
  activeCell=button.dataset.cell;const selected=JSON.parse(activeCell);const value=pool(currentCells().filter(c=>c.type===selected.type&&c.phase===selected.phase&&(!selected.person||c.person===selected.person)));
  $('detail').innerHTML=`<strong>${esc(selected.person||$('person').value||'All selected participants')} / ${TYPE_NAME[selected.type]} / Bin ${fmtPhase(selected.phase)} (${range(selected.phase)})</strong><br>`+(value.scored?`Primary clip-equal recall: ${pct(value.clip_accuracy)}. Ordinary frame recall: ${pct(value.accuracy)} (${number(value.correct)} / ${number(value.scored)} correct), from ${number(value.n_clips)} clips. `:'No evaluated frames. ')+(value.unscored?`${number(value.unscored)} available frames have no evaluated prediction. `:'')+`Selected classifier${value.models.length===1?'':'s'}: ${esc(value.models.join('; ')||'unavailable')}.`;
  $('phase').value=String(selected.phase);renderModels();heatmap('typeHeat',false);heatmap('personHeat',true);
 });
}
function metricSummary(value){return value||{};}
function renderModels(){
 const selected=$('phase').value;
 const results=currentResults().filter(r=>selected===''||r.phase===Number(selected));
 const rows=results.map(r=>{
  const train=metricSummary(r.train),val=metricSummary(r.validation),test=metricSummary(r.test);
  const ready=!!r.selected_model&&finite(test.accuracy);
  return '<tr><td>'+esc(r.person||'All participants')+'</td><td class="num">'+fmtPhase(r.phase)+'<small>'+range(r.phase)+'</small></td><td>'+esc(r.selected_model||'—')+'</td><td class="num">'+number(train.frames)+' / '+number(train.clips)+'<small>groups='+number(train.groups)+'</small></td><td class="num">'+pct(val.clip_equal_accuracy)+'<small>frame ACC '+pct(val.accuracy)+'<br>n='+number(val.frames)+'; clips='+number(val.clips)+'</small></td><td class="num">'+(finite(val.clip_equal_macro_f1)?val.clip_equal_macro_f1.toFixed(3):'—')+'<small>frame '+(finite(val.macro_f1)?val.macro_f1.toFixed(3):'—')+'</small></td><td class="num">'+pct(test.clip_equal_accuracy)+'<small>frame '+pct(test.accuracy)+'</small></td><td class="num">'+pct(test.clip_equal_balanced_accuracy)+'<small>frame '+pct(test.balanced_accuracy)+'</small></td><td class="num">'+(finite(test.clip_equal_macro_f1)?test.clip_equal_macro_f1.toFixed(3):'—')+'<small>frame '+(finite(test.macro_f1)?test.macro_f1.toFixed(3):'—')+'</small></td><td class="num">'+number(test.frames)+' / '+number(test.clips)+'<small>groups='+number(test.groups)+'</small></td><td><span class="badge '+(ready?'':'pending')+'">'+esc(r.status||(ready?'Evaluated':'Unavailable'))+'</span>'+(r.reason?'<small>'+esc(r.reason)+'</small>':'')+'</td></tr>';
 }).join('');
 $('modelRows').innerHTML=rows||'<tr><td colspan="11">No model results match this filter.</td></tr>';
 $('modelCount').textContent=`${results.length} phase/model result${results.length===1?'':'s'} shown. `+(DATA.mode==='general'&&$('person').value?'General-model metrics remain pooled; the heatmaps show the selected participant.':'Model metrics are reported for their original evaluation population.');
}
function render(){
 const value=pool(currentCells());
 $('accuracy').textContent=pct($('metric').value==='frame'?value.accuracy:value.clip_accuracy);$('accuracyLabel').textContent=$('metric').value==='frame'?'Ordinary test-frame accuracy':'Clip-equal ACC · pooled clip–bin contributions';$('testFrames').textContent=number(value.scored);
 $('availableFrames').textContent=number(value.available);$('models').textContent=number(currentResults().filter(r=>r.selected_model).length);
 $('scope').textContent=($('person').value||'All participants')+' · '+($('type').value?TYPE_NAME[$('type').value]:'All three clip labels');
 $('legendMetric').textContent=$('metric').value==='support'?'Evaluated frame count (relative within each heatmap)':($('metric').value==='frame'?'Ordinary frame recall: 0% → 100%':'Primary clip-equal recall: 0% → 100%');
 heatmap('typeHeat',false);heatmap('personHeat',true);renderModels();
}
function exportCells(){
 const rows=currentCells(),fields=['person','type','phase','n_frames','n_clips','correct','accuracy','clip_equal_accuracy','clip_accuracy_sum','selected_model','status'];
 const quote=x=>'"'+String(x??'').replace(/"/g,'""')+'"';
 const csv=[fields.join(','),...rows.map(r=>fields.map(k=>quote(k==='accuracy'?(finite(r.correct)&&r.n_frames>0?r.correct/r.n_frames:null):r[k])).join(','))].join('\r\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));a.download=DATA.key+'_test_cells.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
$('person').innerHTML='<option value="">All participants</option>'+PERSONS.map(p=>'<option>'+esc(p)+'</option>').join('');
$('phase').innerHTML='<option value="">All phases</option>'+Array.from({length:10},(_,i)=>`<option value="${i}">Bin ${fmtPhase(i)} · ${range(i)}</option>`).join('');
['person','type','metric'].forEach(id=>$(id).onchange=render);$('phase').onchange=renderModels;$('export').onclick=exportCells;render();
"""


def nav(current: str) -> str:
    entries = [("index", "Overview"), ("general_3d", "General · 3D"), ("personal_3d", "Personal · 3D"),
               ("general_18d", "General · 18D"), ("personal_18d", "Personal · 18D")]
    return '<nav aria-label="Experiment reports">' + ''.join(
        f'<a href="{key}.html"' + (' aria-current="page"' if key == current else '') + f'>{label}</a>'
        for key, label in entries) + '</nav>'


def protocol_html(protocol: dict, dimension: int) -> str:
    definitions = protocol.get("feature_space_definitions", {})
    feature_text = definitions.get(str(dimension), definitions.get(f"{dimension}d", f"{dimension} raw single-frame model responses."))
    if not isinstance(feature_text, str):
        feature_text = json.dumps(feature_text, ensure_ascii=False)
    items = [
        ("Frame representation", feature_text),
        ("Time bins", protocol.get("phase_rule", "Ten equal bins of normalized onset-to-peak elapsed time. Bins 01–09 are left-closed/right-open; bin 10 includes the endpoint. Time phase is not intensity level.")),
        ("Labels", protocol.get("label_rule", "Every tracked frame inherits its source clip's polite, bitter or true label; frames are not independently relabeled.")),
        ("Primary metric", "Clip-equal accuracy: compute frame correctness within each clip and time bin, then average those clip accuracies equally. Predictions remain per frame; this is not majority voting."),
        ("Selection", protocol.get("selection_rule", "Classifiers are fitted on training clips and selected using validation clip-equal accuracy, then clip-equal macro-F1.")),
        ("Test use", protocol.get("test_policy", "Selected models are frozen before test evaluation; interactive filters do not retrain or select models.")),
    ]
    return '<dl class="definition">' + ''.join(f'<dt>{html.escape(title)}</dt><dd>{html.escape(str(text))}</dd>' for title, text in items) + '</dl>'


def validate_report(key: str, report: dict):
    if not isinstance(report.get("phase_results"), list):
        raise ValueError(f"{key}: phase_results must be a list")
    seen = set()
    for row in report["phase_results"]:
        phase, person = row.get("phase"), row.get("person")
        if not isinstance(phase, int) or not 0 <= phase < 10:
            raise ValueError(f"{key}: invalid phase {phase}")
        model_key = (person, phase)
        if model_key in seen:
            raise ValueError(f"{key}: duplicate result {model_key}")
        seen.add(model_key)
        if key.startswith("general_") and person is not None:
            raise ValueError(f"{key}: general model rows require person=null")
        if key.startswith("personal_") and not person:
            raise ValueError(f"{key}: personal model result has no participant")
        for split in ("train", "validation", "test"):
            values = row.get(split) or {}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "clip_equal_accuracy", "clip_equal_balanced_accuracy", "clip_equal_macro_f1"):
                if values.get(metric) is not None and not 0 <= values[metric] <= 1:
                    raise ValueError(f"{key}: invalid {split} {metric}")
        cell_ids = set()
        for cell in row.get("cells", []):
            identity = (cell.get("person"), cell.get("type"))
            if identity in cell_ids or not identity[0] or identity[1] not in ("polite", "bitter", "true"):
                raise ValueError(f"{key}: invalid/duplicate person-type cell {identity}")
            cell_ids.add(identity)
            if person and cell["person"] != person:
                raise ValueError(f"{key}: personal cell belongs to another participant")
            n = cell.get("n_frames", 0)
            if not isinstance(n, int) or n < 0 or cell.get("n_clips", 0) < 0:
                raise ValueError(f"{key}: invalid support count")
            if cell.get("correct") is not None and not 0 <= cell["correct"] <= n:
                raise ValueError(f"{key}: correct count exceeds support")


def render_page(key: str, report: dict, protocol: dict, source_name: str) -> str:
    dimension = 18 if key.endswith("18d") else 3
    mode = "personal" if key.startswith("personal") else "general"
    report = dict(report, key=key, dimension=dimension, mode=mode)
    title = report.get("title") or TITLES[key]
    description = report.get("mode_description") or (
        "A separate classifier is selected for each participant and each normalized time bin."
        if mode == "personal" else "For each normalized time bin, one classifier is fitted using pooled training participants.")
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    notes = protocol.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    note_html = ''.join('<p>' + html.escape(str(note)) + '</p>' for note in notes)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head>
<body><header>{nav(key)}<div class="eyebrow">Single-frame classification · phase-specific evaluation</div><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p><p class="muted">Unsmoothed frame responses. Source clips remain in their frozen train / validation / test partitions. This is evaluation on already-personalized participants, not unseen-person generalization.</p></header>
<main>
<section class="card protocol"><h2>What this page measures</h2>{protocol_html(protocol,dimension)}<p class="caption">Primary heatmap cells are clip-equal class recall. The ordinary-frame option gives correct / evaluated frames. Zero-support or unevaluated cells are shown as “—”, never as 0%.</p></section>
<section class="card"><div class="controls"><label>Participant<select id="person"></select></label><label>Clip label<select id="type"><option value="">All three labels</option><option value="polite">Polite</option><option value="bitter">Bitter</option><option value="true">True smile</option></select></label><label>Heatmap display<select id="metric"><option value="clip_equal">Clip-equal recall · primary</option><option value="frame">Ordinary frame recall</option><option value="support">Evaluated frame count</option></select></label><button id="export">Download displayed cell data</button></div><p class="caption" id="scope"></p></section>
<div class="summary"><div class="stat"><strong id="accuracy">—</strong><span id="accuracyLabel">Clip-equal ACC · pooled clip–bin contributions</span></div><div class="stat"><strong id="testFrames">—</strong><span>Evaluated test frames</span></div><div class="stat"><strong id="availableFrames">—</strong><span>Available test frames in reported cells</span></div><div class="stat"><strong id="models">—</strong><span>Selected phase classifiers in scope</span></div></div>
<section class="card"><h2>Clip label × normalized time</h2><p class="caption">Counts are pooled across the participants selected above. A clip may contribute frames to several bins; these are frame counts, not independent observations.</p><div class="scroll" id="typeHeat"></div><div class="legend"><span class="gradient"></span><span id="legendMetric"></span><span>Hatched cell: no evaluated samples</span></div><div id="detail" class="detail">Click any cell to inspect its support and selected classifier. The model table will focus on that time bin.</div></section>
<section class="card"><h2>Participant × clip label × normalized time</h2><p class="caption">The same test predictions separated by participant. Models and splits remain fixed when filters change.</p><div class="scroll" id="personHeat"></div></section>
<section class="card"><h2>Validation selection and held-out results</h2><div class="controls"><label>Time bin<select id="phase"></select></label></div><p class="caption" id="modelCount"></p><div class="scroll"><table class="metrics"><thead><tr><th>Model population</th><th>Time bin</th><th>Selected classifier</th><th>Train frames / clips</th><th>Validation clip-equal ACC</th><th>Validation clip-equal mF1</th><th>Test clip-equal ACC</th><th>Test clip-equal balanced ACC</th><th>Test clip-equal mF1</th><th>Test frames / clips</th><th>Status</th></tr></thead><tbody id="modelRows"></tbody></table></div><p class="caption">Primary figures use equal clip weights within each phase; smaller “frame” figures use ordinary frame weights. ACC = accuracy; balanced ACC = class-balanced recall; mF1 = macro-F1. Clip-equal balanced ACC and mF1 are derived from the clip-weighted confusion matrix, not an average of individual clips’ F1 values. A dash means unavailable. The report does not average phase-specific macro-F1 values into an overall macro-F1.</p></section>
<section class="card"><h2>Interpretation</h2><p>Frames inherit clip-level labels; the task tests separability of those annotated clip categories. These measurements are not independently annotated frame-level affect labels.</p><p>Phase is normalized elapsed time from the reviewed onset to peak. It does not imply uniform smile intensity, equal frame spacing, or a monotonic ranking response. Every real frame is assigned once; no interpolation or frame duplication is used.</p><p>The summary across bins averages evaluated clip–bin contributions. A clip can contribute to several bins, so this summary is not a count or vote of unique clips. Training uses inverse frame-count weights within clip–bin groups, normalized to mean one; scaling is fitted using training data and weights only.</p>{note_html}<p class="caption"><a href="{html.escape(source_name)}">Machine-readable experiment results</a>. This offline viewer performs no training and no test-based model selection.</p></section>
</main><footer>Single-frame phase classification · 10 time bins · raw model responses · frozen clip partitions</footer><script>{SCRIPT.replace('__DATA__',serialized)}</script></body></html>'''


def pooled_test(report: dict):
    cells = [c for r in report["phase_results"] for c in r.get("cells", []) if c.get("correct") is not None and c.get("n_frames", 0) > 0]
    count = sum(c["n_frames"] for c in cells)
    correct = sum(c["correct"] for c in cells)
    return count, correct, correct / count if count else None


def pooled_clip_equal(report: dict, phase: int | None = None):
    cells = [c for r in report["phase_results"] if phase is None or r["phase"] == phase
             for c in r.get("cells", []) if c.get("correct") is not None and c.get("n_frames", 0) > 0
             and c.get("n_clips", 0) > 0 and (c.get("clip_accuracy_sum") is not None or c.get("clip_equal_accuracy") is not None)]
    count = sum(c["n_clips"] for c in cells)
    total = sum(c["clip_accuracy_sum"] if c.get("clip_accuracy_sum") is not None
                else c["clip_equal_accuracy"] * c["n_clips"] for c in cells)
    return total / count if count else None


def comparison_svg(data: dict, mode: str) -> str:
    width, height = 690, 300
    left, right, top, bottom = 59, 25, 35, 55
    plot_width, plot_height = width-left-right, height-top-bottom
    x = lambda i: left + i * plot_width / 9
    y = lambda value: top + (1-value) * plot_height
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{mode.capitalize()} 3D versus 18D clip-equal accuracy by time bin" style="width:100%;max-width:760px">']
    for value in (0, .25, .5, .75, 1):
        elements.append(f'<line x1="{left}" x2="{width-right}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="#d9e3eb"/><text x="{left-9}" y="{y(value)+4:.1f}" text-anchor="end" font-size="11" fill="#586b7a">{value:.0%}</text>')
    for i in range(10):
        elements.append(f'<text x="{x(i):.1f}" y="{height-bottom+19}" text-anchor="middle" font-size="11" fill="#586b7a">{i+1:02d}</text>')
    elements.append(f'<text x="{left+plot_width/2:.1f}" y="{height-9}" text-anchor="middle" font-size="12" fill="#586b7a">Normalized time bin (01 = earliest; 10 = latest)</text>')
    for dimension, color, legend_x in ((3, "#176ca4", left), (18, "#8054a4", left+120)):
        report = data["reports"][f"{mode}_{dimension}d"]
        values = [pooled_clip_equal(report, i) for i in range(10)]
        segments, segment = [], []
        for i, value in enumerate(values):
            if value is None:
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append((i, value))
        if segment:
            segments.append(segment)
        for segment in segments:
            coordinates = ' '.join(f'{x(i):.2f},{y(v):.2f}' for i, v in segment)
            elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for i, value in segment:
                elements.append(f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="4" fill="{color}"><title>{dimension}D, bin {i+1:02d}: {value:.2%} clip-equal accuracy</title></circle>')
        elements.append(f'<line x1="{legend_x}" x2="{legend_x+22}" y1="15" y2="15" stroke="{color}" stroke-width="3"/><text x="{legend_x+29}" y="19" font-size="12" fill="#153047">{dimension}D</text>')
    return ''.join(elements) + '</svg>'


def render_index(data: dict, source_name: str) -> str:
    cards = []
    for key in KEYS:
        report = data["reports"][key]
        frames, correct, accuracy = pooled_test(report)
        accuracy_text = f"{accuracy:.1%}" if accuracy is not None else "—"
        clip_accuracy = pooled_clip_equal(report)
        clip_text = f"{clip_accuracy:.1%}" if clip_accuracy is not None else "—"
        selected = sum(bool(r.get("selected_model")) for r in report["phase_results"])
        mode = "One model per person and time bin" if key.startswith("personal") else "Pooled participants; one model per time bin"
        cards.append(f'<article class="card"><h2>{html.escape(report.get("title") or TITLES[key])}</h2><p class="muted">{mode}</p><p><strong>{clip_text}</strong> clip-equal ACC over evaluated clip–bin contributions</p><p class="caption">Ordinary frame ACC: {accuracy_text} · {frames:,} evaluated test frames · {selected} selected phase classifiers</p><p class="caption">Explore clip-label × time and participant × clip-label × time recall, support, validation selection and held-out metrics.</p><a class="open" href="{key}.html">Open report</a></article>')
    protocol = data.get("protocol", {})
    split = protocol.get("split_counts", {})
    split_html = html.escape(json.dumps(split, ensure_ascii=False)) if split else "See the machine-readable experiment protocol."
    comparisons = ''.join('<section class="card"><h2>'+mode.capitalize()+' models · 3D versus 18D</h2><p class="caption">Primary test clip-equal accuracy by time bin; every prediction remains frame-level. Matching test frames are used in both dimensions. Hover over a point for its value.</p>'+comparison_svg(data,mode)+'</section>' for mode in ('general','personal'))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Single-frame classification by time phase</title><style>{STYLE}</style></head><body><header>{nav('index')}<div class="eyebrow">Four fixed experiments</div><h1>Single-frame classification by normalized time</h1><p>Compare pooled and personal classifiers using 3D and 18D raw frame responses. All four reports use ten normalized onset-to-peak time bins and preserve source-clip partitions.</p></header><main><section class="card protocol"><h2>Read these results as frame-level predictions of clip labels</h2><p>Frames inherit the annotated category of their source clip. No temporal smoothing is used in the frame features. Time bins represent elapsed-time fractions, not smile-intensity levels. The 3D input is the full-control subset of the 18D input; test rows match.</p><p>Primary accuracy averages frame correctness within each clip and bin, then weights those clip accuracies equally; it is not majority voting. Summaries across all bins pool clip–bin contributions, not unique clips. Ordinary frame accuracy remains available in every report.</p><p>Every classifier is chosen using validation clip-equal accuracy, then clip-equal macro-F1; held-out test predictions are reported after selection. Differences between these four predeclared reports should not be used for another round of test-driven tuning.</p><p class="caption">Frozen source split counts: <code>{split_html}</code></p></section><div class="index-grid">{''.join(cards)}</div><div class="index-grid" style="margin-top:18px">{comparisons}</div><section class="card" style="margin-top:18px"><h2>Reading the heatmaps</h2><p>Each cell shows class recall and the number of evaluated frames and contributing clips. Empty or unevaluated cells display “—”. Several frames can come from the same clip, so the frame count is not a count of independent observations.</p><p><a href="{html.escape(source_name)}">Download machine-readable results</a></p></section></main><footer>Reports are static artifacts with local interactive filters; no network connection or retraining is required.</footer></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT / "report_data.json")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if set(KEYS) - set(data.get("reports", {})):
        raise ValueError("Input must contain all four named reports")
    for key in KEYS:
        validate_report(key, data["reports"][key])
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.resolve().parent != args.output.resolve():
        source_name = "report_data.json"
        (args.output / source_name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    else:
        source_name = args.input.name
    for key in KEYS:
        (args.output / (key + ".html")).write_text(render_page(key, data["reports"][key], data.get("protocol", {}), source_name), encoding="utf-8")
    (args.output / "index.html").write_text(render_index(data, source_name), encoding="utf-8")
    print(json.dumps(dict(pages=[str(args.output / (key + ".html")) for key in ("index", *KEYS)],
                         phase_rows={key: len(data["reports"][key]["phase_results"]) for key in KEYS}), indent=2))


if __name__ == "__main__":
    main()
