"""Publish lazy-loaded full curves and a separate onset/peak review inventory."""
import hashlib,json,csv,subprocess,re
from collections import Counter
from pathlib import Path
from score_full_original_clips import OUT,read

def render_review(template, inventory, output_dir):
 def js_data(value):
  return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')

 clip_ids = {record['id'] for record in inventory['records']}
 def optional_records(filename):
  path = output_dir / filename
  if not path.is_file():
   return {}
  return {key: value for key, value in read(path).items() if key in clip_ids}

 values = {'__PAYLOAD__': inventory,
           '__AUTO__': optional_records('auto_boundary_proposals.json'),
           '__HUMAN_SEED__': optional_records('human_seed_decisions.json')}
 return re.sub(r'__PAYLOAD__|__AUTO__|__HUMAN_SEED__', lambda match: js_data(values[match.group()]), template)

def main():
 results=read(OUT/'results.json');sources=read(OUT/'sources.json');models=read(OUT/'models.json');assert len(results)==len(sources)
 originals={r['id']:r for r in sources};records=[]
 for r in results:
  curve=r['curve'];source=originals[r['id']];assert r['model_sha256']==models[r['person']+'/'+r['label']]['model_sha256']
  assert r['points']==source['tracking_frames']==len(curve)
  if curve:
   assert all(0<=p['raw_score']<=9 and 0<=p['smoothed_score']<=9 for p in curve)
   assert all(a['source_frame']<b['source_frame'] and a['seconds']<b['seconds'] for a,b in zip(curve,curve[1:]))
   assert all(abs(r['source_frame_times'][p['source_frame']]-p['seconds'])<1e-6 for p in curve)
  folder=Path(r['curve_file']).parent;script=folder/'curve.js'
  script.write_text('window.receiveCurve('+json.dumps(r['id'])+','+json.dumps(curve,ensure_ascii=False).replace('<','\\u003c')+');',encoding='utf8')
  records.append({**{k:v for k,v in r.items() if k!='curve'},'curve_js':script.relative_to(OUT).as_posix(),'tracked_source_frames':[p['source_frame'] for p in curve]})
 inv=dict(schema_version=1,task='full_clip_onset_peak_review',records=records,protocol=read(OUT/'protocol.json'))
 inv['dataset_id']=hashlib.sha256(json.dumps(inv,sort_keys=True).encode()).hexdigest()
 (OUT/'inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf8')
 template=Path(__file__).with_name('full_clip_onset_peak_review.html').read_text(encoding='utf8')
 html=render_review(template,inv,OUT)
 subprocess.run(['node','--check'],input=re.search(r'<script>(.*?)</script>',html,re.S)[1].encode(),check=True)
 (OUT/'index.html').write_text(html,encoding='utf8')
 summary=dict(clips=len(records),scored_frames=sum(r['points'] for r in records),by_person_category={p+'/'+l:dict(clips=sum(r['person']==p and r['label']==l for r in records),frames=sum(r['points'] for r in records if r['person']==p and r['label']==l)) for p,l in sorted({(r['person'],r['label']) for r in records})},insufficient_temporal_clips=[dict(person=r['person'],label=r['label'],filename=r['filename'],frames=r['points']) for r in records if r['points']<2])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 (OUT/'README.txt').write_text('Full original-clip ranking and manual onset/peak review\n\nOpen index.html. Select person/category, then set Onset and Peak on the chart or from video time. Frame times use original source PTS; boundaries snap to observed tracked frames. Accept requires onset<peak. Reject records an analysis decision only.\n\nEvery saved frame in the selected original-video face track was scored; no old start/peak interval or temporal subsampling is used. Old rejected ranking-training candidates and temporal-analysis exclusions in labelled source folders are included. Classification discards outside these folders are not silently relabelled. REN and MATSUDA are not included in the new 15-model inference.\n\nModels: matching person/category latest grouped accepted-only model. References: its matching train-only 10-level scale. Score=clamp(sum symmetric comparison probabilities-0.5,0,9). CSV also retains mean comparison probability. Median smoothing is only for display; raw scores are preserved. Untracked source frames have no score.\n\nExport ONE full_clip_onset_peak_*.json to this folder. Partial progress is exportable and can be re-imported. It is a separate review task; old monotonic training decisions are not overwritten. No video cutting, moves or training is triggered by this page/export.\n\nSources and model provenance: sources.json, models.json, protocol.json. Full CSVs and lazy JS data: curves/PERSON/LABEL/ID/. Results: results.json; summary: summary.json. Validate a later export using validate_full_clip_review.py EXPORT.json (read-only).\n',encoding='utf8')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
