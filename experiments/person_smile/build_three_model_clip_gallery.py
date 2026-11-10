"""Render reviewed onset/peak clips and score the same tracked frames with all three personal models."""
import argparse,csv,gc,hashlib,json,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
import numpy as np
from pipeline import ROOT,digest,frame_number,save_json,uniform_indices
from onset_peak import Scorer
from validate_full_clip_review import validate

SOURCE=ROOT/'output/full_clip_ranking_review_20261102'
OUT=ROOT/'output/three_model_clip_gallery_20261102'
LABELS=['polite','bitter','true']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare():
 inv=read(SOURCE/'inventory.json');export=max(SOURCE.glob('full_clip_onset_peak_*.json'),key=lambda p:p.stat().st_mtime_ns);d=read(export);counts=validate(d,inv);assert d['complete']
 sources={r['id']:r for r in read(SOURCE/'sources.json')};models=read(SOURCE/'models.json');rows=[]
 for r in inv['records']:
  v=d['decisions'][r['id']]
  if v['state']!='accepted':continue
  assert r['person']!='REN' and all(r['person']+'/'+label in models for label in LABELS)
  paths=[Path(p) for p in sources[r['id']]['face_paths'] if v['onset_frame']<=frame_number(p)<=v['peak_frame']]
  points=[dict(source_frame=frame_number(p),source_seconds=r['source_frame_times'][frame_number(p)],seconds=r['source_frame_times'][frame_number(p)]-v['onset_seconds'],face_path=str(p)) for p in paths]
  assert len(points)>=2 and points[0]['source_frame']==v['onset_frame'] and points[-1]['source_frame']==v['peak_frame']
  dest=OUT/'clips'/r['person']/r['label']/r['filename']
  rows.append(dict(id=r['id'],person=r['person'],label=r['label'],filename=r['filename'],source_video=r['source_video'],source_sha256=r['source_sha256'],onset_frame=v['onset_frame'],peak_frame=v['peak_frame'],onset_seconds=v['onset_seconds'],peak_seconds=v['peak_seconds'],interval_seconds=v['peak_seconds']-v['onset_seconds'],frames=points,clip=str(dest),video_url=dest.relative_to(OUT).as_posix()))
 assert len(rows)==1215 and counts.get('discarded')==20
 OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==rows,'Existing input snapshot differs'
 else:save_json(OUT/'manifest.json',rows);shutil.copy2(export,OUT/'review_export.json');save_json(OUT/'models.json',models)
 save_json(OUT/'protocol.json',dict(clips=len(rows),discarded=20,source_export=str(export),source_export_sha256=digest(export),models='All three matching-person models on every selected tracked frame',score='Each model uses its own ten scale references: clamp(sum symmetric probabilities-0.5,0,9)',time='Original source PTS minus reviewed onset time; actual source frames from onset through peak inclusive',smooth='0.10-second median calculated inside the selected interval for all three models',stills='Up to ten unique tracked face frames uniformly spanning the interval',warning='Model-specific ordinal responses, not calibrated category probabilities'))
 return rows,models

def export_clip(r):
 dest=Path(r['clip']);done=OUT/'verification'/r['person']/r['label']/(r['id']+'.json')
 if done.exists():
  audit=read(done);assert dest.is_file() and audit['onset_frame']==r['onset_frame'] and audit['peak_frame']==r['peak_frame'] and dest.stat().st_size==audit['bytes'];return
 assert digest(r['source_video'])==r['source_sha256']
 dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.stem+'.partial.mp4')
 subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-i',r['source_video'],'-map','0:v:0','-vf',f"trim=start_frame={r['onset_frame']}:end_frame={r['peak_frame']+1},setpts=PTS-STARTPTS",'-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-fps_mode','vfr','-movflags','+faststart',str(temp)],check=True)
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames:format=duration','-of','json',str(temp)],text=True))
 n=int(info['streams'][0]['nb_read_frames']);assert n==r['peak_frame']-r['onset_frame']+1
 temp.replace(dest);save_json(done,dict(onset_frame=r['onset_frame'],peak_frame=r['peak_frame'],source_sha256=r['source_sha256'],frames=n,duration=float(info['format']['duration']),bytes=dest.stat().st_size))

def score(rows,models):
 import torch
 for person in sorted({r['person'] for r in rows}):
  specs={label:models[person+'/'+label] for label in LABELS};scorers={}
  for label,m in specs.items():
   assert digest(m['weights'])==m['model_sha256'] and all(digest(a['path'])==a['sha256'] for a in m['anchors'])
   scorers[label]=Scorer(m['config'],Path(m['weights']),[Path(a['path']) for a in m['anchors']],32)
  reference=scorers['polite'].model.extractor.state_dict()
  shared=all(set(s.model.extractor.state_dict())==set(reference) and all(torch.equal(reference[k],s.model.extractor.state_dict()[k]) for k in reference) for s in scorers.values())
  print(person,'identical feature extractors:',shared,flush=True)
  features=None;s=None
  subset=[r for r in rows if r['person']==person]
  for n,r in enumerate(subset,1):
   folder=OUT/'curves'/person/r['label']/r['id'];cache=folder/'scores.json';fingerprint=dict(source=r['source_sha256'],onset=r['onset_frame'],peak=r['peak_frame'],models={l:m['model_sha256'] for l,m in specs.items()},anchors={l:[a['sha256'] for a in m['anchors']] for l,m in specs.items()})
   if cache.exists():assert read(cache)['fingerprint']==fingerprint;continue
   paths=[Path(f['face_path']) for f in r['frames']];values={l:[] for l in LABELS}
   with torch.inference_mode():
    for start in range(0,len(paths),32):
     batch=paths[start:start+32];features=scorers['polite'].features(batch) if shared else None
     for label,s in scorers.items():
      z=s.score_batch(s.model,features if shared else s.features(batch),s.anchors);values[label].extend(map(float,z['score_0_9']))
   t=np.array([f['seconds'] for f in r['frames']]);smoothed={l:[float(np.median(np.array(values[l])[abs(t-v)<=.05])) for v in t] for l in LABELS}
   points=[dict(source_frame=f['source_frame'],source_seconds=f['source_seconds'],seconds=f['seconds'],**{l:values[l][i] for l in LABELS},**{l+'_smooth':smoothed[l][i] for l in LABELS}) for i,f in enumerate(r['frames'])]
   assert len(points)==len(paths) and all(0<=p[l]<=9 for p in points for l in LABELS)
   folder.mkdir(parents=True,exist_ok=True)
   with (folder/'curve.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
   save_json(cache,dict(fingerprint=fingerprint,shared_feature_extractor=shared,points=points))
   if n%25==0 or n==len(subset):print(person,n,'/',len(subset),'scored',flush=True)
  del reference,scorers,features,s;gc.collect();torch.cuda.empty_cache()

def publish(rows):
 cards=[]
 for r in rows:
  folder=OUT/'curves'/r['person']/r['label']/r['id'];data=read(folder/'scores.json');assert len(data['points'])==len(r['frames'])
  assert Path(r['clip']).is_file() and (OUT/'verification'/r['person']/r['label']/(r['id']+'.json')).exists()
  stilldir=OUT/'stills'/r['person']/r['label']/r['id'];stilldir.mkdir(parents=True,exist_ok=True);stills=[]
  for j,i in enumerate(uniform_indices(len(r['frames']),10)):
   frame=r['frames'][i];dest=stilldir/f'{j+1:02d}.png';shutil.copy2(frame['face_path'],dest);stills.append(dict(seconds=frame['seconds'],source_frame=frame['source_frame'],url=dest.relative_to(OUT).as_posix()))
  payload=dict(points=data['points'],stills=stills);js=folder/'curve.js';js.write_text('window.showClipData('+json.dumps(r['id'])+','+json.dumps(payload)+');',encoding='utf8')
  cards.append({k:v for k,v in r.items() if k!='frames'}|dict(curve_js=js.relative_to(OUT).as_posix(),curve_csv=(folder/'curve.csv').relative_to(OUT).as_posix(),tracked_frames=len(r['frames'])))
 template=Path(__file__).with_name('three_model_clip_gallery.html').read_text(encoding='utf8');(OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(cards,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8')
 save_json(OUT/'gallery_manifest.json',cards);save_json(OUT/'complete.json',dict(clips=len(rows),curves=len(rows)*3,scored_tracked_frames=sum(len(r['frames']) for r in rows),by_person=dict(Counter(r['person'] for r in rows))))
 print('Gallery complete:',OUT/'index.html',flush=True)

def main():
 rows,models=prepare()
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(export_clip,r) for r in rows]
  score(rows,models)
  for i,f in enumerate(futures,1):
   f.result()
   if i%100==0:print('Clips verified',i,'/',len(rows),flush=True)
 publish(rows)

if __name__=='__main__':main()
