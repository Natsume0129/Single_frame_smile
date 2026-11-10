"""Score every saved tracked frame of raw labelled clips with matching new models."""
import argparse,csv,gc,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from pipeline import ROOT,read_csv,frame_number,digest,save_json
from onset_peak import Scorer,timestamps

OUT=ROOT/'output/full_clip_ranking_review_20261102'
RUN=ROOT/'output/monotonic_accepted_training_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare():
 OUT.mkdir(exist_ok=True);assert not (OUT/'sources.json').exists(),'Frozen inventory exists; use --score to resume'
 configs={name:read(ROOT/f'experiments/person_smile/config.{name}.json') for name in ['JIANG','JIANG_true','XU','XU_true','YAMAMOTO','YAMAMOTO_true','TAKEDA','JIE']}
 trackmap=defaultdict(list);choices={}
 for name,cfg in configs.items():
  person=name.split('_')[0];base=Path(cfg['output_root'])/person
  for s in read_csv(base/'sequences.csv'):choices[(person,s['sequence_id'])]=s['track_id']
  for done in (base/'tracking').glob('*/complete.json'):
   data=read(done);trackmap[(person,data['source_sha256'])].append(done.parent)
 records=[];models={}
 for name,cfg in configs.items():
  person=name.split('_')[0]
  for label,rawfolder in cfg['label_folders'].items():
   key=person+'/'+label;latest=read(RUN/'models'/person/label/'latest.json');weights=Path(latest['model']);assert (Path(latest['directory'])/'complete.json').is_file()
   scale=RUN/'scale10'/person/label;anchors=[scale/f'level_{i:02d}.png' for i in range(10)]
   models[key]=dict(person=person,label=label,config=cfg,weights=str(weights),model_sha256=digest(weights),anchors=[dict(path=str(p),sha256=digest(p)) for p in anchors])
   for source in sorted((ROOT/'new_dataset'/person/rawfolder).glob('*.mp4')):
    h=digest(source);candidates=trackmap.get((person,h),[]);tracks=[]
    for base in candidates:
     sid=base.name;found=[p/'0' for p in (base/'DetectedFaces'/sid).glob('*') if p.is_dir() and list((p/'0').glob('*.png'))]
     chosen=choices.get((person,sid))
     if chosen is not None:found=[p for p in found if p.parent.name==str(chosen)]
     if len(found)==1:tracks.append((sid,found[0]))
    # Several copies may reference the same source. Prefer the richest valid saved track, never merge identities.
    tracks.sort(key=lambda x:len(list(x[1].glob('*.png'))),reverse=True)
    item=dict(id=hashlib.sha256((key+'/'+source.name).encode()).hexdigest()[:24],person=person,label=label,source_video=str(source),source_sha256=h,filename=source.name,model_key=key,video_url=source.as_uri(),status='pending')
    if tracks:
     sid,folder=tracks[0];paths=sorted(folder.glob('*.png'),key=frame_number);frames=[frame_number(p) for p in paths];assert len(frames)==len(set(frames))
     item.update(sequence_id=sid,face_directory=str(folder),face_paths=[str(p) for p in paths],tracking_frames=len(paths))
    else:item.update(status='no_usable_tracking',tracking_frames=0,reason='No unique usable saved face track; no scores will be fabricated')
    records.append(item)
 assert len({r['source_video'] for r in records})==len(records)
 save_json(OUT/'models.json',models);save_json(OUT/'sources.json',records)
 save_json(OUT/'protocol.json',dict(scope='All raw labelled videos in five participant folders, including nonaccepted ranking clips and temporal exclusions; REN/MATSUDA excluded; unclassified discards have no assigned model',models='matching person/category latest accepted-only model',anchors='matching 10-level scale built from training images',score='clamp(sum of 10 symmetric P(frame stronger than reference) - 0.5, 0, 9)',smooth='0.10-second centered median on observed scores, display only',frames='Every saved frame in the selected face track, no temporal subsampling and no old boundary cropping',missing='Untracked source frames remain missing; annotations snap to observed tracked frames'))
 print('SOURCES',len(records),dict(Counter((r['person'],r['label']) for r in records)),flush=True)
 print('NO TRACKING',[(r['person'],r['label'],r['filename']) for r in records if r['status']!='pending'],flush=True)

def score():
 import torch
 records=read(OUT/'sources.json');models=read(OUT/'models.json');results=[]
 for key,m in models.items():
  assert digest(m['weights'])==m['model_sha256'];assert all(digest(a['path'])==a['sha256'] for a in m['anchors'])
  scorer=None;subset=[r for r in records if r['model_key']==key]
  for n,r in enumerate(subset,1):
   folder=OUT/'curves'/r['person']/r['label']/r['id'];folder.mkdir(parents=True,exist_ok=True);cache=folder/'result.json'
   if cache.exists():
    prior=read(cache);assert prior['model_sha256']==m['model_sha256'] and prior['source_sha256']==r['source_sha256'];results.append(prior);continue
   assert digest(r['source_video'])==r['source_sha256'];t=timestamps(r['source_video']);base={k:v for k,v in r.items() if k!='face_paths'}
   base.update(model_sha256=m['model_sha256'],model_path=m['weights'],source_frame_times=t.tolist(),duration=float(t[-1]),source_frame_count=len(t),curve_file=str(folder/'curve.csv'))
   if r['status']!='pending':base.update(curve=[],points=0,coverage=0);save_json(cache,base);results.append(base);continue
   paths=[Path(p) for p in r['face_paths']];indices=np.array([frame_number(p) for p in paths]);assert indices[-1]<len(t)
   if scorer is None:scorer=Scorer(m['config'],Path(m['weights']),[Path(a['path']) for a in m['anchors']],32)
   values=[];probs=[];cons=[]
   with torch.inference_mode():
    for start in range(0,len(paths),32):
     z=scorer.score_batch(scorer.model,scorer.features(paths[start:start+32]),scorer.anchors)
     values.extend(map(float,z['score_0_9']));probs.extend(map(float,z['mean_probability']));cons.extend(map(float,z['mean_anchor_consistency']))
   times=t[indices];raw=np.array(values);smooth=np.array([np.median(raw[np.abs(times-time)<=.05]) for time in times])
   curve=[dict(source_frame=int(i),seconds=float(time),raw_score=float(v),smoothed_score=float(y),mean_probability=float(p),swap_consistency=float(c),face_url=path.as_uri()) for i,time,v,y,p,c,path in zip(indices,times,values,smooth,probs,cons,paths)]
   with (folder/'curve.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
   base.update(status='scored',curve=curve,points=len(curve),coverage=len(curve)/len(t));assert len(curve)==r['tracking_frames']
   save_json(cache,base);results.append(base)
   print(key,n,'/',len(subset),r['filename'],len(curve),'frames',flush=True)
  del scorer;gc.collect();torch.cuda.empty_cache()
  save_json(OUT/'progress.json',dict(finished=len(results),total=len(records),last_group=key))
 save_json(OUT/'results.json',results)
 print('DONE',len(results),'clips',sum(r['points'] for r in results),'frames',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--score',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.score:score()
