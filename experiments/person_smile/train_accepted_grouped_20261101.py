"""Repartition accepted non-REN clips by source/duplicate groups and train frozen snapshots."""
import argparse,hashlib,json,os,random
from pathlib import Path
from collections import defaultdict,Counter
from PIL import Image
import monotonic_auto_train as auto
from build_monotonic_ranking_review import ROOT,OUT,read,sha

RUN=ROOT/'output/monotonic_accepted_training_20261101'
SEED=20261101

def prepare():
 assert not (RUN/'accepted_manifest.json').exists(),'Prepared snapshot already exists'
 inv=read(OUT/'inventory.json');export=max(OUT.glob('monotonic_ranking_decisions_*.json'),key=lambda p:p.stat().st_mtime_ns);d=read(export)
 rows=[r for r in auto.accepted_partial(d,inv) if r['person']!='REN'];assert len(rows)==323
 parent={r['id']:r['id'] for r in rows}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):parent[find(b)]=find(a)
 seen={};image_hashes={}
 for r in rows:
  assert sha(r['source_video'])==r['source_sha256']
  keys=[('group',r['person'],r['group_id']),('source',r['person'],r['source_sha256'])]
  for f in r['sample_frames']:
   assert sha(f['path'])==f['sha256']
   with Image.open(f['path']) as im:
    im=im.convert('RGB');h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   keys.append(('image',r['person'],h));image_hashes[(r['id'],f['source_frame'])]=h
  for key in keys:
   if key in seen:union(r['id'],seen[key])
   else:seen[key]=r['id']
 components=defaultdict(list)
 for r in rows:components[find(r['id'])].append(r)
 assignment={}
 for person in sorted({r['person'] for r in rows}):
  units=sorted(k for k,v in components.items() if v[0]['person']==person);rng=random.Random(SEED+sum(map(ord,person)));best=None
  # Choose a deterministic group-level split near 70/15/15 by pair counts; never optimize model accuracy.
  n=len(units);nv=max(2,round(n*.15));nt=max(2,round(n*.15))
  for attempt in range(500):
   ids=units[:];rng.shuffle(ids);a={g:('val' if i<nv else 'test' if i<nv+nt else 'train') for i,g in enumerate(ids)}
   cost=0;valid=True
   for label in ['polite','bitter','true']:
    sub=[r for r in rows if r['person']==person and r['label']==label];counts=Counter();gs=defaultdict(set)
    for r in sub:split=a[find(r['id'])];counts[split]+=len(auto.usable(r));gs[split].add(find(r['id']))
    if any(len(gs[s])<2 or counts[s]<1 for s in ['train','val','test']):valid=False;break
    total=sum(counts.values());cost+=sum((counts[s]/total-target)**2 for s,target in [('train',.7),('val',.15),('test',.15)])
   if valid and (best is None or cost<best[0]):best=(cost,a)
  assert best is not None,f'Cannot form independent splits for {person}'
  assignment.update(best[1])
 for r in rows:
  r['review_split']=r['split'];r['split']=assignment[find(r['id'])];r['split_group_id']=find(r['id']);r['split_policy']='accepted_source_group_70_15_15_seed20261101'
 for keyfn in [lambda r:r['source_video'],lambda r:r['group_id'],lambda r:r['split_group_id']]:
  check=defaultdict(set)
  for r in rows:check[keyfn(r)].add(r['split'])
  assert all(len(v)==1 for v in check.values())
 imsplit=defaultdict(set)
 for r in rows:
  for f in r['sample_frames']:imsplit[(r['person'],image_hashes[(r['id'],f['source_frame'])])].add(r['split'])
 assert all(len(s)==1 for s in imsplit.values())
 summary={}
 for person,label in sorted({(r['person'],r['label']) for r in rows}):
  sub=[r for r in rows if (r['person'],r['label'])==(person,label)]
  summary[person+'/'+label]=dict(clips=len(sub),pairs=sum(len(auto.usable(r)) for r in sub),splits={s:dict(clips=sum(r['split']==s for r in sub),pairs=sum(len(auto.usable(r)) for r in sub if r['split']==s),groups=len({r['split_group_id'] for r in sub if r['split']==s})) for s in ['train','val','test']})
 RUN.mkdir(exist_ok=True);auto.save(RUN/'review_export.json',d);auto.save(RUN/'accepted_manifest.json',rows);auto.save(RUN/'split_summary.json',summary)
 auto.save(RUN/'split_audit.json',dict(seed=SEED,target_ratio=[.7,.15,.15],export=str(export),export_sha256=sha(export),accepted_clips=len(rows),excluded_person='REN',clip_group_and_decoded_image_leakage=False,split_selection_uses_model_results=False))
 print(json.dumps(summary,indent=2),flush=True)

def run():
 rows=read(RUN/'accepted_manifest.json');d=read(RUN/'review_export.json');summary=read(RUN/'split_summary.json')
 assert all(r['person']!='REN' for r in rows)
 auto.DEST=RUN/'models'
 state=dict(worker_pid=os.getpid(),worker='running',run=str(RUN),note='Accepted clips only; REN excluded; new grouped 70/15/15 split.',groups={k:{**v,'state':'queued'} for k,v in summary.items()})
 for key in summary:
  person,label=key.split('/');sub=[r for r in rows if r['person']==person and r['label']==label];auto.train(sub,d,person,label,state)
  auto.save(RUN/'training_status.json',state)
 state['worker']='finished';auto.status(state);auto.save(RUN/'training_status.json',state)
 print('FINISHED', {k:v['state'] for k,v in state['groups'].items()},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.train:run()
