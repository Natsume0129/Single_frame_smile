"""Local export watcher: validate reviewed groups, freeze datasets, train sequentially.

No deletion of old models or source data. Only matching explicit review exports are used.
"""
import argparse,csv,hashlib,json,os,random,shutil,subprocess,sys,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from build_monotonic_ranking_review import ROOT,OUT,read,sha
from validate_monotonic_review import validate

DEST=ROOT/'output/monotonic_ranking_models_20261029'
STATE=OUT/'auto_training_status.json'
def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(p)
def status(data):
 data['updated_at']=datetime.now().isoformat();save(STATE,data)
 p=OUT/'training_status.js';temp=p.with_suffix('.tmp');temp.write_text('window.onTrainingStatus && window.onTrainingStatus('+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+');',encoding='utf8');temp.replace(p)
def usable(r):return [p for p in r['pairs'] if p['later_frame']-p['earlier_frame']>=2]
def ready(rows):
 counts={s:sum(len(usable(r)) for r in rows if r['split']==s) for s in ['train','val','test']}
 groups={s:len({r['group_id'] for r in rows if r['split']==s and usable(r)}) for s in counts}
 return dict(pairs=counts,groups=groups,ready=sum(counts.values())>=280 and groups['train']>=2 and groups['val']>=1 and groups['test']>=1)

def accepted_partial(d,inv):
 # Validate only explicitly reviewed rows using the same frozen inventory and recomputation.
 assert d['dataset_id']==inv['dataset_id'] and d['schema_version']==3 and d['task']==inv['task']
 ids=set(d['decisions']);allids={r['id'] for r in inv['records']};assert ids<=allids
 files={r['id']:r for r in d['files']};assert len(files)==len(d['files'])==len(allids) and set(files)==allids
 subset={**inv,'records':[r for r in inv['records'] if r['id'] in ids]}
 payload={**d,'complete':True,'total':len(ids),'files':[f for f in d['files'] if f['id'] in ids]}
 return validate(payload,subset)

def build_dataset(rows,d,folder,cfg):
 assert not folder.exists();images=folder/'images';images.mkdir(parents=True)
 copied={};decoded_splits=defaultdict(set)
 from PIL import Image
 # Copy independently, never hard-link: deleting old experimental data cannot remove new images.
 for r in rows:
  assert sha(r['source_video'])==r['source_sha256'],'Source video changed'
  for f in r['sample_frames']:
   assert sha(f['path'])==f['sha256'],'Selected image changed'
   with Image.open(f['path']) as im:
    im=im.convert('RGB');decoded=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   decoded_splits[decoded].add(r['split'])
   assert len(decoded_splits[decoded])==1,'Decoded image leakage between train/val/test'
   name=r['id']+'_'+str(f['source_frame'])+'.png';shutil.copy2(f['path'],images/name);copied[(r['id'],f['source_frame'])]=name
 rng=random.Random(cfg['seed']);all_pairs=[];pair_records=[];unique_pairs=set()
 for split in ['train','val','test']:
  lines=[]
  for r in rows:
   if r['split']!=split:continue
   for p in usable(r):
    a=copied[(r['id'],p['earlier_frame'])];b=copied[(r['id'],p['later_frame'])];key=(r['source_sha256'],p['earlier_frame'],p['later_frame'])
    if key in unique_pairs:continue
    unique_pairs.add(key);swap=rng.random()<.5;left,right=(b,a) if swap else (a,b);label=1 if swap else -1
    lines.append(f'{left},{right},{label}')
    pair_records.append(dict(clip_id=r['id'],group_id=r['group_id'],split=split,image1=left,image2=right,label=label,label_source='human_accepted_clip_temporal_order'))
  (folder/(split+'.dat')).write_text('\n'.join(lines)+'\n',encoding='utf8');all_pairs+=lines
 assert len(pair_records)>=280,'Less than 280 unique total pairs'
 (folder/'all.dat').write_text('\n'.join(all_pairs)+'\n',encoding='utf8')
 with (folder/'pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pair_records[0]));w.writeheader();w.writerows(pair_records)
 save(folder/'accepted_clips.json',rows);save(folder/'review_export.json',d);save(folder/'config.json',cfg)
 return dict(pair_counts={s:sum(p['split']==s for p in pair_records) for s in ['train','val','test']},independent_clip_counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']})

def train(rows,d,person,label,state):
 assert person!='REN','REN training disabled by user'
 config_name=person+('_true' if label=='true' and person in ['JIANG','XU','YAMAMOTO'] else '')
 cfg=read(ROOT/f'experiments/person_smile/config.{config_name}.json')
 fingerprint=hashlib.sha256(json.dumps(dict(rows=rows,config=cfg,trainer=sha(ROOT/'annotation/train_smile_rank_with_holdout.py')),sort_keys=True).encode()).hexdigest()
 folder=DEST/person/label/fingerprint[:16];key=person+'/'+label
 if (folder/'complete.json').exists():state['groups'][key].update(state='complete',output=str(folder),metrics=read(folder/'model/metrics.json'));status(state);return
 if folder.exists():state['groups'][key].update(state='needs_inspection',output=str(folder),message='Previous incomplete attempt preserved; not overwritten.');status(state);return
 state['groups'][key].update(state='preparing',output=str(folder));status(state)
 try:
  info=build_dataset(rows,d,folder,cfg);model=folder/'model';model.mkdir()
  cmd=[sys.executable,'-u',str(ROOT/'annotation/train_smile_rank_with_holdout.py'),'--smilecomp_dir',cfg['smilecomp_dir'],'--weights',cfg['initial_weights'],
   '--dataset_file',str(folder/'all.dat'),'--dataset_dir',str(folder/'images'),'--train_file',str(folder/'train.dat'),'--val_file',str(folder/'val.dat'),'--holdout_file',str(folder/'test.dat'),'--use_existing_splits',
   '--dst_weights',str(model/'model.pth'),'--dst_history',str(model/'history.png'),'--dst_metrics',str(model/'metrics.json'),'--epochs',str(cfg['epochs']),'--batch_size',str(cfg['batch_size']),'--learning_rate',str(cfg['learning_rate']),'--seed',str(cfg['seed'])]
  save(folder/'training_request.json',dict(command=cmd,fingerprint=fingerprint,counts=info))
  state['groups'][key].update(state='training',counts=info,log=str(folder/'train.log'));status(state)
  with (folder/'train.log').open('w',encoding='utf8') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  metrics=read(model/'metrics.json');assert (model/'model.pth').is_file() and (model/'history.png').is_file()
  save(folder/'complete.json',dict(fingerprint=fingerprint,model_sha256=sha(model/'model.pth'),metrics=metrics,completed_at=datetime.now().isoformat()))
  save(DEST/person/label/'latest.json',dict(directory=str(folder),model=str(model/'model.pth'),metrics=metrics))
  state['groups'][key].update(state='complete',metrics=metrics);status(state)
 except Exception as e:state['groups'][key].update(state='failed',error=str(e));status(state)

def process(export,state,check_only=False):
 inv=read(OUT/'inventory.json');d=read(export);rows=accepted_partial(d,inv)
 groups=defaultdict(list)
 for r in rows:
  if r['person']!='REN':groups[(r['person'],r['label'])].append(r)
 report={p+'/'+l:ready(rs) for (p,l),rs in groups.items()}
 if check_only:return report
 receipt=OUT/'training_exports'/Path(export).name;save(receipt,d)
 state.update(last_export=str(receipt),groups={k:{**v,'state':'queued' if v['ready'] else 'waiting_for_data'} for k,v in report.items()});status(state)
 for (person,label),rs in sorted(groups.items()):
  if report[person+'/'+label]['ready']:train(rs,d,person,label,state)
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--check',type=Path);a=ap.parse_args()
 if a.check:print(json.dumps(process(a.check,{},True),indent=2));return
 assert a.watch
 import msvcrt
 lock=(OUT/'auto_train.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Worker already running')
 state=dict(worker_pid=os.getpid(),worker='running',groups={},watch_directories=[str(OUT),str(Path.home()/'Downloads')],note='Training starts only after a matching exported review JSON is found. Existing models/data are preserved.')
 status(state);seen=set()
 while True:
  try:
   paths=[p for folder in [OUT,Path.home()/'Downloads'] if folder.exists() for p in folder.glob('monotonic_ranking_decisions_*.json')]
   if paths:
    p=max(paths,key=lambda p:p.stat().st_mtime_ns);signature=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
    if signature not in seen:
     try:read(p)
     except (ValueError,OSError):time.sleep(3);continue
     seen.add(signature)
     try:process(p,state)
     except Exception as e:state['last_export_error']=dict(file=str(p),error=str(e));status(state)
   time.sleep(3)
  except KeyboardInterrupt:break
  except Exception as e:state['worker_error']=str(e);status(state);time.sleep(5)

if __name__=='__main__':main()
