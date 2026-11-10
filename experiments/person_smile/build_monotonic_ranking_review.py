"""Build one review/export for a new temporal-ranking dataset; never alter old datasets."""
import csv,hashlib,itertools,json,math,re,subprocess
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/monotonic_ranking_review_20261029'
FINAL=ROOT/'output/all_people_relaxed_start_20261026'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(p):return int(Path(p).stem.rsplit('_',1)[-1])

def sample_frames(frames):
 """~0.2 s spacing; short clips fall back to five frames >=2 source ordinals apart."""
 if len(frames)<5:return [],'Fewer than five tracked/scored frames'
 chosen=[frames[0]]
 for f in frames[1:]:
  if f['source_frame']-chosen[-1]['source_frame']>=2 and f['seconds']-chosen[-1]['seconds']>=.2-1e-5:chosen.append(f)
 if len(chosen)>=5:
  last=frames[-1]
  if last['source_frame']!=chosen[-1]['source_frame'] and last['source_frame']-chosen[-2]['source_frame']>=2:
   chosen[-1]=last
  return chosen,'Approximately 0.2 s between frames; final sample includes interval end'
 # Build earliest possible suffix for each of the five positions, then sample evenly without blocking later positions.
 latest=[0]*5;latest[4]=len(frames)-1
 for j in range(3,-1,-1):
  options=[i for i in range(latest[j+1]) if frames[latest[j+1]]['source_frame']-frames[i]['source_frame']>=2]
  if not options:return [],'Cannot select five frames with source-frame gaps of at least two'
  latest[j]=options[-1]
 selected=[]
 for j in range(5):
  options=[i for i in range(latest[j]+1) if not selected or frames[i]['source_frame']-selected[-1]['source_frame']>=2]
  target=frames[0]['seconds']+(frames[-1]['seconds']-frames[0]['seconds'])*j/4
  i=min(options,key=lambda i:abs(frames[i]['seconds']-target));selected.append(frames[i])
 return selected,'Short interval: five frames, minimum two source frames apart'

def probe(r):
 p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,r_frame_rate','-of','json',r['source_video']],text=True))['streams'][0]
 def fps(s):
  a,b=map(int,s.split('/'));return a/b if b else None
 r['average_fps']=fps(p['avg_frame_rate']);r['nominal_fps']=fps(p['r_frame_rate']);r['source_sha256']=sha(r['source_video'])
 return r

def main():
 OUT.mkdir(exist_ok=True)
 assert not (OUT/'inventory.json').exists(),'Inventory frozen: use a new version/directory to rebuild.'
 final=read(FINAL/'final_manifest.json');locs=read(FINAL/'DATASET_LOCATIONS.json')
 seqs={};seen=set()
 for loc in locs:
  if loc['sequences_csv'] in seen:continue
  seen.add(loc['sequences_csv'])
  for s in table(loc['sequences_csv']):seqs[(loc['person'],s['sequence_id'])]=(s,loc['experiment_directory'])
 known_groups=defaultdict(set)
 for (person,sid),(s,base) in seqs.items():known_groups[(person,s['source_group'])].add(s['split'])
 assert all(len(v)==1 for v in known_groups.values())
 legacy={r['final_clip']:r for r in table(ROOT/'output/neutral_to_peak_delivery_20261023/face_tracking_index.csv')}
 groups=defaultdict(list);splits=defaultdict(set);missing=[]
 for r in final:
  if r['final_clip'] in legacy:
   m=legacy[r['final_clip']];sid=m['sequence_id'];face=Path(m['face_images_directory'])
  else:sid=Path(r['curve_csv']).parent.name;face=None
  split_origin='existing_sequence_split'
  if (r['person'],sid) in seqs:s,base=seqs[(r['person'],sid)]
  else:
   # Some final accepted clips were tracking-only and absent from the old training manifest.
   group=re.sub(r'_P\d+$','',Path(r['source_video']).stem);existing=known_groups.get((r['person'],group))
   bucket=int(hashlib.sha256((r['person']+'/'+group).encode()).hexdigest()[:8],16)%10
   split=next(iter(existing)) if existing else ('test' if bucket<2 else 'val' if bucket==2 else 'train')
   s=dict(source_group=group,split=split,sequence_dir=str(face));split_origin='existing_sibling_group' if existing else 'new_group_fixed_hash'
  face=face or Path(s['sequence_dir'])
  gid=r['person']+'/'+s['source_group'];splits[gid].add(s['split'])
  curve=[dict(source_frame=int(x['source_frame']),seconds=float(x['seconds']),raw_score=float(x['raw_score']),smoothed_score=float(x['smoothed_score'])) for x in table(r['curve_csv'])]
  interval=[x for x in curve if r['start_frame']<=x['source_frame']<=r['peak_frame']]
  faces={number(p):p for p in face.glob('*.png')}
  frames=[dict(**x,path=str(faces[x['source_frame']]),url=faces[x['source_frame']].as_uri()) for x in interval if x['source_frame'] in faces]
  samples,note=sample_frames(frames)
  dy=np.diff([x['smoothed_score'] for x in interval]);drop=float(-np.minimum(dy,0).sum());gain=interval[-1]['smoothed_score']-interval[0]['smoothed_score']
  pairs=[dict(first=i+1,second=i+gap+1,earlier_frame=samples[i]['source_frame'],later_frame=samples[i+gap]['source_frame'],stronger='later',old_score_increases=samples[i+gap]['smoothed_score']>samples[i]['smoothed_score']) for gap in range(1,len(samples)) for i in range(len(samples)-gap)]
  item=dict(id=hashlib.sha256(r['source_video'].encode()).hexdigest()[:24],person=r['person'],label=r['label'],filename=Path(r['source_video']).name,
   source_video=r['source_video'],clip=r['final_clip'],video_url=Path(r['final_clip']).as_uri(),original_url=Path(r['source_video']).as_uri(),
   start_seconds=r['start_seconds'],peak_seconds=r['peak_seconds'],onset_seconds=r['onset_seconds'],start_frame=r['start_frame'],peak_frame=r['peak_frame'],
   sequence_id=sid,source_group=s['source_group'],group_id=gid,split=s['split'],split_origin=split_origin,face_directory=str(face),curve_csv=r['curve_csv'],curve=curve,
   sample_frames=samples,pairs=pairs,sampling_note=note,eligible=len(samples)>=5 and gain>0,net_gain=gain,total_downward_change=drop,
   strict_nondecreasing=bool(np.all(dy>=0)),pair_score_warnings=sum(not p['old_score_increases'] for p in pairs),
   tracked_frames_in_interval=len(frames),curve_sha256=sha(r['curve_csv']))
  groups[(r['person'],r['label'])].append(item)
 assert all(len(v)==1 for v in splits.values()),'Existing source groups cross splits; resolve before preparing.'
 selected=[]
 for key,items in sorted(groups.items()):
  # Freeze old source-group splits; 28/4/8 target quotas, filling shortages without reassigning any split.
  def quality(r):return (not r['eligible'],r['total_downward_change']/max(r['net_gain'],.01),-r['net_gain'],r['filename'])
  ordered=sorted(items,key=quality)
  if len(items)<=40:pick=ordered
  else:
   pick=[]
   for split,n in [('train',28),('val',4),('test',8)]:pick.extend([r for r in ordered if r['split']==split][:n])
   used={r['id'] for r in pick};pick.extend([r for r in ordered if r['id'] not in used][:40-len(pick)])
  selected.extend(pick)
 assert len(selected)==520 and len({r['id'] for r in selected})==520
 with ThreadPoolExecutor(max_workers=6) as pool:selected=list(pool.map(probe,selected))
 for r in selected:
  f=r['sample_frames'];assert all(b['source_frame']-a['source_frame']>=2 for a,b in zip(f,f[1:]))
  assert len(r['pairs'])==len(f)*(len(f)-1)//2
  for x in f:x['sha256']=sha(x['path'])
 dataset_id=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
 inventory=dict(schema_version=1,task='monotonic_clip_ranking_review',dataset_id=dataset_id,created_date='2026-10-29',
  policy=dict(max_per_person_category=40,monotonicity='Human acceptance of approximately increasing old-model curve; small fluctuations permitted',
   sampling='Approx. 0.2 s; short clips use five frames >=2 source frames apart; long clips use all sampled frames',
   pairing='Within accepted clip only; all i<j, ordered by index gap; later assumed stronger',
   split='Preserve prior source-group train/val/test; never split frames or pairs from a group',
   labels='Temporal weak labels after human clip acceptance; not independent pairwise human ground truth',
   independence_limit='Old model curves guide selection; test performance measures this selected distribution, not independent human ranking validity'),records=selected)
 (OUT/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
 payload=json.dumps(inventory,ensure_ascii=False).replace('<','\\u003c')
 template=Path(__file__).with_name('monotonic_ranking_review.html').read_text(encoding='utf8')
 (OUT/'index.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf8')
 summary=dict(total=len(selected),groups={p+'/'+l:dict(total=sum(r['person']==p and r['label']==l for r in selected),splits=dict(Counter(r['split'] for r in selected if r['person']==p and r['label']==l))) for p,l in groups},ineligible=sum(not r['eligible'] for r in selected),prospective_pairs=sum(len(r['pairs']) for r in selected))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 (OUT/'README.txt').write_text('Open index.html. Review all groups; export ONE monotonic_ranking_decisions_*.json into this directory.\nThe browser may download to Downloads; choose this directory in Save As or move the JSON here.\nNo model is trained and no old annotations are changed. inventory.json freezes source-group splits and frame/pair proposals.\nAccept implies later sampled frames are stronger; old scores may fluctuate. Test labels derived this way are weak temporal labels, not independently annotated pairwise ground truth.\nAfter review, run validate_monotonic_review.py EXPORT.json to validate and materialize per-person/category/split pair manifests (not train).\n',encoding='utf8')
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
