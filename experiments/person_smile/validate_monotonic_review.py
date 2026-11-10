"""Validate the single review export and materialize split-specific pair manifests; no training."""
import argparse,hashlib,json,random,csv,subprocess
from collections import Counter,defaultdict
from pathlib import Path
from build_monotonic_ranking_review import OUT,read,sha

def validate(d,inv):
 assert d.get('schema_version')==inv['schema_version'] and d.get('task')==inv['task'] and d.get('dataset_id')==inv['dataset_id'],'Wrong dataset/version'
 records={r['id']:r for r in inv['records']};decisions=d.get('decisions',{});files={r['id']:r for r in d['files']}
 assert d.get('complete') is True and d['total']==len(records),'Review is incomplete'
 assert len(files)==len(d['files'])==len(records) and set(files)==set(records)==set(decisions),'Missing/duplicate/unknown IDs'
 if inv['schema_version'] in (2,3):
  engine=Path(__file__).with_name('manual_ranking_sampling.js' if inv['schema_version']==3 else 'active_rise_sampling.js').read_text(encoding='utf8')
  code=engine+"\nlet s='';process.stdin.on('data',b=>s+=b);process.stdin.on('end',()=>{const d=JSON.parse(s);process.stdout.write(JSON.stringify(d.records.map(r=>activeRise(r,d.decisions[r.id].bounds||null,d.decisions[r.id].extra_frames||[],d.decisions[r.id].removed_frames||[]))));});"
  rows=json.loads(subprocess.check_output(['node','-e',code],input=json.dumps(dict(records=list(records.values()),decisions=decisions)).encode('utf8')))
  records={r['id']:r for r in rows}
 group_splits=defaultdict(set);frame_splits=defaultdict(set)
 for ident,r in records.items():
  choice=decisions[ident]['state'];f=files[ident]
  assert choice in ('accept','reject') and f['decision']==choice
  assert all(f[k]==r[k] for k in ['person','label','source_video','source_sha256','group_id','split','start_frame','peak_frame','sample_frames','pairs']),'Export data modified'
  if inv['schema_version']<3:
   assert choice!='accept' or r['eligible'],'Accepted clip cannot supply five valid frames'
  group_splits[r['group_id']].add(r['split'])
  if choice=='accept':
   for x in r['sample_frames']:frame_splits[(r['person'],x['sha256'])].add(r['split'])
 assert all(len(s)==1 for s in group_splits.values()),'Source-group leakage'
 assert all(len(s)==1 for s in frame_splits.values()),'Identical sampled image files cross splits'
 return [r for r in records.values() if decisions[r['id']]['state']=='accept']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('export',type=Path);ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
 inv=read(OUT/'inventory.json');d=read(a.export);accepted=validate(d,inv);report={}
 for p,l in sorted({(r['person'],r['label']) for r in inv['records']}):
  selected=[r for r in accepted if r['person']==p and r['label']==l]
  counts={s:len({r['group_id'] for r in selected if r['split']==s}) for s in ['train','val','test']}
  sampling_warnings=[r['id'] for r in selected if not r['eligible']]
  report[p+'/'+l]=dict(accepted_clips=len(selected),independent_groups=counts,pairs=sum(len(r['pairs']) for r in selected),sampling_review_required=sampling_warnings,ready_for_training=all(n>=2 for n in counts.values()) and not sampling_warnings)
 print(json.dumps(report,indent=2))
 if a.check_only:return
 dest=OUT/'approved_pairs';assert not dest.exists(),'Already materialized; inspect or version the dataset'
 # Validate referenced sources before writing any output.
 for r in accepted:
  assert sha(r['source_video'])==r['source_sha256'],'Source changed'
  for x in r['sample_frames']:assert sha(x['path'])==x['sha256'],'Face frame changed'
 dest.mkdir();(dest/'review_export.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
 rng=random.Random(20261029)
 for person,label in sorted({(r['person'],r['label']) for r in inv['records']}):
  folder=dest/person/label;folder.mkdir(parents=True)
  for split in ['train','val','test']:
   pairs=[]
   for r in accepted:
    if (r['person'],r['label'],r['split'])!=(person,label,split):continue
    for p in r['pairs']:
     first=r['sample_frames'][p['first']-1];second=r['sample_frames'][p['second']-1];swap=rng.random()<.5
     left,right=(second,first) if swap else (first,second)
     pairs.append(dict(clip_id=r['id'],group_id=r['group_id'],split=split,image1=left['path'],image2=right['path'],label=1 if swap else -1,
      image1_source_frame=left['source_frame'],image2_source_frame=right['source_frame'],label_source='human_accepted_clip_temporal_order',
      earlier_number=p['first'],later_number=p['second'],old_scores_agree=p['old_score_increases'],suggested_clip_balancing_weight=1/len(r['pairs'])))
   with (folder/(split+'_pairs.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    fields=list(pairs[0]) if pairs else ['clip_id','group_id','split','image1','image2','label'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pairs)
 (dest/'readiness.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print('Materialized manifests only. Check readiness.json; do not train groups without adequate split coverage. Old datasets untouched.')

if __name__=='__main__':main()
