"""Read-only check of exported full-clip onset/peak annotations."""
import argparse,json
from collections import Counter
from score_full_original_clips import OUT,read

def validate(d,inv):
 assert d['schema_version']==1 and d['task']==inv['task'] and d['dataset_id']==inv['dataset_id']
 rows={r['id']:r for r in inv['records']};files={r['id']:r for r in d['files']};decisions=d['decisions']
 assert len(files)==len(d['files'])==d['total']==len(rows) and set(files)==set(rows) and set(decisions)<=set(rows)
 for ident,r in rows.items():
  f=files[ident];v=decisions.get(ident,{})
  assert all(f[k]==r[k] for k in ['person','label','source_video','source_sha256','model_path','model_sha256','curve_file'])
  state=v.get('state','pending');assert state in ['pending','accepted','rejected','discarded'] and f['state']==state
  for name in ['onset','peak']:
   frame=v.get(name+'_frame');seconds=v.get(name+'_seconds')
   assert f[name+'_frame']==frame and f[name+'_seconds']==seconds
   if frame is not None:
    assert isinstance(frame,int) and frame in r['tracked_source_frames'] and isinstance(seconds,(int,float))
    assert abs(r['source_frame_times'][frame]-seconds)<1e-6
  if state=='accepted':assert isinstance(v.get('onset_frame'),int) and isinstance(v.get('peak_frame'),int) and v['onset_frame']<v['peak_frame']
 counts=Counter(decisions.get(i,{}).get('state','pending') for i in rows)
 assert d['complete']==(counts['pending']==0)
 return dict(counts)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('export');a=p.parse_args();print(json.dumps(validate(read(a.export),read(OUT/'inventory.json')),indent=2))
