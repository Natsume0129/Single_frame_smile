"""Fit simple curve-boundary heuristics on a grouped development set; report held-out error."""
import hashlib,itertools,json,re
from pathlib import Path
from collections import Counter
import numpy as np
from score_full_original_clips import OUT,read
from validate_full_clip_review import validate

def detect(curve,params):
 if len(curve)<5:return dict(onset_frame=None,peak_frame=None,flags=['insufficient_frames'])
 t=np.array([p['seconds'] for p in curve]);raw=np.array([p['raw_score'] for p in curve]);sigma,of,pf=params
 # Gaussian time smoothing on observations; do not borrow samples across tracking gaps.
 groups=np.cumsum(np.r_[0,np.diff(t)>.15]);y=np.array([np.average(raw[(groups==groups[i])&(abs(t-time)<=3*sigma)],weights=np.exp(-.5*((t[(groups==groups[i])&(abs(t-time)<=3*sigma)]-time)/sigma)**2)) for i,time in enumerate(t)])
 m=int(np.argmax(y));flags=[]
 if m<2:return dict(onset_frame=None,peak_frame=int(curve[m]['source_frame']),flags=['peak_at_start_no_rise'])
 low=float(np.quantile(y[:m+1],.15));amp=float(y[m]-low)
 if amp<.5:flags.append('weak_amplitude')
 threshold=low+of*amp
 runs=[];a=None
 for i in range(m):
  if y[i]<=threshold:
   if a is None:a=i
   elif t[i]-t[i-1]>.15:runs.append((a,i-1));a=i
  elif a is not None:runs.append((a,i-1));a=None
 if a is not None:runs.append((a,m-1))
 stable=[(a,b) for a,b in runs if t[b]-t[a]>=.15]
 if stable:
  # Initial supported low interval; preserve temporary drops and subsequent rebound as one event.
  a,b=stable[0];onset=b
  for i in range(b+1,m):
   end=np.searchsorted(t,t[i]+.08)
   if end<=m and np.all(y[i:end+1]>threshold):onset=i;break
  if any(t[b]-t[a]>=.4 for a,b in stable[1:]):flags.append('multiple_low_returns')
 else:
  flags.append('no_stable_low_baseline');onset=int(np.argmin(y[:m]))
 peak_threshold=low+pf*amp
 below=np.where(y[onset:m+1]<peak_threshold)[0]
 peak=onset+int(below[-1])+1 if len(below) else m
 peak=min(m,max(onset+1,peak))
 if onset>=peak:return dict(onset_frame=None,peak_frame=int(curve[m]['source_frame']),flags=flags+['no_valid_rise'])
 if np.any(np.diff(t[onset:peak+1])>.15):flags.append('tracking_gap')
 if t[m]>=t[-1]-.1:flags.append('peak_near_video_end')
 if y[0]>threshold+.1*max(amp,0):flags.append('starts_above_baseline')
 return dict(onset_frame=int(curve[onset]['source_frame']),onset_seconds=float(t[onset]),peak_frame=int(curve[peak]['source_frame']),peak_seconds=float(t[peak]),flags=flags,amplitude=amp)

def main():
 inv=read(OUT/'inventory.json');export=max(OUT.glob('full_clip_onset_peak_*.json'),key=lambda p:p.stat().st_mtime_ns);d=read(export);print('Export',validate(d,inv))
 results=read(OUT/'results.json');curves={r['id']:r['curve'] for r in results};meta={r['id']:r for r in inv['records']}
 accepted=[(ident,v) for ident,v in d['decisions'].items() if v.get('state')=='accepted']
 grouped={}
 for ident,v in accepted:
  r=meta[ident];g=r['person']+'/'+re.sub(r'_P\d+$','',Path(r['filename']).stem);grouped[ident]=g
 groupnames=sorted(set(grouped.values()),key=lambda g:hashlib.sha256(('20261102/'+g).encode()).hexdigest());holdout=set(groupnames[:max(1,round(len(groupnames)*.3))])
 dev=[x for x in accepted if grouped[x[0]] not in holdout];test=[x for x in accepted if grouped[x[0]] in holdout];assert dev and test
 def evaluate(rows,params):
  errors=[];failed=0
  for ident,v in rows:
   p=detect(curves[ident],params)
   if p.get('onset_seconds') is None:failed+=1;continue
   errors.append(dict(id=ident,onset_error=p['onset_seconds']-v['onset_seconds'],peak_error=p['peak_seconds']-v['peak_seconds']))
  return dict(n=len(rows),valid=len(errors),failed=failed,onset_mae=float(np.mean([abs(e['onset_error']) for e in errors])) if errors else None,peak_mae=float(np.mean([abs(e['peak_error']) for e in errors])) if errors else None,within_020s_both=sum(abs(e['onset_error'])<=.2 and abs(e['peak_error'])<=.2 for e in errors),errors=errors)
 grid=list(itertools.product([.04,.08],[.05,.10,.20],[.90,.95,.98,1.0]));scores=[]
 for params in grid:
  v=evaluate(dev,params);loss=(v['onset_mae'] or 0)+(v['peak_mae'] or 0)+2*v['failed']/len(dev);scores.append((loss,params))
 _,chosen=min(scores);dev_result=evaluate(dev,chosen);test_result=evaluate(test,chosen)
 proposals={r['id']:detect(curves[r['id']],chosen) for r in inv['records']}
 report=dict(export=str(export),export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),calibration_scope=dict(Counter(meta[i]['person']+'/'+meta[i]['label'] for i,v in accepted)),parameters=dict(gaussian_sigma_seconds=chosen[0],onset_amplitude_fraction=chosen[1],peak_amplitude_fraction=chosen[2]),development=dev_result,held_out=test_result,held_out_source_groups=sorted(holdout),development_ids=[i for i,v in dev],held_out_ids=[i for i,v in test],candidate_count=len(grid),selection='Parameters chosen using development errors only; held-out groups not used for selection',limitations='Calibration labels are not a representative sample of other people/categories. No automatic discard and no automatically accepted annotations.')
 (OUT/'auto_boundary_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'auto_boundary_proposals.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2),encoding='utf8')
 seed={i:v for i,v in d['decisions'].items() if v.get('state') in ['accepted','discarded','rejected']}
 (OUT/'human_seed_decisions.json').write_text(json.dumps(seed,ensure_ascii=False,indent=2),encoding='utf8')
 print('Parameters',chosen,'dev', {k:v for k,v in dev_result.items() if k!='errors'},'held_out',{k:v for k,v in test_result.items() if k!='errors'},flush=True)
 print('Proposals',len(proposals),'valid',sum(p.get('onset_seconds') is not None for p in proposals.values()),flush=True)

if __name__=='__main__':main()
