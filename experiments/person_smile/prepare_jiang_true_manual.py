"""Face tracking and source-group splits only; no ranking curves or auto pair labels."""
import json,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pipeline import ROOT,track_clip,digest,frame_number,assign_sequence_splits,write_csv,save_json

def main():
 cfg=json.loads((ROOT/'experiments/person_smile/config.JIANG_true.json').read_text());out=Path(cfg['output_root'])/'JIANG';out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
 clips=sorted((ROOT/'new_dataset/JIANG/true').glob('*.mp4'));assert len(clips)==75
 jobs=[(p,'true_'+digest(p)[:16]) for p in clips];assert len({s for p,s in jobs})==len(jobs)
 def run(job):
  p,s=job;track_clip(cfg,p,s,out);print('Tracked',p.name,flush=True)
 with ThreadPoolExecutor(max_workers=cfg['tracking_workers']) as pool:list(pool.map(run,jobs))
 rows=[];issues=[]
 for p,sid in jobs:
  detected=out/'tracking'/sid/'DetectedFaces'/sid
  tracks=[q/'0' for q in detected.iterdir() if q.is_dir() and (q/'0').is_dir() and list((q/'0').glob('*.png'))] if detected.exists() else []
  if len(tracks)!=1:issues.append(dict(source=str(p),sequence_id=sid,reason='missing_or_multiple_tracks',tracks=[str(t) for t in tracks]));continue
  frames=sorted(tracks[0].glob('*.png'),key=frame_number);dest=out/'sequences/true'/sid;dest.mkdir(parents=True,exist_ok=True)
  for f in frames:
   target=dest/f.name
   if not target.exists():shutil.copy2(f,target)
  rows.append(dict(label='true',sequence_id=sid,source_video=str(p),sequence_dir=str(dest),frame_count=len(frames),track_id=tracks[0].parent.name,filled_count=0))
 save_json(out/'preparation_qc.json',dict(total_sources=len(jobs),tracked_sequences=len(rows),issues=issues))
 if rows:assign_sequence_splits(rows,cfg);write_csv(out/'sequences.csv',rows)
 print('SUMMARY',len(rows),'tracked;',len(issues),'issues',flush=True)

if __name__=='__main__':main()
