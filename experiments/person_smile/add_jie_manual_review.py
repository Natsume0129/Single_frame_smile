"""Append three scoreless JIE categories to the existing manual-node review."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from build_monotonic_ranking_review import ROOT,OUT,read,table,sha,number,probe
from onset_peak import timestamps

def main():
 base=Path('E:/Dataset/person_smile_JIE_20261031/JIE');seq=table(base/'sequences.csv');qc=read(base/'preparation_qc.json')
 assert len(seq)==222 and not qc['issues'],'Resolve tracking issues before publication'
 inv=read(OUT/'inventory.json');assert not any(r['person']=='JIE' for r in inv['records'])
 def build(s):
  source=Path(s['source_video']);times=timestamps(source);faces=sorted(Path(s['sequence_dir']).glob('*.png'),key=number)
  curve=[dict(source_frame=i,seconds=float(t),raw_score=None,smoothed_score=None) for i,t in enumerate(times)]
  available=[dict(**curve[number(p)],path=str(p),url=p.as_uri(),sha256=sha(p)) for p in faces]
  return probe(dict(id=hashlib.sha256(str(source).encode()).hexdigest()[:24],person='JIE',label=s['label'],filename=source.name,source_video=str(source),clip=str(source),video_url=source.as_uri(),original_url=source.as_uri(),sequence_id=s['sequence_id'],source_group=s['source_group'],group_id='JIE/'+s['source_group'],split=s['split'],split_origin='source_group_fixed_before_manual_annotation',face_directory=s['sequence_dir'],curve_csv=None,curve_sha256=None,curve=curve,available_frames=available,final_start_seconds=float(times[0]),final_peak_seconds=float(times[-1]),final_start_frame=0,final_peak_frame=len(times)-1,original_onset_seconds=None,onset_seconds=None,extra_frames=[],removed_frames=[],manual_only=True,candidate_origin='New JIE category: manual temporal nodes; no ranking model',tracked_frames_in_interval=len(available)))
 with ThreadPoolExecutor(max_workers=4) as ex:records=list(ex.map(build,seq))
 engine=Path(__file__).with_name('manual_ranking_sampling.js').read_text(encoding='utf8')
 code=engine+"\nlet s='';process.stdin.on('data',b=>s+=b);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(r=>activeRise(r)))));"
 records=json.loads(subprocess.check_output(['node','-e',code],input=json.dumps(records).encode()))
 assert all(not r['sample_frames'] and r['net_gain'] is None for r in records)
 before=OUT/'before_JIE_manual_20261031';before.mkdir()
 for name in ['inventory.json','index.html','summary.json']:shutil.copy2(OUT/name,before/name)
 inv['records']+=records;inv['inventory_revision']='JIE-manual-20261031';inv['inventory_revision_sha256']=hashlib.sha256(json.dumps(inv['records'],sort_keys=True).encode()).hexdigest()
 inv['policy']['manual_only']='JIANG true and all JIE categories use manual temporal nodes; score fields are null.'
 page=Path(__file__).with_name('monotonic_ranking_review_v3.html').read_text(encoding='utf8')
 (OUT/'inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf8')
 (OUT/'index.html').write_text(page.replace('__PAYLOAD__',json.dumps(inv,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8')
 summary=dict(total=len(inv['records']),added_JIE=len(records),JIE={label:dict(clips=sum(r['label']==label for r in records),splits=dict(Counter(r['split'] for r in records if r['label']==label))) for label in ['polite','bitter','true']})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 (OUT/'review_version.js').write_text('window.onReviewVersion && window.onReviewVersion("JIE-manual-20261031");',encoding='utf8')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
