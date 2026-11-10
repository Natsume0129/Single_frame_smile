"""Append raw JIANG true videos with real face crops and no model scores."""
import hashlib,json,re,shutil,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from build_monotonic_ranking_review import ROOT,OUT,read,table,sha,number,probe
from onset_peak import timestamps

def main():
 base=Path('E:/Dataset/person_smile_JIANG_true_20261031/JIANG');seq=table(base/'sequences.csv');qc=read(base/'preparation_qc.json')
 assert len(seq)==75 and not qc['issues'],'Resolve face tracking before publishing'
 inv=read(OUT/'inventory.json');assert not any(r['person']=='JIANG' and r['label']=='true' for r in inv['records'])
 def build(s):
  source=Path(s['source_video']);times=timestamps(source);faces=sorted(Path(s['sequence_dir']).glob('*.png'),key=number)
  curve=[dict(source_frame=i,seconds=float(t),raw_score=None,smoothed_score=None) for i,t in enumerate(times)]
  available=[dict(**curve[number(p)],path=str(p),url=p.as_uri(),sha256=sha(p)) for p in faces]
  r=dict(id=hashlib.sha256(str(source).encode()).hexdigest()[:24],person='JIANG',label='true',filename=source.name,source_video=str(source),clip=str(source),video_url=source.as_uri(),original_url=source.as_uri(),sequence_id=s['sequence_id'],source_group=s['source_group'],group_id='JIANG/'+s['source_group'],split=s['split'],split_origin='source_group_fixed_before_manual_annotation',face_directory=s['sequence_dir'],curve_csv=None,curve_sha256=None,curve=curve,available_frames=available,final_start_seconds=float(times[0]),final_peak_seconds=float(times[-1]),final_start_frame=0,final_peak_frame=len(times)-1,original_onset_seconds=None,onset_seconds=None,extra_frames=[],removed_frames=[],manual_only=True,candidate_origin='New JIANG true: manually selected temporal nodes; no smile ranking model',tracked_frames_in_interval=len(available))
  return probe(r)
 with ThreadPoolExecutor(max_workers=4) as ex:records=list(ex.map(build,seq))
 engine=Path(__file__).with_name('manual_ranking_sampling.js').read_text(encoding='utf8');code=engine+"\nlet s='';process.stdin.on('data',b=>s+=b);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(r=>activeRise(r)))));"
 records=json.loads(subprocess.check_output(['node','-e',code],input=json.dumps(records).encode()))
 assert all(not r['sample_frames'] and r['net_gain'] is None for r in records)
 backup=OUT/'before_JIANG_true_manual_20261031';backup.mkdir()
 for name in ['inventory.json','index.html','summary.json']:shutil.copy2(OUT/name,backup/name)
 inv['records']+=records;inv['inventory_revision']='JIANG-true-manual-20261031';inv['inventory_revision_sha256']=hashlib.sha256(json.dumps(inv['records'],sort_keys=True).encode()).hexdigest()
 inv['policy']['manual_only']='JIANG true has no scores. User-selected temporal nodes only; accepted later-is-stronger labels remain human segment assumptions.'
 page=Path(__file__).with_name('monotonic_ranking_review_v3.html').read_text(encoding='utf8')
 page=re.sub(r'<script>.*?const DATA=__PAYLOAD__;',lambda m:'<script>\n'+engine+'\nconst DATA=__PAYLOAD__;',page,count=1,flags=re.S)
 page=page.replace("score ${f.smoothed_score.toFixed(3)}","${r.manual_only?'manual node':'score '+f.smoothed_score.toFixed(3)}")
 page=page.replace("$('meta').textContent=(r.previously_excluded?", "$('meta').textContent=(r.manual_only?'MANUAL TIMELINE — no ranking model or scores. ':'')+(r.previously_excluded?")
 page=page.replace("$('warnings').textContent=(!r.eligible?", "$('warnings').textContent=r.manual_only?'Choose start/end and add intermediate frames. Accept confirms your intended temporal strength ordering; there are no model scores.':(!r.eligible?")
 page=page.replace("$('metrics').textContent=`Net score change:","$('metrics').textContent=r.manual_only?'No curve available. The horizontal line represents time only.':`Net score change:")
 page=page.replace("${p.old_score_increases?'yes':'NO / tie'}","${r.manual_only?'Not available':p.old_score_increases?'yes':'NO / tie'}")
 page=page.replace('<div class="small">Original video time','<div class="small" id="chartLegend">Original video time')
 page=page.replace("$('accept').disabled=false;", "$('chartLegend').textContent=r.manual_only?'Time axis only. Choose a click mode to set Start / End or add a middle node. No model score is displayed.':'Original video time. Grey: raw score; blue: original smooth; green/red: training bounds; numbered dots: selected frames.';$('accept').disabled=false;")
 marker='function draw(r){'
 timeline="""function draw(r){
 if(r.manual_only){const end=r.final_peak_seconds,x=t=>55+800*t/Math.max(.001,end);let h='<line x1="55" x2="855" y1="150" y2="150" stroke="#2775a8" stroke-width="3"/>';
 for(let i=0;i<=4;i++){let t=end*i/4;h+=`<line x1="${x(t)}" x2="${x(t)}" y1="145" y2="155" stroke="#2775a8"/><text x="${x(t)}" y="185" text-anchor="middle" font-size="13">${t.toFixed(2)} s</text>`;}
 if(r.manual_bounds){h+=`<rect x="${x(r.start_seconds)}" y="110" width="${x(r.peak_seconds)-x(r.start_seconds)}" height="80" fill="#15976a" opacity=".09"/><text x="${x(r.start_seconds)}" y="95" fill="#168b63">Start</text><text x="${x(r.peak_seconds)}" y="95" fill="#cf4141">End</text>`;}
 r.sample_frames.forEach((f,i)=>h+=`<circle cx="${x(f.seconds)}" cy="150" r="6" fill="#ed8626"/><text x="${x(f.seconds)}" y="130" text-anchor="middle" font-size="13">${i+1}</text>`);
 h+='<text x="450" y="240" text-anchor="middle">Timeline only — click to select real tracked frames</text>';$('curve').innerHTML=h;return;}
 """
 page=page.replace(marker,timeline)
 Path(__file__).with_name('monotonic_ranking_review_v3.html').write_text(page,encoding='utf8')
 (OUT/'inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'index.html').write_text(page.replace('__PAYLOAD__',json.dumps(inv,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8')
 (OUT/'summary.json').write_text(json.dumps(dict(total=len(inv['records']),added_JIANG_true=len(records),manual_true_split={s:sum(r['split']==s for r in records) for s in ['train','val','test']}),indent=2),encoding='utf8')
 (OUT/'review_version.js').write_text('window.onReviewVersion && window.onReviewVersion("JIANG-true-manual-20261031");',encoding='utf8')
 print('Published',len(records),'manual JIANG true videos; total',len(inv['records']))

if __name__=='__main__':main()
