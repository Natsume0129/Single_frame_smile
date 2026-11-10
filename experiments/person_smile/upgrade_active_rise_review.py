"""Version the existing 520-clip review without replacing old decisions or final cuts."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from build_monotonic_ranking_review import ROOT,OUT,read,sha,number

def main():
 old=read(OUT/'inventory.json');assert old['schema_version']==1
 backup=OUT/'v1_before_active_rise';assert not backup.exists();backup.mkdir()
 for name in ['inventory.json','index.html','summary.json','README.txt']:shutil.copy2(OUT/name,backup/name)
 for r in old['records']:
  r['final_start_seconds']=r['start_seconds'];r['final_peak_seconds']=r['peak_seconds'];r['final_start_frame']=r['start_frame'];r['final_peak_frame']=r['peak_frame']
  images={number(p):p for p in Path(r['face_directory']).glob('*.png')}
  r['available_frames']=[dict(**p,path=str(images[p['source_frame']]),url=images[p['source_frame']].as_uri(),sha256=sha(images[p['source_frame']])) for p in r['curve'] if r['start_frame']<=p['source_frame']<=r['peak_frame'] and p['source_frame'] in images]
 engine=Path(__file__).with_name('active_rise_sampling.js').read_text(encoding='utf8')
 code=engine+"\nlet s='';process.stdin.on('data',b=>s+=b);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(r=>activeRise(r)))));"
 records=json.loads(subprocess.check_output(['node','-e',code],input=json.dumps(old['records']).encode('utf8')))
 inv={**old,'schema_version':2,'previous_dataset_id':old['dataset_id'],'records':records,'policy':{**old['policy'],'active_rise':'Time Gaussian sigma=0.07s, radius=0.18s; 8%-92% rise range; no isotonic fitting','sampling':'Approx .2s and score gap max(.02,.05*range); if fewer than5, relax time spacing while preserving >=2 source frame gap and score gap; otherwise ineligible','manual_bounds':'Allowed within existing final start/peak; editing invalidates prior acceptance'}}
 inv['dataset_id']=hashlib.sha256(json.dumps(inv,sort_keys=True).encode()).hexdigest()
 html=Path(__file__).with_name('monotonic_ranking_review.html').read_text(encoding='utf8')
 html=html.replace('const DATA=__PAYLOAD__;',engine+'\nconst DATA=__PAYLOAD__;')
 html=html.replace('let decisions={},currentId=null,visible=[];','let decisions={},priorDecisions={},currentId=null,visible=[];')
 marker="function persist(){"
 migration="""try{const previous=JSON.parse(localStorage.getItem('monotonic-review-'+DATA.previous_dataset_id)||'{}');for(const [id,d] of Object.entries(previous)){if(!byId[id]||decisions[id])continue;if(d.state==='reject')decisions[id]=d;else priorDecisions[id]=d;}}catch(e){}
function restoreBounds(){for(const [id,d] of Object.entries(decisions)){if(d.bounds&&byId[id]){const next=activeRise(byId[id],d.bounds);Object.assign(byId[id],next);if(d.state==='accept'&&!next.eligible){priorDecisions[id]=d;delete decisions[id];}}}}
restoreBounds();
"""
 html=html.replace(marker,migration+marker)
 html=html.replace("&&(d.state!=='accept'||byId[id].eligible)","")
 html=html.replace('<h4>Proposed training frames</h4>','''<div class="bar"><label>Training start (source s) <input id="boundStart" type="number" step="0.001" style="width:100px"></label><label>Training end (source s) <input id="boundEnd" type="number" step="0.001" style="width:100px"></label><button id="applyBounds">Apply interval</button><button id="resetBounds">Reset automatic interval</button></div><p class="small">Original final clip boundaries are unchanged. The green interval is only the new training range. Editing it clears any previous acceptance. Grey: raw; light blue: old smooth; dark blue: additional time-based smooth. Smoothing does not force an increasing curve.</p><h4>Proposed training frames</h4>''')
 html=html.replace("$('video').src=r.video_url;$('video').play().catch(()=>{});", "$('video').src=r.original_url;$('video').onloadedmetadata=()=>{$('video').currentTime=r.start_seconds;$('video').play().catch(()=>{});};$('boundStart').value=r.start_seconds.toFixed(6);$('boundEnd').value=r.peak_seconds.toFixed(6);")
 html=html.replace("$('video').currentTime=Math.max(0,f.seconds-r.start_seconds)","$('video').currentTime=f.seconds")
 html=html.replace("score ${f.smoothed_score.toFixed(3)}","score ${f.active_score.toFixed(3)}")
 html=html.replace("'Not reviewed';draw(r);","(priorDecisions[currentId]?'Previous acceptance retained for reference — interval/samples changed; please confirm again.':'Not reviewed');draw(r);")
 html=html.replace("[['raw_score','#adb9c5',1],['smoothed_score','#1d70a5',2.5]]","[['raw_score','#adb9c5',1],['smoothed_score','#b0cce2',1.5],['active_score','#1d70a5',2.5]]")
 html=html.replace('y(f.smoothed_score)','y(f.active_score)')
 html=html.replace("'Start'],[r.onset_seconds,'#d58d12','Onset'],[r.peak_seconds,'#cf4141','Peak']","'Train start'],[r.onset_seconds,'#d58d12','Old onset'],[r.peak_seconds,'#cf4141','Train end']")
 html=html.replace("label==='Onset'","label==='Old onset'")
 html=html.replace('source time approximately ${(r.start_seconds+$(\'video\').currentTime).toFixed(3)}','source time ${$(\'video\').currentTime.toFixed(3)}')
 html=html.replace("if(r)$('time').textContent=", "if(r&&$('video').currentTime>=r.peak_seconds){$('video').currentTime=r.start_seconds;}if(r)$('time').textContent=")
 html=html.replace("Clip time ${$('video').currentTime.toFixed(3)}", "Training time ${Math.max(0,$('video').currentTime-r.start_seconds).toFixed(3)}")
 html=html.replace("decisions[id]={state,note:","decisions[id]={state,bounds:[byId[id].start_seconds,byId[id].peak_seconds],note:")
 html=html.replace('schema_version:1,task:DATA.task','schema_version:2,task:DATA.task')
 html=html.replace('policy:DATA.policy,decisions,files:', 'policy:DATA.policy,decisions,prior_decisions:priorDecisions,files:')
 html=html.replace('start_frame:r.start_frame,peak_frame:r.peak_frame,decision:', 'start_frame:r.start_frame,peak_frame:r.peak_frame,start_seconds:r.start_seconds,peak_seconds:r.peak_seconds,decision:')
 html=html.replace("if(d.task!==DATA.task||d.dataset_id!==DATA.dataset_id)throw Error('Wrong dataset/version');", "if(d.task!==DATA.task)throw Error('Wrong task');if(d.dataset_id===DATA.previous_dataset_id){for(const [id,v] of Object.entries(d.decisions||{})){if(!byId[id])throw Error('Unknown clip');if(decisions[id])continue;if(v.state==='reject')decisions[id]=v;else priorDecisions[id]=v;}persist();refresh(currentId);alert('Old rejections imported. Old acceptances require confirmation of the new interval.');return;}if(d.dataset_id!==DATA.dataset_id)throw Error('Wrong dataset/version');")
 html=html.replace('decisions={...decisions,...d.decisions};persist();','decisions={...decisions,...d.decisions};priorDecisions={...priorDecisions,...(d.prior_decisions||{})};restoreBounds();persist();')
 html=html.replace('refresh();\n</script>',"""function updateBounds(bounds){try{const r=byId[currentId],next=activeRise(r,bounds);if(decisions[currentId])priorDecisions[currentId]=decisions[currentId];delete decisions[currentId];Object.assign(r,next);persist();render();}catch(e){alert(e.message);}}
$('applyBounds').onclick=()=>updateBounds([Number($('boundStart').value),Number($('boundEnd').value)]);
$('resetBounds').onclick=()=>updateBounds(null);
refresh();
</script>""")
 html=html.replace('Review the old-model curve and sampled faces.', 'Review the central rising interval and sampled faces. Automatic suggestions exclude the low and high plateaus; you can adjust the interval below the curve.')
 html=html.replace('Original video time (seconds). Grey: raw score; blue: smoothed score; green/red: final start/peak;', 'Original video time (seconds). Grey: raw; light blue: old smooth; dark blue: new smooth; green/red: training start/end;')
 (OUT/'inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2),encoding='utf8')
 (OUT/'index.html').write_text(html.replace('__PAYLOAD__',json.dumps(inv,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8')
 Path(__file__).with_name('monotonic_ranking_review_v2.html').write_text(html,encoding='utf8')
 summary=dict(total=len(records),eligible=sum(r['eligible'] for r in records),need_adjustment=sum(not r['eligible'] for r in records),prospective_pairs=sum(len(r['pairs']) for r in records if r['eligible']))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(summary)

if __name__=='__main__':main()
