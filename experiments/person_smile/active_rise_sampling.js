/* Shared deterministic rule used by the review UI and export validator. */
function activeRise(record, bounds=null) {
 const curve=record.curve.map(p=>({...p})),all=record.available_frames;
 // Time-based Gaussian smoothing. Never fit a monotonic/isotonic curve.
 curve.forEach((p,i)=>{
  let a=i,b=i;while(a>0&&curve[a].seconds-curve[a-1].seconds<=.15)a--;while(b+1<curve.length&&curve[b+1].seconds-curve[b].seconds<=.15)b++;
  let sum=0,weight=0;for(let j=a;j<=b;j++){const dt=curve[j].seconds-p.seconds;if(Math.abs(dt)>.18)continue;const w=Math.exp(-.5*(dt/.07)**2);sum+=w*curve[j].smoothed_score;weight+=w;}p.active_score=sum/weight;
 });
 const base=curve.filter(p=>p.source_frame>=record.final_start_frame&&p.source_frame<=record.final_peak_frame);
 let start=base[0].seconds,end=base.at(-1).seconds;
 if(bounds){start=bounds[0];end=bounds[1];if(!Number.isFinite(start)||!Number.isFinite(end)||start<record.final_start_seconds-1e-6||end>record.final_peak_seconds+1e-6||start>=end)throw Error('Interval must remain inside the existing final start-to-peak range.');}
 else {
  const peak=base.reduce((best,p,i)=>p.active_score>base[best].active_score?i:best,0);
  const low=base.slice(0,peak+1).reduce((best,p,i)=>p.active_score<base[best].active_score?i:best,0);
  const floor=base[low].active_score,amp=base[peak].active_score-floor;
  if(amp>.04){const lo=floor+.08*amp,hi=floor+.92*amp;const a=base.findIndex((p,i)=>i>=low&&i<=peak&&p.active_score>=lo);const b=base.findIndex((p,i)=>i>=a&&i<=peak&&p.active_score>=hi);if(a>=0&&b>a){start=base[a].seconds;end=base[b].seconds;}}
 }
 const scores=new Map(curve.map(p=>[p.source_frame,p.active_score]));
 const fs=all.filter(p=>p.seconds>=start-1e-6&&p.seconds<=end+1e-6).map(p=>({...p,active_score:scores.get(p.source_frame)}));
 if(fs.length){start=fs[0].seconds;end=fs.at(-1).seconds;}
 const gain=fs.length?fs.at(-1).active_score-fs[0].active_score:0,margin=Math.max(.02,.05*Math.max(0,gain));
 function pick(interval){
  if(!fs.length)return[];const result=[fs[0]];
  for(const f of fs.slice(1)){const prev=result.at(-1);if(f.source_frame-prev.source_frame>=2&&f.seconds-prev.seconds>=interval-1e-5&&f.active_score-prev.active_score>=margin)result.push(f);}
  return result;
 }
 let sampled=pick(.2),mode='~0.2 s + minimum score difference';if(sampled.length<5){sampled=pick(0);mode='Short/fast rise: minimum 2 source frames + minimum score difference';}
 const pairs=[];for(let gap=1;gap<sampled.length;gap++)for(let i=0;i+gap<sampled.length;i++)pairs.push({first:i+1,second:i+gap+1,earlier_frame:sampled[i].source_frame,later_frame:sampled[i+gap].source_frame,stronger:'later',old_score_increases:sampled[i+gap].smoothed_score>sampled[i].smoothed_score});
 const inRange=curve.filter(p=>p.seconds>=start&&p.seconds<=end);let drop=0;for(let i=1;i<inRange.length;i++)drop+=Math.max(0,inRange[i-1].active_score-inRange[i].active_score);
 return {...record,curve,start_seconds:start,peak_seconds:end,start_frame:fs[0]?.source_frame??record.final_start_frame,peak_frame:fs.at(-1)?.source_frame??record.final_peak_frame,
  onset_seconds:record.onset_seconds!==null&&record.onset_seconds>=start&&record.onset_seconds<=end?record.onset_seconds:null,
  sample_frames:sampled,pairs,eligible:sampled.length>=5&&gain>0,net_gain:gain,total_downward_change:drop,strict_nondecreasing:drop===0,pair_score_warnings:pairs.filter(p=>!p.old_score_increases).length,
  sampling_note:mode+`; score gap ≥ ${margin.toFixed(3)}. No forced monotonic fitting.`,score_margin:margin,manual_bounds:bounds};
}
if(typeof module!=='undefined')module.exports={activeRise};
