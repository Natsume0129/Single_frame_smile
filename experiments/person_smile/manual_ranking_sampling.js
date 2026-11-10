/* V3: original curve and temporal sampling, optional human bounds and extra frames. */
function activeRise(record,bounds=null,extras=[],removed=[]) {
 const start=bounds?bounds[0]:record.final_start_seconds,end=bounds?bounds[1]:record.final_peak_seconds;
 if(!Number.isFinite(start)||!Number.isFinite(end)||start<record.final_start_seconds-1e-6||end>record.final_peak_seconds+1e-6||start>=end)throw Error('Choose an interval inside the original final clip.');
 const fs=record.available_frames.filter(f=>f.seconds>=start-1e-6&&f.seconds<=end+1e-6).map(f=>({...f}));
 let samples=[];
 if(!record.manual_only&&fs.length>=5){
  samples=[fs[0]];for(const f of fs.slice(1))if(f.source_frame-samples.at(-1).source_frame>=2&&f.seconds-samples.at(-1).seconds>=.2-1e-5)samples.push(f);
  if(samples.length>=5){if(fs.at(-1).source_frame!==samples.at(-1).source_frame&&fs.at(-1).source_frame-samples.at(-2).source_frame>=2)samples[samples.length-1]=fs.at(-1);}
  else{
   const latest=[0,0,0,0,fs.length-1];let possible=true;
   for(let j=3;j>=0;j--){const ids=fs.map((f,i)=>i).filter(i=>i<latest[j+1]&&fs[latest[j+1]].source_frame-fs[i].source_frame>=2);if(!ids.length){possible=false;break;}latest[j]=ids.at(-1);}
   samples=[];if(possible)for(let j=0;j<5;j++){const target=fs[0].seconds+(fs.at(-1).seconds-fs[0].seconds)*j/4;const opts=fs.filter((f,i)=>i<=latest[j]&&(!samples.length||f.source_frame-samples.at(-1).source_frame>=2));samples.push(opts.reduce((a,b)=>Math.abs(a.seconds-target)<=Math.abs(b.seconds-target)?a:b));}
  }
 }
 if(record.manual_only&&bounds&&fs.length){samples=[fs[0]];if(fs.length>1)samples.push(fs.at(-1));}
 const extraIds=[...new Set(extras)].sort((a,b)=>a-b);
 for(const id of extraIds){const f=fs.find(x=>x.source_frame===id);if(!f)throw Error('An extra frame is outside the interval or has no tracked face. Remove it or widen the interval.');if(!samples.some(x=>x.source_frame===id))samples.push(f);}
 const removedIds=[...new Set(removed)].sort((a,b)=>a-b);
 samples=samples.filter(f=>!removedIds.includes(f.source_frame));
 samples.sort((a,b)=>a.source_frame-b.source_frame);
 const gaps=samples.slice(1).some((f,i)=>f.source_frame-samples[i].source_frame<2);
 const curve=record.curve.map(p=>{const {active_score,...q}=p;return q;});
 const segment=curve.filter(p=>p.seconds>=start-1e-6&&p.seconds<=end+1e-6);let drop=0;for(let i=1;i<segment.length;i++)drop+=Math.max(0,segment[i-1].smoothed_score-segment[i].smoothed_score);
 const gain=segment.length?segment.at(-1).smoothed_score-segment[0].smoothed_score:0,pairs=[];
 for(let gap=1;gap<samples.length;gap++)for(let i=0;i+gap<samples.length;i++)pairs.push({first:i+1,second:i+gap+1,earlier_frame:samples[i].source_frame,later_frame:samples[i+gap].source_frame,stronger:'later',old_score_increases:record.manual_only?null:samples[i+gap].smoothed_score>samples[i].smoothed_score});
 return {...record,curve,sample_frames:samples,pairs,start_seconds:start,peak_seconds:end,start_frame:Math.abs(start-record.final_start_seconds)<1e-6?record.final_start_frame:(fs[0]?.source_frame??record.final_start_frame),peak_frame:Math.abs(end-record.final_peak_seconds)<1e-6?record.final_peak_frame:(fs.at(-1)?.source_frame??record.final_peak_frame),onset_seconds:record.original_onset_seconds,
  manual_bounds:bounds,extra_frames:extraIds,removed_frames:removedIds,eligible:record.manual_only?samples.length>=2&&!gaps:samples.length>=5&&!gaps&&gain>0,net_gain:record.manual_only?null:gain,total_downward_change:record.manual_only?null:drop,strict_nondecreasing:record.manual_only?null:drop===0,pair_score_warnings:record.manual_only?0:pairs.filter(p=>!p.old_score_increases).length,
  sampling_note:record.manual_only?'Manual timeline nodes only; no ranking model or score.':gaps?'Extra frame is adjacent to another sample (<2 source frames). Remove that extra frame before acceptance.':'Original temporal sampling (~0.2 s; short clips: five frames >=2 source frames apart) + manual extra frames'};
}
if(typeof module!=='undefined')module.exports={activeRise};
