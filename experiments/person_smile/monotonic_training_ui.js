/* UI extension: does not replace inventory ID, decisions, drafts, or storage keys. */
(function(){
 const target=280;
 const box=document.createElement('section');box.className='panel';box.id='trainingProgress';
 box.innerHTML='<h3>Training readiness — 280 total valid pairs per person / category</h3><p>Total = training + validation + test pairs from accepted clips. The three splits remain independent. Pairs less than two source frames apart are excluded from the valid count. Train requires ≥2 source groups; validation and test each require ≥1 independent source group.</p><div id="trainingCounts"></div><p id="exportReminder"></p><p><b>Export ONE JSON</b> when ready. The local worker detects the export in this review folder or your Downloads folder, prepares independent image datasets, and trains ready models sequentially. Unreviewed groups may remain pending.</p><div id="workerStatus">Checking local training worker…</div>';
 $('content').before(box);
 const validPairs=r=>r.pairs.filter(p=>p.later_frame-p.earlier_frame>=2);
 let oldReady=new Set(),initialized=false,groupedTraining=null;
 function readiness(){
  if(groupedTraining){let h='<table><tr><th>Person / category</th><th>Total pairs</th><th>Train</th><th>Val</th><th>Test</th><th>Status</th></tr>';for(const [g,v] of Object.entries(groupedTraining.groups)){if(!v.splits)continue;h+=`<tr><td>${g}</td><td>${v.pairs}</td><td>${v.splits.train.pairs}</td><td>${v.splits.val.pairs}</td><td>${v.splits.test.pairs}</td><td>${v.state}</td></tr>`;}$('trainingCounts').innerHTML=h+'</table>';$('exportReminder').textContent='Current training uses the frozen accepted-only export with newly assigned 70/15/15 source-group splits. REN excluded. Original per-clip review split labels are historical. New annotation edits do not change this running batch.';return;}

  const readyGroups=[];let html='<table><tr><th>Person / category</th><th>Total pairs / 280</th><th>Train pairs</th><th>Val pairs</th><th>Test pairs</th><th>Source groups T/V/T</th><th>Status</th></tr>';
  for(const g of groups){
   const selected=records.filter(r=>groupKey(r)===g&&decisions[r.id]?.state==='accept');
   const count=s=>selected.filter(r=>r.split===s).reduce((n,r)=>n+validPairs(r).length,0);
   const gc=s=>new Set(selected.filter(r=>r.split===s&&validPairs(r).length).map(r=>r.group_id)).size;
   const n=count('train')+count('val')+count('test'),ok=n>=target&&gc('train')>=2&&gc('val')>=1&&gc('test')>=1;
   if(n>=target)readyGroups.push(g);
   html+=`<tr><td>${g}</td><td><b>${n} / ${target}</b></td><td>${count('train')}</td><td>${count('val')}</td><td>${count('test')}</td><td>${['train','val','test'].map(gc).join(' / ')}</td><td>${ok?'Ready: export to train':n>=target?'280 reached; add missing independent split data':'Keep reviewing'}</td></tr>`;
  }
  $('trainingCounts').innerHTML=html+'</table>';
  $('exportReminder').textContent=readyGroups.length?'280 total pairs reached: '+readyGroups.join(', ')+'. Export your decisions; the worker will train groups with independent validation/test data.':'No group has reached 280 total valid pairs yet.';
  const newGroups=readyGroups.filter(g=>!oldReady.has(g));
  if(initialized&&newGroups.length)alert('280 TOTAL valid pairs reached: '+newGroups.join(', ')+'. Export one JSON to save the snapshot and queue eligible models.');
  oldReady=new Set(readyGroups);initialized=true;
 }
 const originalCounts=updateCounts;updateCounts=function(){originalCounts();readiness();};readiness();
 window.onTrainingStatus=data=>{
  if(data.run&&Object.values(data.groups||{}).some(v=>v.splits)){groupedTraining=data;readiness();}
  const div=$('workerStatus');div.replaceChildren();const title=document.createElement('p');title.textContent=`Local worker: ${data.worker||'unknown'} · PID ${data.worker_pid||'?'} · Updated ${data.updated_at||''}`;div.append(title);
  if(data.last_export_error){const e=document.createElement('p');e.className='warn';e.textContent='Export check failed: '+data.last_export_error.error;div.append(e);}
  for(const [name,v] of Object.entries(data.groups||{})){const p=document.createElement('p');p.textContent=name+': '+v.state+(v.output?' · '+v.output:'')+(v.error?' · '+v.error:'');div.append(p);}
 };
 function poll(){const s=document.createElement('script');s.src='training_status.js?v='+Date.now();s.onload=()=>s.remove();s.onerror=()=>{s.remove();$('workerStatus').textContent='Worker status unavailable. Run start_auto_training.ps1 in this folder.';};document.body.append(s);}
 poll();setInterval(poll,10000);
 // A future local page version can request a safe reload. Persist the current note and selection first.
 const viewKey=key+'-view';
 try{const v=JSON.parse(localStorage.getItem(viewKey)||'null');if(v&&byId[v.id]){$('group').value=groupKey(byId[v.id]);$('status').value='all';refresh(v.id);}}catch(e){}
 const saveView=()=>{try{localStorage.setItem(viewKey,JSON.stringify({id:currentId}));if(currentId&&decisions[currentId])decisions[currentId].note=$('note').value;persist();}catch(e){}};
 window.addEventListener('beforeunload',saveView);
 let loadedVersion=null;
 window.onReviewVersion=version=>{if(loadedVersion===null){loadedVersion=version;return;}if(version!==loadedVersion){saveView();location.reload();}};
 setInterval(()=>{const s=document.createElement('script');s.src='review_version.js?v='+Date.now();s.onload=s.onerror=()=>s.remove();document.body.append(s);},10000);
})();
