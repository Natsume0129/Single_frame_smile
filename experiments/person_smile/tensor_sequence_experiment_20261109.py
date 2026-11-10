"""Clip classification from full 18-channel trajectories: fixed-grid vs native length."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,copy,hashlib,json,random,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score
from tensor_kernel_20261109 import run_arm,run_raw73_control

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'output/tensor_sequence_classification_20261109'
FRAME=ROOT/'output/single_frame_phase_classification_20261105'; SPLITS=ROOT/'output/three_class_curve_classifier_20261102/feature_manifest.json'; BASE=ROOT/'output/regional_ranking_extension_20261103/classifier'
LABELS=['polite','bitter','true']; L=20; SEEDS=[20261109,20261119]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p):return dict(n=len(y),correct=int(np.sum(np.asarray(y)==np.asarray(p))),accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),confusion_matrix=confusion_matrix(y,p,labels=LABELS).tolist())
def rank(m):return m['accuracy'],m['macro_f1']

def fixed_grid(r):
 t=r['times'];x=r['x'];grid=np.linspace(0,r['duration'],L);seg=np.cumsum(r['breaks']);keep=np.ones(L,bool);sids=[]
 for i,u in enumerate(grid):
  j=int(np.searchsorted(t,u));near=min((k for k in [j-1,j] if 0<=k<len(t)),key=lambda k:abs(t[k]-u))
  if abs(t[near]-u)<1e-8:sids.append(seg[near]);continue
  assert 0<j<len(t)
  if seg[j]!=seg[j-1]:keep[i]=False
  sids.append(seg[j-1])
 vals=np.stack([np.interp(grid,t,x[:,j]) for j in range(x.shape[1])],axis=1);vals[~keep]=0
 ids=np.asarray(sids)[keep];b=np.r_[True,ids[1:]!=ids[:-1]]
 return {**r,'x':vals[keep],'times':grid[keep],'breaks':b},vals,grid,keep

def dataset():
 clips=read(SPLITS);fm=read(FRAME/'frame_metadata.json');columns=read(FRAME/'protocol.json')['feature_space_definitions']['18d'];X=np.load(FRAME/'dataset.npz')['X'];assert X.shape==(26680,18) and len(columns)==18 and np.isfinite(X).all()
 index={r['id']:[] for r in clips}
 for i,f in enumerate(fm):assert i==f['row_id'];index[f['clip_id']].append(i)
 native=[];fixed=[];fx=[];ft=[];ok=[];offsets=[0];sourceframes=[]
 for c in clips:
  ids=index[c['id']];fs=[fm[i] for i in ids];assert all((f['person'],f['type'],f['split'],f['component'])==(c['person'],c['label'],c['split'],c['component']) for f in fs)
  t=np.array([f['clip_seconds'] for f in fs],float);frames=np.array([f['source_frame'] for f in fs]);assert t[0]==0 and np.all(np.diff(t)>0) and np.all(np.diff(frames)>0) and abs(t[-1]-c['features']['duration'])<1e-6
  r={k:c[k] for k in ['id','person','label','component','split','source_video']};r.update(times=t,x=X[ids],duration=float(t[-1]),breaks=np.r_[True,np.diff(t)>.15]);native.append(r)
  f,vals,grid,keep=fixed_grid(r);fixed.append(f);fx.append(vals);ft.append(grid);ok.append(keep);sourceframes.extend(frames.tolist());offsets.append(offsets[-1]+len(t))
 assert len(native)==1215 and len(set(c['id'] for c in native))==1215
 splitsets={s:{r['component'] for r in native if r['split']==s} for s in ['train','val','test']}
 assert not (splitsets['train']&splitsets['val'] or splitsets['train']&splitsets['test'] or splitsets['val']&splitsets['test'])
 counts={s:dict(clips=sum(r['split']==s for r in native),groups=len(gs),frames=sum(len(r['times']) for r in native if r['split']==s)) for s,gs in splitsets.items()}
 assert [counts[s]['clips'] for s in ['train','val','test']]==[879,159,177]
 np.savez_compressed(OUT/'tensor_dataset.npz',clip_ids=np.array([r['id'] for r in native]),columns=np.array(columns),native_values=np.concatenate([r['x'] for r in native]).astype('float32'),native_times=np.concatenate([r['times'] for r in native]),native_offsets=offsets,source_frames=sourceframes,native_breaks=np.concatenate([r['breaks'] for r in native]),fixed_values=np.asarray(fx,dtype='float32'),fixed_times=ft,fixed_valid=ok)
 save(OUT/'clip_index.json',[{k:r[k] for k in ['id','person','label','component','split','source_video','duration']}|dict(native_length=len(r['times']),fixed_supported=int(np.sum(k))) for r,k in zip(native,ok)])
 return {'fixed':fixed,'native':native},counts,columns

def time_weights(seq):
 t=seq['times'];dt=np.diff(t)*~seq['breaks'][1:];w=np.r_[dt,0]+np.r_[0,dt];return w/w.sum() if w.sum()>0 else np.full(len(t),1/len(t))

def norm_stats(rs):
 mean=np.mean([r['x'].mean(0) for r in rs],axis=0);second=np.mean([(r['x']**2).mean(0) for r in rs],axis=0)
 d=np.log1p([r['duration'] for r in rs]);return dict(mean=mean.tolist(),std=np.maximum(np.sqrt(np.maximum(0,second-mean**2)),1e-5).tolist(),dur_mean=float(d.mean()),dur_std=float(max(1e-5,d.std())))

def arrays(rs,norm,device):
 lens=np.array([len(r['x']) for r in rs],dtype=int);tokens=np.zeros((len(rs),max(lens),21),np.float32);weights=np.zeros(tokens.shape[:2],np.float32);ctx=[]
 for i,r in enumerate(rs):
  t=r['times'];x=(r['x']-norm['mean'])/norm['std'];dw=time_weights(r);dt=np.diff(t,prepend=t[0]);br=r['breaks'].copy();br[0]=False
  tokens[i,:len(t)]=np.column_stack([x,t/r['duration'],np.log1p(dt/.04),br]);weights[i,:len(t)]=dw
  ctx.append(np.r_[(np.log1p(r['duration'])-norm['dur_mean'])/norm['dur_std'],x[-1]-x[0],x[0],x[-1],dw@x])
 return (torch.from_numpy(tokens).to(device),torch.from_numpy(weights).to(device),torch.tensor(np.array(ctx),dtype=torch.float32,device=device),torch.tensor(lens,dtype=torch.long))

def batch(arr,ids):
 x,w,c,lens=arr;length=lens[ids];n=int(length.max());return x[ids,:n],w[ids,:n],c[ids],length

class TemporalNet(nn.Module):
 def __init__(self,arch,context):
  super().__init__();self.arch=arch;self.context=context;h=48
  if arch=='bigru':self.enc=nn.GRU(21,h//2,bidirectional=True,batch_first=True)
  else:
   self.input=nn.Linear(21,h);self.conv=nn.ModuleList([nn.Conv1d(h,h,3,dilation=d,padding=d) for d in [1,2,4]]);self.norm=nn.ModuleList([nn.LayerNorm(h) for _ in self.conv]);self.drop=nn.Dropout(.12)
  self.head=nn.Sequential(nn.Linear(h*3+(19 if context=='delta' else 73),64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,3))
 def forward(self,x,w,c,lengths):
  device=x.device;mask=torch.arange(x.shape[1],device=device)[None,:]<lengths.to(device)[:,None];m=mask[:,:,None]
  if self.arch=='bigru':
   enc,_=self.enc(pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False));h,_=pad_packed_sequence(enc,batch_first=True,total_length=x.shape[1])
  else:
   h=self.input(x)*m
   for conv,norm in zip(self.conv,self.norm):h=norm(h+self.drop(torch.nn.functional.gelu(conv((h*m).transpose(1,2)).transpose(1,2))))*m
  pooled=(h*w[:,:,None]).sum(1);last=h[torch.arange(len(x),device=device),lengths.to(device)-1];context=c[:,:19] if self.context=='delta' else c
  return self.head(torch.cat([pooled,h[:,0],last,context],dim=1))

@torch.no_grad()
def probabilities(model,arr):
 model.eval();chunks=[]
 for ids in np.array_split(np.arange(len(arr[3])),max(1,int(np.ceil(len(arr[3])/96)))):chunks.append(model(*batch(arr,ids)).softmax(1).cpu().numpy())
 return np.concatenate(chunks)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def fit_seed(arm,arch,context,seed,train,val,train_arr,val_arr,norm,device):
 dest=OUT/arm/(arch+'_'+context)/str(seed);dest.mkdir(parents=True,exist_ok=True);checkpoint=dest/'model.pt'
 seed_all(seed);model=TemporalNet(arch,context).to(device)
 if checkpoint.exists():
  model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state_dict']);return model
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.002);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=6,min_lr=.00008)
 y=np.array([LABELS.index(r['label']) for r in train]);v=np.array([LABELS.index(r['label']) for r in val]);yt=torch.as_tensor(y,device=device);cw=torch.as_tensor(len(y)/(3*np.bincount(y,minlength=3)),dtype=torch.float32,device=device)
 loss_fn=nn.CrossEntropyLoss(weight=cw,label_smoothing=.025);bestloss=float('inf');best=None;history=[];bestep=0;rng=np.random.default_rng(seed)
 for epoch in range(1,101):
  model.train();order=rng.permutation(len(y));trainloss=0.
  for k in range(0,len(order),64):
   ids=order[k:k+64];opt.zero_grad(set_to_none=True);z=model(*batch(train_arr,ids));loss=loss_fn(z,yt[ids]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();trainloss+=float(loss.detach())*len(ids)
  vp=probabilities(model,val_arr);vl=float(-np.log(vp[np.arange(len(v)),v].clip(1e-9)).mean());scheduler.step(vl)
  history.append(dict(epoch=epoch,training_loss=trainloss/len(y),validation_log_loss=vl,validation_accuracy=float((vp.argmax(1)==v).mean())))
  if vl<bestloss-1e-4:bestloss=vl;bestep=epoch;best=copy.deepcopy({k:a.detach().cpu() for k,a in model.state_dict().items()})
  if epoch>=22 and epoch-bestep>=18:break
 assert best is not None
 torch.save(dict(state_dict=best,arm=arm,arch=arch,context=context,seed=seed,norm=norm,labels=LABELS,selection_epoch=bestep),checkpoint);model.load_state_dict(best)
 save(dest/'history.json',history);save(dest/'checkpoint_info.json',dict(epoch=bestep,epochs_run=epoch,validation_log_loss=bestloss));return model

def paired(pred0,pred,y,components):
 c0=pred0==y;c=pred==y;gain=int((~c0&c).sum());loss=int((c0&~c).sum());gs=sorted(set(components));inds=[np.where(components==g)[0] for g in gs];rng=np.random.default_rng(SEEDS[0]);diff=(c.astype(float)-c0.astype(float))*100
 boot=[float(diff[np.concatenate([inds[i] for i in ii])].mean()) for ii in rng.integers(0,len(gs),(2000,len(gs)))];return dict(corrected=gain,new_errors=loss,difference_pp=float(diff.mean()),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),groups=len(gs))

def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'report_data.json').exists(),'Completed experiment exists; preserve result.'
 data,counts,cols=dataset();parts={a:{s:[r for r in rs if r['split']==s] for s in counts} for a,rs in data.items()}
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
 protocol=dict(signal='18 raw, simultaneous model responses (full, eyes, nose, mouth, upper face, lower face × true/bitter/polite); not smoothed',columns=cols,
  preprocessing='Fixed = 20 uniform real-time grid slots (train median 19); linearly interpolate scores only. Interior points in source gaps >0.15s masked. Native = all actual observations, zero padding only for batching, mask/packed lengths respected.',
  timing='Neural tokens: 18 training-standardized scores, normalized elapsed time, log1p(actual step dt / 0.04s), indicator of gap crossing. Global log1p(duration), endpoint changes; summary-readout additionally includes endpoints and observed-span time mean. No person ID, true category, or filename as predictor.',
  design='Same clip IDs, grouped split, native interval, global duration, kernel grid and neural budgets in both arms. One clip = one prediction. General classifier for five known participants. REN excluded.',
  neural='Two small architectures: bidirectional GRU (24 per direction) and masked dilated TCN (48 channels). Each with delta or summary readout, two fixed seeds; average probabilities within a configuration. AdamW, class-balanced CE, validation-log-loss early stopping.',
  non_neural='Cosine trajectory embeddings from observed spans + endpoint/mean/delta and log duration; validation selects degrees 3 or 7 and RBF C=.1/1/10. Separate raw 73D control uses SVM C=10.',
  selection='Freeze representation, architecture/readout and hyperparameters using validation ACC, then macro-F1, then declared order. Do not swap winner based on test ACC. Report prespecified family comparisons after freeze.',
  prior='Frozen published 73D RBF-SVM used smoothed scores; separate native raw73 SVM checks the effect of substituting raw inputs.',
  limits='Previously inspected test cohort; category-informed human-reviewed boundaries and participant-specific upstream models. Results are exploratory, offline, not new-person generalization. Sampling cadence/gaps can reflect recording conditions.',
  source_hashes={str(p):sha(p) for p in [SPLITS,FRAME/'dataset.npz',FRAME/'frame_metadata.json',BASE/'test_predictions.json']},seeds=SEEDS,labels=LABELS,grid_slots=L)
 save(OUT/'protocol.json',protocol);save(OUT/'split_counts.json',counts)
 yval={arm:[r['label'] for r in pp['val']] for arm,pp in parts.items()};winners={};allchoices={};fitted={};nrm={};rawctrl=run_raw73_control(parts['native']['train'],parts['native']['val'],OUT)
 for arm in ['fixed','native']:
  train,val=parts[arm]['train'],parts[arm]['val'];norm=norm_stats(train);nrm[arm]=norm;ta,va=arrays(train,norm,device),arrays(val,norm,device)
  kernel=run_arm(train,val,arm,OUT);choices=[];models={};name=kernel['selection']['selected_model'];models[name]=('kernel',kernel['model']);choices.append(dict(model=name,family='kernel',val=metrics(yval[arm],kernel['val_pred']),paths=[kernel['model_path']]))
  for arch in ['bigru','tcn']:
   for context in ['delta','summary']:
    members=[fit_seed(arm,arch,context,sd,train,val,ta,va,norm,device) for sd in SEEDS];pr=np.mean([probabilities(m,va) for m in members],axis=0);name=arch+'_'+context
    metric=metrics(yval[arm],np.array(LABELS)[pr.argmax(1)]);models[name]=('torch',members);choices.append(dict(model=name,family=arch,val=metric,paths=[str(OUT/arm/name/str(sd)/'model.pt') for sd in SEEDS]));print(arm,name,'val',round(metric['accuracy'],4),flush=True)
  best=max(choices,key=lambda c:rank(c['val']));winners[arm]=best;allchoices[arm]=choices;fitted[arm]=models;print('Selected',arm,best['model'],best['val']['accuracy'],flush=True)
 primary=max(['fixed','native'],key=lambda a:rank(winners[a]['val']));save(OUT/'selection_before_test.json',dict(primary_arm=primary,arms=winners,all_candidates=allchoices,raw73=rawctrl['selection']))
 # First use of held-out predictions for the new candidates is after all validation choices are frozen.
 test=parts['native']['test'];y=np.array([r['label'] for r in test]);components=np.array([r['component'] for r in test]);orig={r['id']:r for r in read(BASE/'test_predictions.json')};assert set(orig)=={r['id'] for r in test} and all(orig[r['id']]['label']==r['label'] for r in test)
 bp=np.array([orig[r['id']]['prediction'] for r in test]);rp=rawctrl['model'].predict(test);predictions={};arms={};family_comparison={}
 for arm in ['fixed','native']:
  tt=parts[arm]['test'];assert [r['id'] for r in tt]==[r['id'] for r in test];arr=arrays(tt,nrm[arm],device);by_family={}
  for family in ['kernel','bigru','tcn']:
   choice=max([c for c in allchoices[arm] if c['family']==family],key=lambda c:rank(c['val']));kind,model=fitted[arm][choice['model']]
   pred=model.predict(tt) if kind=='kernel' else np.array(LABELS)[np.mean([probabilities(m,arr) for m in model],axis=0).argmax(1)]
   by_family[family]=dict(selected_model=choice['model'],val=choice['val'],test=metrics(y,pred))
   if choice['model']==winners[arm]['model']:predictions[arm]=pred
  v=winners[arm];arms[arm]=dict(name=arm,selected_model=v['model'],val=v['val'],test=metrics(y,predictions[arm]),n_features=18,model_paths=v['paths']);family_comparison[arm]=by_family
 bl=metrics(y,bp);assert bl['correct']==158 and bl['n']==177
 pp={person:{k:metrics(y[[r['person']==person for r in test]],pred[[r['person']==person for r in test]]) for k,pred in {'baseline':bp,**predictions}.items()} for person in sorted({r['person'] for r in test})}
 predictions_out=[dict(id=r['id'],person=r['person'],label=r['label'],component=r['component'],fixed_prediction=str(predictions['fixed'][i]),native_prediction=str(predictions['native'][i]),baseline_prediction=str(bp[i]),raw73_prediction=str(rp[i])) for i,r in enumerate(test)]
 ex_i=next(i for i,r in enumerate(data['native']) if r['split']=='train' and not r['breaks'][1:].any());ex=data['native'][ex_i];fixed=data['fixed'][ex_i]
 audit=dict(native_length_min=min(len(r['times']) for r in data['native']),native_length_median=float(np.median([len(r['times']) for r in data['native']])),native_length_max=max(len(r['times']) for r in data['native']),clips_with_gaps=sum(int(r['breaks'].sum()>1) for r in data['native']),masked_fixed_slots=sum(L-len(r['times']) for r in data['fixed']),native_frames=sum(len(r['times']) for r in data['native']))
 report=dict(protocol=protocol,split_counts=counts,length_audit=audit,labels=LABELS,primary_arm=primary,arms=arms,baseline=dict(name='Published smoothed 73D RBF-SVM',test=bl),raw73_control=dict(name='Native raw 73D RBF-SVM',val=metrics(yval['native'],rawctrl['val_pred']),test=metrics(y,rp)),paired={a:paired(bp,predictions[a],y,components) for a in predictions},fixed_vs_native=paired(predictions['fixed'],predictions['native'],y,components),per_person=pp,family_comparison=family_comparison,test_predictions=predictions_out,
  input_example=dict(id=ex['id'],person=ex['person'],label=ex['label'],columns=cols,duration=ex['duration'],native=dict(times=ex['times'].tolist(),x=ex['x'].tolist()),fixed=dict(times=fixed['times'].tolist(),x=fixed['x'].tolist())))
 save(OUT/'report_data.json',report);save(OUT/'test_predictions.json',predictions_out);print('FINAL',primary,json.dumps(arms),flush=True)

if __name__=='__main__':main()
