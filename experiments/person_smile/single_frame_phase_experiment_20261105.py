"""Raw-frame 3D/18D classification by normalized phase, global and personal.

Frozen upstream-aware clip split. Never interpolate scores or fit on test data.
Each frame is independently predicted; clip weighting is only an evaluation /
training weighting, never a vote or probability aggregation for prediction.
"""
import os
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(_name,'1')
import argparse,csv,hashlib,json,time,warnings
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/three_class_curve_classifier_20261102'
REGIONAL=ROOT/'output/regional_ranking_extension_20261103'
OUT=ROOT/'output/single_frame_phase_classification_20261105'
REGIONS=['full','periocular','nose','mouth_perioral','upper_face','lower_face']
MODEL_TYPES=['true','bitter','polite'];LABELS=['polite','bitter','true'];SEED=20261105
COLS=[r+'__'+m for r in REGIONS for m in MODEL_TYPES]

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(p)

def prepare():
 OUT.mkdir(exist_ok=True);assert not (OUT/'dataset.npz').exists(),'Dataset snapshot exists; use --train / --report to resume'
 clips=read(BASE/'feature_manifest.json');assert len(clips)==1215
 metadata=[];features=[];coverage=Counter();group_splits=defaultdict(set)
 for r in clips:
  p=REGIONAL/'curves'/r['person']/r['label']/r['id']/'data.json';d=read(p);points=d['points'];duration=d['interval_seconds']
  assert duration>0 and abs(duration-r['features']['duration'])<1e-6
  bins=set()
  for f in points:
   u=f['seconds']/duration;assert -1e-6<=u<=1+1e-6
   phase=min(9,max(0,int(np.floor(10*u+1e-10))));bins.add(phase)
   values=[float(f[c]) for c in COLS];assert np.isfinite(values).all() and all(-1e-6<=v<=9+1e-6 for v in values)
   metadata.append(dict(row_id=len(metadata),frame_id=r['id']+':'+str(f['source_frame']),clip_id=r['id'],person=r['person'],type=r['label'],split=r['split'],component=r['component'],source_group=r['source_group'],source_frame=f['source_frame'],source_seconds=f['source_seconds'],clip_seconds=f['seconds'],normalized_time=u,phase=phase))
   features.append(values)
  coverage[len(bins)]+=1;group_splits[r['component']].add(r['split'])
 assert len(metadata)==26680 and len({r['frame_id'] for r in metadata})==len(metadata)
 assert all(len(x)==1 for x in group_splits.values())
 X=np.array(features,np.float64);np.savez_compressed(OUT/'dataset.npz',X=X)
 save(OUT/'frame_metadata.json',metadata)
 with (OUT/'frame_labels.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
 protocol=dict(source_manifest=str(BASE/'feature_manifest.json'),source_manifest_sha256=hashlib.sha256((BASE/'feature_manifest.json').read_bytes()).hexdigest(),feature_space_definitions={'3d':COLS[:3],'18d':COLS},
  split_counts={s:dict(clips=sum(r['split']==s for r in clips),frames=sum(r['split']==s for r in metadata),groups=len({r['component'] for r in clips if r['split']==s})) for s in ['train','val','test']},
  phase_rule='u=(source_time-reviewed_onset)/(reviewed_peak-reviewed_onset); bins [0,.1),...,[.9,1]; machine-precision boundary epsilon1e-10. Each observed frame assigned once; no interpolation or duplicated filling.',
  label_rule='Frame inherits the human clip category. Early neutral/ambiguous frames are not independently human-labelled as visually recognizable smiles.',
  selection_rule='For each stage and scope, choose validation clip-equal frame accuracy, then clip-equal macro-F1. Only training frames fit scaling/model. Same candidate grid in3D/18D.',
  test_policy='Same frozen clip/source components and upstream exposure protections as baseline. Previously inspected test cohort; exploratory extension. Every frame of a clip and every phase retain one split.',
  training_weight='1/(number of train frames in the same clip and phase), normalized to mean1. This balances clip contributions; no clip voting is used.',
  primary_metric='clip_equal_accuracy = mean of within-clip single-frame accuracy in the phase. Ordinary frame accuracy also reported. TYPE cells are class-conditional accuracy/recall.',
  seed=SEED,frames=len(metadata),clips=len(clips),coverage_histogram=dict(coverage),
  notes=['3D uses the matched full-face control trio, a strict subset of18D; not the earlier stochastic-augmentation full-face trio.',
   'No smooth scores, temporal deltas, duration, phase, person ID, label or filename enter the feature vectors.',
   'Phase stratification uses reviewed onset and future peak, so this is offline relative-phase separability, not online early recognition.',
   'Upstream ranking models and masks are participant-specific; general here means pooled downstream classifier, not unseen-person end-to-end generalization.',
   'Some clips have no frame in a bin. Coverage and supports vary; no missing-bin frames are synthesized.',
   'Personal validation/test have relatively few independent clips/source groups; uncertainty is larger than frame counts suggest.'])
 save(OUT/'protocol.json',protocol);print(json.dumps(protocol['split_counts']),flush=True)

_DATA=None
def load_worker():
 global _DATA
 meta=read(OUT/'frame_metadata.json');arr=np.load(OUT/'dataset.npz')['X'];_DATA=(meta,arr)

def weights(rows):
 counts=Counter(r['clip_id'] for r in rows);w=np.array([1/counts[r['clip_id']] for r in rows],float);return w/w.mean()

def support(rows):
 return dict(frames=len(rows),clips=len({r['clip_id'] for r in rows}),groups=len({r['component'] for r in rows}),by_type={l:dict(frames=sum(r['type']==l for r in rows),clips=len({r['clip_id'] for r in rows if r['type']==l})) for l in LABELS})

def measure(rows,pred):
 y=np.array([r['type'] for r in rows]);w=weights(rows)
 return {**support(rows),'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),
  'clip_equal_accuracy':float(accuracy_score(y,pred,sample_weight=w)), 'clip_equal_balanced_accuracy':float(balanced_accuracy_score(y,pred,sample_weight=w)),
  'clip_equal_macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0,sample_weight=w)), 'confusion_matrix':confusion_matrix(y,pred,labels=LABELS).tolist()}

def candidates():
 for c in [.1,1,10]:yield 'logistic_C'+str(c),LogisticRegression(C=c,max_iter=3000,random_state=SEED)
 for c in [.1,1,10,100]:yield 'rbf_C'+str(c),SVC(C=c,gamma='scale',random_state=SEED)
 for depth in [6,None]:yield 'forest_depth'+str(depth),RandomForestClassifier(n_estimators=200,max_depth=depth,min_samples_leaf=3,n_jobs=1,random_state=SEED)
 for leaves in [7,15]:yield 'hist_leaves'+str(leaves),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=leaves,learning_rate=.05,min_samples_leaf=10,l2_regularization=1,early_stopping=False,random_state=SEED)
 yield 'mlp_32_16',MLPClassifier(hidden_layer_sizes=(32,16),alpha=.01,batch_size=64,learning_rate_init=.003,max_iter=300,early_stopping=False,n_iter_no_change=25,random_state=SEED)

def train_job(job):
 scope,dimension,person,phase=job;ident=(person or 'all')+f'/phase_{phase:02d}';dest=OUT/'models'/f'{scope}_{dimension}d'/ident
 if (dest/'result.json').exists():return read(dest/'result.json')
 meta,allX=_DATA;indices=[r['row_id'] for r in meta if r['phase']==phase and (person is None or r['person']==person)];rows=[meta[i] for i in indices];X=allX[indices,:dimension];y=np.array([r['type'] for r in rows]);masks={s:np.array([r['split']==s for r in rows]) for s in ['train','val','test']};parts={s:[r for r in rows if r['split']==s] for s in masks}
 assert all(set(y[m])==set(LABELS) for m in masks.values()),'Insufficient class coverage'
 trainw=weights(parts['train']);best=None;trials=[];started=time.time();dest.mkdir(parents=True,exist_ok=True)
 with threadpool_limits(limits=1):
  for name,estimator in candidates():
   model=Pipeline([('scale',StandardScaler()),('clf',estimator)])
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always');model.fit(X[masks['train']],y[masks['train']],scale__sample_weight=trainw,clf__sample_weight=trainw)
   predicted=model.predict(X[masks['val']]);metrics=measure(parts['val'],predicted);rank=(metrics['clip_equal_accuracy'],metrics['clip_equal_macro_f1'])
   trials.append(dict(model=name,validation=metrics,warnings=list(dict.fromkeys(str(w.message) for w in ws))))
   if best is None or rank>best[0]:best=(rank,name,model,metrics)
  # Validation choice committed before evaluating test predictions.
  save(dest/'selection_before_test.json',dict(selected_model=best[1],validation=best[3],trials=trials))
  predicted=best[2].predict(X[masks['test']]);test=measure(parts['test'],predicted)
 joblib.dump(dict(model=best[2],columns=COLS[:dimension],scope=scope,person=person,phase=phase,labels=LABELS),dest/'classifier.joblib')
 cells=[]
 for who in sorted({r['person'] for r in parts['test']}):
  for label in LABELS:
   ix=[i for i,r in enumerate(parts['test']) if r['person']==who and r['type']==label];sub=[parts['test'][i] for i in ix];correct=np.array([predicted[i]==label for i in ix],int)
   clip_acc=[]
   for clip in sorted({r['clip_id'] for r in sub}):clip_acc.append(float(np.mean([ok for r,ok in zip(sub,correct) if r['clip_id']==clip])))
   cells.append(dict(person=who,type=label,n_frames=len(ix),n_clips=len(clip_acc),n_groups=len({r['component'] for r in sub}),correct=int(correct.sum()),accuracy=float(correct.mean()) if len(ix) else None,clip_equal_accuracy=float(np.mean(clip_acc)) if clip_acc else None,clip_accuracy_sum=float(sum(clip_acc))))
 predictions=[dict(row_id=r['row_id'],frame_id=r['frame_id'],clip_id=r['clip_id'],person=r['person'],type=r['type'],phase=phase,predicted=str(p)) for r,p in zip(parts['test'],predicted)]
 save(dest/'test_predictions.json',predictions)
 result=dict(scope=scope,dimension=dimension,phase=phase,phase_label=f'{phase*10}–{(phase+1)*10}%',person=person,selected_model=best[1],status='complete',selection_metric='validation clip-equal accuracy then clip-equal macro-F1',train=support(parts['train']),validation=best[3],test=test,cells=cells,model_path=str(dest/'classifier.joblib'),prediction_path=str(dest/'test_predictions.json'),elapsed_seconds=time.time()-started)
 save(dest/'result.json',result);return result

def report():
 protocol=read(OUT/'protocol.json');reports={}
 for scope in ['general','personal']:
  for dimension in [3,18]:
   key=f'{scope}_{dimension}d';results=[read(p) for p in sorted((OUT/'models'/key).glob('*/phase_*/result.json'))];expected=10 if scope=='general' else 50
   assert len(results)==expected,(key,len(results));reports[key]=dict(title=f'{scope.capitalize()} single-frame classifier — {dimension}D',mode=scope,dimension=dimension,phase_results=results)
 # The identical test frame keys must be evaluated in every experiment.
 frame_sets=[]
 for key,r in reports.items():
  ids=[x['frame_id'] for v in r['phase_results'] for x in read(v['prediction_path'])];assert len(ids)==len(set(ids))==3826;frame_sets.append(set(ids))
 assert all(s==frame_sets[0] for s in frame_sets)
 save(OUT/'report_data.json',dict(protocol=protocol,reports=reports));print('Four reports assembled; identical 3826 test frames in every experiment.',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
 if a.prepare:prepare()
 if a.train:
  persons=sorted({r['person'] for r in read(OUT/'frame_metadata.json')});jobs=[(scope,dim,person,phase) for scope in ['general','personal'] for dim in [3,18] for person in ([None] if scope=='general' else persons) for phase in range(10)]
  save(OUT/'experiment_plan.json',dict(jobs=[dict(scope=s,dimension=d,person=p,phase=t) for s,d,p,t in jobs],candidate_names=[name for name,_ in candidates()],selection_rule='validation clip-equal accuracy then clip-equal macro F1; model fitting and weighted standardization on train only',test_rule='One evaluation after choosing each phase model; no test feedback',model_count=120))
  done=0
  with ProcessPoolExecutor(max_workers=a.workers,initializer=load_worker) as pool:
   fs=[pool.submit(train_job,job) for job in jobs]
   for f in as_completed(fs):
    r=f.result();done+=1;save(OUT/'progress.json',dict(completed=done,total=len(jobs),last=dict(scope=r['scope'],dimension=r['dimension'],person=r['person'],phase=r['phase'])));print(done,'/120',r['scope'],r['dimension'],r['person'],r['phase'],r['selected_model'],'validation clip-equal ACC',round(r['validation']['clip_equal_accuracy'],4),flush=True)
 if a.report:report()

if __name__=='__main__':main()
