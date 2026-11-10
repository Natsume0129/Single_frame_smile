"""Exploratory clip-level three-class classification with upstream-aware grouped holdout."""
import csv,hashlib,json,re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import joblib
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
GALLERY=ROOT/'output/three_model_clip_gallery_20261102'
OUT=ROOT/'output/three_class_curve_classifier_20261102'
LABELS=['polite','bitter','true']
SEED=20261102
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def group(r):return r['person']+'/'+re.sub(r'_P\d+$','',Path(r['source_video']).stem)

def main():
 assert not (OUT/'report.json').exists(),'Frozen experiment already exists; inspect before rerunning'
 OUT.mkdir(exist_ok=True);raw=read(GALLERY/'manifest.json');up=read(ROOT/'output/monotonic_accepted_training_20261101/accepted_manifest.json')
 # Link source families, duplicate source files and exact decoded face frames, including upstream train/val.
 parent={};seen={};pixel_cache={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(a,b):parent[find(b)]=find(a)
 def pixels(path):
  if path not in pixel_cache:
   with Image.open(path) as im:im=im.convert('RGB');pixel_cache[path]=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
  return pixel_cache[path]
 for r,upstream in [(r,True) for r in up]+[(r,False) for r in raw]:
  g=r['group_id'] if upstream else group(r);find(g)
  keys=[(r['person'],'source',r['source_sha256'])]
  if upstream:keys.append((r['person'],'upstream_component',r['split_group_id']))
  faces=[f['path'] for f in r['sample_frames']] if upstream else [f['face_path'] for f in r['frames']]
  keys.extend((r['person'],'pixels',pixels(p)) for p in faces)
  for key in keys:
   if key in seen:union(g,seen[key])
   else:seen[key]=g
 print('Source/face grouping complete',len(pixel_cache),'distinct face files',flush=True)
 exposure=defaultdict(set)
 for r in up:exposure[find(r['group_id'])].add(r['split'])
 assignments={};split_origins={}
 for r in raw:
  g=find(group(r));e=exposure.get(g,set())
  if e:
   assignments[g]='train' if 'train' in e else 'val' if 'val' in e else 'test';split_origins[g]='upstream_exposure_priority'
  elif g not in assignments:
   q=int(hashlib.sha256(('classifier-20261102/'+g).encode()).hexdigest()[:8],16)%100
   assignments[g]='train' if q<70 else 'val' if q<85 else 'test';split_origins[g]='new_group_fixed_hash'
 assert all(not (exposure.get(g,set())&{'train','val'}) for g,s in assignments.items() if s=='test')
 # No ground-truth label is used to choose a model feature: all three responses are always included.
 examples=[]
 for r in raw:
  p=GALLERY/'curves'/r['person']/r['label']/r['id']/'curve.csv'
  with p.open(encoding='utf-8-sig') as f:pts=list(csv.DictReader(f))
  t=np.array([float(p['seconds']) for p in pts]);duration=float(t[-1]-t[0]);assert duration>0
  features={'duration':duration};curves={}
  for label in LABELS:
   y=np.array([float(p[label+'_smooth']) for p in pts]);curves[label]=y;span=max(float(np.ptp(y)),.25)
   values=dict(start=float(y[0]),end=float(y[-1]),delta=float(y[-1]-y[0]),mean=float(np.trapezoid(y,t)/duration),range=float(np.ptp(y)),downward=float(-np.minimum(np.diff(y),0).sum()/span),variation=float(abs(np.diff(y)).sum()/span))
   for fraction in [.25,.5,.75]:values['progress_'+str(int(fraction*100))]=float((np.interp(t[0]+fraction*duration,t,y)-y[0])/span)
   features.update({label+'_'+k:v for k,v in values.items()})
  g=find(group(r));examples.append(dict(id=r['id'],person=r['person'],label=r['label'],source_video=r['source_video'],source_group=group(r),component=g,split=assignments[g],split_origin=split_origins[g],features=features))
 save(OUT/'feature_manifest.json',examples)
 masks={s:np.array([r['split']==s for r in examples]) for s in ['train','val','test']};y=np.array([r['label'] for r in examples])
 assert all(set(y[mask])==set(LABELS) for mask in masks.values())
 feature_sets={'duration_only':['duration'],'three_deltas':[l+'_delta' for l in LABELS],'deltas_and_duration':['duration']+[l+'_delta' for l in LABELS],'levels_and_shape':list(examples[0]['features'])}
 trials=[];selected={};models={}
 for name,columns in feature_sets.items():
  X=np.array([[r['features'][k] for k in columns] for r in examples]);assert np.all(np.isfinite(X))
  candidates=[('logistic_C'+str(c),make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,class_weight='balanced',random_state=SEED))) for c in [.1,1,10]]
  candidates += [('forest_depth'+str(depth),RandomForestClassifier(n_estimators=250,max_depth=depth,min_samples_leaf=5,class_weight='balanced_subsample',random_state=SEED,n_jobs=4)) for depth in [4,8]]
  best=None
  for candidate,model in candidates:
   model.fit(X[masks['train']],y[masks['train']]);pred=model.predict(X[masks['val']]);score=float(f1_score(y[masks['val']],pred,average='macro',zero_division=0));trials.append(dict(features=name,model=candidate,val_macro_f1=score))
   if best is None or score>best[0]:best=(score,candidate,model)
  selected[name]=dict(model=best[1],val_macro_f1=best[0],feature_columns=columns);models[name]=best[2]
 winner=max(selected,key=lambda n:selected[n]['val_macro_f1'])
 # Freeze the validation selection before evaluating the test set once.
 save(OUT/'selection_before_test.json',dict(winner=winner,selected=selected,trials=trials))
 def metrics(actual,pred):return dict(n=len(actual),accuracy=float(accuracy_score(actual,pred)),balanced_accuracy=float(balanced_accuracy_score(actual,pred)),macro_f1=float(f1_score(actual,pred,labels=LABELS,average='macro',zero_division=0)),confusion_matrix=confusion_matrix(actual,pred,labels=LABELS).tolist())
 test_results={};predictions={}
 for name,meta in selected.items():
  X=np.array([[r['features'][k] for k in meta['feature_columns']] for r in examples]);pred=models[name].predict(X[masks['test']]);predictions[name]=pred;test_results[name]=metrics(y[masks['test']],pred)
 best_pred=predictions[winner];test_rows=[r for r in examples if r['split']=='test'];per_person={}
 for person in sorted({r['person'] for r in test_rows}):
  mask=np.array([r['person']==person for r in test_rows]);per_person[person]=metrics(y[masks['test']][mask],best_pred[mask])
 baseline={}
 for name,fn in [('highest_mean',lambda r:max(LABELS,key=lambda l:r['features'][l+'_mean'])),('largest_delta',lambda r:max(LABELS,key=lambda l:r['features'][l+'_delta']))]:baseline[name]=metrics(y[masks['test']],np.array([fn(r) for r in test_rows]))
 majority=Counter(y[masks['train']]).most_common(1)[0][0];baseline['majority']=metrics(y[masks['test']],np.array([majority]*len(test_rows)))
 report=dict(seed=SEED,total_clips=len(examples),labels=LABELS,split_counts={s:dict(clips=int(m.sum()),groups=len({r['component'] for r in examples if r['split']==s}),classes=dict(Counter(y[m]))) for s,m in masks.items()},selected_by_validation=winner,selected_models=selected,test_results=test_results,baselines=baseline,test_by_person=per_person,winner_classification_report=classification_report(y[masks['test']],best_pred,labels=LABELS,output_dict=True,zero_division=0),audit=dict(test_groups_exposed_to_upstream_train_or_val=0,source_groups_shared_across_splits=0,decoded_duplicate_components_shared_across_splits=0,person_id_used_as_feature=False,label_used_to_choose_curve=False,standardization_fit_on_train_only=True),limitations=['Reviewed onset/peak boundaries were selected with the corresponding category model; this remains category-informed window selection.','Some ranking-training labels were temporally inferred from human-accepted clips.','This experiment is for already-personalized participants, not unseen people or blind full-video classification.','Several feature sets are reported as a predeclared exploratory ablation; model/feature selection used validation macro-F1 only.'])
 save(OUT/'report.json',report);joblib.dump(dict(model=models[winner],feature_columns=selected[winner]['feature_columns'],labels=LABELS),OUT/'classifier.joblib')
 for r,pred in zip(test_rows,best_pred):r['prediction']=str(pred)
 save(OUT/'test_predictions.json',test_rows)
 cm=np.array(test_results[winner]['confusion_matrix']);fig,ax=plt.subplots(figsize=(6.6,5.3));im=ax.imshow(cm/cm.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=[x.capitalize() for x in LABELS],yticklabels=[x.capitalize() for x in LABELS],xlabel='Predicted category',ylabel='Annotated category',title='Held-out clips: three-class curve classifier')
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{cm[i,j]/cm[i].sum():.1%}',ha='center',va='center',color='white' if cm[i,j]/cm[i].sum()>.55 else '#172c42',fontsize=13)
 fig.colorbar(im,ax=ax,label='Within annotated category');fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=180);plt.close(fig)
 bars=[test_results[n]['balanced_accuracy'] for n in feature_sets];fig,ax=plt.subplots(figsize=(9,4));ax.bar(['Duration','3 deltas','Deltas + duration','Levels + shape'],bars,color=['#98b4ce','#1b78b5','#208f73','#694b9c']);ax.axhline(1/3,ls='--',color='#888');ax.set(ylim=(0,1),ylabel='Held-out balanced accuracy',title='Predeclared feature comparison')
 for i,v in enumerate(bars):ax.text(i,v+.02,f'{v:.1%}',ha='center')
 fig.tight_layout();fig.savefig(OUT/'feature_comparison.png',dpi=180);plt.close(fig)
 html='<html lang="en"><meta charset="utf-8"><title>Three-class curve experiment</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:35px auto}img{max-width:100%}td,th{padding:9px;border-bottom:1px solid #ddd}</style><h1>Three-class curve classification — exploratory experiment</h1><p>Clip/source-group split. Test groups exclude the union of all three personal ranking models’ training and validation exposures. Participants already have personal ranking models. Human-reviewed, category-informed boundaries remain a limitation.</p><p>Selected on validation: '+winner+' / '+selected[winner]['model']+'</p><img src="feature_comparison.png"><img src="confusion_matrix.png"><p><a href="report.json">Full metrics and audit</a> · <a href="test_predictions.json">Test predictions</a> · <a href="feature_manifest.json">Features and splits</a></p></html>'
 (OUT/'index.html').write_text(html,encoding='utf8');print(json.dumps({k:report[k] for k in ['split_counts','selected_by_validation','test_results','test_by_person']},indent=2),flush=True)

if __name__=='__main__':main()
