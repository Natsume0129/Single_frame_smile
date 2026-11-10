"""Frozen-split exploratory extension: choose features/models on validation, test once."""
import json,csv,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/three_class_curve_classifier_20261102'
OUT=ROOT/'output/regional_ranking_extension_20261103/classifier'
REGIONS=['full','periocular','nose','mouth_perioral','upper_face','lower_face']
LABELS=['polite','bitter','true'];SEED=20261103
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def curve_features(t,y):
 span=max(float(np.ptp(y)),.25);duration=t[-1]-t[0]
 v=dict(start=float(y[0]),end=float(y[-1]),delta=float(y[-1]-y[0]),mean=float(np.trapezoid(y,t)/duration),range=float(np.ptp(y)),downward=float(-np.minimum(np.diff(y),0).sum()/span),variation=float(abs(np.diff(y)).sum()/span))
 for q in [.25,.5,.75]:v['progress_'+str(int(q*100))]=float((np.interp(t[0]+q*duration,t,y)-y[0])/span)
 return v
def metric(y,p):return dict(n=len(y),accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),confusion_matrix=confusion_matrix(y,p,labels=LABELS).tolist())

def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'report.json').exists(),'Completed experiment is frozen'
 rows=read(BASE/'feature_manifest.json');assert len(rows)==1215
 for r in rows:
  p=OUT.parent/'curves'/r['person']/r['label']/r['id']/'data.json';d=read(p);pts=d['points'];t=np.array([v['seconds'] for v in pts]);assert len(t)>=2
  features={'old__'+k:v for k,v in r['features'].items()};features['duration']=r['features']['duration']
  assert abs((t[-1]-t[0])-features['duration'])<1e-6
  for region in REGIONS:
   for label in LABELS:
    y=np.array([v[region+'__'+label+'_smooth'] for v in pts]);assert np.isfinite(y).all()
    features.update({region+'__'+label+'__'+k:v for k,v in curve_features(t,y).items()})
  r['expanded_features']=features
 sets={s:np.array([r['split']==s for r in rows]) for s in ['train','val','test']};truth=np.array([r['label'] for r in rows])
 assert [int(sets[s].sum()) for s in sets]==[879,159,177]
 for a,b in [('train','val'),('train','test'),('val','test')]:assert not {r['component'] for r in rows if r['split']==a}&{r['component'] for r in rows if r['split']==b}
 old=['old__duration']+['old__'+l+'_delta' for l in LABELS]
 regional=[reg+'__'+l+'__delta' for reg in REGIONS if reg!='full' for l in LABELS]
 families={'old_deltas_duration':old,'matched_full_deltas':['duration']+['full__'+l+'__delta' for l in LABELS]}
 families.update({reg+'_only':['duration']+[reg+'__'+l+'__delta' for l in LABELS] for reg in REGIONS if reg!='full'})
 families.update(regional_deltas=['duration']+regional,old_plus_regional_deltas=old+regional,matched_plus_regional_deltas=['duration']+[reg+'__'+l+'__delta' for reg in REGIONS for l in LABELS],all_levels=[k for k in rows[0]['expanded_features'] if k=='duration' or k.endswith(('__start','__end','__delta','__mean'))],all_shapes=list(rows[0]['expanded_features']))
 # Lock families and model-selection rule before looking at this experiment's test predictions.
 save(OUT/'experiment_plan.json',dict(feature_families=families,selection='validation accuracy first, macro-F1 as tie breaker; test excluded',frozen_split_source=str(BASE/'feature_manifest.json'),split_sha256=hashlib.sha256((BASE/'feature_manifest.json').read_bytes()).hexdigest(),test_status='Previously inspected baseline test: paired exploratory extension, not a new untouched confirmatory test',roi_mode='Fixed train-derived region-only masks',matched_control='Six fixed augmentation views for full and all five ROIs'))
 selected={};models={};trials=[]
 for family,columns in families.items():
  X=np.array([[r['expanded_features'][k] for k in columns] for r in rows]);assert np.isfinite(X).all()
  candidates=[('logistic_C'+str(c),make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=5000,random_state=SEED))) for c in [.01,.1,1,10]]
  candidates += [('forest_d'+str(depth),RandomForestClassifier(n_estimators=300,max_depth=depth,min_samples_leaf=4,class_weight='balanced_subsample',random_state=SEED,n_jobs=4)) for depth in [4,8,None]]
  candidates += [('hist_leaves'+str(leaves),HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=leaves,learning_rate=.05,min_samples_leaf=15,l2_regularization=1,early_stopping=False,random_state=SEED)) for leaves in [7,15]]
  candidates += [('rbf_C'+str(c),make_pipeline(StandardScaler(),SVC(C=c,gamma='scale',class_weight='balanced',random_state=SEED))) for c in [1,10]]
  best=None
  for name,model in candidates:
   model.fit(X[sets['train']],truth[sets['train']]);pred=model.predict(X[sets['val']]);m=metric(truth[sets['val']],pred);rank=(m['accuracy'],m['macro_f1']);trials.append(dict(family=family,model=name,validation=m))
   if best is None or rank>best[0]:best=(rank,name,model,m)
  selected[family]=dict(model=best[1],validation=best[3],feature_columns=columns);models[family]=best[2];print(family,best[1],'val',best[3]['accuracy'],flush=True)
 winner=max(selected,key=lambda k:(selected[k]['validation']['accuracy'],selected[k]['validation']['macro_f1']))
 save(OUT/'selection_before_test.json',dict(winner=winner,selected=selected,trials=trials))
 test_results={};predictions={}
 for family,meta in selected.items():
  X=np.array([[r['expanded_features'][k] for k in meta['feature_columns']] for r in rows]);pred=models[family].predict(X[sets['test']]);predictions[family]=pred;test_results[family]=metric(truth[sets['test']],pred)
 baseline=joblib.load(BASE/'classifier.joblib');Xold=np.array([[r['features'][k] for k in baseline['feature_columns']] for r in rows]);basepred=baseline['model'].predict(Xold[sets['test']]);actual=truth[sets['test']];newpred=predictions[winner]
 bcorrect=basepred==actual;ncorrect=newpred==actual;gained=int((~bcorrect&ncorrect).sum());lost=int((bcorrect&~ncorrect).sum());pvalue=binomtest(gained,gained+lost,.5).pvalue if gained+lost else 1.0
 testrows=[r for r in rows if r['split']=='test'];per={}
 cluster_ids=sorted({r['component'] for r in testrows});cluster_indices=[np.array([i for i,r in enumerate(testrows) if r['component']==g]) for g in cluster_ids];rng=np.random.default_rng(SEED);differences=[]
 for _ in range(2000):
  ix=np.concatenate([cluster_indices[i] for i in rng.integers(0,len(cluster_indices),len(cluster_indices))]);differences.append(float(ncorrect[ix].mean()-bcorrect[ix].mean()))
 cluster_ci=np.quantile(differences,[.025,.975]).tolist()
 for person in sorted({r['person'] for r in testrows}):
  m=np.array([r['person']==person for r in testrows]);per[person]=dict(baseline=metric(actual[m],basepred[m]),extended=metric(actual[m],newpred[m]))
 report=dict(winner=winner,winner_model=selected[winner]['model'],baseline=metric(actual,basepred),selected_test=test_results[winner],all_test_results=test_results,selected_validation=selected[winner]['validation'],paired_comparison=dict(corrected=gained,new_errors=lost,naive_clip_mcnemar_p=float(pvalue),source_group_bootstrap_accuracy_difference_ci95=cluster_ci,bootstrap_groups=len(cluster_ids),note='Exploratory paired comparison on a previously inspected test. Clip McNemar p ignores within-source dependence; use grouped CI descriptively, not as fresh confirmatory evidence'),by_person=per,limitations=['Fixed reviewed windows are category-informed.','Personal ranking models are already calibrated on these participants; not unseen-person end-to-end evaluation.','Full matched control and ROI models use fixed six-view augmentation, unlike old stochastic-augmentation baseline.','Appearance/recording confounds including eyeglasses and lighting may remain; ROI gains do not establish causal muscle importance.'])
 save(OUT/'report.json',report);joblib.dump(dict(model=models[winner],feature_columns=selected[winner]['feature_columns'],labels=LABELS),OUT/'classifier.joblib')
 save(OUT/'feature_manifest.json',rows)
 save(OUT/'test_predictions.json',[dict(id=r['id'],person=r['person'],label=r['label'],baseline_prediction=str(bp),prediction=str(npred)) for r,bp,npred in zip(testrows,basepred,newpred)])
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for ax,title,pred in zip(axes,['Existing baseline','Validation-selected regional extension'],[basepred,newpred]):
  cm=confusion_matrix(actual,pred,labels=LABELS);scaled=cm/cm.sum(axis=1,keepdims=True);ax.imshow(scaled,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted',ylabel='Annotated',title=title)
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{scaled[i,j]:.1%}',ha='center',va='center',color='white' if scaled[i,j]>.55 else '#17334c')
 fig.tight_layout();fig.savefig(OUT/'confusion_comparison.png',dpi=170);plt.close(fig)
 rows_html=''.join(f'<tr><td>{k}</td><td>{selected[k]["model"]}</td><td>{selected[k]["validation"]["accuracy"]:.1%}</td><td>{v["accuracy"]:.1%}</td><td>{v["macro_f1"]:.3f}</td></tr>' for k,v in test_results.items())
 (OUT/'index.html').write_text('<html lang="en"><meta charset="utf-8"><title>Regional curve classification</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>Regional curve classification — exploratory extension</h1><p>Selected using validation accuracy, then macro-F1. Same frozen 879 /159 /177 split. Test was previously inspected for the baseline.</p><p>Selected: '+winner+' / '+selected[winner]['model']+'</p><img src="confusion_comparison.png"><table><tr><th>Features</th><th>Model</th><th>Validation ACC</th><th>Test ACC</th><th>Test macro-F1</th></tr>'+rows_html+'</table><p><a href="report.json">Full report</a> · <a href="test_predictions.json">Paired test predictions</a> · <a href="experiment_plan.json">Frozen experiment plan</a></p></html>',encoding='utf8')
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
