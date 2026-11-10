"""Classify one reviewed interval from its three-by-six regional response JSON."""
import argparse,json
from pathlib import Path
import numpy as np
import joblib,torch
from tensor_sequence_experiment_20261109 import OUT,LABELS,TemporalNet,fixed_grid,arrays,probabilities,read

def sequence(path):
 data=read(path);pts=data['points'];columns=read(OUT/'protocol.json')['columns'];t=np.array([p['seconds'] for p in pts]);x=np.array([[p[c] for c in columns] for p in pts],float)
 if len(t)<2 or not np.all(np.diff(t)>0) or not np.isfinite(x).all() or abs(t[0])>1e-6 or abs(t[-1]-data['interval_seconds'])>1e-6:raise ValueError('Expected a full onset-to-peak interval on observed frames, in source time minus onset.')
 return dict(times=t,x=x,duration=data['interval_seconds'],breaks=np.r_[True,np.diff(t)>.15])

def classify(path,arm='primary',family='winner',device='cpu'):
 s=read(OUT/'selection_before_test.json');arm=s['primary_arm'] if arm=='primary' else arm
 choices=[c for c in s['all_candidates'][arm] if family=='winner' or c['family']==family];chosen=max(choices,key=lambda c:(c['val']['accuracy'],c['val']['macro_f1']))
 r=sequence(path);r=fixed_grid(r)[0] if arm=='fixed' else r
 if chosen['family']=='kernel':pred=str(joblib.load(chosen['paths'][0]).predict([r])[0])
 else:
  probs=[]
  for ck in chosen['paths']:
   d=torch.load(ck,map_location=device,weights_only=False);model=TemporalNet(d['arch'],d['context']).to(device);model.load_state_dict(d['state_dict']);probs.append(probabilities(model,arrays([r],d['norm'],device)))
  pred=LABELS[int(np.mean(probs,axis=0)[0].argmax())]
 return dict(arm=arm,model=chosen['model'],prediction=pred)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('curve_json',type=Path);p.add_argument('--arm',choices=['primary','fixed','native'],default='primary');p.add_argument('--family',choices=['winner','kernel','bigru','tcn'],default='winner');a=p.parse_args();print(json.dumps(classify(a.curve_json,a.arm,a.family),indent=2))
