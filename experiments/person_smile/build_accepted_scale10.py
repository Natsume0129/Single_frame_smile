"""Reuse the existing all-pairs voting scale builder with train-only candidate images."""
import csv,json,subprocess,sys,html
from pathlib import Path
from pipeline import ROOT,digest,save_json
from PIL import Image,ImageDraw,ImageFont

RUN=ROOT/'output/monotonic_accepted_training_20261101'
OUT=RUN/'scale10'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def render_scale(dest,person,label,cols=10):
 tile=176;gap=10;margin=18;header=48;cell=214;rows=(10+cols-1)//cols
 canvas=Image.new('RGB',(margin*2+cols*tile+(cols-1)*gap,header+rows*cell+margin),'white');draw=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 draw.text((margin,12),f'{person} | {label.capitalize()} smile | 0 weakest -> 9 strongest',font=font,fill='#17324b')
 for i in range(10):
  x=margin+(i%cols)*(tile+gap);y=header+(i//cols)*cell
  with Image.open(dest/f'level_{i:02d}.png') as im:canvas.paste(im.convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS),(x,y))
  draw.text((x+52,y+182),f'Level {i}',font=small,fill='#17324b')
 canvas.save(dest/('scale10_montage.png' if cols==10 else 'scale10_montage_grid.png'))
 return canvas
def main():
 summary=read(RUN/'split_summary.json');OUT.mkdir(exist_ok=True);results=[]
 for key in summary:
  person,label=key.split('/');assert person!='REN'
  latest=read(RUN/'models'/person/label/'latest.json');folder=Path(latest['directory']);weights=Path(latest['model']);assert (folder/'complete.json').exists()
  rows=read(folder/'accepted_clips.json');lookup={r['id']+'_'+str(f['source_frame'])+'.png':(r,f,i) for r in rows for i,f in enumerate(r['sample_frames'])}
  names=sorted({n for line in (folder/'train.dat').read_text().splitlines() if line for n in line.split(',')[:2]})
  items=[]
  for name in names:
   r,f,i=lookup[name];assert r['split']=='train'
   items.append(dict(image_file=name,sequence_id=r['sequence_id'],provisional_level=i,source_frame_index=f['source_frame'],source_frame_name=Path(f['path']).name,source_frame_path=f['path'],source_video=r['source_video'],person=person,label=label,split='train'))
  assert len(items)>=10
  dest=OUT/person/label;dest.mkdir(parents=True,exist_ok=True);manifest=dest/'scale_candidates.csv'
  with manifest.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
  fingerprint=dict(weights=digest(weights),candidates=digest(manifest),script=digest(ROOT/'annotation/build_true_smile_scale_from_model.py'))
  done=dest/'complete.json'
  if done.exists():assert read(done)['fingerprint']==fingerprint
  else:
   cfg=read(folder/'config.json');cmd=[sys.executable,'-u',str(ROOT/'annotation/build_true_smile_scale_from_model.py'),'--smilecomp_dir',cfg['smilecomp_dir'],'--ranking_items',str(manifest),'--images_dir',str(folder/'images'),'--weights',str(weights),'--manual_pairs',str(folder/'train.dat'),'--output_dir',str(dest),'--levels','10','--batch_size','32']
   with (dest/'build.log').open('w',encoding='utf8') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   with (dest/'scale10_items.csv').open(encoding='utf-8-sig') as f:scale=list(csv.DictReader(f))
   assert len(scale)==10 and [int(r['scale_level']) for r in scale]==list(range(10))
   assert all((dest/r['reference_image']).is_file() and r['source_image_file'] in names for r in scale)
   save_json(done,dict(fingerprint=fingerprint,levels=10,train_candidates=len(items)))
  render_scale(dest,person,label);render_scale(dest,person,label,5)
  results.append(dict(person=person,label=label,directory=str(dest),model=str(weights),train_candidates=len(items)))
  print(key,'complete',len(items),'training candidates',flush=True)
 save_json(OUT/'manifest.json',results)
 cards=[]
 for r in results:
  rel=r['person']+'/'+r['label'];cards.append(f'<article data-person="{r["person"]}"><h2>{r["person"]} / {r["label"].capitalize()}</h2><p>{r["train_candidates"]} training images · <a href="{rel}/scale10_items.csv">Source mapping</a> · <a href="{rel}/scale10_montage.png">Download strip</a></p><div class="strip">'+''.join(f'<figure><a href="{rel}/level_{i:02d}.png" target="_blank"><img loading="lazy" src="{rel}/level_{i:02d}.png"></a><figcaption>Level {i}</figcaption></figure>' for i in range(10))+'</div></article>')
 page='<!doctype html><html lang="en"><meta charset="utf-8"><title>10-Level Smile Scales</title><style>body{font:16px/1.5 system-ui;background:#eef3f8;color:#183047;margin:25px}article{background:white;padding:20px;margin:22px 0;border-radius:12px}.strip{display:grid;grid-template-columns:repeat(10,minmax(90px,1fr));gap:8px;overflow-x:auto}figure{margin:0;text-align:center}img{width:100%;max-width:224px}select{padding:8px}a{color:#16689c}</style><h1>10-Level Smile Scales</h1><p>15 models · REN excluded · Level 0 = weakest, level 9 = strongest within the selected training-image pool. Candidate images come only from the new training split. Pairwise voting orders the pool; 10 evenly spaced rank positions supply the reference images. Levels are ordinal, not equal physical-intensity intervals or cross-model calibrated scores.</p><select id="person"><option value="">All people</option>'+''.join(f'<option>{p}</option>' for p in sorted({r['person'] for r in results}))+'</select>'+''.join(cards)+'<script>document.getElementById("person").onchange=function(){document.querySelectorAll("article").forEach(a=>a.hidden=!!this.value&&a.dataset.person!==this.value)};</script></html>'
 (OUT/'index.html').write_text(page,encoding='utf8');print('Gallery:',OUT/'index.html',flush=True)

if __name__=='__main__':main()
