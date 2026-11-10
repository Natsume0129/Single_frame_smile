"""Freeze a current-folder inventory and build a non-destructive label review page."""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'new_dataset'
OUT=ROOT/'output/all_smile_label_review_20261022'


def main():
    records=[]
    for person in ['JIANG','XU','YAMAMOTO']:
        folders={'polite':'polite','bitter':'bitter'} if person=='JIANG' else {'polite':'polite smile','bitter':'bitter smile'}
        for label,folder in folders.items():
            for path in sorted((SOURCE/person/folder).glob('*.mp4')):
                rel=path.relative_to(SOURCE).as_posix();stat=path.stat()
                records.append(dict(id=hashlib.sha256(rel.encode()).hexdigest()[:24],person=person,
                    current_label=label,filename=path.name,source_video=str(path.resolve()),
                    source_relative_path=rel,size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    video_url=quote(os.path.relpath(path,OUT).replace('\\','/'),safe='/')))
    if not records:raise ValueError('No source videos')
    dataset=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    payload=dict(schema_version=1,task='all_smile_category_review',dataset_id=dataset,
                 source_root=str(SOURCE),records=records)
    OUT.mkdir(parents=True,exist_ok=True)
    existing=OUT/'inventory.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf8'))['dataset_id']!=dataset:
        raise ValueError('Inventory changed; choose a new output folder to preserve ongoing review')
    existing.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    template=Path(__file__).with_name('all_label_review.html').read_text(encoding='utf8')
    (OUT/'index.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8')
    print(OUT/'index.html')
    for person in ['JIANG','XU','YAMAMOTO']:
        print(person,{label:sum(r['person']==person and r['current_label']==label for r in records) for label in ['polite','bitter']})
    print('Total:',len(records))


if __name__=='__main__':main()
