"""Reuse the category-review page for JIE, including a non-destructive discard decision."""
import hashlib,json,os,re
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'new_dataset'
INPUT=SOURCE/'JIE/jie'
OUT=ROOT/'output/JIE_label_review_20261031'


def main():
    records=[]
    for p in sorted(INPUT.glob('*.mp4')):
        relative=p.relative_to(SOURCE).as_posix();stat=p.stat()
        records.append(dict(id=hashlib.sha256(relative.encode()).hexdigest()[:24],person='JIE',
            current_label='unclassified',filename=p.name,source_video=str(p.resolve()),
            source_relative_path=relative,size_bytes=stat.st_size,mtime_ns=str(stat.st_mtime_ns),
            video_url=quote(os.path.relpath(p,OUT).replace('\\','/'),safe='/')))
    assert len(records)==150,'Unexpected source count; inspect before freezing inventory'
    dataset=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    payload=dict(schema_version=1,task='jie_polite_bitter_classification',dataset_id=dataset,
                 source_root=str(SOURCE),allowed_labels=['polite','bitter','discard'],records=records)
    OUT.mkdir(exist_ok=True)
    if (OUT/'inventory.json').exists():
        assert json.loads((OUT/'inventory.json').read_text(encoding='utf8'))['dataset_id']==dataset,'Inventory changed; preserve existing review and use a new directory'
    (OUT/'inventory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    page=Path(__file__).with_name('all_label_review.html').read_text(encoding='utf8')
    page=page.replace('Smile Category Review','JIE — Polite / Bitter Annotation')
    page=page.replace('Review every original video. Choose Polite, Bitter, or Discard (exclude from the dataset).','Classify 150 unclassified JIE videos. Choose Polite, Bitter, or Discard. Discard records exclusion without deleting the original video. Only videos from JIE/jie are included.')
    page=re.sub(r'(<select id="person">).*?(</select>)',r'\1<option value="">JIE — all videos</option><option>JIE</option>\2',page,count=1)
    page=page.replace('<option value="changed">Label changed</option>','<option value="changed">Classified</option>')
    page=page.replace(' · ${changedCount} label changes · ${discardCount} discarded',' · ${changedCount} classified · ${discardCount} discarded')
    page=page.replace("a.download='all_smile_labels_'","a.download='JIE_smile_labels_'")
    page=page.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('Prepared',len(records),'videos:',OUT/'index.html')


if __name__=='__main__':main()
