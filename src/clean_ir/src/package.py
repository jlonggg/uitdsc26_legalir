"""Deterministic corpus-only empty-document slug channel and submission validation."""
import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path


def validate(questions, prediction, catalog):
    if set(prediction) != set(questions):
        raise ValueError('Question IDs do not match')
    for q, row in prediction.items():
        a=row.get('answer')
        if not isinstance(a,list) or not 1<=len(a)<=5 or not all(isinstance(d,str) for d in a):
            raise ValueError('Invalid answer at '+q)
        if len(set(a))!=len(a) or not set(a)<=set(catalog):
            raise ValueError('Duplicate or unknown document at '+q)


def slug(s):
    s=re.sub(r'\.(aspx?|html?)$','',s.rstrip('/').split('/')[-1])
    s=re.sub(r'-\d{5,}$','',s)
    s=''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
    return re.sub(r'\s+',' ',s.replace('-',' ').replace('_',' ')).strip().lower()


def package(questions,prediction,catalog,out,threshold=.45):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    # Documents with an empty passage are never retrieved by BM25/dense/CE. For a question whose
    # top-5 is full, replace the 5th answer with the best empty document when the char 3-5-gram
    # TF-IDF cosine between the unaccented question and that document's slug is >= threshold.
    ids=sorted(catalog);empty=[i for i,d in enumerate(ids) if catalog[d]['empty']]
    changes=[]
    if empty:
        vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True)
        X=normalize(vec.fit_transform([slug(catalog[d]['name'] or catalog[d]['link']) for d in ids]))
        qs=sorted(questions)
        Q=normalize(vec.transform([''.join(c for c in unicodedata.normalize('NFD',questions[q]['question']) if unicodedata.category(c)!='Mn').lower() for q in qs]))
        scores=(Q@X[empty].T).toarray()
        for i,q in enumerate(qs):
            j=int(scores[i].argmax());d=ids[empty[j]];a=prediction[q]['answer']
            if scores[i,j]>=threshold and d not in a and len(a)==5:
                changes.append({'qid':q,'inserted':d,'evicted':a[-1],'score':float(scores[i,j])});a[-1]=d
    validate(questions,prediction,catalog)
    out.mkdir(parents=True,exist_ok=True)
    data=json.dumps(prediction,ensure_ascii=False,sort_keys=True,indent=1).encode()
    (out/'submission.json').write_bytes(data)
    with zipfile.ZipFile(out/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:
        info=zipfile.ZipInfo('submission.json',date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,data)
    (out/'validation.json').write_text(json.dumps({'valid':True,'questions':len(questions),'slug_changes':changes},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('questions',type=Path);ap.add_argument('prediction',type=Path)
    ap.add_argument('catalog',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    package(json.loads(a.questions.read_text()),json.loads(a.prediction.read_text()),json.loads(a.catalog.read_text()),a.out)
