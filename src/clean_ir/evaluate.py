"""Create a label-isolated development split and score it without official feedback."""
import argparse
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path
from pipeline import questions,fold,write,sha


def normalized(s):
    return ' '.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFC',s).lower()))


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('split');p.add_argument('--train',type=Path,required=True)
    p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('score');p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--submission',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.stage=='split':
        q=questions(a.train,training=True)
        tr={k:v for k,v in q.items() if fold(v['question'])!=a.fold}
        held={k:v for k,v in q.items() if fold(v['question'])==a.fold}
        assert tr and held
        if a.out.exists() and any(a.out.iterdir()):raise ValueError('Split output must be empty')
        write(a.out/'train.json',tr);write(a.out/'held_gold.json',held)
        write(a.out/'held_questions.json',{k:{'question':v['question']} for k,v in held.items()})
        write(a.out/'manifest.json',{'source_sha':sha(a.train),'fold':a.fold,'train':len(tr),'held':len(held)})
        return
    gold=questions(a.gold,training=True)
    tr=questions(a.work/'train/questions.json',training=True)
    train_text={normalized(r['question']) for r in tr.values()}
    assert not train_text & {normalized(r['question']) for r in gold.values()}, 'Held questions leaked into training'
    with zipfile.ZipFile(a.submission) as z:
        assert z.namelist()==['submission.json'];pred=json.loads(z.read('submission.json'))
    import sys
    sys.path.insert(0,str(Path(__file__).parent/'src'))
    from package import validate
    validate(gold,pred,json.loads((a.work/'corpus/catalog.json').read_text()))
    rec={q:len(set(pred[q]['answer'])&set(r['answer']))/len(r['answer']) for q,r in gold.items()}
    pre={q:len(set(pred[q]['answer'])&set(r['answer']))/len(pred[q]['answer']) for q,r in gold.items()}
    write(a.out,{'recall':sum(rec.values())/len(rec),'precision':sum(pre.values())/len(pre),'queries':len(rec),
        'per_query_recall':rec,'gold_sha':sha(a.gold),'submission_sha':sha(a.submission),
        'train_manifest_sha':sha(a.work/'model/manifest.json'),
        'scope':'Upstream and downstream excluded every normalized held question; development fold, no official tuning.'})


if __name__=='__main__':main()
