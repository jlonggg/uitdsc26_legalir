#!/usr/bin/env python3
"""Train-exact overlay: a test question whose normalized text equals a train question gets that train question's gold
documents put first (union over all matching train rows, then the model's answer, cut to 5). Label-blind w.r.t. the test
set: only issued train labels are read. Same normalizer as the fold hash (mil_common.normalize).
usage: train_exact_overlay.py <train_questions.json> <test_questions.json> <prediction.json> <catalog.json> <out_prediction.json>"""
import json, sys
from pathlib import Path
from mil_common import normalize


def overlay(train, test, prediction, catalog):
    gold = {}
    for r in train.values():
        g = gold.setdefault(normalize(r['question']), [])
        g += [d for d in r['answer'] if d not in g]
    out, changes = {}, []
    for q, row in prediction.items():
        a = list(row['answer']); g = gold.get(normalize(test[q]['question']), [])
        assert set(g) <= set(catalog), q
        new = list(dict.fromkeys(g + a))[:5]
        if new != a: changes.append({'qid': q, 'train_gold': g, 'before': a, 'after': new})
        out[q] = {'answer': new}
    return out, changes


if __name__ == '__main__':
    if len(sys.argv) == 1:  # self-check
        tr = {'t1': {'question': 'Thời hạn cấp đăng ký xe máy là bao lâu?', 'answer': ['g1']}, 't2': {'question': 'thời hạn  cấp đăng ký xe máy là bao lâu', 'answer': ['g2']}}
        te = {'p1': {'question': 'Thời hạn cấp đăng ký xe máy là bao lâu?'}, 'p2': {'question': 'Câu khác hẳn?'}}
        pr = {'p1': {'answer': ['a', 'g1', 'c', 'd', 'e']}, 'p2': {'answer': ['x']}}
        o, ch = overlay(tr, te, pr, {'g1', 'g2', 'a', 'c', 'd', 'e', 'x'})
        assert o['p1']['answer'] == ['g1', 'g2', 'a', 'c', 'd'] and o['p2']['answer'] == ['x'] and len(ch) == 1, (o, ch)
        print('self-check OK'); sys.exit(0)
    train, test, pred, cat, out = sys.argv[1:6]
    o, ch = overlay(*(json.loads(Path(p).read_text()) for p in (train, test, pred, cat)))
    Path(out).write_text(json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1))
    Path(out).with_suffix('.overlay.json').write_text(json.dumps(ch, ensure_ascii=False, indent=1))
    print(f'train-exact overlay: {len(ch)} of {len(o)} questions changed', flush=True)
