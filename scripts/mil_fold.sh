#!/bin/bash
# mil_fold.sh — runs INSIDE the container (run_from_scratch.sh s1|s2 K). One outer fold of the MIL scorer:
# train LoRA on the 4 other folds (train labels only), then score the held fold -> held_mil.jsonl (OOF column for the selector).
# Same recipe for S1 and S2; only the base reranker differs (CLEAN_CE). env: OUTER=K, CLEAN_CE, MICRO (micro-batch, default 13).
# Resumable: training is skipped if the adapter exists; the held-fold evidence subset is picked with mil_common.fold (question hash).
set -euo pipefail
T=/work/train; S=/app/src; K=$OUTER; D=/out/mil/outer$K; EV=/s1/evidence_train.jsonl; mkdir -p $D
RANK=$T/ce_train.s0.jsonl,$T/ce_xtrain.jsonl,$T/ce_x2train.jsonl
[ -f $D/adapter/adapter_config.json ] || python3 $S/train_mil.py --outer $K --evidence $EV --ranking $RANK --out $D --micro ${MICRO:-13}
OUTER=$K python3 - <<'PY'
import json,sys,os; sys.path.insert(0,'/app/src'); from mil_common import fold
k=int(os.environ['OUTER']); q=json.load(open('/work/train/questions.json'))
with open(f'/out/mil/outer{k}/held_evidence.jsonl','w') as o:
    for line in open('/s1/evidence_train.jsonl'):
        if fold(q[json.loads(line)['qid']]['question'])==k: o.write(line)
PY
python3 $S/score_mil.py $CLEAN_CE $T/questions.json $D/held_evidence.jsonl $D/held_mil.jsonl --adapter $D/adapter --batch 16
touch /out/DONE_outer$K
