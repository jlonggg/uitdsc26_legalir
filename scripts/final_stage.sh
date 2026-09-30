#!/bin/bash
# final_stage.sh — CPU only, ~1-2 min: selector a3 predict -> train-exact overlay -> package. Called by run_private.sh.
#   env: O (has questions.json, mil_public.jsonl, mil2_public.jsonl) P (ce predict dir <work>/predict_<hash>) W SELMODEL
#        OVERLAY=1 (0 = skip the train-exact overlay)  TAG=a (container name dsc26repro-$TAG-final)
# Output: $O/final/{submission.zip,submission.json,validation.json}; $O/predict holds the selector inputs (copies, never symlinks).
set -euo pipefail
. "$(dirname "$0")/env.sh"
O=$(readlink -f $O); P=$(readlink -f $P); W=$(readlink -f $W); SELMODEL=$(readlink -f $SELMODEL); OVERLAY=${OVERLAY:-1}; TAG=${TAG:-a}
D=$O/predict; mkdir -p $D; for f in $P/*; do [ -d $f ] && continue; cp -n $f $D/; done
cp $O/mil_public.jsonl $D/mil_public.jsonl; cp $O/mil2_public.jsonl $D/mil2_public.jsonl; rm -f $D/prediction.json $D/best_chunk_text.jsonl
PRED=/out/predict/prediction.json
OV=""; [ "$OVERLAY" = 1 ] && { OV="python3 train_exact_overlay.py /work/train/questions.json /out/predict/questions.json /out/predict/prediction.json /work/corpus/catalog.json /out/predict/prediction_overlay.json &&"; PRED=/out/predict/prediction_overlay.json; }
rm -rf $O/final
docker run --rm --name dsc26repro-$TAG-final --network host $BASE_ENV $(cfg_env $W) -e RB_OUT=/out/predict \
  -v $APP:/app:ro -v $W:/work:ro -v $SELMODEL:/model:ro -v $O:/out --entrypoint bash $IMG -c \
  "cd /app/src && python3 selector_stage.py predict /out/predict /out/predict/questions.json /model && $OV \
   python3 package.py /out/predict/questions.json $PRED /work/corpus/catalog.json /out/final" > $O/final.log 2>&1 || { log "final stage FAILED, see $O/final.log"; exit 1; }
cat $O/final.log
python3 - $O $P <<'PY'
import json,zipfile,hashlib,sys
O,P=sys.argv[1:3]; z=O+'/final/submission.zip'; p=json.loads(zipfile.ZipFile(z).read('submission.json')); v=json.load(open(O+'/final/validation.json'))
n=len(json.load(open(O+'/questions.json')))
assert v['valid'] and len(p)==n and all(1<=len(set(x['answer']))==len(x['answer'])<=5 for x in p.values()), 'validation failed'
print('sha256',hashlib.sha256(open(z,'rb').read()).hexdigest(),'| questions',len(p),'| slug changes',len(v['slug_changes']))
PY
log "FINISHED $O/final/submission.zip"
