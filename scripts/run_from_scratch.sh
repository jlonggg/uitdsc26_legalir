#!/bin/bash
# run_from_scratch.sh — from zero: raw contexts + train_IR.json + 5 base models -> private submission.
# The only file read from artifacts/ is the reference ZIP used by the final compare (step private).
#   bash run_from_scratch.sh <step> [fold]     steps, in order (each is idempotent; resume = rerun the same step):
#     ce       prepare (corpus, graph, ES index on :19212, 3 dense encodes) + train (teacher passes, 6 LoRA students,
#              LightGBM selector 69 cols) + verify                                            ≈ 9 + 44 GPU-h, 1 card, serial
#     evidence K=4 evidence rows for the 7,000 train questions                                  ≈ 0.2 GPU-h
#     s1 K     S1 MIL LoRA on Vietnamese_Reranker, outer fold K (0..4) + held-fold scoring    ≈ 10-12 GPU-h / fold
#     s2 K     S2 MIL LoRA on Qwen3-Reranker-0.6B, outer fold K (0..3; fold 4 is unused by a3) ≈ 30-35 GPU-h / fold
#     a3       lexalign + fit dir + price.py build/final:ce+mil2 (selector a3, S2 cols NaN on fold 4)   CPU ≈ 15 min
#     private  run_private.sh on data/private-official_IR.json with the work dir/adapters/selector above ≈ 70 GPU-h, 4 cards
#              then compare.py against artifacts/reference/private_V2_4bb3380c_submission.zip
#     all      every step above in order on GPU $GPU (≈ 300 GPU-h serial; run s1/s2 folds on separate cards to go faster)
#   env: GPU=0 (card for ce/evidence/s1/s2), ES_PORT=19212 (same value as scripts/setup.sh), MICRO=13 (7 if a 11 GB card OOMs; same gradients)
set -euo pipefail
. "$(dirname "$0")/env.sh"
STEP=${1:?step}; K=${2:-}; GPU=${GPU:-0}; ES_PORT=${ES_PORT:-19212}
B=$RUNS/scratch; W=$B/work; S1D=$B/s1; S2D=$B/s2; A3=$B/a3; mkdir -p $W $S1D $S2D $A3
run() { local name=$1; shift; log "docker $name"; docker run --rm --name dsc26repro-train-$name --gpus "\"device=$GPU\"" --network host --shm-size 2g "$@"; }
milenv() { echo "$BASE_ENV $(cfg_env $W) -e CLEAN_WORK=/out -e RB_OUT=/out"; }

case $STEP in
ce)
  IR="-e OMP_NUM_THREADS=3 -e OPENBLAS_NUM_THREADS=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v $CLEAN:/app:ro -v $CTX:/contexts:ro -v $MODELS:/models:ro -v $DATA:/data:ro -v $W:/work"
  run ce $IR --entrypoint bash $IMG -c "set -e
    python3 /app/pipeline.py --work /work --threads 3 --batch 8 prepare --contexts /contexts --models /models --es http://127.0.0.1:$ES_PORT
    python3 /app/pipeline.py --work /work --threads 3 --batch 8 train --train /data/train_IR.json
    python3 /app/pipeline.py --work /work --threads 3 verify";;
evidence)
  [ -f $S1D/evidence_train.jsonl.done ] || run evidence $(milenv) -v $APP:/app:ro -v $W:/work:ro -v $MODELS:/models:ro -v $S1D:/out --entrypoint bash $IMG -c \
    "python3 /app/src/evidence.py /work/train/questions.json /work/train/pool_train.jsonl /work/train/extra_train.jsonl,/work/train/extra2_train.jsonl /out/evidence_train.jsonl --k 4 && touch /out/evidence_train.jsonl.done";;
s1|s2)
  [ -n "$K" ] || { echo "fold K required"; exit 1; }; [ -f $S1D/evidence_train.jsonl.done ] || { echo "run step evidence first"; exit 1; }
  if [ $STEP = s1 ]; then D=$S1D; BASE=$XLMR; else D=$S2D; BASE=$QWEN; fi
  [ -f $D/DONE_outer$K ] || run ${STEP}_$K $(milenv) -e CLEAN_CE=$BASE -e OUTER=$K -e MICRO=${MICRO:-13} \
    -v $APP:/app:ro -v $W:/work:ro -v $MODELS:/models:ro -v $S1D:/s1:ro -v $D:/out -v $ROOT/scripts/mil_fold.sh:/mil_fold.sh:ro --entrypoint bash $IMG /mil_fold.sh;;
a3)
  for k in 0 1 2 3 4; do [ -f $S1D/DONE_outer$k ] || { echo "missing s1 fold $k"; exit 1; }; done
  for k in 0 1 2 3; do [ -f $S2D/DONE_outer$k ] || { echo "missing s2 fold $k"; exit 1; }; done
  F=$A3/fit; T=$W/train
  if [ ! -f $F/.ready ]; then rm -rf $F; mkdir -p $F
    cp $T/{ce_train.s0,ce_xtrain,ce_x2train,ce2_train,p3teach_train,dense_train,dense_viemb2_train,dense_harrier_train}.jsonl $T/questions.json $F/
    for k in 0 1 2 3 4; do mkdir -p $F/ce_ft_hard_kl0.0/outer$k $F/mil/outer$k
      cp $T/ce_ft_hard_kl0.0/outer$k/held_{scores,scores2,xscores,xscores2}.jsonl $F/ce_ft_hard_kl0.0/outer$k/; cp $S1D/mil/outer$k/held_mil.jsonl $F/mil/outer$k/; done
    for k in 0 1 2 3; do mkdir -p $F/mil2/outer$k; cp $S2D/mil/outer$k/held_mil.jsonl $F/mil2/outer$k/; done
    touch $F/.ready; fi
  PR="$BASE_ENV -e FIT=/out/fit -e OUT=/out -e RB_OUT=/out/fit -e OMP_NUM_THREADS=8 -e MIL2_FOLDS=0,1,2,3 -e NAN_FOLDS=4 -v $APP:/app:ro -v $W:/work:ro -v $ROOT/src/price.py:/pr/price.py:ro -v $S1D:/s1:ro -v $A3:/out"
  # lexalign cols (77-82) are not used by ce+mil2, but price.py build asserts the file covers every train question
  [ -f $F/lexalign_train.jsonl ] || docker run --rm $PR --entrypoint python3 $IMG /app/src/lexalign.py /work/train/questions.json /s1/evidence_train.jsonl /out/fit/lexalign_train.jsonl
  [ -f $A3/X83.npz ] || docker run --rm $PR --entrypoint python3 $IMG /pr/price.py build | tee $A3/build.log
  [ -f $A3/model_ce+mil2/selector.json ] || docker run --rm $PR --entrypoint python3 $IMG /pr/price.py final:ce+mil2 | tee $A3/final.log;;
private)
  W=$W S1D=$S1D S2D=$S2D SELMODEL=$A3/model_ce+mil2 S2AD=0,1 TAG=private bash $ROOT/scripts/run_private.sh $DATA/private-official_IR.json $B/private
  python3 $ROOT/scripts/compare.py $B/private/final/submission.zip $ART/reference/private_V2_4bb3380c_submission.zip;;
all)
  s=$0; bash $s ce; bash $s evidence; for k in 0 1 2 3 4; do bash $s s1 $k; done; for k in 0 1 2 3; do bash $s s2 $k; done; bash $s a3; bash $s private;;
*) echo "unknown step $STEP"; exit 1;;
esac
log "step $STEP $K done"
