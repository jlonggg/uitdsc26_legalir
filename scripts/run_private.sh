#!/bin/bash
# run_private.sh — questions.json -> submission.zip with the shipped system (private LB 0.9515):
#   ce  (clean_ir/pipeline.py predict: BM25 + 3 dense + RRF-free pool, teacher pass 1-3, LoRA student, LightGBM selector 69 cols)
#   ev  (evidence K=4 per pool doc)  ->  S1 = Vietnamese_Reranker MIL, 5 fold adapters bagged  |  S2 = Qwen3-Reranker MIL, adapters S2AD
#   final: selector a3 `ce+mil2` (77 cols) -> train-exact overlay (train labels only) -> package (slug rule + validate) -> zip
# Same logic as the run that produced the reference submission (sha 4bb3380c); index names / ES url are read from
# <work>/config.json, and the adapter check covers the 5 S1 adapters plus only the S2 adapters listed in S2AD.
# GPU scheduler: each job is a detached container ${PFX}<job>, launched on a card from GPUS with enough free memory and
# utilisation < BUSY %; at most MAXGPU of our containers run at once; a failed job is retried up to MAXTRY times (S1/S2 scoring jobs halve their batch each retry).
#   usage: W=<work> S1D=<s1 dir> S2D=<s2 dir> SELMODEL=<selector dir> bash run_private.sh <questions.json> <out_dir>
#   env:   NSHARD=4 MAXGPU=4 S2AD=0,1 GPUS="0 1 2 3" (allowed cards) TAG=a (container prefix dsc26repro-$TAG-) DRY=1 (plan only)
# Idempotent: rerun = resume (stage markers in <work>/steps, score files resume by qid). Stop: touch <out>/.stop
set -uo pipefail   # no -e: a failing docker call must not kill the scheduler
. "$(dirname "$0")/env.sh"
Q=${1:?questions.json}; O=${2:?out_dir}; mkdir -p "$O"; O=$(readlink -f "$O")
W=$(readlink -f ${W:?work dir}); S1D=$(readlink -f ${S1D:?}); S2D=$(readlink -f ${S2D:?}); SELMODEL=$(readlink -f ${SELMODEL:?})
NSHARD=${NSHARD:-4}; MAXGPU=${MAXGPU:-4}; S2AD=${S2AD:-0,1}; DRY=${DRY:-0}; TAG=${TAG:-a}; GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
BUSY=40; MAXTRY=4; PFX=dsc26repro-$TAG-
exec > >(tee -a "$O/run_private.log") 2>&1; log "start Q=$Q O=$O W=$W NSHARD=$NSHARD MAXGPU=$MAXGPU S2AD=$S2AD SELMODEL=$SELMODEL"
ENV="$BASE_ENV $(cfg_env $W)"; ESURL=$(python3 -c "import json;print(json.load(open('$W/config.json'))['es'])")
MNT="-v $APP:/app:ro -v $W:/work:ro -v $MODELS:/models:ro -v $S1D:/s1:ro -v $S2D:/s2:ro -v $O:/out"
S1AD=/s1/mil/outer0/adapter,/s1/mil/outer1/adapter,/s1/mil/outer2/adapter,/s1/mil/outer3/adapter,/s1/mil/outer4/adapter

# ---------- preflight (host, seconds) ----------
python3 - "$Q" "$O/questions.json" <<'PY' || exit 1   # accept dict {qid:{question}} or list [{question_id,question}]; drop any answer field
import json,sys; src,dst=sys.argv[1:3]; d=json.load(open(src,encoding='utf-8-sig'))
if isinstance(d,list): d={str(r['question_id']):r for r in d}
out={str(k):{'question':v['question']} for k,v in d.items()}
assert out and all(isinstance(v['question'],str) and v['question'].strip() for v in out.values()), 'bad question rows'
json.dump(out,open(dst,'w'),ensure_ascii=False,indent=1); print('questions:',len(out))
PY
N=$(python3 -c "import json;print(len(json.load(open('$O/questions.json'))))")
PYTHONPATH=$CLEAN python3 - <<PY || exit 1   # clean_ir source must equal the work's config (pipeline refuses drift); ES index complete
import json,urllib.request; from pipeline import inventory,sha,questions,digest
from pathlib import Path
cfg=json.load(open('$W/config.json')); src=inventory(Path('$CLEAN/src'),{'.py'}); src['pipeline.py']=sha('$CLEAN/pipeline.py')
assert src==cfg['source'], 'clean_ir source drifted from work config: '+str([k for k in set(src)|set(cfg['source']) if src.get(k)!=cfg['source'].get(k)])
assert cfg['contexts']=='/contexts' and cfg['models']=='/models', cfg
n=json.load(urllib.request.urlopen('$ESURL/clean_ir_'+cfg['owner']+'_chunks/_count'))['count']
assert n==json.load(open('$W/corpus/statistics.json'))['chunks'], ('ES index incomplete',n)
sel=json.load(open('$SELMODEL/selector.json')); assert sel['scorer']=='ce+mil2', sel
open('$O/predict_hash','w').write(digest(questions('$O/questions.json'))[:20]); print('preflight OK; selector',sel['scorer'],sel['features'],'cols fit on',sel['train_queries'],'q')
PY
for a in 0 1 2 3 4; do [ -f $S1D/mil/outer$a/adapter/adapter_config.json ] || { echo "missing S1 adapter outer$a"; exit 1; }; done
for a in ${S2AD//,/ }; do [ -f $S2D/mil/outer$a/adapter/adapter_config.json ] || { echo "missing S2 adapter outer$a"; exit 1; }; done
docker image inspect $IMG >/dev/null 2>&1 || { echo "image $IMG missing (scripts/setup.sh)"; exit 1; }
H=$(cat $O/predict_hash); P=$W/predict_$H; NA=$(echo $S2AD | tr ',' ' ' | wc -w)
python3 -c "n=$N/1000; ce=5.9*n; s1=5.2*n; s2=3.0*n*$NA; par=max(1,$MAXGPU-1); print(f'estimate (2080Ti): ce {ce:.1f} h serial | S1 {s1:.1f} + S2 {s2:.1f} GPU-h on {par} cards | wall ~{max(ce,(s1+s2)/par)+0.7:.1f} h')"
echo "predict dir: $P"

# ---------- jobs ----------
JOBS="ce ev"; for i in $(seq 0 $((NSHARD-1))); do JOBS="$JOBS s1_$i"; done; for a in ${S2AD//,/ }; do for i in $(seq 0 $((NSHARD-1))); do JOBS="$JOBS s2_${a}_$i"; done; done
mygpus() { docker ps --format '{{.Names}}' | grep -c "^$PFX" || true; }
freegpu() { used=$(docker ps -q | xargs -r docker inspect -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}},{{end}}{{end}}' | tr -d '\n')
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v n=$1 -v b=$BUSY -v ex=",$used" -v ok=" $GPUS " \
    '$3-$2>=n && $4<b && index(ex,","$1",")==0 && index(ok," "$1" ")>0 {print $1" "($3-$2)}' | sort -k2 -nr | head -1 | cut -d' ' -f1; }
isdone() { case $1 in ce) [ -f $W/steps/predict_${H}_package.json ];; ev) [ -f $O/evidence.done ];; *) [ -f $O/$1.done ];; esac; }
dep() { case $1 in ce) echo /dev/null;; ev) echo $W/steps/predict_${H}_extra2.json;; s1_*) echo $O/ev_${1#s1_}.jsonl;; s2_*) echo $O/ev_${1##*_}.jsonl;; esac; }
need() { case $1 in ce) echo 6000;; ev) echo 4000;; s1_*) echo 4500;; *) echo 5500;; esac; }
batch() { n=$(cat $O/$1.attempts 2>/dev/null || echo 0); m=$2; echo $(( m >> n > 2 ? m >> n : 2 )); }
mnt() { case $1 in ce) echo "-v $CLEAN:/app:ro -v $W:/work -v $CTX:/contexts:ro -v $MODELS:/models:ro -v $O:/out";; *) echo "$MNT -e RB_OUT=/out";; esac; }
cmd() { j=$1; case $j in
  ce) echo "python3 /app/pipeline.py --work /work --threads 3 --batch 8 predict --questions /out/questions.json";;   # batch fixed: step contracts hash the cmd
  ev) echo "rm -f /out/evidence.jsonl; python3 /app/src/evidence.py /work/predict_$H/questions.json /work/predict_$H/pool_public.jsonl /work/predict_$H/extra_public.jsonl,/work/predict_$H/extra2_public.jsonl /out/evidence.jsonl --k 4";;
  s1_*) i=${j#s1_}; echo "python3 /app/src/score_mil.py $XLMR /work/predict_$H/questions.json /out/ev_$i.jsonl /out/s1_part$i.jsonl --adapter $S1AD --batch $(batch $j 16)";;
  s2_*) a=${j#s2_}; i=${a#*_}; a=${a%_*}; echo "python3 /app/src/score_mil.py $QWEN /work/predict_$H/questions.json /out/ev_$i.jsonl /out/s2_a${a}_part$i.jsonl --adapter /s2/mil/outer$a/adapter --batch $(batch $j 16)";; esac; }
if [ "$DRY" = 1 ]; then for j in $JOBS; do echo "PLAN $j need=$(need $j)MiB dep=$(dep $j) :: docker run --gpus device=G $(mnt $j) --entrypoint bash $IMG -c \"$(cmd $j)\""; done; echo "DRY done"; exit 0; fi

agg() { python3 - "$@" <<'PY'   # agg <out> <expect_n> <files...>: jsonl by qid, first occurrence wins, half-written line skipped
import json,sys,os; out,n=sys.argv[1],int(sys.argv[2]); seen={}
for p in sys.argv[3:]:
    if not os.path.exists(p): continue
    for l in open(p):
        try: r=json.loads(l)
        except ValueError: continue
        seen.setdefault(str(r['qid']), l if l.endswith('\n') else l+'\n')
assert len(seen)==n, f'{out}: {len(seen)} rows != {n}'
open(out+'.tmp','w').writelines(seen.values()); os.replace(out+'.tmp',out)
PY
}
meanlogits() { python3 - "$@" <<'PY'   # per (qid, doc, chunk) mean over adapter files == score_mil multi-adapter bagging (1e-4)
import json,sys,os; out=sys.argv[1]; rows=[{str(json.loads(l)['qid']):json.loads(l) for l in open(p)} for p in sys.argv[2:]]
qids=list(rows[0]); assert all(set(r)==set(qids) for r in rows), 'qid sets differ'
with open(out+'.tmp','w') as o:
    for q in qids:
        docs=[r[q]['docs'] for r in rows]; key=lambda x:[(d['doc_id'],d['chunks']) for d in x]; assert all(key(x)==key(docs[0]) for x in docs), q
        o.write(json.dumps({'qid':q,'docs':[{**d,'logits':[round(sum(v)/len(v),4) for v in zip(*(x[j]['logits'] for x in docs))]} for j,d in enumerate(docs[0])]})+'\n')
os.replace(out+'.tmp',out)
PY
}
housekeep() {   # evidence done -> shards; all shards scored -> mil files
  if [ -f $O/ev.done ] && [ ! -f $O/evidence.done ]; then
    [ $(wc -l < $O/evidence.jsonl) -eq $N ] || { log "evidence rows != $N"; rm -f $O/ev.done; return; }
    python3 -c "
rows=open('$O/evidence.jsonl').readlines()
for i in range($NSHARD): open(f'$O/ev_{i}.jsonl','w').writelines(rows[i::$NSHARD])" && touch $O/evidence.done && log "evidence $N rows -> $NSHARD shards"; fi
  if [ ! -f $O/mil_public.jsonl ]; then ok=1; for i in $(seq 0 $((NSHARD-1))); do [ -f $O/s1_$i.done ] || ok=0; done
    [ $ok = 1 ] && agg $O/mil_public.jsonl $N $O/s1_part*.jsonl && log "mil_public.jsonl: $N rows (S1, 5 adapters)"; fi
  if [ ! -f $O/mil2_public.jsonl ]; then files=""; for a in ${S2AD//,/ }; do ok=1; for i in $(seq 0 $((NSHARD-1))); do [ -f $O/s2_${a}_$i.done ] || ok=0; done
      [ $ok = 1 ] || return; [ -f $O/s2_a$a.jsonl ] || agg $O/s2_a$a.jsonl $N $O/s2_a${a}_part*.jsonl || return; files="$files $O/s2_a$a.jsonl"; done
    meanlogits $O/mil2_public.jsonl $files && log "mil2_public.jsonl: $N rows (S2, mean of $NA adapters)"; fi; }

while true; do
  [ -f $O/.stop ] && { log "stop file, exit (containers keep running)"; exit 0; }
  housekeep; alldone=1; failed=0
  for j in $JOBS; do isdone $j && continue; alldone=0; [ -f $O/$j.FAILED ] && { failed=1; continue; }
    n=$PFX$j; st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null || echo none)
    [ "$st" = running ] && continue
    [ "$st" = created ] && { docker rm $n >/dev/null; st=none; }
    if [ "$st" = exited ]; then code=$(docker inspect -f '{{.State.ExitCode}}' $n); docker logs $n > $O/${j}_$(date +%m%d_%H%M%S)_exit$code.log 2>&1; docker rm $n >/dev/null
      if [ "$code" = 0 ]; then case $j in ce) isdone ce || code=99;; ev) touch $O/ev.done;; *) touch $O/$j.done;; esac; fi
      if [ "$code" = 0 ]; then log "$j done"; continue; fi
      t=$(( $(cat $O/$j.attempts 2>/dev/null || echo 0) + 1 )); echo $t > $O/$j.attempts; log "$j exited $code (attempt $t; resume by step/qid, batch halves)"
      [ $t -ge $MAXTRY ] && { touch $O/$j.FAILED; log "$j FAILED after $t attempts"; failed=1; continue; }
    fi
    [ -e "$(dep $j)" ] || continue
    [ $(mygpus) -ge $MAXGPU ] && continue
    g=$(freegpu $(need $j)); [ -z "$g" ] && continue
    log "launch $j on GPU $g: $(cmd $j)"
    docker run -d --name $n --gpus "\"device=$g\"" --network host --shm-size 2g $ENV $(mnt $j) --entrypoint bash $IMG -c "$(cmd $j)" >/dev/null; sleep 45
  done
  [ $alldone = 1 ] && break
  [ $failed = 1 ] && [ $(mygpus) -eq 0 ] && { log "FAILED jobs and nothing running: exit 1"; exit 1; }
  sleep 120
done
housekeep
for f in $O/mil_public.jsonl $O/mil2_public.jsonl $P/prediction.json; do [ -f $f ] || { log "missing $f"; exit 1; }; done
O=$O P=$P W=$W S1D=$S1D S2D=$S2D SELMODEL=$SELMODEL TAG=$TAG bash $ROOT/scripts/final_stage.sh || exit 1
touch $O/FINISHED
