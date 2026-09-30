# Sourced by every script. All paths are relative to the repo root; override any of them from the environment.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMG=${IMG:-dsc26-clean-ir:20260908-v3}            # built from src/clean_ir/Dockerfile (scripts/setup.sh)
MODELS=${MODELS:-$ROOT/models}                     # 5 HF base models, see README §3
CTX=${CTX:-$ROOT/data/selected-contexts}           # 8,532 context_*.json (real dir, not a symlink)
DATA=${DATA:-$ROOT/data}
CLEAN=$ROOT/src/clean_ir                           # stage `ce` code (pipeline.py; its source hash is locked by <work>/config.json)
APP=$ROOT/src/clean_ir_s1                          # MIL scorers + final stage: evidence / train_mil / score_mil / selector / overlay / package
ART=$ROOT/artifacts                                # only artifacts/reference/private_V2_4bb3380c_submission.zip is shipped
RUNS=${RUNS:-$ROOT/runs}                           # all outputs (run_from_scratch.sh writes to $RUNS/scratch)
QWEN=/models/Qwen3-Reranker-0.6B; XLMR=/models/Vietnamese_Reranker   # container paths
# container env shared by every docker run (index names + ES url are derived per work dir by cfg_env)
BASE_ENV="-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python -e COVFEAT=1 -e DETRANK=1 -e CLEAN_THREADS=8 -e OMP_NUM_THREADS=3 -e CLEAN_CORPUS=/work/corpus -e CLEAN_MODELS=/models -e CLEAN_TRAIN=/work/train/questions.json"

# cfg_env <work>: -e CLEAN_ES/CLEAN_INDEX/CLEAN_WSEG from the work's config.json (owner hash + es url written by `prepare`)
cfg_env() { python3 - "$1/config.json" <<'PY'
import json,sys; c=json.load(open(sys.argv[1])); o=c['owner']
print(f"-e CLEAN_ES={c['es']} -e CLEAN_INDEX=clean_ir_{o}_chunks -e CLEAN_WSEG=clean_ir_{o}_wseg")
PY
}
log() { echo "$(date '+%F %T') $*"; }
