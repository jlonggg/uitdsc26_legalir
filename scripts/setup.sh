#!/bin/bash
# setup.sh — build image, kiểm model và dữ liệu, bật Elasticsearch rỗng ở 127.0.0.1:${ES_PORT:-19212}.
set -euo pipefail
. "$(dirname "$0")/env.sh"

# 1. image (mã nguồn luôn được mount vào /app; bản copy bên trong image không bao giờ được chạy)
docker image inspect $IMG >/dev/null 2>&1 || { log "building $IMG"; docker build -t $IMG $CLEAN; }

# 2. model gốc phải trùng từng byte với bản đã tạo ra bài nộp (pipeline.py kiểm lại 4 model trong số đó)
[ -d $MODELS ] || { echo "missing $MODELS (bash scripts/download_models.sh, README §3)"; exit 1; }
(cd $MODELS && sha256sum --quiet -c $ROOT/scripts/MODELS.sha256) || { echo "model sha256 mismatch (bash scripts/download_models.sh)"; exit 1; }
log "models OK ($(wc -l < $ROOT/scripts/MODELS.sha256) files)"

# 3. dữ liệu: chỉ đọc từ data/ của repo; đủ 8.532 file context + 2 file câu hỏi
for f in train_IR.json private-official_IR.json; do [ -f $DATA/$f ] || { echo "missing $DATA/$f"; exit 1; }; done
[ $(ls $CTX | grep -c '^context_.*\.json$') = 8532 ] || { echo "$CTX must hold 8532 context_*.json"; exit 1; }

# 4. Elasticsearch (compose project dsc26irrepro, volume dsc26irrepro_es); index do `run_from_scratch.sh ce` dựng
P=dsc26irrepro; PORT=${ES_PORT:-19212}
COMPOSE_PROJECT_NAME=$P ES_PORT=$PORT docker compose -p $P -f $ROOT/scripts/compose.es.yaml up -d
for i in $(seq 120); do curl -sf http://127.0.0.1:$PORT/_cluster/health?wait_for_status=yellow >/dev/null && break; [ $i = 120 ] && { echo "ES not ready after 10 min"; exit 1; }; sleep 5; done
log "ES $P on 127.0.0.1:$PORT"; curl -s "http://127.0.0.1:$PORT/_cat/indices?v"
