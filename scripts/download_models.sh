#!/bin/bash
# download_models.sh — tải 5 model gốc (danh mục BTC) về models/ đúng revision đã dùng, rồi kiểm sha256 theo MODELS.sha256.
# Chạy huggingface-cli bên trong image của repo (tự build nếu chưa có), host không cần cài gì thêm. Cần Internet, khoảng 6 GB.
# Nếu mạng yêu cầu đăng nhập HuggingFace: HF_TOKEN=... bash scripts/download_models.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"
docker image inspect $IMG >/dev/null 2>&1 || docker build -t $IMG $CLEAN
mkdir -p $MODELS
while read -r repo rev; do
  log "$repo @ $rev"
  docker run --rm -e HF_HUB_OFFLINE=0 ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} -v $MODELS:/models --entrypoint huggingface-cli $IMG \
    download $repo --revision $rev --local-dir /models/${repo#*/} --exclude "onnx/*" </dev/null
  docker run --rm -v $MODELS:/models --entrypoint rm $IMG -rf /models/${repo#*/}/.cache   # metadata của huggingface-cli, không thuộc model
done <<'EOF'
Qwen/Qwen3-Embedding-0.6B 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
AITeamVN/Vietnamese_Embedding_v2 18b44161e041bf1d3a333ab5144b5b7b93f914d2
mainguyen9/vietlegal-harrier-0.6b 91a0e1ebe4b63b4475bbae40658b8ca9231bea74
AITeamVN/Vietnamese_Reranker f536976248403314225d7fdfdbc87f0e9516a54e
Qwen/Qwen3-Reranker-0.6B e61197ed45024b0ed8a2d74b80b4d909f1255473
EOF
(cd $MODELS && sha256sum --quiet -c $ROOT/scripts/MODELS.sha256) && log "models OK"
