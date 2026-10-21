# UIT-DSC 2026 — Legal IR (Task 1) — UIT - Kitchen

Mã nguồn để tái lập bản nộp **Private Phase** của đội **UIT - Kitchen** từ dữ liệu BTC và 5 model gốc trên HuggingFace:

| Leaderboard private | Recall | Precision | Ngày nộp |
|---|---:|---:|---|
| hạng 8 | **0.9515** | 0.2042 | 2026-09-21 07:52 |

Bản nộp gốc là `submission.zip` có sha256 `4bb3380cbb48b72276d9c8579e2d261d27c322f47846a3b3ed39b33f5708af26`. Repo kèm bản sao ở `artifacts/reference/private_V2_4bb3380c_submission.zip` để so sánh ở bước cuối.

- **Lần chấm đầu (2.080 câu):** recall 0.951643, precision 0.204135.
- **Chấm lại:** BTC loại 103 câu private trùng warmup, còn 1.977 câu. Recall của cùng ZIP là 0.9515, đúng như trên leaderboard.

Hệ thống **không đọc nhãn public, warmup hay private ở bất kỳ bước nào**. Nhãn duy nhất được đọc là `train_IR.json`.

---

## 1. Hệ thống

```
câu hỏi ─► [ce] clean_ir/pipeline.py predict ─────────────────────────────────────────────┐
           BM25 top-60 doc (ES, underthesea word-seg) + dense exact search 3 kênh           │
           (Qwen3-Embedding-0.6B theo Điều, Vietnamese_Embedding_v2 & vietlegal-harrier theo chunk)
           → pool ≈85 doc/câu (+≤30 Qwen3, +≤40 viemb2/harrier) → teacher CE pass 1/2/3     │
           → student CE (LoRA trên Vietnamese_Reranker) → 69 đặc trưng/doc                  │
           │                                                                                 │
           ├─► [ev] evidence K=4 chunk/doc (2 BM25 + 2 cosine viemb2 trong doc)              │
           │     ├─► [S1] Vietnamese_Reranker + LoRA MIL, bagging 5 adapter (5 fold)  → 4 cột │
           │     └─► [S2] Qwen3-Reranker-0.6B + LoRA MIL, trung bình adapter fold 0,1 → 4 cột │
           ▼                                                                                 ▼
        [final] selector a3: LightGBM LambdaRank, 77 cột, 6 seed × 1200 cây, z-score cộng dồn
                → overlay train-exact (câu trùng khít văn bản với câu train ⇒ đưa gold train lên đầu)
                → package: kênh slug cho văn bản rỗng (TF-IDF char 3–5, ngưỡng 0.45) → validate → top-5
```

### Huấn luyện (chỉ trên `train_IR.json`, 7.000 câu)

Fold được gán bằng `sha256(câu hỏi chuẩn hoá)[:8] mod 5`. Cùng một cách chia fold dùng cho mọi tầng, nên mọi cột đưa vào selector đều là out-of-fold.

| Tầng | Recipe |
|---|---|
| Student CE | LoRA r16/α32, lr 1e-4, 1 epoch, max_len 768, hard negative. Có 5 fold để sinh cột OOF, cộng 1 bản fit toàn bộ dùng lúc suy luận. |
| Selector-69 (tầng `ce`) | LightGBM 69 cột, 6 seed |
| S1 / S2 MIL | Mỗi nhóm gồm 1 gold, 6 hard negative (theo logit teacher) và 6 random. Điểm doc = LSE trên 4 chunk; loss là CE trên 13 doc. LoRA r32/α64, lr 1e-4, 2 epoch, max_len 768. |
| Selector a3 | LightGBM trên 77 cột = 69 + S1 (4) + S2 (4). Cột S2 OOF có cho fold 0–3; fold 4 để NaN (LightGBM missing). |

### Mô hình

Cả 5 mô hình đều nằm trong danh sách BTC duyệt. Tham số đếm bằng `numel()` trên model đã nạp.

| Mô hình (HF) | Vai trò | Tham số |
|---|---|---:|
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | dense theo Điều | 595,776,512 |
| [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) | dense chunk + evidence | 567,754,752 |
| [mainguyen9/vietlegal-harrier-0.6b](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b) | dense chunk | 596,049,920 |
| [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) | teacher, student (+LoRA 3.4M), S1 (+5 LoRA × 7.3M) | 567,755,777 |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | S2 (+2 LoRA × 9.2M) | 595,776,512 |
| **Tổng** (mỗi base đếm 1 lần + mọi adapter) | | **2,981,583,879 < 4B** |

Tầng `ce` dùng 4 base đầu cùng Vietnamese_Reranker và LoRA student, tổng 2,330,746,882 tham số. Word segmentation dùng [underthesea](https://github.com/undertheseanlp/underthesea) 9.5.0 (CRF). Hệ thống không gọi API nào và không tải gì lúc chạy (`HF_HUB_OFFLINE=1`).

**Cách đếm tham số:** mỗi base được tính một lần, cộng thêm mọi adapter LoRA. Các adapter của cùng một base chỉ được nạp song song để tăng tốc, nên không làm tăng số tham số phân biệt.

---

## 2. Môi trường

| | Đã dùng |
|---|---|
| GPU | 8× RTX 2080 Ti 11 GB; mỗi container 1 card, cần ≥6 GB trống |
| Driver | 580.82.09 |
| Docker image | `dsc26-clean-ir:20260908-v3`, build từ `src/clean_ir/Dockerfile` (`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`, Python 3.11.9) |
| Thư viện chính | torch 2.4.1+cu121, transformers 4.53.2, peft 0.15.2, sentence-transformers 3.4.1, lightgbm 4.6.0, scikit-learn 1.6.1, numpy 1.26.4, elasticsearch-py 7.17.12, underthesea 9.5.0, regex 2026.9.3 |
| Elasticsearch | 7.17.0 single-node, không plugin (`scripts/compose.es.yaml`) |
| Host | bash, python3 ≥3.8 (chỉ stdlib), Docker + NVIDIA Container Toolkit, docker compose v2, curl, nvidia-smi |

- Danh sách pin đầy đủ ở `src/clean_ir/requirements.txt`.
- `pip freeze` của image ở `env/pip_freeze_image.txt`.
- Mã nguồn luôn được mount vào `/app` khi chạy, nên bản copy nằm sẵn trong image không bao giờ được thực thi.

---

## 3. Chuẩn bị

```bash
git clone <repo> DSC26_IR_Repro && cd DSC26_IR_Repro

# 1. Dữ liệu BTC (không kèm trong repo) đặt trong data/ của repo (thư mục thật, không symlink):
#    data/train_IR.json  data/private-official_IR.json  data/selected-contexts/context_*.json (8.532 file)
#    Mọi script chỉ đọc dữ liệu từ data/.

# 2. 5 model gốc, đúng revision đã dùng, kiểm sha256 theo scripts/MODELS.sha256 (≈9 GB, cần Internet)
bash scripts/download_models.sh        # đã có sẵn model: MODELS=/đường/dẫn/models cho mọi lệnh

# 3. build image, kiểm model + dữ liệu, bật Elasticsearch rỗng ở 127.0.0.1:19212 (đổi bằng ES_PORT=...)
bash scripts/setup.sh
```

---

## 4. Chạy lại từ đầu

Đầu vào chỉ gồm corpus, `train_IR.json` và 5 model gốc. Mọi index, embedding, LoRA, selector đều được sinh lại; đầu ra nằm trong `runs/scratch/`. Mỗi bước bỏ qua phần đã xong, nên chạy lại cùng lệnh là resume.

```bash
GPU=0 bash scripts/run_from_scratch.sh ce            # prepare + train tầng ce   ≈ 9 + 44 GPU-h, 1 card
GPU=0 bash scripts/run_from_scratch.sh evidence      # evidence train            ≈ 0.2 GPU-h
GPU=1 bash scripts/run_from_scratch.sh s1 0          # S1 fold 0..4 (mỗi fold một card, chạy song song được)   ≈ 10–12 GPU-h/fold
GPU=2 bash scripts/run_from_scratch.sh s2 0          # S2 fold 0..3                                           ≈ 30–35 GPU-h/fold
bash scripts/run_from_scratch.sh a3                  # fit selector a3, CPU ≈ 15 phút
MAXGPU=4 bash scripts/run_from_scratch.sh private    # suy luận private ≈ 70 GPU-h, rồi so với bản đã nộp
# hoặc tuần tự mọi thứ trên 1 card:  GPU=0 bash scripts/run_from_scratch.sh all   (≈ 310 GPU-h)
```

Kết quả cuối: `runs/scratch/private/final/submission.zip`. `scripts/compare.py` in số câu có tập đáp án trùng với bản đã nộp.

**Mỗi bước làm gì:**

| Bước | Lệnh |
|---|---|
| `ce` | `pipeline.py prepare` (corpus, graph, index ES, 3 lượt encode dense) → `train --train /data/train_IR.json` → `verify` |
| `evidence` | `evidence.py` trên pool/extra/extra2 của train |
| `s1 K`, `s2 K` | `scripts/mil_fold.sh` = `train_mil.py --outer K`, rồi `score_mil.py` trên fold K bị giữ lại |
| `a3` | `lexalign.py`, dựng thư mục fit, `price.py build` (ma trận 83 cột), `price.py final:ce+mil2` với `MIL2_FOLDS=0,1,2,3 NAN_FOLDS=4` |
| `private` | `run_private.sh`: suy luận tầng ce, evidence, S1, S2 trên câu private rồi chạy `final_stage.sh` |

- Tất cả seed đều cố định: LoRA dùng `368+outer`, LightGBM dùng seed 2026/7/99/314/2718/1618, và tie-break xác định.
- Dù vậy, huấn luyện và suy luận trên GPU (fp16 autocast, cuBLAS) **không bảo đảm trùng byte**. Kết quả được đánh giá theo recall và tỷ lệ câu có tập đáp án trùng với bản đã nộp, không theo sha.
- S2 chỉ train fold 0–3 vì selector a3 không dùng fold 4 (cột S2 của fold 4 để NaN).
- Cùng cấu hình (selector fit trên 7.000 câu) đạt public recall 0.954583, precision 0.2058 trên 1.000 câu. Repo không kèm script chấm public.

---

## 5. Chi phí và thời gian

Số đo thật trên 2080 Ti.

| | GPU-h | Ghi chú |
|---|---:|---|
| prepare (3 encode) | 9.2 | cộng 1.4 giờ CPU (corpus, graph, index ES) |
| train tầng ce | 44.4 | teacher pass 1–3 trên 7.000 câu và 6 lần train student |
| S1 × 5 fold | ≈ 55 | |
| S2 × 4 fold | ≈ 130 | 8 group/phút, khoảng 25 giờ train mỗi fold |
| suy luận private 2.080 câu | ≈ 70 | ce 14 giờ tuần tự; S1 23 GPU-h, S2 24 GPU-h |

- **Tổng cộng khoảng 310 GPU-h.**
- Máy dùng chung dễ gặp OOM (`CUBLAS_STATUS_ALLOC_FAILED`). Ở bước `private`, scheduler chạy mỗi job tối đa 4 lần; job chấm S1/S2 giảm batch một nửa mỗi lần thử lại, và mọi job resume theo qid nên không mất tiến độ. Các bước train không tự thử lại: chạy lại cùng lệnh để resume.
- Container chạy bằng root, nên `models/` và `runs/` thuộc root; muốn xoá thì dùng `sudo` hoặc một container.

---

## 6. Cấu trúc thư mục

```
README.md
src/clean_ir/         tầng ce: pipeline.py (prepare/train/predict/verify, stage contract + khoá SHA source/model/corpus), src/*.py, Dockerfile, tests
src/clean_ir_s1/      tầng MIL + final: evidence.py, train_mil.py, score_mil.py, qwen_rr.py, mil_common.py, lexalign.py,
                      selector_stage.py, features.py, train_exact_overlay.py, package.py, common.py, dense.py, tests
src/price.py          fit selector a3 (build ma trận 83 cột, final:ce+mil2)
scripts/              env.sh, setup.sh, download_models.sh, MODELS.sha256, compose.es.yaml,
                      run_from_scratch.sh, mil_fold.sh, run_private.sh, final_stage.sh, compare.py
env/                  pip_freeze_image.txt
artifacts/reference/  ZIP đã nộp (4bb3380c), dùng để so sánh ở bước cuối
data/                 (không commit) dữ liệu BTC
models/               (không commit) 5 model gốc
runs/                 (không commit) đầu ra khi chạy
```

- Hai index ES là `clean_ir_<owner>_chunks` và `clean_ir_<owner>_wseg`, với `owner = sha(source, corpus, '/work')[:20]`. Owner không phụ thuộc đường dẫn host, nên work dir di chuyển được, miễn giữ nguyên các mount `/work /contexts /models /data`.
- Các script đọc ES URL và tên index từ `<work>/config.json`.

---

## 7. Tuân thủ thể lệ

- **Dữ liệu:** chỉ dùng `train_IR.json` (nhãn) và `selected-contexts` (corpus), đều đọc từ `data/`. Câu hỏi private chỉ được đọc phần văn bản. `LegalIR_warmup.json` không được dùng.
- **Overlay train-exact:** nếu một câu hỏi trùng khít văn bản (sau chuẩn hoá) với một câu train, gold của câu train được đưa lên đầu. Bước này chỉ đọc `train_IR.json`. Trên private, overlay đổi đúng 1/2.080 câu.
- **Mô hình và giới hạn:** 5 mô hình BTC duyệt, tổng 2.98B tham số, không API, chạy offline.
- **Định dạng nộp:** mỗi câu có 1–5 doc_id, không trùng lặp, và đều thuộc catalog 8.532 văn bản. `package.py` kiểm tra rồi ghi `validation.json`.
