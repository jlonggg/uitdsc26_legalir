# clean_ir — tầng `ce` (LegalIR)

Pipeline độc lập đi trọn **dữ liệu BTC → huấn luyện → submission** trong mã nguồn. Public/private dùng cùng một lệnh và nhận file câu hỏi bất kỳ. Đầu vào runtime chỉ gồm `context_*.json`, file câu hỏi, file train có nhãn và bốn checkpoint gốc; không dùng nhãn public, bảng tra câu hỏi, vector hay index dựng sẵn.

## Đường xử lý

1. Đọc `context_*.json`: tự sinh chunk 1500/1800 ký tự, Article units, catalog gồm cả văn bản rỗng, doc metadata và legal-reference graph.
2. Tạo hai index Elasticsearch riêng cho work directory. Tự encode Qwen3 (Article), Vietnamese_Embedding_v2 và Harrier (chunk); lưu vector cục bộ. Dense search là exact search, tie-break cố định.
3. BM25 top60 + Qwen top30 bổ sung + union Vietnamese_Embedding_v2/Harrier top20 → teacher CE, pass2 (top10 doc), pass3 (doc hạng 11–40).
4. Huấn luyện student CE LoRA theo fold hash câu hỏi để sinh đặc trưng out-of-fold; student full-fit trên toàn bộ train dùng cho suy luận.
5. Fit ensemble LightGBM lambdarank 69 đặc trưng (6 seed), lưu model. Teacher/student pass2 dùng cùng chính sách ở train và inference.
6. `predict` chỉ dùng model đã fit và nội dung câu hỏi; không fit lại selector. Kênh slug (TF-IDF ký tự, ngưỡng 0.45) chỉ chèn văn bản rỗng từ metadata corpus. Validate theo đủ catalog BTC và tạo ZIP duy nhất chứa `submission.json`.

## Chạy

Chạy qua các script ở `scripts/` của repo (README gốc §3–4): `scripts/setup.sh` build image và bật ES, `scripts/run_from_scratch.sh ce` gọi `pipeline.py prepare/train/verify` với code mount vào `/app` và dữ liệu lấy từ `data/` của repo. `model_audit.py` đếm tham số thực của toàn DAG (teacher, student có adapter, ba encoder) và bắt buộc tổng < 4 tỷ.

Có thể chia prepare thành `--stage corpus`, `--stage index`, `--stage dense`. `train` yêu cầu prepare đã xong. Giữ nguyên mount `/data`, `/contexts`, `/models`, `/work` giữa các lần gọi. `verify` kiểm SHA input/model/artifact và ownership/số bản ghi của index. Work đã có sẽ từ chối khi source hoặc dữ liệu train đổi (kể cả chỉ đổi comment trong `src/*.py`); dùng work mới. Không chạy hai tiến trình trên cùng work, khóa `flock` sẽ từ chối.

## Đánh giá trên fold dev

Tách một fold train theo hash câu hỏi đã chuẩn hóa, không dùng nhãn held-out ở bất kỳ stage nào:

```bash
python3 evaluate.py split --train /path/train_IR.json --fold 0 --out /path/dev_split
```

Dựng work mới, train bằng `dev_split/train.json`, predict bằng `dev_split/held_questions.json`, rồi:

```bash
python3 evaluate.py score --gold /path/dev_split/held_gold.json \
  --submission /path/dev_work/predict_HASH/final/submission.zip \
  --work /path/dev_work --out /path/dev_result.json
```

Lệnh score từ chối nếu có câu hỏi chuẩn hóa trùng giữa train và held, và báo Recall, Precision và recall từng câu.

## Kiểm tra và artifact

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

`train --smoke --trees 10 --seeds 2026` chỉ kiểm tra đường chạy trên tập nhỏ; model manifest đánh dấu `smoke_only`.

`config.json` ghi SHA source/corpus/checkpoint; `steps/` giữ command, thời gian, SHA output; `logs/` giữ stdout/stderr; `model/manifest.json` ghi student/selector/parameter audit. Một file tồn tại không được coi là stage đã hoàn tất.

Giới hạn: một GPU, không tự chuyển GPU khi OOM; không cam kết bit-exact giữa GPU/thư viện khác nhau.
