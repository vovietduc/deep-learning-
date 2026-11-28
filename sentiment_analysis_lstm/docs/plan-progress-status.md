# Plan & Progress Status

> Bảng tiến độ theo phân công tại team-assignment.md.
> Cập nhật: 28/11/2026. Mốc tổng duyệt 26/11/2026 đã qua; nhóm cần xác nhận lại lịch nộp/thuyết trình với giảng viên.
> Chỉ đánh dấu hoàn thành khi source đã nằm trong branch chung, chạy được và có bằng chứng kiểm thử.

## 1. Theo dõi 7 thành viên

| Thành viên | Module / phạm vi | Trạng thái hiện tại | Bằng chứng / việc còn lại |
|---|---|---|---|
| Võ Việt Đức – Nhóm trưởng | Config, seed/device, orchestration, TensorBoard helpers, GitHub integration | Đã triển khai trên feature/duc-integration-clean; chờ chạy trên máy RTX 3050 | Commits cho config, seed, device, TensorBoard, runner, default.yaml, tests và docs; cần xác nhận bằng pytest + stage check trên máy nhóm trưởng |
| Quốc Anh | Download, inspect, audit, EDA, data pipeline | Đã nhận ZIP và review; chưa đưa toàn bộ source vào branch chung | tests/test_data_pipeline.py: 5 passed với temporary shims cho config/seed; cần bàn giao/merge các module audit, download, inspect, EDA, reporting và scripts |
| Duy Căn | Tokenizer, vocabulary, padding, dataset adapter | Source đã có trên main và được đưa vào branch tích hợp | dataset/tokenizer/vocabulary/padding và test_preprocessing.py; đã bổ sung compatibility aliases để kết nối data pipeline |
| Thành Phát | LSTM, training loop, loss, checkpoint, TensorBoard logging | Chưa có bàn giao source để tích hợp trong branch này | Cần model/training smoke, checkpoint, history, TensorBoard event |
| Minh Khang | TF-IDF + Logistic Regression, GRU, BiLSTM, GloVe, configs | Đã nhận ZIP và review; chưa merge source vào branch tích hợp | tests/test_khang_models.py: 5 passed; cần dùng chung preprocessing/manifest và chuyển training sang CUDA |
| Đặng Đức | Metrics, confusion matrix, visualization, error analysis | Chưa có bàn giao source để tích hợp trong branch này | Cần metrics tests, confusion matrix và error samples |
| Đình Phú | Demo, predict service, HTML/PDF reporting | Chưa có bàn giao source để tích hợp trong branch này | Cần demo chạy được và report lấy số liệu từ artifacts |

## 2. Tình trạng branch tích hợp

Branch: feature/duc-integration-clean (base: feature/duc-integration-core).

Đã có:
- Shared ProjectConfig và load_config tương thích nested YAML.
- Seed API gồm set_seed và seed_everything.
- Device resolver auto/cpu/cuda.
- TensorBoard SummaryWriter helpers.
- configs/default.yaml với seed=42, validation_size=0.2, batch_size=64.
- Project runner check/data/train/evaluate/all (các stage phụ thuộc module của thành viên).
- Exporter chuyển split_manifest.json sang CSV cho Khang.
- Duy Căn's tokenizer/vocabulary/padding/dataset trên branch tích hợp.
- Preprocessing adapter kết nối API pipeline Quốc Anh với module Duy Căn.
- README, project design và integration review.

Lưu ý: branch này chưa phải bản end-to-end hoàn chỉnh. Các module Quốc Anh trong ZIP, Minh Khang trong ZIP và training/evaluation/demo của các thành viên chưa được merge đầy đủ.

## 3. Tiêu chí hoàn thành module

Mỗi thành viên phải bàn giao đủ:
1. Source code trong đúng thư mục đã thống nhất.
2. Smoke test hoặc unit test phù hợp.
3. Lệnh chạy và kết quả thực tế.
4. Tài liệu giải thích input, output và luồng xử lý.
5. Phần trình bày vấn đáp 3–5 phút và demo module.

## 4. Quy tắc phối hợp

- Quốc Anh sở hữu download/inspect/audit/EDA; Duy Căn sở hữu tokenizer/vocabulary/padding/Dataset API. Không ghi đè file của nhau.
- Thành Phát sở hữu LSTM/training; Võ Việt Đức hỗ trợ config/device/TensorBoard integration.
- Minh Khang dùng chung split, tokenizer và vocabulary; Đặng Đức đánh giá model bằng cùng test set.
- Đình Phú nhận artifact từ Quốc Anh và Đặng Đức để sinh report; không hard-code kết quả.
- Võ Việt Đức review Pull Request và tích hợp vào main.
- Mỗi người làm trên feature branch, không push trực tiếp main.
- Chỉ dùng TensorBoard; không triển khai ThingSpeak.
- Không commit dataset gốc, GloVe vectors, checkpoint lớn, event logs và secrets.

## 5. Lệnh kiểm chứng branch

Từ thư mục sentiment_analysis_lstm:

```powershell
python -m pytest -q
python -m scripts.run_project --stage check
```

Khi Quốc Anh bàn giao đủ module data:

```powershell
python -m scripts.run_project --stage data
python -m scripts.export_khang_manifest
```

Sau khi Thành Phát và Đặng Đức hoàn thành:

```powershell
python -m scripts.run_project --stage train
python -m scripts.run_project --stage evaluate
python -m scripts.run_project --stage all
tensorboard --logdir runs
```

## 6. Nhật ký cập nhật

- 28/11/2026: tạo branch chuẩn hóa; cập nhật config, seed, device, TensorBoard helpers, runner, default config, tests, docs; đưa module preprocessing của Duy Căn vào branch; thêm adapter cho API pipeline Quốc Anh và exporter manifest cho Khang.
- 28/11/2026: review ZIP Quốc Anh và Khang; kiểm thử độc lập trên source ZIP. Chưa xác nhận end-to-end trên branch tích hợp.
