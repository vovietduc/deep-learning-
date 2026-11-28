# Integration Review – IMDb Sentiment Analysis

Ngày rà soát: 28/11/2026

Tài liệu này phân biệt rõ: (1) code đã có trong GitHub, (2) code nhận qua ZIP, và (3) thay đổi tích hợp đã đưa lên branch chuẩn hóa. Không xem file ZIP hoặc kết quả chạy độc lập là đã merge vào repository.

## 1. Branch tích hợp

Branch: feature/duc-integration-standardized  
Base: feature/duc-integration-core

Các thay đổi của nhóm trưởng đã commit:
- src/config.py: typed ProjectConfig và load_config dạng nested để tương thích data scripts.
- src/utils/seed.py: set_seed và alias seed_everything.
- src/utils/device.py: auto/cpu/cuda và thông tin thiết bị.
- src/utils/tensorboard.py: SummaryWriter helpers.
- configs/default.yaml: seed 42, validation 0.2, batch 64, vocab 30k, max length 300.
- scripts/run_project.py: check/data/train/evaluate/all orchestration; từ chối chạy module còn placeholder.
- scripts/export_khang_manifest.py: chuyển split_manifest.json sang CSV split,path,label cho runner của Khang.
- tests/test_integration_core.py và tests/test_preprocessing.py.
- src/data/dataset.py, padding.py, tokenizer.py, vocabulary.py: đưa preprocessing của Duy Căn từ main vào branch tích hợp.
- src/data/preprocessing.py: adapter dùng tokenizer/vocabulary chung, giữ API mà pipeline Quốc Anh đang gọi.
- README.md và docs/project-design.md: hướng dẫn CUDA và data contract.

## 2. Duy Căn – code trên main

Đã tìm thấy bốn module trên main: dataset.py, padding.py, tokenizer.py, vocabulary.py; kèm tests/test_preprocessing.py và conftest.py.

Đánh giá:
- Thiết kế tách tokenizer, vocabulary, padding và Dataset rõ ràng.
- Dataset có tùy chọn return_lengths=True, phù hợp pack_padded_sequence của GRU/BiLSTM.
- split_train_val stratified, deterministic; build_vocabulary nhận train_samples riêng.
- encode_text dùng chung tokenizer/vocabulary cho inference.
- PAD=0, UNK=1.

Điểm tích hợp:
- Vocabulary ban đầu của Duy Căn dùng token_to_id/PAD_ID/UNK_ID/build(), trong khi pipeline Quốc Anh gọi token_to_idx/PAD_IDX/UNK_IDX/build_from_tokens().
- Branch tích hợp bổ sung alias tương thích trong vocabulary.py, giữ API gốc của Duy Căn.
- Dataset của Duy Căn là Dataset canonical; không ghi đè bằng dataset.py của Quốc Anh (dataset npz adapter). Nếu cần dùng NPZ, đổi tên file đó thành npz_dataset.py hoặc dùng adapter riêng.
- Pipeline preprocessing của Quốc Anh không được giữ tokenizer/vocabulary thứ hai. Adapter mới trong src/data/preprocessing.py dùng Tokenizer và Vocabulary của Duy Căn.

Trạng thái: đã đưa code Duy Căn và adapter vào branch tích hợp; chưa merge vào main.

## 3. Quốc Anh – ZIP data pipeline

Các module trong ZIP: download, inspect, audit, EDA, preprocessing, vocabulary, dataset, reporting, scripts và tests.

Kiểm thử độc lập:
- tests/test_data_pipeline.py: 5 passed khi chạy với temporary compatibility shims cho src.config.load_config và src.utils.seed.seed_everything.
- ZIP nguyên trạng không tự import được trong repository hiện tại vì thiếu load_config và seed_everything ở thời điểm bàn giao ban đầu.
- Branch chuẩn hóa hiện đã cung cấp hai interface này.

Xung đột cần xử lý:
- dataset.py và vocabulary.py của ZIP trùng tên và khác API với module của Duy Căn.
- Branch tích hợp giữ Dataset/Vocabulary của Duy Căn làm chuẩn; adapter preprocessing giữ các hàm mà audit/EDA của Quốc Anh cần.
- Các file audit.py, download.py, inspect.py, eda.py, reporting.py và scripts/run_data_*.py từ ZIP vẫn cần được đưa vào branch sau review; chúng chưa được xem là đã merge chỉ vì ZIP đã được kiểm thử.

## 4. Minh Khang – ZIP model extensions

Các module đã nhận: TF-IDF + Logistic Regression, GRU, BiLSTM, GRU + GloVe, runner, data helper, configs, tests và metrics/predictions.

Kiểm thử độc lập:
- tests/test_khang_models.py: 5 passed.
- Các file metrics và predictions trong ZIP có số mẫu test 25,000; accuracy tính lại từ predictions.csv khớp metrics.json trong các run đã bàn giao.
- Kết quả thực nghiệm trong ZIP được tạo bằng CPU, Python 3.12.14 và PyTorch 2.15.0+cpu.

Các điểm cần chuẩn hóa trước khi so sánh với LSTM chính:
- scripts/khang_data.py tự tokenize và tạo vocabulary riêng; cần dùng cùng Tokenizer/Vocabulary canonical.
- Khang runner hiện ghi device=cpu và chưa chuyển model/batch sang CUDA.
- Runner cần đọc split manifest dùng chung. scripts/export_khang_manifest.py đã được thêm để chuyển manifest JSON của data pipeline thành CSV split,path,label.
- GloVe experiment hiện là GRU được khởi tạo embedding từ GloVe; tên kết quả nên thể hiện rõ GRU + GloVe.
- Chỉ so sánh kết quả sau khi các model dùng cùng split, tokenizer, vocabulary, max sequence length và test set.

Trạng thái: source ZIP đã review và unit test độc lập pass; chưa được commit vào branch tích hợp và chưa chạy lại trên RTX 3050.

## 5. Data contract được chọn

- Label: neg=0, pos=1.
- Seed: 42.
- Full IMDb supervised data: 25,000 train + 25,000 test; train/unsup loại khỏi supervised classification.
- Validation: stratified 20% từ 25,000 train -> 20,000 train + 5,000 validation.
- Vocabulary chỉ fit trên train; validation/test không fit lại.
- PAD=0, UNK=1.
- Max vocabulary mặc định 30,000; max sequence length 300.
- Device mặc định auto; CUDA được ưu tiên khi khả dụng.
- AMP mặc định false cho đến khi training loop của Thành Phát xác nhận hỗ trợ an toàn.

## 6. Những gì chưa được xác nhận

- Chưa có full data pipeline Quốc Anh trong branch để chạy end-to-end.
- Chưa có LSTM/training loop hoàn chỉnh của Thành Phát trong branch.
- Chưa có evaluation/report/demo hoàn chỉnh của các thành viên còn lại.
- Chưa chạy integration test trên máy RTX 3050 của người dùng trong phiên này.
- Vì vậy chưa được tuyên bố toàn bộ project đã chạy end-to-end hoặc sẵn sàng merge main.

## 7. Lệnh kiểm tra sau khi clone branch

Từ thư mục sentiment_analysis_lstm:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.run_project --stage check
```

Khi Quốc Anh bàn giao các module còn lại:
```powershell
python -m scripts.run_project --stage data
python -m scripts.export_khang_manifest
```
