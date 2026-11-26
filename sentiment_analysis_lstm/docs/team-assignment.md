# Phân công công việc – Sentiment Analysis with LSTM

> Phiên bản phân công theo cấu trúc repository hiện tại: `sentiment_analysis_lstm/`.
> Mục tiêu: cả 7 thành viên đều trực tiếp viết code, chạy kiểm thử và chuẩn bị phần vấn đáp của module mình phụ trách.
> Mốc tổng duyệt theo kế hoạch nhóm: **Thứ Năm, 26/11/2026**.

## 1. Nguyên tắc phân công

- Phân công bám theo các thư mục đã có trên GitHub: `src/data`, `src/models`, `src/training`, `src/evaluation`, `src/reporting`, `src/demo`, `src/utils`, `scripts`, `tests`, `configs`, `reports`, `runs`, `models`, `notebooks`, `docs`.
- Mỗi người có phần code cụ thể, đầu ra kiểm chứng được và nội dung tự trình bày khi vấn đáp.
- Đường dẫn file bên dưới là **file dự kiến cần tạo trong các thư mục hiện có**, không có nghĩa các file code đó đã được triển khai.
- Các mô hình phải dùng chung quy tắc chia dữ liệu và preprocessing đã thống nhất; không fit vocabulary/TF-IDF trên test.
- Chỉ dùng TensorBoard để theo dõi training; không triển khai ThingSpeak.
- Dataset, checkpoint lớn, TensorBoard event logs và secret không commit lên Git.

## 2. Phân công chi tiết theo thành viên

### 1) Võ Việt Đức – Nhóm trưởng: Project Integration, Configuration & TensorBoard

**Thư mục/file phụ trách**
- `src/config.py`: đọc và kiểm tra cấu hình chung.
- `src/utils/seed.py`: cố định random seed.
- `src/utils/device.py`: chọn CPU/CUDA và thông tin thiết bị.
- `scripts/run_project.py`: điều phối các bước chạy theo hướng dẫn.
- Phối hợp Thành Phát tích hợp TensorBoard vào training; thống nhất quy ước tên run và vị trí log trong `runs/lstm/`, `runs/gru/`, `runs/bilstm/`.
- Quản lý `README.md`, `docs/project-design.md`, GitHub Issues/branches/PR và tích hợp cuối.

**Bàn giao/kiểm chứng**
- Cấu hình dùng thống nhất giữa train/evaluate/demo.
- Smoke test kiểm tra seed, device, đọc config và gọi được entry point.
- Hướng dẫn clone, cài đặt, chạy data, train, evaluate, demo và TensorBoard.

**Chuẩn bị vấn đáp**
- Kiến trúc tổng thể và luồng dữ liệu.
- Vì sao cố định seed, chọn device và tách config?
- TensorBoard nhận scalar từ training loop như thế nào?
- Git branch, Pull Request và tích hợp module.

### 2) Quốc Anh – Data Acquisition, Inspection, Audit & EDA

**Thư mục/file phụ trách**
- `src/data/download.py`: tải/giải nén IMDb Large Movie Review Dataset.
- `src/data/inspect.py`: kiểm tra cấu trúc thư mục, file, encoding và nhãn.
- `src/data/audit.py`: kiểm tra số lượng, nhãn hợp lệ, file rỗng/lỗi, trùng lặp và tính toàn vẹn split phù hợp.
- `src/data/eda.py`: thống kê và trực quan hóa phân bố sentiment, độ dài review và đặc điểm dữ liệu.
- `scripts/run_data_audit.py`: chạy audit/EDA theo pipeline.
- `notebooks/`: notebook khám phá dữ liệu nếu cần.
- `reports/data_audit/`: `data_audit.json`, `data_audit.txt`, `data_profile.json`, biểu đồ và `data_report.html` (phối hợp Đình Phú phần sinh HTML).

**Bàn giao/kiểm chứng**
- Hướng dẫn tải dữ liệu, không commit dataset gốc.
- Data smoke run trên subset và normal data run trên toàn bộ dữ liệu.
- Audit có PASS/FAIL rõ ràng; không ghi số liệu chưa chạy.

**Chuẩn bị vấn đáp**
- Cấu trúc IMDb và nhãn positive/negative.
- Audit kiểm tra gì, phát hiện lỗi nào?
- EDA cho biết điều gì?
- Vì sao phải kiểm tra dữ liệu trước khi train?

### 3) Duy Căn – Text Preprocessing, Tokenization & Vocabulary

**Thư mục/file phụ trách**
- `src/data/tokenizer.py`: chuẩn hóa văn bản và tokenization.
- `src/data/vocabulary.py`: xây dựng vocabulary, mapping token ↔ index, xử lý UNK/PAD.
- `src/data/padding.py`: truncate/pad sequence về độ dài thống nhất.
- `src/data/dataset.py`: Dataset/DataLoader adapter để trả về tensor và label (phối hợp Quốc Anh về định dạng dữ liệu đầu vào).
- `tests/test_preprocessing.py`: kiểm thử tokenizer, vocabulary, padding và dữ liệu đầu ra.
- `notebooks/`: notebook minh họa review → tokens → IDs → padded sequence.
- Cập nhật nội dung kỹ thuật tương ứng trong `docs/data.md` và phần Embedding trong `docs/model.md`.

**Bàn giao/kiểm chứng**
- Unit test cho văn bản rỗng, token ngoài vocabulary, sequence ngắn/dài.
- Một ví dụ minh họa review chuyển thành tensor.
- Quy ước PAD/UNK và độ dài chuỗi được ghi rõ để model dùng thống nhất.

**Chuẩn bị vấn đáp**
- Tokenization, vocabulary, token ID.
- Padding/truncation và lý do cần thiết.
- Embedding biến token ID thành vector như thế nào?
- RNN/LSTM xử lý chuỗi theo thứ tự ra sao?

### 4) Thành Phát – LSTM Architecture & Training Pipeline

**Thư mục/file phụ trách**
- `src/models/lstm.py`: Embedding + LSTM + Dense/Linear cho phân loại nhị phân.
- `src/training/trainer.py`: training/validation loop.
- `src/training/loss.py`: loss function và cách tính.
- `src/training/checkpoint.py`: lưu/tải checkpoint.
- `src/train.py`: entry point huấn luyện.
- `runs/lstm/`: TensorBoard event logs được tạo khi chạy, không commit logs lên Git.
- `models/` và `reports/lstm/`: checkpoint/history/metrics theo quy ước lưu trữ; checkpoint lớn không commit nếu vượt giới hạn repository.
- Phối hợp Võ Việt Đức ghi scalar loss/accuracy theo epoch vào TensorBoard.
- Cập nhật `docs/training.md`.

**Bàn giao/kiểm chứng**
- Model smoke: forward pass, output shape phù hợp batch.
- Training smoke: chạy vài batch/epoch nhỏ, loss hữu hạn, backward và optimizer step chạy được.
- Normal training: lưu checkpoint, history và log thực nghiệm.
- Ghi rõ hyperparameters thực tế đã dùng.

**Chuẩn bị vấn đáp**
- Embedding, hidden state, cell state.
- Forget/Input/Output Gate.
- Forward pass, loss, backward, optimizer.
- Epoch, batch size, learning rate, early stopping, checkpoint.
- TensorBoard ghi nhận metric ở đâu trong training loop?

### 5) Minh Khang – Baseline & Model Extensions

**Thư mục/file phụ trách**
- `src/models/baseline.py`: TF-IDF + Logistic Regression baseline.
- `src/models/gru.py`: GRU classifier.
- `src/models/bilstm.py`: Bidirectional LSTM classifier.
- `src/models/glove.py`: nạp GloVe embedding và kết nối với mô hình phù hợp.
- `configs/`: cấu hình baseline/GRU/BiLSTM/GloVe và các hyperparameters.
- `reports/gru/`, `reports/bilstm/`, `reports/glove/`: kết quả từng thử nghiệm.
- Cập nhật phần kiến trúc mở rộng trong `docs/model.md`.

**Bàn giao/kiểm chứng**
- Baseline chạy được trên cùng train/validation/test split.
- Model smoke cho từng kiến trúc neural đã triển khai.
- Báo cáo cấu hình, kết quả và giới hạn của từng extension.
- Nếu không đủ thời gian chạy đủ mọi extension, ghi rõ phần nào đã code, phần nào đã chạy và phần nào chưa hoàn thành.

**Chuẩn bị vấn đáp**
- TF-IDF và Logistic Regression.
- GRU khác LSTM ở đâu?
- Bidirectional LSTM hoạt động thế nào?
- GloVe là pretrained word embedding gì?
- So sánh công bằng giữa các mô hình cần giữ cố định những gì?

### 6) Đặng Đức – Evaluation, Metrics & Error Analysis

**Thư mục/file phụ trách**
- `src/evaluation/metrics.py`: accuracy, precision, recall, F1.
- `src/evaluation/confusion_matrix.py`: confusion matrix và visualization.
- `src/evaluation/error_analysis.py`: trích xuất review dự đoán sai.
- `src/evaluation/visualization.py`: biểu đồ so sánh và kết quả đánh giá.
- `src/evaluate.py`: entry point đánh giá trên test set.
- `reports/lstm/`, `reports/gru/`, `reports/bilstm/`, `reports/glove/`: metrics và biểu đồ tương ứng.
- Cập nhật `docs/evaluation.md`.

**Bàn giao/kiểm chứng**
- Đánh giá trên test set độc lập.
- Lưu metrics có thể đọc lại (JSON/CSV) và confusion matrix.
- Có ví dụ false positive/false negative để phân tích.
- Kiểm tra metric bằng test case nhỏ có kết quả kỳ vọng.

**Chuẩn bị vấn đáp**
- Ý nghĩa Accuracy, Precision, Recall, F1.
- TP, TN, FP, FN.
- Vì sao accuracy có thể gây hiểu nhầm?
- Đọc confusion matrix và phân tích dự đoán sai.

### 7) Đình Phú – Demo Application & Report Generation

**Thư mục/file phụ trách**
- `src/demo/predict_service.py`: nhận review, gọi preprocessing và model, trả nhãn sentiment.
- `src/demo/app.py`: giao diện demo nhập review và hiển thị kết quả.
- `src/reporting/data_report.py`: tạo báo cáo HTML dữ liệu từ artifact do Quốc Anh xuất.
- `src/reporting/model_report.py`: tạo báo cáo HTML kết quả mô hình từ metrics/plots do Đặng Đức xuất.
- `scripts/generate_reports.py`: chạy sinh báo cáo.
- `reports/final_report/`: tổng hợp `final_report.html` và `final_report.pdf` (phối hợp cả nhóm cung cấp nội dung và kết quả).
- Cập nhật `docs/demo.md`.

**Bàn giao/kiểm chứng**
- Demo chạy được với review tự nhập và hiển thị Positive/Negative.
- Kiểm tra review rỗng và đầu vào bất thường.
- HTML/PDF lấy dữ liệu thực tế từ JSON/CSV/plots, không hard-code số liệu.
- Hướng dẫn chạy demo và mở báo cáo.

**Chuẩn bị vấn đáp**
- Luồng từ input review đến prediction.
- Vì sao inference phải dùng đúng tokenizer/vocabulary lúc train?
- Cách báo cáo HTML lấy metrics/biểu đồ.
- Phân biệt nhãn dự đoán và confidence score.

## 3. Bảng quyền sở hữu thư mục hiện có

| Thư mục hiện có trên GitHub | Người phụ trách chính | Phối hợp |
|---|---|---|
| `src/data/` | Quốc Anh (download/inspect/audit/EDA); Duy Căn (tokenizer/vocabulary/padding/dataset) | Hai người thống nhất interface |
| `src/models/` | Thành Phát (LSTM); Minh Khang (baseline/GRU/BiLSTM/GloVe) | Dùng chung interface model |
| `src/training/` | Thành Phát | Đức (TensorBoard/config) |
| `src/evaluation/` | Đặng Đức | Minh Khang, Thành Phát |
| `src/reporting/` | Đình Phú | Quốc Anh, Đặng Đức |
| `src/demo/` | Đình Phú | Thành Phát, Duy Căn |
| `src/utils/` | Võ Việt Đức | Cả nhóm dùng chung |
| `scripts/` | Đức (run_project); Quốc Anh (run_data_audit); Đình Phú (generate_reports) | Thành viên liên quan |
| `tests/` | Mỗi người viết test cho module mình; Duy Căn điều phối test preprocessing | Đức tích hợp test |
| `configs/` | Minh Khang (model configs); Đức (config loader/schema) | Thành Phát |
| `notebooks/` | Quốc Anh (EDA); Duy Căn (preprocessing demo) | — |
| `reports/data_audit/` | Quốc Anh | Đình Phú sinh HTML |
| `reports/lstm/` | Thành Phát | Đặng Đức |
| `reports/gru/`, `reports/bilstm/`, `reports/glove/` | Minh Khang | Đặng Đức |
| `reports/final_report/` | Đình Phú | Cả nhóm |
| `reports/smoke_tests/` | Mỗi người lưu kết quả smoke của module mình | Đức tổng hợp |
| `runs/` | Thành Phát thiết lập log LSTM; Đức chuẩn hóa tracking | Minh Khang cho extension |
| `models/` | Thành viên sở hữu model tạo artifact/checkpoint tương ứng | Không commit checkpoint lớn |
| `docs/` | Mỗi người cập nhật tài liệu module mình; Đức rà soát | Cả nhóm |

## 4. Quy trình Git và quy tắc bàn giao

- Mỗi người tạo branch riêng, ví dụ: `feature/data-audit`, `feature/preprocessing`, `feature/lstm-training`, `feature/model-extensions`, `feature/evaluation`, `feature/demo-report`, `feature/integration`.
- Không push trực tiếp lên `main`; tạo Pull Request để Võ Việt Đức review và merge.
- Mỗi Pull Request phải có: mục đích, file thay đổi, lệnh chạy, kết quả smoke/test, ảnh/log nếu phù hợp.
- Không tự ý đổi tên thư mục hoặc interface dùng chung mà chưa thống nhất.
- Mỗi module cần README/docstring hoặc cập nhật file docs tương ứng.
- Không đánh dấu hoàn thành chỉ vì đã viết code: phải chạy được và có bằng chứng kiểm thử.

## 5. Chuẩn vấn đáp bắt buộc cho từng thành viên

Mỗi người chuẩn bị phần trình bày 3–5 phút, gồm:
1. Module giải quyết vấn đề gì?
2. Input và output là gì?
3. Giải thích luồng xử lý trong code.
4. Chạy demo/smoke test trực tiếp.
5. Nêu một lỗi có thể gặp và cách xử lý.
6. Giải thích module kết nối với các phần còn lại thế nào.

Ngoài phần riêng, tất cả thành viên phải nắm: mục tiêu sentiment classification, IMDb, pipeline dữ liệu, Embedding + LSTM, cách train, metric đánh giá và kết quả thực nghiệm cuối cùng.

## 6. Trạng thái

Các đường dẫn file code chi tiết trong tài liệu là kế hoạch triển khai theo cấu trúc repository hiện tại. Repository hiện mới có bộ khung thư mục và entry-point placeholder; mỗi thành viên cần tạo code thực tế, kiểm thử và cập nhật trạng thái sau khi bàn giao.
