# Plan & Progress Status

> Bảng tiến độ theo phân công chi tiết tại [team-assignment.md](team-assignment.md).
> Mốc tổng duyệt theo kế hoạch nhóm: **Thứ Năm, 26/11/2026**.
> Cấu trúc và các đường dẫn code là kế hoạch triển khai; chỉ đánh dấu hoàn thành khi code đã chạy và có bằng chứng test.

## 1. Theo dõi 7 thành viên

| Thành viên | Module / phạm vi | Trạng thái | Bằng chứng bàn giao |
|---|---|---|---|
| Võ Việt Đức – Nhóm trưởng | Config, seed/device, orchestration, TensorBoard integration, GitHub và tích hợp | Đang triển khai | PR, lệnh chạy, smoke test, hướng dẫn tích hợp |
| Quốc Anh | Download, inspect, audit, EDA, data pipeline | Chưa cập nhật | Data smoke, normal run, audit JSON/TXT, EDA, data profile |
| Duy Căn | Tokenizer, vocabulary, padding, dataset adapter | Chưa cập nhật | Unit tests preprocessing, ví dụ review → tensor |
| Thành Phát | LSTM, training loop, loss, checkpoint, TensorBoard logging | Chưa cập nhật | Model/training smoke, checkpoint, history, TensorBoard |
| Minh Khang | TF-IDF + Logistic Regression, GRU, BiLSTM, GloVe, configs | Chưa cập nhật | Baseline/model smoke, configs và kết quả từng experiment |
| Đặng Đức | Metrics, confusion matrix, visualization, error analysis | Chưa cập nhật | Evaluation tests, metrics, confusion matrix, error samples |
| Đình Phú | Demo, predict service, HTML/PDF reporting | Chưa cập nhật | Demo chạy được, data/model/final report HTML/PDF |

## 2. Tiêu chí hoàn thành module

Mỗi thành viên phải bàn giao đủ:
1. Source code trong đúng thư mục đã thống nhất.
2. Smoke test hoặc unit test phù hợp.
3. Lệnh chạy và kết quả thực tế.
4. Tài liệu giải thích input, output và luồng xử lý.
5. Phần trình bày vấn đáp 3–5 phút và demo module.

## 3. Quy tắc phối hợp

- Quốc Anh phụ trách download/inspect/audit/EDA; Duy Căn phụ trách tokenizer/vocabulary/padding/dataset adapter. Hai người thống nhất schema dữ liệu.
- Thành Phát phụ trách LSTM/training; Võ Việt Đức phối hợp config và TensorBoard integration.
- Minh Khang dùng chung split và preprocessing; Đặng Đức đánh giá các model bằng cùng test set và metric.
- Đình Phú nhận artifact dữ liệu từ Quốc Anh và metrics/plots từ Đặng Đức để sinh báo cáo; không tự hard-code kết quả.
- Võ Việt Đức review Pull Request và tích hợp vào main.
- Mỗi người làm trên feature branch, không push trực tiếp main.
- Chỉ dùng TensorBoard; không triển khai ThingSpeak.
- Không commit dataset gốc, checkpoint lớn, event logs và secrets.

## 4. Mốc tổng duyệt

| Mốc | Kết quả cần có |
|---|---|
| Trước 22/11 | Data audit/EDA và preprocessing smoke; model/training smoke |
| Trước 23/11 | Normal data run; LSTM baseline train được |
| Trước 24/11 | TensorBoard, checkpoint, history; baseline và evaluation bước đầu |
| Trước 25/11 | Extension trong phạm vi khả thi; demo và HTML/PDF report |
| **26/11/2026** | Tổng duyệt: clone/run, trình bày module và trả lời vấn đáp |

## 5. Cập nhật tiến độ

Thành viên cập nhật trạng thái, PR và bằng chứng chạy thực tế sau mỗi lần bàn giao. Không dùng trạng thái “Hoàn thành” khi chưa chạy kiểm thử.
