# Plan & Progress Status

> Nguồn phân công: file `Ke_hoach_bai_tap_lon_Sentiment_Analysis_LSTM.docx`, mục 10–11.
> Mốc tổng duyệt: **Thứ Năm, 26/11/2026**.

## 1. Thành viên và phân công chính thức

| Thành viên | Nhiệm vụ chính | Sản phẩm bàn giao / trách nhiệm |
|---|---|---|
| **Võ Việt Đức – Nhóm trưởng** | Quản lý tiến độ; thống nhất kiến trúc; quản lý repository và quy tắc Git; tích hợp các nhánh; kiểm tra chất lượng; điều phối tổng duyệt. | `README.md`, `project-design.md`, cấu trúc repository, `plan-progress-status.md`; tích hợp code; kiểm thử toàn project; hướng dẫn chạy và điều phối tổng duyệt. Phối hợp Thành Phát tích hợp TensorBoard. |
| **Quốc Anh** | Data pipeline: download, inspect, audit, EDA, preprocessing, vocabulary, data smoke và normal data run. | `src/data/`, `docs/data.md`, `data_audit.json/txt`, `data_profile.json`, `data_report.html` và biểu đồ EDA. Bảo đảm dữ liệu và preprocessing dùng được cho các mô hình. |
| **Duy Căn** | Phụ trách lý thuyết: sentiment analysis, ANN/RNN, hạn chế RNN, LSTM gates, GRU, BiLSTM và GloVe. | `docs/model.md`, sơ đồ kiến trúc, nội dung slide lý thuyết và tài liệu tham khảo. |
| **Thành Phát** | LSTM baseline architecture; training loop; loss, optimizer, validation, checkpoint; model/training smoke. | `src/models/lstm.py`, `src/training/`, `src/train.py`, `docs/training.md`, history và training logs; phối hợp nhóm trưởng tích hợp TensorBoard. |
| **Minh Khang** | TF-IDF + Logistic Regression baseline và các thử nghiệm GRU, BiLSTM, GloVe; bảo đảm dùng chung data split. | Code baseline/mở rộng, configs, kết quả thực nghiệm và dữ liệu đầu vào phục vụ so sánh mô hình. |
| **Đặng Đức** | Evaluation, metrics, confusion matrix, visualization, error analysis và rà soát chất lượng kết quả. | `src/evaluation/`, `docs/evaluation.md`, `metrics.json`, confusion matrix, `predictions.csv`, `model_report.html` và error analysis. Rà soát độc lập kết quả audit/metrics cùng Quốc Anh. |
| **Đình Phú** | Demo; tổng hợp nội dung báo cáo; định dạng HTML/PDF; thiết kế slide; kiểm tra luồng demo. | `src/demo/`, `docs/demo.md`, `final_report.html/pdf`, slide và hướng dẫn demo. |

## 2. Quy tắc phối hợp

- Quốc Anh chịu trách nhiệm chất lượng dữ liệu gốc và preprocessing.
- Đặng Đức rà soát độc lập các kết quả audit/metrics và phân tích lỗi.
- Thành Phát phụ trách LSTM và training; Võ Việt Đức phối hợp tích hợp TensorBoard vào quá trình training.
- Minh Khang dùng chung data split và quy tắc preprocessing đã thống nhất khi so sánh baseline/mô hình mở rộng.
- Võ Việt Đức là người tích hợp cuối. Mỗi thành viên phải bàn giao code có thể chạy, hướng dẫn sử dụng và kết quả kiểm chứng.
- Nhóm sử dụng TensorBoard để theo dõi training. ThingSpeak không nằm trong phạm vi triển khai hiện tại theo quyết định thống nhất mới nhất của nhóm.
- Không commit dataset gốc, checkpoint dung lượng lớn, TensorBoard event logs hoặc API keys lên GitHub.

## 3. Timeline đến ngày tổng duyệt 26/11/2026

| Ngày | Mốc công việc | Người phụ trách | Điều kiện hoàn thành |
|---|---|---|---|
| **20/11 (Thứ Sáu)** | Chốt repository, cấu trúc, đề cương, phân công; tải dataset. | Đức + cả nhóm; Quốc Anh | Repository hoạt động; dataset tải/giải nén được; docs mô tả ban đầu. |
| **21/11 (Thứ Bảy)** | Inspect, audit, EDA, preprocessing; viết tài liệu lý thuyết. | Quốc Anh; Duy Căn | Có audit sơ bộ, biểu đồ EDA và data smoke pass. |
| **22/11 (Chủ Nhật)** | Normal data run; hoàn thành LSTM baseline và model smoke. | Quốc Anh + Thành Phát | Processed data đúng; model forward/backward và training smoke pass. |
| **23/11 (Thứ Hai)** | Huấn luyện LSTM; lưu checkpoint, history, training curves; tích hợp tracking. | Thành Phát + Đức | Có normal training log; metrics lưu local; TensorBoard hoạt động. |
| **24/11 (Thứ Ba)** | Baseline, GRU/BiLSTM/GloVe theo khả năng; evaluation. | Minh Khang + Đặng Đức | Có baseline và kết quả LSTM; extension ghi rõ trạng thái. |
| **25/11 (Thứ Tư)** | Sinh HTML/PDF, hoàn thiện demo, README, slide, chạy test. | Đình Phú + cả nhóm | Các artifact cần thiết có thật; demo chạy; không còn lỗi blocker. |
| **26/11 (Thứ Năm)** | Tổng duyệt; mỗi người trình bày phần mình và trả lời câu hỏi. | Cả 7 thành viên | Clone/run theo README; trình bày được pipeline, mô hình, kết quả, lỗi và giới hạn. |

## 4. Bảng cập nhật tiến độ

Mỗi thành viên cập nhật trạng thái và đường dẫn bàn giao khi có thay đổi. Không đánh dấu hoàn thành nếu chưa có sản phẩm và bước kiểm chứng.

| Thành viên | Hạng mục | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|---|
| Võ Việt Đức | Repository, kiến trúc, tài liệu chung, tích hợp và kiểm thử | Đang triển khai | Cấu trúc project đã được push; tiếp tục hoàn thiện README/docs và quy trình tích hợp. |
| Quốc Anh | Data pipeline, audit, EDA, preprocessing | Chưa cập nhật | Bổ sung link commit và báo cáo dữ liệu. |
| Duy Căn | Lý thuyết và tài liệu mô hình | Chưa cập nhật | Bổ sung tài liệu/sơ đồ/slide. |
| Thành Phát | LSTM, training loop, smoke và TensorBoard | Chưa cập nhật | Bổ sung code, log và kết quả smoke. |
| Minh Khang | Baseline và mô hình mở rộng | Chưa cập nhật | Bổ sung configs và kết quả thực nghiệm. |
| Đặng Đức | Evaluation và error analysis | Chưa cập nhật | Bổ sung metrics, confusion matrix và báo cáo lỗi. |
| Đình Phú | Demo, báo cáo HTML/PDF và slide | Chưa cập nhật | Bổ sung demo và artifact báo cáo. |

## 5. Nguyên tắc ưu tiên

Nếu thời gian hoặc tài nguyên hạn chế, ưu tiên hoàn thiện theo thứ tự:

1. Data Audit và preprocessing đúng.
2. TF-IDF + Logistic Regression baseline và LSTM baseline.
3. Training có smoke run, checkpoint, history và TensorBoard.
4. Evaluation trên test: Accuracy, Precision, Recall, F1-score, Confusion Matrix và Error Analysis.
5. Data report, model report, final report và demo.
6. GRU, BiLSTM, GloVe là phần mở rộng; chỉ triển khai sau khi phần bắt buộc ổn định.

Không tự tạo số liệu hoặc ghi kết quả dự kiến thành kết quả thực nghiệm. Mọi thành viên cần báo sớm nếu gặp blocker và cập nhật GitHub khi bàn giao.
