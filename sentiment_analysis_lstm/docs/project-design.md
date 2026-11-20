# Project Design

Mô tả kiến trúc tổng thể, luồng dữ liệu, module và cách các thành phần kết nối.

Pipeline dự kiến: Download/Inspect → Audit/EDA → Preprocessing → Dataset/DataLoader → Baseline/LSTM → Evaluation → Reporting/Demo.

TensorBoard ghi nhận các chỉ số huấn luyện theo epoch; log cục bộ được lưu trong `runs/`.
