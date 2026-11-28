# Sentiment Analysis with LSTM

Bài tập lớn phân loại cảm xúc tích cực/tiêu cực trong IMDb movie reviews.

- Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- Model chính: Embedding + LSTM + Dense (Linear).
- Mở rộng: GRU, Bidirectional LSTM, GloVe embeddings.
- Monitoring: TensorBoard, log lưu cục bộ trong runs/.

## Cấu trúc
- configs/: cấu hình dùng chung.
- data/raw/: IMDb gốc (không commit).
- data/interim/ và data/processed/: dữ liệu trung gian/đã mã hóa.
- src/data/: tokenizer, vocabulary, padding và Dataset dùng chung.
- src/models/ và src/training/: kiến trúc và huấn luyện.
- src/evaluation/ và src/reporting/: đánh giá và báo cáo.
- src/demo/: demo dự đoán review.
- src/utils/: config/runtime/seed/TensorBoard helpers.
- scripts/: runner và các lệnh hỗ trợ.
- tests/: unit/integration tests.
- models/: checkpoint (không commit checkpoint lớn).
- runs/: TensorBoard event logs (không commit).
- reports/: kết quả audit, metrics và báo cáo.

## Môi trường

Khuyến nghị Python 3.12+ phù hợp với phiên bản PyTorch được nhóm kiểm thử. Trên máy Windows có RTX 3050, cài PyTorch CUDA 12.8 trước:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu máy không có GPU NVIDIA, cài PyTorch theo hướng dẫn chính thức cho CPU tại https://pytorch.org/get-started/locally/ rồi cài requirements còn lại.

## Chạy project

Chạy tại thư mục sentiment_analysis_lstm:

```powershell
# Kiểm tra config, seed và thiết bị
python -m scripts.run_project --stage check

# Tải, inspect, preprocess, audit và EDA (sau khi module Quốc Anh được tích hợp)
python -m scripts.run_project --stage data

# Sau khi module training/evaluation được bàn giao
python -m scripts.run_project --stage train
python -m scripts.run_project --stage evaluate
python -m scripts.run_project --stage all

# TensorBoard
tensorboard --logdir runs
```

Xuất split manifest chung sang định dạng CSV cho runner extension của Minh Khang:

```powershell
python -m scripts.export_khang_manifest --manifest data/processed/split_manifest.json --output reports/khang_split.csv
```

## Quy ước dữ liệu
- Negative = 0; Positive = 1.
- Seed mặc định = 42.
- Validation lấy stratified 20% từ tập train: 20,000 train / 5,000 validation / 25,000 test trên full IMDb.
- Vocabulary chỉ fit trên train; validation/test chỉ transform.
- Các model phải dùng cùng split, tokenizer, vocabulary và max sequence length khi so sánh.

## Git và artifact
Dataset gốc, file GloVe, checkpoint lớn, TensorBoard event logs và secrets không commit lên GitHub. Chỉ cập nhật số liệu báo cáo sau khi có lần chạy thực tế và lưu artifact có thể kiểm chứng.

## Thành viên
Võ Việt Đức (nhóm trưởng), Quốc Anh, Duy Căn, Thành Phát, Minh Khang, Đặng Đức, Đình Phú.
