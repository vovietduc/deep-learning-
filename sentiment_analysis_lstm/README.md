# Sentiment Analysis with LSTM

Bài tập lớn phân loại cảm xúc tích cực/tiêu cực trong IMDb movie reviews.

- Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- Model chính: Embedding + LSTM + Dense (Linear).
- Mở rộng: GRU, Bidirectional LSTM, GloVe embeddings.
- Monitoring: TensorBoard, log lưu cục bộ trong `runs/`.

## Cấu trúc
`docs/` tài liệu; `configs/` cấu hình; `data/` dữ liệu; `notebooks/` EDA; `src/` mã nguồn; `scripts/` lệnh chạy; `tests/` kiểm thử; `models/` checkpoint; `runs/` TensorBoard; `reports/` báo cáo.

## Cài đặt
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dataset, checkpoint và TensorBoard event logs không commit lên GitHub. Các kết quả thực nghiệm chỉ cập nhật sau khi chạy thực tế.

## Thành viên
Võ Việt Đức (nhóm trưởng), Quốc Anh, Duy Căn, Thành Phát, Minh Khang, Đặng Đức, Đình Phú.
