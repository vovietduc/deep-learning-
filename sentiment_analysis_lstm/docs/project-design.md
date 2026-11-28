# Project Design

## 1. Mục tiêu
Phân loại review phim IMDb thành Negative (0) hoặc Positive (1). Mô hình chính theo đề bài là Embedding + LSTM + Dense/Linear; các extension gồm GRU, Bidirectional LSTM và GloVe embeddings.

## 2. Luồng xử lý
1. Quốc Anh: download, inspect, data audit và EDA.
2. Duy Căn: text normalization, tokenization, vocabulary, encoding, padding và Dataset adapter.
3. Thành Phát: LSTM chính, training loop, checkpoint và history.
4. Minh Khang: TF-IDF + Logistic Regression, GRU, BiLSTM và GRU khởi tạo bằng GloVe.
5. Đặng Đức: metrics, confusion matrix, visualization và error analysis.
6. Đình Phú: demo và HTML/PDF reporting.
7. Võ Việt Đức: shared config, seed, device, orchestration, TensorBoard convention, Git review và integration.

## 3. Data contract cần áp dụng chung
- Raw dataset: data/raw/aclImdb/.
- Supervised data: 25,000 train reviews và 25,000 test reviews; train/unsup không dùng cho supervised classification.
- Label mapping: neg = 0, pos = 1.
- Validation: lấy stratified 20% từ train với seed 42; full split là 20,000 train, 5,000 validation, 25,000 test.
- Vocabulary chỉ fit trên train split; validation/test chỉ transform.
- Test set không dùng để chọn checkpoint hoặc điều chỉnh hyperparameters.
- PAD ID = 0, UNK ID = 1 là convention của module preprocessing Khang; phải xác nhận với Duy Căn trước khi merge chung tokenizer/vocabulary.

## 4. Configuration và runtime
- Cấu hình mặc định: configs/default.yaml.
- ProjectConfig là typed config; load_config trả nested dictionary để tương thích các script data hiện có.
- Seed mặc định 42 cho Python, NumPy, PyTorch CPU và CUDA.
- runtime.device = auto ưu tiên CUDA; có thể đặt cpu hoặc cuda.
- RTX 3050 được dùng khi PyTorch CUDA khả dụng. AMP mặc định tắt; chỉ bật sau khi kiểm tra training loop hỗ trợ autocast/GradScaler.
- Data path: data/raw/aclImdb; processed artifacts: data/processed; reports: reports; checkpoints: models; TensorBoard: runs.

## 5. TensorBoard contract
Training loop gọi create_summary_writer() một lần khi bắt đầu run; sau mỗi epoch gọi log_epoch_metrics() với các giá trị số, ví dụ:
- {"loss/train": train_loss, "loss/val": val_loss}
- {"accuracy/train": train_accuracy, "accuracy/val": val_accuracy}

Kết thúc run phải gọi close_writer() để flush event file. History CSV/JSON vẫn được lưu riêng; TensorBoard không thay thế artifact báo cáo.

## 6. Entry points
Từ thư mục sentiment_analysis_lstm:
- python -m scripts.run_project --stage check
- python -m scripts.run_project --stage data
- python -m scripts.run_project --stage train
- python -m scripts.run_project --stage evaluate
- python -m scripts.run_project --stage all
- tensorboard --logdir runs

run_project.py chỉ điều phối module do từng thành viên sở hữu. Stage thiếu module hoặc còn TODO sẽ báo lỗi rõ ràng; không giả lập rằng pipeline end-to-end đã hoàn tất.
