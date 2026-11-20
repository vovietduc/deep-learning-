# Practice 3 - Get started with Hugging Face

Source hoàn chỉnh cho cả Exercise 1 và Exercise 2 của Chapter 4.

## Mở đúng thư mục

Giải nén ZIP trước. Trong VS Code, chọn File > Open Folder > chọn `Practice3_HuggingFace` (thư mục chứa `setup_env.py`). Không chạy trực tiếp từ cửa sổ xem ZIP.

**3.14.6 là phiên bản Python, không phải phiên bản VS Code.** Dự án nhắm tới CPython 3.14.6 64-bit tiêu chuẩn trên Windows x64. Không cần GPU. Cần Internet để cài thư viện và tải mô hình gốc ở lần chạy đầu. Mô hình đã fine-tune kèm trong ZIP dùng được offline sau khi cài thư viện.

## Cách chạy trong VS Code trên Windows

1. Cài extension **Python** của Microsoft trong VS Code.
2. Mở **Terminal > New Terminal**. Các lệnh dưới đây dùng PowerShell.
3. Kiểm tra Python:

```powershell
py -3.14 --version
```

Kết quả mục tiêu: `Python 3.14.6`. Nếu `py` không được nhận diện, cài Python 3.14 bản 64-bit cùng Python Launcher, rồi mở lại VS Code. Nếu máy có `python` đúng phiên bản, có thể dùng `python setup_env.py` thay cho lệnh tiếp theo.

4. Cài môi trường và thư viện (chỉ cần làm lần đầu):

```powershell
py -3.14 setup_env.py
```

Đợi đến dòng `SETUP COMPLETE`. Script tạo `.venv`, cài PyTorch CPU, rồi cài các thư viện đã ghim phiên bản. Không đóng cửa sổ khi đang tải. Nếu cài lỗi, sửa kết nối rồi chạy lại lệnh trên.

5. Nhấn **Ctrl+Shift+P > Python: Select Interpreter > Enter interpreter path > Find** và chọn `.venv\Scripts\python.exe` trong thư mục dự án.
6. Chạy kiểm tra:

```powershell
.\.venv\Scripts\python.exe check_env.py
```

Phải có dòng `Trainer imports: OK`. Không cần kích hoạt môi trường bằng `Activate.ps1`; gọi trực tiếp Python như trên tránh lỗi ExecutionPolicy.

7. Chạy **Bài 1**:

```powershell
.\.venv\Scripts\python.exe src\exercise1.py
```

Thử câu khác bằng tiếng Anh:

```powershell
.\.venv\Scripts\python.exe src\exercise1.py --text "I love this movie!" "The plot is boring."
```

Kết quả có nhãn NEGATIVE/POSITIVE, điểm softmax, tokens, input_ids và attention_mask. Xem `outputs/exercise1/results.json`.

8. Chạy **Bài 2** (fine-tune thật, 5 epoch, CPU):

```powershell
.\.venv\Scripts\python.exe src\exercise2.py --cpu
```

Đợi hết tiến trình đến `TEST RESULTS`. Thời gian phụ thuộc CPU, RAM và tốc độ tải. Đừng kết luận bị treo khi lần đầu đang tải mô hình. Dữ liệu CSV đã kèm source nên không cần tải lại dataset.

9. Dự đoán bằng mô hình đã fine-tune:

```powershell
.\.venv\Scripts\python.exe src\predict.py --text "The acting was excellent and the story was touching."
```

10. Mở `report/Bao_cao_Practice3.pdf`. Báo cáo ghi lần chạy kiểm chứng đi kèm. Khi bạn huấn luyện lại, file metrics được cập nhật nhưng PDF không tự cập nhật.

## Cách chạy bằng nhấp đúp

Trong File Explorer, sau khi giải nén:
- `01_SETUP_WINDOWS.cmd`: tạo môi trường và cài thư viện.
- `02_RUN_EXERCISE1.cmd`: chạy bài 1.
- `03_RUN_EXERCISE2.cmd`: huấn luyện và đánh giá bài 2.
- `04_CHECK_ENV.cmd`: kiểm tra Python và thư viện.

Các cửa sổ giữ lại kết quả bằng `pause`. Nếu VS Code mở file CMD dưới dạng văn bản thì đó là bình thường; chạy từ File Explorer hoặc gõ `.\02_RUN_EXERCISE1.cmd` trong terminal.

## Cấu trúc source

| Đường dẫn | Vai trò |
|---|---|
| `README.md` | Hướng dẫn chính |
| `README/HUONG_DAN_VSCODE.md` | Bản hướng dẫn trong thư mục README theo yêu cầu |
| `README/GO_LOI.md` | Lỗi thường gặp và cách sửa |
| `src/exercise1.py` | Tokenize và suy luận sentiment bằng mô hình sẵn có |
| `src/exercise2.py` | Huấn luyện, chọn checkpoint theo validation F1, đánh giá test |
| `src/prepare_data.py` | Tải dataset gốc và tạo lại các CSV |
| `src/predict.py` | Suy luận offline với mô hình đã fine-tune |
| `src/common.py` | Seed, đường dẫn, lưu JSON, tính metrics |
| `data/` | 1.000 train, 200 validation, 200 test và thông tin nguồn |
| `outputs/exercise1/` | Kết quả tokenize và sentiment |
| `outputs/exercise2/` | Mô hình, metrics, dự đoán test và lịch sử huấn luyện |
| `report/` | Báo cáo PDF, bản Markdown và kết quả kiểm chứng |
| `requirements.txt` | Phiên bản các thư viện trực tiếp |
| `requirements-windows.lock` | Bộ phiên bản phụ thuộc đã phân giải cho Windows/Python 3.14 |

## Dữ liệu và phương pháp

- Dữ liệu: `cornell-movie-review-data/rotten_tomatoes`, văn bản review phim tiếng Anh.
- Nhãn: 0 = NEGATIVE; 1 = POSITIVE. Không có lớp trung tính.
- Giữ các split train/validation/test gốc; shuffle seed 42 và lấy tập con để máy cá nhân chạy nhanh hơn.
- Bài 1: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, chỉ suy luận, không huấn luyện.
- Tokenizer Bài 2: `google-bert/bert-base-uncased` (cùng từ vựng WordPiece với BERT-Tiny); cấu hình dùng `BertConfig` để tương thích config gốc thiếu model_type.
- Bài 2: `prajjwal1/bert-tiny`, backbone pretrained và đầu phân loại 2 lớp mới; cập nhật toàn bộ trọng số bằng Trainer.
- Bài 2 dùng cross-entropy do model tính từ labels, AdamW, learning rate 5e-5, batch size 16, 5 epoch, weight decay 0.01, warmup_steps = ceil(10% tổng bước), linear scheduler, gradient clipping 1.0.
- Dùng validation F1 chọn checkpoint tốt nhất; test chỉ đánh giá sau khi huấn luyện. Không dùng test để chọn mô hình.
- `evaluate` được cài theo đề. Các công thức accuracy/precision/recall/F1 được tính trực tiếp bằng NumPy để không cần tải metric script từ mạng.
- Không có early stopping trong cấu hình mặc định. Các ví dụ regularization trong slide là kiến thức liên quan, không phải tất cả đều bắt buộc trong Practice 3.

## Kiểm tra nhanh và tùy chỉnh

Smoke test chỉ kiểm tra luồng chạy, không phải kết quả để báo cáo độ chính xác:

```powershell
.\.venv\Scripts\python.exe src\exercise2.py --cpu --smoke --output-dir outputs\smoke
```

Tăng số epoch, lưu riêng để giữ kết quả mẫu:

```powershell
.\.venv\Scripts\python.exe src\exercise2.py --cpu --epochs 8 --output-dir outputs\my_run
.\.venv\Scripts\python.exe src\predict.py --model-dir outputs\my_run\model --text "A wonderful movie."
```

Tạo lại tập con từ Hugging Face (ghi đè CSV hiện có; chỉ làm khi cần):

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py --train-size 1000 --validation-size 200 --test-size 200
```

Không so trực tiếp accuracy Bài 1 và Bài 2 để kết luận mô hình nào tốt hơn: hai bài dùng mô hình, quy trình và dữ liệu huấn luyện khác nhau. Kết quả không thể bảo đảm đúng cho mọi câu, tiếng Việt hay ngôn ngữ mỉa mai.

## Nguồn

- Đề bài: Chapter 4, Practice 3, trang 25, Nguyễn Thị Khánh Tiên.
- https://huggingface.co/docs/transformers/en/training
- https://huggingface.co/docs/transformers/en/main_classes/trainer
- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- https://huggingface.co/prajjwal1/bert-tiny
- https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
- https://pytorch.org/get-started/locally/

Xem `report/VALIDATION.md` để biết chính xác môi trường và phạm vi kiểm chứng.
