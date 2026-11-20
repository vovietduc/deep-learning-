PRACTICE 3  /  01

BÁO CÁO THỰC HÀNH
Hugging Face



**Chương 4: Fine-tune Models and Training Algorithms**
Phân tích cảm xúc và tinh chỉnh mô hình phân loại văn bản nhị phân.

Sinh viên: ........................................  MSSV: ........................................
Lớp: ................................................  Ngày thực nghiệm: 07/11/2026

1. Mục tiêu và phạm vi

Bài thực hành gồm hai nhiệm vụ. Exercise 1 sử dụng mô hình đã được huấn luyện cho phân tích cảm xúc để tokenize và suy luận trên câu mẫu. Exercise 2 bắt đầu từ backbone pretrained, tạo đầu phân loại nhị phân và huấn luyện trên dữ liệu review phim bằng Hugging Face Trainer.

Source được thiết kế cho Python 3.14.6, chạy trong VS Code trên Windows x64. Lần chạy kiểm chứng được thực hiện bằng Python 3.14.6 trên Linux CPU; hướng dẫn Windows và bộ phụ thuộc Windows được cung cấp riêng. Báo cáo không coi việc phân giải thư viện Windows là kiểm thử thực thi trên Windows.

2. Dữ liệu và nhãn

Dữ liệu được lấy từ cornell-movie-review-data/rotten_tomatoes trên Hugging Face. Mỗi mẫu gồm text (review phim tiếng Anh) và label. Nhãn 0 là NEGATIVE; nhãn 1 là POSITIVE. Tập dữ liệu không có nhãn trung tính.

Tập | Số mẫu | Nhãn 0 | Nhãn 1 | Vai trò
Train | 1000 | 512 | 488 | Cập nhật trọng số
Validation | 200 | 107 | 93 | Chọn checkpoint
Test | 200 | 107 | 93 | Đánh giá cuối


Giữ nguyên các split gốc của dataset, shuffle với seed 42 rồi chọn tập con. Bộ CSV được đóng gói để tái sử dụng đúng dữ liệu. Kiểm tra chuỗi sau strip/lower không phát hiện văn bản trùng giữa ba tập con. Kiểm tra này không chứng minh các câu diễn đạt gần giống hoàn toàn độc lập.

Giới hạn 1.000 mẫu huấn luyện giúp giảm chi phí CPU. Kết quả chỉ đại diện cho tập con đã chọn, không phải benchmark trên toàn bộ Rotten Tomatoes.

PRACTICE 3  /  02

Cơ sở lý thuyết và mô hình



3. Tokenization và biểu diễn đầu vào

Tokenizer chia câu thành token hoặc subword và ánh xạ sang input_ids. Các token đặc biệt đánh dấu biên chuỗi. attention_mask chỉ ra vị trí có nội dung và vị trí padding; với BERT còn có token_type_ids. Văn bản được cắt tối đa 128 token ở Bài 2 và padding động theo câu dài nhất trong từng batch.

4. Mô hình của hai bài

Nội dung | Exercise 1 | Exercise 2
Mô hình | DistilBERT fine-tuned SST-2 | prajjwal1/bert-tiny
Trạng thái ban đầu | Đã có đầu phân tích cảm xúc | Backbone pretrained; đầu 2 lớp mới
Thao tác | Suy luận; không cập nhật trọng số | Huấn luyện toàn bộ trọng số
Tokenizer | Đi kèm mô hình DistilBERT | google-bert/bert-base-uncased
Đầu ra | Nhãn + điểm softmax | 2 logits, metrics và model đã lưu


BERT-Tiny có 2 transformer layer, hidden size 128 và 2 attention head. Mô hình phân loại trong lần chạy này có 4,386,178 tham số. Chọn cấu hình nhỏ nhằm thực hành fine-tuning trên CPU; không kỳ vọng chất lượng ngang mô hình lớn.

Tokenizer BERT-base-uncased dùng bộ từ vựng WordPiece tương ứng với BERT-Tiny. Source chỉ rõ BertConfig vì config gốc của prajjwal1/bert-tiny không có model_type, và dùng tokenizer chuẩn để tránh lỗi nạp tokenizer cũ trên Transformers mới.

5. Fine-tuning và hàm mất mát

Fine-tuning khởi tạo backbone từ trọng số pretrained thay vì ngẫu nhiên. Đầu phân loại 2 lớp mới được khởi tạo rồi học từ nhãn. Không đóng băng backbone trong bài này. Model nhận labels và tự tính cross-entropy trên logits; không cần tự thêm sigmoid hay softmax trước khi tính loss.

Với mẫu i, loss = -log p(y_i | x_i). Trainer thực hiện forward, tính loss, backpropagation và cập nhật tham số. AdamW cập nhật theo gradient và áp dụng weight decay tách khỏi gradient update. Gradient clipping giới hạn norm ở 1.0 trước bước cập nhật.

Điểm softmax biểu thị mức ưu tiên giữa hai lớp của mô hình; không phải xác suất đã được hiệu chỉnh hay cam kết dự đoán chính xác.

PRACTICE 3  /  03

Thiết kế chương trình



6. Quy trình Exercise 1

exercise1.py nạp AutoTokenizer và AutoModelForSequenceClassification; tokenize câu mẫu; chuyển model sang eval(); dùng torch.no_grad() để suy luận; áp dụng softmax, lấy lớp có điểm lớn nhất và xuất JSON. JSON lưu cả tokens, input_ids và attention_mask để thể hiện bước tokenize theo đề.

7. Quy trình Exercise 2

exercise2.py đọc ba CSV bằng datasets, tokenize qua Dataset.map và ghép batch bằng DataCollatorWithPadding. Chương trình khai báo TrainingArguments, tạo Trainer, gọi train(), đánh giá validation và test, rồi lưu mô hình cùng tokenizer. predict.py tải mô hình đã lưu để dự đoán mà không huấn luyện lại.

Tham số | Cấu hình
Epoch / batch size | 5 / 16
Learning rate / optimizer | 0.00005 / AdamW (adamw_torch)
Weight decay / gradient clipping | 0.01 / max_grad_norm = 1.0
Scheduler / warmup | Linear / 10% tổng bước
Độ dài tối đa / seed | 128 token / 42
Checkpoint tốt nhất | F1 validation lớn nhất; đánh giá mỗi epoch
Thiết bị / precision | CPU / float32; không fp16 hoặc bf16
Nạp dữ liệu trên Windows | dataloader_num_workers = 0


Cấu hình mặc định không dùng early stopping; chạy đủ 5 epoch rồi khôi phục checkpoint tốt nhất theo validation F1. Dropout sử dụng cấu hình có sẵn trong BERT. Test không được dùng để chọn checkpoint hoặc điều chỉnh tham số trong lần thực nghiệm.

8. Chỉ số đánh giá và kết quả lưu

Accuracy = số dự đoán đúng / tổng mẫu. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). Lớp dương là POSITIVE (1). Khi mẫu số bằng 0, chương trình trả 0 để tránh lỗi. Các chỉ số được tính bằng NumPy; evaluate được cài đúng yêu cầu thư viện của đề.

outputs/exercise2 chứa metrics.json, training_history.json, test_predictions.csv và model/. Lịch sử huấn luyện và dự đoán từng mẫu giúp đối chiếu số liệu báo cáo. Các checkpoint trung gian không được đóng gói để giảm kích thước; model/ là checkpoint được chọn sau huấn luyện.

PRACTICE 3  /  04

Kết quả thực nghiệm



9. Kết quả Exercise 1

I really enjoyed this movie. The acting was excellent!

**POSITIVE** — điểm softmax 0.999872

This movie was boring and disappointing.

**NEGATIVE** — điểm softmax 0.999784

Ví dụ tokens câu thứ nhất: [CLS] | i | really | enjoyed | this | movie | . | the | acting | was | excellent | ! | [SEP]

10. Kết quả Exercise 2

Chỉ số | Validation | Test
Accuracy | 0.6200 | 0.6300
Precision | 0.6076 | 0.6267
Recall | 0.5161 | 0.5054
F1 | 0.5581 | 0.5595
Loss | 0.6542 | 0.6556


Ma trận nhầm lẫn trên test

Nhãn thực tế / dự đoán | NEGATIVE (0) | POSITIVE (1)
NEGATIVE (0) | 79 | 28
POSITIVE (1) | 46 | 47


Mô hình dự đoán đúng 126/200 mẫu test. Có 28 câu tiêu cực bị dự đoán thành tích cực và 46 câu tích cực bị dự đoán thành tiêu cực. Thời gian train() ghi nhận là 19.0 giây; tổng huấn luyện, đánh giá và lưu kết quả là 19.4 giây trên môi trường kiểm chứng, không bao gồm tải dữ liệu/mô hình ban đầu.

Báo cáo dùng kết quả từ lần chạy thực tế, không phải smoke test. Kết quả đi kèm là một lần chạy với seed cố định; chưa có trung bình và độ lệch chuẩn qua nhiều seed.

PRACTICE 3  /  05

Phân tích và giới hạn



11. Diễn biến validation qua các epoch

Epoch | Val loss | Val accuracy | Val F1
1 | 0.6764 | 0.6050 | 0.5093
2 | 0.6697 | 0.6100 | 0.4583
3 | 0.6609 | 0.6350 | 0.5576
4 | 0.6542 | 0.6200 | 0.5581
5 | 0.6520 | 0.6200 | 0.5529


Biểu đồ: F1 validation (trục dọc) theo epoch (trục ngang).

Checkpoint được chọn ở epoch 4, có validation F1 = 0.5581. Việc chọn dựa trên validation giúp tránh chọn mô hình theo kết quả test. Loss và F1 đo các khía cạnh khác nhau nên có thể không cùng đạt tối ưu ở một epoch.

12. Giới hạn và hướng cải thiện

Tập huấn luyện nhỏ và mô hình nhỏ có thể bỏ sót quan hệ ngữ nghĩa dài, phủ định, mỉa mai hoặc câu chứa cảm xúc hỗn hợp. Nhãn nhị phân buộc mỗi câu thuộc một trong hai lớp, kể cả câu không bộc lộ cảm xúc rõ. Văn bản dài bị cắt ở 128 token có thể mất thông tin.

Các bước mở rộng hợp lý: tăng số mẫu train, thử learning rate/batch size trên validation, chạy nhiều seed và so sánh mô hình lớn hơn bằng cùng một quy trình. Chỉ dùng test để đánh giá cuối cùng. Không suy rộng kết quả này sang tiếng Việt hoặc bài toán thực tế ngoài review phim mà chưa đánh giá thêm.

Exercise 1 chỉ kiểm chứng trên hai câu minh họa; điểm softmax cao ở hai câu không chứng minh accuracy tổng quát. Không so sánh hai điểm đó với accuracy test của Exercise 2.

PRACTICE 3  /  06

Chạy chương trình và đối chiếu đề



13. Chạy trên VS Code / Windows

Giải nén source; File > Open Folder và chọn thư mục chứa setup_env.py. Cài extension Python của Microsoft, mở Terminal > New Terminal rồi chạy:

py -3.14 --version

py -3.14 setup_env.py

.\.venv\Scripts\python.exe check_env.py

.\.venv\Scripts\python.exe src\exercise1.py

.\.venv\Scripts\python.exe src\exercise2.py --cpu

Chọn interpreter .venv/Scripts/python.exe bằng Ctrl+Shift+P > Python: Select Interpreter. Hướng dẫn chi tiết và gỡ lỗi nằm trong README/. Có thể dùng các file CMD để nhấp đúp chạy. Cần Internet cho lần cài đặt và tải mô hình gốc; mô hình đã fine-tune kèm source hỗ trợ predict.py offline sau khi cài thư viện.

14. Đối chiếu yêu cầu

Yêu cầu | Thành phần đáp ứng
Exercise 1: cài thư viện, model, tokenize, sentiment | requirements + setup_env.py + exercise1.py
Exercise 2: dataset, tokenizer, preprocessing | data/ + prepare_data.py + Dataset.map
TrainingArguments, Trainer, fine-tune, evaluate | exercise2.py + outputs/exercise2/
Hướng dẫn và báo cáo trong source | README/ + report/Bao_cao_Practice3.pdf


15. Kiểm chứng và tài liệu tham khảo

Đã chạy hai bài bằng CPython 3.14.6 trên Linux CPU, nạp lại mô hình và kiểm tra các chỉ số. Bộ dependency Windows/Python 3.14 đã được phân giải với wheel; chưa chạy thực tế trên Windows. Xem report/VALIDATION.md.

[1] Chapter 4, Practice 3, trang 25 — Nguyễn Thị Khánh Tiên.

[2] huggingface.co/docs/transformers/en/training

[3] huggingface.co/docs/transformers/en/main_classes/trainer

[4] huggingface.co/prajjwal1/bert-tiny

[5] huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

[6] huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes

[7] pytorch.org/get-started/locally/
