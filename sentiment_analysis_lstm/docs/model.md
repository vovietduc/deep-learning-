<!--
PHẦN BỔ SUNG CHO docs/model.md (Duy Căn: phần Embedding).
Dán vào docs/model.md hiện có, đặt trước phần LSTM của Thành Phát.
-->

# Embedding

## 1. Vai trò

Sau tiền xử lý, mỗi review là một chuỗi token ID, ví dụ `[12, 45, 78, 9, 31, 0, 0, …]`.
ID chỉ là chỉ số, không mang ý nghĩa ngữ nghĩa (ID 45 không “gần” ID 46). `nn.Embedding` là bảng tra
`(vocab_size, embedding_dim)`: mỗi ID được thay bằng một vector thực có số chiều cố định, và các vector này
được học trong lúc training để phù hợp với bài toán phân loại cảm xúc.

```text
Token ID → Embedding Layer → vector → LSTM
```

## 2. Tensor shape

| Bước | Shape |
|---|---|
| `input_ids` (đầu ra DataLoader) | `(batch_size, max_length)` — `torch.long` |
| Sau Embedding | `(batch_size, max_length, embedding_dim)` — `torch.float` |
| `lengths` (tuỳ chọn) | `(batch_size,)` — `torch.long` |

Ví dụ `batch_size=32`, `max_length=200`, `embedding_dim=128`: `(32, 200)` → `(32, 200, 128)`.

## 3. Quy ước giữa preprocessing và model

Các điểm sau cần thống nhất với Thành Phát (`src/models/lstm.py`) trước khi viết model:

1. `vocab_size = len(vocabulary)` (đã gồm `<PAD>` và `<UNK>`).
2. `nn.Embedding(vocab_size, embedding_dim, padding_idx=0)`: vector của `<PAD>` luôn bằng 0 và không được cập nhật gradient.
3. Dataset dùng **padding ở cuối** (`post`). Nếu model lấy hidden state cuối cùng (`h_n`) của LSTM mà không
   xử lý padding, LSTM sẽ đọc hàng loạt `<PAD>` sau phần review thật và trạng thái cuối bị “loãng”.
   Có hai cách xử lý đúng:
   - Dùng `lengths`: tạo dataset với `return_lengths=True`, rồi
     `pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)`
     trước khi đưa vào LSTM (khuyến nghị); hoặc
   - Dùng `padding="pre"` để review thật nằm sát cuối chuỗi.
4. Interface của dataset (xem `docs/data.md`, mục 8) cố định: không đổi thứ tự `(input_ids, [length,] label)`
   sau khi model bắt đầu dùng.
5. Nhãn là `torch.long`; nếu dùng `BCEWithLogitsLoss` thì ép `label.float()` trong training loop.

## 4. Embedding khởi tạo

Mặc định khởi tạo ngẫu nhiên và học trong training. Phần mở rộng GloVe (Minh Khang) khởi tạo ma trận
embedding từ vector pretrained theo vocabulary này (`<PAD>` = vector 0); cần ghi rõ freeze hay fine-tune.

## 5. Tham số

Số tham số của lớp Embedding = `vocab_size × embedding_dim`. Vocabulary lớn làm tăng đáng kể số tham số,
nên `min_freq` và `max_size` ảnh hưởng trực tiếp đến kích thước mô hình.
