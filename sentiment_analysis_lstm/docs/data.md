<!--
PHẦN BỔ SUNG CHO docs/data.md (Duy Căn: tiền xử lý văn bản).
Dán vào docs/data.md hiện có, ngay sau phần Download/Inspect/Audit/EDA của Quốc Anh.
-->

# Text Preprocessing, Tokenization & Vocabulary

Người phụ trách: Duy Căn. Code: `src/data/tokenizer.py`, `vocabulary.py`, `padding.py`, `dataset.py`.
Test: `tests/test_preprocessing.py`. Demo: `notebooks/preprocessing_demo.ipynb`.

## 1. Pipeline

```text
Raw review
   ↓ Tokenizer        (src/data/tokenizer.py)
Tokens
   ↓ Vocabulary       (src/data/vocabulary.py)
Token IDs
   ↓ Padding/Truncate (src/data/padding.py)
Padded sequence
   ↓ IMDBDataset      (src/data/dataset.py)
Tensor input_ids + label
   ↓
Embedding → LSTM → Linear → logit
```

## 2. Dataset và nhãn

IMDb Large Movie Review Dataset: 25.000 review train + 25.000 review test (mỗi tập 12.500 pos / 12.500 neg).
Thư mục `unsup` (50.000 review không nhãn) **không** dùng cho mô hình supervised.

| Thư mục | Nhãn |
|---|---:|
| `neg/` | 0 |
| `pos/` | 1 |

## 3. Chia dữ liệu (train / validation / test)

- Validation được tách **từ tập train có nhãn**: 20.000 train + 5.000 validation (`split_train_val`, `val_ratio=0.2`).
- Tách có phân tầng theo nhãn (validation vẫn cân bằng pos/neg), tái lập được bằng `seed`.
- Tập test 25.000 giữ nguyên, chỉ dùng ở bước đánh giá cuối.
- Danh sách file mỗi tập lưu bằng `save_split_manifest` để tái lập.

## 4. Tokenization

`Tokenizer.normalize` rồi `Tokenizer.tokenize`:

1. `None`, chuỗi rỗng, chuỗi chỉ có khoảng trắng → `[]`.
2. Giải mã HTML entity (`&amp;` → `&`).
3. Chuẩn hoá Unicode (NFKC) và dấu nháy cong `’` → `'`.
4. Bỏ thẻ HTML (`<br />`, `</i>`…) và thay bằng khoảng trắng; `a < b > c` không bị coi là thẻ.
5. Chuẩn hoá khoảng trắng, chuyển chữ thường.
6. Tách token: từ/số (giữ `don't`, `wasn't` là một token) và dấu `!`, `?`.

Ví dụ: `"I LOVE this movie!<br />"` → `["i", "love", "this", "movie", "!"]`

Lý do giữ `!` và `?`: chúng mang tín hiệu cảm xúc (“great!!!”, “what a waste?”). Có thể tắt bằng
`Tokenizer(keep_punctuation=False)`; tham số này cần được nhóm thống nhất một lần và dùng nhất quán cho
train, evaluate và demo (cấu hình được lưu bằng `save_preprocessing_config`).

## 5. Vocabulary

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| token khác | 2, 3, … (tần suất giảm dần; hoà thì theo thứ tự chữ cái) |

- **Chỉ build từ tập train (20.000 review sau khi đã tách validation)**. Validation/test dùng lại
  vocabulary đã build, không bao giờ fit lại. Nếu build từ cả 25.000 review train rồi mới tách,
  validation đã rò rỉ vào vocabulary.
- `min_freq=2` (mặc định của `build_vocabulary`) bỏ từ chỉ xuất hiện 1 lần; `max_size` giới hạn kích thước.
- Kết quả không phụ thuộc thứ tự review đầu vào (deterministic).
- Lưu/nạp: `vocab.save("data/processed/vocab.json")`, `Vocabulary.load(...)`.

## 6. Token ngoài vocabulary

Token xuất hiện ở validation/test nhưng không có trong vocabulary → `<UNK>` → ID 1.
Tỉ lệ này (OOV rate) được báo cáo bằng `preprocessing_stats`.

## 7. Padding và truncation

Mặc định `padding="post"`, `truncating="post"`:

- Ngắn hơn `max_length`: thêm `<PAD>` (ID 0) vào cuối. `[2, 5, 8]`, `max_length=6` → `[2, 5, 8, 0, 0, 0]`.
- Dài hơn `max_length`: cắt phần cuối. `[1, 2, 3, 4, 5, 6]`, `max_length=4` → `[1, 2, 3, 4]`.
- Có thể đổi sang `"pre"` cho cả hai.

`max_length` là hyperparameter của nhóm, **không** phải giá trị bắt buộc của IMDb; chốt dựa trên EDA độ dài
review của Quốc Anh (xem `truncated_rate` và `padded_rate` trong `preprocessing_stats`).

**Độ dài thật:** với `return_lengths=True`, dataset trả thêm độ dài thật (không tính PAD, tối thiểu 1 để
`pack_padded_sequence` không lỗi với review rỗng).

## 8. Dataset output

`IMDBDataset(tokenizer, vocabulary, max_length, split_dir=... | samples=...)`

| `return_lengths` | Mỗi phần tử | Batch từ DataLoader |
|---|---|---|
| `False` (mặc định) | `(input_ids, label)` | `input_ids`: `(batch, max_length)` `long`; `label`: `(batch,)` `long` |
| `True` | `(input_ids, length, label)` | thêm `length`: `(batch,)` `long` |

Dataset cache token ID (chưa pad) trong bộ nhớ nên epoch thứ hai trở đi không đọc lại file.
Tập train nên dùng `create_dataloader(..., shuffle=True, seed=...)`.

## 9. Dùng chung cho demo

`encode_text(text, tokenizer, vocabulary, max_length)` trả `(input_ids, length)` và dùng đúng hàm xử lý như
dataset. `src/demo/predict_service.py` nên gọi hàm này (cùng tokenizer + vocabulary lúc train) và tự xử lý
trường hợp input rỗng.

## 10. Quy trình sử dụng

```python
from src.data.tokenizer import Tokenizer
from src.data.dataset import (collect_samples, split_train_val, build_vocabulary,
                              IMDBDataset, save_preprocessing_config)

tokenizer = Tokenizer()
train_all = collect_samples("data/raw/aclImdb/train")
train_s, val_s = split_train_val(train_all, val_ratio=0.2, seed=42)
vocab = build_vocabulary(train_s, tokenizer, min_freq=2)      # CHỈ train

train_ds = IMDBDataset(tokenizer, vocab, max_length=200, samples=train_s)
val_ds   = IMDBDataset(tokenizer, vocab, max_length=200, samples=val_s)
test_ds  = IMDBDataset(tokenizer, vocab, max_length=200, split_dir="data/raw/aclImdb/test")

vocab.save("data/processed/vocab.json")
save_preprocessing_config("data/processed/preprocessing_config.json", tokenizer, vocab,
                          max_length=200, seed=42)
```

## 11. Hạn chế

Tokenizer theo regex, không có sửa lỗi chính tả hay stemming; từ hiếm thành `<UNK>`; mỉa mai và phủ định
phức tạp vẫn là thách thức cho mô hình. Phần truncation mặc định giữ phần đầu review, trong khi kết luận
của review IMDb thường ở cuối (có thể thử `truncating="pre"` như một thí nghiệm có kiểm soát).
