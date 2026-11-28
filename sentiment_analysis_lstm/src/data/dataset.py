"""Dataset IMDb: nối preprocessing với Embedding + LSTM.

Luồng đúng:
    1. collect_samples(train_dir)      -> 25.000 (path, label)
    2. split_train_val(samples, seed)  -> 20.000 train + 5.000 validation
    3. build_vocabulary(train_samples) -> CHỈ dùng 20.000 train (không dùng validation/test)
    4. IMDBDataset(train / val / test, cùng tokenizer + vocabulary)

Đầu ra của dataset:
    return_lengths=False (mặc định): (input_ids [max_length], label)
    return_lengths=True            : (input_ids [max_length], length, label)
"""

import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .padding import effective_length, pad_or_truncate
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary

Sample = Tuple[str, int]  # (đường dẫn file review, nhãn)
PathLike = Union[str, Path]

LABEL_MAP: Dict[str, int] = {"neg": 0, "pos": 1}


# --------------------------------------------------------------------------
# Đọc dữ liệu / chia split
# --------------------------------------------------------------------------
def read_text(path: PathLike) -> str:
    """Đọc 1 review (UTF-8; byte lỗi được thay bằng ký tự thay thế, không bị mất lặng lẽ)."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def collect_samples(split_dir: PathLike, label_map: Optional[Dict[str, int]] = None) -> List[Sample]:
    """Liệt kê (path, label) trong split_dir/{neg,pos}/*.txt, theo thứ tự cố định."""
    label_map = label_map or LABEL_MAP
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    samples: List[Sample] = []
    for label_name, label_id in label_map.items():
        label_dir = split_dir / label_name
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing directory: {label_dir}")
        for file_path in sorted(label_dir.glob("*.txt")):
            samples.append((str(file_path), label_id))
    return samples


def split_train_val(
    samples: Sequence[Sample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Chia train/validation có phân tầng theo nhãn, tái lập được bằng seed.

    Ví dụ 25.000 review, val_ratio=0.2 -> 20.000 train + 5.000 validation.
    Hai tập không giao nhau.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    rng = random.Random(seed)
    by_label: Dict[int, List[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    train: List[Sample] = []
    val: List[Sample] = []
    for label in sorted(by_label):
        items = sorted(by_label[label])  # thứ tự cố định trước khi xáo
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    train.sort()
    val.sort()
    return train, val


def save_split_manifest(
    train: Sequence[Sample],
    val: Sequence[Sample],
    path: PathLike,
    seed: int,
    val_ratio: float,
    root: Optional[PathLike] = None,
) -> None:
    """Lưu danh sách file của train/val (manifest) để tái lập chính xác."""

    def rel(p: str) -> str:
        if root is None:
            return Path(p).as_posix()
        try:
            return Path(p).relative_to(root).as_posix()
        except ValueError:
            return Path(p).as_posix()

    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "n_train": len(train),
        "n_val": len(val),
        "train": [[rel(p), y] for p, y in train],
        "val": [[rel(p), y] for p, y in val],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# Vocabulary / encode dùng chung cho train, evaluate và demo
# --------------------------------------------------------------------------
def iter_token_sequences(samples: Iterable[Sample], tokenizer: Tokenizer) -> Iterator[List[str]]:
    for path, _ in samples:
        yield tokenizer.tokenize(read_text(path))


def build_vocabulary(
    train_samples: Iterable[Sample],
    tokenizer: Tokenizer,
    min_freq: int = 2,
    max_size: Optional[int] = None,
) -> Vocabulary:
    """Xây vocabulary từ CHỈ các sample train.

    min_freq=2 (mặc định) bỏ từ chỉ xuất hiện 1 lần: giảm kích thước embedding
    và để <UNK> có dữ liệu thật trong lúc train.
    """
    vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    return vocab.build(iter_token_sequences(train_samples, tokenizer))


def encode_text(
    text: Optional[str],
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
    max_length: int,
    padding: str = "post",
    truncating: str = "post",
) -> Tuple[List[int], int]:
    """Review thô -> (input_ids đã pad, độ dài thật).

    Hàm này là nguồn duy nhất cho tiền xử lý: dataset (train/val/test) và demo
    (predict_service) đều gọi nó để đảm bảo giống nhau.
    """
    ids = vocabulary.encode(tokenizer.tokenize(text))
    return _finalize(ids, vocabulary, max_length, padding, truncating)


def _finalize(
    ids: List[int],
    vocabulary: Vocabulary,
    max_length: int,
    padding: str,
    truncating: str,
) -> Tuple[List[int], int]:
    length = effective_length(len(ids), max_length)
    padded = pad_or_truncate(
        ids, max_length, pad_id=vocabulary.PAD_ID, padding=padding, truncating=truncating
    )
    return padded, length


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class IMDBDataset(Dataset):
    """Dataset IMDb.

    Truyền MỘT trong hai:
        split_dir: thư mục chứa neg/ và pos/ (vd. aclImdb/test)
        samples  : list (path, label), vd. kết quả của split_train_val()

    Nhãn: neg -> 0, pos -> 1.
    """

    LABEL_MAP = LABEL_MAP

    def __init__(
        self,
        tokenizer: Tokenizer,
        vocabulary: Vocabulary,
        max_length: int,
        split_dir: Optional[PathLike] = None,
        samples: Optional[Sequence[Sample]] = None,
        padding: str = "post",
        truncating: str = "post",
        return_lengths: bool = False,
        cache: bool = True,
    ):
        if (split_dir is None) == (samples is None):
            raise ValueError("Provide exactly one of split_dir or samples")
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating
        self.return_lengths = return_lengths
        self.cache = cache

        self.split_dir = Path(split_dir) if split_dir is not None else None
        self.samples: List[Sample] = (
            collect_samples(split_dir) if split_dir is not None else list(samples)
        )
        # Cache token ID (chưa pad) để epoch sau không phải đọc file + tokenize lại
        self._cache: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _token_ids(self, index: int) -> List[int]:
        if self.cache and index in self._cache:
            return self._cache[index]
        path, _ = self.samples[index]
        ids = self.vocabulary.encode(self.tokenizer.tokenize(read_text(path)))
        if self.cache:
            self._cache[index] = ids
        return ids

    def __getitem__(self, index: int):
        _, label = self.samples[index]
        padded, length = _finalize(
            self._token_ids(index), self.vocabulary, self.max_length, self.padding, self.truncating
        )

        input_ids = torch.tensor(padded, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_lengths:
            return input_ids, torch.tensor(length, dtype=torch.long), label_tensor
        return input_ids, label_tensor


def create_dataloader(
    dataset: IMDBDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    """Tạo DataLoader. Tập train nên shuffle=True (seed để tái lập thứ tự xáo)."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


# --------------------------------------------------------------------------
# Thống kê và cấu hình (phục vụ báo cáo dữ liệu / tái lập)
# --------------------------------------------------------------------------
def preprocessing_stats(
    samples: Iterable[Sample],
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
    max_length: int,
) -> Dict[str, Any]:
    """Thống kê sau tiền xử lý: OOV rate, tỷ lệ truncate/pad, độ dài review."""
    lengths: List[int] = []
    total_tokens = 0
    unk_tokens = 0
    truncated = 0
    empty = 0

    for path, _ in samples:
        ids = vocabulary.encode(tokenizer.tokenize(read_text(path)))
        lengths.append(len(ids))
        total_tokens += len(ids)
        unk_tokens += sum(1 for i in ids if i == vocabulary.UNK_ID)
        truncated += len(ids) > max_length
        empty += len(ids) == 0

    n = len(lengths)
    if n == 0:
        return {"num_reviews": 0}

    padded = sum(1 for length in lengths if length < max_length)
    return {
        "num_reviews": n,
        "vocab_size": len(vocabulary),
        "max_length": max_length,
        "empty_reviews": empty,
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": round(statistics.fmean(lengths), 2),
        "length_median": statistics.median(lengths),
        "oov_rate": round(unk_tokens / total_tokens, 6) if total_tokens else 0.0,
        "truncated_rate": round(truncated / n, 6),
        "padded_rate": round(padded / n, 6),
    }


def save_preprocessing_config(
    path: PathLike,
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
    max_length: int,
    seed: Optional[int] = None,
    padding: str = "post",
    truncating: str = "post",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Lưu cấu hình tiền xử lý để Thành Phát/Đình Phú dùng lại đúng khi train/demo."""
    config: Dict[str, Any] = {
        "tokenizer": tokenizer.get_config(),
        "vocab": {
            "size": len(vocabulary),
            "min_freq": vocabulary.min_freq,
            "max_size": vocabulary.max_size,
            "pad_token": vocabulary.PAD_TOKEN,
            "pad_id": vocabulary.PAD_ID,
            "unk_token": vocabulary.UNK_TOKEN,
            "unk_id": vocabulary.UNK_ID,
        },
        "max_length": max_length,
        "padding": padding,
        "truncating": truncating,
        "label_map": LABEL_MAP,
        "seed": seed,
    }
    if extra:
        config.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
