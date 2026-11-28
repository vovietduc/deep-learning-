"""Vocabulary: ánh xạ token <-> ID.

Special tokens:
    <PAD> -> 0   (độ dài padding; truyền padding_idx=0 cho nn.Embedding)
    <UNK> -> 1   (token ngoài vocabulary)

QUAN TRỌNG: chỉ gọi build() bằng dữ liệu TRAIN (sau khi đã tách validation).
Validation/test dùng lại vocabulary đã build, không được fit lại.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None):
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        if max_size is not None and max_size < 2:
            raise ValueError("max_size must be >= 2 (to fit <PAD> and <UNK>)")

        self.min_freq = min_freq
        self.max_size = max_size  # tổng kích thước tối đa, đã gồm <PAD>, <UNK>
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._reset()

    # ------------------------------------------------------------------ build
    def _reset(self) -> None:
        self.token_to_id = {self.PAD_TOKEN: self.PAD_ID, self.UNK_TOKEN: self.UNK_ID}
        self.id_to_token = {self.PAD_ID: self.PAD_TOKEN, self.UNK_ID: self.UNK_TOKEN}

    def build(self, token_sequences: Iterable[List[str]]) -> "Vocabulary":
        """Xây vocabulary từ các chuỗi token của tập TRAIN.

        Gọi lại build() sẽ xây lại từ đầu (không cộng dồn).
        Thứ tự ID: tần suất giảm dần, hoà thì theo thứ tự chữ cái,
        nên kết quả không phụ thuộc thứ tự review đầu vào.
        """
        self._reset()

        counter: Counter = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

        next_id = len(self.token_to_id)
        for token, freq in ordered:
            if freq < self.min_freq:
                break  # đã sắp giảm dần, phần còn lại đều thấp hơn
            if self.max_size is not None and next_id >= self.max_size:
                break
            if token in self.token_to_id:  # trùng với <PAD>/<UNK>
                continue
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1
        return self

    # ------------------------------------------------------- encode / decode
    def encode(self, tokens: List[str]) -> List[int]:
        """tokens -> IDs; token lạ thành UNK_ID."""
        unk = self.UNK_ID
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: List[int], skip_pad: bool = False) -> List[str]:
        """IDs -> tokens; ID lạ thành <UNK>."""
        out = []
        for i in ids:
            i = int(i)
            if skip_pad and i == self.PAD_ID:
                continue
            out.append(self.id_to_token.get(i, self.UNK_TOKEN))
        return out

    # ------------------------------------------------------------ save / load
    def save(self, path: Union[str, Path]) -> None:
        """Lưu vocabulary ra JSON (token theo thứ tự ID)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = [self.id_to_token[i] for i in range(len(self.id_to_token))]
        payload = {
            "pad_token": self.PAD_TOKEN,
            "unk_token": self.UNK_TOKEN,
            "pad_id": self.PAD_ID,
            "unk_id": self.UNK_ID,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "size": len(tokens),
            "tokens": tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokens = data["tokens"]
        if tokens[cls.PAD_ID] != cls.PAD_TOKEN or tokens[cls.UNK_ID] != cls.UNK_TOKEN:
            raise ValueError("Vocabulary file has unexpected special tokens")

        vocab = cls(min_freq=data.get("min_freq", 1), max_size=data.get("max_size"))
        vocab.token_to_id = {t: i for i, t in enumerate(tokens)}
        vocab.id_to_token = {i: t for i, t in enumerate(tokens)}
        return vocab

    # ------------------------------------------------------------------ misc
    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
