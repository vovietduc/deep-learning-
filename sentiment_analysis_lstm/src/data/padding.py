"""Padding / truncation cho chuỗi token ID.

Quy ước mặc định: padding="post", truncating="post"
    - chuỗi ngắn: thêm PAD vào CUỐI
    - chuỗi dài : cắt phần CUỐI (giữ phần đầu review)

Lưu ý khi dùng với LSTM: padding="post" nghĩa là LSTM sẽ đọc cả các PAD ở cuối.
Nếu lấy hidden state cuối cùng thì cần dùng `lengths` (pack_padded_sequence)
hoặc chọn padding="pre". Xem docs/model.md.
"""

from typing import List, Sequence

_VALID_SIDES = ("pre", "post")


def effective_length(sequence_length: int, max_length: int) -> int:
    """Độ dài thật (không tính PAD) sau truncation, tối thiểu 1.

    Tối thiểu 1 vì pack_padded_sequence không chấp nhận độ dài 0
    (review rỗng vẫn có 1 vị trí là PAD).
    """
    return max(1, min(sequence_length, max_length))


def pad_or_truncate(
    sequence: Sequence[int],
    max_length: int,
    pad_id: int = 0,
    padding: str = "post",
    truncating: str = "post",
) -> List[int]:
    """Đưa sequence về đúng max_length. Luôn trả về list MỚI."""
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")
    if padding not in _VALID_SIDES or truncating not in _VALID_SIDES:
        raise ValueError("padding and truncating must be 'pre' or 'post'")

    seq = list(sequence)

    if len(seq) > max_length:
        return seq[:max_length] if truncating == "post" else seq[-max_length:]

    pad = [pad_id] * (max_length - len(seq))
    return seq + pad if padding == "post" else pad + seq


def pad_batch(
    sequences: Sequence[Sequence[int]],
    max_length: int,
    pad_id: int = 0,
    padding: str = "post",
    truncating: str = "post",
) -> List[List[int]]:
    """Pad/truncate nhiều sequence."""
    return [
        pad_or_truncate(s, max_length, pad_id=pad_id, padding=padding, truncating=truncating)
        for s in sequences
    ]
