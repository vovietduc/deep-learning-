"""Tokenizer cho review phim IMDb.

Pipeline: raw text -> normalize -> tokens

Quy ước (phải giống nhau ở train, evaluate và demo):
    - None / chuỗi rỗng / chỉ có khoảng trắng  -> []
    - Giải mã HTML entity (&amp; -> &), bỏ thẻ HTML (<br />)
    - Chuẩn hoá Unicode (NFKC) và dấu nháy cong (’ -> ')
    - Mặc định chuyển chữ thường
    - Giữ từ viết tắt dạng don't, wasn't là MỘT token
    - Mặc định giữ "!" và "?" như token riêng vì mang tín hiệu cảm xúc
"""

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional

# Chỉ bỏ thẻ HTML thật sự: "<" phải đi NGAY SAU bởi chữ cái hoặc "/" + chữ cái
# (<br />, </i>, <a href=...>), để câu như "a < b and c > d" không bị xoá nhầm.
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_WHITESPACE_RE = re.compile(r"\s+")

# [^\W_] = chữ cái hoặc số (có cả ký tự Unicode như "é"), không gồm "_"
_WORD = r"[^\W_]+(?:'[^\W_]+)*"
_TOKEN_RE = re.compile(_WORD)
_TOKEN_WITH_PUNCT_RE = re.compile(_WORD + r"|[!?]")


class Tokenizer:
    """Tokenizer đơn giản, xác định (deterministic) và không có trạng thái học."""

    def __init__(self, lowercase: bool = True, keep_punctuation: bool = True):
        self.lowercase = lowercase
        self.keep_punctuation = keep_punctuation
        self._pattern = _TOKEN_WITH_PUNCT_RE if keep_punctuation else _TOKEN_RE

    def normalize(self, text: Optional[Any]) -> str:
        """Chuẩn hoá review thô thành chuỗi sạch."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")

        # Thay thẻ HTML bằng khoảng trắng để "good<br />movie" tách thành 2 từ
        text = _HTML_TAG_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()

        if self.lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: Optional[Any]) -> List[str]:
        """Chuyển review thô thành danh sách token."""
        text = self.normalize(text)
        if not text:
            return []
        return self._pattern.findall(text)

    def get_config(self) -> Dict[str, Any]:
        """Cấu hình để lưu cùng vocabulary (phục vụ tái lập kết quả)."""
        return {
            "type": "regex_word_tokenizer",
            "lowercase": self.lowercase,
            "keep_punctuation": self.keep_punctuation,
        }

    def __call__(self, text: Optional[Any]) -> List[str]:
        return self.tokenize(text)
