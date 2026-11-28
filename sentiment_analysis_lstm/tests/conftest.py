"""Thêm thư mục gốc project vào sys.path để `from src.data... import ...` chạy
được khi gọi `pytest -q` từ bất kỳ đâu (kể cả Windows/VS Code)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
