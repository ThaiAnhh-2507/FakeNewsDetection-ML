"""
Làm sạch văn bản - hàm DUY NHẤT dùng cho cả huấn luyện lẫn dự đoán.

Ngoài các bước thông thường (lowercase, bỏ URL...), file này còn xóa những
"cụm lộ nhãn" của bộ ISOT: những thứ không phải nội dung tin nhưng lại xuất
hiện gần như chỉ ở một lớp (ví dụ "(Reuters)" chỉ có ở tin thật,
"Featured image via Getty Images" chỉ có ở tin giả). Nếu không xóa, model
sẽ học các dấu hiệu này thay vì học nội dung, và cho điểm cao ảo.
"""
from __future__ import annotations

import re
import unicodedata

# "WASHINGTON (Reuters) - ..." ở đầu bài (chỉ tìm trong 200 ký tự đầu)
_DATELINE = re.compile(r"^.{0,200}?\(\s*reuters\s*\)\s*[-\u2013\u2014:]*\s*", re.I | re.S)

_LEAK_PATTERNS = [
    re.compile(r"\breuters\b", re.I),
    re.compile(r"(video\s*/\s*)?featured\s+image(\s+(via|from|by|courtesy\s+of))?[^.]*", re.I),
    re.compile(r"\b(getty\s+images|screen\s+capture|image\s+via)\b", re.I),
    re.compile(r"pic\.twitter\.com/\S+", re.I),
    re.compile(r"[\(\[]\s*(video|videos|images?|photos?|pics?|tweets?|gif|watch)\s*[\)\]]", re.I),
    re.compile(r"\b21st\s+century\s+wire\b", re.I),
]

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Trả về chuỗi chữ thường, chỉ gồm a-z, 0-9 và khoảng trắng."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _DATELINE.sub("", value, count=1)
    for pattern in _LEAK_PATTERNS:
        value = pattern.sub(" ", value)
    value = value.lower()
    value = _URL.sub(" ", value)
    value = _EMAIL.sub(" ", value)
    value = _HTML.sub(" ", value)
    value = _NON_ALNUM.sub(" ", value)
    return _SPACES.sub(" ", value).strip()


def build_input(title, body) -> str:
    """Ghép tiêu đề + nội dung (đã làm sạch) thành đầu vào cho model."""
    parts = [clean_text(title), clean_text(body)]
    return " ".join(p for p in parts if p)
