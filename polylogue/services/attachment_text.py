from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".rs",
    ".go",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".html",
    ".htm",
    ".xml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".css",
    ".scss",
    ".less",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
}

MAX_BYTES_DEFAULT = 5 * 1024 * 1024
MAX_CHARS_DEFAULT = 120_000


@dataclass
class AttachmentText:
    text: Optional[str]
    mime: Optional[str]
    truncated: bool
    ocr_used: bool
    size_bytes: int


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _read_text_with_limits(path: Path, *, max_bytes: int, max_chars: int) -> Tuple[str, bool]:
    raw = path.read_bytes()
    truncated = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True
    text = raw.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, truncated


def _extract_pdf_text(path: Path, *, max_chars: int) -> Tuple[str, bool]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pieces = []
    for page in reader.pages:
        content = page.extract_text() or ""
        pieces.append(content)
        if sum(len(p) for p in pieces) > max_chars:
            break
    text = "\n".join(pieces)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated


def _extract_image_text(path: Path, *, max_chars: int) -> Tuple[str, bool]:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency required when OCR requested
        raise RuntimeError(
            "OCR requested for attachments but Pillow/pytesseract are not installed. "
            "Install with `pip install 'polylogue[ocr]'` or add pillow+pytesseract to your environment."
        ) from exc

    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated


def extract_attachment_text(
    path: Path,
    *,
    ocr: bool = False,
    max_bytes: int = MAX_BYTES_DEFAULT,
    max_chars: int = MAX_CHARS_DEFAULT,
) -> AttachmentText:
    """Extract best-effort text from an attachment for indexing.

    Returns an AttachmentText with optional text content and metadata about
    truncation and OCR usage.
    """

    size = path.stat().st_size if path.exists() else 0
    mime, _ = mimetypes.guess_type(path.name)
    suffix = path.suffix.lower()

    if not path.exists() or not path.is_file():
        return AttachmentText(text=None, mime=mime, truncated=False, ocr_used=False, size_bytes=size)

    truncated = False
    ocr_used = False

    # Text-like files
    if suffix in TEXT_EXTENSIONS or (mime and mime.startswith("text/")):
        text, truncated = _read_text_with_limits(path, max_bytes=max_bytes, max_chars=max_chars)
        if suffix in {".html", ".htm"} or (mime and mime.startswith("text/html")):
            text = _strip_html(text)
        return AttachmentText(text=text, mime=mime, truncated=truncated, ocr_used=False, size_bytes=size)

    # JSON and YAML can be normalised to reduce noise
    if suffix in {".json", ".yaml", ".yml"}:
        text, truncated = _read_text_with_limits(path, max_bytes=max_bytes, max_chars=max_chars)
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return AttachmentText(text=text, mime=mime, truncated=truncated, ocr_used=False, size_bytes=size)

    # PDFs
    if suffix == ".pdf" or (mime and mime == "application/pdf"):
        text, truncated = _extract_pdf_text(path, max_chars=max_chars)
        return AttachmentText(text=text, mime=mime or "application/pdf", truncated=truncated, ocr_used=False, size_bytes=size)

    # Images with OCR (opt-in)
    if ocr and suffix in IMAGE_EXTENSIONS:
        text, truncated = _extract_image_text(path, max_chars=max_chars)
        ocr_used = True
        return AttachmentText(text=text, mime=mime, truncated=truncated, ocr_used=ocr_used, size_bytes=size)

    # Fallback: best-effort binary decode up to limits
    text, truncated = _read_text_with_limits(path, max_bytes=max_bytes, max_chars=max_chars)
    return AttachmentText(text=text, mime=mime, truncated=truncated, ocr_used=ocr_used, size_bytes=size)


__all__ = [
    "AttachmentText",
    "extract_attachment_text",
    "TEXT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "MAX_BYTES_DEFAULT",
    "MAX_CHARS_DEFAULT",
]
