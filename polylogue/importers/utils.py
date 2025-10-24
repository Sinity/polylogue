from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..render import AttachmentInfo

PREVIEW_LINES = 5
LINE_THRESHOLD = 40
CHAR_THRESHOLD = 4000

_ESCAPED_FOOTNOTE_RE = re.compile(r"\\\[([^\]\n]{1,12})\\\]")


def normalise_inline_footnotes(text: str) -> str:
    if not text or "\\[" not in text:
        return text
    # Exports can double escape the brackets (``\\\[``); collapse those first.
    text = text.replace("\\\\[", "\\[").replace("\\\\]", "\\]")

    def is_candidate(label: str) -> bool:
        if not label:
            return False
        if re.fullmatch(r"\d+[A-Za-z]?", label):
            return True
        if len(label) <= 5 and re.fullmatch(r"[A-Za-z0-9-]+", label):
            return True
        return False

    def replacer(match: re.Match[str]) -> str:
        label = match.group(1)
        if is_candidate(label):
            return f"[{label}]"
        return match.group(0)

    return _ESCAPED_FOOTNOTE_RE.sub(replacer, text)

try:  # optional dependency for accurate counts
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

_TOKENIZER_CACHE: Dict[str, object] = {}


def _get_tokenizer(model: Optional[str]) -> Optional[object]:
    if tiktoken is None:
        return None
    key = model or "cl100k_base"
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]
    try:
        if model:
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    _TOKENIZER_CACHE[key] = enc
    return enc


def estimate_token_count(text: str, *, model: Optional[str] = None) -> int:
    """Estimate token usage, preferring tiktoken when available."""

    if not text:
        return 0
    enc = _get_tokenizer(model)
    if enc is None:
        return max(1, len(text.split()))
    try:
        return len(enc.encode(text))  # type: ignore[attr-defined]
    except Exception:
        return max(1, len(text.split()))


def store_large_text(
    text: str,
    *,
    chunk_index: int,
    attachments_dir: Path,
    markdown_dir: Path,
    attachments: List[AttachmentInfo],
    per_chunk_links: Dict[int, List[Tuple[str, Path]]],
    prefix: str = "chunk",
) -> str:
    """Persist oversized text to an attachment and return a preview."""

    lines = text.splitlines()
    if len(lines) <= LINE_THRESHOLD and len(text) <= CHAR_THRESHOLD:
        return text

    attachments_dir.mkdir(parents=True, exist_ok=True)
    attachment_name = f"{prefix}{chunk_index:03d}.txt"
    attachment_path = attachments_dir / attachment_name
    attachment_path.write_text(text, encoding="utf-8")
    try:
        rel = attachment_path.relative_to(markdown_dir)
    except ValueError:
        rel = attachment_path
    attachments.append(
        AttachmentInfo(
            name=attachment_name,
            link=str(rel),
            local_path=rel,
            size_bytes=attachment_path.stat().st_size,
            remote=False,
        )
    )
    per_chunk_links.setdefault(chunk_index, []).append((attachment_name, rel))
    head = lines[:PREVIEW_LINES]
    tail = lines[-PREVIEW_LINES:]
    preview = "\n".join(
        head
        + [
            "…",
            "",
            f"(Full content saved to {attachment_name})",
            "",
        ]
        + tail
    )
    return preview
