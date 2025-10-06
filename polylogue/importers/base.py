from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..render import MarkdownDocument


@dataclass
class ImportResult:
    markdown_path: Path
    html_path: Optional[Path]
    attachments_dir: Optional[Path]
    document: Optional[MarkdownDocument]
    diff_path: Optional[Path] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
