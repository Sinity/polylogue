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
    document: MarkdownDocument

