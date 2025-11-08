from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Set

from ..render import AttachmentInfo


class AttachmentManager:
    """Utility helpers for managing attachment directories."""

    @staticmethod
    def expected_paths(markdown_path: Path, attachments: Iterable[AttachmentInfo]) -> Set[Path]:
        base = markdown_path.parent
        expected: Set[Path] = set()
        for info in attachments:
            if info.remote or info.local_path is None:
                continue
            try:
                rel = Path(info.local_path)
                expected.add((base / rel).resolve())
            except Exception:
                continue
        return expected

    @staticmethod
    def reconcile(directory: Path, expected: Set[Path]) -> None:
        if not directory.exists():
            return
        for file_path in sorted(directory.rglob("*"), reverse=True):
            try:
                resolved = file_path.resolve()
            except Exception:
                resolved = file_path
            if file_path.is_file() and resolved not in expected:
                try:
                    file_path.unlink()
                except OSError:
                    pass
            elif file_path.is_dir() and not any(file_path.iterdir()):
                try:
                    file_path.rmdir()
                except OSError:
                    pass
        if not expected and directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
