from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from polylogue.importers.utils import safe_extractall


def _build_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_safe_extractall_allows_regular_members(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    _build_zip(archive, {"dir/file.txt": b"hello"})

    extract_dir = tmp_path / "out"
    with zipfile.ZipFile(archive) as zf:
        safe_extractall(zf, extract_dir)

    assert (extract_dir / "dir" / "file.txt").read_text(encoding="utf-8") == "hello"


def test_safe_extractall_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    _build_zip(archive, {"../evil.txt": b"boom"})

    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError):
            safe_extractall(zf, tmp_path / "out")
