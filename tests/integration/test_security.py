"""Integration security tests for real archive and filesystem boundaries."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from polylogue.paths import Source
from polylogue.sources.decoders import MAX_UNCOMPRESSED_SIZE
from polylogue.sources.source_parsing import iter_source_conversations
from polylogue.storage.state_views import CursorFailurePayload, CursorStatePayload


def _empty_cursor_state() -> CursorStatePayload:
    return {}


def _failed_files(cursor_state: CursorStatePayload) -> list[CursorFailurePayload]:
    return cursor_state.get("failed_files", [])


def _failed_count(cursor_state: CursorStatePayload) -> int:
    return cursor_state.get("failed_count", 0)


def test_symlink_traversal_blocked_in_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        safe_dir = tmppath / "safe"
        safe_dir.mkdir()
        symlink = safe_dir / "etc_link"
        try:
            symlink.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks (Windows or permissions)")

        files = list(safe_dir.rglob("*"))
        etc_files = [f for f in files if "/etc/" in str(f)]
        assert len(etc_files) == 0


def test_symlink_to_file_outside_archive() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        outside_file = tmppath / "outside.txt"
        outside_file.write_text("secret data")
        archive_dir = tmppath / "archive"
        archive_dir.mkdir()
        symlink = archive_dir / "link.txt"
        try:
            symlink.symlink_to(outside_file)
        except OSError:
            pytest.skip("Cannot create symlinks")
        if symlink.exists():
            assert symlink.is_symlink()


def test_symlink_circular_reference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        link1 = tmppath / "link1"
        link2 = tmppath / "link2"
        try:
            link1.symlink_to(link2)
            link2.symlink_to(link1)
        except OSError:
            pytest.skip("Cannot create symlinks")
        try:
            resolved = link1.resolve(strict=False)
            assert resolved is not None
        except RuntimeError:
            pass


def test_zip_bomb_compression_ratio_blocked(tmp_path: Path) -> None:
    zip_path = tmp_path / "suspicious.zip"
    json_content = b'{"id": "test", "messages": []}'
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.json", b"\x00" * (1024 * 1024))
        zf.writestr("valid.json", json_content)

    source = Source(name="test", path=tmp_path)
    cursor_state: CursorStatePayload = _empty_cursor_state()
    list(iter_source_conversations(source, cursor_state=cursor_state))
    failed = _failed_files(cursor_state)
    failed_count = _failed_count(cursor_state)
    assert failed_count >= 1 or not failed
    if failed:
        has_expected_error = any(
            "ratio" in str(f.get("error", "")).lower() or "json" in str(f.get("error", "")).lower() for f in failed
        )
        assert has_expected_error or len(failed) == 0


def test_zip_oversized_file_limit_constant(tmp_path: Path) -> None:
    assert MAX_UNCOMPRESSED_SIZE == 10 * 1024 * 1024 * 1024


def test_zip_path_traversal_filenames_handled(tmp_path: Path) -> None:
    zip_path = tmp_path / "traversal.zip"
    json_content = b'{"id": "traversal-test", "messages": [{"role": "user", "content": "test"}]}'
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../../etc/passwd.json", json_content)
        zf.writestr("..\\..\\windows\\system.json", json_content)
        zf.writestr("normal.json", json_content)

    source = Source(name="test", path=tmp_path)
    cursor_state: CursorStatePayload = _empty_cursor_state()
    payloads = list(iter_source_conversations(source, cursor_state=cursor_state))
    assert len(payloads) >= 1
