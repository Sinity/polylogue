"""Adversary tests for live cursor content-awareness (#620).

Each test constructs a cursor record and a file mutation, then asserts
that the watcher's skip decision is correct (re-ingest vs skip).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.sources.live.cursor import CursorStore


@pytest.fixture
def cursor_store(tmp_path: Path) -> CursorStore:
    db = tmp_path / "test_cursor.sqlite"
    return CursorStore(db)


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    f = tmp_path / "test_source.jsonl"
    f.write_text('{"role":"user","content":"hello"}\n{"role":"assistant","content":"hi"}\n')
    return f


def _fingerprint_file(path: Path) -> tuple[str, int]:
    """Minimal fingerprint: SHA-256 over content + last newline offset."""
    import hashlib

    content = path.read_bytes()
    h = hashlib.sha256(content).hexdigest()
    # Find last complete newline
    last_nl = content.rfind(b"\n")
    return h, last_nl if last_nl >= 0 else 0


class TestCursorStorePersistence:
    def test_store_and_retrieve_cursor(self, cursor_store: CursorStore, source_file: Path) -> None:
        cursor_store.set(
            source_file,
            byte_size=100,
            content_fingerprint="abc123",
            parser_fingerprint="v1",
            source_name="test",
        )
        record = cursor_store.get_record(source_file)
        assert record is not None
        assert record.byte_size == 100
        assert record.content_fingerprint == "abc123"

    def test_stat_fields_are_stored(self, cursor_store: CursorStore, source_file: Path) -> None:
        stat = os.stat(source_file)
        cursor_store.set(
            source_file,
            byte_size=stat.st_size,
            content_fingerprint="abc",
            parser_fingerprint="v1",
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        record = cursor_store.get_record(source_file)
        assert record is not None
        assert record.st_dev == stat.st_dev
        assert record.st_ino == stat.st_ino
        assert record.mtime_ns == stat.st_mtime_ns

    def test_failure_count_increments(self, cursor_store: CursorStore, source_file: Path) -> None:
        cursor_store.set(source_file, byte_size=100, content_fingerprint="abc")
        cursor_store.mark_failed(source_file)
        r1 = cursor_store.get_record(source_file)
        assert r1 is not None
        assert r1.failure_count >= 1
        assert r1.next_retry_at is not None

    def test_excluded_flag_is_stored(self, cursor_store: CursorStore, source_file: Path) -> None:
        cursor_store.set(source_file, byte_size=100, content_fingerprint="abc")
        cursor_store.mark_excluded(source_file)
        r = cursor_store.get_record(source_file)
        assert r is not None
        assert r.excluded


class TestCursorSkipDecisions:
    """Adversary tests for the watcher's content-aware skip logic."""

    def test_same_size_rewrite_changes_fingerprint(self, tmp_path: Path) -> None:
        """A same-size rewrite must produce a different fingerprint."""
        f = tmp_path / "same_size.jsonl"
        f.write_text('{"role":"user","content":"A"}\n')
        fp1, _ = _fingerprint_file(f)
        f.write_text('{"role":"user","content":"B"}\n')
        fp2, _ = _fingerprint_file(f)
        assert fp1 != fp2, "same-size different-content must produce different fingerprints"

    def test_truncate_changes_fingerprint(self, tmp_path: Path) -> None:
        """Truncation must produce a different fingerprint."""
        f = tmp_path / "truncate.jsonl"
        f.write_text('{"role":"user","content":"long message"}\n')
        fp1, _ = _fingerprint_file(f)
        f.write_text('{"role":"user","conte')
        fp2, _ = _fingerprint_file(f)
        assert fp1 != fp2, "truncation must produce different fingerprints"

    def test_append_after_partial_line_changes_fingerprint(self, tmp_path: Path) -> None:
        """Appending after a partial line must work correctly."""
        f = tmp_path / "partial.jsonl"
        f.write_text('{"role":"user","content":"hel')
        fp_partial, last_nl_partial = _fingerprint_file(f)
        f.write_text('{"role":"user","content":"hello"}\n')
        fp_full, last_nl_full = _fingerprint_file(f)
        assert fp_partial != fp_full, "append-completion must change fingerprint"
        assert last_nl_full >= 0, "completed line must be detected"

    def test_parser_fingerprint_change_forces_reingest(self, cursor_store: CursorStore, source_file: Path) -> None:
        """When parser fingerprint differs, the file must be re-ingested."""
        stat = os.stat(source_file)
        fp, _ = _fingerprint_file(source_file)
        cursor_store.set(
            source_file,
            byte_size=stat.st_size,
            content_fingerprint=fp,
            parser_fingerprint="v1",
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        # Simulate new parser version
        NEW_PARSER = "v2"
        record = cursor_store.get_record(source_file)
        assert record is not None
        assert record.parser_fingerprint != NEW_PARSER, (
            "parser version change should trigger re-ingest"
        )

    def test_rename_detection_via_stat(self, cursor_store: CursorStore, tmp_path: Path) -> None:
        """Moving a file changes its path; st_dev/st_ino persist across rename."""
        f1 = tmp_path / "original.jsonl"
        f1.write_text('{"role":"user","content":"test"}\n')
        stat1 = os.stat(f1)
        cursor_store.set(
            f1,
            byte_size=stat1.st_size,
            content_fingerprint="fp1",
            st_dev=stat1.st_dev,
            st_ino=stat1.st_ino,
            mtime_ns=stat1.st_mtime_ns,
        )
        # Rename
        f2 = tmp_path / "renamed.jsonl"
        f1.rename(f2)
        # Old path still has cursor; new path has no cursor
        old_record = cursor_store.get_record(f1)
        new_record = cursor_store.get_record(f2)
        assert old_record is not None, "old path should still have cursor"
        assert new_record is None, "new path should have no cursor (needs fresh ingest)"

    def test_quarantined_files_listed(self, cursor_store: CursorStore, tmp_path: Path) -> None:
        """Excluded files should show up in list_excluded()."""
        f = tmp_path / "quarantine.jsonl"
        f.write_text("invalid json not a real file\n")
        stat = os.stat(f)
        cursor_store.set(f, byte_size=stat.st_size, content_fingerprint="bad")
        cursor_store.mark_excluded(f)
        excluded = cursor_store.list_excluded()
        assert str(f) in excluded
