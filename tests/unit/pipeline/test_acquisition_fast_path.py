"""Breaker test for #736: cursor mtime fast path must skip full reads.

When the cursor holds matching stat metadata (st_dev, st_ino, st_size,
mtime_ns) for a source file, the acquisition fast path MUST skip that file
entirely — no ``Path.stat()`` bypasses the cursor check (the stat is what
we compare) but no further file I/O (read/write/blob) should occur.
"""

from __future__ import annotations

import os
from pathlib import Path

from polylogue.sources import cursor as cursor_module


def test_select_paths_skips_file_when_cursor_stat_matches(tmp_path: Path) -> None:
    """If the cursor matches file stat, the file must not appear in paths_to_process."""
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"test": "data"}\n')
    st = os.stat(test_file)

    cursor_data: dict[str, object] = {
        "st_dev": st.st_dev,
        "st_ino": st.st_ino,
        "st_size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }

    paths, skipped = cursor_module._select_paths_for_processing(
        [test_file],
        include_file_mtime=False,
        known_cursors={str(test_file): cursor_data},
    )

    # The file should be skipped — no paths returned.
    assert paths == [], f"Expected file to be skipped by cursor match, got {paths}"
    assert skipped == 1, f"Expected skipped=1, got {skipped}"


def test_select_paths_processes_file_when_cursor_has_wrong_size(tmp_path: Path) -> None:
    """If the cursor has a stale size, the file must still be processed."""
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("x" * 500)
    st = os.stat(test_file)

    # Deliberately wrong size.
    cursor_data: dict[str, object] = {
        "st_dev": st.st_dev,
        "st_ino": st.st_ino,
        "st_size": 999999,
        "mtime_ns": st.st_mtime_ns,
    }

    paths, skipped = cursor_module._select_paths_for_processing(
        [test_file],
        include_file_mtime=False,
        known_cursors={str(test_file): cursor_data},
    )

    assert paths == [(test_file, None)], f"Expected file to be processed (stale cursor), got paths={paths}"
    assert skipped == 0, f"Expected skipped=0, got {skipped}"


def test_select_paths_processes_file_when_cursor_has_wrong_inode(tmp_path: Path) -> None:
    """If the cursor has a stale inode (file was replaced), process the file."""
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"version": 1}\n')
    st = os.stat(test_file)

    # Wrong inode (simulates file replacement).
    cursor_data: dict[str, object] = {
        "st_dev": st.st_dev,
        "st_ino": st.st_ino + 1,
        "st_size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }

    paths, skipped = cursor_module._select_paths_for_processing(
        [test_file],
        include_file_mtime=False,
        known_cursors={str(test_file): cursor_data},
    )

    assert len(paths) == 1, f"Expected file to be processed (inode mismatch), got {paths}"
    assert skipped == 0, f"Expected skipped=0, got {skipped}"


def test_select_paths_no_known_cursors_falls_through_to_mtime(tmp_path: Path) -> None:
    """When known_cursors is None, the function falls through to mtime check."""
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("data")
    file_mtime = cursor_module._get_file_mtime(test_file)
    assert file_mtime is not None

    paths, skipped = cursor_module._select_paths_for_processing(
        [test_file],
        include_file_mtime=True,
        known_mtimes={str(test_file): file_mtime},
        known_cursors=None,
    )

    # Falls through to mtime check — file should be skipped.
    assert paths == [], f"Expected file to be skipped by mtime, got {paths}"
    assert skipped == 1, f"Expected skipped=1, got {skipped}"


def test_select_paths_cursor_skips_read_regardless_of_mtime(tmp_path: Path) -> None:
    """Cursor match takes priority: even if known_mtimes would not skip,
    if cursor matches, the file IS skipped."""
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("data")
    st = os.stat(test_file)
    wrong_mtime = cursor_module._get_file_mtime(test_file)
    assert wrong_mtime is not None
    wrong_mtime = "2000-01-01T00:00:00"  # different from actual mtime

    cursor_data: dict[str, object] = {
        "st_dev": st.st_dev,
        "st_ino": st.st_ino,
        "st_size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }

    paths, skipped = cursor_module._select_paths_for_processing(
        [test_file],
        include_file_mtime=True,
        known_mtimes={str(test_file): wrong_mtime},
        known_cursors={str(test_file): cursor_data},
    )

    # Cursor matches so file should be skipped, even though mtime doesn't.
    assert paths == [], f"Expected file skipped by cursor (overrides mtime), got {paths}"
    assert skipped == 1, f"Expected skipped=1, got {skipped}"


def test_stat_matches_cursor_full_match() -> None:
    """_stat_matches_cursor returns True when all cursor fields match the stat."""
    tmp = Path("/tmp")
    st = tmp.stat()
    cursor_data: dict[str, object] = {
        "st_dev": st.st_dev,
        "st_ino": st.st_ino,
        "st_size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }
    assert cursor_module._stat_matches_cursor(st, cursor_data) is True


def test_stat_matches_cursor_sparse_dict_fails() -> None:
    """_stat_matches_cursor returns False when cursor has missing fields."""
    tmp = Path("/tmp")
    st = tmp.stat()

    # Missing all cursor fields.
    assert cursor_module._stat_matches_cursor(st, {}) is False

    # Only one field present.
    assert cursor_module._stat_matches_cursor(st, {"st_dev": st.st_dev}) is False
