"""Breaker test for #736: cursor mtime fast path must skip full reads.

When the cursor holds matching stat metadata (st_dev, st_ino, st_size,
mtime_ns) for a source file, the acquisition fast path MUST skip that file
entirely — no ``Path.stat()`` bypasses the cursor check (the stat is what
we compare) but no further file I/O (read/write/blob) should occur.
"""

from __future__ import annotations

import os
import sqlite3
import zipfile
from contextlib import closing
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


def test_select_paths_without_cursor_cannot_skip_by_mtime(tmp_path: Path) -> None:
    """A timestamp alone cannot certify the current byte input."""
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

    assert paths == [(test_file, file_mtime)]
    assert skipped == 0


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


def test_source_walk_reacquires_replaced_codex_with_preserved_mtime(tmp_path: Path) -> None:
    from polylogue.config import Source
    from polylogue.sources.source_walk import _setup_source_walk

    path = tmp_path / "2026/01/02/rollout-session.jsonl"
    path.parent.mkdir(parents=True)
    fixture = Path(__file__).parents[2] / "fixtures/origin-capability/codex-session.jsonl"
    path.write_bytes(fixture.read_bytes())
    stamp = 1800000000000000000
    os.utime(path, ns=(stamp, stamp))
    previous = path.stat()
    known = {key: getattr(previous, attr) for key, attr in cursor_module._CURSOR_STAT_MAP.items()}
    mtime = cursor_module._get_file_mtime(path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(
        path.read_bytes()
        + b'{"type":"response_item","payload":{"type":"message","id":"late-user","role":"user","content":[{"type":"input_text","text":"A late valid turn."}]}}\n'
    )
    os.utime(replacement, ns=(stamp, stamp))
    replacement.replace(path)
    walk = _setup_source_walk(
        Source(name="codex", path=tmp_path),
        cursor_state={},
        include_mtime=True,
        known_mtimes={str(path): mtime},
        known_cursors={str(path): known},
        discover_sidecars=False,
    )
    assert walk is not None
    assert walk.paths_to_process == [(path, mtime)]
    assert walk.skipped_mtime == 0


def test_source_walk_reacquires_committed_sqlite_wal(tmp_path: Path) -> None:
    from polylogue.config import Source
    from polylogue.sources.source_walk import _setup_source_walk

    path = tmp_path / "state_5.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE threads(id TEXT)")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        previous = path.stat()
        known = {key: getattr(previous, attr) for key, attr in cursor_module._CURSOR_STAT_MAP.items()}
        mtime = cursor_module._get_file_mtime(path)
        connection.execute("INSERT INTO threads VALUES ('new-session')")
        connection.commit()
        assert cursor_module._stat_matches_cursor(path.stat(), known)
        assert connection.execute("SELECT id FROM threads").fetchall() == [("new-session",)]
        walk = _setup_source_walk(
            Source(name="codex-state", path=tmp_path),
            cursor_state={},
            include_mtime=True,
            known_mtimes={str(path): mtime},
            known_cursors={str(path): known},
            discover_sidecars=False,
        )
        assert walk is not None
        assert walk.paths_to_process == [(path, mtime)]
        assert walk.skipped_mtime == 0


def test_zip_same_member_names_and_mtime_still_requires_acquisition(tmp_path: Path) -> None:
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("session.json", b'{"version":1}')
    stamp = 1800000000000000000
    os.utime(path, ns=(stamp, stamp))
    mtime = cursor_module._get_file_mtime(path)
    previous = path.stat()
    known = {key: getattr(previous, attr) for key, attr in cursor_module._CURSOR_STAT_MAP.items()}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("session.json", b'{"version":2}')
    os.utime(path, ns=(stamp, stamp))
    assert cursor_module._stat_matches_cursor(path.stat(), known)
    paths, skipped = cursor_module._select_paths_for_processing(
        [path],
        include_file_mtime=True,
        known_cursors={str(path): known},
        known_mtimes={f"{path}:session.json": mtime},
    )
    assert paths == [(path, mtime)]
    assert skipped == 0
