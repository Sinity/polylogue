"""A logical-export reconstruction is never readable outside the owner.

``open_logical_source`` rebuilds a retained logical export into a temporary
SQLite file before reading it.  That reconstruction carries every row of the
export -- for a Hermes ``state.db`` member, the operator's whole conversation
state -- and it lives under the shared system temporary directory for the
entire materialization window.  If SQLite creates the pathname itself, the
process umask decides the mode: 0644 under the ordinary 0022, world-readable
in a shared ``/tmp``.

Anti-vacuity: restore the ``reconstruction.unlink()`` that used to run before
``materialize_export`` in ``open_logical_source`` -- so SQLite recreates the
pathname instead of reusing the ``mkstemp`` inode -- and the observed mode
becomes 0o644 and this test fails.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

import polylogue.sources.sqlite_export as sqlite_export
from polylogue.sources.sqlite_export import logical_export_bytes, open_logical_source


def _source_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE secrets (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO secrets (id, body) VALUES (1, 'operator transcript')")
        conn.commit()


def test_reconstructed_export_is_owner_only_while_it_is_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    del frozen_clock
    source = tmp_path / "state.db"
    _source_database(source)
    export = tmp_path / "state.db.export"
    export.write_bytes(logical_export_bytes(source))

    observed: list[int] = []
    real_materialize = sqlite_export.materialize_export

    def _recording_materialize(path: Path, destination: Path, **kwargs: object) -> None:
        real_materialize(path, destination, **kwargs)  # type: ignore[arg-type]
        observed.append(stat.S_IMODE(os.stat(destination).st_mode))

    monkeypatch.setattr(sqlite_export, "materialize_export", _recording_materialize)

    previous_umask = os.umask(0o022)
    try:
        conn = open_logical_source(export)
    finally:
        os.umask(previous_umask)
    try:
        assert conn.execute("SELECT body FROM secrets").fetchone()[0] == "operator transcript"
    finally:
        conn.close()

    assert observed, "materialize_export was never reached; the reconstruction route did not run"
    mode = observed[0]
    assert not mode & 0o077, f"reconstruction was group/world accessible: {oct(mode)}"
