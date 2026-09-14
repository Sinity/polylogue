"""Dispatch-sidecar resolution matches literally and refuses hostile sidecars.

``_sidecar_dispatch_tool_ids`` runs in the synchronous database writer, and it
selects ``raw_sessions`` rows without requiring a successful parse, a current
revision or any size bound -- then reads the blob and decodes it. Three
consequences, all reachable from an imported Claude Code export:

- the child stem is provider-derived and went into a ``LIKE`` pattern
  unescaped, so ``_`` matched any character and pulled a sibling session's
  sidecar in. More than one tool id is treated as a dispatch-identity
  contradiction, so the stray match does not mis-bind an edge -- it refuses a
  correct one;
- ``RecursionError`` is a ``RuntimeError``, not a ``ValueError``, so a deeply
  nested sidecar escaped the handler and aborted the session write. The raw row
  persists, so it aborted again on every later replay of the same lineage;
- ZIP admission permits a 10 GiB member, and the writer agreed to ``read_all``
  whatever the row pointed at.

Anti-vacuity: drop the ``ESCAPE``/``_escape_like`` pair and the first test sees
both tool ids; drop ``RecursionError`` from the handler and the second test
raises instead of returning; drop the ``blob_size`` bound and the third test's
fake store is asked for the oversized blob.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import polylogue.storage.sqlite.archive_tiers.write as write_mod
from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import _sidecar_dispatch_tool_ids

_ORIGIN = Origin.CLAUDE_CODE_SESSION.value


class _FakeBlobStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requested: list[str] = []

    def read_all(self, hash_hex: str) -> bytes:
        self.requested.append(hash_hex)
        return self.payloads[hash_hex]


def _source_conn(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "source.db"
    initialize_archive_database(db_path, ArchiveTier.SOURCE)
    return sqlite3.connect(db_path)


def _insert_sidecar(conn: sqlite3.Connection, *, raw_id: str, source_path: str, digest: str, size: int) -> None:
    conn.execute(
        "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (raw_id, _ORIGIN, source_path, bytes.fromhex(digest), size),
    )
    conn.commit()


def _meta(tool_use_id: str) -> bytes:
    return (
        b'{"type":"subagent","agent_type":"general-purpose","tool_use_id":"' + tool_use_id.encode() + b'","uuid":"u1"}'
    )


def test_a_sibling_stem_is_not_matched_through_a_like_wildcard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    del frozen_clock
    wanted = "aa" * 32
    sibling = "bb" * 32
    conn = _source_conn(tmp_path)
    try:
        _insert_sidecar(
            conn,
            raw_id="raw-wanted",
            source_path="/export/parent-1/subagents/agent-a_b.meta.json",
            digest=wanted,
            size=128,
        )
        _insert_sidecar(
            conn,
            raw_id="raw-sibling",
            source_path="/export/parent-1/subagents/agent-axb.meta.json",
            digest=sibling,
            size=128,
        )
        store = _FakeBlobStore({wanted: _meta("toolu_wanted"), sibling: _meta("toolu_sibling")})
        monkeypatch.setattr(write_mod, "get_blob_store", lambda: store)
        tool_ids = _sidecar_dispatch_tool_ids(
            conn,
            origin=_ORIGIN,
            parent_values={"parent-1"},
            child_values={"agent-a_b"},
        )
    finally:
        conn.close()
    assert tool_ids == {"toolu_wanted"}, tool_ids


def test_a_deeply_nested_sidecar_is_refused_instead_of_aborting_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    del frozen_clock
    digest = "cc" * 32
    conn = _source_conn(tmp_path)
    try:
        _insert_sidecar(
            conn,
            raw_id="raw-nested",
            source_path="/export/parent-1/subagents/agent-deep.meta.json",
            digest=digest,
            size=4096,
        )
        # 200k is where CPython's JSON decoder actually overflows the stack in
        # this build; 20k decodes cleanly and would make this test vacuous.
        nested = b"[" * 200_000 + b"]" * 200_000
        monkeypatch.setattr(write_mod, "get_blob_store", lambda: _FakeBlobStore({digest: nested}))
        # No raise: the writer records a refusal and keeps going.
        tool_ids = _sidecar_dispatch_tool_ids(
            conn,
            origin=_ORIGIN,
            parent_values={"parent-1"},
            child_values={"agent-deep"},
        )
    finally:
        conn.close()
    assert tool_ids == set()


def test_an_oversized_sidecar_is_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    del frozen_clock
    digest = "dd" * 32
    conn = _source_conn(tmp_path)
    try:
        _insert_sidecar(
            conn,
            raw_id="raw-huge",
            source_path="/export/parent-1/subagents/agent-huge.meta.json",
            digest=digest,
            size=write_mod._SIDECAR_DISPATCH_MAX_BYTES + 1,
        )
        store = _FakeBlobStore({digest: _meta("toolu_huge")})
        monkeypatch.setattr(write_mod, "get_blob_store", lambda: store)
        tool_ids = _sidecar_dispatch_tool_ids(
            conn,
            origin=_ORIGIN,
            parent_values={"parent-1"},
            child_values={"agent-huge"},
        )
    finally:
        conn.close()
    assert tool_ids == set()
    assert store.requested == [], "the oversized blob was read despite the bound"
