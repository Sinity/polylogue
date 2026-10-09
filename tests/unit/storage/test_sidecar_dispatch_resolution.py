"""Dispatch-sidecar resolution matches exactly and refuses hostile sidecars.

``_sidecar_paths_dispatch_tool_ids`` runs in the synchronous database writer,
and it selects ``raw_sessions`` rows without requiring a successful parse, a
current revision or any size bound -- then reads the blob and decodes it.
Three consequences, all reachable from an imported Claude Code export:

- the child stem is provider-derived; a pattern match on it (an unescaped
  ``LIKE``, where ``_`` matches any character) pulled a sibling session's
  sidecar in. More than one tool id is treated as a dispatch-identity
  contradiction, so the stray match does not mis-bind an edge -- it refuses a
  correct one. The lookup is now an exact ``source_path`` probe;
- ``RecursionError`` is a ``RuntimeError``, not a ``ValueError``, so a deeply
  nested sidecar escaped the handler and aborted the session write. The raw row
  persists, so it aborted again on every later replay of the same lineage;
- ZIP admission permits a 10 GiB member, and the writer agreed to ``read_all``
  whatever the row pointed at.

Anti-vacuity: match the sidecar path by pattern instead of equality and the
first test sees both tool ids; drop ``RecursionError`` from the handler and the
second test raises instead of returning; reinstate a byte ceiling and the third
test's parent tool id is not resolved.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

import polylogue.storage.sqlite.archive_tiers.write as write_mod
from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead, _sidecar_paths_dispatch_tool_ids
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture

_ORIGIN = Origin.CLAUDE_CODE_SESSION.value


class _FakeBlobStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requested: list[str] = []

    def open(self, hash_hex: str) -> io.BytesIO:
        self.requested.append(hash_hex)
        return io.BytesIO(self.payloads[hash_hex])


def _source_conn(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "source.db"
    initialize_runtime_source_fixture(db_path)
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
        monkeypatch.setattr(write_mod, "blob_store_for_connection", lambda _conn: store)
        tool_ids = _sidecar_paths_dispatch_tool_ids(
            ConnectionSessionSourceRead(conn),
            origin=_ORIGIN,
            sidecar_paths={"/export/parent-1/subagents/agent-a_b.meta.json"},
            parent_values={"parent-1"},
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
        monkeypatch.setattr(write_mod, "blob_store_for_connection", lambda _conn: _FakeBlobStore({digest: nested}))
        # No raise: the writer records a refusal and keeps going.
        tool_ids = _sidecar_paths_dispatch_tool_ids(
            ConnectionSessionSourceRead(conn),
            origin=_ORIGIN,
            sidecar_paths={"/export/parent-1/subagents/agent-deep.meta.json"},
            parent_values={"parent-1"},
        )
    finally:
        conn.close()
    assert tool_ids == set()


def test_a_large_sidecar_is_streamed_to_its_dispatch_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    """Sidecar size never refuses dispatch identity.

    Anti-vacuity: reinstate a byte ceiling on the sidecar read (the removed
    8 MiB refusal) and the parent tool_use id is not resolved.
    """
    del frozen_clock
    digest = "dd" * 32
    payload = (
        b'{"type":"subagent","padding":"' + b"x" * (9 * 1024 * 1024) + b'","tool_use_id":"toolu_large","uuid":"u1"}'
    )
    conn = _source_conn(tmp_path)
    try:
        _insert_sidecar(
            conn,
            raw_id="raw-large",
            source_path="/export/parent-1/subagents/agent-large.meta.json",
            digest=digest,
            size=len(payload),
        )
        store = _FakeBlobStore({digest: payload})
        monkeypatch.setattr(write_mod, "blob_store_for_connection", lambda _conn: store)
        tool_ids = _sidecar_paths_dispatch_tool_ids(
            ConnectionSessionSourceRead(conn),
            origin=_ORIGIN,
            sidecar_paths={"/export/parent-1/subagents/agent-large.meta.json"},
            parent_values={"parent-1"},
        )
    finally:
        conn.close()
    assert tool_ids == {"toolu_large"}


def test_a_scalar_sidecar_root_carries_no_dispatch_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    """A JSON string whose text looks like a sidecar is not a sidecar.

    Anti-vacuity: hand the decoded scalar to the artifact parser and it is
    decoded a second time into a document naming ``toolu_parent``.
    """
    del frozen_clock
    digest = "ee" * 32
    payload = b'"{\\"toolUseId\\":\\"toolu_parent\\"}"'
    conn = _source_conn(tmp_path)
    try:
        _insert_sidecar(
            conn,
            raw_id="raw-scalar",
            source_path="/export/parent-1/subagents/agent-scalar.meta.json",
            digest=digest,
            size=len(payload),
        )
        monkeypatch.setattr(write_mod, "blob_store_for_connection", lambda _conn: _FakeBlobStore({digest: payload}))
        tool_ids = _sidecar_paths_dispatch_tool_ids(
            ConnectionSessionSourceRead(conn),
            origin=_ORIGIN,
            sidecar_paths={"/export/parent-1/subagents/agent-scalar.meta.json"},
            parent_values={"parent-1"},
        )
    finally:
        conn.close()
    assert tool_ids == set()


def test_retained_sidecar_exact_long_identity_ignores_large_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: object
) -> None:
    del frozen_clock
    from polylogue.schemas import observation_spill
    from polylogue.schemas.observation_spill import _ScalarTokenStore

    identity = "toolu_" + "x" * 9000
    payload = b'{"toolUseId":"' + identity.encode() + b'","ignored":' + b"9" * 65537 + b"}"
    digest = "ef" * 32
    conn = _source_conn(tmp_path)
    original_read = _ScalarTokenStore.read

    def selected_read(self: _ScalarTokenStore, kind: str, ordinal: int) -> object:
        row = self.connection.execute(
            "SELECT decoded_bytes FROM json_scalar_tokens WHERE kind=? AND token=?", (kind, ordinal)
        ).fetchone()
        assert row[0] < 65537, "ignored integer materialized"
        return original_read(self, kind, ordinal)

    monkeypatch.setattr(observation_spill._ScalarTokenStore, "read", selected_read)
    try:
        path = "/export/parent-1/subagents/agent-exact.meta.json"
        _insert_sidecar(conn, raw_id="raw-exact", source_path=path, digest=digest, size=len(payload))
        monkeypatch.setattr(write_mod, "blob_store_for_connection", lambda _conn: _FakeBlobStore({digest: payload}))
        tool_ids = _sidecar_paths_dispatch_tool_ids(
            ConnectionSessionSourceRead(conn), origin=_ORIGIN, sidecar_paths={path}, parent_values={"parent-1"}
        )
    finally:
        conn.close()
    assert tool_ids == {identity}
