"""A grouped Claude Code capture publishes one raw row and distinct members.

The one-shot compatibility entry point uses the live acquisition and
convergence owner. A carried-over parent turn and the child session share
one physical JSONL file, so repeated acquisition must preserve one raw
identity while retaining both session memberships.
"""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


class _RejectUnboundedRead:
    """ZIP handle proxy that rejects a full member read in archive admission."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __enter__(self) -> _RejectUnboundedRead:
        self._handle.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._handle.__exit__(*args)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise AssertionError("ZIP artifact admission must stream to the blob store")
        return cast("bytes", self._handle.read(size))


def _write_carryover_chain(root: Path, *, session_prefix: str = "") -> tuple[Path, Path]:
    """Write parent-session.jsonl (real "parent" session) + child-session.jsonl
    (a 1-record carryover of parent's tail under `sessionId=parent-session`,
    then real "child-session" content) -- the exact structural shape found in
    production, including Claude Code's own naming convention of literally
    naming each file after its own session id (bd polylogue-jc4q: this is
    what dispatch.py's carryover-vs-independent-session detection anchors
    on, so the fixture must use real-shaped filenames, not human-readable
    stand-ins, to exercise it honestly).
    """
    parent_session = f"{session_prefix}parent-session"
    child_session = f"{session_prefix}child-session"
    parent_file = root / "parent-session.jsonl"
    child_file = root / "child-session.jsonl"

    def rec(session_id: str, uuid: str, role: str, text: str, parent_uuid: str | None = None) -> str:
        import json as _json

        return _json.dumps(
            {
                "type": role,
                "sessionId": session_id,
                "uuid": uuid,
                "parentUuid": parent_uuid,
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
                "timestamp": "2026-02-13T00:00:00.000Z",
            }
        )

    parent_file.write_text(
        "\n".join(
            [
                rec(parent_session, "p-u1", "user", "p1"),
                rec(parent_session, "p-a1", "assistant", "p2", parent_uuid="p-u1"),
            ]
        )
        + "\n"
    )
    child_file.write_text(
        "\n".join(
            [
                # Boundary carryover: same conversation thread, still tagged
                # with the PARENT's sessionId, referencing parent's last uuid.
                rec(
                    parent_session,
                    "carryover",
                    "user",
                    "[Request interrupted by user for tool use]",
                    parent_uuid="p-a1",
                ),
                rec(child_session, "c-u1", "user", "c1"),
                rec(child_session, "c-a1", "assistant", "c2", parent_uuid="c-u1"),
            ]
        )
        + "\n"
    )
    return parent_file, child_file


def _raw_rows_for_path(source_db: Path, source_path: str) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT raw_id, native_id FROM raw_sessions WHERE source_path = ? ORDER BY raw_id",
            (source_path,),
        ).fetchall()
    finally:
        conn.close()
    return [(str(r[0]), r[1]) for r in rows]


def _membership_rows(source_db: Path, raw_id: str) -> set[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT logical_source_key, provider_session_id FROM raw_session_memberships WHERE raw_id = ?",
            (raw_id,),
        ).fetchall()
    finally:
        conn.close()
    return {(str(row[0]), str(row[1])) for row in rows}


def _workflow_journal_payload(*, malformed: bool = False, delayed: bool = False) -> bytes:
    if malformed:
        return b'{"contentKey":"broken"\n'
    prefix = b""
    if delayed:
        prefix = b"".join(
            b'{"contentKey":"artifact-' + str(index).encode() + b'","agentId":"workflow-agent"}\n'
            for index in range(32)
        )
    return prefix + (
        b'{"sessionId":"journal-session","parentUuid":null,"type":"user",'
        b'"message":{"role":"user","content":[{"type":"text","text":"recover journal"}]},'
        b'"uuid":"journal-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"sessionId":"journal-session","parentUuid":"journal-user","type":"assistant",'
        b'"message":{"role":"assistant",'
        b'"content":[{"type":"text","text":"repaired reply"}]},"uuid":"journal-assistant",'
        b'"timestamp":"2025-01-01T00:00:01Z"}\n'
    )


def _write_session_shaped_workflow_journal(root: Path, *, malformed: bool = False) -> Path:
    journal = root / "subagents" / "workflows" / "wf-archive" / "journal.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(_workflow_journal_payload(malformed=malformed))
    return journal


def _write_workflow_journal_zip(root: Path, *, malformed: bool = False) -> Path:
    archive = root / "claude-export.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "subagents/workflows/wf-archive/journal.jsonl",
            _workflow_journal_payload(malformed=malformed, delayed=not malformed),
        )
    return archive


def _write_large_zip_member(root: Path, name: str, payload: bytes) -> Path:
    archive = root / "large-export.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(name, payload)
    return archive


def _set_mtime_ms(path: Path, mtime_ms: int) -> None:
    os.utime(path, ns=(mtime_ms * 1_000_000, mtime_ms * 1_000_000))


@pytest.mark.asyncio
async def test_archive_ingest_session_shaped_workflow_journal_reaches_parser_idempotently(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """The production one-shot route must decode a journal before path exclusion."""
    archive_root = one_shot_workspace_env["archive_root"]
    journal = _write_session_shaped_workflow_journal(tmp_path / "sessions")
    sources = [Source(name="claude-code", path=journal)]

    first = await parse_sources_archive(archive_root, sources, parse_workers=1)
    second = await parse_sources_archive(archive_root, sources, parse_workers=1)

    assert first.parse_failures == 0
    assert first.counts["sessions"] == 1
    assert second.parse_failures == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


@pytest.mark.asyncio
async def test_archive_ingest_ordinary_session_records_current_parser_receipt(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """Canonical acquisition must emit source-tier parser authority evidence.

    The fixture enters through live intake and convergence; no receipt is
    inserted in test setup. Losing the admission receipt leaves readiness
    blocked even when the session is indexed.
    """
    archive_root = one_shot_workspace_env["archive_root"]
    journal = _write_session_shaped_workflow_journal(tmp_path / "sessions")

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=journal)],
        parse_workers=1,
    )

    assert result.parse_failures == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        receipt = conn.execute(
            """
            SELECT c.parser_fingerprint, c.status, c.logical_keys_json
            FROM raw_sessions AS r
            JOIN raw_authority_parser_census AS c ON c.raw_id = r.raw_id
            WHERE r.source_path = ?
            """,
            (str(journal),),
        ).fetchone()

    assert receipt == (
        RAW_AUTHORITY_PARSER_FINGERPRINT,
        "complete",
        '["claude-code-session:journal-session"]',
    )


@pytest.mark.asyncio
async def test_archive_ingest_malformed_workflow_journal_remains_typed_evidence(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """A journal with no decodable session evidence remains a typed artifact."""
    archive_root = one_shot_workspace_env["archive_root"]
    journal = _write_session_shaped_workflow_journal(tmp_path / "sessions", malformed=True)
    expected_mtime_ms = 1_735_689_600_123
    _set_mtime_ms(journal, expected_mtime_ms)

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=journal)],
        parse_workers=1,
    )

    assert result.parse_failures == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT file_mtime_ms FROM raw_sessions").fetchone() == (expected_mtime_ms,)
        assert conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone() == (
            "workflow_journal",
            0,
        )
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.asyncio
async def test_archive_ingest_zip_workflow_journal_scans_delayed_session_evidence_idempotently(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """ZIP member routing must decode beyond 32 artifact records before exclusion."""
    archive_root = one_shot_workspace_env["archive_root"]
    journal_zip = _write_workflow_journal_zip(tmp_path / "sessions")
    sources = [Source(name="claude-code", path=journal_zip)]

    first = await parse_sources_archive(archive_root, sources, parse_workers=1)
    second = await parse_sources_archive(archive_root, sources, parse_workers=1)

    assert first.parse_failures == 0
    assert first.counts["sessions"] == 1
    assert second.parse_failures == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


@pytest.mark.asyncio
async def test_archive_ingest_malformed_zip_workflow_journal_remains_typed_evidence(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """Malformed ZIP journals are retained as typed evidence without sessions."""
    archive_root = one_shot_workspace_env["archive_root"]
    journal_zip = _write_workflow_journal_zip(tmp_path / "sessions", malformed=True)
    expected_mtime_ms = 1_735_689_601_456
    _set_mtime_ms(journal_zip, expected_mtime_ms)

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=journal_zip)],
        parse_workers=1,
    )

    assert result.parse_failures == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        assert conn.execute("SELECT file_mtime_ms FROM raw_sessions").fetchone() == (expected_mtime_ms,)
        assert conn.execute("SELECT artifact_kind, parse_as_session FROM raw_artifacts").fetchone() == (
            "workflow_journal",
            0,
        )
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.asyncio
async def test_archive_ingest_large_zip_artifact_streams_to_blob_reference(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large ZIP journal artifact must not be read into an admission payload."""
    from polylogue.sources.decoder_zip import _ZIP_READ_CHUNK_SIZE, MAX_UNCOMPRESSED_SIZE, open_bounded_zip_entry

    archive_root = one_shot_workspace_env["archive_root"]
    payload = b'{"contentKey":"artifact","agentId":"workflow-agent","body":"' + b"x" * _ZIP_READ_CHUNK_SIZE + b'"}\n'
    journal_zip = _write_large_zip_member(
        tmp_path / "sessions",
        "subagents/workflows/wf-archive/journal.jsonl",
        payload,
    )
    original_open = open_bounded_zip_entry

    def reject_unbounded_read(
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        *,
        max_bytes: int = MAX_UNCOMPRESSED_SIZE,
    ) -> _RejectUnboundedRead:
        return _RejectUnboundedRead(original_open(zf, info, max_bytes=max_bytes))

    monkeypatch.setattr("polylogue.sources.decoder_zip.open_bounded_zip_entry", reject_unbounded_read)

    result = await parse_sources_archive(archive_root, [Source(name="claude-code", path=journal_zip)], parse_workers=1)

    assert result.parse_failures == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (1,)
        assert conn.execute("SELECT blob_size FROM raw_sessions").fetchone() == (len(payload),)


@pytest.mark.asyncio
async def test_archive_ingest_large_ordinary_zip_jsonl_skips_delayed_artifact_scan(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unclassified ZIP JSONL follows normal parsing without a second full scan."""
    from polylogue.sources import decoder_zip

    archive_root = one_shot_workspace_env["archive_root"]
    payload = (
        b'{"sessionId":"ordinary-session","parentUuid":null,"type":"user",'
        b'"message":{"role":"user","content":[{"type":"text","text":"' + b"x" * (1024 * 1024) + b'"}]},'
        b'"uuid":"ordinary-user","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"sessionId":"ordinary-session","parentUuid":"ordinary-user","type":"assistant",'
        b'"message":{"role":"assistant","content":[{"type":"text","text":"reply"}]},'
        b'"uuid":"ordinary-assistant","timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    session_zip = _write_large_zip_member(tmp_path / "sessions", "nested/ordinary.jsonl", payload)

    def fail_unexpected_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary ZIP JSONL must not receive a delayed artifact scan")

    monkeypatch.setattr(decoder_zip, "zip_entry_session_artifact", fail_unexpected_scan)

    result = await parse_sources_archive(archive_root, [Source(name="claude-code", path=session_zip)], parse_workers=1)

    assert result.parse_failures == 0
    assert result.counts["sessions"] == 1


@pytest.mark.asyncio
async def test_archive_ingest_path_classified_zip_json_record_array_reaches_parser(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """Decoded Claude records outrank a non-session workflow snapshot path."""
    archive_root = one_shot_workspace_env["archive_root"]
    journal_zip = _write_large_zip_member(
        tmp_path / "sessions",
        "workflows/wf-json.json",
        json.dumps(
            [
                {
                    "sessionId": "json-array-session",
                    "parentUuid": None,
                    "type": "user",
                    "message": {"role": "user", "content": "recover JSON records"},
                    "uuid": "json-array-user",
                    "timestamp": "2025-01-01T00:00:00Z",
                },
                {
                    "sessionId": "json-array-session",
                    "parentUuid": "json-array-user",
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "recovered reply"}],
                    },
                    "uuid": "json-array-assistant",
                    "timestamp": "2025-01-01T00:00:01Z",
                },
            ]
        ).encode(),
    )

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=journal_zip)],
        parse_workers=1,
    )

    assert result.parse_failures == 0
    assert result.counts["sessions"] >= 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (0,)


@pytest.mark.asyncio
async def test_archive_ingest_records_file_mtime_through_canonical_acquisition(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
) -> None:
    archive_root = one_shot_workspace_env["archive_root"]
    source_root = tmp_path / "sessions"
    source_root.mkdir()
    _, source_path = _write_carryover_chain(source_root)
    expected_mtime_ms = 1_777_632_000_000
    _set_mtime_ms(source_path, expected_mtime_ms)

    result = await parse_sources_archive(archive_root, [Source(name="claude-code", path=source_path)], parse_workers=1)

    assert result.parse_failures == 0
    assert result.counts["sessions"] == 2
    with sqlite3.connect(archive_root / "source.db") as conn:
        raw_mtime_ms = conn.execute(
            "SELECT file_mtime_ms FROM raw_sessions WHERE source_path = ?",
            (str(source_path),),
        ).fetchone()[0]
    assert raw_mtime_ms == expected_mtime_ms


async def test_grouped_carryover_sessions_share_one_raw_row(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """Two sessions split from ONE Claude Code file's bytes must NOT produce
    two raw_sessions rows for that file -- the specific bug behind
    polylogue-sjf6's live guard crash."""
    archive_root = one_shot_workspace_env["archive_root"]
    root = tmp_path / "sessions"
    root.mkdir()
    _parent_file, child_file = _write_carryover_chain(root)
    sources = [Source(name="claude-code", path=child_file)]

    result = await parse_sources_archive(archive_root, sources)
    assert result.parse_failures == 0

    rows = _raw_rows_for_path(archive_root / "source.db", str(child_file))
    # A second raw identity for either split session breaks the shared-source
    # membership claim, and this assertion observes that durable failure.
    assert len(rows) == 1, f"expected exactly one raw row for child-session.jsonl's bytes, got {rows}"
    assert rows[0][1] is None
    assert _membership_rows(archive_root / "source.db", rows[0][0]) == {
        ("claude-code-session:parent-session:child-session", "parent-session:child-session"),
        ("claude-code-session:child-session", "child-session"),
    }

    # Both split sessions must still have been indexed -- but as of
    # bd polylogue-jc4q the 1-record carryover no longer collides identity
    # with "parent-session" itself (that was the bug: every fork/resume
    # descendant of one ancestor carrying a boundary record ends up
    # colliding revision membership on the ancestor's own
    # `logical_source_key`). It keeps a distinct, qualified identity while
    # recording "parent-session" as an unresolved lineage edge instead
    # (parent-session.jsonl is written to disk by the fixture but never
    # ingested by this test's `sources`, so the edge stays unresolved).
    conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        native_ids = {
            str(r[0])
            for r in conn.execute(
                "SELECT native_id FROM sessions WHERE native_id IN ('parent-session:child-session', 'child-session')"
            )
        }
        carryover_links = conn.execute(
            "SELECT dst_origin, dst_native_id, resolved_dst_session_id FROM session_links "
            "WHERE src_session_id = 'claude-code-session:parent-session:child-session'"
        ).fetchall()
    finally:
        conn.close()
    assert native_ids == {"parent-session:child-session", "child-session"}
    assert carryover_links == [("claude-code-session", "parent-session", None)]


def _write_sibling_carryover_children(root: Path, ancestor_session_id: str) -> tuple[Path, Path]:
    """Write two SIBLING resume/fork files that both carry a boundary record
    from the SAME ancestor session, but are otherwise unrelated to each other
    -- the exact shape bd polylogue-jc4q measured against the live archive
    (fork/resume files ``a3a274a2...``/``cbea0c3a...`` both echoing ancestor
    ``0213d48f...``): a live/usage-limit interruption forced two independent
    resumes off one ancestor, and each resumed file's leading records still
    carried the ancestor's own sessionId before diverging into their own.
    """

    def rec(session_id: str, uuid: str, role: str, text: str, parent_uuid: str | None = None) -> str:
        import json as _json

        return _json.dumps(
            {
                "type": role,
                "sessionId": session_id,
                "uuid": uuid,
                "parentUuid": parent_uuid,
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
                "timestamp": "2026-02-13T00:00:00.000Z",
            }
        )

    sibling_a = root / "sibling-a.jsonl"
    sibling_b = root / "sibling-b.jsonl"
    sibling_a.write_text(
        "\n".join(
            [
                rec(ancestor_session_id, "carryover-a", "user", "[Request interrupted]", parent_uuid="ancestor-tail"),
                rec("sibling-a", "a-u1", "user", "a1"),
                rec("sibling-a", "a-a1", "assistant", "a2", parent_uuid="a-u1"),
            ]
        )
        + "\n"
    )
    sibling_b.write_text(
        "\n".join(
            [
                rec(ancestor_session_id, "carryover-b", "user", "[Request interrupted]", parent_uuid="ancestor-tail"),
                rec("sibling-b", "b-u1", "user", "b1"),
                rec("sibling-b", "b-a1", "assistant", "b2", parent_uuid="b-u1"),
            ]
        )
        + "\n"
    )
    return sibling_a, sibling_b


@pytest.mark.asyncio
async def test_sibling_fork_carryovers_off_one_ancestor_do_not_collide_identity(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """bd polylogue-jc4q: two UNRELATED resume/fork files that each carry a
    boundary record from the SAME ancestor session must not be assigned the
    SAME `provider_session_id` as each other (or as the ancestor). Before
    this fix, both siblings' carryover fragments composed their identity
    from the bare ancestor sessionId, so `claude-code-session:<ancestor id>`
    became a `logical_source_key` cohort of mutually-incomparable revisions
    (each carryover strictly contained in the real ancestor, but not in each
    other) -- an irreducible conflict that quarantined the whole cohort,
    including the ancestor's own real, large revision. Measured live: only
    56/185 (30.3%) of claude-code-session ambiguous cohorts resolved cleanly
    under this defect, far below chatgpt-export/claude-ai-export (>90%).
    """
    archive_root = one_shot_workspace_env["archive_root"]
    root = tmp_path / "sessions"
    root.mkdir()
    ancestor_session_id = "ancestor-session"
    sibling_a, sibling_b = _write_sibling_carryover_children(root, ancestor_session_id)
    sources = [
        Source(name="claude-code", path=sibling_a),
        Source(name="claude-code", path=sibling_b),
    ]

    result = await parse_sources_archive(archive_root, sources)
    assert result.parse_failures == 0

    conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        native_ids = {str(r[0]) for r in conn.execute("SELECT native_id FROM sessions")}
        carryover_links = {
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                "SELECT src_session_id, dst_native_id FROM session_links WHERE dst_native_id = ?",
                (ancestor_session_id,),
            )
        }
    finally:
        conn.close()

    # Both siblings' real content is indexed under its OWN identity, and
    # neither carryover fragment collides with the other, with either
    # sibling's real content, or with the ancestor's own (never-ingested
    # here) identity -- the core claim: four DISTINCT identities, not one
    # cohort colliding on the ancestor's `logical_source_key`.
    assert native_ids == {
        "sibling-a",
        "sibling-b",
        f"{ancestor_session_id}:sibling-a",
        f"{ancestor_session_id}:sibling-b",
    }
    assert ancestor_session_id not in native_ids
    # Both carryover fragments record the SAME ancestor as an unresolved
    # lineage edge instead -- exactly the "use session_links, don't collide
    # identity" fix bd polylogue-jc4q calls for.
    assert carryover_links == {
        (f"claude-code-session:{ancestor_session_id}:sibling-a", ancestor_session_id),
        (f"claude-code-session:{ancestor_session_id}:sibling-b", ancestor_session_id),
    }


@pytest.mark.asyncio
async def test_reingesting_identical_bytes_resolves_to_the_same_raw_id(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """AC #1: a deterministic fixture re-acquired twice with byte-identical
    bytes must yield the SAME raw_id (and therefore the same downstream
    logical_source_key) both times -- not a second, native_id-differentiated
    raw row that later collides with the first in membership classification.
    """
    archive_root = one_shot_workspace_env["archive_root"]
    root = tmp_path / "sessions"
    root.mkdir()
    _parent_file, child_file = _write_carryover_chain(root)
    sources = [Source(name="claude-code", path=child_file)]

    await parse_sources_archive(archive_root, sources)
    first_rows = _raw_rows_for_path(archive_root / "source.db", str(child_file))
    assert len(first_rows) == 1
    first_raw_id = first_rows[0][0]

    # Re-ingest the exact same byte-identical file again (simulating the
    # daemon's later catch-up revisit of unchanged content).
    await parse_sources_archive(archive_root, sources)
    second_rows = _raw_rows_for_path(archive_root / "source.db", str(child_file))

    assert len(second_rows) == 1, f"re-ingest must not create a second raw row, got {second_rows}"
    assert second_rows[0][0] == first_raw_id, "raw_id must be deterministic across re-acquisitions of identical bytes"
    assert _membership_rows(archive_root / "source.db", first_raw_id) == {
        ("claude-code-session:parent-session:child-session", "parent-session:child-session"),
        ("claude-code-session:child-session", "child-session"),
    }


@pytest.mark.asyncio
async def test_batched_grouped_ingest_commits_census_before_next_raw(
    tmp_path: Path,
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped raw publication must not wait behind the index message batch."""
    monkeypatch.setenv("POLYLOGUE_INGEST_PARSE_WORKERS", "1")
    # Keep the index transaction open across both files. The default production
    # threshold has the same shape for this small demo-sized input.
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "1000")
    archive_root = one_shot_workspace_env["archive_root"]
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _first_parent, first_child = _write_carryover_chain(first_root)
    _second_parent, second_child = _write_carryover_chain(second_root, session_prefix="second-")

    original_census = ArchiveStore.replace_raw_membership_census
    census_calls = 0

    def require_source_transaction(
        archive: ArchiveStore,
        raw_id: str,
        sessions: list[ParsedSession] | None,
        **kwargs: Any,
    ) -> None:
        nonlocal census_calls
        census_calls += 1
        assert kwargs["manage_transaction"] is True
        original_census(archive, raw_id, sessions, **kwargs)

    monkeypatch.setattr(ArchiveStore, "replace_raw_membership_census", require_source_transaction)

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=first_child), Source(name="claude-code", path=second_child)],
        parse_workers=1,
    )

    # Anti-vacuity: this calls the production grouped parser, raw admission,
    # source census, and index writer with a positive batch threshold. Removing
    # the source-tier transaction ownership fix fails at the first real census
    # call, and without that guard the next file's publisher blocks on the
    # census transaction's source.db write lock.
    assert result.parse_failures == 0
    assert census_calls == 4
    assert result.counts["sessions"] == 4
    assert len(_raw_rows_for_path(archive_root / "source.db", str(first_child))) == 1
    assert len(_raw_rows_for_path(archive_root / "source.db", str(second_child))) == 1
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 4


def _write_stem_identity_husk(root: Path, stem: str) -> Path:
    """A JSON document under a Claude Code project whose only identity is its filename.

    The record shape satisfies dispatch's loose "looks like a message list"
    admission but carries no record the Claude Code parser recognizes, so the
    parse falls back to the discovery walk's ``fallback_id`` (the filename
    stem) and yields a session with zero authored messages.
    """
    path = root / ".claude" / "projects" / "proj" / "notes" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "tool output shaped like a chat"}]}),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("stem", ["toolu_01ABCDEFGHIJKLMNOPQRSTUV", "wf_run_1"])
async def test_archive_ingest_refuses_filename_stem_identity_without_authored_content(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path], stem: str
) -> None:
    """Canonical one-shot intake must not mint fragment-identity husks.

    ``require_positive_conversational_evidence`` is the archive's admission law
    for "parsed, but no conversation is present". Every other production write
    path applies it -- the daemon decode worker, live batch convergence, the
    incremental append route, and offline replay. ``parse_sources_archive``
    (reached from the public ``Polylogue.parse_file``/``parse_sources`` API and
    the demo seeder) did not, so a JSON document that merely satisfies the
    loose "has a messages list" shape became a session keyed on its own
    filename stem with zero messages -- the ``toolu_*``/``wf_*`` fragment
    phantom class this bead exists to make unrepresentable. (The ``*.meta``
    sibling shape is refused earlier, by its own declared artifact rule.)

    Anti-vacuity: dropping the authored-evidence gate writes one session
    row per parametrized stem, each ``provider_session_id`` equal to the stem
    and ``COUNT(*) FROM messages`` zero.
    """
    archive_root = one_shot_workspace_env["archive_root"]
    husk = _write_stem_identity_husk(tmp_path / "corpus", stem)

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=husk)],
        parse_workers=1,
    )

    assert result.parse_failures == 0
    assert result.counts.get("sessions", 0) == 0
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_archive_ingest_still_admits_a_real_session_through_the_same_gate(
    tmp_path: Path, one_shot_workspace_env: dict[str, Path]
) -> None:
    """The evidence gate must not refuse a transcript that carries authored content.

    Pairs with the husk law above: the refusal is keyed on absent authored
    content, not on the file's location or its identity shape. Anti-vacuity:
    widening the gate to reject on anything the husk case has in common with
    this one (same directory, same provider, same one-shot route) turns this
    red.
    """
    archive_root = one_shot_workspace_env["archive_root"]
    transcript = _write_session_shaped_workflow_journal(tmp_path / "sessions")

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code", path=transcript)],
        parse_workers=1,
    )

    assert result.parse_failures == 0
    assert result.counts["sessions"] == 1
