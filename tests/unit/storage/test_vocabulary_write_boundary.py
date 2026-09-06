"""Every durable vocabulary is refused at the write boundary, not by the DDL.

Durable-tier DDL carries no enum-membership CHECK, so ``require_vocabulary``
at the source writers is the only thing standing between an out-of-vocabulary
string and a persisted row. One case per vocabulary per production route.

Anti-vacuity is executable rather than asserted:
``test_durable_ddl_admits_what_the_boundary_refuses`` writes the same rejected
value straight through SQLite and shows it lands. Deleting any
``require_vocabulary`` call below therefore turns that route's refusal test
red -- nothing downstream would catch the value.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.enums import ArtifactSupportStatus, Origin, Provider
from polylogue.storage.runtime.raw.records import ArtifactObservationRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    PendingPreParseRawAdmissionRequest,
    plan_raw_admission,
)
from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHookEvent,
    ArchiveSourceArtifact,
    record_capture_mode_observation,
    refine_raw_origin,
    upsert_raw_artifact,
    write_source_raw_session,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.queries.artifacts import artifact_observation_params
from polylogue.storage.sqlite.queries.raw_writes import execute_raw_admission_plan_async

OUT_OF_VOCABULARY = "not-a-declared-member"

PAYLOAD = b'{"kind":"session"}'
BLOB_HASH = hashlib.sha256(PAYLOAD).digest()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def _artifact(**overrides: object) -> ArchiveSourceArtifact:
    fields: dict[str, Any] = {
        "artifact_id": "artifact-1",
        "origin": Origin.CLAUDE_CODE_SESSION,
        "source_path": "/captures/record.jsonl",
        "artifact_kind": "session_export",
        "classification_reason": "expected",
        "support_status": ArtifactSupportStatus.SUPPORTED_PARSEABLE,
        "first_observed_at_ms": 1,
        "last_observed_at_ms": 1,
    }
    fields.update(overrides)
    return ArchiveSourceArtifact(**fields)


def _raw_session(conn: sqlite3.Connection, **overrides: object) -> str:
    kwargs: dict[str, Any] = {
        "origin": Origin.CLAUDE_CODE_SESSION,
        "source_path": "/captures/record.jsonl",
        "source_index": 0,
        "payload": PAYLOAD,
        "acquired_at_ms": 1,
    }
    kwargs.update(overrides)
    return write_source_raw_session(conn, **kwargs)


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("raw_sessions", "raw_artifacts", "raw_hook_events", "raw_capture_observations", "source_items")
    }


# --- raw_sessions: the four vocabularies its columns carry ------------------


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("origin", {"origin": OUT_OF_VOCABULARY}),
        ("capture_mode", {"capture_mode": OUT_OF_VOCABULARY}),
        ("validation_status", {"validation_status": OUT_OF_VOCABULARY}),
        ("validation_mode", {"validation_mode": OUT_OF_VOCABULARY}),
    ],
)
def test_raw_session_writer_refuses_out_of_vocabulary(
    tmp_path: Path,
    field: str,
    overrides: dict[str, object],
) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        before = _row_counts(conn)
        with pytest.raises(ValueError, match=field):
            _raw_session(conn, **overrides)
        assert _row_counts(conn) == before
    finally:
        conn.close()


def test_raw_session_writer_refuses_out_of_vocabulary_artifact_origin(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        before = _row_counts(conn)
        with pytest.raises(ValueError, match="artifact.origin"):
            _raw_session(conn, artifact=_artifact(origin=OUT_OF_VOCABULARY))
        assert _row_counts(conn) == before
    finally:
        conn.close()


def test_raw_session_writer_refuses_out_of_vocabulary_support_status(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        before = _row_counts(conn)
        with pytest.raises(ValueError, match="artifact.support_status"):
            _raw_session(conn, artifact=_artifact(support_status=OUT_OF_VOCABULARY))
        assert _row_counts(conn) == before
    finally:
        conn.close()


def test_raw_session_writer_refuses_out_of_vocabulary_hook_origin(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        before = _row_counts(conn)
        with pytest.raises(ValueError, match="hook_event.origin"):
            _raw_session(
                conn,
                hook_event=ArchiveHookEvent(
                    hook_event_id="hook-1",
                    origin=OUT_OF_VOCABULARY,
                    source_path="/captures/record.jsonl",
                    event_type="source_opened",
                    payload={},
                    observed_at_ms=1,
                    session_native_id="session-1",
                ),
            )
        assert _row_counts(conn) == before
    finally:
        conn.close()


# --- the remaining source-tier routes --------------------------------------


def test_origin_refinement_refuses_out_of_vocabulary(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = _raw_session(conn, origin=Origin.UNKNOWN_EXPORT)
        with pytest.raises(ValueError, match="origin"):
            refine_raw_origin(conn, raw_id=raw_id, origin=OUT_OF_VOCABULARY)
        stored = conn.execute("SELECT origin FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        assert stored == Origin.UNKNOWN_EXPORT.value
    finally:
        conn.close()


def test_capture_mode_observation_refuses_out_of_vocabulary(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = _raw_session(conn)
        with pytest.raises(ValueError, match="capture_mode"):
            record_capture_mode_observation(
                conn,
                raw_id=raw_id,
                capture_mode=OUT_OF_VOCABULARY,
                observed_at_ms=1,
            )
        assert conn.execute("SELECT COUNT(*) FROM raw_capture_observations").fetchone()[0] == 0
    finally:
        conn.close()


def test_artifact_upsert_refuses_out_of_vocabulary_origin(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = _raw_session(conn)
        with pytest.raises(ValueError, match="artifact.origin"):
            upsert_raw_artifact(conn, raw_id, _artifact(origin=OUT_OF_VOCABULARY))
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()[0] == 0
    finally:
        conn.close()


def test_artifact_upsert_refuses_out_of_vocabulary_support_status(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        raw_id = _raw_session(conn)
        with pytest.raises(ValueError, match="artifact.support_status"):
            upsert_raw_artifact(conn, raw_id, _artifact(support_status=OUT_OF_VOCABULARY))
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()[0] == 0
    finally:
        conn.close()


def test_source_generation_publication_refuses_out_of_vocabulary_origin(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "source.db")
    try:
        with pytest.raises(ValueError, match="origin"):
            publish_source_generation(
                conn,
                source_generation_id="generation-1",
                manifest_digest="0" * 64,
                addressing_mode="path",
                coordinates=("/captures/record.jsonl",),
                observed_at_ms=1,
                origin=OUT_OF_VOCABULARY,
            )
        assert conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 0
    finally:
        conn.close()


def test_artifact_observation_projection_refuses_out_of_vocabulary_support_status() -> None:
    """The async artifact route validates again below pydantic's typed field."""
    record = ArtifactObservationRecord.model_construct(
        observation_id="observation-1",
        raw_id="raw-1",
        payload_provider=Provider.CLAUDE_CODE,
        source_path="/captures/record.jsonl",
        source_index=0,
        artifact_kind="session_export",
        classification_reason="expected",
        parse_as_session=True,
        schema_eligible=True,
        support_status=cast(ArtifactSupportStatus, OUT_OF_VOCABULARY),
        first_observed_at="1",
        last_observed_at="1",
    )
    with pytest.raises(ValueError, match="support_status"):
        artifact_observation_params(record)


# --- the async admission adapter --------------------------------------------


class _RefusesEveryStatement:
    """Stands in for the async connection; any use is a boundary that came too late."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"the async adapter reached conn.{name} before validating its vocabulary")


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("origin", {"origin": OUT_OF_VOCABULARY}),
        ("capture_mode", {"capture_mode": OUT_OF_VOCABULARY}),
    ],
)
def test_async_admission_refuses_out_of_vocabulary_before_any_sql(
    field: str,
    overrides: dict[str, Any],
) -> None:
    # Planned from a valid request, then degraded: the planner shares the
    # writer's boundary, and this test is about the async adapter's own.
    plan = plan_raw_admission(
        PendingPreParseRawAdmissionRequest(
            origin=Origin.CHATGPT_EXPORT,
            capture_mode=Provider.CHATGPT,
            source_path="/captures/source.json",
            source_index=0,
            blob_hash=BLOB_HASH,
            blob_size=len(PAYLOAD),
            acquired_at_ms=1,
        )
    )
    degraded = replace(plan, request=replace(plan.request, **overrides))

    with pytest.raises(ValueError, match=field):
        asyncio.run(execute_raw_admission_plan_async(cast(Any, _RefusesEveryStatement()), degraded, 0))


# --- anti-vacuity: the DDL below these routes accepts what they refuse ------


def test_durable_ddl_admits_what_the_boundary_refuses(tmp_path: Path) -> None:
    """Vocabulary membership is unenforced in SQL, so the boundary is the only check."""
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, validation_status, validation_mode,
                source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                logical_source_key, revision_kind, source_revision, acquisition_generation,
                revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'full', ?, 0, 'quarantined')
            """,
            (
                "raw-unchecked",
                OUT_OF_VOCABULARY,
                OUT_OF_VOCABULARY,
                OUT_OF_VOCABULARY,
                OUT_OF_VOCABULARY,
                "/captures/record.jsonl",
                0,
                BLOB_HASH,
                len(PAYLOAD),
                1,
                "logical-key",
                BLOB_HASH.hex(),
            ),
        )
        stored = conn.execute(
            "SELECT origin, capture_mode, validation_status, validation_mode FROM raw_sessions"
        ).fetchone()
        assert tuple(stored) == (OUT_OF_VOCABULARY,) * 4
    finally:
        conn.close()
