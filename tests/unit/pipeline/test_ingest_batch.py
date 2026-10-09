"""Focused tests for retained ingest publication through the batch host."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

import polylogue.storage.sqlite.archive_tiers.write as archive_write
from polylogue.archive.message.roles import Role
from polylogue.config import Config
from polylogue.core.enums import ArtifactSupportStatus, BlockType, Origin, Provider
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.daemon.status import raw_failure_info_for_root
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services.ingest_batch import process_ingest_batch
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sinex.models import PublicationMode
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.revision_backfill import RetainedReplayOutcome
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.repository import SessionRepository
from polylogue.storage.search.cache import get_cache_stats
from polylogue.storage.search.runtime import search_messages
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    upsert_raw_artifact,
    write_source_raw_session,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.index_writer import (
    close_fixture_index_connection,
    fixture_index_mutation_scope,
    write_fixture_index_session,
)
from tests.infra.live_ingest import prepared_live_convergence_owner


def test_timestamp_normalization_preserves_provider_and_session_identity() -> None:
    parsed = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="a6216592-1a36-4748-9198-64b8db6ec05b",
        title="Session",
        messages=[],
    )

    normalized = normalize_session_timestamps(parsed)

    assert str(make_session_id("claude-code-session", normalized.provider_session_id)) == (
        "claude-code-session:a6216592-1a36-4748-9198-64b8db6ec05b"
    )
    assert normalized.source_name == Provider.CLAUDE_CODE
    assert parsed.source_name == Provider.CLAUDE_CODE


def test_timestamp_normalization_replaces_malformed_session_time_with_message_evidence() -> None:
    parsed = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="malformed-session-time",
        created_at="not-a-timestamp",
        updated_at="also-not-a-timestamp",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="evidence",
                timestamp="2026-06-01T12:00:00Z",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="evidence")],
            )
        ],
    )

    normalized = normalize_session_timestamps(parsed, fallback_timestamp="2020-01-01T00:00:00Z")

    assert normalized.created_at == "2026-06-01T12:00:00+00:00"
    assert normalized.updated_at == "2026-06-01T12:00:00+00:00"


def test_stale_observation_repair_derives_created_time_from_session_event(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    conn = connect_measured(archive_root / "index.db")
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="stale-event-observation",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="summary",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="summary")],
                )
            ],
        )
        session_id = write_fixture_index_session(conn, session)
        candidate = session.model_copy(
            update={
                "session_events": [
                    ParsedSessionEvent(
                        event_type="hermes_llm_request_span",
                        timestamp="2026-07-01T00:00:00Z",
                    )
                ]
            }
        )
        with fixture_index_mutation_scope(conn):
            archive_write._retain_stale_session_observations(
                conn, session_id, candidate, fallback_timestamp="2020-01-01T00:00:00Z"
            )
        row = conn.execute("SELECT created_at_ms FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    finally:
        close_fixture_index_connection(conn)

    assert row[0] == 1_782_864_000_000


_CANONICAL_CODEX_PAYLOAD = (
    Path(__file__).parents[2] / "fixtures" / "origin-capability" / "codex-session.jsonl"
).read_bytes()


@asynccontextmanager
async def _canonical_parsing_service(
    root: Path,
    *,
    repository: SessionRepository | None = None,
) -> AsyncIterator[ParsingService]:
    """A real parsing service whose publication is the canonical retained owner."""
    owned_repository = repository is None
    repo = repository or SessionRepository(backend=SQLiteBackend(db_path=root / "index.db"), archive_root=root)
    try:
        async with prepared_live_convergence_owner(root) as owner:
            yield ParsingService(
                repository=repo,
                archive_root=root,
                config=Config(archive_root=root, render_root=root / "render", sources=[]),
                retained_runner=owner.ingest_retained_raw_ids,
            )
    finally:
        if owned_repository:
            await repo.close()


def _seed_retained_raw(root: Path, *, origin: Origin, source_path: str, payload: bytes) -> str:
    BlobStore(root / "blob").write_from_bytes(payload)
    with sqlite3.connect(root / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=origin,
            source_path=source_path,
            canonical_source_path=source_path,
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
        )
        conn.commit()
    return raw_id


class _PublicationConfig:
    """The loaded config with only its publication and validation modes pinned."""

    def __init__(self, loaded: object, mode: str) -> None:
        self._loaded = loaded
        self.schema_validation = "advisory"
        self.sinex_mode = mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loaded, name)


def _publication_mode(monkeypatch: pytest.MonkeyPatch, mode: str = "off") -> None:
    import polylogue.config

    loaded = polylogue.config.load_polylogue_config

    def pinned(*args: Any, **kwargs: Any) -> _PublicationConfig:
        return _PublicationConfig(loaded(*args, **kwargs), mode)

    monkeypatch.setattr("polylogue.config.load_polylogue_config", pinned)


@pytest.mark.asyncio
async def test_process_ingest_batch_requires_its_retained_owner(tmp_path: Path) -> None:
    """The public batch route publishes only through a supplied retained owner.

    Anti-vacuity: falling back to a local parse/write when no owner is
    supplied would publish outside the canonical owner and this refusal
    would disappear.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="no-owner.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(
        repository=repository,
        archive_root=tmp_path,
        config=Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[]),
    )
    try:
        with pytest.raises(PermissionError, match="retained Raw owner"):
            await process_ingest_batch(service, [raw_id], ParseResult(), None)
    finally:
        await repository.close()
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [member.value for member in PublicationMode if member is not PublicationMode.OFF])
async def test_process_ingest_batch_refuses_sinex_publication_before_the_retained_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A non-OFF publication mode is refused before any local projection.

    Replaces the predecessor primary-mode laws: the retained route has no
    accepted-marker/outbox producer, so it must refuse before the owner
    publishes Index or FTS rows.
    """
    from polylogue.sinex.material_adapter import PublicationEncodingError

    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="sinex.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    _publication_mode(monkeypatch, mode)
    calls: list[tuple[str, ...]] = []

    async def owner_must_not_run(raw_ids: Sequence[str], **_refusal_handlers: object) -> RetainedReplayOutcome:
        calls.append(tuple(raw_ids))
        return RetainedReplayOutcome()

    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"), archive_root=tmp_path)
    service = ParsingService(
        repository=repository,
        archive_root=tmp_path,
        config=Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[]),
        retained_runner=owner_must_not_run,
    )
    try:
        with pytest.raises(PublicationEncodingError, match="accepted-marker and outbox producer"):
            await process_ingest_batch(service, [raw_id], ParseResult(), None)
    finally:
        await repository.close()
    assert calls == []
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone() == (0,)


@pytest.mark.asyncio
async def test_process_ingest_batch_publishes_and_invalidates_search_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed session from the canonical route is searchable, never served stale.

    Ported from the predecessor sync-route law: the search cache epoch covers
    ordinary ingest writes within one generation, so publication that changes
    a session must advance it. Anti-vacuity: drop the invalidation and the
    cached empty result is served after publication.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="search.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    _publication_mode(monkeypatch)
    needle = "witness"
    before = search_messages(needle, archive_root=tmp_path, db_path=tmp_path / "index.db", limit=10)
    assert before.hits == ()
    version_before = get_cache_stats()["cache_version"]
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await process_ingest_batch(service, [raw_id], parse_result, None)
    assert len(parse_result.processed_ids) == 1
    assert get_cache_stats()["cache_version"] > version_before
    after = search_messages(needle, archive_root=tmp_path, db_path=tmp_path / "index.db", limit=10)
    assert {hit.session_id for hit in after.hits} == set(parse_result.processed_ids)


@pytest.mark.asyncio
async def test_process_ingest_batch_public_route_retires_deferred_cas_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public batch route supersedes deferred CAS evidence after publication."""
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    payload = (Path(__file__).parents[2] / "fixtures" / "chatgpt" / "native-conversation-v1.json").read_bytes()
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CHATGPT_EXPORT, source_path="public-batch.json", payload=payload
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        upsert_raw_artifact(
            conn,
            raw_id,
            ArchiveSourceArtifact(
                artifact_id="public-deferred-cas",
                origin=Origin.CHATGPT_EXPORT,
                source_path="public-batch.json",
                source_index=0,
                artifact_kind=RawFailureEvidenceKind.DEFERRED_CAS_FRONTIER.value,
                classification_reason="deferred CAS",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                parse_as_session=True,
                schema_eligible=True,
                first_observed_at_ms=1,
                last_observed_at_ms=1,
            ),
        )
        conn.commit()
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await process_ingest_batch(service, [raw_id], parse_result, None)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)
        assert conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (
            RawFailureEvidenceKind.TERMINAL_SUPERSEDED_DEFERRED_CAS_FRONTIER.value,
            "unknown",
        )
    assert parse_result.processed_ids


@pytest.mark.asyncio
async def test_process_ingest_batch_off_mode_supports_repository_without_source_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF publication keeps an index-only repository usable and records accepted input."""
    await asyncio.to_thread(initialize_active_archive_root, tmp_path)
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="index-only.jsonl", payload=_CANONICAL_CODEX_PAYLOAD
    )
    repository = SessionRepository(backend=SQLiteBackend(db_path=tmp_path / "index.db"))
    assert repository.source_backend is None
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    try:
        async with _canonical_parsing_service(tmp_path, repository=repository) as service:
            await process_ingest_batch(service, [raw_id], parse_result, None)
    finally:
        await repository.close()

    assert len(parse_result.processed_ids) == 1
    (session_id,) = parse_result.processed_ids
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone() == (2,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)
        assert conn.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert conn.execute("SELECT raw_id FROM accepted_marker_inputs").fetchall() == [(raw_id,)]
    assert parse_result.parse_failures == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [b"{\n", b'{"type":"session_meta"}\n{nope}\n'],
    ids=["undecodable-sole-record", "undecodable-later-record"],
)
async def test_process_ingest_batch_public_route_persists_corrupt_input_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    """A complete record that does not decode is terminal corrupt input.

    The canonical route settles a non-empty undecodable record stream as
    ``terminal_corrupt_input`` and makes it status-readable. Removing the
    census's decode-refusal settlement (``_persist_terminal_raw_refusal``)
    turns this red.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="public-corrupt.jsonl", payload=payload
    )
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await process_ingest_batch(service, [raw_id], parse_result, None)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        artifact = conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = ?", (raw_id,)
        ).fetchone()
    assert artifact == (RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT.value, "decode_failed")
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 1
    assert lifecycle.unexplained == 0
    assert lifecycle.blocking is False
    status = raw_failure_info_for_root(tmp_path)
    assert status["terminal_rejections"] == 1
    assert status["unexplained_failures"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [b"", b'{"type":"session_meta"'],
    ids=["zero-record-stream", "unterminated-sole-record"],
)
async def test_process_ingest_batch_public_route_retains_unadmitted_tail_without_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    """A zero-record JSONL prefix retains clean non-session authority.

    The shared JSONL parse-prefix rule (``jsonl_parse_prefix_size``) leaves an
    unterminated tail out. The retained census reads immutable bytes and cannot
    decide whether intake should retry or classify a disappeared file.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    raw_id = _seed_retained_raw(
        tmp_path, origin=Origin.CODEX_SESSION, source_path="public-recordless.jsonl", payload=payload
    )
    _publication_mode(monkeypatch)
    parse_result = ParseResult()
    async with _canonical_parsing_service(tmp_path) as service:
        await process_ingest_batch(service, [raw_id], parse_result, None)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchone() == (0,)
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1, None)
        assert conn.execute(
            "SELECT status, member_count FROM raw_membership_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("non_session", 0)
        assert conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("complete", "[]")
    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.terminal == 0
    assert lifecycle.unexplained == 0
    assert parse_result.parse_failures == 0
