"""Raw materialization hands its current output to canonical session derivation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.raw_observation_derivation import converge_raw_observations
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.mark.asyncio
async def test_raw_materialization_hands_current_output_to_the_canonical_session_derivation(
    tmp_path: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """Raw admission targets its actual output after releasing the writer lease.

    Anti-vacuity: removing the raw-to-session query or calling the profile
    owner before raw publication leaves the materialized session without its
    canonical profile partition. Repeating the handoff proves that inspection
    rather than the intake item is the source of idempotence.
    """
    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session("raw-profile-handoff", (("user", "question"), ("assistant", "answer"))),
            source_path="raw-profile-handoff.jsonl",
            acquired_at_ms=1,
        )

    result = converge_raw_observations(archive_root, source_roots=(), limit=1, compute_adapter=bounded_compute_adapter)
    assert result.done == 1 and result.failed == 0
    session_ids = daemon_cli._raw_materialized_session_ids(archive_root, raw_id)
    assert session_ids == ("codex-session:raw-profile-handoff",)

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, composed.callback)
        with _connect(archive_root / "index.db") as conn:
            first = tuple(
                conn.execute(
                    "SELECT session_id, input_content_hash FROM session_profiles WHERE session_id = ?",
                    session_ids,
                ).fetchall()
            )
        assert len(first) == 1

        await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, composed.callback)
        with _connect(archive_root / "index.db") as conn:
            second = tuple(
                conn.execute(
                    "SELECT session_id, input_content_hash FROM session_profiles WHERE session_id = ?",
                    session_ids,
                ).fetchall()
            )
        assert second == first
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_two_accepted_revisions_survive_one_periodic_profile_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted R1/R2 marker inputs survive a coalesced profile publication."""
    from polylogue.core.enums import Origin, Role
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.pipeline.services.ingest_batch import _core as ingest_batch_core
    from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.pipeline.services.parsing_models import ParseResult
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.runtime import RawSessionRecord
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)
    config = _config(archive_root)
    repository = SessionRepository(backend=SQLiteBackend(db_path=archive_root / "index.db"), archive_root=archive_root)
    service = ParsingService(repository=repository, archive_root=archive_root, config=config, ingest_workers=1)
    raw_notes: dict[str, str] = {}
    for revision, note in ((1, "first retained note"), (2, "second retained note")):
        payload_bytes = f"synthetic-revision-{revision}".encode()
        BlobStore(archive_root / "blob").write_from_bytes(payload_bytes)
        with sqlite3.connect(archive_root / "source.db") as source:
            raw_id = write_source_raw_session(
                source,
                origin=Origin.CODEX_SESSION,
                source_path=f"coalesced-{revision}.jsonl",
                source_index=0,
                payload=payload_bytes,
                acquired_at_ms=revision,
            )
        raw_notes[raw_id] = note

    def fresh_ingest(record: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        parsed = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="coalesced-profile",
            created_at="2026-01-01T00:00:00Z",
            messages=[
                ParsedMessage(provider_message_id="question", role=Role.USER, text="question"),
                ParsedMessage(
                    provider_message_id="note", role=Role.ASSISTANT, text=f"::note: {raw_notes[record.raw_id]}"
                ),
            ],
        )
        return IngestRecordResult(
            raw_id=record.raw_id,
            payload_provider=Provider.CODEX.value,
            validation_status="passed",
            outcome_code="success",
            sessions=[
                SessionWritePayload(
                    session_id="codex-session:coalesced-profile",
                    content_hash=str(session_content_hash(parsed)),
                    parsed_session=parsed,
                    message_count=len(parsed.messages),
                    raw_id=record.raw_id,
                )
            ],
        )

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fresh_ingest)
    monkeypatch.setattr(
        "polylogue.config.load_polylogue_config",
        lambda: type("Settings", (), {"schema_validation": "advisory", "sinex_mode": "off"})(),
    )
    try:
        for raw_id in raw_notes:
            await ingest_batch_core.process_ingest_batch(service, repository.backend, [raw_id], ParseResult(), None)
    finally:
        await repository.close()

    with sqlite3.connect(archive_root / "source.db") as source:
        retained = source.execute("SELECT sequence, raw_id FROM accepted_marker_inputs ORDER BY sequence").fetchall()
        assert len(retained) == 2
        assert retained[0][0] < retained[1][0]

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        periodic = await composed.callback(None)
        assert periodic.outcomes
        # Marker delivery intentionally advances one accepted source batch per
        # transaction. The next periodic tick consumes R2 without republishing
        # the already-current profile of the coalesced index revision.
        second = await composed.callback(None)
        assert second.outcomes
        assert all(item.key.domain != "session_profile" for item in second.outcomes)
        with sqlite3.connect(archive_root / "index.db") as index:
            assert index.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", ("codex-session:coalesced-profile",)
            ).fetchone() == (1,)
            assert (
                index.execute(
                    "SELECT 1 FROM session_profile_demand WHERE session_id = ?", ("codex-session:coalesced-profile",)
                ).fetchone()
                is None
            )
        with sqlite3.connect(archive_root / "user.db") as user:
            bodies = [str(row[0]) for row in user.execute("SELECT body_text FROM assertions ORDER BY body_text")]
            assert any("first retained note" in body for body in bodies)
            assert any("second retained note" in body for body in bodies)
        unchanged = await composed.callback(None)
        assert unchanged.made_no_publication_attempts
        assert unchanged.work.inspected == 0
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_raw_materialized_session_ids_exclude_stale_component_sessions_without_current_heads(
    tmp_path: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """Raw-to-profile handoff follows authoritative heads, not residual session rows.

    Anti-vacuity: querying ``sessions`` by raw component alone includes the
    deliberately orphaned split member below and schedules a non-current
    session partition.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload: list[dict[str, object]] = [
        {
            "id": native_id,
            "title": native_id,
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [native_id]},
                    },
                }
            },
        }
        for native_id in ("active", "stale")
    ]
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path="split.json",
            acquired_at_ms=1,
        )
    result = converge_raw_observations(archive_root, source_roots=(), limit=1, compute_adapter=bounded_compute_adapter)
    assert result.done == 1 and result.failed == 0
    with sqlite3.connect(archive_root / "index.db") as index:
        index.execute("DELETE FROM raw_revision_heads WHERE session_id = ?", ("chatgpt-export:stale",))
        index.commit()

    assert daemon_cli._raw_materialized_session_ids(archive_root, raw_id) == ("chatgpt-export:active",)
