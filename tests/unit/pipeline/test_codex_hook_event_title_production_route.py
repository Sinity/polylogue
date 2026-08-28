"""Acquired ``codex_thread_title`` hook events reach Codex assembly (polylogue-zco96).

Both routes that will run during the production reindex are driven end to
end here: ``_process_ingest_batch_sync`` (pipeline ingest) and
``_enrich_retained_parse_results`` (retained-raw replay). Neither test hands
assembly a ``hook_event_titles`` key -- the evidence is produced by the real
writer (``write_codex_thread_state_evidence`` over a real ``state_5.sqlite``
snapshot) and must be found by the route itself. Sever either consumer and
both sessions fall back to the content-heuristic first-prompt title, which
is exactly what these assertions reject.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider, TitleSource
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.sources.codex_state_evidence import write_codex_thread_state_evidence
from polylogue.sources.parsers.codex_state import parse_codex_state_db
from polylogue.sources.revision_backfill import _enrich_retained_parse_results
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_THREAD_ID = "3f2a9c10-7b41-4d55-9a6e-1c2b3d4e5f60"
_CURATED_TITLE = "Curated thread title from state db"
_FIRST_PROMPT = "opening prompt that must not become the title"


def _rollout_bytes() -> bytes:
    lines = [
        {"type": "session_meta", "payload": {"id": _THREAD_ID, "timestamp": "2026-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _FIRST_PROMPT}],
            },
        },
    ]
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode("utf-8")


def _write_state_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title TEXT, cwd TEXT,
                created_at_ms INTEGER, updated_at_ms INTEGER, source TEXT,
                model TEXT, agent_nickname TEXT, agent_role TEXT, archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT, child_thread_id TEXT, status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO threads (id, title, cwd, created_at_ms, updated_at_ms, source, model, "
            "agent_nickname, agent_role, archived) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_THREAD_ID, _CURATED_TITLE, "/repo", 1000, 2000, "cli", "gpt-synthetic", None, None, 0),
        )
        conn.commit()


def _archive_with_acquired_title_evidence(tmp_path: Path) -> Path:
    """Produce the durable hook event through its only production writer."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    state_path = tmp_path / "state_5.sqlite"
    _write_state_db(state_path)
    snapshot = parse_codex_state_db(state_path, immutable=True)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        write_codex_thread_state_evidence(
            archive,
            snapshot,
            source_path=str(state_path),
            acquired_at_ms=1_767_000_000_000,
        )
        archive.commit()
    return archive_root


def test_pipeline_ingest_resolves_acquired_hook_event_title(tmp_path: Path) -> None:
    archive_root = _archive_with_acquired_title_evidence(tmp_path)
    content = _rollout_bytes()
    store = BlobStore(archive_root / "blob")
    raw_id, blob_size = store.write_from_bytes(content)
    record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path=str(tmp_path / "sessions" / f"rollout-{_THREAD_ID}.jsonl"),
        payload_provider=Provider.CODEX,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
    )
    assert record.sidecar_snapshot is None, "the route, not the test, must supply the evidence"

    _process_ingest_batch_sync(
        [record],
        db_path=archive_root / "index.db",
        archive_root_str=str(archive_root),
        blob_root_str=str(store.root),
        validation_mode="advisory",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    with sqlite3.connect(archive_root / "index.db") as index_conn:
        row = index_conn.execute(
            "SELECT title, title_source FROM sessions WHERE native_id = ?",
            (_THREAD_ID,),
        ).fetchone()
    assert row is not None, "expected the codex session to be materialized"
    assert row[0] == _CURATED_TITLE
    assert row[1] == TitleSource.ORIGIN.value


def test_retained_replay_resolves_acquired_hook_event_title(tmp_path: Path) -> None:
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.sources.dispatch import parse_stream_payload

    archive_root = _archive_with_acquired_title_evidence(tmp_path)
    content = _rollout_bytes()
    source_path = str(tmp_path / "sessions" / f"rollout-{_THREAD_ID}.jsonl")
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=content,
            source_path=source_path,
            acquired_at_ms=1_767_000_000_000,
        )
        archive.commit()

    sessions = parse_stream_payload(
        Provider.CODEX,
        [json.loads(line) for line in content.decode("utf-8").splitlines()],
        _THREAD_ID,
        source_path=source_path,
    )
    assert sessions and sessions[0].title != _CURATED_TITLE

    descriptors = {raw_id: (Provider.CODEX, "", source_path, RawRevisionKind.FULL, len(content), _THREAD_ID)}
    results: dict[str, object] = {raw_id: (sessions, len(content), RawRevisionKind.FULL)}
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _enrich_retained_parse_results(archive, descriptors=descriptors, results=results)  # type: ignore[arg-type]

    enriched = results[raw_id][0]  # type: ignore[index]
    assert enriched[0].title == _CURATED_TITLE
    assert enriched[0].title_source is TitleSource.ORIGIN
