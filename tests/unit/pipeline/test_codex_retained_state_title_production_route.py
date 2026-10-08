"""Projected Codex thread titles reach Codex assembly.

The supplied resident owner drives both first ingestion and retained replay
end to end here. Neither route hands
assembly a ``retained_state_titles`` key -- the evidence is produced by the
real writer (``prepare_codex_state_source_terminal`` over a real ``state_5.sqlite``
export) and must be found by the route itself. Sever either consumer and both
sessions fall back to the content-heuristic first-prompt title, which is
exactly what these assertions reject.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, TitleSource
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.agent_thread_state import (
    read_provenance,
    read_spawn_edges,
    read_thread_titles,
    thread_id_from_context_ref,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

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


def _write_state_db(
    path: Path,
    *,
    thread_id: str = _THREAD_ID,
    title: str = _CURATED_TITLE,
    edge: tuple[str, str, str] | None = None,
) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
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
            (thread_id, title, "/repo", 1000, 2000, "cli", "gpt-synthetic", None, None, 0),
        )
        if edge is not None:
            conn.execute("INSERT INTO thread_spawn_edges VALUES (?, ?, ?)", edge)
        conn.commit()


def _archive_with_retained_state_export(tmp_path: Path) -> Path:
    """Retain the state export and project it through its production route."""
    archive_root = tmp_path / "archive"
    state_path = tmp_path / "state_5.sqlite"
    _write_state_db(state_path)
    _record_state_export(archive_root, state_path, acquired_at_ms=1_767_000_000_000)
    return archive_root


async def _record_state_export_async(archive_root: Path, state_path: Path, *, acquired_at_ms: int) -> None:
    """Acquire the actual logical export and settle its original resident owner."""

    def acquire() -> str:
        bootstrap_archive_root(archive_root)
        store = BlobStore(archive_root / "blob")
        export = snapshot_sqlite_to_blob(state_path, store)
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=store.blob_path(export.blob_hash).read_bytes(),
                source_path=str(state_path),
                canonical_source_path=str(state_path),
                acquired_at_ms=acquired_at_ms,
                captured_profile_key=export.captured_profile_key,
            )
            archive.commit()
            return raw_id

    raw_id = await run_archive_fixture_write(archive_root, acquire)
    async with prepared_live_convergence_owner(archive_root) as owner:
        (await owner.replay_retained_raw_ids((raw_id,))).require_complete()


def _record_state_export(archive_root: Path, state_path: Path, *, acquired_at_ms: int) -> None:
    asyncio.run(_record_state_export_async(archive_root, state_path, acquired_at_ms=acquired_at_ms))


async def _record_rollout(archive_root: Path, source_path: str, *, ingest: bool) -> None:
    def acquire() -> str:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_rollout_bytes(),
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=1_767_000_000_000,
            )
            archive.commit()
            return raw_id

    raw_id = await run_archive_fixture_write(archive_root, acquire)
    async with prepared_live_convergence_owner(archive_root) as owner:
        if ingest:
            (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        else:
            (await owner.replay_retained_raw_ids((raw_id,))).require_complete()


def test_resident_ingest_resolves_the_projected_state_title(tmp_path: Path) -> None:
    archive_root = _archive_with_retained_state_export(tmp_path)
    asyncio.run(_record_rollout(archive_root, str(tmp_path / "sessions" / f"rollout-{_THREAD_ID}.jsonl"), ingest=True))

    with closing(sqlite3.connect(archive_root / "index.db")) as index_conn:
        row = index_conn.execute(
            "SELECT title, title_source FROM sessions WHERE native_id = ?",
            (_THREAD_ID,),
        ).fetchone()
    assert row is not None, "expected the codex session to be materialized"
    assert row[0] == _CURATED_TITLE
    assert row[1] == TitleSource.ORIGIN.value


def test_retained_replay_resolves_the_projected_state_title(tmp_path: Path) -> None:
    from polylogue.sources.dispatch import parse_stream_payload

    archive_root = _archive_with_retained_state_export(tmp_path)
    content = _rollout_bytes()
    source_path = str(tmp_path / "sessions" / f"rollout-{_THREAD_ID}.jsonl")
    sessions = parse_stream_payload(
        Provider.CODEX,
        [json.loads(line) for line in content.decode("utf-8").splitlines()],
        _THREAD_ID,
        source_path=source_path,
    )
    assert sessions and sessions[0].title != _CURATED_TITLE

    asyncio.run(_record_rollout(archive_root, source_path, ingest=False))
    with closing(sqlite3.connect(archive_root / "index.db")) as index_conn:
        title = index_conn.execute(
            "SELECT title, title_source FROM sessions WHERE native_id = ?", (_THREAD_ID,)
        ).fetchone()
    assert title == (_CURATED_TITLE, TitleSource.ORIGIN.value)


def test_unknown_export_codex_raw_publishes_under_its_resolved_provider(tmp_path: Path) -> None:
    """Publication validates an UNKNOWN raw's carrier with the provider the worker resolved.

    Input: a Codex rollout retained as ``unknown-export`` for a thread with a
    retained state title. The worker resolves the raw to Codex and seals a
    Codex-evidence dependency digest.

    Wrong outcome prevented: the publisher recomputed that digest under the
    stored ``Provider.UNKNOWN``, the digests never matched, and every attempt
    was refused as retryable. Anti-vacuity: validate with the descriptor's
    provider in ``_prepared_retained_outcome`` and the final ``publish`` below
    returns False. A real title change between preparation and publication
    still refuses the carrier.
    """
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    archive_root = _archive_with_retained_state_export(tmp_path)

    async def exercise_original_creator() -> None:
        def acquire() -> str:
            with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
                raw_id = archive.write_raw_payload(
                    provider=Provider.UNKNOWN,
                    payload=_rollout_bytes(),
                    source_path=str(tmp_path / "sessions" / f"rollout-{_THREAD_ID}.jsonl"),
                    canonical_source_path=str(tmp_path / "sessions" / f"rollout-{_THREAD_ID}.jsonl"),
                    acquired_at_ms=1_767_000_000_001,
                )
                archive.commit()
                return raw_id

        raw_id = await run_archive_fixture_write(archive_root, acquire)
        async with prepared_live_convergence_owner(archive_root) as owner:
            # Establish genuine preparatory census/classification before holding
            # the selected parser artifact for the stale-dependency experiment.
            (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

            retained: list[RawObservationReplacement] = []

            def retire(replacement: RawObservationReplacement) -> None:
                primary = sys.exception()
                try:
                    replacement.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "title experiment and physical cleanup failed", [primary, cleanup]
                        ) from primary
                    raise
                retained.remove(replacement)

            def exercise() -> None:
                adapter, index_path, _index_destination = owner._archive.destination_adapter()
                assert isinstance(adapter, RawObservationDerivation)
                frame = raw_observation_frame(archive_root, raw_ids=(raw_id,), index_db_path=index_path)
                stale = adapter.compute(frame, raw_id, replay_current=True)
                retained.append(stale)
                try:
                    assert stale.prepared_inputs is not None
                    artifact = stale.prepared_inputs[raw_id].prepared_artifact
                    assert artifact is not None and artifact.resolved_provider is Provider.CODEX
                    state_path = tmp_path / "state_5.sqlite"
                    state_path.unlink()
                    _write_state_db(state_path, title="Renamed thread title")

                    def acquire_state() -> str:
                        store = BlobStore(archive_root / "blob")
                        export = snapshot_sqlite_to_blob(state_path, store)
                        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
                            state_raw = archive.write_raw_payload(
                                provider=Provider.CODEX,
                                payload=store.blob_path(export.blob_hash).read_bytes(),
                                source_path=str(state_path),
                                canonical_source_path=str(state_path),
                                acquired_at_ms=1_767_000_000_002,
                                captured_profile_key=export.captured_profile_key,
                            )
                            archive.commit()
                            return state_raw

                    state_raw = admit_stage_write("fixture.title.changed-state.acquire", acquire_state)

                    def select_original(reader: PreparedSessionSourceRead) -> tuple[str, ...]:
                        # The selection hook runs once per continued phase of
                        # one compute; it always selects exactly the state raw.
                        expanded, _keys = reader.expand_raw_membership_selection((state_raw,))
                        assert state_raw in expanded
                        return (state_raw,)

                    while True:
                        prepared = adapter.compute(
                            frame, state_raw, replay_current=True, select_retained_raw_ids=select_original
                        )
                        retained.append(prepared)

                        def publish_state(current: RawObservationReplacement = prepared) -> bool:
                            return adapter.publish(frame, current)

                        try:
                            published = admit_stage_write("fixture.title.changed-state.publish", publish_state)
                        finally:
                            retire(prepared)
                        if published:
                            break
                        # A committed prerequisite phase is progress the next
                        # preparation continues from, exactly as the derivation
                        # kernel decides; anything else is a refusal.
                        if not adapter.publication_advanced(prepared):
                            raise RetainedPreparationRetryableError("changed state refused without actual progress")
                    # The stale preparation's original observers moved: its
                    # publication refuses with the typed stale-seal error,
                    # exactly as a moved active Index does.
                    with pytest.raises(ReferenceSealStaleError):
                        admit_stage_write("fixture.title.stale.publish", lambda: adapter.publish(frame, stale))
                finally:
                    retire(stale)
                fresh = adapter.compute(frame, raw_id, replay_current=True)
                retained.append(fresh)
                try:
                    assert (
                        admit_stage_write("fixture.title.fresh.publish", lambda: adapter.publish(frame, fresh)) is True
                    )
                finally:
                    retire(fresh)

            await owner.run_prepared_sync(
                "fixture.title.original-provider",
                exercise,
                settlement_owners=lambda: tuple(retained),
                estimated_bytes=len(_rollout_bytes()),
            )

    asyncio.run(exercise_original_creator())
    with closing(sqlite3.connect(archive_root / "index.db")) as index_conn:
        row = index_conn.execute("SELECT title FROM sessions WHERE native_id = ?", (_THREAD_ID,)).fetchone()
    assert row == ("Renamed thread title",)


@pytest.mark.parametrize("initial_order", (("a", "b"), ("b", "a")))
def test_state_projection_keeps_disjoint_roots_and_omitted_evidence(
    tmp_path: Path, initial_order: tuple[str, str]
) -> None:
    """Production state admission never lets one root erase another's rows.

    A later snapshot in root A is complete only for A: it explicitly updates
    ``a-new`` while the omitted thread/edge stays readable as archived
    evidence.  Reversing the two initial acquisitions yields the same rows.
    """
    archive_root = tmp_path / "archive"
    root_a = tmp_path / "codex-a"
    root_b = tmp_path / "codex-b"
    root_a.mkdir()
    root_b.mkdir()
    initial_a = root_a / "state_5.sqlite"
    initial_b = root_b / "state_5.sqlite"
    _write_state_db(initial_a, thread_id="a-old", title="A old", edge=("a-old", "a-child", "spawned"))
    _write_state_db(initial_b, thread_id="b-thread", title="B title", edge=("b-thread", "b-child", "closed"))

    initial_exports = {"a": initial_a, "b": initial_b}
    for sequence, root in enumerate(initial_order, start=1):
        _record_state_export(archive_root, initial_exports[root], acquired_at_ms=sequence * 100)
    initial_a.unlink()
    _write_state_db(initial_a, thread_id="a-new", title="A new")
    _record_state_export(archive_root, initial_a, acquired_at_ms=300)

    with closing(sqlite3.connect(archive_root / "index.db")) as index_conn:
        threads = sorted(read_thread_titles(index_conn).items())
        edges = sorted(read_spawn_edges(index_conn))
        states = dict(
            index_conn.execute(
                "SELECT node_ref, association_state FROM work_evidence_nodes WHERE node_kind = 'execution-context'"
            ).fetchall()
        )
    # An object the newest snapshot of its own scope is silent about stays
    # readable, marked superseded -- another root's evidence is untouched.
    assert threads == [("a-new", "A new"), ("a-old", "A old"), ("b-thread", "B title")]
    assert edges == [("a-old", "a-child"), ("b-thread", "b-child")]
    assert {thread_id_from_context_ref(ref): state for ref, state in states.items()} == {
        "a-new": "resolved",
        "a-old": "superseded",
        "a-child": "superseded",
        "b-thread": "resolved",
        "b-child": "resolved",
    }
    initial_a.unlink()
    initial_b.unlink()


def test_state_projection_uses_receipt_order_for_a_b_a_observations(tmp_path: Path) -> None:
    """A re-observed old payload is newer evidence, not its first acquisition."""
    archive_root = tmp_path / "archive"
    state_path = tmp_path / "codex" / "state_5.sqlite"
    state_path.parent.mkdir()
    _write_state_db(state_path, thread_id="thread", title="A title")
    _record_state_export(archive_root, state_path, acquired_at_ms=100)
    state_path.unlink()
    _write_state_db(state_path, thread_id="thread", title="B title")
    _record_state_export(archive_root, state_path, acquired_at_ms=200)
    state_path.unlink()
    _write_state_db(state_path, thread_id="thread", title="A title")
    _record_state_export(archive_root, state_path, acquired_at_ms=300)

    with closing(sqlite3.connect(archive_root / "index.db")) as index_conn:
        provenance = read_provenance(index_conn)
    assert provenance is not None
    assert provenance.observed_at_ms == 300
