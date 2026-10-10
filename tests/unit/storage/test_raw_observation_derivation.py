"""Production raw-adapter laws independent of the legacy backlog census."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider, ValidationMode
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, converge
from polylogue.operations.raw_observation_derivation import (
    make_raw_observation_derivation,
    raw_observation_frame,
)
from polylogue.sources.revision_backfill import PreparedRevisionReplayResult, RevisionCensusResult
from polylogue.storage.derived.raw import RawFrame, RawObservationDerivation, RawObservationReplacement
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


def _chatgpt_payload(names: tuple[str, ...]) -> bytes:
    return json.dumps(
        [
            {
                "id": name,
                "title": name,
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
                            "content": {"content_type": "text", "parts": [name]},
                        },
                    },
                },
            }
            for name in names
        ]
    ).encode()


def test_source_census_activity_clocks_do_not_count_as_progress(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with _fixture_archive(tmp_path) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_AI,
            payload=b'{"uuid":"clock","chat_messages":[]}',
            source_path="clock.json",
            canonical_source_path="clock.json",
            acquired_at_ms=1,
        )
        archive.commit()

    def exercise(compute: BoundedComputeAdapter) -> None:
        adapter = make_raw_observation_derivation(tmp_path, compute_adapter=compute)
        before = adapter._census_state((raw_id,))
        with _fixture_archive(tmp_path) as archive:
            archive.source_connection.execute(
                "UPDATE raw_sessions SET parsed_at_ms=2,validated_at_ms=3 WHERE raw_id=?", (raw_id,)
            )
            archive.source_connection.commit()
        assert adapter._census_state((raw_id,)) == before
        with _fixture_archive(tmp_path) as archive:
            archive.source_connection.execute(
                "UPDATE raw_sessions SET validation_status='passed' WHERE raw_id=?", (raw_id,)
            )
            archive.source_connection.commit()
        assert adapter._census_state((raw_id,)) != before

    _run_raw_law(tmp_path, exercise)


@contextmanager
def _fixture_archive(root: Path) -> Iterator[ArchiveStore]:
    with write_lease("synthetic-raw-admission", archive_root=root):
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            yield archive


def _writer(root: Path, actor: str, work: Callable[[], bool]) -> bool:
    with write_lease(actor, archive_root=root):
        return work()


def _publish(
    adapter: RawObservationDerivation,
    frame: RawFrame,
    replacement: RawObservationReplacement,
    *,
    phase_receipt: Callable[
        [Literal["census", "classification", "replay"], RevisionCensusResult | PreparedRevisionReplayResult], None
    ]
    | None = None,
) -> bool:
    return _writer(
        adapter.archive_root,
        "synthetic-raw-publication",
        lambda: adapter.publish(frame, replacement, phase_receipt=phase_receipt),
    )


def _source_progress(root: Path, raw_ids: Sequence[str]) -> tuple[object, ...]:
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with (
        PreparedIndexMutation.source_only(archive_root=root) as seal,
        seal.original_read_snapshot(),
        seal.source_producer(),
    ):
        reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
        expanded, _member_keys = reader.expand_raw_membership_selection(raw_ids)
        current = tuple((item, reader.raw_parser_census_is_current(item)) for item in expanded)
        logical_keys = reader.raw_revision_rebuild_logical_keys(expanded)
        return (
            tuple((item, reader.raw_revision_descriptor(item)) for item in expanded),
            current,
            reader.raw_membership_census_rows(expanded),
            tuple((key, reader.raw_revision_replay_plan(key)) for key in logical_keys)
            if all(valid for _item, valid in current)
            else (),
        )


def _publish_to_valid(
    adapter: RawObservationDerivation, frame: RawFrame, replacement: RawObservationReplacement
) -> bool:
    """Retain the first real carrier while advancing its actual Source phases."""
    selected = replacement.raw_ids
    previous = _source_progress(adapter.archive_root, selected)
    first = replacement
    while True:
        preparatory = replacement.needs_source_census or replacement.needs_source_classification
        phases: list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]] = []

        def record_phase(
            phase: Literal["census", "classification", "replay"],
            receipt: RevisionCensusResult | PreparedRevisionReplayResult,
            *,
            captured_phases: list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]] = phases,
        ) -> None:
            captured_phases.append((phase, receipt))

        published = _publish(adapter, frame, replacement, phase_receipt=record_phase)
        if all(state == "valid" for state in adapter.inspect(frame, selected).values()):
            assert first.reference_seal is not None and first.reference_seal._closed
            return published
        if not preparatory or not phases:
            return False
        assert phases[0][0] in ("census", "classification")
        current = _source_progress(adapter.archive_root, selected)
        assert current != previous, "accepted Source phase retained identical original inputs"
        previous = current
        replacement = adapter.compute(frame, selected[0])


def _prepare_source_phases(adapter: RawObservationDerivation, frame: RawFrame, raw_id: str) -> None:
    """Reach the actual Index boundary without publishing its prepared rows."""
    previous = _source_progress(adapter.archive_root, (raw_id,))
    first: RawObservationReplacement | None = None
    while True:
        replacement = adapter.compute(frame, raw_id)
        if first is None:
            first = replacement
        if not (replacement.needs_source_census or replacement.needs_source_classification):
            replacement.close()
            assert first.reference_seal is not None and first.reference_seal._closed
            return
        phases: list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]] = []

        def record_phase(
            phase: Literal["census", "classification", "replay"],
            receipt: RevisionCensusResult | PreparedRevisionReplayResult,
            *,
            captured_phases: list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]] = phases,
        ) -> None:
            captured_phases.append((phase, receipt))

        _publish(adapter, frame, replacement, phase_receipt=record_phase)
        assert phases and phases[0][0] in ("census", "classification")
        current = _source_progress(adapter.archive_root, (raw_id,))
        assert current != previous, (replacement.needs_source_census, replacement.needs_source_classification)
        previous = current


def _admit(root: Path, names: tuple[str, ...], *, path: str = "bundle.json") -> str:
    with _fixture_archive(root) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=_chatgpt_payload(names),
            source_path=path,
            canonical_source_path=path,
            acquired_at_ms=1,
        )


def _run(root: Path, *, compute_adapter: BoundedComputeAdapter) -> DerivationReport:
    adapter = RawObservationDerivation(root, compute_adapter=compute_adapter)
    registry = DerivationRegistry((adapter,))
    frame = raw_observation_frame(root)
    with sqlite3.connect(root / "source.db") as source:
        selected = tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id"))
    previous = _source_progress(root, selected)

    def run_pass() -> DerivationReport:
        return converge(registry, frame, publisher=lambda actor, work: _writer(root, actor, work))

    first_report = run_pass()
    report = first_report
    while report.pending:
        current = _source_progress(root, selected)
        if current == previous and any(outcome.terminal_refusal is not None for outcome in report.outcomes):
            return report
        assert current != previous, (first_report.outcomes, report.outcomes)
        previous = current
        report = run_pass()
    return report


def _run_raw_law(root: Path, law: Callable[[BoundedComputeAdapter], None]) -> None:
    """Keep the complete law on the supplied original preparation creator."""
    from tests.infra.live_ingest import prepared_live_convergence_owner

    root.mkdir(parents=True, exist_ok=True)

    async def exercise() -> None:
        async with prepared_live_convergence_owner(root) as owner:
            await owner.run_convergence_sync("test.raw-observation.law", law, owner._compute_adapter)

    asyncio.run(exercise())


def _snapshot(root: Path) -> tuple[tuple[tuple[object, ...], ...], ...]:
    with sqlite3.connect(root / "source.db") as conn:
        return tuple(
            tuple(conn.execute(f"SELECT * FROM {table} ORDER BY raw_id"))
            for table in (
                "raw_sessions",
                "raw_session_memberships",
                "raw_membership_census",
                "raw_authority_parser_census",
            )
        )


def test_non_json_retained_worker_replays_past_old_payload_limit(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A 64 MiB cache ceiling must not block a retained non-JSON path.

        The small semantic payload has large JSON whitespace so the test reaches
        the former byte boundary without constructing a large parsed session.
        Losing the sealed worker path makes the old component-size check fail.
        """
        bootstrap_archive_root(tmp_path)
        payload = json.dumps(
            [
                {
                    "id": "large-text-route",
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
                                "content": {"content_type": "text", "parts": ["retained"]},
                            },
                        },
                    },
                }
            ]
        ).encode()
        payload += b" " * (64 * 1024 * 1024 + 1 - len(payload))
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="bundle.txt",
                canonical_source_path="bundle.txt",
                acquired_at_ms=1,
            )
        del payload

        adapter = make_raw_observation_derivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        assert replacement.prepared_inputs is not None
        assert replacement.payload is None
        assert _publish_to_valid(adapter, frame, replacement)
        assert replacement.scratch_directory is not None
        assert not replacement.scratch_directory.exists()
        assert adapter.inspect(raw_observation_frame(tmp_path), (raw_id,))[raw_id] == "valid"
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("large-text-route",)]
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)

    _run_raw_law(tmp_path, run_phase)


def test_retained_text_path_matches_json_session_output(tmp_path: Path) -> None:
    """The isolated non-JSON carrier preserves the canonical parser output."""
    outputs: list[tuple[list[tuple[str]], list[tuple[str]]]] = []
    for suffix in ("txt", "json"):
        root = tmp_path / suffix

        def run_phase(compute_adapter: BoundedComputeAdapter, *, root: Path = root, suffix: str = suffix) -> None:
            bootstrap_archive_root(root)
            raw_id = _admit(root, ("same-session",), path=f"bundle.{suffix}")
            adapter = make_raw_observation_derivation(root, compute_adapter=compute_adapter)
            frame = raw_observation_frame(root)
            replacement = adapter.compute(frame, raw_id)
            assert _publish_to_valid(adapter, frame, replacement)
            with sqlite3.connect(root / "index.db") as conn:
                outputs.append(
                    (
                        conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall(),
                        conn.execute("SELECT search_text FROM blocks ORDER BY block_id").fetchall(),
                    )
                )

        _run_raw_law(root, run_phase)
    assert outputs[0] == outputs[1]


def test_sqlite_page_image_uses_worker_without_materializing_a_session(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """SQLite-shaped retained evidence also stays outside the daemon parse cache."""
        bootstrap_archive_root(tmp_path)
        source = tmp_path / "opaque.db"
        with sqlite3.connect(source) as conn:
            conn.execute("CREATE TABLE unrelated (value TEXT)")
            conn.execute("INSERT INTO unrelated VALUES ('synthetic')")
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.ANTIGRAVITY,
                payload=source.read_bytes(),
                source_path=str(source),
                canonical_source_path=str(source),
                acquired_at_ms=1,
            )
        adapter = make_raw_observation_derivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        # The census commits in place during preparation and settles the
        # page image as non-session evidence: no replay input remains.
        assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
        assert replacement.prepared_inputs is None
        assert replacement.payload is None
        assert _publish(adapter, frame, replacement)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize("retained_name", ("state.db", "backup.json"))
def test_logical_sqlite_export_uses_worker_and_replays_session(tmp_path: Path, retained_name: str) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Logical export bytes outrank a JSON suffix when choosing the worker."""
        from polylogue.core.provider_identity import captured_hermes_profile_key
        from polylogue.sources.sqlite_export import logical_export_bytes
        from polylogue.sources.sqlite_snapshot import member_export_scope

        bootstrap_archive_root(tmp_path)
        source = tmp_path / "hermes-home" / "state.db"
        source.parent.mkdir()
        with sqlite3.connect(source) as conn:
            conn.executescript(
                """
                CREATE TABLE schema_version(version INTEGER NOT NULL);
                INSERT INTO schema_version VALUES (19);
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY, source TEXT, model_config TEXT, parent_session_id TEXT,
                    started_at REAL, ended_at REAL, end_reason TEXT, title TEXT
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
                    timestamp REAL NOT NULL, tool_calls TEXT, observed INTEGER DEFAULT 0,
                    active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0
                );
                INSERT INTO sessions VALUES ('root', 'cli', '{}', NULL, 1.0, 8.0, 'completed', 'root');
                INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, 'root', 'user', 'hi', 2.0);
                """
            )
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.HERMES,
                payload=logical_export_bytes(source, scope=member_export_scope(source)),
                source_path=str(source.with_name(retained_name)),
                canonical_source_path=str(source.with_name(retained_name)),
                # Hermes acquisition captures the profile identity its
                # session ids are qualified by.
                captured_profile_key=captured_hermes_profile_key(source.parent),
                acquired_at_ms=1,
            )
        adapter = make_raw_observation_derivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        assert replacement.prepared_inputs is not None
        assert replacement.payload is None
        assert _publish_to_valid(adapter, frame, replacement)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            native_ids = [str(row[0]) for row in conn.execute("SELECT native_id FROM sessions")]
            assert len(native_ids) == 1 and native_ids[0].startswith("root@profile-")
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)

    _run_raw_law(tmp_path, run_phase)


def test_split_member_loss_is_recovered_by_kernel_without_legacy_scanner(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Anti-vacuity: a raw-id/session-exists probe misses the lost split member."""
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("split-a", "split-b"))

        first = _run(tmp_path, compute_adapter=compute_adapter)
        assert first.done == 1 and first.failed == 0
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute("DELETE FROM sessions WHERE native_id = 'split-b'")
            conn.commit()
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "missing"
        restored = _run(tmp_path, compute_adapter=compute_adapter)
        assert restored.done == 1 and restored.failed == 0
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM sessions ORDER BY native_id").fetchall() == [
                ("split-a",),
                ("split-b",),
            ]
        before = _snapshot(tmp_path)
        unchanged = _run(tmp_path, compute_adapter=compute_adapter)
        assert unchanged.work.computed == unchanged.work.published == 0
        assert _snapshot(tmp_path) == before

    _run_raw_law(tmp_path, run_phase)


def test_one_pass_replays_a_shared_raw_component_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A sibling made current by the first replay must not replay the component again.

        Anti-vacuity: process the initially stale status of every raw ID without
        rechecking before preparation, and the second raw below invokes the real
        replay route a second time despite the first publication having already
        made its whole authoritative component current.
        """
        from polylogue.sources import revision_backfill

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=_chatgpt_payload(("shared-session",)),
                    source_path="shared-component.json",
                    canonical_source_path="shared-component.json",
                    source_index=index,
                    acquired_at_ms=1,
                )
                for index in range(2)
            )
            component, _logical_keys = archive.expand_raw_membership_selection([raw_ids[0]])
        assert set(component) == set(raw_ids)

        _prepare_source_phases(
            RawObservationDerivation(tmp_path, compute_adapter=compute_adapter),
            raw_observation_frame(tmp_path),
            raw_ids[0],
        )

        replay = Mock(wraps=revision_backfill.apply_prepared_revision_replay)
        monkeypatch.setattr(revision_backfill, "apply_prepared_revision_replay", replay)

        report = converge(
            DerivationRegistry((RawObservationDerivation(tmp_path, compute_adapter=compute_adapter),)),
            raw_observation_frame(tmp_path),
            budget=Budget(page=2, discovery=2, inspection=4, compute=2, publication=2),
            publisher=lambda actor, work: _writer(tmp_path, actor, work),
        )

        assert report.done == 2 and report.failed == report.pending == 0
        assert replay.call_count == 1
        from polylogue.storage import raw_authority

        full_plan = Mock(wraps=raw_authority._raw_replay_plan_from_rows)
        monkeypatch.setattr(raw_authority, "_raw_replay_plan_from_rows", full_plan)
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        assert adapter.inspect(raw_observation_frame(tmp_path), raw_ids) == dict.fromkeys(raw_ids, "valid")
        assert full_plan.call_count == 0

    _run_raw_law(tmp_path, run_phase)


def test_component_publications_leave_the_archive_fts_audit_to_one_pass_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """N replayed components cost one archive-wide exact FTS audit, not N.

        Each component publication proves its own sessions' FTS rows; the
        archive-wide exact inspection belongs to the daemon's
        ``fts_readiness_binding`` stage, which runs once after the burst of block
        writes retired its binding.

        Wrong outcome prevented: every component's replay ran the archive-wide
        ``fts_invariant_snapshot_sync`` scan under the writer, so a pass over N
        components scanned every block N times. Anti-vacuity: drop
        ``exact_fts_audit=False`` from ``RawObservationDerivation.publish`` and
        ``audits`` equals the component count.
        """
        from polylogue.daemon.convergence_stages import make_fts_readiness_binding_stage
        from polylogue.storage.fts import fts_lifecycle
        from polylogue.storage.fts.derivation import GLOBAL_PARTITION, FtsDerivationAdapter

        bootstrap_archive_root(tmp_path)
        for index in range(3):
            _admit(tmp_path, (f"component-{index}",), path=f"component-{index}.json")
        audits = 0
        global_inspections = 0
        original_audit = fts_lifecycle.fts_invariant_snapshot_sync
        original_inspect = FtsDerivationAdapter.inspect_partition

        def counting_audit(conn: sqlite3.Connection) -> object:
            nonlocal audits
            audits += 1
            return original_audit(conn)

        def counting_inspect(self: FtsDerivationAdapter, conn: sqlite3.Connection, partition: str) -> object:
            nonlocal global_inspections
            global_inspections += int(partition == GLOBAL_PARTITION)
            return original_inspect(self, conn, partition)

        monkeypatch.setattr(fts_lifecycle, "fts_invariant_snapshot_sync", counting_audit)
        monkeypatch.setattr(FtsDerivationAdapter, "inspect_partition", counting_inspect)

        report = _run(tmp_path, compute_adapter=compute_adapter)

        assert report.done == 3 and report.failed == report.pending == 0
        assert (audits, global_inspections) == (0, 0)
        stage = make_fts_readiness_binding_stage(tmp_path / "index.db")
        assert stage.check(tmp_path / "index.db") is True
        assert stage.execute(tmp_path / "index.db") is True
        assert (audits, global_inspections) == (0, 1)
        assert stage.check(tmp_path / "index.db") is False

    _run_raw_law(tmp_path, run_phase)


def test_duplicate_raws_share_preparation_but_keep_distinct_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Identical bytes need one worker parse while both raw IDs reach authority."""

        worker_calls: list[str] = []
        from polylogue.sources import revision_backfill

        original_prepare = revision_backfill.prepare_retained_jsonl_artifact

        from polylogue.schemas.runtime_registry import SchemaRegistry
        from polylogue.sources.prepared_jsonl import PreparedJsonl
        from polylogue.sources.revision_backfill import RetainedSessionRead

        def counted_prepare(
            reader: RetainedSessionRead,
            raw_id: str,
            *,
            directory: Path,
            validation_mode: ValidationMode,
            schema_registry: SchemaRegistry,
        ) -> PreparedJsonl:
            worker_calls.append(raw_id)
            return original_prepare(
                reader,
                raw_id,
                directory=directory,
                validation_mode=validation_mode,
                schema_registry=schema_registry,
            )

        monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", counted_prepare)

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=b"[]",
                    source_path="duplicate-component.json",
                    canonical_source_path="duplicate-component.json",
                    source_index=index,
                    acquired_at_ms=1,
                )
                for index in range(2)
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_ids[0])
        # The census commits in place during preparation: the shared worker
        # parse settles both raws, each with its own census row.
        assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
        assert len(worker_calls) == 1 and worker_calls[0] in raw_ids
        assert _publish(adapter, frame, replacement)
        assert adapter.inspect(frame, raw_ids) == dict.fromkeys(raw_ids, "valid")
        with sqlite3.connect(tmp_path / "source.db") as conn:
            censused = {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT raw_id, status FROM raw_authority_parser_census")
            }
        assert censused == dict.fromkeys(raw_ids, "complete")

    _run_raw_law(tmp_path, run_phase)


def test_unequal_retained_bytes_do_not_share_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Equal decoded values do not permit sharing unequal source bytes."""

        worker_calls: list[str] = []
        from polylogue.sources import revision_backfill

        original_prepare = revision_backfill.prepare_retained_jsonl_artifact

        from polylogue.schemas.runtime_registry import SchemaRegistry
        from polylogue.sources.prepared_jsonl import PreparedJsonl
        from polylogue.sources.revision_backfill import RetainedSessionRead

        def counted_prepare(
            reader: RetainedSessionRead,
            raw_id: str,
            *,
            directory: Path,
            validation_mode: ValidationMode,
            schema_registry: SchemaRegistry,
        ) -> PreparedJsonl:
            worker_calls.append(raw_id)
            return original_prepare(
                reader,
                raw_id,
                directory=directory,
                validation_mode=validation_mode,
                schema_registry=schema_registry,
            )

        monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", counted_prepare)

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=b"[]" if index == 0 else b"[ ]",
                    source_path="duplicate-component.json",
                    canonical_source_path="duplicate-component.json",
                    source_index=index,
                    acquired_at_ms=1,
                )
                for index in range(2)
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_ids[0])
        # The census commits in place during preparation: the shared worker
        # parse settles both raws, each with its own census row.
        assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
        assert len(worker_calls) == 2 and set(worker_calls) == set(raw_ids)
        assert _publish(adapter, frame, replacement)
        assert adapter.inspect(frame, raw_ids) == dict.fromkeys(raw_ids, "valid")
        with sqlite3.connect(tmp_path / "source.db") as conn:
            censused = {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT raw_id, status FROM raw_authority_parser_census")
            }
        assert censused == dict.fromkeys(raw_ids, "complete")

    _run_raw_law(tmp_path, run_phase)


def test_shared_session_carrier_keeps_each_raw_validation_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Identical bytes need one worker parse while both raw IDs reach authority."""

        worker_calls: list[str] = []
        from polylogue.sources import revision_backfill

        original_prepare = revision_backfill.prepare_retained_jsonl_artifact

        from polylogue.schemas.runtime_registry import SchemaRegistry
        from polylogue.sources.prepared_jsonl import PreparedJsonl
        from polylogue.sources.revision_backfill import RetainedSessionRead

        def counted_prepare(
            reader: RetainedSessionRead,
            raw_id: str,
            *,
            directory: Path,
            validation_mode: ValidationMode,
            schema_registry: SchemaRegistry,
        ) -> PreparedJsonl:
            worker_calls.append(raw_id)
            return original_prepare(
                reader,
                raw_id,
                directory=directory,
                validation_mode=validation_mode,
                schema_registry=schema_registry,
            )

        monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", counted_prepare)

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=_chatgpt_payload(("duplicate-session",)),
                    source_path="duplicate-component.json",
                    canonical_source_path="duplicate-component.json",
                    source_index=index,
                    acquired_at_ms=1,
                )
                for index in range(2)
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_ids[0])
        # The census commits in place during preparation: the shared worker
        # parse settles both raws, each with its own census row.
        assert replacement.committed_phase_receipts[0][0] == "census"
        assert len(worker_calls) == 1 and worker_calls[0] in raw_ids
        assert replacement.prepared_inputs is not None
        inputs = replacement.prepared_inputs
        assert set(inputs) == set(raw_ids)
        assert len({id(item.prepared_artifact) for item in inputs.values()}) == 1
        for raw_id, item in inputs.items():
            assert item.validation_verdict is not None
            assert item.validation_verdict.raw_id == raw_id
            assert item.validation_verdict.evidence_id == raw_id
        assert _publish(adapter, frame, replacement)
        assert adapter.inspect(frame, raw_ids) == dict.fromkeys(raw_ids, "valid")
        with sqlite3.connect(tmp_path / "source.db") as conn:
            censused = {
                str(row[0]): str(row[1])
                for row in conn.execute("SELECT raw_id, status FROM raw_authority_parser_census")
            }
        assert censused == dict.fromkeys(raw_ids, "complete")

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize("cancel_at", ["duplicate_validation", "worker_return"])
def test_cancelled_duplicate_validation_discards_the_shared_carrier_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_at: str
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        from polylogue import schemas
        from polylogue.core.compute import DaemonOperationCancelled
        from polylogue.core.compute_cancel import check_compute_cancelled
        from polylogue.sources import revision_backfill
        from polylogue.sources.prepared_jsonl import PreparedJsonl
        from polylogue.storage.derived import raw as raw_derivation

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_ids = tuple(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=_chatgpt_payload(("cancel-shared-session",)),
                    source_path="cancel-shared.json",
                    canonical_source_path="cancel-shared.json",
                    source_index=index,
                    acquired_at_ms=1,
                )
                for index in range(2)
            )
        prepared: list[PreparedJsonl] = []
        discarded: list[int] = []
        validations: list[str] = []
        original_prepare = revision_backfill.prepare_retained_jsonl_artifact
        original_validate = schemas.validate_retained_document
        original_discard = PreparedJsonl.discard
        original_check = check_compute_cancelled

        def check() -> None:
            if cancel_at == "worker_return" and prepared:
                raise DaemonOperationCancelled("cancel shared validation")
            original_check()

        def discard(artifact: PreparedJsonl) -> None:
            discarded.append(id(artifact))
            original_discard(artifact)

        def validate(*args: Any, **kwargs: Any) -> Any:
            validations.append(kwargs["raw_id"])
            if cancel_at == "duplicate_validation" and len(validations) == 2:
                raise DaemonOperationCancelled("cancel shared validation")
            return original_validate(*args, **kwargs)

        def prepare(*args: Any, **kwargs: Any) -> PreparedJsonl:
            artifact = original_prepare(*args, **kwargs)
            prepared.append(artifact)
            return artifact

        with monkeypatch.context() as patch:
            patch.setattr(PreparedJsonl, "discard", discard)
            patch.setattr(raw_derivation, "check_compute_cancelled", check)
            patch.setattr(schemas, "validate_retained_document", validate)
            patch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", prepare)
            adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
            with pytest.raises(DaemonOperationCancelled, match="cancel shared validation"):
                adapter.compute(raw_observation_frame(tmp_path), raw_ids[0])
        assert len(prepared) == 1
        assert len(validations) == (2 if cancel_at == "duplicate_validation" else 1)
        assert set(validations) <= set(raw_ids)
        assert discarded.count(id(prepared[0])) == 1
        assert prepared[0].sessions_path is not None and not prepared[0].sessions_path.exists()
        assert prepared[0].shard_path is not None and not prepared[0].shard_path.exists()
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert (
                conn.execute("SELECT count(*) FROM raw_authority_parser_census WHERE status='complete'").fetchone()[0]
                == 0
            )

    _run_raw_law(tmp_path, run_phase)


def test_current_parser_census_still_records_missing_validation_policy(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A current parser receipt cannot stand in for a raw's validation receipt."""
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("validation-continuation",))
        initial = _run(tmp_path, compute_adapter=compute_adapter)
        assert initial.done == 1 and initial.failed == initial.pending == 0
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute(
                "UPDATE raw_sessions SET validated_at_ms=NULL,validation_status=NULL,validation_error=NULL,"
                "validation_mode=NULL,validation_drift_count=0 WHERE raw_id=?",
                (raw_id,),
            )
            conn.commit()

        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        try:
            assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
            assert _publish(adapter, frame, replacement)
        finally:
            replacement.close()
        with sqlite3.connect(tmp_path / "source.db") as conn:
            status, mode, validated_at_ms = conn.execute(
                "SELECT validation_status,validation_mode,validated_at_ms FROM raw_sessions WHERE raw_id=?",
                (raw_id,),
            ).fetchone()
        assert status == "passed" and mode == ValidationMode.ADVISORY.value and validated_at_ms is not None
        assert adapter.inspect(frame, (raw_id,)) == {raw_id: "valid"}

    _run_raw_law(tmp_path, run_phase)


def test_restart_without_ops_hints_recovers_index_loss_and_new_admission(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Anti-vacuity: retained pending caches hide reset or later admissions."""
        bootstrap_archive_root(tmp_path)
        _admit(tmp_path, ("first",))
        assert _run(tmp_path, compute_adapter=compute_adapter).done == 1
        with sqlite3.connect(tmp_path / "ops.db") as conn:
            conn.execute("DELETE FROM convergence_debt")
            conn.commit()
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute("DELETE FROM sessions")
            conn.commit()
        _admit(tmp_path, ("second",), path="second.json")
        restarted = _run(tmp_path, compute_adapter=compute_adapter)
        assert restarted.done == 2 and restarted.failed == 0
        assert _run(tmp_path, compute_adapter=compute_adapter).made_no_publication_attempts

    _run_raw_law(tmp_path, run_phase)


def test_absent_retained_blob_is_restored_from_its_exact_direct_source(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A lost retained blob converges once its direct source proves the bytes.

        While the source holds different bytes the raw fails with a typed,
        deterministic refusal and nothing is staged or published. Once the source
        holds the recorded bytes again, the writer restores the blob, consumes its
        publication receipt, and the next pass materializes the session.
        Anti-vacuity: without restoration every pass fails with "retained raw blob
        disappeared" and the session never materializes.
        """
        from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
        from polylogue.storage.blob_store import BlobStore

        bootstrap_archive_root(tmp_path)
        payload = _chatgpt_payload(("restored",))
        source = tmp_path / "exports" / "bundle.json"
        source.parent.mkdir()
        source.write_bytes(payload)
        raw_id = _admit(tmp_path, ("restored",), path=str(source))
        store = BlobStore(tmp_path / "blob")
        blob_path = store.blob_path(hashlib.sha256(payload).hexdigest())
        blob_path.unlink()

        source.write_bytes(payload.replace(b"restored", b"rewritten"))
        with pytest.raises(
            RetainedPreparationRetryableError, match=r"not restorable from its source \(hash_mismatch\)"
        ):
            RawObservationDerivation(tmp_path, compute_adapter=compute_adapter).compute(
                raw_observation_frame(tmp_path), raw_id
            )
        assert not blob_path.exists()
        assert not any(store.staging_root.iterdir())

        source.write_bytes(payload)
        restoring = _run(tmp_path, compute_adapter=compute_adapter)
        assert restoring.failed == 0
        assert blob_path.read_bytes() == payload
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
        assert restoring.done + _run(tmp_path, compute_adapter=compute_adapter).done == 1
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)
        assert not any(store.staging_root.iterdir())

    _run_raw_law(tmp_path, run_phase)


def test_missing_prepared_raw_retries_without_quarantining_source(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A lost worker carrier must not become a parser verdict about retained bytes."""
        from polylogue.sources.revision_backfill import (
            RetainedPreparationRetryableError,
            prepare_revision_source_census,
        )

        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("prepared-retry",))
        from polylogue.storage.blob_store import BlobStore
        from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

        seal = PreparedIndexMutation.source_only(archive_root=tmp_path)
        try:
            with pytest.raises(RetainedPreparationRetryableError, match="missing"):
                with seal.original_read_snapshot(), seal.source_producer():
                    reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(tmp_path / "blob"))
                    prepare_revision_source_census(seal, reader, selected_raw_ids=[raw_id], prepared_inputs={})
        finally:
            seal.close()
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
                None,
            )
            assert conn.execute(
                "SELECT COUNT(*) FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
            ).fetchone() == (0,)

    _run_raw_law(tmp_path, run_phase)


def test_empty_eligible_chatgpt_document_still_requires_validation_evidence(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A current non-session census cannot stand in for eligible JSON validation."""
        from polylogue.sources.revision_backfill import (
            RetainedPreparationRetryableError,
            prepare_revision_source_census,
        )

        bootstrap_archive_root(tmp_path)
        payload = json.dumps(
            {
                "id": "empty-eligible",
                "conversation_id": "empty-eligible",
                "title": "empty eligible document",
                "create_time": 1_700_000_000,
                "current_node": "node-1",
                "mapping": {
                    "node-1": {
                        "id": "node-1",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "message-1",
                            "author": {"role": "user"},
                            "create_time": 1_700_000_000,
                            "content": {"content_type": "text", "parts": []},
                        },
                    }
                },
            }
        ).encode()
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="empty-eligible.json",
                canonical_source_path="empty-eligible.json",
                acquired_at_ms=1,
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        initial = adapter.compute(frame, raw_id)
        assert _publish_to_valid(adapter, frame, initial)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                "SELECT validation_status, validation_mode FROM raw_sessions WHERE raw_id=?", (raw_id,)
            ).fetchone() == ("passed", "advisory")
            assert conn.execute("SELECT COUNT(*) FROM raw_artifacts WHERE raw_id=?", (raw_id,)).fetchone() == (0,)
            assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone() == (
                "non_session",
            )
            conn.execute(
                "UPDATE raw_sessions SET validated_at_ms=NULL, validation_status=NULL, "
                "validation_error=NULL, validation_drift_count=0, validation_mode=NULL WHERE raw_id=?",
                (raw_id,),
            )
            conn.commit()
        from polylogue.storage.blob_store import BlobStore
        from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
            with seal.original_read_snapshot(), seal.source_producer():
                reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(tmp_path / "blob"))
                assert reader.raw_schema_eligible(raw_id)
                with pytest.raises(RetainedPreparationRetryableError, match="lacks captured validation evidence"):
                    prepare_revision_source_census(seal, reader, selected_raw_ids=[raw_id], prepared_inputs={})

    _run_raw_law(tmp_path, run_phase)


def test_retained_jsonl_replay_consumes_worker_carrier_without_inline_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Losing the strict carrier path would reparse this raw inside replay."""
        from polylogue.sources import revision_backfill

        bootstrap_archive_root(tmp_path)
        payload = (
            b'{"type":"session_meta","payload":{"id":"prepared-session"}}\n'
            b'{"type":"response_item","payload":{"type":"message","role":"user",'
            b'"content":[{"type":"input_text","text":"prepared text"}]}}\n'
        )
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="prepared-session.jsonl",
                canonical_source_path="prepared-session.jsonl",
                acquired_at_ms=1,
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        assert replacement.prepared_inputs is not None
        monkeypatch.setattr(
            revision_backfill,
            "parse_retained_raw_sessions",
            lambda *_args: pytest.fail("retained replay parsed inline"),
        )
        assert _publish_to_valid(adapter, frame, replacement)
        assert replacement.scratch_directory is not None and not replacement.scratch_directory.exists()
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM sessions").fetchone() == ("prepared-session",)

    _run_raw_law(tmp_path, run_phase)


def test_retained_json_document_uses_prepared_carrier_past_cache_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A whole-document raw larger than the cache budget still reaches replay."""
        from polylogue.sources import revision_backfill

        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("large-document",))
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        assert replacement.prepared_inputs is not None
        monkeypatch.setattr(
            revision_backfill,
            "parse_retained_raw_sessions",
            lambda *_args: pytest.fail("retained JSON replay parsed inline"),
        )
        assert _publish_to_valid(adapter, frame, replacement)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("large-document",)]

    _run_raw_law(tmp_path, run_phase)


def test_retained_fact_json_is_terminal_without_a_session(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """The prepared document route keeps declared sidecar evidence out of the index."""
        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=b'{"agentId":"synthetic-agent","toolUseId":"synthetic-tool"}',
                source_path="subagents/agent-synthetic.meta.json",
                canonical_source_path="subagents/agent-synthetic.meta.json",
                acquired_at_ms=1,
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        # The census commits in place during preparation and settles the
        # declared fact document: nothing remains for a session replay.
        assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
        assert replacement.prepared_inputs is None
        assert not (replacement.needs_source_census or replacement.needs_source_classification)
        assert _publish(adapter, frame, replacement)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    _run_raw_law(tmp_path, run_phase)


def test_unknown_retained_json_resolves_on_prepared_route_past_cache_budget(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Source-only acquisition keeps its raw identity while replay detects JSON shape."""
        bootstrap_archive_root(tmp_path)
        payload = {
            "id": "detected-json",
            "title": "detected-json",
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
                        "content": {"content_type": "text", "parts": ["detected content"]},
                    },
                }
            },
        }
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.UNKNOWN,
                payload=json.dumps([payload]).encode(),
                source_path="unknown-capture.json",
                canonical_source_path="unknown-capture.json",
                acquired_at_ms=1,
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id)
        assert replacement.prepared_inputs is not None
        # The census commits in place during preparation: the raw keeps its
        # raw identity, records the provider its bytes resolved to, and the
        # parsed origin replaces acquisition's unknown-export placeholder.
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                "SELECT origin, detected_provider FROM raw_sessions WHERE raw_id = ?", (raw_id,)
            ).fetchone() == ("chatgpt-export", "chatgpt")
        artifact = replacement.prepared_inputs[raw_id].prepared_artifact
        assert artifact is not None
        assert artifact.resolved_provider is Provider.CHATGPT
        assert _publish_to_valid(adapter, frame, replacement)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("detected-json",)]

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize(
    ("payload", "source_name"),
    [
        (b'{"unrecognized":"synthetic"}', "unknown-shape.json"),
        ((b'{"opaque":"' + b"x" * 9_000 + b'"}\n') * 32, "opaque.jsonl"),
    ],
    ids=["object", "complete-jsonl"],
)
def test_unsupported_unknown_json_keeps_typed_refusal_in_prepared_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes, source_name: str
) -> None:
    """Original unsupported bytes keep their typed refusal through real publication.

    Anti-vacuity: drop the typed census refusal and the raw replays as a
    session; report no committed receipts and ``phases`` is empty.
    """
    import asyncio
    import sys
    from builtins import BaseExceptionGroup

    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.sources.revision_backfill import (
        RevisionCensusResult,
    )
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.retained_jsonl import retained_raw_fixture

    bootstrap_archive_root(tmp_path)

    blob_hash, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=blob_hash,
        source_path=str(tmp_path / source_name),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == blob_hash

    def refuse_eager_material(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("retained unsupported input eagerly loaded complete Raw bytes")

    monkeypatch.setattr(PreparedSessionSourceRead, "raw_revision_material", refuse_eager_material)

    async def exercise() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            retained: list[RawObservationReplacement] = []

            def run_phase() -> None:
                adapter = RawObservationDerivation(tmp_path, compute_adapter=owner._compute_adapter)
                frame = raw_observation_frame(tmp_path, raw_ids=(raw_id,))
                replacement = adapter.compute(frame, raw_id)
                retained.append(replacement)
                try:
                    # Single preparation: the census commits in place during
                    # compute and settles the deterministic unsupported-shape
                    # refusal, so no replay input remains to prepare.
                    assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
                    assert replacement.prepared_inputs is None
                    assert not (replacement.needs_source_census or replacement.needs_source_classification)
                    phases: list[tuple[str, RevisionCensusResult | PreparedRevisionReplayResult]] = []

                    def record_phase(phase: str, receipt: RevisionCensusResult | PreparedRevisionReplayResult) -> None:
                        phases.append((phase, receipt))

                    # Publication reports the committed census receipt and
                    # replays nothing.
                    admit_stage_write(
                        "test.unsupported.raw-publication",
                        lambda: adapter.publish(frame, replacement, phase_receipt=record_phase),
                    )
                    assert len(phases) == 1
                    phase, receipt = phases[0]
                    assert phase == "census"
                    assert isinstance(receipt, RevisionCensusResult)
                    assert receipt.scanned == 1
                    assert receipt.quarantined == 1

                finally:
                    primary = sys.exception()
                    try:
                        replacement.close()
                    except BaseException as cleanup:
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "unsupported preparation and close failed", [primary, cleanup]
                            ) from None
                        raise
                    else:
                        retained.remove(replacement)

            await owner.run_prepared_sync(
                "test.unsupported.raw-preparation",
                run_phase,
                settlement_owners=lambda: tuple(retained),
                estimated_bytes=0,
            )

    asyncio.run(exercise())
    # The refusal settles: a non-session census with typed terminal evidence,
    # never a failed census the next pass would census again.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone() == (
            "non_session",
        )
        assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchall() == [
            ("terminal_unknown_export_no_session",)
        ]
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.parametrize("source_path", ("worker-exit.jsonl", "worker-exit.txt"))
def test_retained_compute_refusal_keeps_raw_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_path: str
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Shared admission refusal leaves durable source evidence retryable."""
        from polylogue.core.compute import DaemonBackpressureError

        bootstrap_archive_root(tmp_path)
        payload = (
            b'{"type":"session_meta","payload":{"id":"worker-exit"}}\n'
            b'{"type":"response_item","payload":{"type":"message","role":"user",'
            b'"content":[{"type":"input_text","text":"retry"}]}}\n'
        )
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=1,
            )

        def refuse_compute(*args: object, **kwargs: object) -> object:
            raise DaemonBackpressureError("synthetic saturated capacity")

        monkeypatch.setattr(compute_adapter, "require_current_creator", refuse_compute)
        with pytest.raises(DaemonBackpressureError):
            make_raw_observation_derivation(tmp_path, compute_adapter=compute_adapter).compute(
                raw_observation_frame(tmp_path), raw_id
            )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
                None,
            )

    _run_raw_law(tmp_path, run_phase)


def test_retained_blob_io_failure_retries_without_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A transient blob read error must not become a durable parser refusal."""
        from polylogue.sources import revision_backfill
        from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=_chatgpt_payload(("io-retry",)),
                source_path="io-retry.json",
                canonical_source_path="io-retry.json",
                acquired_at_ms=1,
            )

        attempted: list[bool] = []

        def fail_blob_open(*_args: object, **_kwargs: object) -> None:
            attempted.append(True)
            raise OSError(errno.EMFILE, "too many open files")

        monkeypatch.setattr(revision_backfill, "prepare_jsonl_blob", fail_blob_open)
        with pytest.raises(RetainedPreparationRetryableError, match="read failed"):
            RawObservationDerivation(tmp_path, compute_adapter=compute_adapter).compute(
                raw_observation_frame(tmp_path), raw_id
            )
        assert attempted == [True]
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (
                None,
            )
            assert conn.execute(
                "SELECT COUNT(*) FROM raw_membership_census WHERE raw_id = ?", (raw_id,)
            ).fetchone() == (0,)

    _run_raw_law(tmp_path, run_phase)


def test_retained_parser_error_settles_as_terminal_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A parser verdict from a live worker settles through the canonical source census."""
        from polylogue.sources.prepared_jsonl import PreparedJsonl

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b"{bad json}\n",
                source_path="bad-session.jsonl",
                canonical_source_path="bad-session.jsonl",
                acquired_at_ms=1,
            )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        from polylogue.sources import prepared_jsonl

        prepared_calls: list[str] = []

        def refuse_prepared(*args: Any, **kwargs: Any) -> PreparedJsonl:
            prepared_calls.append(str(args[1]))
            return PreparedJsonl(None, None, None, "synthetic parser refusal")

        monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", refuse_prepared)
        replacement = adapter.compute(frame, raw_id)
        assert prepared_calls == ["bad-session.jsonl"]
        # The census commits in place during preparation and settles the
        # refusal; no replay publication follows it.
        assert [phase for phase, _receipt in replacement.committed_phase_receipts] == ["census"]
        assert replacement.prepared_inputs is None
        assert _publish(adapter, frame, replacement)
        assert adapter.inspect(frame, (raw_id,)) == {raw_id: "valid"}
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone() == (
                "non_session",
            )
            assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id = ?", (raw_id,)).fetchall() == [
                ("terminal_unsupported_shape",)
            ]

    _run_raw_law(tmp_path, run_phase)


def test_malformed_retained_jsonl_commits_typed_refusal_and_preserves_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real malformed bytes settle at the original Source phase without replay."""
    from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind, RetainedRawDecodeRefusalError
    from polylogue.sources import prepared_jsonl
    from polylogue.storage.blob_store import BlobStore

    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        bootstrap_archive_root(tmp_path)
        payload = b"{bad json}\n"
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="malformed-session.jsonl",
                canonical_source_path="malformed-session.jsonl",
                acquired_at_ms=1,
            )
        calls: list[str] = []
        original = prepared_jsonl.prepare_jsonl_blob

        def observe(*args: Any, **kwargs: Any) -> Any:
            calls.append(str(args[1]))
            return original(*args, **kwargs)

        monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", observe)
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        with pytest.raises(RetainedRawDecodeRefusalError) as failed:
            adapter.compute(frame, raw_id)
        assert failed.value.raw_id == raw_id
        assert failed.value.kind is RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT
        assert calls == ["malformed-session.jsonl"]
        with sqlite3.connect(tmp_path / "source.db") as conn:
            # Corrupt bytes cannot prove membership or a non-session verdict.
            assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchall() == []
            assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id=?", (raw_id,)).fetchall() == [
                ("terminal_corrupt_input",)
            ]
            (blob_hash,) = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
        assert BlobStore(tmp_path / "blob").read_all(bytes(blob_hash).hex()) == payload
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        settled = _snapshot(tmp_path)
        with pytest.raises(RetainedRawDecodeRefusalError) as repeated:
            adapter.compute(frame, raw_id)
        assert repeated.value.kind is failed.value.kind
        assert calls == ["malformed-session.jsonl"]
        assert _snapshot(tmp_path) == settled

    _run_raw_law(tmp_path, run_phase)


def test_empty_claude_history_remains_non_session_when_validation_mode_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Configured history is raw-only evidence and has no schema policy."""
        from types import SimpleNamespace

        import polylogue.paths as polylogue_paths
        import polylogue.sources.live.watcher as live_watcher
        from polylogue.sources.live.batch import LiveBatchProcessor
        from polylogue.sources.live.cursor import CursorStore
        from polylogue.sources.live.watcher import default_sources
        from tests.infra.raw_owner_routes import run_ingest_files

        bootstrap_archive_root(tmp_path)
        claude_root = tmp_path / "neutral-home" / ".claude"
        claude_root.mkdir(parents=True)
        source_path = claude_root / "history.jsonl"
        source_path.write_bytes(b"")
        monkeypatch.setattr(polylogue_paths, "claude_code_path", lambda: claude_root / "projects")
        history_source = next(source for source in default_sources() if source.name == "claude-code-history")
        polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))
        processor = LiveBatchProcessor(
            polylogue,
            (history_source,),
            cursor=CursorStore(tmp_path / "ops.db"),
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        )
        metrics = run_ingest_files(processor, [source_path], emit_event=False)
        assert metrics.excluded_file_count == 1 and metrics.failed_file_count == 0, metrics

        frame = raw_observation_frame(tmp_path)
        advisory = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            raw_id = str(conn.execute("SELECT raw_id FROM raw_sessions").fetchone()[0])
            assert conn.execute(
                "SELECT artifact_kind, parse_as_session, schema_eligible FROM raw_artifacts WHERE raw_id=?",
                (raw_id,),
            ).fetchall() == [("prompt_history_log", 0, 0)]
            assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone() == (
                "non_session",
            )
            # Non-session sources deliberately have no schema-validation stamp.
        assert advisory.inspect(frame, (raw_id,)) == {raw_id: "valid"}

        strict = RawObservationDerivation(
            tmp_path, compute_adapter=compute_adapter, validation_mode=ValidationMode.STRICT
        )
        strict_frame = raw_observation_frame(tmp_path, validation_mode=ValidationMode.STRICT)
        assert strict.inspect(strict_frame, (raw_id,)) == {raw_id: "valid"}
        repeated = strict.compute(strict_frame, raw_id)
        try:
            assert repeated.already_valid
        finally:
            repeated.close()
        assert advisory.inspect(frame, (raw_id,)) == {raw_id: "valid"}

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize("mutation", ["descriptor", "blob", "generation"])
def test_publish_rejects_changed_source_or_generation(tmp_path: Path, mutation: str) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Anti-vacuity: bypassing publish revalidation materializes stale inputs."""
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("bound",))
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        prepared = adapter.compute(frame, raw_id)
        if mutation == "descriptor":
            with sqlite3.connect(tmp_path / "source.db") as conn:
                conn.execute("UPDATE raw_sessions SET source_path = 'changed.json' WHERE raw_id = ?", (raw_id,))
                conn.commit()
        elif mutation == "blob":
            with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
                _provider, blob_hash, _path, _kind, _size = archive.raw_revision_descriptor(raw_id)
                blob_path = archive.blob_path_for_hash(blob_hash)
                assert blob_path is not None
                blob_path.write_bytes(b"changed")
        else:
            from dataclasses import replace

            frame = replace(frame, source_revision=str(tmp_path / "another-index.db"))
        if mutation == "descriptor":
            # The census commits in place during preparation, so a changed Source
            # row now reaches the replay publication's original observer seal,
            # which refuses before any effect with its typed stale error.
            from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

            with pytest.raises(ReferenceSealStaleError):
                _publish(adapter, frame, prepared)
        else:
            assert _publish(adapter, frame, prepared) is False
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    _run_raw_law(tmp_path, run_phase)


def test_poison_observation_does_not_suppress_healthy_sibling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        bootstrap_archive_root(tmp_path)
        _admit(tmp_path, ("healthy",), path="healthy.json")
        with _fixture_archive(tmp_path) as archive:
            poison = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=b"not json",
                source_path="poison.json",
                canonical_source_path="poison.json",
                acquired_at_ms=1,
            )
        report = _run(tmp_path, compute_adapter=compute_adapter)
        assert report.done == 1 and report.failed == 1
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("healthy",)]

        def refuse_reparse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("settled terminal evidence must not reparse retained bytes")

        monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_retained_jsonl_artifact", refuse_reparse)
        repeated = _run(tmp_path, compute_adapter=compute_adapter)
        assert repeated.done == 0 and repeated.failed == 1
        failure = next(outcome for outcome in repeated.outcomes if outcome.key.key == poison)
        assert failure.transient is False
        assert failure.terminal_refusal is not None

        from polylogue.daemon.cli import _derivation_admission
        from polylogue.daemon.intake import AdmissionOutcome
        from polylogue.operations.intake_adapters import RawMaterializationDiscovery

        assert _derivation_admission(repeated, poison, subject="raw observation").outcome is AdmissionOutcome.EXCLUDED
        discovery = RawMaterializationDiscovery(tmp_path)
        assert all(not discovery.discover_pending_raw_ids(8) for _ in range(4))
        from polylogue.operations.raw_observation_derivation import raw_observation_backlog_snapshot

        assert raw_observation_backlog_snapshot(tmp_path, limit=8)["candidate_count"] == 0
        with sqlite3.connect(tmp_path / "source.db") as source:
            # A census-settled decode carrier records no validation failure, so
            # its typed kind and support status, not worker provenance, carry
            # its authority; an inconsistent carrier is no refusal.
            source.execute("UPDATE raw_artifacts SET support_status='unknown' WHERE raw_id=?", (poison,))
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        assert adapter.terminal_decode_refusals((poison,)) == {}
        assert adapter.inspect(raw_observation_frame(tmp_path), (poison,))[poison] == "stale"
        renewed = RawMaterializationDiscovery(tmp_path)
        assert poison in {raw_id for _ in range(4) for raw_id, _cost in renewed.discover_pending_raw_ids(8)}

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize("lane", ["arrival", "dependents", "sweep"])
def test_every_raw_discovery_lane_preserves_terminal_receipt_authority(tmp_path: Path, lane: str) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        from polylogue.operations.intake_adapters import RawMaterializationDiscovery

        bootstrap_archive_root(tmp_path)
        with _fixture_archive(tmp_path) as archive:
            poison = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=b"not json\n",
                source_path="/synthetic/project/poison.jsonl",
                canonical_source_path="/synthetic/project/poison.jsonl",
                acquired_at_ms=1,
            )
        assert _run(tmp_path, compute_adapter=compute_adapter).failed == 1
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)

        def select() -> tuple[str, ...]:
            discovery = RawMaterializationDiscovery(tmp_path)
            if lane == "dependents":
                # This is the existing disposable project continuation; the row
                # and refusal authority are read from actual Source receipts.
                discovery._evidence_projects.append(("/synthetic/project", "", -1))
            selectors = {
                "arrival": discovery._arrival_selected,
                "dependents": discovery._dependents_selected,
                "sweep": discovery._sweep_selected,
            }
            return selectors[lane](frame, adapter, 8)

        assert select() == ()
        with sqlite3.connect(tmp_path / "source.db") as source:
            # A census-settled decode carrier records no validation failure, so
            # its typed kind and support status, not worker provenance, carry
            # its authority; an inconsistent carrier is no refusal.
            source.execute("UPDATE raw_artifacts SET support_status='unknown' WHERE raw_id=?", (poison,))
        assert select() == (poison,)

    _run_raw_law(tmp_path, run_phase)


def test_zero_output_requires_parser_evidence(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Anti-vacuity: a bare empty index cannot certify a zero-output raw."""
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ())
        assert _run(tmp_path, compute_adapter=compute_adapter).done == 1
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
        assert _run(tmp_path, compute_adapter=compute_adapter).work.published == 0
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("DELETE FROM raw_membership_census WHERE raw_id = ?", (raw_id,))
            conn.execute("DELETE FROM raw_artifacts WHERE raw_id = ?", (raw_id,))
            conn.commit()
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize("fingerprint_field", ["parser_fingerprint", "lowering_fingerprint"])
def test_inspection_rejects_stale_parser_recipe_and_excess_identity(tmp_path: Path, fingerprint_field: str) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("expected",))
        assert _run(tmp_path, compute_adapter=compute_adapter).done == 1
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(f"UPDATE sessions SET {fingerprint_field} = 'stale'")
            conn.commit()
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute("UPDATE sessions SET native_id = 'excess'")
            conn.commit()
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "excess"

    _run_raw_law(tmp_path, run_phase)


def test_existing_head_does_not_certify_missing_observation_application(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Anti-vacuity: a valid sibling head cannot replace this raw's decision."""
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("application",))
        assert _run(tmp_path, compute_adapter=compute_adapter).done == 1
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute("DELETE FROM raw_revision_applications WHERE raw_id = ?", (raw_id,))
            conn.commit()
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        assert adapter.inspect(raw_observation_frame(tmp_path), (raw_id,))[raw_id] == "missing"
        assert _run(tmp_path, compute_adapter=compute_adapter).done == 1

    _run_raw_law(tmp_path, run_phase)


def test_discovery_budget_bounds_raw_enumeration(tmp_path: Path) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """A discovery bound stops enumeration mid-domain and defers, never drops, the rest.

        Anti-vacuity: let the adapter enumerate past its page/discovery bound, mark
        the cursor swept while keys remain unread, or fail to reach the deferred
        raws on later passes, and this goes red.
        """
        bootstrap_archive_root(tmp_path)
        for index in range(5):
            _admit(tmp_path, (f"session-{index}",), path=f"{index}.json")

        registry = DerivationRegistry((RawObservationDerivation(tmp_path, compute_adapter=compute_adapter),))
        budget = Budget(page=2, discovery=2, inspection=2, compute=2)

        report = converge(
            registry,
            raw_observation_frame(tmp_path),
            budget=budget,
            publisher=lambda actor, work: _writer(tmp_path, actor, work),
        )

        # The bound is a bound: one pass may not read the whole five-raw domain.
        assert report.work.discovered == 2
        assert report.work.published <= 2
        assert not report.cursor.position("raw_observation").swept

        # What the bound withheld is deferred, not dropped: bounded passes reach
        # every raw, and the domain then settles with nothing left to enumerate.
        cursor = report.cursor
        for _ in range(16):
            report = converge(
                registry,
                raw_observation_frame(tmp_path),
                budget=budget,
                cursor=cursor,
                publisher=lambda actor, work: _writer(tmp_path, actor, work),
            )
            cursor = report.cursor

        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 5

        assert _run(tmp_path, compute_adapter=compute_adapter).made_no_publication_attempts

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize(
    "field", ["source_revision", "accepted_source_revision", "decision_id", "accepted_content_hash"]
)
def test_inspection_requires_exact_application_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Red twin: bypassing receipt validation falsely certifies a forged application."""
        from polylogue.storage.derived import raw as raw_adapter

        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("receipt",))
        assert _run(tmp_path, compute_adapter=compute_adapter).done == 1
        frame = raw_observation_frame(tmp_path)
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"
        with sqlite3.connect(tmp_path / "index.db") as conn:
            value = bytes(32) if field == "accepted_content_hash" else "forged"
            conn.execute(f"UPDATE raw_revision_applications SET {field} = ? WHERE raw_id = ?", (value, raw_id))
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "stale"
        from polylogue.storage import raw_authority

        with adapter.read_current() as conn:
            statements: list[str] = []
            conn.set_trace_callback(statements.append)
            plan = raw_authority.build_raw_replay_plan(conn, (raw_id,))
            receipt = raw_authority.raw_replay_application_receipt_from_connection(
                conn, plan, index_db_path=tmp_path / "index.db"
            )
            expected = raw_authority.validate_raw_replay_application_receipt(plan, receipt)
            full_reads = sum(statement.lstrip().upper().startswith("SELECT") for statement in statements)
            statements.clear()
            observed = raw_authority.assess_raw_replay_materialization(
                conn, (raw_id,), index_db_path=tmp_path / "index.db"
            )
            projected_reads = sum(statement.lstrip().upper().startswith("SELECT") for statement in statements)
            conn.set_trace_callback(None)
        assert observed == expected
        assert not observed[0]
        assert full_reads == 11
        assert projected_reads == 7
        monkeypatch.setattr(raw_adapter, "assess_raw_replay_materialization", lambda *_args, **_kwargs: (True, ()))
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize(
    "error",
    [
        "OperationalError: database is locked",
        "MembershipReplayConflictError: retained comparison",
        "RuntimeError: raw revision CAS rejected an older accepted frontier",
        "RuntimeError: membership replay cannot replace an unconvertible byte head",
        "decode: No such file or directory retained-blob",
    ],
)
def test_historical_replay_refusal_is_retried_by_canonical_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Red twin: removing the legacy spelling bridge strands retained source work."""
        from polylogue.storage.derived import raw as raw_adapter

        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("retry",))
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET parse_error = ? WHERE raw_id = ?", (error, raw_id))
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "missing"
        monkeypatch.setattr(raw_adapter, "raw_replay_error_is_retryable", lambda *_args: False)
        assert adapter.inspect(frame, (raw_id,))[raw_id] == "valid"

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.timeout(600)
def test_all_valid_prefix_has_a_total_discovery_bound_and_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """One discovery page reads a bounded slice of a large all-valid archive.

        Anti-vacuity: a LIMIT applied after a full sort still consumes every
        row; the red twin below measures that plan.
        """
        from tests.infra.sqlite_work_counter import sqlite_work_counter

        bootstrap_archive_root(tmp_path)
        source = tmp_path / "sources"
        # 2048 members keep the red twin's sorted-scope work near twice the
        # bound while one single-pass census of the whole component fits the
        # managed cutoff.
        members = 2048
        with _fixture_archive(tmp_path) as archive:
            raw_ids = [
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=b"[]",
                    source_path=str(source / "prefix.json"),
                    canonical_source_path=str(source / "prefix.json"),
                    source_index=index,
                    acquired_at_ms=1,
                )
                for index in range(members)
            ]
        assert len(set(raw_ids)) == members
        # A real source component lets one canonical publication census every
        # acquired observation. Single-pass convergence commits that census in
        # place during preparation; the observer counts the members of the
        # census receipts a successful publication reports, not setup rows or
        # a synthetic inspection verdict.
        censused: list[str] = []
        publish = RawObservationDerivation.publish

        def counted_publish(
            self: RawObservationDerivation, frame: RawFrame, replacement: RawObservationReplacement, **kwargs: Any
        ) -> bool:
            committed = publish(self, frame, replacement, **kwargs)
            if committed:
                for _phase, receipt in replacement.committed_phase_receipts:
                    censused.extend(receipt.input_raw_ids)
            return committed

        with monkeypatch.context() as setup:
            setup.setattr(RawObservationDerivation, "publish", counted_publish)
            publication = converge(
                DerivationRegistry((RawObservationDerivation(tmp_path, compute_adapter=compute_adapter),)),
                raw_observation_frame(tmp_path, raw_ids=(raw_ids[0],)),
                budget=Budget(page=1, discovery=1, inspection=2, compute=1, publication=1),
                publisher=lambda actor, work: _writer(tmp_path, actor, work),
            )
        assert publication.done == 1 and publication.failed == publication.pending == 0
        assert len(censused) == len(set(censused)) == members
        assert set(censused) == set(raw_ids)
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        with sqlite_work_counter(step_interval=1) as indexed:
            keys, continuation = adapter.required_page(frame, cursor=None, limit=128)
        assert len(keys) == 128 and continuation is not None
        assert indexed.metric("vm_steps", "source") < 10_000, indexed.summary()
        # Red twin: a LIMIT after sorting by an unindexed column still consumes
        # every row. Returned-row counts alone cannot detect that work.
        with sqlite_work_counter(step_interval=1) as sorted_scope:
            with sqlite3.connect(tmp_path / "source.db") as conn:
                rows = conn.execute(
                    "SELECT raw_id FROM raw_sessions NOT INDEXED ORDER BY acquired_at_ms, raw_id LIMIT 128"
                ).fetchall()
        assert len(rows) == 128
        assert sorted_scope.metric("vm_steps", "source") > 10_000

    _run_raw_law(tmp_path, run_phase)


@pytest.mark.parametrize("admission", ["held", "rejected", "publication_failure"])
def test_computed_raw_carrier_settles_at_its_actual_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admission: str
) -> None:
    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        """Abandonment closes a real Raw carrier; started publication owns its close.

        The admission changes only after the production compute returned, so a
        pre-compute barrier check cannot make the cleanup assertion vacuous.
        """
        from polylogue.sources import revision_backfill

        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("abandoned",))
        held: set[str] = set()
        captured: list[RawObservationReplacement] = []
        closes: list[RawObservationReplacement] = []
        original_compute = RawObservationDerivation.compute
        original_close = RawObservationReplacement.close

        class ObservedRaw(RawObservationDerivation):
            def barrier_sessions(self, frame: RawFrame, keys: tuple[str, ...]) -> dict[str, str]:
                return dict.fromkeys(keys, "chatgpt:abandoned")

            def compute(
                self,
                frame: RawFrame,
                key: str,
                *,
                replay_current: bool = False,
                select_retained_raw_ids: Callable[[PreparedSessionSourceRead], Sequence[str]] | None = None,
            ) -> RawObservationReplacement:
                replacement = original_compute(
                    self,
                    frame,
                    key,
                    replay_current=replay_current,
                    select_retained_raw_ids=select_retained_raw_ids,
                )
                captured.append(replacement)
                return replacement

        def close(replacement: RawObservationReplacement) -> None:
            closes.append(replacement)
            original_close(replacement)

        def fail_publication(*args: object, **kwargs: object) -> None:
            # The census commits in place while compute prepares; the replay
            # publication that publish starts is the one that fails.
            assert captured[0].reference_seal is not None
            assert captured[0].reference_seal.publication_lifetime_bound
            raise OSError("synthetic publication failure")

        def publish(actor: str, work: Callable[[], bool]) -> bool:
            assert len(captured) == 1
            assert captured[0].reference_seal is not None
            assert captured[0].scratch_directory is not None
            assert captured[0].prepared_inputs
            if admission == "rejected":
                raise OSError("synthetic admission rejection")
            if admission == "held":
                held.add("chatgpt:abandoned")
            return _writer(tmp_path, actor, work)

        monkeypatch.setattr(RawObservationReplacement, "close", close)
        if admission == "publication_failure":
            monkeypatch.setattr(revision_backfill, "apply_prepared_revision_replay", fail_publication)
        report = converge(
            DerivationRegistry((ObservedRaw(tmp_path, compute_adapter=compute_adapter),)),
            raw_observation_frame(tmp_path, raw_ids=(raw_id,)),
            publisher=publish,
            barrier=lambda sessions: set(sessions).intersection(held),
        )
        assert report.done == 0
        assert report.pending == int(admission == "held")
        assert report.failed == int(admission != "held")
        assert len(captured) == len(closes) == 1
        replacement = captured[0]
        assert closes[0] is replacement
        assert replacement.reference_seal is not None
        assert replacement.reference_seal.publication_lifetime_bound == (admission == "publication_failure")
        if replacement.scratch_directory is not None:
            assert not replacement.scratch_directory.exists()
        with sqlite3.connect(tmp_path / "index.db") as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    _run_raw_law(tmp_path, run_phase)


def test_raw_publication_rejects_foreign_original_seal_before_binding_lifetime(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path / "other")

    def run_phase(compute_adapter: BoundedComputeAdapter) -> None:
        from dataclasses import replace

        from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

        root = tmp_path / "selected"
        other = tmp_path / "other"
        bootstrap_archive_root(root)
        raw_id = _admit(root, ("original-seal",))
        adapter = RawObservationDerivation(root, compute_adapter=compute_adapter)
        frame = raw_observation_frame(root, raw_ids=(raw_id,))
        prepared = adapter.compute(frame, raw_id)
        assert prepared.reference_seal is not None
        prepared.reference_seal.close()
        foreign = PreparedIndexMutation(other / "index.db", archive_root=other)
        moved = replace(prepared, reference_seal=foreign)
        before = _snapshot(root)
        with pytest.raises(RuntimeError):
            _publish(adapter, frame, moved)
        assert not foreign.publication_lifetime_bound
        assert foreign._closed
        assert _snapshot(root) == before
        assert moved.scratch_directory is not None and not moved.scratch_directory.exists()
        with sqlite3.connect(root / "index.db") as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    _run_raw_law(tmp_path / "selected", run_phase)
