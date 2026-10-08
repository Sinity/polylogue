"""Original acquired Source phase receipts and terminal laws."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.revision_backfill import RevisionCensusResult
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.agent_thread_state import read_thread_titles
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.retained_jsonl import (
    prepared_source_fixture,
    retained_raw_fixture,
    run_retained_source_phase,
)


def test_backfill_persists_detected_provider_for_empty_ordinary_session_path(tmp_path: Path) -> None:
    """Empty replay retains parser identity without changing original acquisition identity."""
    from polylogue.storage.raw_retention import RawRetentionAuthority, active_raw_retention_authority

    bootstrap_archive_root(tmp_path)
    payload = (
        b'{"type":"file-history-snapshot","messageId":"history-message",'
        b'"sessionId":"history-only-session","snapshot":{"trackedFileBackups":{}}}\n'
    )
    path = str(tmp_path / ".claude" / "projects" / "proj" / "history-only-session.jsonl")
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(root=tmp_path, provider=Provider.UNKNOWN, blob_hash=digest, source_path=path) as (
        reader,
        raw_id,
    ):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, (raw_id,)))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute(
            "SELECT origin, detected_provider, parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("unknown-export", "claude-code", 1)
        assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id=?", (raw_id,)).fetchall() == [
            ("file_history_snapshot",)
        ]
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone() == (
            "non_session",
        )
        assert active_raw_retention_authority(conn, index_db_path=tmp_path / "index.db") == RawRetentionAuthority(
            protected_raw_ids=frozenset({raw_id}),
            eligible_raw_ids=frozenset(),
        )
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_parser_census_is_current(raw_id)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=digest,
        source_path=path,
        acquired_at_ms=2,
    ) as (_reader, repeated_raw_id):
        assert repeated_raw_id == raw_id


def test_backfill_settles_an_undetected_unknown_shape_as_typed_non_session(tmp_path: Path) -> None:
    """An unsupported shape on unchanged bytes is a settled, typed refusal.

    The same parser can only fail the same captured bytes the same way, so
    the census settles it as non-session with typed terminal evidence rather
    than a failed census every pass would repeat. It is not a decode refusal.
    """
    bootstrap_archive_root(tmp_path)
    payload = b'{"future_provider_shape":true}\n'
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.UNKNOWN,
        blob_hash=digest,
        source_path=str(tmp_path / "future.jsonl"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, (raw_id,)))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 1 and receipt.quarantined == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone() == (
            "non_session",
        )
        assert conn.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id=?", (raw_id,)).fetchall() == [
            ("terminal_unknown_export_no_session",)
        ]
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_terminal_decode_refusal(raw_id) is None


def test_backfill_fallback_terminalization_preserves_each_source_index(tmp_path: Path) -> None:
    """One actual Source phase retains both acquired artifact coordinates."""
    bootstrap_archive_root(tmp_path)
    path = str(tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / "wf" / "journal.jsonl")
    older = b'{"contentKey":"older","agentId":"agent"}\n'
    head = older + b'{"contentKey":"head","agentId":"agent"}\n'
    acquired: list[str] = []
    for index, payload, instant in ((4, older, 1), (9, head, 2)):
        digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
        with retained_raw_fixture(
            root=tmp_path,
            provider=Provider.CLAUDE_CODE,
            blob_hash=digest,
            source_path=path,
            source_index=index,
            acquired_at_ms=instant,
        ) as (reader, raw_id):
            assert reader.raw_revision_descriptor(raw_id)[1] == digest
            acquired.append(raw_id)
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, acquired))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 2
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT source_index FROM raw_artifacts ORDER BY source_index").fetchall() == [(4,), (9,)]


def test_unbatched_census_persists_parser_observed_receipt_before_close(tmp_path: Path) -> None:
    """An admitted Source phase physically closes before its receipt is independently read."""
    import json

    from tests.infra.retained_parser_payloads import _chatgpt_session

    bootstrap_archive_root(tmp_path)
    payload = json.dumps(_chatgpt_session("receipt-owner", "neutral original text")).encode()
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CHATGPT,
        blob_hash=digest,
        source_path=str(tmp_path / "receipt.json"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest
    published, phase, census = asyncio.run(run_retained_source_phase(tmp_path, (raw_id,)))
    assert published is False and phase == "census"
    assert isinstance(census, RevisionCensusResult) and census.scanned == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        receipt = conn.execute(
            "SELECT status, detail FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone()
    assert receipt is not None and receipt[0] == "complete"
    assert str(receipt[1]).startswith("parser-observed:")


def test_batched_terminal_artifact_receipts_roll_back_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fault at the second actual Source receipt rolls back the entire original cohort."""
    from collections.abc import Iterator
    from contextlib import contextmanager

    from polylogue.storage.derived.raw import RawObservationReplacement
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    bootstrap_archive_root(tmp_path)
    acquired: list[str] = []
    for index in range(2):
        payload = f'{{"contentKey":"workflow-{index}","agentId":"agent"}}\n'.encode()
        digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
        path = str(
            tmp_path / ".claude" / "projects" / "proj" / "subagents" / "workflows" / f"wf-{index}" / "journal.jsonl"
        )
        with retained_raw_fixture(
            root=tmp_path,
            provider=Provider.CLAUDE_CODE,
            blob_hash=digest,
            source_path=path,
            acquired_at_ms=index + 1,
        ) as (reader, raw_id):
            assert reader.raw_revision_descriptor(raw_id)[1] == digest
            acquired.append(raw_id)
    calls = 0
    original_cursor = PreparedIndexMutation._owned_cursor

    def install_fault(replacement: RawObservationReplacement) -> None:
        seal = replacement.reference_seal
        assert seal is not None and replacement.prepared_source_census is not None

        @contextmanager
        def fail_second_receipt(
            current: PreparedIndexMutation,
            connection: sqlite3.Connection,
            sql: str,
            parameters: tuple[object, ...] = (),
        ) -> Iterator[sqlite3.Cursor]:
            nonlocal calls
            permit = current._pending_tier_permits.get("source")
            if (
                current is seal
                and permit is not None
                and connection is permit._connection
                and sql.lstrip().upper().startswith("INSERT")
                and "raw_authority_parser_census" in sql
            ):
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected second terminal census failure")
            with original_cursor(current, connection, sql, parameters) as cursor:
                yield cursor

        monkeypatch.setattr(PreparedIndexMutation, "_owned_cursor", fail_second_receipt)

    with pytest.raises(RuntimeError, match="injected second terminal census failure"):
        asyncio.run(run_retained_source_phase(tmp_path, acquired, before_publication=install_fault))
    assert calls == 2
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        placeholders = ",".join("?" for _raw_id in acquired)
        assert conn.execute(
            f"SELECT COUNT(*) FROM raw_artifacts WHERE raw_id IN ({placeholders})",
            acquired,
        ).fetchone() == (0,)
        assert conn.execute(
            f"SELECT COUNT(*) FROM raw_authority_parser_census WHERE raw_id IN ({placeholders})",
            acquired,
        ).fetchone() == (0,)


def test_backfill_replays_codex_state_by_latest_raw_observation(tmp_path: Path) -> None:
    """A genuinely reacquired A/B/A sequence retains A's latest observation."""
    from tests.infra.retained_parser_payloads import _codex_thread_state_snapshot_bytes

    bootstrap_archive_root(tmp_path)
    path = str(tmp_path / "codex" / "state_5.sqlite")
    snapshot_a = _codex_thread_state_snapshot_bytes(tmp_path, "title A")
    snapshot_b = _codex_thread_state_snapshot_bytes(tmp_path, "title B")
    acquired: list[str] = []
    for payload in (snapshot_a, snapshot_b, snapshot_a):
        digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
        with retained_raw_fixture(
            root=tmp_path,
            provider=Provider.CODEX,
            blob_hash=digest,
            source_path=path,
        ) as (reader, raw_id):
            assert reader.raw_revision_descriptor(raw_id)[1] == digest
            acquired.append(raw_id)
    assert acquired[0] == acquired[2] and acquired[0] != acquired[1]

    async def replay() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            (await owner.replay_retained_raw_ids(acquired)).require_complete()

    asyncio.run(replay())
    with closing(sqlite3.connect(tmp_path / "index.db")) as conn:
        assert read_thread_titles(conn, thread_ids=["codex-state-thread"]) == {"codex-state-thread": "title A"}


def test_backfill_replays_equal_time_codex_state_by_raw_acquisition_order(tmp_path: Path) -> None:
    """Actual durable acquisition order wins over lexical fixture IDs at equal timestamps."""
    from tests.infra.retained_parser_payloads import _codex_thread_state_snapshot_bytes

    bootstrap_archive_root(tmp_path)
    path = str(tmp_path / "codex" / "state_5.sqlite")
    acquired: list[str] = []
    for title, raw_identity in (("older title", "z-older-state"), ("newer title", "a-newer-state")):
        payload = _codex_thread_state_snapshot_bytes(tmp_path, title)
        digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
        with retained_raw_fixture(
            root=tmp_path,
            provider=Provider.CODEX,
            blob_hash=digest,
            source_path=path,
            acquired_at_ms=1,
            raw_id=raw_identity,
        ) as (reader, raw_id):
            assert raw_id == raw_identity
            assert reader.raw_revision_descriptor(raw_id)[1] == digest
            acquired.append(raw_id)

    async def replay() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            (await owner.replay_retained_raw_ids(acquired)).require_complete()

    asyncio.run(replay())
    with closing(sqlite3.connect(tmp_path / "index.db")) as conn:
        assert read_thread_titles(conn, thread_ids=["codex-state-thread"]) == {"codex-state-thread": "newer title"}


def test_codex_state_preparation_preserves_valid_input_without_a_payload_cutoff(tmp_path: Path) -> None:
    """Valid state inputs prepare under the real owner without off-writer evidence effects."""
    from polylogue.storage.derived.raw import RawObservationReplacement
    from tests.infra.retained_parser_payloads import _codex_thread_state_snapshot_bytes

    bootstrap_archive_root(tmp_path)
    payload = _codex_thread_state_snapshot_bytes(tmp_path, "oversized state")
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CODEX,
        blob_hash=digest,
        source_path=str(tmp_path / "codex" / "state_5.sqlite"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest

    def before_publication(replacement: RawObservationReplacement) -> None:
        assert replacement.prepared_source_census is not None
        with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
            assert conn.execute(
                "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE raw_id=?",
                (raw_id,),
            ).fetchone() == (None, None)
            assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone() == (0,)
        with closing(sqlite3.connect(tmp_path / "index.db")) as conn:
            assert conn.execute("SELECT COUNT(*) FROM work_evidence_nodes").fetchone() == (0,)

    published, phase, receipt = asyncio.run(
        run_retained_source_phase(tmp_path, (raw_id,), before_publication=before_publication)
    )
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 1


def test_legacy_codex_page_image_does_not_abort_census(tmp_path: Path) -> None:
    """An explicitly acquired external page image leaves a terminal receipt."""
    from polylogue.sources.revision_backfill import LEGACY_PAGE_IMAGE_CENSUS_DETAIL
    from tests.infra.retained_parser_payloads import _codex_thread_state_page_image_bytes

    bootstrap_archive_root(tmp_path)
    payload = _codex_thread_state_page_image_bytes(tmp_path, "neutral external page image")
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CODEX,
        blob_hash=digest,
        source_path=str(tmp_path / "codex" / "state_5.sqlite"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, (raw_id,)))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        detail = conn.execute("SELECT detail FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone()
    assert detail is not None
    assert LEGACY_PAGE_IMAGE_CENSUS_DETAIL in str(detail[0])


def test_backfill_retires_stale_revision_governance_for_empty_replay(tmp_path: Path) -> None:
    """Current zero-session evidence retires a stale quarantined FULL plan."""
    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease

    bootstrap_archive_root(tmp_path)
    payload = b'{"type":"file-history-snapshot","sessionId":"history-only","snapshot":{}}\n'
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=digest,
        source_path=str(tmp_path / ".claude" / "projects" / "proj" / "history-only-session.jsonl"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest
    with (
        write_lease("test.retained.stale-plan", archive_root=tmp_path),
        ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
    ):
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                logical_source_key="claude-code-session:stale-session",
                kind=RawRevisionKind.FULL,
                source_revision=digest,
                acquisition_generation=0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        archive.commit()
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, (raw_id,)))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute(
            "SELECT logical_source_key, revision_kind, revision_authority FROM raw_sessions WHERE raw_id=?",
            (raw_id,),
        ).fetchone() == (None, "unknown", "quarantined")


def test_backfill_preserves_empty_append_revision_governance(tmp_path: Path) -> None:
    """A real terminal APPEND retains its original reconstructible byte range."""
    from tests.infra.retained_jsonl import retained_append_fixture

    bootstrap_archive_root(tmp_path)
    baseline = b'{"type":"file-history-snapshot","sessionId":"append-only","snapshot":{}}\n'
    delta = b'{"type":"file-history-snapshot","sessionId":"append-only","snapshot":{"trackedFileBackups":{}}}\n'
    with retained_append_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        source_path=tmp_path / ".claude" / "projects" / "proj" / "append.jsonl",
        native_id="append-only",
        logical_source_key="claude-code-session:append-only",
        baseline=baseline,
        delta=delta,
    ) as (reader, baseline_raw_id, raw_id, predecessor, _revision, start, end):
        assert reader.raw_revision_descriptor(raw_id)[0] == Provider.CLAUDE_CODE
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, (baseline_raw_id, raw_id)))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 2
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute(
            "SELECT logical_source_key, revision_kind, predecessor_source_revision, predecessor_raw_id, baseline_raw_id, append_start_offset, append_end_offset, acquisition_generation, revision_authority FROM raw_sessions WHERE raw_id=?",
            (raw_id,),
        ).fetchone() == (
            "claude-code-session:append-only",
            "append",
            predecessor,
            baseline_raw_id,
            baseline_raw_id,
            start,
            end,
            1,
            "byte_proven",
        )
        assert conn.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone() == (
            "non_session",
        )


def test_terminal_non_session_reselection_repairs_legacy_parser_receipt(tmp_path: Path) -> None:
    """Reobservation replaces an incomplete receipt without inventing sessions."""
    from polylogue.storage.raw_authority import iter_parser_census_logical_keys
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease

    bootstrap_archive_root(tmp_path)
    payload = b'{"type":"file-history-snapshot","sessionId":"receipt-only","snapshot":{}}\n'
    digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=tmp_path,
        provider=Provider.CLAUDE_CODE,
        blob_hash=digest,
        source_path=str(tmp_path / ".claude" / "projects" / "proj" / "receipt-only.jsonl"),
    ) as (reader, raw_id):
        assert reader.raw_revision_descriptor(raw_id)[1] == digest
    asyncio.run(run_retained_source_phase(tmp_path, (raw_id,)))
    with (
        write_lease("test.retained.incomplete-receipt", archive_root=tmp_path),
        ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
    ):
        source = archive._ensure_source_conn()
        source.execute(
            "UPDATE raw_authority_parser_census SET status='failed', logical_keys_json='[]', detail='parser-observed: incomplete receipt' WHERE raw_id=?",
            (raw_id,),
        )
        archive.commit()
    with prepared_source_fixture(tmp_path) as reader:
        assert not reader.raw_parser_census_is_current(raw_id)
    published, phase, receipt = asyncio.run(run_retained_source_phase(tmp_path, (raw_id,), replay_current=True))
    assert published is False and phase == "census"
    assert isinstance(receipt, RevisionCensusResult) and receipt.scanned == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        result = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone()
    assert result is not None and result[0] == "complete"
    assert tuple(iter_parser_census_logical_keys(result[1])) == ()
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_parser_census_is_current(raw_id)


def test_census_batching_reduces_commit_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One original cohort publication replaces per-Raw Source publications."""
    import polylogue.sources.revision_backfill as raw_module
    from polylogue.sources.revision_backfill import PreparedRevisionSourceCensus
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    roots = (tmp_path / "individual", tmp_path / "cohort")
    acquisitions: list[list[str]] = []
    for root in roots:
        bootstrap_archive_root(root)
        acquired: list[str] = []
        for index in range(9):
            payload = f'{{"contentKey":"workflow-{index}","agentId":"agent"}}\n'.encode()
            digest, _size = BlobStore(root / "blob").write_from_bytes(payload)
            with retained_raw_fixture(
                root=root,
                provider=Provider.CLAUDE_CODE,
                blob_hash=digest,
                source_path=str(
                    root / ".claude" / "projects" / "proj" / "subagents" / "workflows" / f"wf-{index}" / "journal.jsonl"
                ),
                acquired_at_ms=index + 1,
            ) as (reader, raw_id):
                assert reader.raw_revision_descriptor(raw_id)[1] == digest
                acquired.append(raw_id)
        acquisitions.append(acquired)
    original_apply = raw_module.apply_prepared_revision_census
    publications = 0

    def record_publication(
        seal: PreparedIndexMutation, prepared: PreparedRevisionSourceCensus, *, payload_store: BlobStore
    ) -> RevisionCensusResult:
        nonlocal publications
        result = original_apply(seal, prepared, payload_store=payload_store)
        publications += 1
        return result

    monkeypatch.setattr(raw_module, "apply_prepared_revision_census", record_publication)
    individual_scanned = 0
    for raw_id in acquisitions[0]:
        published, phase, receipt = asyncio.run(run_retained_source_phase(roots[0], (raw_id,)))
        assert published is False and phase == "census" and isinstance(receipt, RevisionCensusResult)
        individual_scanned += receipt.scanned
    assert publications == 9
    publications = 0
    published, phase, receipt = asyncio.run(run_retained_source_phase(roots[1], acquisitions[1]))
    assert published is False and phase == "census" and isinstance(receipt, RevisionCensusResult)
    assert publications == 1 and receipt.scanned == individual_scanned == 9
    outcomes = []
    for root in roots:
        with closing(sqlite3.connect(root / "source.db")) as conn:
            outcomes.append(
                (
                    conn.execute(
                        "SELECT status, COUNT(*) FROM raw_authority_parser_census GROUP BY status ORDER BY status"
                    ).fetchall(),
                    conn.execute(
                        "SELECT status, COUNT(*) FROM raw_membership_census GROUP BY status ORDER BY status"
                    ).fetchall(),
                    conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone(),
                )
            )
    assert outcomes[0] == outcomes[1]


def test_fragment_repair_preserves_durable_membership_while_refreshing_legacy_receipt(tmp_path: Path) -> None:
    """A current fragment receipt refresh retains the actual durable owner key."""
    from polylogue.storage.raw_authority import iter_parser_census_logical_keys
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.retained_jsonl import retained_append_fixture

    bootstrap_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"fragment-owner","timestamp":"2026-08-20T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"baseline"}]}}\n'
    )
    fragment = b'{"type":"response_item","payload":{"type":"message","id":"suffix","role":"assistant","content":[{"type":"output_text","text":"suffix"}]}}\n'
    logical_key = "codex-session:fragment-owner"
    with retained_append_fixture(
        root=tmp_path,
        provider=Provider.CODEX,
        source_path=tmp_path / "fragment.jsonl",
        native_id="fragment-owner",
        logical_source_key=logical_key,
        baseline=baseline,
        delta=fragment,
    ) as (_reader, baseline_raw_id, raw_id, _predecessor, _revision, _start, _end):
        pass

    async def replay() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            (await owner.replay_retained_raw_ids((baseline_raw_id, raw_id))).require_complete()

    asyncio.run(replay())

    async def seed_original_membership() -> None:
        import sys
        from builtins import BaseExceptionGroup

        from polylogue.archive.revision_authority import RawRevisionAuthority
        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.sources.prepared_jsonl import PreparedJsonl
        from polylogue.sources.prepared_merge import prepare_retained_cohort_artifact
        from polylogue.sources.revision_backfill import prepare_retained_jsonl_artifact
        from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
        from polylogue.storage.sqlite.archive_tiers.revision_governance import replace_raw_membership_census
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
        from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

        async with prepared_live_convergence_owner(tmp_path) as owner:
            retained: list[PreparedIndexMutation] = []

            def seed() -> None:
                seal = PreparedIndexMutation.source_only(archive_root=tmp_path)
                retained.append(seal)
                artifacts: list[PreparedJsonl] = []
                try:
                    directory = BlobStore(tmp_path / "blob")._ensure_private_staging_root()
                    with seal.original_read_snapshot(), seal.source_producer():
                        reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(tmp_path / "blob"))
                        ordered: list[tuple[str, PreparedJsonl]] = []
                        for selected_raw_id in (baseline_raw_id, raw_id):
                            artifact = prepare_retained_jsonl_artifact(reader, selected_raw_id, directory=directory)
                            artifacts.append(artifact)
                            assert artifact.error is None and not artifact.deferred
                            ordered.append((selected_raw_id, artifact))
                        aggregate = prepare_retained_cohort_artifact(ordered, directory)
                        artifacts.append(aggregate)
                        assert aggregate.error is None and not aggregate.deferred
                        replace_raw_membership_census(
                            seal,
                            raw_id,
                            aggregate.session_sequence(),
                            parser_fingerprint=raw_authority_parser_fingerprint(),
                            censused_at_ms=1,
                            revision_authority=RawRevisionAuthority.BYTE_PROVEN,
                            retire_full_revision_governance=False,
                        )
                    permit = seal.prepare_source_mutation()

                    def publish() -> None:
                        with permit.hold_authority(), permit.mutation_connection() as source:
                            with closing(source.execute("BEGIN IMMEDIATE")):
                                pass
                            permit.apply_source_statements(source)
                            permit.allow_commit(source)
                            source.commit()
                            seal.accept_known_tier_commit(permit.committed())

                    admit_stage_write("test.retained.fragment-membership.seed", publish)
                finally:
                    primary = sys.exception()
                    failures: list[BaseException] = []
                    for artifact in reversed(artifacts):
                        try:
                            artifact.discard()
                        except BaseException as cleanup:
                            failures.append(cleanup)
                    try:
                        seal.close()
                    except BaseException as cleanup:
                        failures.append(cleanup)
                    if failures:
                        if primary is not None:
                            failures.insert(0, primary)
                        raise BaseExceptionGroup("membership seed and original close failed", failures) from None
                    retained.remove(seal)

            await owner.run_prepared_sync(
                "test.retained.fragment-membership.prepare",
                seed,
                settlement_owners=lambda: tuple(retained),
                estimated_bytes=len(baseline) + len(fragment),
            )

    asyncio.run(seed_original_membership())

    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        original_memberships = conn.execute(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=?",
            (raw_id,),
        ).fetchall()
    assert original_memberships == [(logical_key,)]
    with (
        write_lease("test.retained.fragment-receipt", archive_root=tmp_path),
        ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
    ):
        archive._ensure_source_conn().execute(
            "UPDATE raw_authority_parser_census SET logical_keys_json='[]', detail='parser-observed: incomplete fragment receipt' WHERE raw_id=?",
            (raw_id,),
        )
        archive.commit()
    with prepared_source_fixture(tmp_path) as reader:
        assert not reader.raw_parser_census_is_current(raw_id)
    asyncio.run(run_retained_source_phase(tmp_path, (raw_id,), replay_current=True))
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        memberships = conn.execute(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=?", (raw_id,)
        ).fetchall()
        receipt = conn.execute(
            "SELECT status, logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone()
    assert memberships == original_memberships
    assert receipt is not None and receipt[0] == "complete"
    assert tuple(iter_parser_census_logical_keys(receipt[1])) == (logical_key,)
    with prepared_source_fixture(tmp_path) as reader:
        assert reader.raw_parser_census_is_current(raw_id)


@pytest.mark.parametrize("distinct_scopes", [False, True])
def test_retained_thread_graph_cohort_preserves_absent_evidence_and_scope_parent_laws(
    tmp_path: Path, distinct_scopes: bool
) -> None:
    from polylogue.archive.topology.edge import HOOK_AUTHORITATIVE_LINK_METHOD
    from polylogue.storage.sqlite.agent_thread_state import read_parent_thread_id, thread_state_graph_id
    from tests.infra.retained_thread_graph_payloads import codex_thread_graph_snapshot_bytes

    bootstrap_archive_root(tmp_path)
    scope_a = tmp_path / ".codex"
    scope_b = tmp_path / "other-install" if distinct_scopes else scope_a
    rollouts: list[str] = []
    for native_id in ("parent-a", "parent-b", "child"):
        payload = (
            '{"type":"session_meta","payload":{"id":"'
            + native_id
            + '","timestamp":"2026-08-20T00:00:00Z"}}\n'
            + '{"type":"response_item","payload":{"type":"message","id":"message-'
            + native_id
            + '","role":"user","content":[{"type":"input_text","text":"neutral retained thread"}]}}\n'
        ).encode()
        digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
        with retained_raw_fixture(
            root=tmp_path,
            provider=Provider.CODEX,
            blob_hash=digest,
            source_path=str(scope_a / "sessions" / f"{native_id}.jsonl"),
        ) as (_reader, raw_id):
            rollouts.append(raw_id)

    async def replay_rollouts() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            (await owner.replay_retained_raw_ids(rollouts)).require_complete()

    asyncio.run(replay_rollouts())
    acquired: list[str] = []
    for label, scope, parent, title, absent in (
        ("older", scope_a, "parent-a", "older title", True),
        ("newer", scope_b, "parent-b", "newer title", False),
    ):
        threads = [("child", title)]
        if absent:
            threads.append(("absent-thread", "retained absent title"))
        payload = codex_thread_graph_snapshot_bytes(
            tmp_path, label, threads=threads, spawn_edges=[(parent, "child", "running")]
        )
        digest, _size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
        with retained_raw_fixture(
            root=tmp_path,
            provider=Provider.CODEX,
            blob_hash=digest,
            source_path=str(scope / "state_5.sqlite"),
            acquired_at_ms=1,
            raw_id=f"{label}-graph",
        ) as (reader, raw_id):
            assert reader.raw_revision_descriptor(raw_id)[1] == digest
            acquired.append(raw_id)

    async def replay_graphs() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            (await owner.replay_retained_raw_ids(tuple(reversed(acquired)))).require_complete()

    asyncio.run(replay_graphs())
    with closing(sqlite3.connect(tmp_path / "index.db")) as index:
        assert read_thread_titles(index, thread_ids=["child"]) == {"child": "newer title"}
        assert read_thread_titles(index, thread_ids=["absent-thread"]) == {"absent-thread": "retained absent title"}
        assert read_parent_thread_id(index, "child", source_scope=str(scope_a)) == (
            "parent-a" if distinct_scopes else "parent-b"
        )
        assert read_parent_thread_id(index, "child") == (None if distinct_scopes else "parent-b")
        assert index.execute(
            "SELECT association_state FROM work_evidence_nodes WHERE graph_id=? AND label='retained absent title' "
            "AND node_kind='claim'",
            (thread_state_graph_id(str(scope_a)),),
        ).fetchone() == ("resolved" if distinct_scopes else "superseded",)
        assert index.execute(
            "SELECT dst_native_id FROM session_links WHERE src_session_id='codex-session:child' AND method=?",
            (HOOK_AUTHORITATIVE_LINK_METHOD,),
        ).fetchall() == [("parent-a" if distinct_scopes else "parent-b",)]
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT raw_id FROM raw_sessions WHERE raw_id IN ('older-graph','newer-graph') ORDER BY raw_id"
        ).fetchall() == [("newer-graph",), ("older-graph",)]
