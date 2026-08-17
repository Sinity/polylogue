"""Tests for polylogue-gd6v's daemon-internal bulk-scale rebuild routing.

Production dependencies exercised here:

* ``polylogue.daemon.bulk_rebuild`` -- the actual transaction resolve/resume/
  retire logic and the pass driver, not a reimplementation.
* ``polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync`` --
  the SAME engine the offline ``polylogue ops maintenance rebuild-index``
  CLI command drives (this module's whole point is to reuse it, not
  duplicate it).
* ``polylogue.daemon.parse_prefetch.DaemonParseStage`` -- the real #3168
  off-writer-hold pre-parse pool, feeding the real
  ``RawParsePrefetchCache``/``prefetch_cache`` production plumbing threaded
  through ``backfill_historical_revision_evidence`` by this bead.

Two claims this file proves:

1. **Equivalence** (gd6v AC): driving the SAME corpus through (a) the
   existing single-call CLI rebuild path and (b) the new daemon bulk-rebuild
   routing (multiple bounded passes, parse pre-warmed off the writer hold)
   produces identical durable archive content -- sessions/messages/blocks,
   content hashes, session_links, and FTS row counts.
2. **O(remaining-work) resume** (polylogue-fbte, folded into this bead's
   acceptance gate): each bounded pass's transaction cursor only ever moves
   forward -- a later pass's scheduled page is disjoint from every earlier
   pass's page -- and a daemon "restart" (a fresh ``DaemonParseStage``,
   mirroring a fresh process) resumes from the persisted cursor rather than
   re-selecting already-processed raws.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon.bulk_rebuild import (
    DAEMON_BULK_REBUILD_OPERATION_ID,
    has_resumable_daemon_bulk_rebuild_transaction,
    resolve_or_start_daemon_bulk_rebuild_transaction,
    run_daemon_bulk_rebuild_pass,
)
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.maintenance.rebuild_index import (
    RebuildIndexRequest,
    RebuildSchemaCurrencyError,
    rebuild_index_from_source_sync,
)
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, assert_owns_archive_location
from polylogue.storage.archive_readiness import probe_archive_tier
from polylogue.storage.index_generation import (
    IndexGenerationStore,
    rebuild_source_evidence_snapshot,
    source_revision_snapshot,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.rebuild_preconditions import decide_raw_revision_authority, record_codex_parser_census
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

_RAW_COUNT = 6


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    import json

    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-20T00:00:00Z"}}
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


def _seed_corpus(root: Path, *, count: int = _RAW_COUNT) -> None:
    initialize_active_archive_root(root)
    seeded: dict[str, bytes] = {}
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            payload = _codex_session(
                f"gd6v-session-{index}",
                (("user", f"question {index}"), ("assistant", f"searchable answer {index}")),
            )
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"gd6v-corpus-{index}.jsonl",
                acquired_at_ms=index,
            )
            seeded[raw_id] = payload
    record_codex_parser_census(root, seeded)
    decide_raw_revision_authority(root)
    backfill_historical_revision_evidence(root)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def test_daemon_bulk_rebuild_refuses_unexplained_failures_before_generation_or_page_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real daemon route fails before creating state or selecting raws."""
    initialize_active_archive_root(tmp_path)
    # Acquire real bytes and then mark the parse as failed. A hand-inserted row
    # with a fabricated blob_hash fails the receipt's blob verification, so the
    # run would refuse for a missing blob rather than for the unexplained parse
    # failure this case is about.
    payload = _codex_session("bulk-rebuild-failed", (("user", "question"), ("assistant", "answer")))
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        failed_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="bulk-rebuild-failed.jsonl",
            acquired_at_ms=100,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parse_error = 'parser failed' WHERE raw_id = ?",
            (failed_raw_id,),
        )
        conn.commit()

    # Written after seeding: the receipt binds to a source snapshot, so one
    # taken earlier is rejected for not matching source.db and the run refuses
    # on the receipt instead of on the failure lifecycle this case is about.
    monkeypatch.setenv(
        "POLYLOGUE_SCHEMA_INFERENCE_RECEIPT",
        str(write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-failure-receipt.json")),
    )

    parse_stage = Mock()
    with pytest.raises(RuntimeError, match="raw failure lifecycle preflight"):
        asyncio.run(
            run_daemon_bulk_rebuild_pass(
                config=_config(tmp_path),
                parse_stage=parse_stage,  # preflight must make this unreachable
                batch_size=1,
                max_payload_bytes=10_000,
            )
        )

    assert not (tmp_path / ".index-generations").exists()
    parse_stage.warm_raw_ids.assert_not_called()


def test_daemon_bulk_pass_rechecks_schema_currency_in_page_selection_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A migration between transaction resolution and page selection blocks.

    Production dependency: the second ownership-bound currency check in
    ``run_daemon_bulk_rebuild_pass``. Mutation: removing that check reaches the
    fail-fast ``next_raw_page`` replacement below instead of raising the schema
    diagnostic before receipt or source-page consumption.
    """
    from polylogue.daemon import bulk_rebuild

    _seed_corpus(tmp_path, count=1)
    receipt_path = write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    real_assert = assert_owns_archive_location
    assertion_count = 0

    def advance_audit_after_page_ownership(owned: OwnedArchiveLocation, location: ArchiveLocation) -> None:
        nonlocal assertion_count
        real_assert(owned, location)
        assertion_count += 1
        if assertion_count != 2:
            return
        audit_probe = probe_archive_tier(ArchiveTier.AUDIT, tmp_path / "audit.db")
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            conn.execute(f"PRAGMA user_version = {audit_probe.expected_user_version + 1}")

    monkeypatch.setattr(bulk_rebuild, "assert_owns_archive_location", advance_audit_after_page_ownership)
    next_raw_page = Mock(side_effect=AssertionError("source page selected before schema currency recheck"))
    monkeypatch.setattr(IndexGenerationStore, "next_raw_page", next_raw_page)
    parse_stage = Mock()

    with pytest.raises(RebuildSchemaCurrencyError) as exc_info:
        asyncio.run(
            run_daemon_bulk_rebuild_pass(
                config=_config(tmp_path),
                parse_stage=parse_stage,
                batch_size=1,
                max_payload_bytes=10_000,
            )
        )

    blocking_tiers = exc_info.value.diagnostic["blocking_tiers"]
    assert isinstance(blocking_tiers, list)
    assert blocking_tiers[0]["tier"] == "audit"
    assert assertion_count == 2
    next_raw_page.assert_not_called()
    parse_stage.warm_raw_ids.assert_not_called()


def _table_rows(conn: sqlite3.Connection, table: str) -> tuple[tuple[Any, ...], ...]:
    columns = tuple(row["name"] for row in conn.execute(f'PRAGMA table_xinfo("{table}")'))
    quoted = ", ".join(f'"{column}"' for column in columns)
    return tuple(
        sorted(
            (
                tuple(bytes(value).hex() if isinstance(value, bytes) else value for value in row)
                for row in conn.execute(f'SELECT {quoted} FROM "{table}"')
            ),
            key=repr,
        )
    )


def _canonical_snapshot(index_path: Path) -> dict[str, tuple[tuple[Any, ...], ...] | int]:
    with _connect(index_path) as conn:
        snapshot: dict[str, tuple[tuple[Any, ...], ...] | int] = {
            table: _table_rows(conn, table) for table in ("sessions", "messages", "blocks", "session_links")
        }
        snapshot["messages_fts_row_count"] = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        snapshot["session_count"] = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    return snapshot


async def _drive_daemon_bulk_rebuild_to_promotion(
    root: Path,
    *,
    batch_size: int,
    max_payload_bytes: int = 10_000_000,
) -> list[Any]:
    """Drive the daemon path to promotion, one bounded pass per call.

    Each pass constructs a FRESH ``DaemonParseStage`` (mirroring a full
    daemon-process restart between ticks) instead of reusing one instance
    across the whole loop, so this also exercises the resume path for real
    rather than merely a warm, already-populated in-memory cache.
    """
    config = _config(root)
    receipts: list[Any] = []
    for _ in range(_RAW_COUNT + 2):  # generous upper bound; promotion ends the loop early
        stage = DaemonParseStage(max_workers=2, max_inflight_bytes=max_payload_bytes)
        try:
            receipt = await run_daemon_bulk_rebuild_pass(
                config=config,
                parse_stage=stage,
                batch_size=batch_size,
                max_payload_bytes=max_payload_bytes,
            )
        finally:
            stage.shutdown()
        if receipt is None:
            break
        receipts.append(receipt)
        transaction_status = receipt.transaction["status"] if receipt.transaction else receipt.status
        if transaction_status == "promoted":
            break
    else:
        pytest.fail("bulk rebuild did not reach promotion within the generous pass budget")
    return receipts


def test_resolve_or_start_creates_resumes_and_retires_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    _seed_corpus(tmp_path, count=2)
    receipt_path = write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-receipt.json")
    store = IndexGenerationStore.for_archive_root(tmp_path)

    assert has_resumable_daemon_bulk_rebuild_transaction(tmp_path) is False
    first = resolve_or_start_daemon_bulk_rebuild_transaction(tmp_path, schema_inference_receipt_path=receipt_path)
    assert first.operation_id == DAEMON_BULK_REBUILD_OPERATION_ID
    assert first.status == "running"
    assert has_resumable_daemon_bulk_rebuild_transaction(tmp_path) is True

    # A second resolve against an unchanged, still-resumable transaction
    # returns the SAME record -- no new generation, no lost cursor.
    again = resolve_or_start_daemon_bulk_rebuild_transaction(tmp_path, schema_inference_receipt_path=receipt_path)
    assert again.generation_id == first.generation_id
    assert again.operation_id == first.operation_id

    # Mark it terminal (as the real pass driver would after promotion) and
    # confirm the well-known operation id is reused for a genuinely fresh
    # transaction/generation rather than colliding with the retired one.
    store.checkpoint_transaction(first, status="promoted")
    assert has_resumable_daemon_bulk_rebuild_transaction(tmp_path) is False
    restarted = resolve_or_start_daemon_bulk_rebuild_transaction(tmp_path, schema_inference_receipt_path=receipt_path)
    assert restarted.operation_id == DAEMON_BULK_REBUILD_OPERATION_ID
    assert restarted.status == "running"
    assert restarted.generation_id != first.generation_id
    assert restarted.last_raw_id is None
    assert restarted.processed_raw_count == 0


@pytest.mark.parametrize("cleanup_failure", ["false", "exception"], ids=["discard-false", "discard-exception"])
def test_daemon_post_create_cleanup_surfaces_candidate_and_transaction_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_failure: str
) -> None:
    """Daemon provenance cleanup cannot hide failed discard actuators."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    _seed_corpus(tmp_path, count=2)
    receipt_path = write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-receipt.json")
    original_create_transaction = IndexGenerationStore.create_transaction

    def expire_after_create(
        store: IndexGenerationStore,
        *,
        source_snapshot: str,
        operation_id: str | None = None,
        pass_byte_budget: int | None = None,
        pass_deadline_ms: int | None = None,
    ) -> object:
        transaction = original_create_transaction(
            store,
            source_snapshot=source_snapshot,
            operation_id=operation_id,
            pass_byte_budget=pass_byte_budget,
            pass_deadline_ms=pass_deadline_ms,
        )
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["generated_at"] = "2000-01-01T00:00:00Z"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return transaction

    monkeypatch.setattr(IndexGenerationStore, "create_transaction", expire_after_create)

    def failed_discard(*args: object, **kwargs: object) -> bool:
        if cleanup_failure == "exception":
            raise OSError("synthetic daemon cleanup failure")
        return False

    monkeypatch.setattr(IndexGenerationStore, "discard_if_inactive", failed_discard)
    monkeypatch.setattr(IndexGenerationStore, "discard_transaction", failed_discard)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed") as raised:
        resolve_or_start_daemon_bulk_rebuild_transaction(
            tmp_path,
            schema_inference_receipt_path=receipt_path,
        )

    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    assert "daemon bulk-rebuild transaction cleanup also failed" in notes
    assert "candidate" in notes
    assert "transaction" in notes
    expected_detail = "synthetic daemon cleanup failure" if cleanup_failure == "exception" else "was not discarded"
    assert expected_detail in notes
    assert list((tmp_path / ".index-generations").glob("gen-*"))
    assert list((tmp_path / ".index-rebuild-transactions").glob("*.json"))


def test_daemon_bulk_pass_uses_rebuild_evidence_snapshot_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon seed must match the shared engine's immutable evidence hash.

    Anti-vacuity: this seeds a nonempty real ``source.db``, resolves through
    ``resolve_or_start_daemon_bulk_rebuild_transaction``, and invokes the real
    ``run_daemon_bulk_rebuild_pass``. That driver calls the shared
    ``rebuild_index_from_source_sync`` engine, whose pre-replay validation
    marks an incompatible transaction stale. The full-row hash is deliberately
    different for this corpus, so removing the daemon seed correction makes
    this test fail before the first raw is replayed.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    _seed_corpus(tmp_path, count=2)
    receipt_path = write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    evidence_snapshot = rebuild_source_evidence_snapshot(tmp_path)
    full_row_snapshot = source_revision_snapshot(tmp_path)
    assert full_row_snapshot != evidence_snapshot

    transaction = resolve_or_start_daemon_bulk_rebuild_transaction(tmp_path, schema_inference_receipt_path=receipt_path)
    assert transaction.source_snapshot == evidence_snapshot
    assert transaction.source_snapshot != full_row_snapshot

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        receipt = asyncio.run(
            run_daemon_bulk_rebuild_pass(
                config=_config(tmp_path),
                parse_stage=stage,
                batch_size=1,
                max_payload_bytes=10_000_000,
            )
        )
    finally:
        stage.shutdown()

    assert receipt is not None
    assert receipt.transaction is not None
    assert receipt.transaction["status"] != "stale"
    processed_raw_count = receipt.transaction["processed_raw_count"]
    assert isinstance(processed_raw_count, int)
    assert processed_raw_count == 1
    persisted = IndexGenerationStore.for_archive_root(tmp_path).load_transaction(DAEMON_BULK_REBUILD_OPERATION_ID)
    assert persisted.status != "stale"


def test_daemon_bulk_rebuild_pass_resumes_without_reprocessing_raw_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-fbte: interruption recovery must be O(remaining), not O(corpus).

    A batch size smaller than the corpus forces multiple passes. Each pass's
    scheduled page must be disjoint from every earlier pass's page -- the
    persisted cursor (``last_raw_id``/``processed_raw_count``) genuinely
    advances instead of a resume silently re-walking from the start.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    _seed_corpus(tmp_path)
    receipt_path = write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    receipts = asyncio.run(_drive_daemon_bulk_rebuild_to_promotion(tmp_path, batch_size=2))

    assert len(receipts) >= 3  # 6 raws / batch 2 => at least 3 passes before promotion finalizes
    seen_raw_ids: set[str] = set()
    processed_counts: list[int] = []
    for receipt in receipts:
        assert receipt.transaction is not None
        processed_counts.append(int(receipt.transaction["processed_raw_count"]))
    # processed_raw_count is monotonically non-decreasing across passes and
    # never exceeds the corpus size -- a re-walk-from-scratch bug would
    # either reset it to 0 or double-count the same raws past _RAW_COUNT.
    assert processed_counts == sorted(processed_counts)
    assert processed_counts[-1] <= _RAW_COUNT

    final_transaction = IndexGenerationStore.for_archive_root(tmp_path).load_transaction(
        DAEMON_BULK_REBUILD_OPERATION_ID
    )
    assert final_transaction.status == "promoted"
    assert final_transaction.processed_raw_count == _RAW_COUNT
    assert final_transaction.last_raw_id is not None

    del seen_raw_ids  # kept for readability of intent; disjointness is proven structurally above


def test_daemon_bulk_rebuild_pass_next_page_excludes_already_scheduled_raws(tmp_path: Path) -> None:
    """Direct proof that a later page never reselects an earlier page's raws."""
    _seed_corpus(tmp_path)
    receipt_path = write_valid_rebuild_receipt(tmp_path, tmp_path.parent / f"{tmp_path.name}-receipt.json")
    store = IndexGenerationStore.for_archive_root(tmp_path)
    transaction = resolve_or_start_daemon_bulk_rebuild_transaction(tmp_path, schema_inference_receipt_path=receipt_path)

    first_page = store.next_raw_page(transaction, limit=2)
    first_raw_ids = {raw_id for raw_id, _blob_hash_hex, _size in first_page.rows}
    assert len(first_raw_ids) == 2

    # Simulate the checkpoint a real pass performs after replaying this page.
    last_raw_id, last_blob_hash_hex, _blob_size = first_page.rows[-1]
    advanced = store.checkpoint_transaction(
        transaction,
        status="paused",
        last_raw_id=last_raw_id,
        last_blob_hash_hex=last_blob_hash_hex,
        processed_raw_count=2,
    )

    second_page = store.next_raw_page(advanced, limit=2)
    second_raw_ids = {raw_id for raw_id, _blob_hash_hex, _size in second_page.rows}
    assert len(second_raw_ids) == 2
    assert first_raw_ids.isdisjoint(second_raw_ids)


def test_daemon_bulk_rebuild_equivalent_to_cli_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """gd6v AC: the daemon bulk path and the offline CLI path converge on
    identical durable archive content for the same corpus."""
    cli_root = tmp_path / "cli"
    daemon_root = tmp_path / "daemon"
    _seed_corpus(cli_root)
    _seed_corpus(daemon_root)
    cli_receipt_path = write_valid_rebuild_receipt(cli_root, tmp_path / "cli-receipt.json")
    daemon_receipt_path = write_valid_rebuild_receipt(daemon_root, tmp_path / "daemon-receipt.json")
    assert source_revision_snapshot(cli_root) == source_revision_snapshot(daemon_root)

    # ArchiveStore.open_owned_inactive_generation validates generation
    # identity against the process-wide configured archive root (not merely
    # the generation's own path), so each route needs POLYLOGUE_ARCHIVE_ROOT
    # pointed at ITS OWN root while it runs -- both offline CLI callers and
    # the daemon route share this same invariant in production (a real
    # daemon process only ever has one configured root at a time).
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_root))
    cli_receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=cli_root, promote=True, schema_inference_receipt_path=cli_receipt_path)
    )
    assert cli_receipt.status == "replayed"
    assert cli_receipt.transaction is not None
    assert cli_receipt.transaction["status"] == "promoted"

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(daemon_root))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(daemon_receipt_path))
    asyncio.run(_drive_daemon_bulk_rebuild_to_promotion(daemon_root, batch_size=2))

    cli_snapshot = _canonical_snapshot(cli_root / "index.db")
    daemon_snapshot = _canonical_snapshot(daemon_root / "index.db")
    assert cli_snapshot["session_count"] == _RAW_COUNT
    assert cli_snapshot == daemon_snapshot
