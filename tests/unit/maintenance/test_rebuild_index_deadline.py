"""polylogue-uhgm: a rebuild pass's deadline must be enforced INSIDE replay
work, not only after the whole requested page has replayed to completion.

Live recovery evidence: operation 3f8fa7b0 configured ``pass_deadline_ms=
300000`` (5 minutes), yet 100-row pages ran for roughly 8-9 minutes because
``rebuild_index_from_source`` (``maintenance/rebuild_index.py``) previously
checked elapsed time only after ``replay_source(...)`` -- the WHOLE page's
replay -- returned.

Production dependencies exercised here:

* ``polylogue.sources.revision_backfill.backfill_historical_revision_evidence``
  -- the real REPLAY-phase byte-cohort/membership-cohort loops now call an
  injected ``deadline_check`` between cohorts.
* ``polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync`` --
  the real offline-rebuild orchestrator that builds the ``deadline_check``
  closure from a resumable transaction's ``pass_deadline_ms`` and catches
  ``RebuildDeadlineExceededError`` to checkpoint a "no forward progress"
  pass instead of letting a page run past its budget.

Anti-vacuity: the mutation that makes ``test_deadline_check_invoked_between_
replay_cohorts_not_only_after_return`` fail is moving (or removing) the
``deadline_check()`` calls out of the byte-cohort/membership-cohort loops in
``backfill_historical_revision_evidence`` -- e.g. back to a single call after
the function's own work loop finishes, which is exactly the pre-fix bug
shape. The mutation that makes ``test_rebuild_index_deadline_stops_mid_page_
and_resumes_without_omission_or_duplication`` fail is either (a) reverting
``rebuild_index.py`` to the old post-hoc-only check (the whole page would
complete before any deadline is observed, so the fake clock's huge elapsed
value would never interrupt anything and the first pass would report
``status="replayed"`` instead of ``"deferred"``), or (b) advancing the
transaction's cursor/processed counters on interrupt (the resumed pass would
then either skip the un-replayed raw or duplicate index rows for an already-
replayed one, and the final session count would not equal 3).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources.revision_backfill import (
    RebuildDeadlineExceededError,
    backfill_historical_revision_evidence,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-16T10:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _seed_distinct_codex_sessions(root: Path, count: int) -> list[str]:
    """Write ``count`` raws that each parse to their own logical cohort."""
    initialize_active_archive_root(root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            payload = _codex_session(f"sess-{index}", (("user", f"hello {index}"), ("assistant", f"hi {index}")))
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=payload,
                    source_path=f"deadline-test/{index}.jsonl",
                    acquired_at_ms=index + 1,
                )
            )
    return raw_ids


def _receipt(root: Path) -> Path:
    """A fresh schema-inference receipt: the rebuild preflight refuses without one."""
    return write_valid_rebuild_receipt(root, root.parent / f"{root.name}-schema-receipt.json")


def test_deadline_check_invoked_between_replay_cohorts_not_only_after_return(tmp_path: Path) -> None:
    """The interruption point is BETWEEN cohorts, proven at the direct
    production-function boundary (no rebuild_index.py orchestration): a
    deadline_check that raises on its second call must stop replay after
    exactly one cohort durably commits, never zero and never all three."""
    root = tmp_path / "archive"
    raw_ids = _seed_distinct_codex_sessions(root, 3)

    deadline_check = Mock(side_effect=[None, RebuildDeadlineExceededError("synthetic deadline")])

    with pytest.raises(RebuildDeadlineExceededError):
        backfill_historical_revision_evidence(
            root,
            selected_raw_ids=raw_ids,
            ingest_workers=1,
            # Force per-cohort commits so the interrupted pass's partial
            # progress is durable and directly observable below, matching
            # the crash-recovery contract an open batch already has.
            replay_commit_batch_size=1,
            deadline_check=deadline_check,
        )

    # Called once before the first cohort (proceeds) and once before the
    # second (raises) -- i.e. mid-replay, not only after the function would
    # have returned. A deadline_check wired only after the whole call
    # returns would never be invoked at all here, since the call itself
    # never returns normally.
    assert deadline_check.call_count == 2

    with sqlite3.connect(root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert session_count == 1, "exactly the first cohort must have durably committed before the interrupt"


def test_rebuild_index_deadline_stops_mid_page_and_resumes_without_omission_or_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the real offline-rebuild orchestrator: a page that
    would normally replay all 3 raws in one ``replay_source`` call stops
    before completing, checkpoints a deferred pass with NO cursor advance,
    and a resumed pass -- with the real clock restored -- completes cleanly
    with the exact final session count: no raw skipped, none duplicated.

    The fake clock returns ``0.0`` for exactly the first ``time.time()``
    call reached (verified by source inspection to be ``pass_started_at_ms``
    -- nothing between entering ``rebuild_index_from_source`` and that line
    reads the wall clock) and a huge value for every call after, so the
    first ``_check_pass_deadline`` call inside the replay cohort loop always
    sees a huge elapsed time and interrupts deterministically before any
    cohort completes -- independent of real machine speed. It is scoped to
    only the first (interrupted) call via ``monkeypatch.context()``, so the
    resumed call below runs on the real clock with the transaction's
    genuinely generous durable 30s budget.
    """
    root = tmp_path / "archive"
    # ``ArchiveStore.__init__``'s owned-inactive-generation branch (taken by
    # the real rebuild's replay call) resolves the generation store off
    # ``configured_archive_root()``, not the ``archive_root`` argument it was
    # constructed with -- that must agree with ``root`` here or generation
    # lookups 404 against the wrong (default XDG) location.
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_distinct_codex_sessions(root, 3)

    call_state = {"first": True}

    def fake_time() -> float:
        if call_state["first"]:
            call_state["first"] = False
            return 0.0
        return 999_999.0

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("polylogue.maintenance.rebuild_index.time.time", fake_time)
        first_pass = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=_receipt(root),
                raw_batch_size=10,
                pass_deadline_seconds=30.0,
            )
        )

    assert first_pass.status == "deferred"
    assert first_pass.replay.get("deferred_reason") == "pass-deadline-mid-replay"
    assert first_pass.transaction is not None
    # No forward progress recorded: the cursor/processed counters must stay
    # exactly where they were before this pass attempted anything, so a
    # resume re-derives from the identical source-order position.
    assert first_pass.transaction["processed_raw_count"] == 0
    assert first_pass.transaction["last_raw_id"] is None
    operation_id = first_pass.transaction["operation_id"]
    assert isinstance(operation_id, str)

    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0, (
            "the active archive's own index.db must be untouched: any cohort work this pass did (if any) "
            "lives only in a not-yet-promoted inactive generation directory"
        )

    # Real clock restored (the monkeypatch context exited above); the
    # transaction's durable 30s budget is ample for this tiny fixture.
    final_pass = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=_receipt(root), operation_id=operation_id)
    )
    assert final_pass.status == "replayed"
    assert final_pass.materialized is True

    with sqlite3.connect(root / "index.db") as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert session_count == 3, "no raw/cohort omitted or duplicated across the interrupted + resumed pass"
