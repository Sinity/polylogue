"""polylogue-6mvg (remaining slice): durable phase-timing telemetry for the
rebuild path.

Prior work already lands most of this bead's original scope: planner-stats
refresh, byte-skip, batched commits, pool floor, and dedup shipped via
sibling beads, and polylogue-o56w already threads terminal-stage timings
(``terminal.session_insights``/``terminal.bulk_build.*``/``terminal.
reindex_acceptance``/``terminal.readiness``/``terminal.promote``) plus
``parse_s``/``apply_s`` onto every rebuild receipt. The one piece the bead's
own notes call out as still unowned is a SELECTION-phase timing: the live
full rebuild that motivated this bead spent ~86s of one CPU core choosing
WHICH raws to replay before the inactive generation held a single session,
and nothing durable recorded where that time went.

``RebuildPassCost.selection_s`` (and the matching ``"selection_s"`` key on
the terminal/materialized receipt's ``timings_s``) closes that gap by
extending the EXISTING receipt vocabulary rather than inventing a parallel
one -- every ``RebuildIndexReceipt.timings_s`` shape (deferred, paused,
replayed) now carries the same key for this phase.

Production dependency exercised: ``polylogue.maintenance.rebuild_index.
rebuild_index_from_source_sync`` -- the real offline-rebuild orchestrator.
Anti-vacuity: the mutation that makes these tests fail is removing the
``selection_elapsed_s`` measurement (or the ``selection_s=selection_elapsed_s``
threading into ``RebuildPassCost``/``terminal_timings_s``) added around the
``next_raw_page``/``select_rebuild_raw_ids`` calls in
``_rebuild_index_from_source_owned`` -- the key would then be absent from
every receipt's ``timings_s``, not merely zero.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


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
    initialize_active_archive_root(root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            payload = _codex_session(f"sess-{index}", (("user", f"hello {index}"), ("assistant", f"hi {index}")))
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=payload,
                    source_path=f"phase-timing-test/{index}.jsonl",
                    acquired_at_ms=index + 1,
                )
            )
    return raw_ids


def test_replayed_receipt_carries_selection_phase_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-shot (non-resumable) rebuild's final receipt records how long
    raw-id selection took, distinct from replay/parse/apply/terminal costs."""
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_distinct_codex_sessions(root, 2)

    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

    assert receipt.status == "replayed"
    assert "selection_s" in receipt.timings_s
    assert receipt.timings_s["selection_s"] >= 0.0

    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_deferred_pass_cost_carries_selection_phase_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumable pass that defers on its pass deadline still records the
    selection-phase cost -- the same key as the final "replayed" receipt,
    not a different shape for the deferred path."""
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_distinct_codex_sessions(root, 2)

    # A sub-millisecond deadline truncates to pass_deadline_ms=0, so the
    # first between-cohorts check always trips deterministically (see
    # test_rebuild_index_deadline.py for the same technique in detail).
    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, raw_batch_size=10, pass_deadline_seconds=0.0001)
    )

    assert receipt.status == "deferred"
    assert "selection_s" in receipt.timings_s
    assert receipt.timings_s["selection_s"] >= 0.0
