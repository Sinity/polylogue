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

import polylogue.maintenance.rebuild_index as rebuild_index
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


def test_real_receipt_accounts_for_all_rebuild_phases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production receipt carries one stable rollup for every rebuild phase.

    This is deliberately a real source-to-index rebuild. The assertions bind
    the rollups to the existing replay stage ledger and terminal-stage receipt,
    so a test-only timing fixture cannot make the contract pass.
    """
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_distinct_codex_sessions(root, 3)

    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

    timings = receipt.timings_s
    assert {
        "selection_s",
        "cohort_s",
        "parse_s",
        "apply_s",
        "insight_s",
        "terminal_s",
    }.issubset(timings)
    assert all(timings[key] >= 0.0 for key in timings)

    stage_timings = receipt.replay["stage_timings_s"]
    assert isinstance(stage_timings, dict)
    cohort_stage_seconds = sum(
        float(value)
        for key, value in stage_timings.items()
        if isinstance(key, str)
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and (
            key.startswith("replay.classify_cohort")
            or key.startswith("replay.adoptable_check")
            or key.startswith("membership.")
        )
    )
    assert cohort_stage_seconds > 0.0
    assert timings["cohort_s"] == pytest.approx(cohort_stage_seconds, abs=0.005)
    parse_seconds = receipt.replay["parse_s"]
    apply_seconds = receipt.replay["apply_s"]
    assert isinstance(parse_seconds, int | float) and not isinstance(parse_seconds, bool)
    assert isinstance(apply_seconds, int | float) and not isinstance(apply_seconds, bool)
    assert timings["parse_s"] == pytest.approx(float(parse_seconds), abs=0.005)
    assert timings["apply_s"] == pytest.approx(float(apply_seconds), abs=0.005)
    assert timings["insight_s"] == pytest.approx(timings["terminal.session_insights"], abs=0.005)
    assert timings["terminal_s"] == pytest.approx(
        sum(value for key, value in timings.items() if key.startswith("terminal.") and key != "terminal_s"),
        abs=0.005,
    )


def test_phase_rollups_are_bound_to_production_stage_timing_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt's phase rollups cannot pass from selection timing alone."""
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_distinct_codex_sessions(root, 2)

    original = rebuild_index._repopulate_bulk_build_derived_state

    def instrumented_repopulate(index_path: Path) -> dict[str, float]:
        timings = original(index_path)
        timings["mutation_probe"] = 0.125
        return timings

    monkeypatch.setattr(rebuild_index, "_repopulate_bulk_build_derived_state", instrumented_repopulate)
    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

    assert receipt.timings_s["terminal.bulk_build.mutation_probe"] == pytest.approx(0.125, abs=0.001)
    assert receipt.timings_s["terminal_s"] >= receipt.timings_s["terminal.bulk_build.mutation_probe"]


def test_partial_rebuild_cannot_complete_when_candidate_omits_source_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rebuild_index_from_source_sync`` must reject the real candidate
    generation when its selected raw subset omits a source-backed document.
    Partial requests are independently forbidden from promotion, so this
    uses their allowed ``promote=False`` route. Removing the candidate-index
    corpus report lets that real route report a completed rebuild instead.
    """
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    raw_ids = _seed_distinct_codex_sessions(root, 2)

    with pytest.raises(RuntimeError, match="reindex acceptance gate failed.*corpus-absences"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, raw_ids=(raw_ids[0],), promote=False))

    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("violation", "expected_check"),
    (
        ("attachment", "corpus-attachment-fidelity"),
        ("revision", "corpus-revision-fidelity"),
    ),
)
def test_rebuild_corpus_gate_blocks_mutated_inactive_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
    expected_check: str,
) -> None:
    """The real rebuild acceptance stage rejects each corrupted candidate.

    The hook wraps the production bulk-derived-state step, after replay has
    built the actual inactive index and immediately before the production
    acceptance stage. It never replaces ``verify_archive``: attachment
    corruption is written to the inactive SQLite index, while revision
    corruption adds durable source evidence the already-built candidate lacks.
    Removing the corpus acceptance report lets this full rebuild complete.
    """
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_distinct_codex_sessions(root, 1)
    original_repopulate = rebuild_index._repopulate_bulk_build_derived_state

    def corrupt_candidate_before_acceptance(index_path: Path) -> dict[str, float]:
        timings_s = original_repopulate(index_path)
        with sqlite3.connect(index_path) as candidate:
            session_row = candidate.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()
            assert session_row is not None
            session_id = str(session_row[0])
            if violation == "attachment":
                message_row = candidate.execute(
                    "SELECT message_id FROM messages WHERE session_id = ? LIMIT 1", (session_id,)
                ).fetchone()
                assert message_row is not None
                candidate.execute(
                    "INSERT INTO attachments(attachment_id, acquisition_status) VALUES ('rebuild-unfetched', 'unfetched')"
                )
                candidate.execute(
                    "INSERT INTO attachment_refs(attachment_id, session_id, message_id, position, upload_origin) "
                    "VALUES ('rebuild-unfetched', ?, ?, 99, 'drive')",
                    (session_id, message_row[0]),
                )
        if violation == "revision":
            origin, native_id = session_id.split(":", 1)
            with sqlite3.connect(root / "source.db") as source:
                source.execute(
                    """
                    INSERT INTO raw_sessions(
                        raw_id, origin, native_id, source_path, blob_hash, blob_size,
                        acquired_at_ms, logical_source_key
                    ) VALUES ('rebuild-best-revision', ?, ?, '/fixture/rebuild-best-revision', ?, 10, 100, ?)
                    """,
                    (origin, native_id, b"r" * 32, f"fixture:{native_id}"),
                )
                source.execute(
                    """
                    INSERT INTO raw_session_memberships(
                        raw_id, logical_source_key, provider_session_id, source_revision,
                        normalized_content_hash, message_count
                    ) VALUES ('rebuild-best-revision', ?, ?, 'rebuild-best', ?, 100000)
                    """,
                    (f"fixture:{native_id}", native_id, b"r" * 32),
                )
        return timings_s

    monkeypatch.setattr(rebuild_index, "_repopulate_bulk_build_derived_state", corrupt_candidate_before_acceptance)

    with pytest.raises(RuntimeError, match=f"reindex acceptance gate failed.*{expected_check}"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=False))

    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


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
