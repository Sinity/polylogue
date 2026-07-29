"""polylogue-623q: the rebuild's parse-vs-apply split, end to end.

``backfill_historical_revision_evidence`` (``polylogue.sources.
revision_backfill``) has always computed a rich per-stage timing breakdown
-- ``census``/``spill_load`` (decode, read-only, parallel) and
``revision_replay.*``/``membership_replay.*`` (writer-side index/FTS/
projection writes, serialized through the single SQLite writer) -- and
logged it as ``"backfill stage timings: ..."``. Before this change it was
discarded at the function's return boundary: ``RevisionBackfillResult`` had
no field for it, so ``polylogue.maintenance.replay.rebuild_index_from_source``
(the SAME function ``polylogue.maintenance.rebuild_index.
rebuild_index_from_source_sync`` calls -- the real offline-rebuild and
daemon-bulk-rebuild engine, not a reimplementation) had nothing to thread
into ``RebuildPassCost``, whose ``replay_s`` field covered parse and apply
as one opaque number.

This module proves the split survives the full real route:
``rebuild_index_from_source_sync`` -> ``rebuild_index_from_source`` (replay
module) -> ``backfill_historical_revision_evidence`` -> a real
``ArchiveStore`` writing real index rows. No mocks. Reverting either the
``RevisionBackfillResult.stage_timings_s`` field or ``replay.py``'s
``parse_s``/``apply_s`` computation makes
``test_rebuild_records_parse_apply_split_summing_to_stage_total`` fail with
a ``KeyError``/``None`` lookup, not merely a wrong number -- the two
components are not independently fabricatable from this test's assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources.revision_backfill import split_parse_and_apply_seconds
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus


def test_split_parse_and_apply_seconds_sums_to_total() -> None:
    """Pure rollup: parse is census+spill_load, apply is everything else."""
    stage_timings_s = {
        "census": 2.0,
        "spill_load": 0.5,
        "census_receipt": 0.1,
        "revision_replay.index_parsed_write": 1.5,
        "membership_replay.index_parsed_write": 0.4,
        "total": 4.5,
    }
    parse_s, apply_s = split_parse_and_apply_seconds(stage_timings_s)
    assert parse_s == 2.5  # census + spill_load only
    assert apply_s == 2.0  # total - parse_s (census_receipt + writer stages)
    assert parse_s + apply_s == stage_timings_s["total"]


def test_split_parse_and_apply_seconds_floors_at_zero_when_total_absent() -> None:
    """An empty/early-returned timings dict (e.g. zero raws replayed) must
    not report a negative or fabricated apply cost."""
    parse_s, apply_s = split_parse_and_apply_seconds({})
    assert (parse_s, apply_s) == (0.0, 0.0)


def test_rebuild_records_parse_apply_split_summing_to_stage_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real production route: rebuild_index_from_source_sync must surface a
    parse-vs-apply split that sums to the SAME 'total' stage timing that was
    already being logged -- not a second, independently-computed number.
    """
    root = tmp_path / "archive"
    build_independent_raw_corpus(root, raw_count=8, avg_payload_bytes=20_000)
    # ArchiveStore.open_owned_inactive_generation resolves the generation
    # store via the CONFIGURED archive root (polylogue.paths.archive_root),
    # not the archive_root argument threaded through the call -- it must
    # point at this test's own root (see
    # tests/unit/storage/test_rebuild_paging_content_order.py for the same
    # requirement on the same code path).
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=True,
            raw_batch_size=500,  # single page: whole corpus fits in one pass
        )
    )

    assert receipt.status == "replayed"
    replay = receipt.replay
    assert "parse_s" in replay
    assert "apply_s" in replay
    stage_timings_s = replay["stage_timings_s"]
    assert isinstance(stage_timings_s, dict)
    # Real production stage names must be present -- proof this came from the
    # actual backfill, not a stand-in.
    assert "census" in stage_timings_s
    assert "total" in stage_timings_s

    parse_s = replay["parse_s"]
    apply_s = replay["apply_s"]
    assert isinstance(parse_s, float)
    assert isinstance(apply_s, float)
    assert parse_s >= 0.0
    assert apply_s >= 0.0
    # Each of the three numbers is independently rounded to 6 decimals in
    # the return dict, so allow a tiny tolerance rather than exact equality.
    assert parse_s + apply_s == pytest.approx(stage_timings_s["total"], abs=1e-5)
    # This corpus is genuinely replayed (not empty), so real writer work
    # happened: apply_s must be strictly positive, not merely non-negative.
    assert apply_s > 0.0
