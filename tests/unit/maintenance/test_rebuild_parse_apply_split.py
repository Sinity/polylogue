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

import time
from pathlib import Path

import pytest

from devtools.production_reachability import ProductionSeamSpec, check_production_seam
from polylogue.config import Config
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources import revision_backfill
from polylogue.sources.census_parse_stage import CensusParseStage
from polylogue.sources.revision_backfill import split_parse_and_apply_seconds
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus

REINDEX_PRODUCTION_SEAMS = (
    ProductionSeamSpec(
        test_path="tests/unit/maintenance/test_rebuild_parse_apply_split.py",
        test_function="test_rebuild_records_parse_apply_split_summing_to_stage_total",
        production_entrypoint="polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync",
        tested_symbols=("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync",),
        required_symbols=(
            "polylogue.sources.revision_backfill.backfill_historical_revision_evidence",
            "polylogue.storage.repair.repair_session_insights",
        ),
    ),
    ProductionSeamSpec(
        test_path="tests/unit/maintenance/test_rebuild_parse_apply_split.py",
        test_function="test_rebuild_index_from_source_sync_warms_prefetch_cache_when_caller_omits_one",
        production_entrypoint="polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync",
        tested_symbols=("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync",),
        required_symbols=(
            "polylogue.sources.revision_backfill.backfill_historical_revision_evidence",
            "polylogue.storage.repair.repair_session_insights",
        ),
    ),
    ProductionSeamSpec(
        test_path="tests/unit/maintenance/test_rebuild_parse_apply_split.py",
        test_function="test_rebuild_index_from_source_sync_auto_engages_pipelined_decode",
        production_entrypoint="polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync",
        tested_symbols=("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync",),
        required_symbols=(
            "polylogue.sources.revision_backfill.backfill_historical_revision_evidence",
            "polylogue.storage.repair.repair_session_insights",
        ),
    ),
)


def test_selected_reindex_proof_tests_are_production_reachable() -> None:
    """The selected reindex proofs bind to replay and terminal convergence."""
    root = Path(__file__).resolve().parents[3]
    for spec in REINDEX_PRODUCTION_SEAMS:
        report = check_production_seam(spec, source_root=root)
        assert report.ok, report.to_json()


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


def test_rebuild_index_from_source_sync_warms_prefetch_cache_when_caller_omits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-czq2: the offline CLI route must get the SAME off-writer
    census prefetch seam ``daemon/bulk_rebuild.py`` has always had, not just
    a caller that remembers to construct its own ``CensusParseStage``.

    Anti-vacuity: this drives the real production entry point
    (``rebuild_index_from_source_sync``, the exact function the CLI's
    ``rebuild-index`` command and ``polylogued``'s own HTTP maintenance route
    call) with a request that leaves ``RebuildIndexRequest.prefetch_cache``
    at its default ``None`` -- exactly what those two callers do today. Before
    ``_rebuild_index_from_source_owned`` grew its internal
    ``_warm_offline_prefetch_cache`` call, ``CensusParseStage.warm_raw_ids``
    was reached by exactly one caller in the whole codebase
    (``daemon/bulk_rebuild.py``): deleting the internal warm call this test
    exercises makes ``warm_raw_ids`` unreached again from this route and this
    assertion fails, proving the spy is wired to production code, not a
    self-authorized double.
    """
    root = tmp_path / "archive"
    raw_ids = build_independent_raw_corpus(root, raw_count=6, avg_payload_bytes=20_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    warmed_raw_id_batches: list[tuple[str, ...]] = []
    real_warm_raw_ids = CensusParseStage.warm_raw_ids

    def _spy_warm_raw_ids(self: CensusParseStage, config: Config, *, raw_ids: list[str], max_payload_bytes: int) -> int:
        warmed_raw_id_batches.append(tuple(raw_ids))
        return real_warm_raw_ids(self, config, raw_ids=raw_ids, max_payload_bytes=max_payload_bytes)

    monkeypatch.setattr(CensusParseStage, "warm_raw_ids", _spy_warm_raw_ids)

    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=True,
            raw_batch_size=500,  # single page: whole corpus fits in one pass
        )
    )

    assert receipt.status == "replayed"
    assert warmed_raw_id_batches, "offline rebuild_index_from_source_sync never warmed a prefetch cache"
    # Every raw the census phase went on to process was offered to the warmer
    # first -- the exact set this pass selected, not a subset/superset.
    assert set(warmed_raw_id_batches[0]) == set(raw_ids)


def _give_replay_spill_prefetcher_a_head_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic race pin, mirroring ``test_revision_backfill.py``'s
    identically-named helper: production makes no ordering promise between
    the background decode worker and the writer, so a tiny corpus can let
    the writer finish before the worker buffers anything, making
    ``spill_prefetch.consumed`` a coin flip. Give the worker a bounded head
    start before the writer's own replay loop begins."""
    original_start_phase = revision_backfill._ReplaySpillPrefetcher.start_phase

    def start_phase_with_head_start(
        self: revision_backfill._ReplaySpillPrefetcher,
        ordered_keys: object,
        extra_members: object,
    ) -> None:
        original_start_phase(self, ordered_keys, extra_members)  # type: ignore[arg-type]
        worker = self._thread
        for _ in range(1000):  # bounded ~10s; normally exits in milliseconds
            with self._lock:
                if len(self._buffer) >= 2:
                    break
            if worker is None or not worker.is_alive():
                break
            time.sleep(0.01)

    monkeypatch.setattr(revision_backfill._ReplaySpillPrefetcher, "start_phase", start_phase_with_head_start)


def test_rebuild_index_from_source_sync_auto_engages_pipelined_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-2cuv: BOTH production rebuild routes -- the offline CLI
    (``polylogue ops maintenance rebuild-index``) and the daemon bulk-rebuild
    loop (``daemon/bulk_rebuild.py::run_daemon_bulk_rebuild_pass``) -- drive
    this exact ``rebuild_index_from_source_sync`` entry point (the daemon via
    its own write coordinator, unmodified). Both therefore inherit the SAME
    ``pipeline_decode`` auto-engagement inside ``backfill_historical_revision_
    evidence`` (Lever A / PR #3478's ``_ReplaySpillPrefetcher``) with zero
    per-route wiring -- there is no separate knob either route could forget
    to set.

    Anti-vacuity: this drives the real production entry point with a corpus
    sized at/above ``_PIPELINE_DECODE_MIN_COHORTS`` (independent raws, so
    every raw is its own logical cohort) and shrinks the spill's RAM cache
    tiers to 1 byte so every replay ``for_raw`` misses RAM and must go
    through the prefetcher-or-inline decode fork. If pipeline_decode were
    hardcoded ``False`` somewhere between ``rebuild_index_from_source_sync``
    and ``backfill_historical_revision_evidence`` (e.g. a parameter dropped
    while threading a future kwarg), ``spill_prefetch.consumed`` would never
    appear in the stage timings and this assertion would fail -- proving the
    auto-engagement path is actually reached from the production route, not
    only from the lower-level unit tests that call
    ``backfill_historical_revision_evidence`` directly.
    """
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", 1)
    monkeypatch.setattr(revision_backfill._ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", 1)
    _give_replay_spill_prefetcher_a_head_start(monkeypatch)

    root = tmp_path / "archive"
    cohort_count = revision_backfill._PIPELINE_DECODE_MIN_COHORTS + 4
    raw_ids = build_independent_raw_corpus(root, raw_count=cohort_count, avg_payload_bytes=20_000)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=True,
            raw_batch_size=500,  # single page: whole corpus fits in one pass
        )
    )

    assert receipt.status == "replayed"
    stage_timings_s = receipt.replay["stage_timings_s"]
    assert isinstance(stage_timings_s, dict)
    assert stage_timings_s.get("spill_prefetch.consumed", 0.0) > 0, (
        "rebuild_index_from_source_sync did not auto-engage the background "
        "replay-spill prefetcher (Lever A) for a cohort count above "
        "_PIPELINE_DECODE_MIN_COHORTS -- census+spill_load decode is no "
        "longer proven to overlap the writer's apply work on this route"
    )
    assert receipt.selected_raw_count == len(raw_ids)
