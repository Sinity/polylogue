"""Regression benchmark for polylogue-odm1: whale-aware census spill.

Live receipts (2026-07-21 v42 walk, resume26): on whale-bearing rebuild
pages ``spill_load`` dominates -- 598.2s of a 1440.2s page (41%), plus
236.6s/523.1s, 228.1s/607.5s, 139.6s/454.6s across other pages. The census
spill (``polylogue.sources.revision_backfill._ParsedSessionSpill``) pickles
decoded ``ParsedSession`` trees to a scratch sqlite file during census and
reloads them one authority cohort at a time during replay -- for a session
too large to fit the spill's own small in-RAM hot cache, every reload paid a
full ``pickle.loads`` (or, once the sqlite spill's own payload budget was
also exhausted, a full re-parse from raw bytes).

The fix (whale-aware residency): a parsed tree too large for the hot cache's
own budget but within a wider "whale ceiling" (see
``_ParsedSessionSpill._WHALE_CACHE_MAX_TREE_BYTES``) is held resident as a
plain Python object, bypassing the sqlite spill's pickle round trip
entirely.

This benchmark proves the win using the ``_ParsedSessionSpill`` class
itself, in its default configuration, on a corpus that reproduces the shape
(one oversized raw sharing a page with many ordinary ones) at a scale that
keeps CI wall time bounded. Reproduce the exact same threshold-crossing
behavior at any payload size by shrinking ``_DECODED_CACHE_MIN_TREE_BYTES``/
``_DECODED_CACHE_MAX_TREE_BYTES`` via monkeypatch, exactly as done here --
the class computes its budgets from ``effective_physical_memory_bytes()``,
so leaving them at production defaults would require a fixture sized to a
real host's RAM/16 (>= 256 MiB) to ever classify as a "whale" in the first
place.

"Baseline" (pre-lever) behavior is reproduced by monkeypatching
``_ParsedSessionSpill._retain_whale`` to always return ``False`` -- this
exercises the IDENTICAL unmodified code for every other path (the sqlite
spill write/read, the small-session hot cache) and only neutralizes the new
whale-residency branch, which is a more faithful "before" than reconstructing
a separate implementation.

Run with::

    pytest tests/benchmarks/test_whale_census_spill_bench.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import _parse_retained_raw, _ParsedSessionSpill
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.revision_backfill_benchmark import WHALE_BEARING_SHAPE, build_whale_bearing_corpus

# Shrink the hot-cache budget constants so the benchmark's whale (a few MB,
# not a few hundred MB) reliably fails to fit the hot cache and is
# classified as a whale -- see module docstring. The whale ceiling itself is
# left at its production formula (still generously above this fixture's
# whale, exactly as it would be on any real host processing a genuine
# multi-hundred-MB rollout against a multi-GiB whale ceiling).
_SHRUNK_DECODED_MIN_TREE_BYTES = 64 * 1024
_SHRUNK_DECODED_MAX_TREE_BYTES = 64 * 1024

# Generous spill-cache budget: large enough that every raw (small population
# + whale) fits the sqlite spill's own payload-byte cap in the baseline run,
# so the measured delta is purely "resident object vs pickle round trip",
# not confounded by the sqlite budget also rejecting the whale (a different,
# already-existing failure mode this benchmark does not target).
_SPILL_CACHE_BYTES = 64 * 1024 * 1024

# How many times replay-shaped code re-reads each raw via for_raw() in one
# pass -- mirrors polylogue-odm1's evidence that the same raw_id can be
# looked up more than once across the census/replay boundary.
_REREAD_PASSES = 2


def _run_spill_load_pass(
    archive_root: Path,
    small_raw_ids: list[str],
    whale_raw_id: str,
    *,
    disable_whale_lever: bool,
) -> tuple[float, float, dict[str, list[ParsedSession]]]:
    """Census (add) then replay (for_raw) every raw once per pass; return
    ``(whale_reload_seconds, small_reload_seconds, sessions_by_raw_id)``.

    ``sessions_by_raw_id`` captures the FIRST pass's reloaded sessions
    (keyed raw_id -> list[ParsedSession]) so the caller can assert output
    equivalence between the baseline and lever-enabled runs.

    Uses its own scoped ``pytest.MonkeyPatch`` context (not the caller's
    fixture) so the ``_retain_whale`` neutralization used to reproduce
    pre-lever behavior is guaranteed undone before the next call in the same
    test -- the shared test-scoped ``monkeypatch`` fixture only undoes
    patches at test teardown, which would otherwise leak the baseline's
    neutralized ``_retain_whale`` into the lever-enabled run.
    """
    whale_patch = pytest.MonkeyPatch()
    try:
        if disable_whale_lever:
            whale_patch.setattr(_ParsedSessionSpill, "_retain_whale", lambda self, *a, **k: False)

        with (
            ArchiveStore.open_existing(archive_root, read_only=False) as archive,
            _ParsedSessionSpill(archive_root, max_cached_payload_bytes=_SPILL_CACHE_BYTES) as spill,
        ):
            return _census_then_reload(archive, spill, small_raw_ids, whale_raw_id)
    finally:
        whale_patch.undo()


def _census_then_reload(
    archive: ArchiveStore,
    spill: _ParsedSessionSpill,
    small_raw_ids: list[str],
    whale_raw_id: str,
) -> tuple[float, float, dict[str, list[ParsedSession]]]:
    """Census (add) every raw once, then replay (for_raw) every raw once per
    pass; return ``(whale_reload_seconds, small_reload_seconds, sessions_by_raw_id)``.
    """
    for raw_id in [*small_raw_ids, whale_raw_id]:
        sessions, payload_bytes, _kind = _parse_retained_raw(archive, raw_id)
        spill.add(raw_id, sessions, payload_bytes=payload_bytes)

    sessions_by_raw_id: dict[str, list[ParsedSession]] = {}
    whale_seconds = 0.0
    small_seconds = 0.0
    for _pass in range(_REREAD_PASSES):
        for raw_id in small_raw_ids:
            started = time.perf_counter()
            sessions, _payload_bytes = spill.for_raw(archive, raw_id)
            small_seconds += time.perf_counter() - started
            sessions_by_raw_id.setdefault(raw_id, sessions)
        started = time.perf_counter()
        whale_sessions, _payload_bytes = spill.for_raw(archive, whale_raw_id)
        whale_seconds += time.perf_counter() - started
        sessions_by_raw_id.setdefault(whale_raw_id, whale_sessions)

    return whale_seconds, small_seconds, sessions_by_raw_id


@pytest.mark.benchmark
def test_whale_bypasses_pickle_round_trip_on_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whale-aware residency must cut ``for_raw()`` reload cost for an
    oversized session versus the pre-lever pickle-to-sqlite round trip, and
    must not change what content is returned (output equivalence)."""
    monkeypatch.setattr(_ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_DECODED_MIN_TREE_BYTES)
    monkeypatch.setattr(_ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_DECODED_MAX_TREE_BYTES)

    baseline_root = tmp_path / "baseline"
    small_raw_ids, whale_raw_id = build_whale_bearing_corpus(
        baseline_root,
        small_raw_count=WHALE_BEARING_SHAPE["small_raw_count"],
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=WHALE_BEARING_SHAPE["whale_payload_bytes"],
    )
    baseline_whale_s, baseline_small_s, baseline_sessions = _run_spill_load_pass(
        baseline_root, small_raw_ids, whale_raw_id, disable_whale_lever=True
    )

    lever_root = tmp_path / "lever"
    small_raw_ids_2, whale_raw_id_2 = build_whale_bearing_corpus(
        lever_root,
        small_raw_count=WHALE_BEARING_SHAPE["small_raw_count"],
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=WHALE_BEARING_SHAPE["whale_payload_bytes"],
    )
    assert small_raw_ids_2 == small_raw_ids and whale_raw_id_2 == whale_raw_id, (
        "corpus builder must be deterministic across the two roots for a fair before/after comparison"
    )
    lever_whale_s, lever_small_s, lever_sessions = _run_spill_load_pass(
        lever_root, small_raw_ids, whale_raw_id, disable_whale_lever=False
    )

    speedup = baseline_whale_s / max(lever_whale_s, 1e-9)
    print(
        f"\nwhale spill_load reload ({_REREAD_PASSES} passes, "
        f"whale_payload_bytes={WHALE_BEARING_SHAPE['whale_payload_bytes']}, "
        f"small_raw_count={WHALE_BEARING_SHAPE['small_raw_count']}): "
        f"baseline(pickle round trip)={baseline_whale_s:.4f}s, lever(resident)={lever_whale_s:.4f}s, "
        f"speedup={speedup:.1f}x; small-population reload baseline={baseline_small_s:.4f}s "
        f"lever={lever_small_s:.4f}s (unaffected -- both paths spill small sessions identically)"
    )

    assert lever_whale_s * 3 < baseline_whale_s, (
        f"Expected the whale-resident reload to be at least 3x faster than the pickle round trip; "
        f"got baseline={baseline_whale_s:.4f}s lever={lever_whale_s:.4f}s (only {speedup:.1f}x)"
    )

    # Output equivalence: the reloaded whale session's content must be
    # byte-identical between the two paths (proves the resident-object
    # bypass is not silently returning stale or partial content).
    baseline_whale_session = baseline_sessions[whale_raw_id][0]
    lever_whale_session = lever_sessions[whale_raw_id][0]
    assert lever_whale_session.provider_session_id == baseline_whale_session.provider_session_id
    assert len(lever_whale_session.messages) == len(baseline_whale_session.messages)
    assert [m.text for m in lever_whale_session.messages] == [m.text for m in baseline_whale_session.messages]
    for small_raw_id in small_raw_ids:
        baseline_small_session = baseline_sessions[small_raw_id][0]
        lever_small_session = lever_sessions[small_raw_id][0]
        assert lever_small_session.provider_session_id == baseline_small_session.provider_session_id
        assert [m.text for m in lever_small_session.messages] == [m.text for m in baseline_small_session.messages]


@pytest.mark.benchmark
def test_whale_exceeding_whale_ceiling_falls_back_to_spill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tree too large for EITHER tier (hot cache or whale ceiling) must
    still be spilled to sqlite exactly as the pre-lever code did --
    correctness-over-speed fallback required by polylogue-odm1's AC."""
    monkeypatch.setattr(_ParsedSessionSpill, "_DECODED_CACHE_MIN_TREE_BYTES", _SHRUNK_DECODED_MIN_TREE_BYTES)
    monkeypatch.setattr(_ParsedSessionSpill, "_DECODED_CACHE_MAX_TREE_BYTES", _SHRUNK_DECODED_MAX_TREE_BYTES)
    # Shrink the whale ceiling itself below the fixture's whale tree size so
    # the fallback path is exercised deterministically regardless of host RAM.
    monkeypatch.setattr(_ParsedSessionSpill, "_WHALE_CACHE_MAX_TREE_BYTES", _SHRUNK_DECODED_MAX_TREE_BYTES)

    archive_root = tmp_path / "archive"
    small_raw_ids, whale_raw_id = build_whale_bearing_corpus(
        archive_root,
        small_raw_count=5,
        small_avg_payload_bytes=WHALE_BEARING_SHAPE["small_avg_payload_bytes"],
        whale_payload_bytes=WHALE_BEARING_SHAPE["whale_payload_bytes"],
    )

    with (
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
        _ParsedSessionSpill(archive_root, max_cached_payload_bytes=_SPILL_CACHE_BYTES) as spill,
    ):
        sessions, payload_bytes, _kind = _parse_retained_raw(archive, whale_raw_id)
        spill.add(whale_raw_id, sessions, payload_bytes=payload_bytes)

        assert whale_raw_id not in spill._whales, "whale ceiling was shrunk below the fixture; must not be resident"
        row = spill.conn.execute("SELECT COUNT(*) FROM parsed_sessions WHERE raw_id = ?", (whale_raw_id,)).fetchone()
        assert row is not None and row[0] > 0, (
            "oversized tree exceeding the whale ceiling must fall back to the sqlite spill"
        )

        reloaded_sessions, reloaded_payload_bytes = spill.for_raw(archive, whale_raw_id)
        assert reloaded_payload_bytes == payload_bytes
        assert [m.text for m in reloaded_sessions[0].messages] == [m.text for m in sessions[0].messages]
