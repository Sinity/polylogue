"""Committed workload profiles with their size tails clipped, for behavior tests.

The committed profiles reproduce real tails: a draw can be a multi-gigabyte
tool result, a quarter-million-record rollout, or a session fanning out to
hundreds of subagents. A session's size is the product of those draws, so
clipping text lengths alone still lets one seed render a gigabyte session.
Tests that materialize a generated corpus to inspect record shapes therefore
declare a bounded profile: every size parameter -- text and list lengths
(``max_bucket``), records per stream (``max_records_bucket``), and subagent
fan-out and nesting (``max_fanout_bucket``) -- keeps only its measured buckets
at or below the bound (log2), renormalized. The generator renders that
profile faithfully; nothing it draws is cut afterwards. Tail behavior itself
is covered by tests that sample a histogram directly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest

from polylogue.schemas.synthetic import workload
from polylogue.schemas.synthetic.workload import Histogram, WorkloadProfile

#: 4 MiB: large enough for lazy-text and sidecar paths, small enough to materialize.
DEFAULT_MAX_BUCKET = 23
#: Up to 2,047 records per stream: long enough for multi-turn tool pairing.
DEFAULT_MAX_RECORDS_BUCKET = 10
#: Up to 63 subagents per session and 63 nested descendants per subagent.
DEFAULT_MAX_FANOUT_BUCKET = 5


def _clip(histogram: Histogram, max_bucket: int) -> Histogram:
    kept = [
        (bucket, weight)
        for bucket, weight in zip(histogram.buckets, histogram.weights, strict=True)
        if bucket <= max_bucket
    ]
    if not kept:
        return Histogram((max_bucket,), (1.0,))
    return Histogram(tuple(bucket for bucket, _ in kept), tuple(weight for _, weight in kept))


def _clip_paths(by_kind: Mapping[str, Mapping[str, Histogram]], max_bucket: int) -> dict[str, dict[str, Histogram]]:
    return {kind: {path: _clip(h, max_bucket) for path, h in paths.items()} for kind, paths in by_kind.items()}


def clipped_profile(
    profile: WorkloadProfile,
    max_bucket: int = DEFAULT_MAX_BUCKET,
    *,
    max_records_bucket: int = DEFAULT_MAX_RECORDS_BUCKET,
    max_fanout_bucket: int = DEFAULT_MAX_FANOUT_BUCKET,
) -> WorkloadProfile:
    streams = {
        name: dataclasses.replace(
            stream,
            records=_clip(stream.records, max_records_bucket),
            lengths={kind: _clip(h, max_bucket) for kind, h in stream.lengths.items()},
        )
        for name, stream in profile.streams.items()
    }
    return dataclasses.replace(
        profile,
        streams=streams,
        subagents_per_session=_clip(profile.subagents_per_session, max_fanout_bucket),
        nested_descendants=_clip(profile.nested_descendants, max_fanout_bucket),
        nested_spawns=_clip(profile.nested_spawns, max_fanout_bucket),
        template_strings=_clip_paths(profile.template_strings, max_bucket),
        template_lists=_clip_paths(profile.template_lists, max_bucket),
    )


def clip_committed_profiles(monkeypatch: pytest.MonkeyPatch, *, max_bucket: int = DEFAULT_MAX_BUCKET) -> None:
    """Make every ``load_workload_profile`` in the generator return a bounded profile."""
    original = workload.load_workload_profile
    cache: dict[str, WorkloadProfile] = {}

    def load(origin: str) -> WorkloadProfile:
        if origin not in cache:
            cache[origin] = clipped_profile(original(origin), max_bucket)
        return cache[origin]

    monkeypatch.setattr(workload, "load_workload_profile", load)


__all__ = [
    "DEFAULT_MAX_BUCKET",
    "DEFAULT_MAX_FANOUT_BUCKET",
    "DEFAULT_MAX_RECORDS_BUCKET",
    "clip_committed_profiles",
    "clipped_profile",
]
