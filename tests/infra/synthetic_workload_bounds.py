"""Committed workload profiles with their length tails clipped, for behavior tests.

The committed profiles reproduce real tails: a draw can be a multi-gigabyte
tool result. Tests that materialize a generated corpus to inspect record
shapes clip each length histogram to ``max_bucket`` (log2), so a seed that
lands in the tail cannot exhaust the test host's memory. Tail behavior itself
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


def clipped_profile(profile: WorkloadProfile, max_bucket: int = DEFAULT_MAX_BUCKET) -> WorkloadProfile:
    streams = {
        name: dataclasses.replace(stream, lengths={kind: _clip(h, max_bucket) for kind, h in stream.lengths.items()})
        for name, stream in profile.streams.items()
    }
    return dataclasses.replace(
        profile,
        streams=streams,
        template_strings=_clip_paths(profile.template_strings, max_bucket),
        template_lists=_clip_paths(profile.template_lists, max_bucket),
    )


def clip_committed_profiles(monkeypatch: pytest.MonkeyPatch, *, max_bucket: int = DEFAULT_MAX_BUCKET) -> None:
    """Make every ``load_workload_profile`` in the generator return a clipped profile."""
    original = workload.load_workload_profile
    cache: dict[str, WorkloadProfile] = {}

    def load(origin: str) -> WorkloadProfile:
        if origin not in cache:
            cache[origin] = clipped_profile(original(origin), max_bucket)
        return cache[origin]

    monkeypatch.setattr(workload, "load_workload_profile", load)


__all__ = ["DEFAULT_MAX_BUCKET", "clip_committed_profiles", "clipped_profile"]
