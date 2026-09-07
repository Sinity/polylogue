"""The batch policy is the only place cold and live differ."""

from __future__ import annotations

import pytest

from polylogue.daemon.batch_policy import (
    COLD_BACKLOG_MIN_QUEUE_AGE_S,
    COLD_BACKLOG_MIN_QUEUE_DEPTH,
    LIVE_TRICKLE_MAX_BYTES,
    LIVE_TRICKLE_MAX_FILES,
    IngestMode,
    WriteDestination,
    select_batch_shape,
)

_OWNED_INDEX = WriteDestination(tier="index", owned_rebuildable_generation=True)
_LIVE_INDEX = WriteDestination(tier="index")
_SOURCE = WriteDestination(tier="source")


def test_an_idle_queue_takes_the_live_trickle_shape() -> None:
    shape = select_batch_shape(queue_depth=1, queue_age_s=0.0, destination=_LIVE_INDEX)
    assert shape.mode is IngestMode.LIVE_TRICKLE
    assert (shape.max_files, shape.max_bytes) == (LIVE_TRICKLE_MAX_FILES, LIVE_TRICKLE_MAX_BYTES)


def test_a_deep_and_old_queue_takes_the_cold_backlog_shape() -> None:
    shape = select_batch_shape(
        queue_depth=COLD_BACKLOG_MIN_QUEUE_DEPTH,
        queue_age_s=COLD_BACKLOG_MIN_QUEUE_AGE_S,
        destination=_OWNED_INDEX,
    )
    assert shape.mode is IngestMode.COLD_BACKLOG
    assert shape.max_bytes > LIVE_TRICKLE_MAX_BYTES
    assert shape.defer_secondary_indexes is False
    assert shape.fresh_build is False


def test_an_empty_owned_generation_admits_fresh_writer_and_index_deferral() -> None:
    shape = select_batch_shape(
        queue_depth=COLD_BACKLOG_MIN_QUEUE_DEPTH,
        queue_age_s=COLD_BACKLOG_MIN_QUEUE_AGE_S,
        destination=_OWNED_INDEX,
        archive_empty=True,
    )
    assert shape.defer_secondary_indexes is True
    assert shape.fresh_build is True


@pytest.mark.parametrize(
    ("depth", "age"),
    [(COLD_BACKLOG_MIN_QUEUE_DEPTH, 0.0), (1, COLD_BACKLOG_MIN_QUEUE_AGE_S)],
)
def test_depth_or_age_alone_is_not_a_cold_backlog(depth: int, age: float) -> None:
    """Depth alone is a burst of live appends; age alone is one stale file."""
    shape = select_batch_shape(queue_depth=depth, queue_age_s=age, destination=_OWNED_INDEX)
    assert shape.mode is IngestMode.LIVE_TRICKLE


def test_bulk_pragmas_need_an_owned_rebuildable_destination_not_a_deep_backlog() -> None:
    """The licence is ownership of a disposable artifact, never backlog size.

    Anti-vacuity: gate ``bulk_pragmas`` on the cold-backlog decision instead of
    on the destination and this goes green while a durable tier is written with
    ``synchronous=OFF``, where a host crash is unrecoverable rather than a
    discarded generation.
    """

    def shape(destination: WriteDestination) -> bool:
        return select_batch_shape(queue_depth=10_000, queue_age_s=3600.0, destination=destination).bulk_pragmas

    assert shape(_OWNED_INDEX) is True
    assert shape(_LIVE_INDEX) is False
    assert shape(_SOURCE) is False


@pytest.mark.parametrize("tier", ["source", "user", "audit", "ops"])
def test_a_durable_or_disposable_tier_never_admits_bulk_pragmas(tier: str) -> None:
    destination = WriteDestination(tier=tier, owned_rebuildable_generation=True)  # type: ignore[arg-type]
    assert destination.admits_bulk_pragmas is False


def test_live_trickle_never_takes_bulk_pragmas_even_on_an_owned_generation() -> None:
    """A live writer shares its tier with readers; the tradeoff does not apply."""
    shape = select_batch_shape(queue_depth=1, queue_age_s=0.0, destination=_OWNED_INDEX)
    assert shape.bulk_pragmas is False


def test_archive_wide_derivations_require_an_admitted_input_boundary() -> None:
    """A drained queue is an instant, not a boundary.

    Anti-vacuity: derive this from ``queue_depth == 0`` and an archive-wide pass
    runs against an input set that refills underneath it, binding output to a
    frame that never existed.
    """
    at_boundary = select_batch_shape(
        queue_depth=0, queue_age_s=0.0, destination=_LIVE_INDEX, at_admitted_input_boundary=True
    )
    mid_run = select_batch_shape(queue_depth=0, queue_age_s=0.0, destination=_LIVE_INDEX)
    assert at_boundary.archive_wide_derivations is True
    assert mid_run.archive_wide_derivations is False


@pytest.mark.parametrize(("depth", "age"), [(-1, 0.0), (0, -1.0)])
def test_a_nonsense_queue_measurement_is_refused(depth: int, age: float) -> None:
    with pytest.raises(ValueError):
        select_batch_shape(queue_depth=depth, queue_age_s=age, destination=_LIVE_INDEX)
