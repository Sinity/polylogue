"""The append-only pickle spool that parse and publication passes replay."""

from __future__ import annotations

import gc

import pytest

from polylogue.sources.pickle_spool import PickleSpool


def test_spool_replays_values_in_write_order_every_time() -> None:
    spool = PickleSpool[object]()
    values: list[object] = [{"a": 1}, "x" * 100_000, [1, 2, {"b": None}], ""]
    for value in values:
        spool.append(value)

    assert len(spool) == len(values)
    assert list(spool) == values
    assert list(spool) == values
    spool.close()


def test_indexed_spool_replays_any_suffix() -> None:
    spool = PickleSpool[int](indexed=True)
    for value in range(50):
        spool.append(value)

    assert list(spool.iter_from(0)) == list(range(50))
    assert list(spool.iter_from(37)) == list(range(37, 50))
    assert list(spool.iter_from(50)) == []
    with pytest.raises(IndexError):
        list(spool.iter_from(51))
    spool.close()


def test_unindexed_spool_refuses_a_suffix_replay() -> None:
    spool = PickleSpool[int]()
    spool.append(1)

    with pytest.raises(ValueError):
        list(spool.iter_from(1))
    spool.close()


def test_interleaved_replays_keep_their_own_positions() -> None:
    """Replays read by offset, so interleaving two never skips or repeats a value.

    Anti-vacuity: read through the shared file position (``seek`` + ``read``)
    and the second replay starts where the first one stopped.
    """
    spool = PickleSpool[int](indexed=True)
    for value in range(10):
        spool.append(value)
    first = spool.iter_from(0)
    second = spool.iter_from(0)

    seen = [(next(first), next(second)) for _ in range(10)]

    assert seen == [(value, value) for value in range(10)]
    spool.close()


def test_dropped_spool_releases_its_file_after_the_last_replay() -> None:
    """A replay in flight keeps reading a spool its registry already dropped."""
    spool = PickleSpool[int]()
    for value in range(3):
        spool.append(value)
    replay = iter(spool)
    handle = spool._file
    del spool
    gc.collect()

    assert list(replay) == [0, 1, 2]
    del replay
    gc.collect()
    assert handle.closed


@pytest.mark.parametrize("block", [1, 5, 64])
def test_block_reads_cross_value_boundaries(monkeypatch: pytest.MonkeyPatch, block: int) -> None:
    """Values straddling or exceeding the read block replay intact.

    Anti-vacuity: drop the exact read for a value larger than the block and
    those values come back truncated.
    """
    from polylogue.sources import pickle_spool

    monkeypatch.setattr(pickle_spool, "_READ_BLOCK_BYTES", block)
    spool = PickleSpool[tuple[str, int]](indexed=True)
    values = [("v" * (index * 7 % 97), index) for index in range(120)]
    for value in values:
        spool.append(value)

    assert list(spool) == values
    assert list(spool.iter_from(61)) == values[61:]
    spool.close()
