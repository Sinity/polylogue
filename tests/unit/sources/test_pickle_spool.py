"""The append-only pickle spool that parse and publication passes replay."""

from __future__ import annotations

import gc
import pickle
from typing import Any

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


def test_indexed_spool_large_ordinal_tape_preserves_interleaved_suffixes() -> None:
    """A large count retains exact ordinal replay without a resident offset array."""
    spool = PickleSpool[tuple[int, str]](indexed=True)
    try:
        for ordinal in range(100_003):
            spool.append((ordinal, "neutral"))
        first = spool.iter_from(99_997)
        second = spool.iter_from(37)
        assert next(first) == (99_997, "neutral")
        assert next(second) == (37, "neutral")
        assert list(first) == [(ordinal, "neutral") for ordinal in range(99_998, 100_003)]
        assert next(second) == (38, "neutral")
        assert len(spool) == 100_003
    finally:
        spool.close()


def test_indexed_spool_closes_both_original_files() -> None:
    spool = PickleSpool[int](indexed=True)
    values = spool._file
    offsets = spool._offsets
    assert offsets is not None
    spool.append(0)
    assert list(spool) == [0]
    spool.close()
    assert values.closed and offsets.closed
    spool.close()


def test_indexed_spool_retained_python_memory_does_not_grow_per_ordinal() -> None:
    import tracemalloc

    count = 250_003
    tracemalloc.start()
    spool = PickleSpool[int](indexed=True)
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        for ordinal in range(count):
            spool.append(ordinal)
        retained, _ = tracemalloc.get_traced_memory()
        # Even a four-byte resident ordinal array exceeds this allowance;
        # transient per-record serialization does not remain after append.
        assert retained - baseline < count * 4
        assert next(spool.iter_from(count - 1)) == count - 1
        assert len(spool) == count
    finally:
        spool.close()
        tracemalloc.stop()


def test_indexed_spool_close_failure_settles_sibling_and_keeps_original_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    actual = tempfile.TemporaryFile
    handles: list[object] = []
    failure = OSError("synthetic original value-file close failure")

    class Handle:
        def __init__(self, underlying: object, fail_once: bool) -> None:
            self.underlying = underlying
            self.fail_once = fail_once

        def __getattr__(self, name: str) -> object:
            return getattr(self.underlying, name)

        def close(self) -> None:
            if self.fail_once:
                self.fail_once = False
                raise failure
            self.underlying.close()  # type: ignore[attr-defined]

    def opened(*args: Any, **kwargs: Any) -> Handle:
        handle = Handle(actual(*args, **kwargs), not handles)
        handles.append(handle)
        return handle

    monkeypatch.setattr("polylogue.sources.pickle_spool.tempfile.TemporaryFile", opened)
    spool = PickleSpool[int](indexed=True)
    spool.append(7)
    with pytest.raises(OSError) as observed:
        spool.close()
    assert observed.value is failure
    assert len(handles) == 2
    assert not handles[0].closed and handles[1].closed  # type: ignore[attr-defined]
    assert spool._release.alive
    spool.close()
    assert all(handle.closed for handle in handles)  # type: ignore[attr-defined]
    assert not spool._release.alive


@pytest.mark.parametrize("indexed", [False, True])
def test_streamed_pickle_nested_value_and_failed_append_keep_original_tape(indexed: bool) -> None:
    spool = PickleSpool[object](indexed=indexed)
    first = {"text": "neutral" * 200_003, "nested": [{"empty": [], "number": index} for index in range(1001)]}
    second = [None, False, 0, {"exact": "e\u0301"}]
    try:
        spool.append(first)
        original_size = spool._size
        with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
            spool.append({"valid_prefix": "neutral" * 100_003, "invalid": lambda: None})
        assert len(spool) == 1 and spool._size == original_size
        assert spool._file.seek(0, 2) == original_size
        if spool._offsets is not None:
            assert spool._offsets.seek(0, 2) == 8
        assert list(spool) == [first]
        spool.append(second)
        assert list(spool) == [first, second]
        if indexed:
            assert list(spool.iter_from(1)) == [second]
    finally:
        spool.close()


def test_streamed_pickle_replay_refuses_truncated_original_frame() -> None:
    spool = PickleSpool[object](indexed=True)
    try:
        spool.append({"first": "neutral" * 100_003})
        spool.append({"second": "neutral"})
        spool._file.flush()
        spool._file.truncate(spool._size - 3)
        replay = iter(spool)
        assert next(replay) == {"first": "neutral" * 100_003}
        with pytest.raises(EOFError):
            next(replay)
    finally:
        spool.close()


def test_streamed_pickle_large_bytes_avoids_a_whole_serialized_record_copy() -> None:
    import tracemalloc

    # Allocate the actual input before measuring spool working memory.
    parts = [bytes([97 + index]) * (4 * 1024 * 1024) for index in range(3)]
    value = {"parts": parts, "repeated": parts[0]}
    total = sum(map(len, parts))
    largest = max(map(len, parts))
    spool = PickleSpool[object](indexed=True)
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        spool.append(value)
        _, append_peak = tracemalloc.get_traced_memory()
        assert append_peak - baseline < total

        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        replay = iter(spool)
        restored = next(replay)
        _, replay_peak = tracemalloc.get_traced_memory()
        assert replay_peak - baseline < total + 2 * largest
        assert restored == value
        assert isinstance(restored, dict)
        assert restored["repeated"] is restored["parts"][0]
        with pytest.raises(StopIteration):
            next(replay)
        del restored, replay
        assert list(spool) == [value]
    finally:
        spool.close()
        tracemalloc.stop()
