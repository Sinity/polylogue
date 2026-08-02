"""Tests for shared utilities in polylogue.core.common."""

from __future__ import annotations

from collections.abc import Iterator

from polylogue.core.common import forward_bounded_slice, peek_truthy


class _CountingIterable:
    def __init__(self, items: list[int]) -> None:
        self._items = items
        self.touched = 0

    def __iter__(self) -> Iterator[int]:
        for item in self._items:
            self.touched += 1
            yield item


class TestPeekTruthy:
    def test_true_for_non_empty_touches_only_first_item(self) -> None:
        source = _CountingIterable(list(range(10_000)))

        assert peek_truthy(source) is True
        assert source.touched == 1

    def test_false_for_empty(self) -> None:
        assert peek_truthy(_CountingIterable([])) is False

    def test_false_for_generator_expression(self) -> None:
        empty: list[int] = []
        assert peek_truthy(item for item in empty) is False


class TestForwardBoundedSlice:
    def test_open_slice_is_bounded(self) -> None:
        assert forward_bounded_slice(slice(None, 5)) == (0, 5, 1)

    def test_explicit_start_stop_step_is_bounded(self) -> None:
        assert forward_bounded_slice(slice(2, 10, 2)) == (2, 10, 2)

    def test_full_open_slice_is_bounded_with_none_stop(self) -> None:
        assert forward_bounded_slice(slice(None, None)) == (0, None, 1)

    def test_negative_start_is_not_bounded(self) -> None:
        assert forward_bounded_slice(slice(-5, None)) is None

    def test_negative_stop_is_not_bounded(self) -> None:
        assert forward_bounded_slice(slice(None, -1)) is None

    def test_negative_step_is_not_bounded(self) -> None:
        assert forward_bounded_slice(slice(None, None, -1)) is None

    def test_zero_step_is_not_bounded(self) -> None:
        assert forward_bounded_slice(slice(None, None, 0)) is None
