"""Tests for _timestamp_sort_key (relocated to polylogue.core.timestamps)."""

from __future__ import annotations

from polylogue.core.timestamps import _timestamp_sort_key


def test_timestamp_sort_key_accepts_iso_datetime() -> None:
    assert _timestamp_sort_key("2024-01-15T10:30:00Z") == 1705314600.0


def test_timestamp_sort_key_normalizes_millisecond_epoch() -> None:
    assert _timestamp_sort_key("1705314600000") == 1705314600.0
