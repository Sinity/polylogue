"""Tests for shared conversation materialization helpers."""

from __future__ import annotations

import pytest

from polylogue.archive.message.roles import Role
from polylogue.pipeline.materialization_runtime import _timestamp_sort_key
from polylogue.pipeline.prepare_transform_content import canonicalize_message_content
from polylogue.sources.parsers.base import ParsedMessage


def test_timestamp_sort_key_accepts_iso_datetime() -> None:
    assert _timestamp_sort_key("2024-01-15T10:30:00Z") == 1705314600.0


def test_timestamp_sort_key_normalizes_millisecond_epoch() -> None:
    assert _timestamp_sort_key("1705314600000") == 1705314600.0


def test_canonicalize_message_skips_harmonizer_without_provider_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_harmonize(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("messages without provider_meta do not need viewport harmonization")

    monkeypatch.setattr("polylogue.pipeline.prepare_transform_content.harmonize_parsed_message", fail_harmonize)
    message = ParsedMessage(
        provider_message_id="m1",
        role=Role.USER,
        text="hello",
    )

    assert canonicalize_message_content("codex", message) is message
