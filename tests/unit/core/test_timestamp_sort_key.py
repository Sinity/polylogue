"""Tests for shared session materialization helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.pipeline.materialization_runtime import _timestamp_sort_key, materialize_session
from polylogue.pipeline.prepare_transform_content import canonicalize_message_content
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.types import Provider


def test_timestamp_sort_key_accepts_iso_datetime() -> None:
    assert _timestamp_sort_key("2024-01-15T10:30:00Z") == 1705314600.0


def test_timestamp_sort_key_normalizes_millisecond_epoch() -> None:
    assert _timestamp_sort_key("1705314600000") == 1705314600.0


def test_materialize_session_canonicalizes_epoch_session_timestamps(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="codex-epoch",
        title="Epoch timestamps",
        created_at="1705312200.123",
        updated_at="1705314600",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="hello",
                timestamp="1705312200.123",
            )
        ],
    )

    materialized = materialize_session(session, source_name="codex", archive_root=tmp_path)

    assert materialized.created_at == "2024-01-15T09:50:00.123000+00:00"
    assert materialized.updated_at == "2024-01-15T10:30:00+00:00"
    assert materialized.sort_key == 1705314600.0


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
