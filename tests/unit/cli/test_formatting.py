"""Focused tests for CLI formatting helpers."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.formatting import (
    announce_plain_mode,
    format_counts,
    format_cursors,
    format_index_status,
    format_source_label,
    format_sources_summary,
    should_use_plain,
)
from polylogue.config import Source


@pytest.mark.parametrize(
    ("plain", "env_value", "tty", "expected"),
    [
        (True, None, True, True),
        (False, "1", True, True),
        (False, None, False, True),
        (False, None, True, False),
    ],
)
def test_should_use_plain_contract(
    plain: bool,
    env_value: str | None,
    tty: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env_value is None:
        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", env_value)

    with patch("sys.stdout.isatty", return_value=tty), patch("sys.stderr.isatty", return_value=tty):
        assert should_use_plain(plain=plain) is expected


@pytest.mark.parametrize("falsey", ["0", "false", "no"])
def test_should_use_plain_falsey_env_values_do_not_force_plain(
    falsey: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", falsey)
    with patch("sys.stdout.isatty", return_value=True), patch("sys.stderr.isatty", return_value=True):
        assert should_use_plain(plain=False) is False


def test_announce_plain_mode_writes_to_stderr() -> None:
    captured = StringIO()
    with patch.object(sys, "stderr", captured):
        announce_plain_mode()
    assert "Plain output active" in captured.getvalue()


@pytest.mark.parametrize(
    ("cursors", "expected_parts"),
    [
        ({}, None),
        ({"inbox": {"file_count": 10}}, ("10 files", "inbox")),
        ({"source": {"file_count": 5, "error_count": 2}}, ("2 errors",)),
        ({"source": {"file_count": 5, "error_count": 0}}, ("5 files",)),
        ({"source": {"latest_mtime": 1704067200}}, ("latest",)),
        ({"source": {"latest_file_name": "chat.json"}}, ("latest chat.json",)),
        ({"source": {"latest_path": "/tmp/export.json"}}, ("latest export.json",)),
        ({"src": "plain_string"}, ("src", "unknown")),
        ({"inbox": {"file_count": 5}, "drive": {"file_count": 3}}, ("inbox", "drive", ";")),
    ],
)
def test_format_cursors_contract(cursors: dict[str, object], expected_parts: tuple[str, ...] | None) -> None:
    result = format_cursors(cursors)
    if expected_parts is None:
        assert result is None
        return
    assert result is not None
    for part in expected_parts:
        assert part in result


@pytest.mark.parametrize(
    ("counts", "expected_parts"),
    [
        ({"conversations": 10, "messages": 100}, ("10 conv", "100 msg")),
        ({"conversations": 5, "messages": 50, "rendered": 5}, ("5 rendered",)),
        ({"acquired": 4, "validated": 4, "validation_drift": 2}, ("4 acquired", "4 validated", "2 drift")),
        ({"conversations": 5, "messages": 50, "rendered": 0}, ("5 conv", "50 msg")),
        ({}, ("0 conv", "0 msg")),
    ],
)
def test_format_counts_contract(counts: dict[str, object], expected_parts: tuple[str, ...]) -> None:
    result = format_counts(counts)
    for part in expected_parts:
        assert part in result


@pytest.mark.parametrize(
    ("stage", "indexed", "error", "expected"),
    [
        ("parse", False, None, "Index: skipped"),
        ("render", True, None, "Index: skipped"),
        ("index", False, "boom", "Index: error"),
        ("index", True, None, "Index: ok"),
        ("all", False, None, "Index: up-to-date"),
    ],
)
def test_format_index_status_contract(stage: str, indexed: bool, error: str | None, expected: str) -> None:
    assert format_index_status(stage, indexed, error) == expected


@pytest.mark.parametrize(
    ("source_name", "provider_name", "expected"),
    [
        ("inbox", "claude", "inbox/claude"),
        ("chatgpt", "chatgpt", "chatgpt"),
        (None, "codex", "codex"),
    ],
)
def test_format_source_label_contract(source_name: str | None, provider_name: str, expected: str) -> None:
    assert format_source_label(source_name, provider_name) == expected


def test_format_sources_summary_contract() -> None:
    sources = [
        Source(name="inbox", path=Path("/inbox")),
        Source(name="gemini", folder="folder-id"),
    ]
    result = format_sources_summary(sources)
    assert "inbox" in result
    assert "gemini (drive)" in result


def test_format_sources_summary_marks_missing() -> None:
    source = MagicMock()
    source.name = "broken"
    source.path = None
    source.folder = None
    assert "broken (missing)" in format_sources_summary([source])


def test_format_sources_summary_truncates_long_lists() -> None:
    sources = [Source(name=f"source{i}", path=Path(f"/src{i}")) for i in range(12)]
    result = format_sources_summary(sources)
    assert "+4 more" in result
    assert result.count(",") == 8
