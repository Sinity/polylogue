"""Focused edge-regression coverage for source iteration.

Broader provider/source semantics now live in the law and contract suites:
- test_source_laws.py
- test_parse_laws.py
- test_drive_client_laws.py
- test_unified_semantic_laws.py

This file keeps only the handful of concrete operational edge cases that are
still clearer as direct examples.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.config import Source
from polylogue.sources.source import (
    _iter_json_stream,
    iter_source_conversations,
    iter_source_conversations_with_raw,
    parse_payload,
)


@pytest.fixture
def cursor_state() -> dict[str, object]:
    return {}


def test_jsonl_multiple_errors_logging() -> None:
    content = b'{"a": 1}\n{bad}\n{bad}\n{bad}\n{bad}\n{"b": 2}\n'
    handle = BytesIO(content)
    with patch("polylogue.sources.source.logger") as mock_logger:
        results = list(_iter_json_stream(handle, "test.jsonl"))

    assert len(results) == 2
    assert mock_logger.warning.call_count >= 1


def test_iter_source_conversations_with_raw_ignores_stat_os_error(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    json_file = source_dir / "conv.json"
    json_file.write_text('{"mapping": {}}')
    source = Source(name="test", path=source_dir)

    original_stat = Path.stat

    def mock_stat(self, **kwargs):
        if self == json_file or str(self) == str(json_file):
            raise OSError("no stat")
        return original_stat(self, **kwargs)

    with patch.object(Path, "stat", mock_stat):
        items = list(iter_source_conversations_with_raw(source, capture_raw=True))

    assert len(items) == 1


@pytest.mark.parametrize(
    ("error_type", "setup"),
    [
        ("file_not_found", None),
        ("json_decode", lambda path: path.write_text("{invalid}")),
        ("unicode_decode", lambda path: path.write_bytes(b"\xff\xfe{invalid}")),
    ],
    ids=["file_not_found_toctou", "json_decode_error", "unicode_decode_error"],
)
def test_iter_source_conversations_error_variants_increment_failed_count(
    tmp_path: Path,
    cursor_state: dict[str, object],
    error_type: str,
    setup,
) -> None:
    json_file = tmp_path / "test.json"

    if error_type == "file_not_found":
        json_file.write_text('{"mapping": {}}')
        source = Source(name="test", path=json_file)
        with patch("polylogue.sources.source.Path.open", side_effect=FileNotFoundError("deleted")):
            list(iter_source_conversations(source, cursor_state=cursor_state))
    else:
        assert setup is not None
        setup(json_file)
        source = Source(name="test", path=json_file)
        list(iter_source_conversations(source, cursor_state=cursor_state))

    assert cursor_state.get("failed_count", 0) > 0


def test_iter_source_conversations_latest_mtime_oserror_is_ignored(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    conv_file = source_dir / "conv.json"
    conv_file.write_text(json.dumps({"mapping": {}}))

    source = Source(name="test", path=source_dir)
    cursor_state: dict[str, object] = {}
    original_stat = Path.stat

    def mock_stat_error(self, **kwargs):
        if self == conv_file or str(self) == str(conv_file):
            raise OSError("Stat failed")
        return original_stat(self, **kwargs)

    with patch.object(Path, "stat", mock_stat_error):
        list(iter_source_conversations(source, cursor_state=cursor_state))

    assert cursor_state.get("file_count") == 1
    assert "latest_mtime" not in cursor_state


def test_parse_payload_max_depth_returns_empty() -> None:
    assert parse_payload("chatgpt", {}, "test", _depth=11) == []
