"""Tests for RawFailureSample Pydantic model and its integration into DaemonStatus."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from polylogue.daemon.status import (
    DaemonStatus,
    RawFailureSample,
    _raw_failure_info,
)


class TestRawFailureSampleModel:
    """Contract tests for the RawFailureSample Pydantic model."""

    def test_constructs_with_valid_fields(self) -> None:
        sample = RawFailureSample(
            failure_kind="parse_error",
            provider_hint="claude-code",
            redacted_error="malformed JSONL: unexpected token at line 5",
        )
        assert sample.failure_kind == "parse_error"
        assert sample.provider_hint == "claude-code"
        assert sample.redacted_error == "malformed JSONL: unexpected token at line 5"

    def test_constructs_with_minimal_fields(self) -> None:
        sample = RawFailureSample(failure_kind="unknown")
        assert sample.failure_kind == "unknown"
        assert sample.provider_hint is None
        assert sample.redacted_error == ""

    def test_rejects_invalid_failure_kind(self) -> None:
        with pytest.raises(ValidationError):
            RawFailureSample(failure_kind="invalid_kind")  # type: ignore[arg-type]

    def test_all_valid_failure_kinds(self) -> None:
        for kind in ("decode_error", "parse_error", "schema_violation", "unknown"):
            sample = RawFailureSample(failure_kind=kind)
            assert sample.failure_kind == kind

    def test_redacts_absolute_file_paths(self) -> None:
        sample = RawFailureSample(
            failure_kind="decode_error",
            redacted_error="JSONDecodeError at /home/user/project/src/file.py:42",
        )
        assert "/home/user/project/src/file.py" not in sample.redacted_error
        assert "JSONDecodeError" in sample.redacted_error

    def test_redacts_realm_paths(self) -> None:
        sample = RawFailureSample(
            failure_kind="parse_error",
            redacted_error="Failed to parse /realm/project/polylogue/data/input.json",
        )
        assert "/realm/project/polylogue/data/input.json" not in sample.redacted_error
        assert "Failed to parse" in sample.redacted_error

    def test_redacts_multiple_paths_in_single_error(self) -> None:
        sample = RawFailureSample(
            failure_kind="schema_violation",
            redacted_error="Schema mismatch: /tmp/a.json vs /nix/store/hash-name/file.py",
        )
        assert "/tmp/a.json" not in sample.redacted_error
        assert "/nix/store" not in sample.redacted_error
        assert "Schema mismatch" in sample.redacted_error

    def test_preserves_non_path_text(self) -> None:
        sample = RawFailureSample(
            failure_kind="parse_error",
            redacted_error="JSONDecodeError: Expecting value: line 1 column 1",
        )
        assert sample.redacted_error == "JSONDecodeError: Expecting value: line 1 column 1"

    def test_handles_empty_redacted_error(self) -> None:
        sample = RawFailureSample(failure_kind="unknown", redacted_error="")
        assert sample.redacted_error == ""

    def test_handles_none_redacted_error_coerced(self) -> None:
        sample = RawFailureSample(failure_kind="unknown", redacted_error=None)  # type: ignore[arg-type]
        assert sample.redacted_error == ""

    def test_handles_non_string_redacted_error_coerced(self) -> None:
        sample = RawFailureSample(failure_kind="unknown", redacted_error=42)  # type: ignore[arg-type]
        assert sample.redacted_error == ""

    def test_model_dump_excludes_raw_paths(self) -> None:
        sample = RawFailureSample(
            failure_kind="decode_error",
            provider_hint="claude-code",
            redacted_error="Failure in /home/user/secret/data.json: decoding error",
        )
        dumped = sample.model_dump()
        assert "/home/user/secret/data.json" not in dumped["redacted_error"]


class TestRawFailureSampleInDaemonStatus:
    """Integration: DaemonStatus.raw_failure_samples uses typed model."""

    def test_daemon_status_accepts_typed_samples(self) -> None:
        samples = [
            RawFailureSample(
                failure_kind="parse_error",
                provider_hint="claude-code",
                redacted_error="bad record at line 3",
            )
        ]
        status = DaemonStatus(raw_failure_samples=samples)
        assert status.raw_failure_samples == samples
        assert status.raw_failure_samples[0].failure_kind == "parse_error"

    def test_daemon_status_defaults_to_empty_list(self) -> None:
        status = DaemonStatus()
        assert status.raw_failure_samples == []

    def test_daemon_status_model_dump_serializes_samples(self) -> None:
        samples = [
            RawFailureSample(
                failure_kind="schema_violation",
                provider_hint=None,
                redacted_error="Missing required field: title",
            )
        ]
        status = DaemonStatus(raw_failure_samples=samples)
        dumped = status.model_dump()
        assert isinstance(dumped["raw_failure_samples"], list)
        assert dumped["raw_failure_samples"][0]["failure_kind"] == "schema_violation"
        assert dumped["raw_failure_samples"][0]["provider_hint"] is None
        assert "Missing required field" in dumped["raw_failure_samples"][0]["redacted_error"]

    def test_rejects_non_raw_failure_sample_items(self) -> None:
        with pytest.raises(ValidationError):
            DaemonStatus(raw_failure_samples=["not a model"])  # type: ignore[list-item]


class TestRawFailureInfoProducesTypedSamples:
    """_raw_failure_info() returns typed RawFailureSample instances."""

    def test_raw_failure_info_samples_are_typed(self, tmp_path: Path) -> None:
        db = tmp_path / "polylogue.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE raw_conversations (
                    raw_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    payload_provider TEXT,
                    source_name TEXT,
                    source_path TEXT NOT NULL,
                    source_index INTEGER,
                    blob_size INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    file_mtime TEXT,
                    parsed_at TEXT,
                    parse_error TEXT,
                    validated_at TEXT,
                    validation_status TEXT,
                    validation_error TEXT,
                    validation_drift_count INTEGER DEFAULT 0,
                    validation_provider TEXT,
                    validation_mode TEXT,
                    detection_warnings TEXT
                );
                INSERT INTO raw_conversations (
                    raw_id, provider_name, source_path, blob_size, acquired_at,
                    parse_error, validation_status
                ) VALUES (
                    'raw-1', 'claude-code', '/data/session.jsonl', 1024,
                    '2025-01-01T00:00:00Z',
                    'JSONDecodeError: Unterminated string at /home/user/file.py:202',
                    NULL
                );
                """
            )

        with patch("polylogue.daemon.status.db_path", return_value=db):
            info = _raw_failure_info()

        samples = info["samples"]
        assert isinstance(samples, list)
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        assert sample.failure_kind == "decode_error"
        assert sample.provider_hint == "claude-code"
        assert "/home/user/file.py" not in sample.redacted_error
        assert "JSONDecodeError" in sample.redacted_error

    def test_raw_failure_info_schema_violation_kind(self, tmp_path: Path) -> None:
        db = tmp_path / "polylogue.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE raw_conversations (
                    raw_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    payload_provider TEXT,
                    source_name TEXT,
                    source_path TEXT NOT NULL,
                    source_index INTEGER,
                    blob_size INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    file_mtime TEXT,
                    parsed_at TEXT,
                    parse_error TEXT,
                    validated_at TEXT,
                    validation_status TEXT,
                    validation_error TEXT,
                    validation_drift_count INTEGER DEFAULT 0,
                    validation_provider TEXT,
                    validation_mode TEXT,
                    detection_warnings TEXT
                );
                INSERT INTO raw_conversations (
                    raw_id, provider_name, source_path, blob_size, acquired_at,
                    parse_error, validation_status, validation_error
                ) VALUES (
                    'raw-2', 'chatgpt', '/data/conv.json', 512,
                    '2025-02-01T00:00:00Z',
                    NULL, 'FAILED', 'Missing required field: mapping'
                );
                """
            )

        with patch("polylogue.daemon.status.db_path", return_value=db):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        assert sample.failure_kind == "schema_violation"
        assert sample.provider_hint == "chatgpt"

    def test_raw_failure_info_generic_parse_error_kind(self, tmp_path: Path) -> None:
        db = tmp_path / "polylogue.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE raw_conversations (
                    raw_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    payload_provider TEXT,
                    source_name TEXT,
                    source_path TEXT NOT NULL,
                    source_index INTEGER,
                    blob_size INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    file_mtime TEXT,
                    parsed_at TEXT,
                    parse_error TEXT,
                    validated_at TEXT,
                    validation_status TEXT,
                    validation_error TEXT,
                    validation_drift_count INTEGER DEFAULT 0,
                    validation_provider TEXT,
                    validation_mode TEXT,
                    detection_warnings TEXT
                );
                INSERT INTO raw_conversations (
                    raw_id, provider_name, source_path, blob_size, acquired_at,
                    parse_error, validation_status
                ) VALUES (
                    'raw-3', 'unknown', '/data/bad.json', 256,
                    '2025-03-01T00:00:00Z',
                    'Some weird error', NULL
                );
                """
            )

        with patch("polylogue.daemon.status.db_path", return_value=db):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        # Non-JSON parse error → "parse_error" (the error IS a parse error, just not JSON-specific)
        assert sample.failure_kind == "parse_error"

    def test_raw_failure_info_empty_when_no_failures(self, tmp_path: Path) -> None:
        db = tmp_path / "polylogue.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE raw_conversations (
                    raw_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    payload_provider TEXT,
                    source_name TEXT,
                    source_path TEXT NOT NULL,
                    source_index INTEGER,
                    blob_size INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    file_mtime TEXT,
                    parsed_at TEXT,
                    parse_error TEXT,
                    validated_at TEXT,
                    validation_status TEXT,
                    validation_error TEXT,
                    validation_drift_count INTEGER DEFAULT 0,
                    validation_provider TEXT,
                    validation_mode TEXT,
                    detection_warnings TEXT
                );
                """
            )

        with patch("polylogue.daemon.status.db_path", return_value=db):
            info = _raw_failure_info()

        assert info["parse_failures"] == 0
        assert info["validation_failures"] == 0
        assert info["quarantined"] == 0
        assert info["samples"] == []


class TestRawFailureSampleRedactionPattern:
    """Verify redaction covers realistic error path patterns."""

    def test_traceback_path_redacted(self) -> None:
        sample = RawFailureSample(
            failure_kind="decode_error",
            redacted_error=(
                "Traceback (most recent call last):\n"
                '  File "/nix/store/abc123-python3-3.12.5/lib/python3.12/json/decoder.py", line 355, in raw_decode\n'
                '    raise JSONDecodeError("Expecting value", s, err.value) from None\n'
                "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
            ),
        )
        assert "/nix/store" not in sample.redacted_error
        assert "json.decoder.JSONDecodeError" in sample.redacted_error
        assert "Expecting value" in sample.redacted_error

    def test_single_segment_path_preserved(self) -> None:
        """Single-segment paths like /bin or /tmp should be redacted too."""
        sample = RawFailureSample(
            failure_kind="parse_error",
            redacted_error="Failed to open /tmp/out.json for reading",
        )
        assert "/tmp/out.json" not in sample.redacted_error
        assert "Failed to open" in sample.redacted_error

    def test_relative_paths_not_redacted(self) -> None:
        """Relative paths like src/file.py should not be redacted."""
        sample = RawFailureSample(
            failure_kind="parse_error",
            redacted_error="Missing module: src/file.py not found",
        )
        # Relative paths should not match the absolute-path pattern
        assert "src/file.py" in sample.redacted_error

    def test_urls_not_redacted(self) -> None:
        """URLs should not be confused with file paths."""
        sample = RawFailureSample(
            failure_kind="unknown",
            redacted_error="Connection failed to https://api.example.com/v1/data",
        )
        assert "https://api.example.com/v1/data" in sample.redacted_error

    def test_redaction_marker_present(self) -> None:
        sample = RawFailureSample(
            failure_kind="decode_error",
            redacted_error="Error in /home/user/private/file.py at line 42",
        )
        assert "[redacted]" in sample.redacted_error
        assert "/home/user/private/file.py" not in sample.redacted_error

    def test_null_provider_hint_defaults_to_none(self) -> None:
        """provider_hint should be None when not provided, not empty string."""
        sample = RawFailureSample(failure_kind="unknown")
        assert sample.provider_hint is None
