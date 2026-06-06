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
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


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
        for kind in ("decode_error", "parse_error", "schema_violation", "maintenance", "unknown"):
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


def _seed_archive_raw_session(
    tmp_path: Path,
    *,
    raw_id: str,
    origin: str,
    native_id: str,
    source_path: str,
    parse_error: str | None = None,
    validation_status: str | None = None,
    validation_error: str | None = None,
    detection_warnings_json: str = "[]",
    blob_size: int = 256,
    acquired_at_ms: int = 1_770_000_000_000,
) -> tuple[Path, Path]:
    """Seed one archive `source.db` ``raw_sessions`` row.

    Returns ``(legacy_db, index_db)`` paths to patch onto
    ``polylogue.daemon.status`` so ``_active_status_db_path`` resolves to
    ``tmp_path/index.db`` and reads ``tmp_path/source.db``.
    """
    legacy_db = tmp_path / "polylogue.db"
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parse_error, validation_status, validation_error,
                detection_warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                origin,
                native_id,
                source_path,
                bytes(32),
                blob_size,
                acquired_at_ms,
                parse_error,
                validation_status,
                validation_error,
                detection_warnings_json,
            ),
        )
        conn.commit()
    return legacy_db, index_db


class TestRawFailureInfoProducesTypedSamples:
    """_raw_failure_info() returns typed RawFailureSample instances."""

    def test_raw_failure_info_reads_archive_file_set_without_polylogue_db(self, tmp_path: Path) -> None:
        legacy_db = tmp_path / "polylogue.db"
        index_db = tmp_path / "index.db"
        archive_db = tmp_path / "source.db"
        initialize_archive_database(archive_db, ArchiveTier.SOURCE)
        with sqlite3.connect(archive_db) as conn:
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, blob_hash, blob_size,
                    acquired_at_ms, parse_error, detection_warnings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "raw-1",
                    "codex-session",
                    "native-1",
                    "/data/bad.jsonl",
                    bytes(32),
                    128,
                    1_770_000_000_000,
                    "JSONDecodeError: bad token at /home/user/private.py:1",
                    '["unknown envelope"]',
                ),
            )
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, blob_hash, blob_size,
                    acquired_at_ms, validation_status, validation_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "raw-2",
                    "chatgpt-export",
                    "native-2",
                    "/data/chatgpt.json",
                    bytes([1]) * 32,
                    256,
                    1_770_000_001_000,
                    "failed",
                    "Missing required field: mapping",
                ),
            )
            conn.commit()

        with (
            patch("polylogue.daemon.status.db_path", return_value=legacy_db),
            patch("polylogue.daemon.status.index_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        assert info["parse_failures"] == 1
        assert info["validation_failures"] == 1
        assert info["quarantined"] == 2
        assert info["detection_warnings"] == 1
        samples = cast(list[RawFailureSample], info["samples"])
        assert [sample.failure_kind for sample in samples] == ["schema_violation", "decode_error"]
        assert samples[0].provider_hint == "chatgpt-export"
        assert samples[1].provider_hint == "codex-session"
        assert "/home/user/private.py" not in samples[1].redacted_error

    def test_raw_failure_info_samples_are_typed(self, tmp_path: Path) -> None:
        legacy_db, index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-1",
            origin="claude-code-session",
            native_id="native-1",
            source_path="/data/session.jsonl",
            parse_error="JSONDecodeError: Unterminated string at /home/user/file.py:202",
            blob_size=1024,
        )

        with (
            patch("polylogue.daemon.status.db_path", return_value=legacy_db),
            patch("polylogue.daemon.status.index_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        samples = info["samples"]
        assert isinstance(samples, list)
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        assert sample.failure_kind == "decode_error"
        assert sample.provider_hint == "claude-code-session"
        assert "/home/user/file.py" not in sample.redacted_error
        assert "JSONDecodeError" in sample.redacted_error

    def test_raw_failure_info_schema_violation_kind(self, tmp_path: Path) -> None:
        legacy_db, index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-2",
            origin="chatgpt-export",
            native_id="native-2",
            source_path="/data/conv.json",
            validation_status="failed",
            validation_error="Missing required field: mapping",
            blob_size=512,
        )

        with (
            patch("polylogue.daemon.status.db_path", return_value=legacy_db),
            patch("polylogue.daemon.status.index_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        assert sample.failure_kind == "schema_violation"
        assert sample.provider_hint == "chatgpt-export"

    def test_raw_failure_info_generic_parse_error_kind(self, tmp_path: Path) -> None:
        legacy_db, index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-3",
            origin="unknown-export",
            native_id="native-3",
            source_path="/data/bad.json",
            parse_error="Some weird error",
            blob_size=256,
        )

        with (
            patch("polylogue.daemon.status.db_path", return_value=legacy_db),
            patch("polylogue.daemon.status.index_db_path", return_value=index_db),
        ):
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
                CREATE TABLE raw_sessions (
                    raw_id TEXT PRIMARY KEY,
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
