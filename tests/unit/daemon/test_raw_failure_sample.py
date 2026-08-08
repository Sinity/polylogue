"""Tests for RawFailureSample Pydantic model and its integration into DaemonStatus."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from polylogue.core.enums import ArtifactSupportStatus
from polylogue.core.json import JSONDocument
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_EVIDENCE_KINDS,
    RAW_FAILURE_TERMINAL_EVIDENCE_KINDS,
    RawFailureEvidenceKind,
)
from polylogue.daemon.status import (
    DaemonStatus,
    RawFailureSample,
    _raw_failure_info,
    format_daemon_status_lines,
    raw_failure_info_for_root,
)
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact
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
        for kind in (
            "decode_error",
            "parse_error",
            "schema_violation",
            "maintenance",
            "unknown",
            *RAW_FAILURE_DEFERRED_EVIDENCE_KINDS,
            *RAW_FAILURE_TERMINAL_EVIDENCE_KINDS,
        ):
            sample = RawFailureSample(failure_kind=cast(Any, kind))
            assert sample.failure_kind == kind

    def test_raw_evidence_kinds_have_closed_lifecycle_partition(self) -> None:
        assert (
            frozenset(
                {
                    "deferred_hot_jsonl_capture",
                    "deferred_claude_code_partial_jsonl",
                    "deferred_cas_frontier",
                    "deferred_codex_cas_frontier",
                }
            )
            == RAW_FAILURE_DEFERRED_EVIDENCE_KINDS
        )
        assert (
            frozenset(
                {
                    "terminal_corrupt_input",
                    "terminal_unknown_json_decode",
                    "terminal_unknown_export_no_session",
                    "terminal_unsupported_shape",
                }
            )
            == RAW_FAILURE_TERMINAL_EVIDENCE_KINDS
        )
        for value in (*RAW_FAILURE_DEFERRED_EVIDENCE_KINDS, *RAW_FAILURE_TERMINAL_EVIDENCE_KINDS):
            assert RawFailureEvidenceKind(value).lifecycle in {"deferred", "terminal"}

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
) -> Path:
    """Seed one archive `source.db` ``raw_sessions`` row.

    Returns the ``index.db`` path to patch onto
    ``polylogue.daemon.status._active_status_db_path`` so it resolves into
    ``tmp_path`` and reads the sibling ``tmp_path/source.db``.
    """
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
    return index_db


class TestRawFailureInfoProducesTypedSamples:
    """_raw_failure_info() returns typed RawFailureSample instances."""

    def test_raw_failure_info_reads_archive_file_set_from_archive_tiers(self, tmp_path: Path) -> None:
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
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
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
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-1",
            origin="claude-code-session",
            native_id="native-1",
            source_path="/data/session.jsonl",
            parse_error="JSONDecodeError: Unterminated string at /home/user/file.py:202",
            blob_size=1024,
        )

        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
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
        index_db = _seed_archive_raw_session(
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
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        assert sample.failure_kind == "schema_violation"
        assert sample.provider_hint == "chatgpt-export"

    def test_raw_failure_info_generic_parse_error_kind(self, tmp_path: Path) -> None:
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-3",
            origin="unknown-export",
            native_id="native-3",
            source_path="/data/bad.json",
            parse_error="Some weird error",
            blob_size=256,
        )

        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, RawFailureSample)
        # Non-JSON parse error → "parse_error" (the error IS a parse error, just not JSON-specific)
        assert sample.failure_kind == "parse_error"

    def test_raw_failure_info_prefers_failure_evidence_over_newer_artifact(self, tmp_path: Path) -> None:
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-multi-artifact",
            origin="claude-code-session",
            native_id="native-multi-artifact",
            source_path="/data/failure.jsonl",
            parse_error="captured JSONL payload ends before a complete record boundary",
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            upsert_raw_artifact(
                conn,
                "raw-multi-artifact",
                ArchiveSourceArtifact(
                    artifact_id="failure-evidence",
                    origin="claude-code-session",
                    source_path="/data/failure.jsonl",
                    source_index=0,
                    artifact_kind="deferred_claude_code_partial_jsonl",
                    classification_reason="deferred_claude_code_partial_jsonl",
                    support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                    parse_as_session=True,
                    schema_eligible=True,
                    first_observed_at_ms=100,
                    last_observed_at_ms=100,
                ),
            )
            upsert_raw_artifact(
                conn,
                "raw-multi-artifact",
                ArchiveSourceArtifact(
                    artifact_id="newer-unrelated-artifact",
                    origin="claude-code-session",
                    source_path="/data/newer.sqlite",
                    source_index=0,
                    artifact_kind="sqlite_state_database",
                    classification_reason="sqlite_state_database",
                    support_status=ArtifactSupportStatus.UNKNOWN,
                    first_observed_at_ms=200,
                    last_observed_at_ms=200,
                ),
            )

        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 1
        assert samples[0].failure_kind == "deferred_claude_code_partial_jsonl"
        assert samples[0].lifecycle == "deferred"

    def test_raw_failure_info_separates_closed_lifecycle_evidence(self, tmp_path: Path) -> None:
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-deferred",
            origin="claude-code-session",
            native_id="native-deferred",
            source_path="/data/deferred.jsonl",
            parse_error="captured JSONL payload ends before a complete record boundary",
            acquired_at_ms=1_770_000_000_001,
        )
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-terminal",
            origin="unknown-export",
            native_id="native-terminal",
            source_path="/data/terminal.json",
            parse_error="parsed raw payload produced no sessions",
            acquired_at_ms=1_770_000_000_002,
        )
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-unexplained",
            origin="codex-session",
            native_id="native-unexplained",
            source_path="/data/unexplained.jsonl",
            parse_error="raw revision CAS rejected an older accepted frontier",
            acquired_at_ms=1_770_000_000_003,
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            upsert_raw_artifact(
                conn,
                "raw-deferred",
                ArchiveSourceArtifact(
                    artifact_id="deferred-evidence",
                    origin="claude-code-session",
                    source_path="/data/deferred.jsonl",
                    source_index=0,
                    artifact_kind="deferred_hot_jsonl_capture",
                    classification_reason="deferred_hot_jsonl_capture",
                    support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                    parse_as_session=True,
                    schema_eligible=True,
                ),
            )
            upsert_raw_artifact(
                conn,
                "raw-terminal",
                ArchiveSourceArtifact(
                    artifact_id="terminal-evidence",
                    origin="unknown-export",
                    source_path="/data/terminal.json",
                    source_index=0,
                    artifact_kind="terminal_unsupported_shape",
                    classification_reason="terminal_unsupported_shape",
                    support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                ),
            )

        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        assert info["deferred_failures"] == 1
        assert info["terminal_rejections"] == 1
        assert info["unexplained_failures"] == 1
        samples = cast(list[RawFailureSample], info["samples"])
        assert {sample.lifecycle for sample in samples} == {"deferred", "terminal", "unexplained"}
        by_kind = {sample.provider_hint: sample.failure_kind for sample in samples}
        assert by_kind["claude-code-session"] == "deferred_hot_jsonl_capture"
        assert by_kind["unknown-export"] == "terminal_unsupported_shape"
        assert by_kind["codex-session"] == "parse_error"

    def test_raw_failure_info_uses_root_source_tier_for_pointer_index(self, tmp_path: Path) -> None:
        generation = tmp_path / "generation"
        generation.mkdir()
        active_index = generation / "index.db"
        sqlite3.connect(active_index).close()
        (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-terminal",
            origin="codex-session",
            native_id="terminal",
            source_path="/data/terminal.jsonl",
            parse_error="captured JSONL payload ends before a complete record boundary",
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            upsert_raw_artifact(
                conn,
                "raw-terminal",
                ArchiveSourceArtifact(
                    artifact_id="terminal-evidence",
                    origin="codex-session",
                    source_path="/data/terminal.jsonl",
                    source_index=0,
                    artifact_kind="terminal_corrupt_input",
                    classification_reason="terminal_corrupt_input",
                    support_status=ArtifactSupportStatus.DECODE_FAILED,
                ),
            )

        with patch("polylogue.daemon.status.archive_root", return_value=tmp_path):
            info = _raw_failure_info()

        assert info["parse_failures"] == 1
        assert info["terminal_rejections"] == 1

    def test_raw_failure_lifecycle_rejects_mismatched_or_malformed_artifacts(self, tmp_path: Path) -> None:
        """Status cannot bless evidence with the wrong support or raw identity."""
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-malformed",
            origin="codex-session",
            native_id="malformed",
            source_path="/data/malformed.jsonl",
            parse_error="parser failed",
        )
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-target",
            origin="codex-session",
            native_id="target",
            source_path="/data/target.jsonl",
            parse_error="parser failed",
        )
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-other",
            origin="codex-session",
            native_id="other",
            source_path="/data/other.jsonl",
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            # The evidence kind is terminal_corrupt_input, but its support
            # status belongs to terminal_unsupported_shape.
            upsert_raw_artifact(
                conn,
                "raw-malformed",
                ArchiveSourceArtifact(
                    artifact_id="malformed-evidence",
                    origin="codex-session",
                    source_path="/data/malformed.jsonl",
                    source_index=0,
                    artifact_kind="terminal_corrupt_input",
                    classification_reason="terminal_corrupt_input",
                    support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                ),
            )
            # A source-identity match is insufficient when the durable raw_id
            # points at a different retained payload.
            upsert_raw_artifact(
                conn,
                "raw-other",
                ArchiveSourceArtifact(
                    artifact_id="mismatched-evidence",
                    origin="codex-session",
                    source_path="/data/target.jsonl",
                    source_index=0,
                    artifact_kind="terminal_corrupt_input",
                    classification_reason="terminal_corrupt_input",
                    support_status=ArtifactSupportStatus.DECODE_FAILED,
                ),
            )
            conn.commit()

        snapshot = read_raw_failure_lifecycle(tmp_path / "source.db")
        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        assert snapshot.terminal == 0
        assert snapshot.unexplained == 2
        assert info["terminal_rejections"] == snapshot.terminal
        assert info["unexplained_failures"] == snapshot.unexplained

    def test_raw_failure_status_does_not_project_unvalidated_typed_kinds(self, tmp_path: Path) -> None:
        """Status uses the lifecycle reader's validation, support, and kind checks."""
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-validation-failed",
            origin="codex-session",
            native_id="validation-failed",
            source_path="/data/validation-failed.jsonl",
            validation_status="failed",
            validation_error="schema drift",
        )
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-kind-contradiction",
            origin="codex-session",
            native_id="kind-contradiction",
            source_path="/data/kind-contradiction.jsonl",
            parse_error="parser failed after artifact observation",
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            upsert_raw_artifact(
                conn,
                "raw-validation-failed",
                ArchiveSourceArtifact(
                    artifact_id="validation-failed-evidence",
                    origin="codex-session",
                    source_path="/data/validation-failed.jsonl",
                    source_index=0,
                    artifact_kind="terminal_corrupt_input",
                    classification_reason="terminal_corrupt_input",
                    support_status=ArtifactSupportStatus.DECODE_FAILED,
                ),
            )
            upsert_raw_artifact(
                conn,
                "raw-kind-contradiction",
                ArchiveSourceArtifact(
                    artifact_id="kind-contradiction-evidence",
                    origin="codex-session",
                    source_path="/data/kind-contradiction.jsonl",
                    source_index=0,
                    artifact_kind="terminal_corrupt_input",
                    classification_reason="terminal_corrupt_input",
                    support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                ),
            )
            conn.commit()

        snapshot = read_raw_failure_lifecycle(tmp_path / "source.db")
        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        by_origin_error = {sample.redacted_error: sample for sample in samples}
        assert snapshot.unexplained == 2
        assert info["unexplained_failures"] == 2
        assert by_origin_error["schema drift"].failure_kind == "schema_violation"
        assert by_origin_error["parser failed after artifact observation"].failure_kind == "parse_error"
        assert all(sample.failure_kind != "terminal_corrupt_input" for sample in samples)

    def test_daemon_status_lifecycle_counts_match_the_shared_projection(self, tmp_path: Path) -> None:
        """The health/status source is the same lifecycle projection as preflight."""
        index_db = _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-terminal",
            origin="codex-session",
            native_id="terminal",
            source_path="/data/terminal.jsonl",
            parse_error="bad input",
        )
        _seed_archive_raw_session(
            tmp_path,
            raw_id="raw-deferred",
            origin="codex-session",
            native_id="deferred",
            source_path="/data/deferred.jsonl",
            parse_error="hot capture",
            acquired_at_ms=1_770_000_000_001,
        )
        with sqlite3.connect(tmp_path / "source.db") as conn:
            upsert_raw_artifact(
                conn,
                "raw-terminal",
                ArchiveSourceArtifact(
                    artifact_id="terminal-evidence",
                    origin="codex-session",
                    source_path="/data/terminal.jsonl",
                    source_index=0,
                    artifact_kind="terminal_corrupt_input",
                    classification_reason="terminal_corrupt_input",
                    support_status=ArtifactSupportStatus.DECODE_FAILED,
                ),
            )
            upsert_raw_artifact(
                conn,
                "raw-deferred",
                ArchiveSourceArtifact(
                    artifact_id="deferred-evidence",
                    origin="codex-session",
                    source_path="/data/deferred.jsonl",
                    source_index=0,
                    artifact_kind="deferred_hot_jsonl_capture",
                    classification_reason="deferred_hot_jsonl_capture",
                    support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                ),
            )
            conn.commit()

        snapshot = read_raw_failure_lifecycle(tmp_path / "source.db")
        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        ):
            info = _raw_failure_info()

        assert info["deferred_failures"] == snapshot.deferred == 1
        assert info["terminal_rejections"] == snapshot.terminal == 1
        assert info["unexplained_failures"] == snapshot.unexplained == 0

    def test_daemon_status_uses_the_lifecycle_sample_rows(self, tmp_path: Path) -> None:
        """Status must retain classified rows selected ahead of newer unexplained rows."""
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        with sqlite3.connect(source_db) as conn:
            conn.executemany(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                    acquired_at_ms, parse_error
                ) VALUES (?, ?, ?, ?, 0, ?, 0, ?, ?)
                """,
                [
                    (
                        f"raw-unexplained-{index}",
                        "codex-session",
                        f"native-{index}",
                        "/data/unexplained.jsonl",
                        bytes(index.to_bytes(32, "big")),
                        1_770_000_000_100 + index,
                        "newer unexplained failure",
                    )
                    for index in range(50)
                ]
                + [
                    (
                        "raw-terminal",
                        "unknown-export",
                        "terminal-native",
                        "/data/terminal.json",
                        bytes(32),
                        1_770_000_000_000,
                        "parsed raw payload produced no sessions",
                    )
                ],
            )
            upsert_raw_artifact(
                conn,
                "raw-terminal",
                ArchiveSourceArtifact(
                    artifact_id="terminal-evidence",
                    origin="unknown-export",
                    source_path="/data/terminal.json",
                    source_index=0,
                    artifact_kind="terminal_unsupported_shape",
                    classification_reason="terminal_unsupported_shape",
                    support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                ),
            )
            conn.commit()

        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        ):
            info = _raw_failure_info()

        samples = cast(list[RawFailureSample], info["samples"])
        assert len(samples) == 50
        assert any(sample.provider_hint == "unknown-export" and sample.lifecycle == "terminal" for sample in samples)

    def test_lifecycle_samples_are_sql_bounded_for_large_file_backed_failure_sets(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        with sqlite3.connect(source_db) as conn:
            conn.executemany(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, parse_error
                ) VALUES (?, 'codex-session', ?, '/data/large-failures.jsonl', ?, ?, 0, ?, 'bad input')
                """,
                [
                    (f"raw-{index}", f"native-{index}", index, bytes(32), 1_770_000_000_000 + index)
                    for index in range(200)
                ],
            )
            conn.commit()

        statements: list[str] = []

        def open_traced_readonly(path: Path) -> sqlite3.Connection:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr("polylogue.storage.raw_failure_lifecycle.open_readonly_connection", open_traced_readonly)
        snapshot = read_raw_failure_lifecycle(source_db, sample_limit=3)

        assert snapshot.parse_failures == snapshot.unexplained == 200
        assert len(snapshot.samples) == 3
        summary_queries = [statement for statement in statements if "GROUP BY f.origin" in statement]
        sample_queries = [statement for statement in statements if "FROM sampled" in statement]
        assert len(summary_queries) == 1
        assert len(sample_queries) == 1
        assert "ROW_NUMBER()" not in sample_queries[0]
        assert "NOT EXISTS" in sample_queries[0]
        assert "LIMIT 3" in sample_queries[0]

    def test_lifecycle_samples_are_sql_bounded_without_artifact_table(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        with sqlite3.connect(source_db) as conn:
            conn.executemany(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, parse_error
                ) VALUES (?, 'codex-session', ?, '/data/no-artifacts.jsonl', ?, ?, 0, ?, 'bad input')
                """,
                [
                    (f"raw-{index}", f"native-{index}", index, bytes(32), 1_770_000_000_000 + index)
                    for index in range(5)
                ],
            )
            conn.execute("DROP TABLE raw_artifacts")
            conn.commit()

        statements: list[str] = []

        def open_traced_readonly(path: Path) -> sqlite3.Connection:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr("polylogue.storage.raw_failure_lifecycle.open_readonly_connection", open_traced_readonly)
        snapshot = read_raw_failure_lifecycle(source_db, sample_limit=3)

        assert snapshot.parse_failures == snapshot.unexplained == 5
        assert snapshot.deferred == snapshot.terminal == 0
        assert len(snapshot.samples) == 3
        assert all(sample["artifact_kind"] is None and sample["support_status"] is None for sample in snapshot.samples)
        sample_queries = [statement for statement in statements if "ORDER BY acquired_at_ms DESC" in statement]
        assert len(sample_queries) == 1
        assert "LIMIT 3" in sample_queries[0]

    @pytest.mark.parametrize("source_state", ["missing", "malformed"])
    def test_status_fails_closed_when_source_lifecycle_is_unavailable(
        self,
        tmp_path: Path,
        source_state: str,
    ) -> None:
        """An index fallback cannot turn unavailable source evidence into zero failures."""
        source_db = tmp_path / "source.db"
        if source_state == "malformed":
            source_db.write_bytes(b"not a sqlite database")

        with patch("polylogue.daemon.status.archive_root", return_value=tmp_path):
            info = _raw_failure_info()
            root_info = raw_failure_info_for_root(tmp_path)

        for status in (info, root_info):
            assert status["raw_failure_lifecycle_available"] is False
            assert status["raw_failure_lifecycle_state"] == "unavailable"
            assert status["raw_failure_lifecycle_reason"]
            rendered = "\n".join(
                format_daemon_status_lines(
                    cast(
                        JSONDocument,
                        {
                            "raw_failure_lifecycle_available": status["raw_failure_lifecycle_available"],
                            "raw_failure_lifecycle_state": status["raw_failure_lifecycle_state"],
                            "raw_failure_lifecycle_reason": status["raw_failure_lifecycle_reason"],
                        },
                    )
                )
            )
            assert "unavailable" in rendered
            assert "no raw failures" not in rendered

    def test_status_reports_healthy_zero_failure_lifecycle_for_valid_source(self, tmp_path: Path) -> None:
        initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)

        with patch("polylogue.daemon.status.archive_root", return_value=tmp_path):
            info = raw_failure_info_for_root(tmp_path)

        assert info["raw_failure_lifecycle_available"] is True
        assert info["raw_failure_lifecycle_state"] == "healthy"
        assert info["parse_failures"] == 0
        assert info["validation_failures"] == 0
        assert info["unexplained_failures"] == 0

    def test_raw_failure_info_streams_lifecycle_counts_beyond_sample_cap(self, tmp_path: Path) -> None:
        """Lifecycle counts cover every failed raw without bulk-fetching rows."""
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        rows = []
        with sqlite3.connect(source_db) as conn:
            for index in range(120):
                raw_id = f"raw-{index}"
                rows.append(
                    (
                        raw_id,
                        "codex-session",
                        raw_id,
                        f"/data/{raw_id}.jsonl",
                        index.to_bytes(32, "big"),
                        128,
                        1_770_000_000_000 + index,
                        "captured JSONL payload ends before a complete record boundary",
                        "[]",
                    )
                )
            conn.executemany(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, blob_hash, blob_size,
                    acquired_at_ms, parse_error, detection_warnings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            for index, row in enumerate(rows):
                deferred = index % 2 == 0
                upsert_raw_artifact(
                    conn,
                    str(row[0]),
                    ArchiveSourceArtifact(
                        artifact_id=f"artifact-{index}",
                        origin="codex-session",
                        source_path=str(row[3]),
                        source_index=0,
                        artifact_kind="deferred_hot_jsonl_capture" if deferred else "terminal_corrupt_input",
                        classification_reason="raw-failure",
                        support_status=(
                            ArtifactSupportStatus.PARTIAL_DECODE if deferred else ArtifactSupportStatus.DECODE_FAILED
                        ),
                    ),
                )

        class GuardedCursor:
            def __init__(self, cursor: sqlite3.Cursor, sql: str) -> None:
                self._cursor = cursor
                self._sql = sql

            def fetchone(self) -> object:
                return self._cursor.fetchone()

            def fetchall(self) -> list[object]:
                normalized = " ".join(self._sql.split()).lower()
                if "from raw_sessions as r" in normalized and "order by acquired_at_ms desc" in normalized:
                    raise AssertionError("raw-failure lifecycle rows must stream instead of fetchall")
                return self._cursor.fetchall()

            def __iter__(self) -> object:
                normalized = " ".join(self._sql.split()).lower()
                if (
                    "from raw_sessions as r" in normalized
                    and "limit 50" not in normalized
                    and "r.raw_id in (" not in normalized
                ):
                    raise AssertionError("raw-failure lifecycle totals must aggregate in SQL")
                return iter(self._cursor)

            def __getattr__(self, name: str) -> object:
                return getattr(self._cursor, name)

        class GuardedConnection:
            def __init__(self, connection: sqlite3.Connection) -> None:
                self._connection = connection

            def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> GuardedCursor:
                return GuardedCursor(self._connection.execute(sql, parameters), sql)

            def close(self) -> None:
                self._connection.close()

        with (
            patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
            patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
            patch(
                "polylogue.daemon.status.open_readonly_connection",
                side_effect=lambda *_args, **_kwargs: GuardedConnection(sqlite3.connect(source_db)),
            ),
        ):
            info = _raw_failure_info()

        assert info["parse_failures"] == 120
        assert info["deferred_failures"] == 60
        assert info["terminal_rejections"] == 60
        assert info["unexplained_failures"] == 0
        assert len(cast(list[RawFailureSample], info["samples"])) == 50

    def test_raw_failure_info_empty_when_no_failures(self, tmp_path: Path) -> None:
        db = tmp_path / "index.db"
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

        with patch("polylogue.daemon.status._active_status_db_path", return_value=db):
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
