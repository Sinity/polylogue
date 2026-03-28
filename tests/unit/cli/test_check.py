"""Tests for the check CLI command."""

from __future__ import annotations

import json
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.health import HealthCheck, HealthReport, VerifyStatus
from polylogue.rendering.semantic_proof import (
    ProviderSemanticProof,
    SemanticConversationProof,
    SemanticMetricCheck,
    SemanticProofReport,
    SemanticProofSuiteReport,
)
from polylogue.schemas.verification import ArtifactProofReport, ProviderArtifactProof
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import ArtifactCohortSummary, ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider
from tests.infra.storage_records import DbFactory


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


def _extract_json(output: str) -> dict:
    """Extract JSON from CLI output, unwrapping the success envelope."""
    lines = output.strip().split("\n")
    # Find first line that starts with { and join all subsequent lines
    json_start = next((i for i, line in enumerate(lines) if line.strip().startswith("{")), None)
    if json_start is None:
        raise ValueError(f"No JSON found in output: {output}")
    json_str = "\n".join(lines[json_start:])
    data = json.loads(json_str)
    # Unwrap success envelope
    if isinstance(data, dict) and data.get("status") == "ok" and "result" in data:
        return data["result"]
    return data


def _make_semantic_report(*, critical: bool = False) -> SemanticProofSuiteReport:
    checks = [
        SemanticMetricCheck(
            metric="renderable_messages",
            status="critical_loss" if critical else "preserved",
            policy="canonical markdown must preserve every renderable message section",
            input_value=2,
            output_value=1 if critical else 2,
        ),
        SemanticMetricCheck(
            metric="thinking_semantics",
            status="declared_loss",
            policy="canonical markdown preserves display text but not typed thinking markers",
            input_value=1,
            output_value=0,
        ),
    ]
    canonical_report = SemanticProofReport(
        surface="canonical_markdown_v1",
        conversations=[
            SemanticConversationProof(
                conversation_id="conv-1",
                provider="chatgpt",
                surface="canonical_markdown_v1",
                input_facts={"renderable_messages": 2},
                output_facts={"message_sections": 1 if critical else 2},
                checks=checks,
            )
        ],
        provider_reports={
            "chatgpt": ProviderSemanticProof(
                provider="chatgpt",
                total_conversations=1,
                clean_conversations=0 if critical else 1,
                critical_conversations=1 if critical else 0,
                preserved_checks=0 if critical else 1,
                declared_loss_checks=1,
                critical_loss_checks=1 if critical else 0,
                metric_summary={
                    "renderable_messages": {
                        "preserved": 0 if critical else 1,
                        "declared_loss": 0,
                        "critical_loss": 1 if critical else 0,
                    },
                    "thinking_semantics": {
                        "preserved": 0,
                        "declared_loss": 1,
                        "critical_loss": 0,
                    },
                },
            )
        },
    )
    html_report = SemanticProofReport(
        surface="export_html_v1",
        conversations=[
            SemanticConversationProof(
                conversation_id="conv-1",
                provider="chatgpt",
                surface="export_html_v1",
                input_facts={"text_messages": 2},
                output_facts={"message_sections": 1 if critical else 2},
                checks=[
                    SemanticMetricCheck(
                        metric="text_messages",
                        status="critical_loss" if critical else "preserved",
                        policy="export_html_v1 must preserve visible message sections for text-bearing messages",
                        input_value=2,
                        output_value=1 if critical else 2,
                    ),
                    SemanticMetricCheck(
                        metric="branch_structure",
                        status="preserved",
                        policy="export_html_v1 must preserve visible branch groupings for branched messages",
                        input_value=0,
                        output_value=0,
                    ),
                ],
            )
        ],
        provider_reports={
            "chatgpt": ProviderSemanticProof(
                provider="chatgpt",
                total_conversations=1,
                clean_conversations=0 if critical else 1,
                critical_conversations=1 if critical else 0,
                preserved_checks=1 if critical else 2,
                declared_loss_checks=0,
                critical_loss_checks=1 if critical else 0,
                metric_summary={
                    "text_messages": {
                        "preserved": 0 if critical else 1,
                        "declared_loss": 0,
                        "critical_loss": 1 if critical else 0,
                    },
                    "branch_structure": {
                        "preserved": 1,
                        "declared_loss": 0,
                        "critical_loss": 0,
                    },
                },
            )
        },
    )
    return SemanticProofSuiteReport(
        surface_reports={
            "canonical_markdown_v1": canonical_report,
            "export_html_v1": html_report,
        },
    )


class TestHealthReportConstruction:
    """Tests for proper HealthReport instantiation."""

    def test_health_report_requires_summary(self):
        """HealthReport derives summary dict from check statuses."""
        checks = [
            HealthCheck("database", VerifyStatus.OK, summary="DB reachable"),
            HealthCheck("archive", VerifyStatus.WARNING, summary="Not found"),
        ]

        report = HealthReport(checks=checks)

        assert len(report.checks) == 2
        assert report.summary == {"ok": 1, "warning": 1, "error": 0}

    def test_health_report_summary_counts(self):
        """Summary should accurately reflect check status counts."""
        checks = [
            HealthCheck("check1", VerifyStatus.OK),
            HealthCheck("check2", VerifyStatus.OK),
            HealthCheck("check3", VerifyStatus.WARNING),
            HealthCheck("check4", VerifyStatus.ERROR),
        ]

        report = HealthReport(checks=checks)

        # Verify counts match
        assert report.summary["ok"] == 2
        assert report.summary["warning"] == 1
        assert report.summary["error"] == 1

    def test_health_report_to_dict_serialization(self):
        """HealthReport should serialize to dict with all required fields."""
        checks = [HealthCheck("test", VerifyStatus.OK, summary="OK")]
        report = HealthReport(checks=checks)

        data = report.to_dict()

        assert "checks" in data
        assert "summary" in data
        assert "timestamp" in data
        assert data["summary"] == {"ok": 1, "warning": 0, "error": 0}

    def test_health_report_empty_checks(self):
        """HealthReport with no checks should still have summary."""
        report = HealthReport(checks=[])

        assert len(report.checks) == 0
        assert report.summary == {"ok": 0, "warning": 0, "error": 0}


class TestCheckCommand:
    """Tests for polylogue check command."""

    def test_check_clean_database(self, db_path, cli_runner):
        """Check command succeeds on clean database with valid data."""
        factory = DbFactory(db_path)

        # Create valid conversation with messages
        factory.create_conversation(
            id="conv1",
            provider="chatgpt",
            title="Test Conversation",
            messages=[
                {"id": "m1", "role": "user", "text": "hello"},
                {"id": "m2", "role": "assistant", "text": "world"},
            ],
        )

        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0
        assert "ok" in result.output.lower() or "✓" in result.output

    def test_check_json_output(self, db_path, cli_runner):
        """Check --json flag produces valid JSON."""
        factory = DbFactory(db_path)

        factory.create_conversation(
            id="conv1",
            provider="claude-ai",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "check", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        data = _extract_json(result.output)
        assert "checks" in data
        assert "summary" in data
        assert isinstance(data["checks"], list)
        assert isinstance(data["summary"], dict)

        # Check summary has expected keys
        assert "ok" in data["summary"]
        assert "warning" in data["summary"]
        assert "error" in data["summary"]

    def test_check_detects_orphan_messages(self, db_path, cli_runner):
        """Check detects messages without conversations."""

        # Disable foreign key constraints temporarily to insert orphan message
        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, role, text,
                    content_hash, version
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "orphan-msg",
                    "non-existent-conv",
                    "user",
                    "orphaned text",
                    "abc123",
                    1,
                ),
            )
            conn.commit()  # Explicit commit to ensure orphan persists
            conn.execute("PRAGMA foreign_keys = ON")

        result = cli_runner.invoke(cli, ["--plain", "check", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the orphaned_messages check
        orphan_check = next(
            (c for c in data["checks"] if c["name"] == "orphaned_messages"),
            None,
        )
        assert orphan_check is not None
        assert orphan_check["status"] == "error"
        assert orphan_check["count"] == 1

        # Summary should show at least one error
        assert data["summary"]["error"] >= 1

    def test_check_verbose_output(self, db_path, cli_runner):
        """Check -v flag increases detail with provider breakdown."""
        factory = DbFactory(db_path)

        # Create conversations from multiple providers
        factory.create_conversation(
            id="conv1",
            provider="chatgpt",
            messages=[{"id": "m1", "role": "user", "text": "hello"}],
        )
        factory.create_conversation(
            id="conv2",
            provider="claude-ai",
            messages=[{"id": "m2", "role": "user", "text": "world"}],
        )

        # Run verify without verbose
        result_normal = cli_runner.invoke(cli, ["check"])
        assert result_normal.exit_code == 0

        # Run verify with verbose
        result_verbose = cli_runner.invoke(cli, ["check", "-v"])
        assert result_verbose.exit_code == 0

        # Verbose output should contain provider names for breakdowns
        # (provider_distribution check always has breakdown)
        assert "chatgpt" in result_verbose.output or "claude-ai" in result_verbose.output

    def test_check_detects_empty_conversations(self, db_path, cli_runner):
        """Check detects conversations with no messages (warning status)."""

        # Create a conversation with no messages
        with open_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, provider_name, provider_conversation_id,
                    title, created_at, updated_at, content_hash, version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "empty-conv",
                    "test",
                    "ext-empty",
                    "Empty Conversation",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "def456",
                    1,
                ),
            )
            conn.commit()  # Explicit commit to ensure conversation persists

        result = cli_runner.invoke(cli, ["--plain", "check", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the empty_conversations check
        empty_check = next(
            (c for c in data["checks"] if c["name"] == "empty_conversations"),
            None,
        )
        assert empty_check is not None
        assert empty_check["status"] == "warning"
        assert empty_check["count"] == 1

    def test_check_no_duplicate_conversation_ids(self, db_path, cli_runner):
        """Check duplicate_conversations check passes when there are no duplicates."""
        factory = DbFactory(db_path)

        # Create unique conversations (duplicates prevented by UNIQUE constraint)
        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test1"}],
        )
        factory.create_conversation(
            id="conv2",
            provider="test",
            messages=[{"id": "m2", "role": "user", "text": "test2"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "check", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the duplicate_conversations check
        dup_check = next(
            (c for c in data["checks"] if c["name"] == "duplicate_conversations"),
            None,
        )
        assert dup_check is not None
        assert dup_check["status"] == "ok"
        assert dup_check["count"] == 0  # No duplicates found

    def test_check_detects_fts_sync_issues(self, db_path, cli_runner):
        """Check detects FTS sync issues when FTS table is missing."""
        factory = DbFactory(db_path)

        # Create some messages
        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        # Manually drop FTS table to simulate desync
        with open_connection(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.commit()  # Explicit commit

        result = cli_runner.invoke(cli, ["--plain", "check", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the fts_sync check
        fts_check = next(
            (c for c in data["checks"] if c["name"] == "fts_sync"),
            None,
        )
        assert fts_check is not None
        # Should be warning due to missing FTS table
        assert fts_check["status"] == "warning"

    def test_check_plain_output_format(self, db_path, cli_runner):
        """Check --plain flag produces plain text output without colors."""
        factory = DbFactory(db_path)

        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "check"])
        assert result.exit_code == 0

        # Plain output should use OK/WARN/ERR instead of symbols
        assert "OK" in result.output or "ok" in result.output.lower()
        # Should not contain ANSI color codes
        assert "[green]" not in result.output
        assert "[red]" not in result.output

    def test_check_summary_counts(self, db_path, cli_runner):
        """Check summary shows correct counts of ok/warning/error checks."""
        factory = DbFactory(db_path)

        # Create valid data
        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0

        # Check summary line format
        assert "Summary:" in result.output
        assert "ok" in result.output
        assert "warnings" in result.output or "warning" in result.output
        assert "errors" in result.output or "error" in result.output

    def test_check_empty_database(self, db_path, cli_runner):
        """Check command succeeds on empty database."""
        # Don't create any data - just verify empty DB (db_path fixture ensures DB exists)

        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0

        # Should show healthy status checks (no integrity_check without --deep)
        assert "OK database" in result.output

    def test_check_sqlite_integrity_check(self, db_path, cli_runner):
        """Check includes SQLite integrity check when --deep is passed."""
        factory = DbFactory(db_path)

        factory.create_conversation(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "check", "--deep", "--json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the sqlite_integrity check
        integrity_check = next(
            (c for c in data["checks"] if c["name"] == "sqlite_integrity"),
            None,
        )
        assert integrity_check is not None
        assert integrity_check["status"] == "ok"
        assert integrity_check["detail"] == "ok"


# --- Merged from test_supplementary_coverage.py ---


class TestCheckCommandSupplementary:
    """Tests for check command edge cases."""

    # --- Flag validation: invalid combos rejected with correct error ---

    INVALID_FLAG_COMBOS = [
        (["check", "--vacuum"], "--vacuum requires --repair"),
        (["check", "--preview"], "--preview requires --repair"),
        (["check", "--schema-provider", "chatgpt"], "--schema-provider requires --schemas"),
        (["check", "--schema-record-limit", "100"], "--schema-record-limit requires --schemas"),
        (["check", "--schema-record-offset", "10"], "--schema-record-offset requires --schemas"),
        (["check", "--schema-quarantine-malformed"], "--schema-quarantine-malformed requires --schemas"),
        (["check", "--semantic-provider", "chatgpt"], "--semantic-provider requires --semantic-proof"),
        (["check", "--semantic-surface", "html"], "--semantic-surface requires --semantic-proof"),
        (["check", "--semantic-limit", "10"], "--semantic-limit requires --semantic-proof"),
        (["check", "--semantic-offset", "10"], "--semantic-offset requires --semantic-proof"),
        (["check", "--schemas", "--schema-samples", "0"], "--schema-samples must be a positive integer or 'all'"),
        (["check", "--schemas", "--schema-record-limit", "0"], "--schema-record-limit must be a positive integer"),
        (["check", "--schemas", "--schema-record-offset", "-1"], "--schema-record-offset must be >= 0"),
        (["check", "--semantic-proof", "--semantic-limit", "0"], "--semantic-limit must be a positive integer"),
        (["check", "--semantic-proof", "--semantic-offset", "-1"], "--semantic-offset must be >= 0"),
    ]

    @pytest.mark.parametrize("args,expected_error", INVALID_FLAG_COMBOS)
    def test_invalid_flag_combinations_rejected(self, cli_workspace, args, expected_error):
        """Flag dependencies and value constraints are enforced."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, args)
        assert result.exit_code != 0
        assert expected_error in result.output

    # --- Remaining non-repetitive tests ---

    def test_json_output_with_repair(self, cli_workspace):
        """--json with --repair includes repair results."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--json", "--repair", "--preview"])
        assert result.exit_code == 0
        envelope = json.loads(result.output.split("\n", 1)[-1] if "Plain" in result.output else result.output)
        data = envelope.get("result", envelope)
        assert "repairs" in data

    def test_repair_with_no_issues_shows_message(self, cli_workspace):
        """When repair finds no issues, should show 'No issues' message."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--repair"])
        assert result.exit_code == 0
        assert "No issues" in result.output or "Repaired" in result.output or "repair" in result.output.lower()

    def test_vacuum_with_repair(self, cli_workspace):
        """--vacuum with --repair should attempt VACUUM."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--repair", "--vacuum"])
        assert result.exit_code == 0
        assert "VACUUM" in result.output

    def test_json_output_with_repair_and_vacuum_is_machine_safe(self, cli_workspace):
        """`--json --repair --vacuum` should stay valid JSON."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "check", "--json", "--repair", "--preview", "--vacuum"])

        assert result.exit_code == 0
        envelope = json.loads(result.output)
        data = envelope.get("result", envelope)
        assert "repairs" in data
        assert data["vacuum"]["ok"] is True
        assert data["vacuum"]["preview"] is True

    def test_check_schemas_json_output(self, cli_workspace):
        """--schemas adds schema_verification block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli
        from polylogue.schemas.verification import ProviderSchemaVerification, SchemaVerificationReport

        fake_report = SchemaVerificationReport(
            providers={
                "chatgpt": ProviderSchemaVerification(
                    provider="chatgpt",
                    total_records=3,
                    valid_records=3,
                )
            },
            max_samples=None,
            total_records=3,
        )

        with patch(
            "polylogue.schemas.verification.verify_raw_corpus",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--json", "--schemas", "--schema-samples", "all"])

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "schema_verification" in data
        assert data["schema_verification"]["total_records"] == 3
        assert data["schema_verification"]["max_samples"] == "all"
        assert data["schema_verification"]["record_limit"] == "all"
        assert data["schema_verification"]["record_offset"] == 0

    def test_check_schemas_forwards_record_chunk_options(self, cli_workspace):
        """Chunking options are forwarded to verify_raw_corpus."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli
        from polylogue.schemas.verification import SchemaVerificationReport

        fake_report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
            record_limit=250,
            record_offset=500,
        )

        with patch(
            "polylogue.schemas.verification.verify_raw_corpus",
            return_value=fake_report,
        ) as mock_verify:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "check",
                    "--json",
                    "--schemas",
                    "--schema-provider",
                    "claude-code",
                    "--schema-samples",
                    "16",
                    "--schema-record-limit",
                    "250",
                    "--schema-record-offset",
                    "500",
                ],
            )

        assert result.exit_code == 0
        mock_verify.assert_called_once_with(
            providers=["claude-code"],
            max_samples=16,
            record_limit=250,
            record_offset=500,
            quarantine_malformed=False,
            progress_callback=ANY,
        )

    def test_check_schemas_forwards_quarantine_flag(self, cli_workspace):
        """Quarantine option is forwarded to verify_raw_corpus."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli
        from polylogue.schemas.verification import SchemaVerificationReport

        fake_report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
        )

        with patch(
            "polylogue.schemas.verification.verify_raw_corpus",
            return_value=fake_report,
        ) as mock_verify:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["--plain", "check", "--json", "--schemas", "--schema-quarantine-malformed"],
            )

        assert result.exit_code == 0
        mock_verify.assert_called_once_with(
            providers=None,
            max_samples=None,
            record_limit=None,
            record_offset=0,
            quarantine_malformed=True,
            progress_callback=ANY,
        )

    def test_check_proof_json_output(self, cli_workspace):
        """--proof adds artifact_proof block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = ArtifactProofReport(
            providers={
                "claude-code": ProviderArtifactProof(
                    provider="claude-code",
                    total_records=2,
                    recognized_non_parseable_records=1,
                    unsupported_parseable_records=1,
                    linked_sidecars=1,
                    subagent_streams=1,
                    streams_with_sidecars=1,
                    package_versions={"v7": 1},
                    element_kinds={"subagent_conversation_stream": 1},
                    resolution_reasons={"bundle_scope": 1},
                )
            },
            total_records=2,
        )

        with patch(
            "polylogue.schemas.verification.prove_raw_artifact_coverage",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--json", "--proof"])

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "artifact_proof" in data
        assert data["artifact_proof"]["total_records"] == 2
        assert data["artifact_proof"]["summary"]["linked_sidecars"] == 1
        assert data["artifact_proof"]["summary"]["unsupported_parseable_records"] == 1
        assert data["artifact_proof"]["summary"]["package_versions"] == {"v7": 1}

    def test_check_proof_plain_output(self, cli_workspace):
        """--proof renders the artifact proof summary in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = ArtifactProofReport(
            providers={
                "chatgpt": ProviderArtifactProof(
                    provider="chatgpt",
                    total_records=1,
                    contract_backed_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"conversation_document": 1},
                    resolution_reasons={"exact_structure": 1},
                )
            },
            total_records=1,
        )

        with patch(
            "polylogue.schemas.verification.prove_raw_artifact_coverage",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--proof"])

        assert result.exit_code == 0
        assert "Artifact proof:" in result.output
        assert "Resolved packages: v1=1" in result.output
        assert "Resolved elements: conversation_document=1" in result.output
        assert "Resolution reasons: exact_structure=1" in result.output
        assert "chatgpt: contract_backed=1" in result.output
        assert "packages: v1=1" in result.output

    def test_check_proof_forwards_artifact_scope(self, cli_workspace):
        """Artifact provider/limit/offset are forwarded to the proof workflow."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = ArtifactProofReport(providers={}, total_records=0)

        with patch(
            "polylogue.schemas.verification.prove_raw_artifact_coverage",
            return_value=fake_report,
        ) as mock_prove:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "check",
                    "--json",
                    "--proof",
                    "--artifact-provider",
                    "claude-code",
                    "--artifact-limit",
                    "25",
                    "--artifact-offset",
                    "50",
                ],
            )

        assert result.exit_code == 0
        mock_prove.assert_called_once_with(
            providers=["claude-code"],
            record_limit=25,
            record_offset=50,
        )

    def test_check_semantic_proof_json_output(self, cli_workspace):
        """--semantic-proof adds semantic_proof block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_semantic_report()

        with patch(
            "polylogue.rendering.semantic_proof.prove_semantic_surface_suite",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--json", "--semantic-proof"])

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "semantic_proof" in data
        assert data["semantic_proof"]["summary"]["surface_count"] == 2
        assert data["semantic_proof"]["summary"]["clean_surfaces"] == 2
        assert data["semantic_proof"]["summary"]["metric_summary"]["thinking_semantics"]["declared_loss"] == 1
        assert "export_html_v1" in data["semantic_proof"]["surfaces"]

    def test_check_semantic_proof_plain_output(self, cli_workspace):
        """--semantic-proof renders the semantic proof summary in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_semantic_report(critical=True)

        with patch(
            "polylogue.rendering.semantic_proof.prove_semantic_surface_suite",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--semantic-proof"])

        assert result.exit_code == 0
        assert "Semantic proof:" in result.output
        assert "critical=2" in result.output
        assert "renderable_messages(preserved=0, declared_loss=0, critical_loss=1)" in result.output
        assert "canonical_markdown_v1: conversations=1 clean=0 critical=1" in result.output
        assert "export_html_v1: conversations=1 clean=0 critical=1" in result.output
        assert "chatgpt: conversations=1 clean=0 critical=1" in result.output

    def test_check_semantic_proof_forwards_scope(self, cli_workspace):
        """Semantic provider/surface/limit/offset are forwarded to the proof workflow."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_semantic_report()

        with patch(
            "polylogue.rendering.semantic_proof.prove_semantic_surface_suite",
            return_value=fake_report,
        ) as mock_prove:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "check",
                    "--json",
                    "--semantic-proof",
                    "--semantic-provider",
                    "chatgpt",
                    "--semantic-surface",
                    "html",
                    "--semantic-limit",
                    "25",
                    "--semantic-offset",
                    "50",
                ],
            )

        assert result.exit_code == 0
        mock_prove.assert_called_once_with(
            providers=["chatgpt"],
            surfaces=["html"],
            record_limit=25,
            record_offset=50,
        )

<<<<<<< ours
||||||| base
    def test_check_roundtrip_proof_json_output(self, cli_workspace):
        """--roundtrip-proof adds roundtrip_proof block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_roundtrip_report()

        with patch(
            "polylogue.schemas.roundtrip_proof.prove_schema_evidence_roundtrip_suite",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--json", "--roundtrip-proof"])

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["roundtrip_proof"]["summary"]["provider_count"] == 1
        assert data["roundtrip_proof"]["summary"]["clean"] is True
        assert "chatgpt" in data["roundtrip_proof"]["providers"]

    def test_check_roundtrip_proof_plain_output(self, cli_workspace):
        """--roundtrip-proof renders the roundtrip proof summary in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_roundtrip_report(clean=False)

        with patch(
            "polylogue.schemas.roundtrip_proof.prove_schema_evidence_roundtrip_suite",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--roundtrip-proof"])

        assert result.exit_code == 0
        assert "Roundtrip proof:" in result.output
        assert "failed=1" in result.output
        assert "chatgpt: failed, package=v1" in result.output

    def test_check_roundtrip_proof_forwards_scope(self, cli_workspace):
        """Roundtrip provider/count are forwarded to the proof workflow."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_roundtrip_report()

        with patch(
            "polylogue.schemas.roundtrip_proof.prove_schema_evidence_roundtrip_suite",
            return_value=fake_report,
        ) as mock_roundtrip:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "check",
                    "--json",
                    "--roundtrip-proof",
                    "--roundtrip-provider",
                    "chatgpt",
                    "--roundtrip-count",
                    "2",
                ],
            )

        assert result.exit_code == 0
        mock_roundtrip.assert_called_once_with(providers=["chatgpt"], count=2)

=======
    def test_check_semantic_contracts_json_output(self, cli_workspace):
        """--semantic-contracts exposes the declared surface catalog in JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "check",
                "--json",
                "--semantic-contracts",
                "--semantic-surface",
                "html",
                "--semantic-surface",
                "stream_markdown",
            ],
        )

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["semantic_contracts"]["count"] == 2
        assert data["semantic_contracts"]["items"] == [
            {
                "surface": "export_html_v1",
                "category": "export",
                "aliases": ["html"],
                "export_format": "html",
                "stream_format": None,
                "contract_count": 10,
                "contracts": [
                    {
                        "metric": "title_metadata",
                        "mode": "preserve",
                        "policy": "export_html_v1 must preserve the display title",
                        "input_key": "title",
                        "output_key": "title",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "provider_identity",
                        "mode": "preserve",
                        "policy": "export_html_v1 must preserve provider identity at document level",
                        "input_key": "provider",
                        "output_key": "provider",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "date_metadata",
                        "mode": "presence",
                        "policy": "export_html_v1 must preserve conversation date presence at document level",
                        "input_key": "date",
                        "output_key": "has_date",
                        "input_transform": "presence_bool",
                        "output_transform": "bool",
                        "default_output": 0,
                    },
                    {
                        "metric": "text_messages",
                        "mode": "preserve",
                        "policy": "export_html_v1 must preserve visible message sections for text-bearing messages",
                        "input_key": "text_messages",
                        "output_key": "message_sections",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "role_sections",
                        "mode": "preserve",
                        "policy": "export_html_v1 must preserve visible role labels for text-bearing messages",
                        "input_key": "text_role_counts",
                        "output_key": "role_counts",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "timestamp_values",
                        "mode": "preserve",
                        "policy": "export_html_v1 must preserve visible message timestamps",
                        "input_key": "timestamped_text_messages",
                        "output_key": "timestamp_lines",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "branch_structure",
                        "mode": "preserve",
                        "policy": "export_html_v1 must preserve visible branch groupings for branched messages",
                        "input_key": "branch_messages",
                        "output_key": "branch_labels",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "attachment_semantics",
                        "mode": "declared_loss",
                        "policy": "export_html_v1 intentionally omits attachment payload semantics",
                        "input_key": "attachment_count",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "thinking_semantics",
                        "mode": "declared_loss",
                        "policy": "export_html_v1 preserves display text but not typed thinking markers",
                        "input_key": "thinking_messages",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "tool_semantics",
                        "mode": "declared_loss",
                        "policy": "export_html_v1 preserves display text but not typed tool markers",
                        "input_key": "tool_messages",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                ],
            },
            {
                "surface": "query_stream_markdown_v1",
                "category": "query_stream",
                "aliases": ["stream_markdown"],
                "export_format": None,
                "stream_format": "markdown",
                "contract_count": 11,
                "contracts": [
                    {
                        "metric": "title_metadata",
                        "mode": "preserve",
                        "policy": "query_stream_markdown_v1 must preserve the conversation title",
                        "input_key": "title",
                        "output_key": "title",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "provider_identity",
                        "mode": "preserve",
                        "policy": "query_stream_markdown_v1 must preserve provider identity in the stream header",
                        "input_key": "provider",
                        "output_key": "provider",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "date_metadata",
                        "mode": "presence",
                        "policy": "query_stream_markdown_v1 must preserve conversation date presence in the stream header",
                        "input_key": "date",
                        "output_key": "has_date",
                        "input_transform": "presence_bool",
                        "output_transform": "bool",
                        "default_output": 0,
                    },
                    {
                        "metric": "text_messages",
                        "mode": "preserve",
                        "policy": "query_stream_markdown_v1 must preserve one visible section per text-bearing message",
                        "input_key": "text_messages",
                        "output_key": "message_sections",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "role_sections",
                        "mode": "preserve",
                        "policy": "query_stream_markdown_v1 must preserve visible role headings for streamed messages",
                        "input_key": "text_role_counts",
                        "output_key": "role_counts",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "footer_count",
                        "mode": "preserve",
                        "policy": "query_stream_markdown_v1 must report the number of emitted messages honestly",
                        "input_key": "text_messages",
                        "output_key": "footer_count",
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "timestamp_values",
                        "mode": "declared_loss",
                        "policy": "query_stream_markdown_v1 intentionally omits per-message timestamps",
                        "input_key": "timestamped_text_messages",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "attachment_semantics",
                        "mode": "declared_loss",
                        "policy": "query_stream_markdown_v1 intentionally omits attachment semantics",
                        "input_key": "attachment_count",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "thinking_semantics",
                        "mode": "declared_loss",
                        "policy": "query_stream_markdown_v1 preserves display text but not typed thinking markers",
                        "input_key": "thinking_messages",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "tool_semantics",
                        "mode": "declared_loss",
                        "policy": "query_stream_markdown_v1 preserves display text but not typed tool markers",
                        "input_key": "tool_messages",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                    {
                        "metric": "branch_structure",
                        "mode": "declared_loss",
                        "policy": "query_stream_markdown_v1 intentionally omits explicit branch topology",
                        "input_key": "branch_messages",
                        "output_key": None,
                        "input_transform": "identity",
                        "output_transform": "identity",
                        "default_output": 0,
                    },
                ],
            },
        ]

    def test_check_semantic_contracts_plain_output(self, cli_workspace):
        """--semantic-contracts renders the declared surface catalog in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "check", "--semantic-contracts", "--semantic-surface", "read_all"])

        assert result.exit_code == 0
        assert "Semantic contracts: 9 surfaces" in result.output
        assert "query_summary_json_v1: category=query_summary; aliases=query_summary_json; contracts=7" in result.output
        assert "query_stream_markdown_v1: category=query_stream; aliases=stream_markdown; stream_format=markdown; contracts=11" in result.output
        assert "mcp_detail_json_v1: category=mcp; aliases=mcp_detail; contracts=13" in result.output
        assert "metrics=conversation_id:preserve, provider_identity:preserve" in result.output

    def test_check_roundtrip_proof_json_output(self, cli_workspace):
        """--roundtrip-proof adds roundtrip_proof block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_roundtrip_report()

        with patch(
            "polylogue.schemas.roundtrip_proof.prove_schema_evidence_roundtrip_suite",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--json", "--roundtrip-proof"])

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["roundtrip_proof"]["summary"]["provider_count"] == 1
        assert data["roundtrip_proof"]["summary"]["clean"] is True
        assert "chatgpt" in data["roundtrip_proof"]["providers"]

    def test_check_roundtrip_proof_plain_output(self, cli_workspace):
        """--roundtrip-proof renders the roundtrip proof summary in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_roundtrip_report(clean=False)

        with patch(
            "polylogue.schemas.roundtrip_proof.prove_schema_evidence_roundtrip_suite",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--roundtrip-proof"])

        assert result.exit_code == 0
        assert "Roundtrip proof:" in result.output
        assert "failed=1" in result.output
        assert "chatgpt: failed, package=v1" in result.output

    def test_check_roundtrip_proof_forwards_scope(self, cli_workspace):
        """Roundtrip provider/count are forwarded to the proof workflow."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = _make_roundtrip_report()

        with patch(
            "polylogue.schemas.roundtrip_proof.prove_schema_evidence_roundtrip_suite",
            return_value=fake_report,
        ) as mock_roundtrip:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "check",
                    "--json",
                    "--roundtrip-proof",
                    "--roundtrip-provider",
                    "chatgpt",
                    "--roundtrip-count",
                    "2",
                ],
            )

        assert result.exit_code == 0
        mock_roundtrip.assert_called_once_with(providers=["chatgpt"], count=2)

>>>>>>> theirs
    def test_check_artifacts_json_output(self, cli_workspace):
        """--artifacts adds artifact_observations rows to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_rows = [
            ArtifactObservationRecord(
                observation_id="obs-1",
                raw_id="raw-1",
                provider_name="chatgpt",
                payload_provider=Provider.CHATGPT,
                source_name="chatgpt",
                source_path="/tmp/chatgpt.json",
                source_index=0,
                file_mtime=None,
                wire_format="json",
                artifact_kind="conversation_document",
                classification_reason="conversation-bearing document",
                parse_as_conversation=True,
                schema_eligible=True,
                support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
                malformed_jsonl_lines=0,
                decode_error=None,
                bundle_scope="chatgpt",
                cohort_id="cohort-1",
                resolved_package_version="v1",
                resolved_element_kind="conversation_document",
                resolution_reason="exact_structure",
                link_group_key=None,
                sidecar_agent_type=None,
                first_observed_at="2026-03-21T00:00:00+00:00",
                last_observed_at="2026-03-21T00:00:00+00:00",
            )
        ]

        with patch(
            "polylogue.schemas.verification.list_artifact_observation_rows",
            return_value=fake_rows,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "check",
                    "--json",
                    "--artifacts",
                    "--artifact-provider",
                    "chatgpt",
                    "--artifact-status",
                    "supported_parseable",
                    "--artifact-kind",
                    "conversation_document",
                ],
            )

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert data["artifact_observations"]["count"] == 1
        assert data["artifact_observations"]["items"][0]["support_status"] == "supported_parseable"

    def test_check_cohorts_plain_output(self, cli_workspace):
        """--cohorts renders durable cohort summaries in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_rows = [
            ArtifactCohortSummary(
                provider_name="claude-code",
                payload_provider=Provider.CLAUDE_CODE,
                artifact_kind="agent_sidecar_meta",
                support_status=ArtifactSupportStatus.RECOGNIZED_UNPARSED,
                cohort_id="cohort-sidecar",
                observation_count=1,
                unique_raw_ids=1,
                first_observed_at="2026-03-21T00:00:00+00:00",
                last_observed_at="2026-03-21T00:00:00+00:00",
                bundle_scope_count=1,
                sample_source_paths=["/tmp/subagents/agent-a123.meta.json"],
                resolved_package_version=None,
                resolved_element_kind=None,
                resolution_reason=None,
                link_group_count=1,
                linked_sidecar_count=1,
            )
        ]

        with patch(
            "polylogue.schemas.verification.list_artifact_cohort_rows",
            return_value=fake_rows,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "check", "--cohorts"])

        assert result.exit_code == 0
        assert "Artifact cohorts: 1 cohorts" in result.output
        assert "claude-code agent_sidecar_meta recognized_unparsed count=1" in result.output
