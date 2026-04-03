"""Tests for the check CLI command."""

from __future__ import annotations

import json
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.health import HealthCheck, HealthReport, VerifyStatus
from polylogue.schemas.operator_models import (
    ArtifactCohortListResult,
    ArtifactObservationListResult,
    ArtifactProofResult,
)
from polylogue.schemas.verification_models import (
    ArtifactProofReport,
    ProviderArtifactProof,
    ProviderSchemaVerification,
    SchemaVerificationReport,
)
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.state_views import ArtifactCohortSummary
from polylogue.storage.store import ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider
from tests.infra.storage_records import ConversationBuilder, DbFactory


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


def test_check_records_scoped_maintenance_preview(cli_workspace, cli_runner):
    from polylogue.storage.session_product_rebuild import rebuild_session_products_sync

    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-check-products")
        .provider("claude-code")
        .title("Scoped Check Repair")
        .add_message("u1", role="user", text="Plan the cleanup")
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        conn.execute("DELETE FROM session_profiles")
        conn.commit()

    result = cli_runner.invoke(
        cli,
        [
            "doctor",
            "--json",
            "--repair",
            "--preview",
            "--target",
            "session_products",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["maintenance"]["targets"] == ["session_products"]
    assert payload["maintenance"]["items"][0]["name"] == "session_products"
    assert payload["maintenance"]["items"][0]["repaired_count"] > 0


def test_check_records_scoped_maintenance_apply(cli_workspace, cli_runner):
    from polylogue.storage.session_product_rebuild import rebuild_session_products_sync

    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-check-products-apply")
        .provider("claude-code")
        .title("Scoped Check Repair Apply")
        .add_message("u1", role="user", text="Repair the durable products")
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        conn.execute("DELETE FROM session_profiles")
        conn.commit()

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "doctor",
            "--json",
            "--repair",
            "--target",
            "session_products",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["maintenance"]["targets"] == ["session_products"]
    assert payload["maintenance"]["items"][0]["name"] == "session_products"
    assert payload["maintenance"]["items"][0]["success"] is True


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

        result = cli_runner.invoke(cli, ["doctor"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--json"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--json"])
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
        result_normal = cli_runner.invoke(cli, ["doctor"])
        assert result_normal.exit_code == 0

        # Run verify with verbose
        result_verbose = cli_runner.invoke(cli, ["doctor", "-v"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--json"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--json"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--json"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor"])
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

        result = cli_runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0

        # Check summary line format
        assert "Summary:" in result.output
        assert "ok" in result.output
        assert "warnings" in result.output or "warning" in result.output
        assert "errors" in result.output or "error" in result.output

    def test_check_empty_database(self, db_path, cli_runner):
        """Check command succeeds on empty database."""
        # Don't create any data - just verify empty DB (db_path fixture ensures DB exists)

        result = cli_runner.invoke(cli, ["doctor"])
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

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--deep", "--json"])
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
        (["doctor", "--vacuum"], "--vacuum requires --repair or --cleanup"),
        (["doctor", "--preview"], "--preview requires --repair or --cleanup"),
        (["doctor", "--schema-provider", "chatgpt"], "--schema-provider requires --schemas"),
        (["doctor", "--schema-record-limit", "100"], "--schema-record-limit requires --schemas"),
        (["doctor", "--schema-record-offset", "10"], "--schema-record-offset requires --schemas"),
        (["doctor", "--schema-quarantine-malformed"], "--schema-quarantine-malformed requires --schemas"),
        (["doctor", "--schemas", "--schema-samples", "0"], "--schema-samples must be a positive integer or 'all'"),
        (["doctor", "--schemas", "--schema-record-limit", "0"], "--schema-record-limit must be a positive integer"),
        (["doctor", "--schemas", "--schema-record-offset", "-1"], "--schema-record-offset must be >= 0"),
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
        """--json with --repair includes maintenance results."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json", "--repair", "--preview"])
        assert result.exit_code == 0
        envelope = json.loads(result.output.split("\n", 1)[-1] if "Plain" in result.output else result.output)
        data = envelope.get("result", envelope)
        assert "maintenance" in data

    def test_repair_with_no_issues_shows_message(self, cli_workspace):
        """When repair finds no issues, should show a maintenance status message."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--repair"])
        assert result.exit_code == 0
        assert "No selected maintenance work" in result.output or "Changed" in result.output or "maintenance" in result.output.lower()

    def test_vacuum_with_repair(self, cli_workspace):
        """--vacuum with --repair should attempt VACUUM."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--repair", "--vacuum"])
        assert result.exit_code == 0
        assert "VACUUM" in result.output

    def test_json_output_with_repair_and_vacuum_is_machine_safe(self, cli_workspace):
        """`--json --repair --vacuum` should stay valid JSON."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "doctor", "--json", "--repair", "--preview", "--vacuum"])

        assert result.exit_code == 0
        envelope = json.loads(result.output)
        data = envelope.get("result", envelope)
        assert "maintenance" in data
        assert data["vacuum"]["ok"] is True
        assert data["vacuum"]["preview"] is True

    def test_check_schemas_json_output(self, cli_workspace):
        """--schemas adds schema_verification block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

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
            "polylogue.cli.check_workflow.run_schema_verification",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--json", "--schemas", "--schema-samples", "all"])

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

        fake_report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
            record_limit=250,
            record_offset=500,
        )

        with patch(
            "polylogue.cli.check_workflow.run_schema_verification",
            return_value=fake_report,
        ) as mock_verify:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "doctor",
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
        request = mock_verify.call_args.args[0]
        assert request.providers == ["claude-code"]
        assert request.max_samples == 16
        assert request.record_limit == 250
        assert request.record_offset == 500
        assert request.quarantine_malformed is False
        assert request.progress_callback is not None
        assert mock_verify.call_args.kwargs["db_path"] == ANY

    def test_check_schemas_forwards_quarantine_flag(self, cli_workspace):
        """Quarantine option is forwarded to verify_raw_corpus."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
        )

        with patch(
            "polylogue.cli.check_workflow.run_schema_verification",
            return_value=fake_report,
        ) as mock_verify:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["--plain", "doctor", "--json", "--schemas", "--schema-quarantine-malformed"],
            )

        assert result.exit_code == 0
        request = mock_verify.call_args.args[0]
        assert request.providers is None
        assert request.max_samples is None
        assert request.record_limit is None
        assert request.record_offset == 0
        assert request.quarantine_malformed is True
        assert request.progress_callback is not None

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
            "polylogue.cli.check_workflow.run_artifact_proof",
            return_value=ArtifactProofResult(report=fake_report),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--json", "--proof"])

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
            "polylogue.cli.check_workflow.run_artifact_proof",
            return_value=ArtifactProofResult(report=fake_report),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--proof"])

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
            "polylogue.cli.check_workflow.run_artifact_proof",
            return_value=ArtifactProofResult(report=fake_report),
        ) as mock_prove:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "doctor",
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
        request = mock_prove.call_args.args[0]
        assert request.providers == ["claude-code"]
        assert request.record_limit == 25
        assert request.record_offset == 50

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
            "polylogue.cli.check_workflow.list_artifact_observations",
            return_value=ArtifactObservationListResult(rows=fake_rows),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "doctor",
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
            "polylogue.cli.check_workflow.list_artifact_cohorts",
            return_value=ArtifactCohortListResult(rows=fake_rows),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--cohorts"])

        assert result.exit_code == 0
        assert "Artifact cohorts: 1 cohorts" in result.output
        assert "claude-code agent_sidecar_meta recognized_unparsed count=1" in result.output
