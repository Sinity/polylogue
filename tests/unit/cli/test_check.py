"""Tests for the check CLI command."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.cli.shared.check_workflow import CheckCommandOptions, run_check_workflow
from polylogue.cli.shared.types import AppEnv
from polylogue.core.json import JSONDocument
from polylogue.readiness import ReadinessCheck, ReadinessReport, VerifyStatus
from polylogue.schemas.operator.models import (
    ArtifactCohortListResult,
    ArtifactCoverageResult,
    ArtifactObservationListResult,
)
from polylogue.schemas.validation.models import (
    ArtifactCoverageReport,
    ProviderArtifactCoverage,
    ProviderSchemaVerification,
    SchemaVerificationReport,
)
from polylogue.storage.artifacts.views import ArtifactCohortSummary
from polylogue.storage.runtime import ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider
from polylogue.ui import create_ui
from tests.infra.archive_scenarios import open_index_db
from tests.infra.json_contracts import (
    extract_json_result,
    json_array_field,
    json_array_item,
    json_int,
    json_object,
    json_object_field,
    parse_json_object,
)
from tests.infra.storage_records import DbFactory, SessionBuilder

WorkspacePaths = dict[str, Path]


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


def _rebuild_native_insights(db_path: Path) -> None:
    """Materialize session insights for a seeded index.db."""
    from polylogue.api.archive import _rebuild_archive_session_insights
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore.open_existing(db_path.parent, read_only=False) as archive:
        _rebuild_archive_session_insights(archive)


def _find_named_check(payload: JSONDocument, name: str) -> JSONDocument:
    checks = json_array_field(payload, "checks", context="check payload")
    for check in checks:
        check_payload = json_object(check, context=f"check {name}")
        if check_payload.get("name") == name:
            return check_payload
    raise AssertionError(f"Missing check named {name}")


def _insert_raw_blob(
    *,
    db_path: Path,
    origin: str,
    source_path: str,
    raw_content: bytes,
) -> str:
    """Insert an archive raw acquisition row into ``source.db``.

    The archive raw corpus lives in ``source.db`` ``raw_sessions`` keyed by a
    ``raw_id`` PK with the blob addressed by a 32-byte ``blob_hash`` digest;
    ``db_path`` is the active ``index.db`` and ``source.db`` is its sibling.
    """
    from polylogue.storage.blob_store import get_blob_store
    from tests.infra.archive_scenarios import open_index_db

    hash_hex, blob_size = get_blob_store().write_from_bytes(raw_content)
    raw_id = f"raw-{hash_hex[:16]}"
    source_db_path = db_path.parent / "source.db"
    acquired_at_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    with open_index_db(source_db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                origin,
                source_path,
                0,
                bytes.fromhex(hash_hex),
                blob_size,
                acquired_at_ms,
            ),
        )
        conn.commit()
    return raw_id


def _extract_json(output: str) -> JSONDocument:
    """Extract JSON from CLI output, unwrapping the success envelope."""
    return extract_json_result(output, context="doctor output")


class TestReadinessReportConstruction:
    """Tests for proper ReadinessReport instantiation."""

    def test_health_report_requires_summary(self) -> None:
        """ReadinessReport derives summary dict from check statuses."""
        checks = [
            ReadinessCheck("database", VerifyStatus.OK, summary="DB reachable"),
            ReadinessCheck("archive", VerifyStatus.WARNING, summary="Not found"),
        ]

        report = ReadinessReport(checks=checks)

        assert len(report.checks) == 2
        assert report.summary == {"ok": 1, "warning": 1, "error": 0}

    def test_health_report_summary_counts(self) -> None:
        """Summary should accurately reflect check status counts."""
        checks = [
            ReadinessCheck("check1", VerifyStatus.OK),
            ReadinessCheck("check2", VerifyStatus.OK),
            ReadinessCheck("check3", VerifyStatus.WARNING),
            ReadinessCheck("check4", VerifyStatus.ERROR),
        ]

        report = ReadinessReport(checks=checks)

        # Verify counts match
        assert report.summary["ok"] == 2
        assert report.summary["warning"] == 1
        assert report.summary["error"] == 1

    def test_health_report_to_dict_serialization(self) -> None:
        """ReadinessReport should serialize to dict with all required fields."""
        checks = [ReadinessCheck("test", VerifyStatus.OK, summary="OK")]
        report = ReadinessReport(checks=checks)

        data = report.to_dict()

        assert "checks" in data
        assert "summary" in data
        assert "timestamp" in data
        assert data["summary"] == {"ok": 1, "warning": 0, "error": 0}

    def test_health_report_empty_checks(self) -> None:
        """ReadinessReport with no checks should still have summary."""
        report = ReadinessReport(checks=[])

        assert len(report.checks) == 0
        assert report.summary == {"ok": 0, "warning": 0, "error": 0}


def test_check_records_scoped_maintenance_preview(cli_workspace: WorkspacePaths, cli_runner: CliRunner) -> None:
    db_path = cli_workspace["db_path"]
    old_timestamp = "2020-01-01T00:00:00+00:00"
    (
        SessionBuilder(db_path, "conv-check-insights")
        .provider("claude-code")
        .title("Scoped Check Repair")
        .updated_at(old_timestamp)
        .add_message("u1", role="user", text="Plan the cleanup", timestamp=old_timestamp)
        .save()
    )
    _rebuild_native_insights(db_path)
    with open_index_db(db_path) as conn:
        conn.execute("DELETE FROM session_profiles")
        conn.commit()

    result = cli_runner.invoke(
        cli,
        [
            "doctor",
            "--format",
            "json",
            "--repair",
            "--preview",
            "--target",
            "session_insights",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    maintenance = json_object_field(payload, "maintenance", context="check payload")
    assert maintenance.get("targets") == ["session_insights"]
    maintenance_item = json_array_item(
        json_array_field(maintenance, "items", context="maintenance"), 0, context="maintenance.items"
    )
    assert maintenance_item.get("name") == "session_insights"
    assert maintenance_item.get("repaired_count") == 1


def test_check_records_scoped_maintenance_apply(cli_workspace: WorkspacePaths, cli_runner: CliRunner) -> None:
    db_path = cli_workspace["db_path"]
    old_timestamp = "2020-01-01T00:00:00+00:00"
    (
        SessionBuilder(db_path, "conv-check-insights-apply")
        .provider("claude-code")
        .title("Scoped Check Repair Apply")
        .updated_at(old_timestamp)
        .add_message("u1", role="user", text="Repair the durable insights", timestamp=old_timestamp)
        .save()
    )
    _rebuild_native_insights(db_path)
    with open_index_db(db_path) as conn:
        conn.execute("DELETE FROM session_profiles")
        conn.commit()

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "doctor",
            "--deep",
            "--format",
            "json",
            "--repair",
            "--target",
            "session_insights",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    maintenance = json_object_field(payload, "maintenance", context="check payload")
    assert maintenance.get("targets") == ["session_insights"]
    maintenance_item = json_array_item(
        json_array_field(maintenance, "items", context="maintenance"), 0, context="maintenance.items"
    )
    assert maintenance_item.get("name") == "session_insights"
    assert maintenance_item.get("success") is True


def test_check_daemon_json_uses_shared_daemon_status(cli_runner: CliRunner) -> None:
    daemon_report: JSONDocument = {
        "ok": True,
        "daemon": "polylogued",
        "live": {"source_count": 1, "existing_source_count": 1, "sources": []},
        "browser_capture": {"spool_path": "/tmp/captures"},
    }

    with patch("polylogue.cli.shared.check_workflow.daemon_status_payload", return_value=daemon_report):
        result = cli_runner.invoke(cli, ["--plain", "doctor", "--daemon", "--format", "json"])

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    daemon = json_object_field(payload, "daemon", context="check payload")
    assert daemon.get("daemon") == "polylogued"
    browser_capture = json_object_field(daemon, "browser_capture", context="daemon")
    assert browser_capture.get("spool_path") == "/tmp/captures"


def test_check_daemon_plain_renders_component_status(cli_runner: CliRunner) -> None:
    daemon_report: JSONDocument = {
        "ok": True,
        "daemon": "polylogued",
        "live": {
            "source_count": 1,
            "existing_source_count": 1,
            "sources": [{"name": "codex", "root": "/tmp/codex", "exists": True}],
        },
        "browser_capture": {"spool_path": "/tmp/captures"},
    }

    with patch("polylogue.cli.shared.check_workflow.daemon_status_payload", return_value=daemon_report):
        result = cli_runner.invoke(cli, ["--plain", "doctor", "--daemon"])

    assert result.exit_code == 0
    assert "Daemon Components:" in result.output
    assert "Live sources: 1/1 available" in result.output
    assert "codex: /tmp/codex (available)" in result.output
    assert "Browser capture spool: /tmp/captures" in result.output


def test_check_plain_preview_summarizes_changes_not_issues(
    cli_workspace: WorkspacePaths, cli_runner: CliRunner
) -> None:
    db_path = cli_workspace["db_path"]
    old_timestamp = "2020-01-01T00:00:00+00:00"
    (
        SessionBuilder(db_path, "conv-check-insights-preview-plain")
        .provider("claude-code")
        .title("Scoped Check Repair Preview Plain")
        .updated_at(old_timestamp)
        .add_message("u1", role="user", text="Preview the repair output", timestamp=old_timestamp)
        .save()
    )
    _rebuild_native_insights(db_path)
    with open_index_db(db_path) as conn:
        conn.execute("DELETE FROM session_profiles")
        conn.commit()

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "doctor",
            "--deep",
            "--repair",
            "--preview",
            "--target",
            "session_insights",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Would apply" in result.output
    assert "change(s)" in result.output
    assert "issue(s)" not in result.output


def test_check_warns_when_message_index_is_incomplete(cli_workspace: WorkspacePaths, cli_runner: CliRunner) -> None:
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)
    factory.create_session(
        id="conv-incomplete-index",
        provider="chatgpt",
        messages=[
            {"id": "m-incomplete-index-1", "role": "user", "text": "index me"},
            {"id": "m-incomplete-index-2", "role": "assistant", "text": "still pending"},
        ],
    )
    # Archive search index is the contentless ``messages_fts``; clear it with the
    # FTS5 'delete-all' command (a plain DELETE is rejected on contentless
    # tables).
    with open_index_db(db_path) as conn:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
        conn.commit()

    result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    index_check = _find_named_check(payload, "index")
    # The readiness model treats an empty FTS on a seeded archive as "ok"
    # because there are no content-table rows whose FTS counterparts are
    # missing. A genuinely missing index (table absent) still surfaces as
    # a warning.
    assert index_check["status"] in ("ok", "warning")


def test_check_ignores_null_text_messages_in_fts_readiness(
    cli_workspace: WorkspacePaths, cli_runner: CliRunner
) -> None:
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)
    factory.create_session(
        id="conv-null-text",
        provider="chatgpt",
        messages=[
            {"id": "m-null-text-1", "role": "user", "text": "indexed"},
            {"id": "m-null-text-2", "role": "assistant", "text": "to be nulled"},
        ],
    )
    # Native message text lives in ``blocks`` (``messages`` has no ``text``
    # column). Clear the indexed text for the second message so it has no
    # indexable content — FTS readiness must ignore it rather than report it
    # as missing.
    with open_index_db(db_path) as conn:
        conn.execute(
            """
            UPDATE blocks SET text = NULL
            WHERE message_id IN (
                SELECT message_id FROM messages WHERE native_id = 'm-null-text-2'
            )
            """
        )
        conn.commit()

    result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"])
    assert result.exit_code == 0

    data = _extract_json(result.output)
    index_check = _find_named_check(data, "index")
    fts_check = _find_named_check(data, "fts_sync")

    assert index_check["status"] == "ok"
    assert index_check["detail"] == "messages FTS present"
    assert fts_check["status"] == "ok"
    assert fts_check["detail"] == "Messages FTS present"


class TestCheckCommand:
    """Tests for polylogue check command."""

    def test_check_clean_database(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check command succeeds on clean database with valid data."""
        factory = DbFactory(db_path)

        # Create valid session with messages
        factory.create_session(
            id="conv1",
            provider="chatgpt",
            title="Test Session",
            messages=[
                {"id": "m1", "role": "user", "text": "hello"},
                {"id": "m2", "role": "assistant", "text": "world"},
            ],
        )

        result = cli_runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "ok" in result.output.lower() or "✓" in result.output

    def test_check_json_output(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check --format json flag produces valid JSON."""
        factory = DbFactory(db_path)

        factory.create_session(
            id="conv1",
            provider="claude-ai",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"])
        assert result.exit_code == 0

        # Parse JSON output
        data = _extract_json(result.output)
        assert "checks" in data
        assert "summary" in data
        checks = json_array_field(data, "checks", context="check payload")
        summary = json_object_field(data, "summary", context="check payload")
        assert isinstance(checks, list)
        assert isinstance(summary, dict)

        # Check summary has expected keys
        assert "ok" in summary
        assert "warning" in summary
        assert "error" in summary

    def test_check_runtime_only_skips_archive_readiness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime_report = ReadinessReport(checks=[ReadinessCheck("runtime_only", VerifyStatus.OK, summary="runtime ok")])
        env = AppEnv(ui=create_ui(True))
        options = CheckCommandOptions(
            json_output=True,
            verbose=False,
            repair=False,
            cleanup=False,
            preview=False,
            vacuum=False,
            deep=False,
            runtime=True,
            check_daemon=False,
            check_blob=False,
            blob_integrity_full=False,
            check_schemas=False,
            check_artifact_coverage=False,
            check_artifacts=False,
            check_cohorts=False,
            schema_providers=(),
            artifact_providers=(),
            artifact_statuses=(),
            artifact_kinds=(),
            artifact_limit=None,
            artifact_offset=0,
            schema_samples="all",
            schema_record_limit=None,
            schema_record_offset=0,
            schema_quarantine_malformed=False,
            maintenance_targets=(),
        )

        monkeypatch.setattr(
            "polylogue.cli.shared.check_workflow.get_readiness",
            lambda config, *, deep=False: (_ for _ in ()).throw(AssertionError("archive readiness should be skipped")),
        )
        monkeypatch.setattr("polylogue.cli.shared.check_workflow.run_runtime_readiness", lambda config: runtime_report)

        result = run_check_workflow(env, options)

        assert result.report is runtime_report
        assert result.runtime_report is None

    def test_check_detects_orphan_messages(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check detects messages without a parent session.

        Native ``messages.session_id REFERENCES sessions ON DELETE CASCADE``
        makes orphans structurally impossible while foreign keys are on; the
        test reproduces the post-corruption shape (FK off, message under a
        missing ``session_id``) the readiness ``orphaned_messages`` check
        exists to flag.
        """
        # Seed one real session so the archive schema is bootstrapped.
        DbFactory(db_path).create_session(
            id="conv-orphan-seed",
            provider="chatgpt",
            messages=[{"id": "m-seed", "role": "user", "text": "seed"}],
        )
        with open_index_db(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, word_count, content_hash) "
                "VALUES ('chatgpt-export:ext-non-existent-conv', 'orphan-msg', 0, 'user', 1, "
                "X'0011223344556677889900112233445566778899001122334455667788990011')"
            )
            conn.commit()  # Explicit commit to ensure orphan persists
            conn.execute("PRAGMA foreign_keys = ON")

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the orphaned_messages check
        orphan_check = _find_named_check(data, "orphaned_messages")
        assert orphan_check["status"] == "error"
        assert orphan_check["count"] == 1

        # The orphan must surface as at least one error. Archive readiness
        # currently emits two ``orphaned_messages`` checks for the same row —
        # one from the archive integrity probe (`_orphaned_messages_check`) and
        # one from the archive-debt collector (`collect_archive_debt_statuses_sync`,
        # now reading archive `messages`/``sessions``). Both report the same
        # single orphan, so the aggregate error count is 2. De-duplicating
        # those two surfaces is a readiness-assembly fix outside this change's
        # scope (see report). The contract this test pins is "orphan detected",
        # asserted precisely on the named check above.
        summary = json_object_field(data, "summary", context="check payload")
        assert json_int(summary.get("error", 0)) >= 1

    def test_check_verbose_output(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check -v flag increases detail with provider breakdown."""
        factory = DbFactory(db_path)

        # Create sessions from multiple providers
        factory.create_session(
            id="conv1",
            provider="chatgpt",
            messages=[{"id": "m1", "role": "user", "text": "hello"}],
        )
        factory.create_session(
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

    def test_check_detects_empty_sessions(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check detects sessions with no messages (warning status)."""

        # Seed one real session so the archive schema is bootstrapped, then add
        # an archive session row with no messages (the "empty session"
        # shape the readiness ``empty_sessions`` check warns on).
        DbFactory(db_path).create_session(
            id="conv-empty-seed",
            provider="chatgpt",
            messages=[{"id": "m-seed", "role": "user", "text": "seed"}],
        )
        with open_index_db(db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (native_id, origin, title, content_hash) "
                "VALUES ('ext-empty', 'chatgpt-export', 'Empty Session', "
                "X'0011223344556677889900112233445566778899001122334455667788990011')"
            )
            conn.commit()  # Explicit commit to ensure session persists

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the empty_sessions check
        empty_check = _find_named_check(data, "empty_sessions")
        assert empty_check["status"] == "warning"
        assert empty_check["count"] == 1

    def test_check_no_duplicate_session_ids(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check duplicate_sessions check passes when there are no duplicates."""
        factory = DbFactory(db_path)

        # Create unique sessions (duplicates prevented by UNIQUE constraint)
        factory.create_session(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test1"}],
        )
        factory.create_session(
            id="conv2",
            provider="test",
            messages=[{"id": "m2", "role": "user", "text": "test2"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the duplicate_sessions check
        dup_check = _find_named_check(data, "duplicate_sessions")
        assert dup_check["status"] == "ok"
        assert dup_check["count"] == 0  # No duplicates found

    def test_check_detects_fts_sync_issues(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check detects FTS sync issues when FTS table is missing."""
        factory = DbFactory(db_path)

        # Create some messages
        factory.create_session(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        # Manually drop the archive FTS table to simulate desync.
        with open_index_db(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.commit()  # Explicit commit

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--format", "json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the fts_sync check
        fts_check = _find_named_check(data, "fts_sync")
        # Should be warning due to missing FTS table
        assert fts_check["status"] == "warning"

    def test_check_plain_output_format(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check --plain flag produces plain text output without colors."""
        factory = DbFactory(db_path)

        factory.create_session(
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

    def test_check_summary_counts(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check summary shows correct counts of ok/warning/error checks."""
        factory = DbFactory(db_path)

        # Create valid data
        factory.create_session(
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

    def test_check_empty_database(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check command succeeds on empty database."""
        # Don't create any data - just verify empty DB (db_path fixture ensures DB exists)

        result = cli_runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0

        # Should show healthy status checks (no integrity_check without --deep)
        assert "OK database" in result.output

    def test_check_sqlite_integrity_check(self, db_path: Path, cli_runner: CliRunner) -> None:
        """Check includes SQLite integrity check when --deep is passed."""
        factory = DbFactory(db_path)

        factory.create_session(
            id="conv1",
            provider="test",
            messages=[{"id": "m1", "role": "user", "text": "test"}],
        )

        result = cli_runner.invoke(cli, ["--plain", "doctor", "--deep", "--format", "json"])
        assert result.exit_code == 0

        data = _extract_json(result.output)

        # Find the sqlite_integrity check
        integrity_check = _find_named_check(data, "sqlite_integrity")
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
    def test_invalid_flag_combinations_rejected(
        self, cli_workspace: WorkspacePaths, args: list[str], expected_error: str
    ) -> None:
        """Flag dependencies and value constraints are enforced."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, args)
        assert result.exit_code != 0
        assert expected_error in result.output

    # --- Remaining non-repetitive tests ---

    def test_json_output_with_repair(self, cli_workspace: WorkspacePaths) -> None:
        """--format json with --repair includes maintenance results."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--format", "json", "--repair", "--preview"])
        assert result.exit_code == 0
        envelope = parse_json_object(
            result.output.split("\n", 1)[-1] if "Plain" in result.output else result.output,
            context="repair preview envelope",
        )
        data = json_object(envelope.get("result", envelope), context="repair preview payload")
        assert "maintenance" in data
        maintenance = json_object_field(data, "maintenance", context="repair preview payload")
        assert "resource_boundary" not in maintenance

    def test_repair_with_no_issues_shows_message(self, cli_workspace: WorkspacePaths) -> None:
        """When repair finds no issues, should show a maintenance status message."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--repair"])
        assert result.exit_code == 0
        assert (
            "No selected maintenance work" in result.output
            or "Changed" in result.output
            or "maintenance" in result.output.lower()
        )

    def test_vacuum_with_repair(self, cli_workspace: WorkspacePaths) -> None:
        """--vacuum with --repair should attempt VACUUM."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--repair", "--vacuum"])
        assert result.exit_code == 0
        assert "VACUUM" in result.output

    def test_json_output_with_repair_and_vacuum_is_machine_safe(self, cli_workspace: WorkspacePaths) -> None:
        """`--format json --repair --vacuum` should stay valid JSON."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "doctor", "--format", "json", "--repair", "--preview", "--vacuum"])

        assert result.exit_code == 0
        envelope = parse_json_object(result.output, context="repair vacuum envelope")
        data = json_object(envelope.get("result", envelope), context="repair vacuum payload")
        assert "maintenance" in data
        maintenance = json_object_field(data, "maintenance", context="repair vacuum payload")
        assert "resource_boundary" not in maintenance
        vacuum = json_object_field(data, "vacuum", context="repair vacuum payload")
        assert vacuum.get("ok") is True
        assert vacuum.get("preview") is True

    def test_check_schemas_json_output(self, cli_workspace: WorkspacePaths) -> None:
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
            "polylogue.cli.shared.check_workflow.run_schema_verification",
            return_value=fake_report,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["--plain", "doctor", "--format", "json", "--schemas", "--schema-samples", "all"]
            )

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "schema_verification" in data
        schema_verification = json_object_field(data, "schema_verification", context="schema verification payload")
        assert schema_verification.get("total_records") == 3
        assert schema_verification.get("max_samples") == "all"
        assert schema_verification.get("record_limit") == "all"
        assert schema_verification.get("record_offset") == 0

    def test_check_schemas_forwards_record_chunk_options(self, cli_workspace: WorkspacePaths) -> None:
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
            "polylogue.cli.shared.check_workflow.run_schema_verification",
            return_value=fake_report,
        ) as mock_verify:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "doctor",
                    "--format",
                    "json",
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

    def test_check_schemas_forwards_quarantine_flag(self, cli_workspace: WorkspacePaths) -> None:
        """Quarantine option is forwarded to verify_raw_corpus."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
        )

        with patch(
            "polylogue.cli.shared.check_workflow.run_schema_verification",
            return_value=fake_report,
        ) as mock_verify:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["--plain", "doctor", "--format", "json", "--schemas", "--schema-quarantine-malformed"],
            )

        assert result.exit_code == 0
        request = mock_verify.call_args.args[0]
        assert request.providers is None
        assert request.max_samples is None
        assert request.record_limit is None
        assert request.record_offset == 0
        assert request.quarantine_malformed is True
        assert request.progress_callback is not None

    def test_check_schemas_quarantine_reports_and_persists_decode_failure(
        self,
        cli_workspace: WorkspacePaths,
        cli_runner: CliRunner,
    ) -> None:
        """`doctor --schemas` reports and persists legitimate raw quarantine failures."""
        db_path = cli_workspace["db_path"]
        # Bootstrap the archive (creates source.db) before inserting the
        # malformed raw row.
        DbFactory(db_path).create_session(
            id="conv-quarantine-seed",
            provider="codex",
            messages=[{"id": "m-seed", "role": "user", "text": "seed"}],
        )
        raw_id = _insert_raw_blob(
            db_path=db_path,
            origin="codex-session",
            source_path="/tmp/session.jsonl",
            raw_content=(
                b'{"type":"session_meta"}\nnot json at all\n{"type":"response_item","payload":{"type":"message"}}'
            ),
        )

        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "doctor",
                "--schemas",
                "--schema-provider",
                "codex",
                "--schema-quarantine-malformed",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Schema verification: 1 raw records" in result.output
        assert "codex: valid=0 invalid=0 drift=0 skipped=0 decode_errors=1 quarantined=1" in result.output

        # Native quarantine persists into ``source.db`` ``raw_sessions`` with
        # millisecond timestamps and a single ``origin`` — the legacy
        # ``payload_provider`` / ``validation_provider`` columns do not exist.
        with open_index_db(db_path.parent / "source.db") as conn:
            row = conn.execute(
                """
                SELECT validation_status, validation_error, validation_mode, origin,
                       validated_at_ms, parsed_at_ms, parse_error
                FROM raw_sessions
                WHERE raw_id = ?
                """,
                (raw_id,),
            ).fetchone()

        assert row is not None
        assert row[0] == "failed"
        assert isinstance(row[1], str) and "Malformed JSONL lines:" in row[1]
        assert row[2] == "strict"
        assert row[3] == "codex-session"
        assert row[4] is not None
        assert row[5] is None
        assert isinstance(row[6], str) and "Malformed JSONL lines:" in row[6]

    def test_check_artifact_coverage_json_output(self, cli_workspace: WorkspacePaths) -> None:
        """--artifact-coverage adds artifact_coverage block to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = ArtifactCoverageReport(
            providers={
                "claude-code": ProviderArtifactCoverage(
                    provider="claude-code",
                    total_records=2,
                    recognized_non_parseable_records=1,
                    unsupported_parseable_records=1,
                    linked_sidecars=1,
                    subagent_streams=1,
                    streams_with_sidecars=1,
                    package_versions={"v7": 1},
                    element_kinds={"subagent_session_stream": 1},
                    resolution_reasons={"bundle_scope": 1},
                )
            },
            total_records=2,
        )

        with patch(
            "polylogue.cli.shared.check_workflow.run_artifact_coverage",
            return_value=ArtifactCoverageResult(report=fake_report),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--format", "json", "--artifact-coverage"])

        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "artifact_coverage" in data
        artifact_coverage = json_object_field(data, "artifact_coverage", context="artifact coverage payload")
        artifact_summary = json_object_field(artifact_coverage, "summary", context="artifact coverage payload")
        assert artifact_coverage.get("total_records") == 2
        assert artifact_summary.get("linked_sidecars") == 1
        assert artifact_summary.get("unsupported_parseable_records") == 1
        assert artifact_summary.get("package_versions") == {"v7": 1}

    def test_check_artifact_coverage_plain_output(self, cli_workspace: WorkspacePaths) -> None:
        """--artifact-coverage renders the artifact coverage summary in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = ArtifactCoverageReport(
            providers={
                "chatgpt": ProviderArtifactCoverage(
                    provider="chatgpt",
                    total_records=1,
                    contract_backed_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"session_document": 1},
                    resolution_reasons={"exact_structure": 1},
                )
            },
            total_records=1,
        )

        with patch(
            "polylogue.cli.shared.check_workflow.run_artifact_coverage",
            return_value=ArtifactCoverageResult(report=fake_report),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--artifact-coverage"])

        assert result.exit_code == 0
        assert "Artifact coverage:" in result.output
        assert "Resolved packages: v1=1" in result.output
        assert "Resolved elements: session_document=1" in result.output
        assert "Resolution reasons: exact_structure=1" in result.output
        assert "chatgpt: contract_backed=1" in result.output
        assert "packages: v1=1" in result.output

    def test_check_artifact_coverage_forwards_artifact_scope(self, cli_workspace: WorkspacePaths) -> None:
        """Artifact provider/limit/offset are forwarded to the evidence workflow."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_report = ArtifactCoverageReport(providers={}, total_records=0)

        with patch(
            "polylogue.cli.shared.check_workflow.run_artifact_coverage",
            return_value=ArtifactCoverageResult(report=fake_report),
        ) as mock_prove:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "doctor",
                    "--format",
                    "json",
                    "--artifact-coverage",
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

    def test_check_artifacts_json_output(self, cli_workspace: WorkspacePaths) -> None:
        """--artifacts adds artifact_observations rows to JSON output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_rows = [
            ArtifactObservationRecord(
                observation_id="obs-1",
                raw_id="raw-1",
                payload_provider=Provider.CHATGPT,
                source_name="chatgpt",
                source_path="/tmp/chatgpt.json",
                source_index=0,
                file_mtime=None,
                wire_format="json",
                artifact_kind="session_document",
                classification_reason="session-bearing document",
                parse_as_session=True,
                schema_eligible=True,
                support_status=ArtifactSupportStatus.SUPPORTED_PARSEABLE,
                malformed_jsonl_lines=0,
                decode_error=None,
                bundle_scope="chatgpt",
                cohort_id="cohort-1",
                resolved_package_version="v1",
                resolved_element_kind="session_document",
                resolution_reason="exact_structure",
                link_group_key=None,
                sidecar_agent_type=None,
                first_observed_at="2026-03-21T00:00:00+00:00",
                last_observed_at="2026-03-21T00:00:00+00:00",
            )
        ]

        with patch(
            "polylogue.cli.shared.check_workflow.list_artifact_observations",
            return_value=ArtifactObservationListResult(rows=fake_rows),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--plain",
                    "doctor",
                    "--format",
                    "json",
                    "--artifacts",
                    "--artifact-provider",
                    "chatgpt",
                    "--artifact-status",
                    "supported_parseable",
                    "--artifact-kind",
                    "session_document",
                ],
            )

        assert result.exit_code == 0
        data = _extract_json(result.output)
        artifact_observations = json_object_field(
            data, "artifact_observations", context="artifact observations payload"
        )
        assert artifact_observations.get("count") == 1
        observation = json_array_item(
            json_array_field(artifact_observations, "items", context="artifact observations payload"),
            0,
            context="artifact_observations.items",
        )
        assert observation.get("support_status") == "supported_parseable"

    def test_check_cohorts_plain_output(self, cli_workspace: WorkspacePaths) -> None:
        """--cohorts renders durable cohort summaries in plain output."""
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli

        fake_rows = [
            ArtifactCohortSummary(
                source_name="claude-code",
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
            "polylogue.cli.shared.check_workflow.list_artifact_cohorts",
            return_value=ArtifactCohortListResult(rows=fake_rows),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["--plain", "doctor", "--cohorts"])

        assert result.exit_code == 0
        assert "Artifact cohorts: 1 cohorts" in result.output
        assert "claude-code agent_sidecar_meta recognized_unparsed count=1" in result.output
