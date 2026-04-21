"""Tests for check --runtime command and run_runtime_readiness.

Validates readiness checks for the runtime environment, including:
- Database writability
- Schema version
- FTS table availability
- sqlite-vec availability
- Archive/render paths
- Configuration home
- Terminal capabilities
- UI libraries
- VHS availability
"""

from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.paths import Source
from polylogue.readiness import ReadinessReport, VerifyStatus, run_runtime_readiness

pytestmark = pytest.mark.machine_contract


class TestRuntimeReadinessCheckNames:
    """Tests that run_runtime_readiness includes expected check names."""

    def test_runtime_health_includes_db_writable_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes db_writable check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "db_writable" in check_names

    def test_runtime_health_includes_schema_version_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes schema_version check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "schema_version" in check_names

    def test_runtime_health_includes_fts_tables_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes fts_tables check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "fts_tables" in check_names

    def test_runtime_health_includes_sqlite_vec_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes sqlite_vec check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "sqlite_vec" in check_names

    def test_runtime_health_includes_archive_root_writable_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes archive_root_writable check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "archive_root_writable" in check_names

    def test_runtime_health_includes_render_root_writable_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes render_root_writable check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "render_root_writable" in check_names

    def test_runtime_health_includes_config_home_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes config_home check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "config_home" in check_names

    def test_runtime_health_includes_terminal_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes terminal check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "terminal" in check_names

    def test_runtime_health_includes_ui_libraries_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes ui_libraries check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "ui_libraries" in check_names

    def test_runtime_health_includes_vhs_check(self, tmp_path: Path) -> None:
        """run_runtime_readiness includes vhs check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        check_names = [c.name for c in report.checks]
        assert "vhs" in check_names


class TestRuntimeReadinessCheckResults:
    """Tests for runtime readiness check status results."""

    def test_runtime_health_with_writable_paths(self, tmp_path: Path) -> None:
        """Runtime health returns OK for writable archive and render roots."""
        archive_root = tmp_path / "archive"
        render_root = tmp_path / "render"
        archive_root.mkdir(parents=True, exist_ok=True)
        render_root.mkdir(parents=True, exist_ok=True)

        config = Config(
            archive_root=archive_root,
            render_root=render_root,
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )

        report = run_runtime_readiness(config)

        # Find writable checks
        archive_check = next(
            (c for c in report.checks if c.name == "archive_root_writable"),
            None,
        )
        render_check = next(
            (c for c in report.checks if c.name == "render_root_writable"),
            None,
        )

        assert archive_check is not None
        assert archive_check.status == VerifyStatus.OK

        assert render_check is not None
        assert render_check.status == VerifyStatus.OK

    def test_runtime_health_with_missing_paths(self, tmp_path: Path) -> None:
        """Runtime health shows OK/WARNING for missing but creatable paths."""
        # Path that can be created (parent exists and is writable)
        archive_root = tmp_path / "archive"
        render_root = tmp_path / "render"
        # Don't create them - should still pass if parent is writable

        config = Config(
            archive_root=archive_root,
            render_root=render_root,
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )

        report = run_runtime_readiness(config)

        archive_check = next(
            (c for c in report.checks if c.name == "archive_root_writable"),
            None,
        )
        render_check = next(
            (c for c in report.checks if c.name == "render_root_writable"),
            None,
        )

        # Both should succeed since parent is writable
        assert archive_check is not None
        assert archive_check.status == VerifyStatus.OK

        assert render_check is not None
        assert render_check.status == VerifyStatus.OK

    def test_runtime_health_returns_health_report(self, tmp_path: Path) -> None:
        """run_runtime_readiness returns a ReadinessReport object."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        assert isinstance(report, ReadinessReport)
        assert isinstance(report.checks, list)
        assert isinstance(report.summary, dict)

    def test_runtime_health_summary_has_ok_warning_error(self, tmp_path: Path) -> None:
        """ReadinessReport.summary includes ok, warning, error counts."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        assert "ok" in report.summary
        assert "warning" in report.summary
        assert "error" in report.summary
        assert isinstance(report.summary["ok"], int)
        assert isinstance(report.summary["warning"], int)
        assert isinstance(report.summary["error"], int)

    def test_runtime_health_summary_counts_match_checks(self, tmp_path: Path) -> None:
        """ReadinessReport.summary counts should match the actual checks."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        # Count checks by status
        ok_count = sum(1 for c in report.checks if c.status == VerifyStatus.OK)
        warning_count = sum(1 for c in report.checks if c.status == VerifyStatus.WARNING)
        error_count = sum(1 for c in report.checks if c.status == VerifyStatus.ERROR)

        assert report.summary["ok"] == ok_count
        assert report.summary["warning"] == warning_count
        assert report.summary["error"] == error_count

    def test_runtime_health_all_checks_have_status(self, tmp_path: Path) -> None:
        """All checks in the report must have a valid status."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        valid_statuses = {VerifyStatus.OK, VerifyStatus.WARNING, VerifyStatus.ERROR}
        for check in report.checks:
            assert check.status in valid_statuses

    def test_runtime_health_all_checks_have_names(self, tmp_path: Path) -> None:
        """All checks in the report must have non-empty names."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        for check in report.checks:
            assert check.name
            assert isinstance(check.name, str)


class TestRuntimeHealthReadOnlyPaths:
    """Tests for runtime readiness with read-only or inaccessible paths."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix-specific permission testing")
    def test_runtime_health_with_readonly_archive_root(self, tmp_path: Path) -> None:
        """Runtime health detects read-only archive_root."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)

        # Make read-only
        archive_root.chmod(0o444)

        try:
            config = Config(
                archive_root=archive_root,
                render_root=tmp_path / "render",
                sources=[Source(name="test", path=tmp_path / "inbox")],
            )
            (tmp_path / "render").mkdir(parents=True, exist_ok=True)

            report = run_runtime_readiness(config)

            archive_check = next(
                (c for c in report.checks if c.name == "archive_root_writable"),
                None,
            )

            # Should show ERROR or WARNING for read-only directory
            assert archive_check is not None
            assert archive_check.status in (
                VerifyStatus.ERROR,
                VerifyStatus.WARNING,
            )
        finally:
            # Restore permissions for cleanup
            archive_root.chmod(0o755)


class TestRuntimeHealthLegacySchema:
    """Tests for explicit reporting of unsupported legacy archive layouts."""

    def test_runtime_health_reports_legacy_inline_raw_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.paths

        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        importlib.reload(polylogue.paths)

        db_path = polylogue.paths.db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE raw_conversations (
                raw_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                payload_provider TEXT,
                source_name TEXT,
                source_path TEXT NOT NULL,
                source_index INTEGER,
                raw_content BLOB NOT NULL,
                acquired_at TEXT NOT NULL,
                file_mtime TEXT,
                parsed_at TEXT,
                parse_error TEXT,
                validated_at TEXT,
                validation_status TEXT,
                validation_error TEXT,
                validation_drift_count INTEGER DEFAULT 0,
                validation_provider TEXT,
                validation_mode TEXT
            );
            PRAGMA user_version = 1;
            """
        )
        conn.commit()
        conn.close()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)

        schema_check = next((c for c in report.checks if c.name == "schema_version"), None)
        assert schema_check is not None
        assert schema_check.status == VerifyStatus.ERROR
        assert "legacy inline raw-content layout" in schema_check.summary


class TestRuntimeHealthToDict:
    """Tests for ReadinessReport serialization from runtime checks."""

    def test_runtime_health_report_to_dict(self, tmp_path: Path) -> None:
        """ReadinessReport from run_runtime_readiness serializes correctly."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_readiness(config)
        data = report.to_dict()

        assert "checks" in data
        assert "summary" in data
        assert "timestamp" in data
        checks = data["checks"]
        assert isinstance(checks, list)
        assert all(isinstance(check, dict) and "name" in check for check in checks)
        assert all(isinstance(check, dict) and "status" in check for check in checks)
