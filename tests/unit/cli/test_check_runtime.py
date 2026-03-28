"""Tests for check --runtime command and run_runtime_health.

Validates health checks for the runtime environment, including:
- Database writability
- Schema version
- FTS table availability
- sqlite-vec availability
- Archive/render paths
- Configuration paths
- Terminal capabilities
- UI libraries
- VHS availability
"""

from __future__ import annotations

import os

import pytest

from polylogue.config import Config
from polylogue.health_models import HealthReport, VerifyStatus
from polylogue.health_runtime import run_runtime_health
from polylogue.paths import Source

pytestmark = pytest.mark.machine_contract


class TestRuntimeHealthCheckNames:
    """Tests that run_runtime_health includes expected check names."""

    def test_runtime_health_includes_db_writable_check(self, tmp_path):
        """run_runtime_health includes db_writable check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "db_writable" in check_names

    def test_runtime_health_includes_schema_version_check(self, tmp_path):
        """run_runtime_health includes schema_version check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "schema_version" in check_names

    def test_runtime_health_includes_fts_tables_check(self, tmp_path):
        """run_runtime_health includes fts_tables check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "fts_tables" in check_names

    def test_runtime_health_includes_sqlite_vec_check(self, tmp_path):
        """run_runtime_health includes sqlite_vec check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "sqlite_vec" in check_names

    def test_runtime_health_includes_archive_root_writable_check(self, tmp_path):
        """run_runtime_health includes archive_root_writable check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "archive_root_writable" in check_names

    def test_runtime_health_includes_render_root_writable_check(self, tmp_path):
        """run_runtime_health includes render_root_writable check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "render_root_writable" in check_names

    def test_runtime_health_includes_config_path_check(self, tmp_path):
        """run_runtime_health includes config_path check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "config_path" in check_names

    def test_runtime_health_includes_terminal_check(self, tmp_path):
        """run_runtime_health includes terminal check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "terminal" in check_names

    def test_runtime_health_includes_ui_libraries_check(self, tmp_path):
        """run_runtime_health includes ui_libraries check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "ui_libraries" in check_names

    def test_runtime_health_includes_vhs_check(self, tmp_path):
        """run_runtime_health includes vhs check."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        check_names = [c.name for c in report.checks]
        assert "vhs" in check_names


class TestRuntimeHealthCheckResults:
    """Tests for runtime health check status results."""

    def test_runtime_health_with_writable_paths(self, tmp_path):
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

        report = run_runtime_health(config)

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

    def test_runtime_health_with_missing_paths(self, tmp_path):
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

        report = run_runtime_health(config)

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

    def test_runtime_health_returns_health_report(self, tmp_path):
        """run_runtime_health returns a HealthReport object."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        assert isinstance(report, HealthReport)
        assert isinstance(report.checks, list)
        assert isinstance(report.summary, dict)

    def test_runtime_health_summary_has_ok_warning_error(self, tmp_path):
        """HealthReport.summary includes ok, warning, error counts."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        assert "ok" in report.summary
        assert "warning" in report.summary
        assert "error" in report.summary
        assert isinstance(report.summary["ok"], int)
        assert isinstance(report.summary["warning"], int)
        assert isinstance(report.summary["error"], int)

    def test_runtime_health_summary_counts_match_checks(self, tmp_path):
        """HealthReport.summary counts should match the actual checks."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        # Count checks by status
        ok_count = sum(1 for c in report.checks if c.status == VerifyStatus.OK)
        warning_count = sum(1 for c in report.checks if c.status == VerifyStatus.WARNING)
        error_count = sum(1 for c in report.checks if c.status == VerifyStatus.ERROR)

        assert report.summary["ok"] == ok_count
        assert report.summary["warning"] == warning_count
        assert report.summary["error"] == error_count

    def test_runtime_health_all_checks_have_status(self, tmp_path):
        """All checks in the report must have a valid status."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        valid_statuses = {VerifyStatus.OK, VerifyStatus.WARNING, VerifyStatus.ERROR}
        for check in report.checks:
            assert check.status in valid_statuses

    def test_runtime_health_all_checks_have_names(self, tmp_path):
        """All checks in the report must have non-empty names."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)

        for check in report.checks:
            assert check.name
            assert isinstance(check.name, str)


class TestRuntimeHealthReadOnlyPaths:
    """Tests for runtime health with read-only or inaccessible paths."""

    @pytest.mark.skipif(
        os.name == "nt", reason="Unix-specific permission testing"
    )
    def test_runtime_health_with_readonly_archive_root(self, tmp_path):
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

            report = run_runtime_health(config)

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


class TestRuntimeHealthToDict:
    """Tests for HealthReport serialization from runtime checks."""

    def test_runtime_health_report_to_dict(self, tmp_path):
        """HealthReport from run_runtime_health serializes correctly."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="test", path=tmp_path / "inbox")],
        )
        (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
        (tmp_path / "render").mkdir(parents=True, exist_ok=True)

        report = run_runtime_health(config)
        data = report.to_dict()

        assert "checks" in data
        assert "summary" in data
        assert "timestamp" in data
        assert isinstance(data["checks"], list)
        assert all("name" in c for c in data["checks"])
        assert all("status" in c for c in data["checks"])
