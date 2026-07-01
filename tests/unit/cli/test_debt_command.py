"""Tests for ``polylogue ops debt``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.surfaces.payloads import ArchiveDebtListPayload, ArchiveDebtRowPayload, ArchiveDebtTotalsPayload


def _payload(root: Path) -> ArchiveDebtListPayload:
    return ArchiveDebtListPayload(
        generated_at=datetime(2026, 6, 19, tzinfo=UTC).isoformat(),
        archive_root=str(root),
        rows=(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:catchup:backlog",
                kind="embedding",
                stage="catchup",
                subject_ref="embedding:pending",
                severity="warning",
                status="actionable",
                owner="daemon",
                summary="3 session(s) pending embedding catch-up",
                affected_count=3,
                evidence_refs=("session:pending-1", "session:pending-2"),
                caveats=("Evidence refs are sampled.",),
            ),
        ),
        totals=ArchiveDebtTotalsPayload(
            total=1,
            warning=1,
            actionable=1,
            affected_total=3,
            affected_warning=3,
            affected_actionable=3,
            affected_open=3,
        ),
    )


def test_debt_list_json_uses_shared_payload(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_db = tmp_path / "index.db"
    index_db.touch()
    captured: dict[str, object] = {}

    def fake_archive_debt_list(**kwargs: object) -> ArchiveDebtListPayload:
        captured.update(kwargs)
        return _payload(tmp_path)

    monkeypatch.setattr("polylogue.cli.commands.debt.active_index_db_path", lambda: index_db)
    monkeypatch.setattr("polylogue.cli.commands.debt.archive_debt_list", fake_archive_debt_list)

    result = cli_runner.invoke(cli, ["--plain", "ops", "debt", "list", "--kind", "embedding", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "archive-debt-list"
    assert payload["rows"][0]["debt_ref"] == "debt:embedding:catchup:backlog"
    assert captured["archive_root"] == tmp_path
    assert captured["kinds"] == ("embedding",)


def test_debt_list_text_renders_actionable_summary(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_db = tmp_path / "index.db"
    index_db.touch()
    monkeypatch.setattr("polylogue.cli.commands.debt.active_index_db_path", lambda: index_db)
    monkeypatch.setattr("polylogue.cli.commands.debt.archive_debt_list", lambda **_kwargs: _payload(tmp_path))

    result = cli_runner.invoke(cli, ["--plain", "ops", "debt", "list"])

    assert result.exit_code == 0, result.output
    assert "Archive debt: 1 row(s)" in result.output
    assert "affected=3 affected_warning=3 affected_actionable=3 affected_open=3 affected_blocked=0" in result.output
    assert "3 session(s) pending embedding catch-up" in result.output
    assert "affected_count=3" in result.output
    assert "evidence: session:pending-1" in result.output
    assert "evidence: session:pending-2" in result.output
    assert "caveat: Evidence refs are sampled." in result.output
