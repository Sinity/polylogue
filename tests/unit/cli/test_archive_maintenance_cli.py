from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands import maintenance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitResult,
    ArchiveTierInitResult,
)
from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitAction, ArchiveInitPlan
from polylogue.types import Provider

_ARCHIVE_TIERS = ("source.db", "index.db", "embeddings.db", "ops.db", "user.db")


def _stage_uninitialized_archive(cli_workspace: dict[str, Path]) -> None:
    """Reset the workspace to an uninitialized state for plan/init tests.

    ``cli_workspace`` bootstraps a full archive (all tiers present),
    which the archive planner would classify as already-initialized. Clear the
    pre-created tiers so the plan reports a ready, "create every tier"
    initialization.
    """
    archive_root = cli_workspace["archive_root"]
    for name in _ARCHIVE_TIERS:
        (archive_root / name).unlink(missing_ok=True)


def test_archive_plan_cli_reports_tier_targets(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "maintenance", "archive-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert {tier["tier"]: tier["action"] for tier in payload["tiers"]} == {
        "source": "create",
        "index": "create",
        "embeddings": "create",
        "user": "create",
        "ops": "create",
    }
    assert {Path(tier["path"]).name for tier in payload["tiers"]} == {
        "source.db",
        "index.db",
        "embeddings.db",
        "user.db",
        "ops.db",
    }
    assert {tier["tier"]: tier["durability"] for tier in payload["tiers"]} == {
        "source": "irreplaceable",
        "index": "rebuildable",
        "embeddings": "expensive_rebuild",
        "user": "human",
        "ops": "disposable",
    }
    assert all(isinstance(tier["expected_user_version"], int) for tier in payload["tiers"])


def test_archive_plan_cli_surfaces_existing_target_blocker(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source_target = cli_workspace["archive_root"] / "source.db"
    source_target.write_bytes(b"not an archive source")

    result = cli_runner.invoke(
        cli,
        ["--plain", "maintenance", "archive-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    source_plan = next(tier for tier in payload["tiers"] if tier["tier"] == "source")
    assert source_plan["action"] == "blocked"
    assert any("source target already exists" in blocker for blocker in payload["blockers"])


def test_archive_init_cli_is_dry_run_without_yes(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "maintenance", "archive-init", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["executed"] is False
    assert payload["ready"] is True
    assert not (cli_workspace["archive_root"] / "index.db").exists()


def test_archive_init_cli_executes_confirmed_initialization(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_init(plan: ArchiveInitPlan) -> ArchiveInitResult:
        return ArchiveInitResult(
            tier_results=(
                ArchiveTierInitResult(
                    tier="index",
                    path=plan.archive_root / "index.db",
                    action=ArchiveInitAction.CREATE,
                    backup_path=None,
                    initialized=True,
                ),
            ),
        )

    monkeypatch.setattr(maintenance, "initialize_archive_tier_files_from_plan", fake_init)

    result = cli_runner.invoke(
        cli,
        ["--plain", "maintenance", "archive-init", "--yes", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["executed"] is True
    assert payload["tiers"] == [
        {
            "action": "create",
            "backup_path": None,
            "initialized": True,
            "path": str(cli_workspace["archive_root"] / "index.db"),
            "tier": "index",
        }
    ]


def test_archive_maintenance_help_omits_copy_activation_surface(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["--plain", "maintenance", "--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "archive-plan" in result.output
    assert "archive-init" in result.output
    assert "archive-read" in result.output
    for removed in (
        "archive-copy-raw",
        "archive-copy-archive",
        "archive-copy-insights",
        "archive-copy-user",
        "archive-copy-all",
        "archive-copy-audit",
        "archive-activate",
    ):
        assert removed not in result.output


def test_archive_read_cli_lists_archive_sessions(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    sqlite3.connect(cli_workspace["archive_root"] / "index.db").close()

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, *, limit: int, origin: str | None) -> list[ArchiveSessionSummary]:
            assert limit == 2
            assert origin == "codex-session"
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    provider=Provider.CODEX,
                    title="Copied",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=("archive",),
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "maintenance",
            "archive-read",
            "--origin",
            "codex-session",
            "--limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "list"
    assert payload["sessions"] == [
        {
            "created_at": "2026-01-02T03:04:05Z",
            "message_count": 3,
            "native_id": "native-1",
            "origin": "codex-session",
            "session_id": "codex-session:native-1",
            "tags": ["archive"],
            "title": "Copied",
            "updated_at": "2026-01-02T03:04:06Z",
            "word_count": 9,
        }
    ]


def test_archive_read_cli_searches_archive_blocks(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    sqlite3.connect(cli_workspace["archive_root"] / "index.db").close()

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_summaries(self, query: str, *, limit: int, origin: str | None) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            assert limit == 5
            assert origin is None
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    provider=Provider.CODEX,
                    title="Copied",
                    snippet="[needle]",
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.commands.maintenance.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "maintenance",
            "archive-read",
            "--query",
            "needle",
            "--limit",
            "5",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "search"
    assert payload["hits"][0]["block_id"] == "codex-session:native-1:m1:0"
    assert payload["hits"][0]["snippet"] == "[needle]"
