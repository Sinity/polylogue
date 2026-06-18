from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands import maintenance
from polylogue.core.enums import Provider
from polylogue.storage.blob_gc import read_gc_history
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitResult,
    ArchiveTierInitResult,
)
from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitAction, ArchiveInitPlan
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

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


def _write_gc_candidate(cli_workspace: dict[str, Path], blob_hash: str) -> Path:
    blob_root = cli_workspace["data_root"] / "polylogue" / "blob"
    path = blob_root / blob_hash[:2] / blob_hash[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gc candidate")
    old_epoch_s = 946684800
    os.utime(path, (old_epoch_s, old_epoch_s))
    return path


def _seed_assertion_export_rows(archive_root: Path) -> None:
    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.USER)
        upsert_assertion(
            conn,
            assertion_id="export-mark",
            target_ref="session:s-1",
            kind=AssertionKind.MARK,
            scope_ref="run:r-1",
            key="export/mark",
            value={"label": "important"},
            body_text="operator mark",
            author_ref="user:operator",
            author_kind="user",
            evidence_refs=["message:s-1:1"],
            status="active",
            visibility="private",
            now_ms=1_700_000_001_000,
        )
        upsert_assertion(
            conn,
            assertion_id="export-deleted-note",
            target_ref="session:s-2",
            kind=AssertionKind.NOTE,
            scope_ref="run:r-2",
            key="export/note",
            body_text="deleted note retained for backup",
            status="deleted",
            visibility="private",
            now_ms=1_700_000_002_000,
        )


def test_archive_plan_cli_reports_tier_targets(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-plan", "--output-format", "json"],
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
        ["--plain", "ops", "maintenance", "archive-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    source_plan = next(tier for tier in payload["tiers"] if tier["tier"] == "source")
    assert source_plan["action"] == "blocked"
    assert any("source target already exists" in blocker for blocker in payload["blockers"])


def test_backup_plan_cli_reports_backup_profiles_and_tier_boundaries(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["mode"] == "backup_plan"
    assert payload["mutates"] is False
    assert payload["archive_root"] == str(cli_workspace["archive_root"])

    tiers = {tier["tier"]: tier for tier in payload["tiers"]}
    assert tiers["source"]["backup_class"] == "critical"
    assert tiers["source"]["backup_required"] is True
    assert tiers["index"]["backup_class"] == "warm_cache"
    assert tiers["index"]["backup_required"] is False
    assert tiers["embeddings"]["backup_policy"] == "back_up_when_present"
    assert tiers["user"]["backup_policy"] == "always_back_up"
    assert tiers["ops"]["backup_policy"] == "diagnostics_only"
    assert all(tier["present"] is True for tier in tiers.values())

    profiles = {profile["name"] for profile in payload["profiles"]}
    assert profiles == {
        "full_evidence",
        "user_overlays",
        "rebuildable_cache_exclude",
        "diagnostics_bundle",
    }
    assert payload["blob_store"]["path"] == str(cli_workspace["data_root"] / "polylogue" / "blob")
    assert payload["blob_store"]["backup_policy"] == "back_up_referenced_blobs_with_source_and_user_tiers"


def test_backup_plan_cli_surfaces_missing_tiers_and_wal_checkpoint_warning(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    archive_root = cli_workspace["archive_root"]
    (archive_root / "index.db").unlink()
    (archive_root / "user.db-wal").write_text("pending", encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    tiers = {tier["tier"]: tier for tier in payload["tiers"]}
    assert tiers["index"]["present"] is False
    assert tiers["user"]["wal_present"] is True
    assert tiers["user"]["checkpoint_recommended"] is True
    assert payload["warnings"] == ["user.db-wal is present; checkpoint before copying user.db"]


def test_backup_plan_cli_renders_plain_summary(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Archive backup plan" in result.output
    assert "source.db: critical policy=back_up present" in result.output
    assert "full_evidence:" in result.output


def test_assertion_export_cli_emits_all_assertions_as_jsonl(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_assertion_export_rows(cli_workspace["archive_root"])

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "assertion-export"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert [row["assertion_id"] for row in rows] == ["export-mark", "export-deleted-note"]
    assert rows[0]["kind"] == "mark"
    assert rows[0]["value"] == {"label": "important"}
    assert rows[0]["evidence_refs"] == ["message:s-1:1"]
    assert rows[1]["status"] == "deleted"


def test_assertion_export_cli_filters_and_writes_json_file(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_assertion_export_rows(cli_workspace["archive_root"])
    out_path = cli_workspace["archive_root"] / "exports" / "assertions.json"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "assertion-export",
            "--format",
            "json",
            "--kind",
            "note",
            "--status",
            "deleted",
            "--out",
            str(out_path),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert result.output == f"Exported 1 assertions to {out_path}\n"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "assertion_export"
    assert payload["count"] == 1
    assert [row["assertion_id"] for row in payload["assertions"]] == ["export-deleted-note"]


def test_blob_gc_cli_dry_run_reports_without_deleting(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    blob_hash = "aa" + "1" * 62
    candidate = _write_gc_candidate(cli_workspace, blob_hash)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--max-batch", "5", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["mode"] == "blob_gc"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 1
    assert payload["inspected_count"] == 1
    assert payload["would_delete_count"] == 1
    assert payload["deleted_count"] == 0
    assert payload["generation_written"] is False
    assert candidate.exists(), "dry-run must not delete the candidate"
    assert read_gc_history(cli_workspace["archive_root"] / "index.db", limit=1) == []


def test_blob_gc_cli_plain_preview_names_skip_counts(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _write_gc_candidate(cli_workspace, "bb" + "2" * 62)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--max-batch", "5"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Blob GC dry-run" in result.output
    assert "Candidates: 1" in result.output
    assert "Result:     would delete 1 blob(s)" in result.output
    assert "referenced=0 leased=0 missing=0 unlink_error=0" in result.output


def test_blob_gc_cli_yes_deletes_and_records_generation(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    blob_hash = "cc" + "3" * 62
    candidate = _write_gc_candidate(cli_workspace, blob_hash)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--yes", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mutates"] is True
    assert payload["dry_run"] is False
    assert payload["would_delete_count"] == 0
    assert payload["deleted_count"] == 1
    assert payload["reclaimed_bytes"] == len(b"gc candidate")
    assert payload["generation_written"] is True
    assert str(payload["generation_id"]).startswith("gc-")
    assert not candidate.exists()

    history = read_gc_history(cli_workspace["archive_root"] / "index.db", limit=1)
    assert len(history) == 1
    assert history[0].generation_id == payload["generation_id"]
    assert history[0].reclaimed_count == 1


def test_archive_init_cli_is_dry_run_without_yes(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    _stage_uninitialized_archive(cli_workspace)
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "archive-init", "--output-format", "json"],
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
        ["--plain", "ops", "maintenance", "archive-init", "--yes", "--output-format", "json"],
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
    result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "--help"], catch_exceptions=False)

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
            "ops",
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
            "ops",
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
