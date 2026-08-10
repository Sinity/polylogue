"""Real file-backed tests for the pre-reindex message-owner compatibility pass."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.daemon import bulk_rebuild
from polylogue.maintenance.message_owner_scope_backfill import (
    MessageOwnerScopeBackfillError,
    apply_message_owner_scope_backfill,
    census_message_owner_scope_backfill,
    validate_message_owner_scope_backfill_receipt,
    write_message_owner_scope_backfill_plan,
)
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_annotation, upsert_mark
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import SessionBuilder, db_setup


def _seed_opaque_archive(workspace_env: dict[str, Path]) -> tuple[Path, list[tuple[str, str]]]:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "opaque:n:session")
        .provider("claude-code")
        .add_message(message_id="native", text="native")
    )
    builder.save()
    root = workspace_env["archive_root"]
    with open_connection(root / "index.db") as conn:
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="opaque:p:session",
                title="Positional",
                messages=[ParsedMessage(provider_message_id="", role=Role.USER, text="positional", position=0)],
            ),
        )
    with sqlite3.connect(root / "index.db") as conn:
        message_rows = [
            (str(row[0]), str(row[1])) for row in conn.execute("SELECT message_id, session_id FROM messages")
        ]
    with sqlite3.connect(root / "user.db") as conn:
        for message_id, _session_id in message_rows:
            upsert_mark(conn, "message", message_id, f"mark-{message_id}")
            upsert_annotation(conn, "message", message_id, f"note-{message_id}", annotation_id=f"ann-{message_id}")
        conn.commit()
    return root, message_rows


def _plan_and_paths(tmp_path: Path, root: Path) -> tuple[Path, Path, Path]:
    plan = census_message_owner_scope_backfill(root)
    plan_path = tmp_path / "owner-plan.json"
    write_message_owner_scope_backfill_plan(plan, plan_path)
    return plan_path, tmp_path / "user-backup" / "manifest.json", tmp_path / "owner-receipt.json"


def _allow_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.maintenance.message_owner_scope_backfill.validate_migration_backup_manifest",
        lambda path, tier, *, connection: path,
    )


def test_census_is_deterministic_and_opaque_for_n_and_p_ids(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    root, message_rows = _seed_opaque_archive(workspace_env)

    first = census_message_owner_scope_backfill(root)
    second = census_message_owner_scope_backfill(root)

    assert first.plan_digest == second.plan_digest
    assert first.counts == {
        "exact-resolvable": 4,
        "already-scoped": 0,
        "missing-index-owner": 0,
        "malformed-scope": 0,
        "conflicting-scope": 0,
    }
    assert {":n:", ":p:"} <= {
        marker for marker in (":n:", ":p:") if any(marker in message_id for message_id, _ in message_rows)
    }
    assert all(row.target_id in {message_id for message_id, _ in message_rows} for row in first.rows)


def test_apply_updates_only_exact_rows_and_writes_complete_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("verified user backup", encoding="utf-8")
    _allow_backup(monkeypatch)

    report = apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    assert report.updated_count == 4
    assert report.terminal_state == "committed"
    assert report.after_plan is not None and report.after_plan.unresolved_denominator == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["before_counts"]["exact-resolvable"] == 4
    assert payload["after_counts"]["already-scoped"] == 4
    assert validate_message_owner_scope_backfill_receipt(root, receipt)["unresolved_denominator"] == 0
    with sqlite3.connect(root / "user.db") as conn:
        assert all(
            row[0].startswith("session:")
            for row in conn.execute("SELECT scope_ref FROM assertions WHERE kind IN ('mark', 'annotation')")
        )


def test_apply_is_idempotent_with_a_fresh_plan(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    _allow_backup(monkeypatch)
    first_plan, backup, first_receipt = _plan_and_paths(tmp_path / "first", root)
    backup.parent.mkdir()
    backup.write_text("verified user backup", encoding="utf-8")
    first = apply_message_owner_scope_backfill(
        root, plan_path=first_plan, backup_manifest=backup, receipt_path=first_receipt, dry_run=False
    )
    assert first.updated_count == 4

    second_plan = census_message_owner_scope_backfill(root)
    second_plan_path = tmp_path / "second-plan.json"
    second_receipt = tmp_path / "second-receipt.json"
    write_message_owner_scope_backfill_plan(second_plan, second_plan_path)
    second = apply_message_owner_scope_backfill(
        root, plan_path=second_plan_path, backup_manifest=backup, receipt_path=second_receipt, dry_run=False
    )
    assert second.updated_count == 0
    assert second.terminal_state == "committed"


def test_missing_owner_is_a_typed_blocker_and_not_a_complete_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    with sqlite3.connect(root / "user.db") as conn:
        upsert_mark(conn, "message", "missing:opaque:n:id", "missing")
        conn.commit()
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("verified user backup", encoding="utf-8")
    _allow_backup(monkeypatch)

    report = apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    assert report.terminal_state == "blocked"
    assert report.after_plan is not None and report.after_plan.unresolved_denominator == 1
    assert json.loads(receipt.read_text(encoding="utf-8"))["complete"] is False
    with pytest.raises(MessageOwnerScopeBackfillError, match="not complete"):
        validate_message_owner_scope_backfill_receipt(root, receipt)


@pytest.mark.parametrize("scope_ref, message", [("session:other", "conflicting"), ("owner:other", "malformed")])
def test_conflict_and_malformed_scope_refuse_before_mutation(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope_ref: str, message: str
) -> None:
    root, message_rows = _seed_opaque_archive(workspace_env)
    target = message_rows[0][0]
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute(
            "UPDATE assertions SET scope_ref = ? WHERE target_ref = ? AND kind = 'mark'",
            (scope_ref, f"message:{target}"),
        )
        conn.commit()
    plan = census_message_owner_scope_backfill(root)
    assert plan.counts["conflicting-scope" if message == "conflicting" else "malformed-scope"] == 1
    plan_path = tmp_path / "plan.json"
    write_message_owner_scope_backfill_plan(plan, plan_path)
    backup = tmp_path / "backup-manifest.json"
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    with pytest.raises(MessageOwnerScopeBackfillError, match="malformed or conflicting"):
        apply_message_owner_scope_backfill(
            root, plan_path=plan_path, backup_manifest=backup, receipt_path=tmp_path / "receipt.json", dry_run=False
        )


def test_stale_plan_wrong_backup_and_active_daemon_are_fail_closed(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("UPDATE assertions SET scope_ref = 'session:changed' WHERE kind = 'mark' LIMIT 1")
        conn.commit()
    _allow_backup(monkeypatch)
    with pytest.raises(MessageOwnerScopeBackfillError, match="plan changed"):
        apply_message_owner_scope_backfill(
            root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
        )

    monkeypatch.setattr(
        "polylogue.maintenance.message_owner_scope_backfill.validate_migration_backup_manifest",
        lambda path, tier, *, connection: (_ for _ in ()).throw(RuntimeError("wrong backup")),
    )
    fresh = census_message_owner_scope_backfill(root)
    fresh_path = tmp_path / "fresh-plan.json"
    write_message_owner_scope_backfill_plan(fresh, fresh_path)
    with pytest.raises(RuntimeError, match="wrong backup"):
        apply_message_owner_scope_backfill(
            root, plan_path=fresh_path, backup_manifest=backup, receipt_path=tmp_path / "wrong.json", dry_run=False
        )

    _allow_backup(monkeypatch)
    monkeypatch.setattr("polylogue.maintenance.message_owner_scope_backfill.running_daemon_pid", lambda config: 1234)
    with pytest.raises(MessageOwnerScopeBackfillError, match="daemon"):
        apply_message_owner_scope_backfill(
            root, plan_path=fresh_path, backup_manifest=backup, receipt_path=tmp_path / "daemon.json", dry_run=False
        )


def test_tampered_plan_and_receipt_are_rejected(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    plan = census_message_owner_scope_backfill(root)
    plan_path = tmp_path / "plan.json"
    write_message_owner_scope_backfill_plan(plan, plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["rows"][0]["target_id"] = "tampered"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MessageOwnerScopeBackfillError, match="digest mismatch"):
        apply_message_owner_scope_backfill(
            root,
            plan_path=plan_path,
            backup_manifest=tmp_path / "backup",
            receipt_path=tmp_path / "receipt",
            dry_run=False,
        )


def test_transaction_rolls_back_and_leaves_recovery_marker_on_mid_apply_failure(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    original = census_message_owner_scope_backfill(root)
    calls = 0

    def fail_under_lock(current_root: Path, plan: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MessageOwnerScopeBackfillError("simulated crash")
        return original

    monkeypatch.setattr("polylogue.maintenance.message_owner_scope_backfill._validate_plan_binding", fail_under_lock)
    with pytest.raises(MessageOwnerScopeBackfillError, match="simulated crash"):
        apply_message_owner_scope_backfill(
            root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
        )
    with sqlite3.connect(root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE scope_ref IS NOT NULL").fetchone()[0] == 0
    assert not receipt.exists()
    assert receipt.with_name(receipt.name + ".prepared").exists()


def test_post_commit_receipt_failure_recovers_only_the_prepared_exact_state(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)

    with monkeypatch.context() as post_commit_failure:
        post_commit_failure.setattr(
            "polylogue.maintenance.message_owner_scope_backfill._final_receipt",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("simulated receipt fsync failure")),
        )
        with pytest.raises(OSError, match="receipt fsync"):
            apply_message_owner_scope_backfill(
                root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
            )

    assert receipt.with_name(receipt.name + ".prepared").exists()
    recovered = apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    assert recovered.terminal_state == "committed"
    assert recovered.updated_count == 4
    assert receipt.exists()
    assert not receipt.with_name(receipt.name + ".prepared").exists()


def test_complete_receipt_rejects_current_durable_scope_drift(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, message_rows = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute(
            "UPDATE assertions SET scope_ref = 'session:wrong-owner' WHERE target_ref = ? AND kind = 'mark'",
            (f"message:{message_rows[0][0]}",),
        )
        conn.commit()

    with pytest.raises(MessageOwnerScopeBackfillError, match="durable message assertion state is stale"):
        validate_message_owner_scope_backfill_receipt(root, receipt, candidate_index_path=root / "index.db")


def test_daemon_bulk_rebuild_refuses_legacy_message_owner_scopes_before_candidate_creation(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _seed_opaque_archive(workspace_env)
    monkeypatch.setattr(bulk_rebuild, "_validate_rebuild_provenance_receipt", lambda _root, _receipt: None)

    with pytest.raises(RuntimeError, match="message-owner scope backfill"):
        bulk_rebuild.resolve_or_start_daemon_bulk_rebuild_transaction(root)

    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


def test_complete_receipt_is_consumable_after_index_rows_are_deleted(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, message_rows = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )
    validate_message_owner_scope_backfill_receipt(root, receipt, candidate_index_path=root / "index.db")
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("DELETE FROM messages")
        conn.commit()
    with pytest.raises(MessageOwnerScopeBackfillError, match="candidate index does not own"):
        validate_message_owner_scope_backfill_receipt(root, receipt, candidate_index_path=root / "index.db")
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        marks = archive.list_marks()
        annotations = archive.list_annotations()
    assert {row["target_id"] for row in marks} == {message_id for message_id, _ in message_rows}
    assert {row["target_id"] for row in annotations} == {message_id for message_id, _ in message_rows}
    assert {row["session_id"] for row in marks} == {session_id for _, session_id in message_rows}
