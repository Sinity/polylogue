"""Real file-backed tests for the pre-reindex message-owner compatibility pass."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

import polylogue.maintenance.message_owner_scope_backfill as owner_backfill_module
from polylogue.annotations.batch import AnnotationBatch
from polylogue.annotations.schema import AnnotationField, AnnotationSchema, AnnotationSchemaRegistry
from polylogue.annotations.write import assertion_id_for_schema_annotation, upsert_annotation_assertion
from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.daemon import bulk_rebuild
from polylogue.maintenance.message_owner_scope_backfill import (
    MessageOwnerScopeBackfillError,
    apply_message_owner_scope_backfill,
    census_message_owner_scope_backfill,
    validate_message_owner_scope_backfill_receipt,
    validate_message_owner_scope_for_index_replacement,
    write_message_owner_scope_backfill_plan,
)
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_annotation, upsert_mark
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt
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


def _write_batched_message_annotation(root: Path, *, target_id: str) -> str:
    """Persist actual annotation-batch provenance for one message assertion."""
    schema = AnnotationSchema(
        schema_id="test.message-owner",
        version=1,
        title="Message owner backfill",
        description="Regression fixture for batch-scoped message annotations.",
        fields=(AnnotationField(name="label", value_type="enum", enum_values=("yes", "no")),),
        target_ref_kinds=("message",),
        evidence_policy="optional",
        status="active",
    )
    batch_ref = "annotation-batch:message-owner-backfill"
    target_ref = f"message:{target_id}"
    author_ref = "agent:message-owner-backfill"
    assertion_id = assertion_id_for_schema_annotation(
        schema_qualified_id=schema.qualified_id,
        target_ref=target_ref,
        author_ref=author_ref,
        row_key="batch-row",
        batch_ref=batch_ref,
    )
    batch = AnnotationBatch(
        batch_id="message-owner-backfill",
        schema_id=schema.schema_id,
        schema_version=schema.version,
        target_ref=target_ref,
        source_result_ref="result-set:message-owner-backfill",
        actor_ref=author_ref,
        model_ref="agent:gpt-5.6-terra",
        prompt_ref="block:message-owner-backfill:0",
        total_count=1,
        valid_count=1,
        invalid_count=0,
        abstained_count=0,
        assertion_refs=(f"assertion:{assertion_id}",),
        created_at_ms=1_000,
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.save_annotation_schema(schema, registered_at_ms=1_000)
        archive.save_annotation_batch(batch)
    registry = AnnotationSchemaRegistry()
    registry.register(schema)
    with sqlite3.connect(root / "user.db") as conn:
        persisted = upsert_annotation_assertion(
            conn,
            schema=schema,
            registry=registry,
            target_ref=target_ref,
            value={"label": "yes"},
            row_key="batch-row",
            author_ref=author_ref,
            batch_ref=batch_ref,
            now_ms=2_000,
        )
        conn.commit()
    assert persisted.assertion_id == assertion_id
    return assertion_id


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


def test_stale_active_index_schema_can_backfill_and_validate_a_current_candidate(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable pass may start from an obsolete derived generation.

    Anti-vacuity: this uses the production census, transaction, receipt, and
    candidate-validation paths over the fixture's file-backed archive.  Restoring
    the old active-index currency rejection makes the initial census fail.
    """

    root, _ = _seed_opaque_archive(workspace_env)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] - 1}")
        conn.commit()
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)

    apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )
    candidate = tmp_path / "candidate-index.db"
    with sqlite3.connect(root / "index.db") as source, sqlite3.connect(candidate) as target:
        source.backup(target)
    with sqlite3.connect(candidate) as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        conn.commit()

    assert (
        validate_message_owner_scope_backfill_receipt(root, receipt, candidate_index_path=candidate)["complete"] is True
    )


def test_completed_message_owner_backfill_requires_a_receipt_for_index_replacement(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt authorizes only the completed durable state, never a planned one."""

    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    with pytest.raises(MessageOwnerScopeBackfillError, match="complete receipt is required"):
        validate_message_owner_scope_for_index_replacement(root, receipt_path=None)


def test_no_promote_rebuild_runs_candidate_message_owner_admission_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real inactive generation must be admitted before the no-promote route returns.

    Anti-vacuity: the raw payload is persisted through ``ArchiveStore`` and the
    production rebuild constructs a real generation.  Dropping the terminal
    gate from no-promote rebuilds leaves ``candidate_paths`` empty.
    """

    root = tmp_path / "candidate-archive"
    initialize_active_archive_root(root)
    payload = (
        b"\n".join(
            (
                b'{"type":"session_meta","payload":{"id":"candidate-gate"}}',
                b'{"type":"response_item","payload":{"type":"message","id":"candidate-gate-m0",'
                b'"role":"user","content":[{"type":"input_text","text":"candidate gate"}]}}',
            )
        )
        + b"\n"
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="candidate-gate.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="codex-session:candidate-gate",
                kind=RawRevisionKind.FULL,
                source_revision="candidate-gate-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.ASSERTED,
            ),
        )
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET baseline_raw_id = raw_id, revision_authority = 'byte_proven'")
        conn.commit()
    backfill_historical_revision_evidence(root)
    schema_receipt = write_valid_rebuild_receipt(root, tmp_path / "schema-receipt.json")
    candidate_paths: list[Path] = []
    original_gate = owner_backfill_module.validate_message_owner_scope_for_index_replacement

    def record_candidate_gate(
        archive_root: Path, *, receipt_path: Path | None, candidate_index_path: Path | None = None
    ) -> None:
        original_gate(archive_root, receipt_path=receipt_path, candidate_index_path=candidate_index_path)
        if candidate_index_path is not None:
            candidate_paths.append(candidate_index_path)

    monkeypatch.setattr(
        owner_backfill_module, "validate_message_owner_scope_for_index_replacement", record_candidate_gate
    )
    rebuilt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=False,
            schema_inference_receipt_path=schema_receipt,
        )
    )

    assert rebuilt.generation["state"] == "inactive"
    assert candidate_paths and candidate_paths[0].is_file()


def test_deleted_message_assertions_do_not_block_the_owner_scope_census(workspace_env: dict[str, Path]) -> None:
    """Soft-deleted overlays are absent from both the census and replacement gate."""

    root, _ = _seed_opaque_archive(workspace_env)
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("UPDATE assertions SET status = 'deleted' WHERE kind IN ('mark', 'annotation')")
        conn.commit()

    assert census_message_owner_scope_backfill(root).rows == ()
    validate_message_owner_scope_for_index_replacement(root, receipt_path=None)


def test_prepared_recovery_rejects_a_nonidentity_assertion_change_after_commit(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit-before-receipt recovery binds the entire durable assertion row.

    Anti-vacuity: this interrupts the production receipt writer after its
    transaction commits, mutates only ``body_text`` in SQLite, then re-enters
    the production recovery path.  Identity-only recovery would publish a
    receipt for the altered durable state.
    """

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
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("UPDATE assertions SET body_text = 'changed after commit' WHERE kind = 'annotation' LIMIT 1")
        conn.commit()

    with pytest.raises(MessageOwnerScopeBackfillError, match="does not prove the committed durable state"):
        apply_message_owner_scope_backfill(
            root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
        )


def test_prepared_recovery_preserves_a_partial_receipt_before_republishing(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-commit partial receipt is quarantined and recovered without reapplying SQL."""

    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)

    def write_partial_then_fail(**kwargs: object) -> object:
        receipt_path = kwargs["receipt_path"]
        assert isinstance(receipt_path, Path)
        receipt_path.write_text('{"partial":', encoding="utf-8")
        raise OSError("simulated partial receipt publication failure")

    with monkeypatch.context() as post_commit_failure:
        post_commit_failure.setattr(
            "polylogue.maintenance.message_owner_scope_backfill._final_receipt", write_partial_then_fail
        )
        with pytest.raises(OSError, match="partial receipt"):
            apply_message_owner_scope_backfill(
                root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
            )

    recovered = apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    fragment = receipt.with_name(receipt.name + ".partial")
    assert recovered.terminal_state == "committed"
    assert json.loads(receipt.read_text(encoding="utf-8"))["recovered_receipt_fragment"]["path"] == str(fragment)
    assert fragment.read_text(encoding="utf-8") == '{"partial":'


def test_annotation_batch_scope_survives_owner_backfill_and_candidate_proof(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch provenance remains immutable while the receipt binds its message owner.

    Anti-vacuity: the assertion is created through ``upsert_annotation_assertion``
    against a durable ``AnnotationBatch`` and passes the production census,
    transaction, receipt, and candidate-index gate.  Treating every non-session
    scope as malformed makes the initial plan block instead.
    """

    root, message_rows = _seed_opaque_archive(workspace_env)
    target_id = message_rows[0][0]
    assertion_id = _write_batched_message_annotation(root, target_id=target_id)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)

    before = census_message_owner_scope_backfill(root)
    batch_row = next(row for row in before.rows if row.assertion_id == assertion_id)
    assert batch_row.scope_ref == "annotation-batch:message-owner-backfill"
    assert batch_row.disposition.value == "already-scoped"

    apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    with sqlite3.connect(root / "user.db") as conn:
        assert conn.execute("SELECT scope_ref FROM assertions WHERE assertion_id = ?", (assertion_id,)).fetchone() == (
            "annotation-batch:message-owner-backfill",
        )
    assert validate_message_owner_scope_backfill_receipt(root, receipt, candidate_index_path=root / "index.db")


def test_daemon_bulk_rebuild_resolves_completed_message_owner_receipt(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon transaction admission consumes the same completed owner receipt.

    Anti-vacuity: this reaches the real daemon transaction resolver over the
    fixture's durable SQLite archive.  Removing the daemon receipt handoff
    makes its preflight pass ``None`` and rejects the completed backfill.
    """

    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )
    monkeypatch.setenv(owner_backfill_module.MESSAGE_OWNER_SCOPE_BACKFILL_RECEIPT_ENV, str(receipt))
    monkeypatch.setattr(bulk_rebuild, "_validate_rebuild_provenance_receipt", lambda _root, _receipt: None)
    monkeypatch.setattr(bulk_rebuild, "validate_rebuild_source_admission", lambda _root, _location: None)

    transaction = bulk_rebuild.resolve_or_start_daemon_bulk_rebuild_transaction(root)

    assert transaction.operation_id == bulk_rebuild.DAEMON_BULK_REBUILD_OPERATION_ID


def test_partial_no_promote_rebuild_skips_archive_wide_candidate_owner_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected subset is admitted but cannot prove unrelated durable rows.

    Anti-vacuity: this drives a real raw-id selected no-promote rebuild to its
    inactive generation.  Restoring unconditional candidate proof supplies a
    candidate path to the production owner gate; the complete-only route does
    not.
    """

    root = tmp_path / "partial-candidate-archive"
    initialize_active_archive_root(root)
    payload = (
        b'{"type":"session_meta","payload":{"id":"partial-candidate"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"partial-candidate-m0",'
        b'"role":"user","content":[{"type":"input_text","text":"partial candidate"}]}}\n'
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="partial-candidate.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key="codex-session:partial-candidate",
                kind=RawRevisionKind.FULL,
                source_revision="partial-candidate-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.ASSERTED,
            ),
        )
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET baseline_raw_id = raw_id, revision_authority = 'byte_proven'")
        conn.commit()
    backfill_historical_revision_evidence(root)
    schema_receipt = write_valid_rebuild_receipt(root, tmp_path / "schema-receipt.json")
    candidate_paths: list[Path] = []
    original_gate = owner_backfill_module.validate_message_owner_scope_for_index_replacement

    def record_candidate_gate(
        archive_root: Path, *, receipt_path: Path | None, candidate_index_path: Path | None = None
    ) -> None:
        original_gate(archive_root, receipt_path=receipt_path, candidate_index_path=candidate_index_path)
        if candidate_index_path is not None:
            candidate_paths.append(candidate_index_path)

    monkeypatch.setattr(
        owner_backfill_module, "validate_message_owner_scope_for_index_replacement", record_candidate_gate
    )
    rebuilt = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=False,
            raw_ids=(raw_id,),
            schema_inference_receipt_path=schema_receipt,
        )
    )

    assert rebuilt.generation["state"] == "inactive"
    assert candidate_paths == []


def test_prepared_recovery_accepts_a_valid_terminal_receipt_left_after_publication(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after immutable receipt publication only needs marker cleanup.

    Anti-vacuity: the production writer first creates a self-hashed terminal
    receipt, then raises before marker removal.  The second production apply
    must accept that receipt; the old recovery path rejected it as a conflict.
    """

    root, _ = _seed_opaque_archive(workspace_env)
    plan_path, backup, receipt = _plan_and_paths(tmp_path, root)
    backup.parent.mkdir()
    backup.write_text("backup", encoding="utf-8")
    _allow_backup(monkeypatch)
    original_final_receipt = owner_backfill_module._final_receipt

    def publish_then_fail(**kwargs: object) -> None:
        original_final_receipt(**cast(Any, kwargs))
        raise OSError("simulated crash after receipt publication")

    with monkeypatch.context() as post_publication_failure:
        post_publication_failure.setattr(
            "polylogue.maintenance.message_owner_scope_backfill._final_receipt", publish_then_fail
        )
        with pytest.raises(OSError, match="after receipt publication"):
            apply_message_owner_scope_backfill(
                root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
            )

    assert receipt.exists()
    assert receipt.with_name(receipt.name + ".prepared").exists()
    recovered = apply_message_owner_scope_backfill(
        root, plan_path=plan_path, backup_manifest=backup, receipt_path=receipt, dry_run=False
    )

    assert recovered.terminal_state == "committed"
    assert not receipt.with_name(receipt.name + ".prepared").exists()
