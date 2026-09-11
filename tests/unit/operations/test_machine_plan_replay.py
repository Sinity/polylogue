"""Real audited plans retain their authority across source-WAL replay."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.operations.audit import AuditRepository
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_actuators import (
    BulkMetadataSetActuator,
    BulkMetadataSetArgs,
    BulkTagActuator,
    BulkTagArgs,
    SessionDeleteActuator,
    SessionDeleteArgs,
)
from polylogue.operations.mutation_transaction import (
    MutationPrincipal,
    OperationExecutor,
    compute_parameter_digest,
    validate_mutation_plan_integrity,
)
from polylogue.storage.archive_identity import ArchiveIdentity
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.storage_records import SessionBuilder


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("family", ["delete", "tag", "metadata"])
def test_real_plan_replays_before_authorization_without_losing_its_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, batch: bool
) -> None:
    """Mutation: replay only context_sha256 and authorize rejects the reconstructed plan."""
    bootstrap_archive_root(tmp_path)
    builder = (
        SessionBuilder(tmp_path / "index.db", "replay-context").provider("codex").add_message(text="Neutral fixture")
    )
    builder.save()
    ids = (builder.native_session_id(),)
    principal = MutationPrincipal(
        "synthetic-replay", frozenset({"archive.delete_session", "archive.tag_session", "archive.set_metadata"}), "cli"
    )
    audit = AuditRepository.for_archive_root(tmp_path)
    instance = audit.ensure_archive_authority(now_ms=1000)
    executor = OperationExecutor(now_ms=lambda: 1000)
    with ArchiveStore.open_existing(tmp_path) as archive:
        if family == "delete":
            actuator, args = SessionDeleteActuator(), SessionDeleteArgs(archive, ids)
        elif family == "tag":
            actuator, args = BulkTagActuator(), BulkTagArgs(archive, ids, ("neutral-tag",))
        else:
            actuator, args = BulkMetadataSetActuator(), BulkMetadataSetArgs(archive, ids, (("purpose", "fixture"),))
        operation = runtime_operation_binding(actuator)
        # Use the operation's declared capability rather than duplicating its vocabulary.
        principal = replace(
            principal,
            capabilities=frozenset(
                capability for policy in operation.spec.target_authority for capability in policy.required_capabilities
            ),
        )
        raw = actuator.prepare(args)
        preview = executor.prepare_bound(
            operation,
            args,
            principal,
            archive_instance_id=instance,
            archive_identity_digest=ArchiveIdentity.resolve(tmp_path).authority_identity_digest,
            parameter_digest=compute_parameter_digest(raw),
            expires_at_ms=2**62,
            raw_plan=raw,
        )
    target_kind = "create_preview_batch" if batch else "create_preview"
    original_phase = AuditContinuityCoordinator._phase

    def crash(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == target_kind and phase == "after_source_prepare":
            raise RuntimeError("synthetic source-WAL crash")
        original_phase(self, phase, mutation)

    with monkeypatch.context() as patch:
        patch.setattr(AuditContinuityCoordinator, "_phase", crash)
        with pytest.raises(RuntimeError, match="source-WAL crash"):
            if batch:
                audit.create_preview_batch((preview.plan,), principal)
            else:
                audit.create_preview(preview.plan, principal)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        pending = json.loads(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
        assert "polylogue.machine-plan-context/v1" in json.dumps(pending)
        assert "authorization_token" not in json.dumps(pending)

    restarted = AuditRepository.for_archive_root(tmp_path)
    restarted.reconcile_continuity()
    with closing(sqlite3.connect(tmp_path / "audit.db")) as connection:
        ref = connection.execute(
            "SELECT preview_id FROM operation_previews WHERE plan_hash=?", (preview.plan.plan_hash,)
        ).fetchone()[0]
    recovered = restarted.preview_for_principal(ref, principal)
    assert recovered.plan.context == preview.plan.context
    assert recovered.plan.plan_hash == preview.plan.plan_hash
    validate_mutation_plan_integrity(recovered.plan)
    authorization = executor.authorize_bound(operation, recovered, principal, confirmation_strength="bound_token")
    target_kind = "issue_authorization"
    with monkeypatch.context() as patch:
        patch.setattr(AuditContinuityCoordinator, "_phase", crash)
        with pytest.raises(RuntimeError, match="source-WAL crash"):
            restarted.issue_authorization(recovered, principal, authorization)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        pending = json.loads(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
    issued_at_ms = pending["command"]["payload"]["issued_at_ms"]
    assert isinstance(issued_at_ms, int)
    assert authorization.token not in json.dumps(pending)
    restarted.reconcile_continuity()
    with closing(sqlite3.connect(tmp_path / "audit.db")) as connection:
        assert (
            connection.execute(
                "SELECT issued_at_ms FROM operation_authorizations WHERE preview_id=?", (ref,)
            ).fetchone()[0]
            == issued_at_ms
        )


def test_machine_replay_refuses_unknown_context_fields() -> None:
    from polylogue.operations.machine_plan_context import replay_context

    with pytest.raises(ValueError):
        replay_context("mutate-delete-session", {"session_ids": [], "credential": "not-a-plan-field"})
