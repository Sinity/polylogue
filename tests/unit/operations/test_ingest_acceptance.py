"""Accepted input and audit identity survive every continuity crash boundary."""

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.operations.audit import AuditRepository, MachineRequestBinding, MachineRequestRecoveredError
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.ingest_acceptance import IngestActuator, ingest_plan
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import (
    AuthorizationMismatchError,
    MutationPreview,
    MutationPrincipal,
    OperationExecutor,
)
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceInput, FrozenSourceManifest
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.frozen_clock import FrozenClock


def _authorize(plan, actuator: IngestActuator, principal: MutationPrincipal):
    return OperationExecutor(now_ms=lambda: plan.prepared_at_ms).authorize_bound(
        runtime_operation_binding(actuator), MutationPreview(f"preview:{plan.plan_hash}", plan), principal
    )


@pytest.mark.parametrize("phase", ["after_source_prepare", "after_audit_commit", "after_source_promotion"])
def test_ingest_acceptance_replays_identity_without_acquiring(tmp_path: Path, phase: str) -> None:
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    publisher.flush()
    publication_id = publisher.receipt_id(blob_hash)
    assert publication_id is not None
    manifest = FrozenSourceManifest(
        "source-generation:test",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", blob_hash, publication_id),),
    )
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    binding = MachineRequestBinding("archive:test", "request:test", principal.actor_ref, "f" * 64, "ingest")
    audit = AuditRepository.for_archive_root(tmp_path)

    def crash(at: str, _mutation: AuditMutation) -> None:
        if at == phase:
            raise RuntimeError("synthetic interruption")

    audit._continuity = AuditContinuityCoordinator(tmp_path, phase_hook=crash)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with audit.bind_machine_request(binding, transition="accept_ingest", deadline_unix_ms=1000):
            audit.accept_ingest(manifest, principal)

    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()
    record = recovered.machine_request(binding)
    assert record is not None
    assert record["artifact_kind"] == "source-generation"
    assert record["artifact_ref"] == manifest.source_generation_id
    assert record["accepted_deadline_unix_ms"] == 1000
    assert machine_request_state(recovered, record)["outcome"] == "accepted"
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 1
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
    with pytest.raises(MachineRequestRecoveredError):
        with recovered.bind_machine_request(binding, transition="accept_ingest"):
            recovered.accept_ingest(manifest, principal)


def test_ingest_principal_is_checked_before_source_prepare(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    manifest = FrozenSourceManifest(
        "generation:test",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", "a" * 64, "not-published"),),
    )
    audit = AuditRepository.for_archive_root(tmp_path)
    principal = MutationPrincipal("actor:test", frozenset({"read"}), "cli", "user")
    binding = MachineRequestBinding("archive:test", "request:test", principal.actor_ref, "f" * 64, "ingest")
    with pytest.raises(AuthorizationMismatchError, match="lacks bound authority"):
        with audit.bind_machine_request(binding, transition="accept_ingest"):
            audit.accept_ingest(manifest, principal)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 0
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None


def test_malformed_runtime_authority_never_prepares_source_manifest(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    """Moving auth validation after source prepare would consume this reservation."""

    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = FrozenSourceManifest("generation:bad", "d" * 64, (FrozenSourceInput("in", "/in", blob_hash, receipt),))
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    now_ms = int(frozen_clock.time() * 1000)
    plan = ingest_plan(
        manifest,
        archive_instance_id="archive:test",
        archive_identity_digest="archive:test",
        now_ms=now_ms,
        expires_at_ms=now_ms + 300_000,
    )
    actuator = IngestActuator(manifest, "archive:test", "archive:test", now_ms, now_ms + 300_000)
    authorization = _authorize(plan, actuator, MutationPrincipal("other", frozenset({"archive.ingest"}), "cli", "user"))
    audit = AuditRepository.for_archive_root(tmp_path)
    binding = MachineRequestBinding("archive:test", "request:bad", principal.actor_ref, "f" * 64, "ingest")
    with pytest.raises(AuthorizationMismatchError):
        with audit.bind_machine_request(binding, transition="accept_ingest"):
            audit.accept_ingest(manifest, principal, plan=plan, authorization=authorization)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 0
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1


def test_runtime_authority_replay_preserves_frozen_ids_and_machine_part(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    """Replay minting replacement authority would change these durable references."""

    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = FrozenSourceManifest("generation:good", "d" * 64, (FrozenSourceInput("in", "/in", blob_hash, receipt),))
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    now_ms = int(frozen_clock.time() * 1000)
    plan = ingest_plan(
        manifest,
        archive_instance_id="archive:test",
        archive_identity_digest="archive:test",
        now_ms=now_ms,
        expires_at_ms=now_ms + 300_000,
    )
    actuator = IngestActuator(manifest, "archive:test", "archive:test", now_ms, now_ms + 300_000)
    authorization = _authorize(plan, actuator, principal)
    audit = AuditRepository.for_archive_root(tmp_path)
    binding = MachineRequestBinding("archive:test", "request:good", principal.actor_ref, "e" * 64, "ingest")

    def crash(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("crash")

    audit._continuity = AuditContinuityCoordinator(tmp_path, phase_hook=crash)
    with pytest.raises(RuntimeError):
        with audit.bind_machine_request(binding, transition="accept_ingest"):
            audit.accept_ingest(manifest, principal, plan=plan, authorization=authorization)
    with sqlite3.connect(tmp_path / "source.db") as source:
        payload = json.loads(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
    expected = payload["command"]["payload"]
    assert "token" not in expected["authorization"]
    assert isinstance(expected["authorization_token_sha256"], str)
    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()
    part = recovered.machine_parts(binding)[0]
    assert (part["preview_ref"], part["authorization_ref"], part["operation_id"]) == (
        expected["preview_id"],
        expected["authorization_id"],
        expected["operation_id"],
    )
    with sqlite3.connect(tmp_path / "audit.db") as audit_db:
        assert audit_db.execute("SELECT attempt_id, started_at_ms FROM operation_attempts").fetchone() == (
            expected["attempt_id"],
            expected["now_ms"],
        )
        assert audit_db.execute("SELECT issued_at_ms, token_sha256 FROM operation_authorizations").fetchone() == (
            expected["issued_at_ms"],
            expected["authorization_token_sha256"],
        )


def test_runtime_authority_normal_accept_commits_linked_run(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    """Skipping frozen authority application would violate the machine-part run FK."""

    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = FrozenSourceManifest(
        "generation:normal", "d" * 64, (FrozenSourceInput("in", "/in", blob_hash, receipt),)
    )
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    now_ms = int(frozen_clock.time() * 1000)
    plan = ingest_plan(
        manifest,
        archive_instance_id="archive:test",
        archive_identity_digest="archive:test",
        now_ms=now_ms,
        expires_at_ms=now_ms + 300_000,
    )
    actuator = IngestActuator(manifest, "archive:test", "archive:test", now_ms, now_ms + 300_000)
    authorization = _authorize(plan, actuator, principal)
    audit = AuditRepository.for_archive_root(tmp_path)
    binding = MachineRequestBinding("archive:test", "request:normal", principal.actor_ref, "c" * 64, "ingest")
    with audit.bind_machine_request(binding, transition="accept_ingest"):
        assert (
            audit.accept_ingest(manifest, principal, plan=plan, authorization=authorization)
            == manifest.source_generation_id
        )
    part = audit.machine_parts(binding)[0]
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit_db:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
        assert audit_db.execute("SELECT COUNT(*) FROM operation_previews").fetchone()[0] == 1
        assert audit_db.execute("SELECT COUNT(*) FROM operation_authorizations").fetchone()[0] == 1
        assert audit_db.execute("SELECT operation_id FROM operation_runs").fetchone()[0] == part["operation_id"]
