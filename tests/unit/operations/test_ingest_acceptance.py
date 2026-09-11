"""Accepted input and audit identity survive every continuity crash boundary."""

import sqlite3
from pathlib import Path

import pytest

from polylogue.operations.audit import AuditRepository, MachineRequestBinding, MachineRequestRecoveredError
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import AuthorizationMismatchError, MutationPrincipal
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceInput, FrozenSourceManifest
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from tests.infra.archive_templates import bootstrap_archive_root


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
