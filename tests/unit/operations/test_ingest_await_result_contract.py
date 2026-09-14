"""``operation.await`` must accept the state a completed ingest actually produces.

Anti-vacuity (verified by reverting the fix, not asserted): with
``source_generation_id`` absent from ``MutationResult``, ``extra="forbid"``
rejects the state that ``machine_request_state`` returns for a
``source-generation`` artifact, and ``validate_operation_result`` raises
``OperationResultContractError`` — which the client surfaces as
``DaemonMutationIndeterminateError`` on a clean, committed ingest.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.operations.audit import AuditRepository, MachineRequestBinding
from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceInput, FrozenSourceManifest
from tests.infra.archive_templates import bootstrap_archive_root


def _accepted_ingest_state(tmp_path: Path) -> dict[str, object]:
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    publisher.flush()
    publication_id = publisher.receipt_id(blob_hash)
    assert publication_id is not None
    manifest = FrozenSourceManifest(
        "source-generation:await-contract",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", blob_hash, publication_id),),
    )
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    binding = MachineRequestBinding("archive:test", "request:test", principal.actor_ref, "f" * 64, "ingest")
    audit = AuditRepository.for_archive_root(tmp_path)
    with audit.bind_machine_request(binding, transition="accept_ingest", deadline_unix_ms=1000):
        audit.accept_ingest(manifest, principal)
    record = audit.machine_request(binding)
    assert record is not None
    assert record["artifact_kind"] == "source-generation"
    return machine_request_state(audit, record)


def test_ingest_await_state_satisfies_its_declared_result_contract(tmp_path: Path) -> None:
    state = _accepted_ingest_state(tmp_path)
    # The concrete field that the declared model must admit.
    assert state["source_generation_id"] == "source-generation:await-contract"
    # Must not raise OperationResultContractError.
    validate_operation_result("operation.await", state)
    validate_operation_result("operation.status", state)
