"""Accepted input and audit identity survive every continuity crash boundary."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.operations.audit import AuditRepository, MachineRequestBinding, MachineRequestRecoveredError
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.ingest_acceptance import IngestActuator, ingest_plan
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_transaction import (
    AuthorizationMismatchError,
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationPrincipal,
    OperationExecutor,
)
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_items import (
    FrozenSourceInput,
    FrozenSourceManifest,
    SealedSourceManifestRef,
    append_prepared_source_inputs,
    begin_prepared_source_manifest,
    seal_prepared_source_manifest,
)
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.frozen_clock import FrozenClock
from tests.infra.operation_recovery import recover_on_admitted_owner
from tests.infra.source_builders import prepared_ingest_manifest


def _leased_ingest_manifest(archive_root: Path, *args: Any, **kwargs: Any) -> SealedSourceManifestRef:
    """Prepare the staged manifest under the archive's writer lease, as production does."""
    with write_lease("test.ingest-manifest", archive_root=archive_root):
        return prepared_ingest_manifest(archive_root, *args, **kwargs)


def _authorize(plan: MutationPlan, actuator: IngestActuator, principal: MutationPrincipal) -> MutationAuthorization:
    return OperationExecutor(now_ms=lambda: plan.prepared_at_ms).authorize_bound(
        runtime_operation_binding(actuator), MutationPreview(f"preview:{plan.plan_hash}", plan), principal
    )


def test_new_acceptance_streams_manifest_beyond_former_input_cap(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic shared input")
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = _leased_ingest_manifest(
        tmp_path,
        "large-acceptance",
        "d" * 64,
        (FrozenSourceInput(f"input:{n:05d}", f"/synthetic/{n}", blob_hash, receipt) for n in range(10_241)),
        publisher_id=publisher.publisher_id,
    )
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    binding = MachineRequestBinding("archive:test", "request:large", principal.actor_ref, "f" * 64, "ingest")
    audit = AuditRepository.for_archive_root(tmp_path)
    with audit.bind_machine_request(binding, transition="accept_ingest", deadline_unix_ms=1000):
        audit.accept_ingest(manifest, principal)
    assert audit.machine_request(binding) is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone() == (10_241,)
        assert source.execute("SELECT COUNT(*) FROM prepared_source_manifest_members").fetchone() == (10_241,)
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)


def test_new_plan_refuses_inline_manifest_without_rewriting_historical_evidence() -> None:
    historical = FrozenSourceManifest(
        "historical",
        "d" * 64,
        (FrozenSourceInput("input", "/synthetic/input", "a" * 64, "receipt"),),
    )
    with pytest.raises(TypeError, match="staged source manifest"):
        ingest_plan(
            historical,  # type: ignore[arg-type]
            archive_instance_id="archive",
            archive_identity_digest="b" * 64,
            now_ms=1,
            expires_at_ms=1000,
        )


@pytest.mark.parametrize("phase", ["after_source_prepare", "after_audit_commit", "after_source_promotion"])
def test_ingest_acceptance_replays_identity_without_acquiring(tmp_path: Path, phase: str) -> None:
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    publication_id = publisher.receipt_id(blob_hash)
    assert publication_id is not None
    manifest = _leased_ingest_manifest(
        tmp_path,
        "source-generation:test",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", blob_hash, publication_id),),
        publisher_id=publisher.publisher_id,
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


def test_startup_reclaims_interrupted_preaccept_pages_and_unattached_reservations(tmp_path: Path) -> None:
    """A killed producer cannot leave staged members or GC-immune receipts."""
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    first_hash, _ = publisher.write_from_bytes(b"first page")
    first_receipt = publisher.receipt_id(first_hash)
    assert first_receipt is not None
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("BEGIN IMMEDIATE")
        begin_prepared_source_manifest(
            source,
            source_generation_id="interrupted:page",
            publisher_id=publisher.publisher_id,
            enumeration_fingerprint="d" * 64,
            source_name=None,
        )
        append_prepared_source_inputs(
            source,
            "interrupted:page",
            0,
            tuple(
                FrozenSourceInput(f"input:{ordinal}", f"/synthetic/{ordinal}", first_hash, first_receipt)
                for ordinal in range(256)
            ),
        )
        source.commit()
    # The next page's publication can commit before its member batch does.
    publisher.write_from_bytes(b"unattached second page")
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()

    # Startup recovery runs on the daemon's admitted preparation owner, which
    # charges the original Source rows the cleanup reads; a bare call has no
    # input admission and is refused before any row is read.
    recover_on_admitted_owner(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM prepared_source_manifests").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM prepared_source_manifest_members").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM source_generations").fetchone() == (0,)


@pytest.mark.parametrize("phase", ["after_source_prepare", "after_audit_commit", "after_source_promotion"])
def test_sealed_manifest_acceptance_promotes_every_member_atomically(tmp_path: Path, phase: str) -> None:
    """A crash cannot expose a machine acceptance with partial source membership."""
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    inputs = []
    for ordinal in range(2):
        blob_hash, _ = publisher.write_from_bytes(f"synthetic-{ordinal}".encode())
        receipt_id = publisher.receipt_id(blob_hash)
        assert receipt_id is not None
        inputs.append(FrozenSourceInput(f"input:{ordinal}", f"/synthetic/{ordinal}", blob_hash, receipt_id))
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("BEGIN IMMEDIATE")
        begin_prepared_source_manifest(
            source,
            source_generation_id="sealed:test",
            publisher_id=publisher.publisher_id,
            enumeration_fingerprint="d" * 64,
            source_name=None,
        )
        append_prepared_source_inputs(source, "sealed:test", 0, tuple(inputs))
        manifest = seal_prepared_source_manifest(source, "sealed:test", sealed_at_ms=1)
        source.commit()
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    binding = MachineRequestBinding("archive:test", "request:sealed", principal.actor_ref, "f" * 64, "ingest")
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
    assert record is not None and record["artifact_ref"] == manifest.source_generation_id
    assert machine_request_state(recovered, record)["outcome"] == "accepted"
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id='sealed:test'"
        ).fetchone() == (2,)
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
    recover_on_admitted_owner(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT COUNT(*) FROM prepared_source_manifest_members WHERE source_generation_id='sealed:test'"
        ).fetchone() == (2,)
        assert source.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id='sealed:test'"
        ).fetchone() == (2,)


def test_ingest_principal_is_checked_before_source_prepare(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    manifest = SealedSourceManifestRef("generation:test", "d" * 64, "a" * 64, "b" * 64, 1)
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
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = _leased_ingest_manifest(
        tmp_path,
        "generation:bad",
        "d" * 64,
        (FrozenSourceInput("in", "/in", blob_hash, receipt),),
        publisher_id=publisher.publisher_id,
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
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = _leased_ingest_manifest(
        tmp_path,
        "generation:good",
        "d" * 64,
        (FrozenSourceInput("in", "/in", blob_hash, receipt),),
        "codex",
        publisher_id=publisher.publisher_id,
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
    recovered_preview, _ = recovered.authorization_for_principal(str(part["authorization_ref"]), principal)
    assert recovered_preview.plan.context["source_name"] == "codex"
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


def test_source_name_changes_accepted_manifest_and_preview_identity(frozen_clock: FrozenClock) -> None:
    input_ref = FrozenSourceInput("in", "/in", "a" * 64, "receipt")
    unnamed = FrozenSourceManifest("generation:source", "d" * 64, (input_ref,))
    named = FrozenSourceManifest("generation:source", "d" * 64, (input_ref,), "codex")
    another = FrozenSourceManifest("generation:source", "d" * 64, (input_ref,), "claude-code")

    assert FrozenSourceManifest.from_dict(named.to_dict()) == named
    assert len({unnamed.manifest_digest, named.manifest_digest, another.manifest_digest}) == 3
    plans = [
        ingest_plan(
            manifest,
            archive_instance_id="archive:test",
            archive_identity_digest="archive:test",
            now_ms=int(frozen_clock.time() * 1000),
            expires_at_ms=int(frozen_clock.time() * 1000) + 300_000,
        )
        for manifest in (
            SealedSourceManifestRef(
                m.source_generation_id, m.enumeration_fingerprint, m.manifest_digest, "b" * 64, 1, m.source_name
            )
            for m in (unnamed, named, another)
        )
    ]
    assert len({plan.plan_hash for plan in plans}) == 3


def test_runtime_authority_normal_accept_commits_linked_run(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    """Skipping frozen authority application would violate the machine-part run FK."""

    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    manifest = _leased_ingest_manifest(
        tmp_path,
        "generation:normal",
        "d" * 64,
        (FrozenSourceInput("in", "/in", blob_hash, receipt),),
        publisher_id=publisher.publisher_id,
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


def test_deterministically_failed_accept_ingest_leaves_a_recoverable_archive(tmp_path: Path) -> None:
    """A non-transient accept_ingest failure must not wedge the audit tier.

    polylogue-2kbrl: ``_execute_serialized`` skipped ``_abort_prepared`` for
    ``accept_ingest`` and ``_abort_prepared`` returned early for it, because the
    prepare phase had already committed the frozen manifest to source.db and
    erasing the WAL entry would have stranded that durable work. The pending row
    therefore survived, ``_prepare`` refused every later audit mutation of every
    kind, ``settled_read`` refused every operation read, and recovery replayed
    the same failing payload forever -- an unbounded wedge of a durable,
    append-only, irreplaceable tier, reached by an ordinary failure.

    The prepare phase now only *validates* the manifest and the promotion phase
    publishes it, so the abort is a true rollback: no audit history is
    discarded and no durable source row is stranded.

    Anti-vacuity: restore either ``if mutation.kind != "accept_ingest"`` in
    ``_execute_serialized`` or the ``accept_ingest`` early return in
    ``_abort_prepared`` and the pending assertion below fails with the wedged
    mutation id, followed by ``AuditContinuityError: another audit continuity
    mutation is already pending``.
    """
    bootstrap_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"synthetic export")
    with write_lease("test.ingest-acceptance", archive_root=tmp_path):
        publisher.flush()
    publication_id = publisher.receipt_id(blob_hash)
    assert publication_id is not None
    manifest = _leased_ingest_manifest(
        tmp_path,
        "source-generation:wedge",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", blob_hash, publication_id),),
        publisher_id=publisher.publisher_id,
    )
    principal = MutationPrincipal("actor:test", frozenset({"archive.ingest"}), "cli", "user")
    binding = MachineRequestBinding("archive:test", "request:wedge", principal.actor_ref, "f" * 64, "ingest")

    class _DefectiveCoordinator(AuditContinuityCoordinator):
        """A non-transient defect in the audit mutation body, not a phase hook."""

        def _apply_prepared(self, prepared, apply, *, allow_rebind=False):  # type: ignore[no-untyped-def]
            raise RuntimeError("deterministic audit-mutation defect")

    audit = AuditRepository.for_archive_root(tmp_path)
    audit._continuity = _DefectiveCoordinator(tmp_path)
    with pytest.raises(RuntimeError, match="deterministic audit-mutation defect"):
        with audit.bind_machine_request(binding, transition="accept_ingest", deadline_unix_ms=1000):
            audit.accept_ingest(manifest, principal)

    with sqlite3.connect(tmp_path / "source.db") as source:
        # The wedge: this held the failed mutation's id and never cleared.
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
        # The rolled-back prepare stranded no durable source authority, and the
        # publication reservation it would have consumed is still spendable.
        assert source.execute("SELECT COUNT(*) FROM source_generations").fetchone()[0] == 0
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1

    # Every later audit mutation of every kind was refused; a read must settle.
    recovered = AuditRepository.for_archive_root(tmp_path)
    recovered.reconcile_continuity()
    with recovered.settled_machine_read() as versions:
        assert set(versions) == {"source", "audit"}
    assert recovered.machine_request(binding) is None

    # And the same acceptance succeeds once the defect is gone.
    healthy = AuditRepository.for_archive_root(tmp_path)
    with healthy.bind_machine_request(binding, transition="accept_ingest", deadline_unix_ms=1000):
        healthy.accept_ingest(manifest, principal)
    record = healthy.machine_request(binding)
    assert record is not None and record["artifact_ref"] == manifest.source_generation_id
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM source_generations").fetchone()[0] == 1
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
