from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel

from polylogue.operations.audit import (
    AuditRepository,
    _attempt_owner_is_live,
    _attempt_owner_liveness,
    _current_process_attempt_owner,
    token_sha256,
)
from polylogue.operations.bindings import OperationBinding
from polylogue.operations.mutation_transaction import (
    AuditFinalizationError,
    AuthorizationMismatchError,
    CapabilityDeniedError,
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationPrincipal,
    MutationReceipt,
    OperationExecutor,
    PlanStaleError,
    RecoveryBlockedError,
    RecoveryDisposition,
    RecoveryOperation,
    RecoveryTargetDisposition,
    TargetAuthorityPolicy,
    TargetDurability,
    TokenConsumedError,
    TokenExpiredError,
    build_plan,
    recover_interrupted_operations,
)
from polylogue.operations.specs import OperationKind, OperationSpec
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    VerifiedAuditLeaf,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
)


@dataclass
class _Actuator:
    operation: str = "mutate-fixture"
    operation_version: int = 1
    changed: bool = False
    calls: int = 0
    crash: bool = False
    recovery_disposition: RecoveryDisposition = field(
        default_factory=lambda: RecoveryDisposition("confirmed-not-applied", "retry-exact")
    )
    recovery_raises: bool = False
    inspections: int = 0
    target_refs: tuple[str, ...] = ("session:fixture",)
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, _args: object) -> MutationPlan:
        targets = ("session:changed",) if self.changed else self.target_refs
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=targets,
            affected_tiers=("user",),
            reversible=True,
        )

    def apply(self, plan: MutationPlan, _args: object) -> MutationReceipt:
        self.calls += 1
        if self.crash:
            raise RuntimeError("simulated actuator crash")
        return MutationReceipt(
            operation=plan.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=len(plan.target_refs),
            detail=None,
            receipt_ref=None,
            applied_at="now",
        )

    def inspect_recovery(self, operation: RecoveryOperation, _args: object) -> RecoveryDisposition:
        self.inspections += 1
        if self.recovery_raises:
            raise OSError("synthetic target is unreadable")
        disposition = self.recovery_disposition
        if disposition.kind == "unknown" or disposition.target_dispositions:
            return disposition
        refs = tuple(target.ref for target in operation.targets)
        state: Literal["applied", "not-applied", "unknown"] = (
            "applied" if disposition.kind == "confirmed-applied" else "not-applied"
        )
        return replace(
            disposition,
            target_dispositions=tuple(RecoveryTargetDisposition(ref, state, disposition.action) for ref in refs),
        )


@dataclass(frozen=True)
class _TypedDomainBatch:
    batch_ref: str
    rows: tuple[str, ...]
    _cached_bytes: bytes = field(init=False, repr=False, default=b"private-cache")


class _TypedDomainOutcome(BaseModel):
    row_ref: str
    status: str


@dataclass
class _TypedReceiptActuator(_Actuator):
    def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
        receipt = super().apply(plan, args)
        return replace(
            receipt,
            domain_receipt={
                "batch": _TypedDomainBatch("annotation-batch:typed", ("assertion:typed",)),
                "outcomes": (_TypedDomainOutcome(row_ref="assertion:typed", status="imported"),),
            },
        )


@dataclass
class _AlreadySatisfiedActuator(_Actuator):
    def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
        return replace(super().apply(plan, args), status="already_satisfied", affected_count=0)


@dataclass
class _FailedReceiptActuator(_Actuator):
    def apply(self, plan: MutationPlan, args: object) -> MutationReceipt:
        return replace(super().apply(plan, args), status="failed", affected_count=0)


def _binding(
    actuator: _Actuator, *, target_durability: TargetDurability = "derived"
) -> OperationBinding[object, object]:
    spec = OperationSpec(
        name="mutate-fixture",
        kind=OperationKind.MAINTENANCE,
        description="fixture",
        mutates_state=True,
        executor_status="executor-routed",
        allowed_surfaces=("internal",),
        target_authority=(
            TargetAuthorityPolicy(
                key="session",
                target_kinds=("session",),
                required_capabilities=("archive.fixture.write",),
                destructive_class="reversible",
                required_confirmation="role_only",
                allowed_durabilities=(target_durability,),
                allowed_recovery=("none",),
            ),
        ),
        affected_tiers=("user",),
    )
    return OperationBinding(spec, actuator)


def _principal() -> MutationPrincipal:
    return MutationPrincipal("actor:test", frozenset({"archive.fixture.write"}), "internal", "system")


def _audit(tmp_path: Path) -> AuditRepository:
    initialize_active_archive_root(tmp_path)
    return AuditRepository.for_archive_root(tmp_path)


def test_token_is_digest_only_and_consumption_run_attempt_are_atomic(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.ensure_archive_authority(now_ms=1)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "raw-secret-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    assert "raw-secret-token" not in (tmp_path / "audit.db").read_bytes().decode("utf-8", errors="ignore")
    digest_as_bearer = replace(authorization, token=f"sha256:{token_sha256('raw-secret-token')}")
    with pytest.raises(AuthorizationMismatchError, match="does not match preview"):
        audit.consume_authorization_and_start(preview, digest_as_bearer)
    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert receipt.operation_id is not None
    operation = audit.get_operation(receipt.operation_id)
    assert operation is not None
    assert operation["status"] == "completed"
    assert operation["affected_count"] == 1
    assert audit.list_events(receipt.operation_id)[-1]["event_type"] == "attempt_finalized"
    with pytest.raises(RuntimeError, match="consumed"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert actuator.calls == 1


def test_execute_bound_routes_the_effect_through_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound execution reaches the executor's sole actuator.apply gate."""

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "bound-route-token")
    binding = _binding(actuator)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    calls: list[object] = []
    original_execute = OperationExecutor.execute

    def record_execute(
        self: OperationExecutor, invoked_actuator: object, plan: object, auth: object, args: object
    ) -> object:
        calls.append(invoked_actuator)
        return original_execute(self, invoked_actuator, plan, auth, args)  # type: ignore[arg-type]

    monkeypatch.setattr(OperationExecutor, "execute", record_execute)

    receipt = executor.execute_bound(binding, preview, authorization, object())

    assert receipt.status == "applied"
    assert calls == [actuator]
    assert actuator.calls == 1


def test_prepare_bound_uses_the_declared_durable_target_for_legacy_actuators() -> None:
    """Fallback target construction cannot downgrade a durable policy to derived."""

    actuator = _Actuator()
    preview = OperationExecutor().prepare_bound(
        _binding(actuator, target_durability="durable"),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )

    assert preview.plan.targets[0].durability == "durable"
    assert preview.plan.targets[0].recovery == "none"


def test_production_executor_factory_persists_audit_preview(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor.for_archive_root(tmp_path, token_factory=lambda: "factory-token")

    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT preview_id FROM operation_previews").fetchone()[0] == preview.preview_ref


def test_audit_authority_rejects_a_symlinked_audit_leaf_without_touching_its_target(tmp_path: Path) -> None:
    """Bootstrap and direct audit access never follow an audit path outside its archive root."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    external_audit = tmp_path.parent / "external-audit.db"
    external_audit.write_bytes(audit_path.read_bytes())
    audit_path.unlink()
    audit_path.symlink_to(external_audit)
    before = external_audit.read_bytes()

    with pytest.raises(RuntimeError, match="archive-owned regular file"):
        initialize_active_archive_root(tmp_path)
    with pytest.raises(RuntimeError, match="archive-owned regular file"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert external_audit.read_bytes() == before


def test_audit_authority_rejects_a_hardlinked_audit_leaf_without_touching_its_target(tmp_path: Path) -> None:
    """A regular-looking audit leaf must still have exactly one archive-owned link."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    external_audit = tmp_path.parent / "external-hardlinked-audit.db"
    external_audit.write_bytes(audit_path.read_bytes())
    audit_path.unlink()
    audit_path.hardlink_to(external_audit)
    before = external_audit.read_bytes()

    with pytest.raises(RuntimeError, match="one link"):
        initialize_active_archive_root(tmp_path)
    with pytest.raises(RuntimeError, match="one link"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert external_audit.read_bytes() == before


def test_audit_authority_rejects_a_foreign_owned_audit_leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The authority leaf must belong to this effective archive owner.

    Anti-vacuity: removing the uid comparison accepts the otherwise-valid
    single-linked regular file and allows the authority check to proceed.
    """

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    before = audit_path.read_bytes()
    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.os.geteuid", lambda: audit_path.stat().st_uid + 1)

    with pytest.raises(RuntimeError, match="current effective user"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert audit_path.read_bytes() == before


def test_audit_leaf_uses_the_verified_native_directory_when_descriptor_children_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A portable native descriptor path is used before pseudo-filesystem traversal.

    Anti-vacuity: removing the F_GETPATH-style route makes this macOS-shaped
    host fail closed because neither pseudo-filesystem child is available.
    """

    initialize_active_archive_root(tmp_path)

    def native_path_from_descriptor(_fd: int, _request: int, _buffer: bytes) -> bytes:
        return os.fsencode(tmp_path) + b"\0"

    monkeypatch.setattr(VerifiedAuditLeaf, "_descriptor_child_path", lambda _self: None)
    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.fcntl.F_GETPATH", 50, raising=False)
    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.fcntl.fcntl", native_path_from_descriptor)
    with VerifiedAuditLeaf(tmp_path) as leaf:
        assert leaf.anchored_path == tmp_path / "audit.db"


def test_audit_leaf_closes_its_directory_descriptor_after_validation_failure(tmp_path: Path) -> None:
    """Rejected leaves do not retain one descriptor per failed authority request.

    Anti-vacuity: the old OSError-only cleanup leaves ``_directory_fd`` set
    after this symlink validation error.
    """

    target = tmp_path.parent / "external-audit-leaf.db"
    target.write_bytes(b"external")
    (tmp_path / "audit.db").symlink_to(target)
    leaf = VerifiedAuditLeaf(tmp_path)

    with pytest.raises(AuditLeafError):
        leaf.__enter__()

    assert leaf._directory_fd is None
    assert leaf._leaf_fd is None


def test_audit_leaf_rejects_a_foreign_sidecar_without_leaking_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SQLite namespace is validated with the main leaf before any writer opens.

    Anti-vacuity: checking only audit.db accepts this foreign-owned WAL leaf
    and lets SQLite consume attacker-controlled sidecar bytes.
    """

    initialize_active_archive_root(tmp_path)
    sidecar = tmp_path / "audit.db-wal"
    sidecar.write_bytes(b"not a sqlite wal")
    leaf = VerifiedAuditLeaf(tmp_path)
    real_stat = sidecar.stat()
    real_os_stat = os.stat

    def foreign_sidecar_metadata(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: Any, **kwargs: Any
    ) -> os.stat_result:
        metadata = real_os_stat(path, *args, **kwargs)
        if path == "audit.db-wal":
            values = list(metadata)
            values[4] = real_stat.st_uid + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.os.stat", foreign_sidecar_metadata)

    with pytest.raises(AuditLeafError, match="sidecar.*current effective user"):
        leaf.__enter__()

    assert leaf._directory_fd is None
    assert leaf._leaf_fd is None


def test_audit_leaf_rejects_group_writable_archive_directory(tmp_path: Path) -> None:
    """A second Unix principal cannot plant an SQLite sidecar in the authority namespace."""

    initialize_active_archive_root(tmp_path)
    tmp_path.chmod(0o770)
    leaf = VerifiedAuditLeaf(tmp_path)

    with pytest.raises(AuditLeafError, match="directory must not be writable by group or other"):
        leaf.__enter__()

    assert leaf._directory_fd is None
    assert leaf._leaf_fd is None


def test_audit_leaf_rejects_group_writable_main_and_sidecar_files(tmp_path: Path) -> None:
    """UID equality alone cannot grant exclusive write authority over SQLite files."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    audit_path.chmod(0o660)
    with pytest.raises(AuditLeafError, match="audit tier must not be writable by group or other"):
        VerifiedAuditLeaf(tmp_path).__enter__()

    audit_path.chmod(0o600)
    sidecar = tmp_path / "audit.db-wal"
    sidecar.write_bytes(b"not a sqlite wal")
    sidecar.chmod(0o660)
    with pytest.raises(AuditLeafError, match="sidecar must not be writable by group or other"):
        VerifiedAuditLeaf(tmp_path).__enter__()


def test_audit_leaf_serializes_writers_across_the_main_and_sidecar_namespace(tmp_path: Path) -> None:
    """A second writer cannot validate then race the first SQLite namespace owner.

    Anti-vacuity: without the nonblocking main-leaf lock, both contexts open
    and can independently create or replace the audit sidecar namespace.
    """

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    with open_verified_audit_connection(audit_path):
        with pytest.raises(AuditLeafError, match="active writer"):
            with open_verified_audit_connection(audit_path):
                pass


def test_audit_authority_rejects_a_leaf_replaced_during_sqlite_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite never yields a connection after the descriptor-checked leaf changes."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    replacement = tmp_path / "replacement-audit.db"
    replacement.write_bytes(audit_path.read_bytes())
    before = replacement.read_bytes()
    original_connect = cast(Callable[..., sqlite3.Connection], sqlite3.connect)
    swapped = False

    def replace_after_open(database: object, *args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal swapped
        connection = original_connect(database, *args, **kwargs)
        database_text = str(database)
        if (
            not swapped
            and database_text.split("?", 1)[0].endswith("/audit.db")
            and ("/dev/fd/" in database_text or "/proc/self/fd/" in database_text)
        ):
            swapped = True
            audit_path.unlink()
            replacement.replace(audit_path)
        return connection

    monkeypatch.setattr("polylogue.storage.sqlite.audit_leaf.sqlite3.connect", replace_after_open)

    with pytest.raises(RuntimeError, match="changed during SQLite open"):
        AuditRepository.for_archive_root(tmp_path).ensure_archive_authority(now_ms=1)

    assert swapped
    assert audit_path.read_bytes() == before


def test_verified_audit_writer_rejects_a_wal_replacement_before_first_application_begin(tmp_path: Path) -> None:
    """The production writer pins WAL/SHM before a caller can start its transaction."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    wal_path = audit_path.with_name("audit.db-wal")

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        with open_verified_audit_connection(audit_path) as connection:
            replacement = tmp_path / "replacement-audit.db-wal"
            replacement.write_bytes(wal_path.read_bytes())
            replacement.replace(wal_path)
            connection.execute("BEGIN IMMEDIATE")


def test_verified_audit_reader_observes_a_committed_live_wal_head(tmp_path: Path) -> None:
    """Read-only authority checks include commits still resident in the live WAL."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    archive_id = "archive:live-wal-read"

    with open_verified_audit_connection(audit_path) as writer:
        writer.execute(
            "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, 1, 1)",
            (archive_id,),
        )
        writer.commit()
        assert audit_path.with_name("audit.db-wal").exists()

        with open_verified_audit_read_connection(audit_path) as reader:
            assert reader.execute(
                "SELECT archive_instance_id FROM archive_authority WHERE archive_instance_id = ?", (archive_id,)
            ).fetchone() == (archive_id,)


def test_verified_audit_writer_coexists_with_an_older_read_transaction(tmp_path: Path) -> None:
    """Persistent WAL mode lets a later writer proceed while a reader retains its snapshot."""

    initialize_active_archive_root(tmp_path)
    audit_path = tmp_path / "audit.db"
    with open_verified_audit_connection(audit_path) as writer:
        assert writer.execute("PRAGMA journal_mode").fetchone() == ("wal",)

    with open_verified_audit_read_connection(audit_path) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT generation FROM audit_continuity_head").fetchone()
        with open_verified_audit_connection(audit_path) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE audit_continuity_head SET advanced_at_ms = advanced_at_ms")
            writer.commit()


def test_production_factory_does_not_abandon_a_live_same_process_attempt(tmp_path: Path) -> None:
    """A second composition-root call recognizes the first executor's owner."""
    initialize_active_archive_root(tmp_path)
    actuator = _Actuator()
    first = OperationExecutor.for_archive_root(tmp_path, token_factory=lambda: "first-owner-token")
    preview = first.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:live-owner",
        archive_identity_digest="identity:live-owner",
        parameter_digest="params:live-owner",
    )
    authorization = first.authorize_bound(_binding(actuator), preview, _principal())
    assert first._audit is not None
    operation_id = first._audit.consume_authorization_and_start(preview, authorization)

    OperationExecutor.for_archive_root(tmp_path)

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert (
            conn.execute(
                "SELECT state, worker_id FROM operation_attempts WHERE operation_id = ?", (operation_id,)
            ).fetchone()[0]
            == "running"
        )


def test_recovery_marks_a_dead_process_owned_attempt_unknown(tmp_path: Path) -> None:
    """Restart recovery remains active when the recorded owner no longer exists."""
    initialize_active_archive_root(tmp_path)
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "dead-owner-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:dead-owner",
        archive_identity_digest="identity:dead-owner",
        parameter_digest="params:dead-owner",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()

    assert audit.recover_abandoned_attempts() == (operation_id,)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "interrupted",
        )


def _dead_nonterminal_operation(tmp_path: Path, actuator: _Actuator) -> tuple[AuditRepository, str]:
    """Create the production durable boundary immediately after attempt start."""

    audit = _audit(tmp_path)
    first = OperationExecutor(audit=audit, token_factory=lambda: "crash-boundary-token")
    binding = _binding(actuator)
    preview = first.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = first.authorize_bound(binding, preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    return audit, operation_id


def test_dead_attempt_is_inspected_before_overlapping_apply_and_exact_retry_is_single_effect(tmp_path: Path) -> None:
    """A crash after attempt start is classified from the domain before retry.

    Anti-vacuity: deleting overlap discovery, target inspection, or the
    durable recovery finalization either leaves the first operation running or
    permits the new apply without recording the classification.
    """

    actuator = _Actuator()
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    retry = OperationExecutor(audit=audit, token_factory=lambda: "retry-boundary-token")
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, _principal())

    receipt = retry.execute_bound(binding, preview, authorization, object())

    assert receipt.status == "applied"
    assert actuator.inspections == 1
    assert actuator.calls == 1
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "failed",
        )
        assert conn.execute(
            "SELECT event_type FROM operation_events WHERE operation_id = ? ORDER BY sequence DESC LIMIT 1",
            (operation_id,),
        ).fetchone() == ("recovery_classified",)


@pytest.mark.parametrize(
    ("disposition", "raises"),
    [
        (RecoveryDisposition("unknown", "operator-blocking", "unreadable target"), False),
        (RecoveryDisposition("confirmed-not-applied", "retry-exact"), True),
    ],
)
def test_unreadable_or_unknown_recovery_never_consumes_overlapping_authority(
    tmp_path: Path, disposition: RecoveryDisposition, raises: bool
) -> None:
    """Unknown target evidence blocks before a second authorization is consumed."""

    actuator = _Actuator(recovery_disposition=disposition, recovery_raises=raises)
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    retry = OperationExecutor(audit=audit, token_factory=lambda: "blocked-retry-token")
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, _principal())

    with pytest.raises(RecoveryBlockedError):
        retry.execute_bound(binding, preview, authorization, object())

    assert actuator.calls == 0
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "failed",
        )
        assert conn.execute(
            "SELECT state FROM operation_authorizations WHERE token_sha256 = ?", (token_sha256("blocked-retry-token"),)
        ).fetchone() == ("active",)


def test_confirmed_applied_recovery_refuses_duplicate_effect(tmp_path: Path) -> None:
    """A target-specific applied classification cannot be replayed as a new effect."""

    actuator = _Actuator(recovery_disposition=RecoveryDisposition("confirmed-applied", "forward"))
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    retry = OperationExecutor(audit=audit, token_factory=lambda: "duplicate-retry-token")
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, _principal())

    with pytest.raises(RecoveryBlockedError, match="already applied"):
        retry.execute_bound(binding, preview, authorization, object())

    assert actuator.calls == 0
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "completed",
        )


def test_confirmed_applied_recovery_installs_a_durable_barrier_for_every_later_retry(tmp_path: Path) -> None:
    """The first blocked retry must not make the second retry a fresh effect.

    Anti-vacuity: this goes through a second preview and authorization after
    the original operation is terminal, so overlap-only implementations pass
    the first refusal and fail the second.
    """

    actuator = _Actuator(recovery_disposition=RecoveryDisposition("confirmed-applied", "forward"))
    audit, _operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    binding = _binding(actuator)
    retry = OperationExecutor(audit=audit, token_factory=lambda: "duplicate-first")
    first_preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    first_authorization = retry.authorize_bound(binding, first_preview, _principal())
    with pytest.raises(RecoveryBlockedError, match="already applied"):
        retry.execute_bound(binding, first_preview, first_authorization, object())

    second = OperationExecutor(audit=audit, token_factory=lambda: "duplicate-second")
    second_preview = second.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    second_authorization = second.authorize_bound(binding, second_preview, _principal())
    with pytest.raises(RecoveryBlockedError, match="semantic effect applied"):
        second.execute_bound(binding, second_preview, second_authorization, object())
    assert actuator.calls == 0


def test_bounded_adjudication_can_unwedge_a_recovered_applied_barrier(tmp_path: Path) -> None:
    """Operator evidence can replace a prior recovered-applied classification.

    Anti-vacuity: the first retry creates the durable barrier. Restricting
    adjudication to interrupted runs leaves the final retry blocked forever.
    """

    actuator = _Actuator(recovery_disposition=RecoveryDisposition("confirmed-applied", "forward"))
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    binding = _binding(actuator)
    first = OperationExecutor(audit=audit, token_factory=lambda: "barrier-first")
    preview = first.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = first.authorize_bound(binding, preview, _principal())
    with pytest.raises(RecoveryBlockedError, match="already applied"):
        first.execute_bound(binding, preview, authorization, object())

    audit.adjudicate_recovery(
        operation_id,
        target_outcomes={"session:fixture": "not-applied"},
        reason="operator verified the recovered effect was not durable",
    )

    retry = OperationExecutor(audit=audit, token_factory=lambda: "barrier-unwedged")
    retry_preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    retry_authorization = retry.authorize_bound(binding, retry_preview, _principal())
    assert retry.execute_bound(binding, retry_preview, retry_authorization, object()).status == "applied"
    assert actuator.calls == 1


def test_applied_adjudication_retains_the_recovered_effect_barrier(tmp_path: Path) -> None:
    """An applied operator decision still refuses every duplicate effect.

    Anti-vacuity: replacing the recovered-applied terminal reason with the
    ordinary completed reason lets the retry reach the actuator.
    """

    actuator = _Actuator()
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    assert audit.recover_abandoned_attempts() == (operation_id,)
    audit.adjudicate_recovery(
        operation_id,
        target_outcomes={"session:fixture": "applied"},
        reason="operator verified the interrupted effect committed",
    )

    retry = OperationExecutor(audit=audit, token_factory=lambda: "adjudicated-applied-retry")
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, _principal())
    with pytest.raises(RecoveryBlockedError, match="semantic effect applied"):
        retry.execute_bound(binding, preview, authorization, object())

    assert actuator.calls == 0
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("completed", "recovered_applied")


def test_recovery_disposition_keeps_create_update_delete_outcomes_per_target(tmp_path: Path) -> None:
    """A mixed mutation remains a per-target audit record, never a delete shortcut."""

    actuator = _Actuator(target_refs=("session:create", "session:update", "session:delete"))
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    audit.record_recovery_disposition(
        operation_id,
        RecoveryDisposition(
            "confirmed-partial",
            "retry-exact",
            "domain inspector distinguished create, update, and delete",
            (
                RecoveryTargetDisposition("session:create", "applied", "forward"),
                RecoveryTargetDisposition("session:update", "not-applied", "retry-exact"),
                RecoveryTargetDisposition("session:delete", "unknown", "operator-blocking"),
            ),
        ),
    )
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ? ORDER BY ordinal", (operation_id,)
        ).fetchall() == [("applied",), ("failed",), ("unknown",)]
        assert conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "recovery_unknown")


def test_adjudication_refuses_malformed_outcomes_and_partial_target_sets(tmp_path: Path) -> None:
    """Audit authority validates adjudication input itself, not only surfaces.

    Anti-vacuity: dropping either guard lets a caller silently terminalize a
    subset of durable targets, or write a target state outside the closed
    ``applied``/``not-applied``/``unknown`` vocabulary.
    """

    actuator = _Actuator(target_refs=("session:one", "session:two"))
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    with pytest.raises(ValueError, match="name every durable target exactly once"):
        audit.adjudicate_recovery(operation_id, target_outcomes={"session:one": "applied"}, reason="partial")
    with pytest.raises(ValueError, match="applied, not-applied, or unknown"):
        audit.adjudicate_recovery(
            operation_id,
            target_outcomes=cast(Any, {"session:one": "applied", "session:two": "probably"}),
            reason="malformed",
        )
    with pytest.raises(ValueError, match="requires a reason"):
        audit.adjudicate_recovery(
            operation_id,
            target_outcomes={"session:one": "applied", "session:two": "applied"},
            reason="   ",
        )
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT DISTINCT state FROM operation_targets WHERE operation_id = ? ORDER BY state", (operation_id,)
        ).fetchall() == [("pending",), ("running",)]


def test_adjudicated_unknown_stays_terminal_and_adjudicable(tmp_path: Path) -> None:
    """An adjudicated unknown does not reopen the run for startup reclassification.

    Anti-vacuity: deriving the run state without the unknown override leaves
    the run ``interrupted``, which the next daemon startup would classify and
    append another durable recovery event for.
    """

    actuator = _Actuator(target_refs=("session:one", "session:two"))
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    audit.adjudicate_recovery(
        operation_id,
        target_outcomes={"session:one": "applied", "session:two": "unknown"},
        reason="one target could not be verified",
    )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "recovery_unknown")
        assert conn.execute(
            "SELECT DISTINCT state FROM operation_attempts WHERE operation_id = ?", (operation_id,)
        ).fetchall() == [("unknown",)]

    # The wedge stays adjudicable: a second decision can still close it.
    audit.adjudicate_recovery(
        operation_id,
        target_outcomes={"session:one": "applied", "session:two": "applied"},
        reason="operator verified the second target",
    )
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "completed",
        )


def test_recovery_disposition_rejects_internally_inconsistent_target_outcomes() -> None:
    """A disposition cannot claim a kind its own per-target outcomes contradict."""

    with pytest.raises(ValueError, match="applied outcomes only"):
        RecoveryDisposition(
            "confirmed-applied",
            "forward",
            "inconsistent",
            (RecoveryTargetDisposition("session:one", "not-applied", "forward"),),
        )
    with pytest.raises(ValueError, match="not-applied outcomes only"):
        RecoveryDisposition(
            "confirmed-not-applied",
            "retry-exact",
            "inconsistent",
            (RecoveryTargetDisposition("session:one", "applied", "retry-exact"),),
        )
    with pytest.raises(ValueError, match="unknown recovery must block an operator"):
        RecoveryDisposition("unknown", "retry-exact", "unknown cannot continue")


def test_confirmed_recovery_must_classify_every_durable_target(tmp_path: Path) -> None:
    """A confirmed recovery cannot terminalize a run it only partly inspected.

    Anti-vacuity: restricting the completeness check to ``confirmed-partial``
    lets a confirmed-applied disposition name one target and silently leave
    the rest applied by the flat fallback.
    """

    actuator = _Actuator(target_refs=("session:one", "session:two"))
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)

    with pytest.raises(ValueError, match="classify every durable target exactly once"):
        audit.record_recovery_disposition(
            operation_id,
            RecoveryDisposition(
                "confirmed-applied",
                "forward",
                "only one target inspected",
                (RecoveryTargetDisposition("session:one", "applied", "forward"),),
            ),
        )
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT DISTINCT state FROM operation_targets WHERE operation_id = ? ORDER BY state", (operation_id,)
        ).fetchall() == [("pending",), ("running",)]


def test_partial_recovery_persists_each_target_continuation(tmp_path: Path) -> None:
    """A mixed recovery cannot collapse target evidence into one flat action."""

    actuator = _Actuator(
        target_refs=("session:first", "session:second"),
        recovery_disposition=RecoveryDisposition(
            "confirmed-partial",
            "retry-exact",
            "first delete applied, second remains",
            (
                RecoveryTargetDisposition("session:first", "applied", "forward"),
                RecoveryTargetDisposition("session:second", "not-applied", "retry-exact"),
            ),
        ),
    )
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    executor = OperationExecutor(audit=audit, token_factory=lambda: "partial-retry")
    binding = _binding(actuator)
    preview = executor.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = executor.authorize_bound(binding, preview, _principal())
    receipt = executor.execute_bound(binding, preview, authorization, object())
    assert receipt.status == "applied"
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ? ORDER BY ordinal", (operation_id,)
        ).fetchall() == [("applied",), ("failed",)]
    event = audit.list_events(operation_id)[-1]
    assert event["event_type"] == "recovery_classified"
    assert '"target_ref":"session:second"' in str(event["detail_json"])


def test_unknown_owner_is_not_stolen_by_an_overlapping_apply(tmp_path: Path) -> None:
    """An unreadable ownership witness is blocking, never an orphan shortcut."""

    actuator = _Actuator()
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'external:unverifiable' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    retry = OperationExecutor(audit=audit, token_factory=lambda: "unknown-owner-token")
    binding = _binding(actuator)
    preview = retry.prepare_bound(
        binding,
        object(),
        _principal(),
        archive_instance_id="archive:recovery",
        archive_identity_digest="identity:recovery",
        parameter_digest="params:recovery",
    )
    authorization = retry.authorize_bound(binding, preview, _principal())

    with pytest.raises(RecoveryBlockedError, match="unknown owner"):
        retry.execute_bound(binding, preview, authorization, object())

    assert actuator.inspections == 0
    assert actuator.calls == 0
    assert audit.list_events(operation_id)[-1]["event_type"] == "authorization_consumed"


@pytest.mark.parametrize("owner_id", [None, "external:unverifiable"])
def test_recovery_preserves_attempts_with_unproven_owners(tmp_path: Path, owner_id: str | None) -> None:
    """Legacy or externally-owned attempts stay running until their owner is proven dead."""

    initialize_active_archive_root(tmp_path)
    audit = _audit(tmp_path)
    executor = OperationExecutor(audit=audit, token_factory=lambda: "unproven-owner-token")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:unproven-owner",
        archive_identity_digest="identity:unproven-owner",
        parameter_digest="params:unproven-owner",
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute("UPDATE operation_attempts SET worker_id = ? WHERE operation_id = ?", (owner_id, operation_id))
        conn.commit()

    assert audit.recover_abandoned_attempts() == ()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone() == (
            "running",
        )


def test_process_owner_liveness_is_unknown_when_start_ticks_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live PID with unreadable identity evidence is not proof that its owner died."""

    monkeypatch.setattr("polylogue.operations.audit.os.kill", lambda _pid, _signal: None)
    monkeypatch.setattr(Path, "read_text", lambda _self, *, encoding: (_ for _ in ()).throw(OSError("denied")))

    assert _attempt_owner_liveness("pid:321:known-start") == "unknown"


def test_process_owner_uses_proc_start_ticks_after_a_spaced_process_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """PID reuse remains detectable when /proc's parenthesized comm has spaces."""

    state = {"stat": "321 (worker process) S " + " ".join(["0"] * 17 + ["stable", "old", "0"])}

    def read_text(self: Path, *, encoding: str) -> str:
        assert self == Path("/proc/321/stat")
        assert encoding == "utf-8"
        return state["stat"]

    monkeypatch.setattr("polylogue.operations.audit.os.getpid", lambda: 321)
    monkeypatch.setattr("polylogue.operations.audit.os.kill", lambda _pid, _signal: None)
    monkeypatch.setattr(Path, "read_text", read_text)

    owner = _current_process_attempt_owner()
    assert owner == "pid:321:old"

    state["stat"] = "321 (worker process) S " + " ".join(["0"] * 17 + ["stable", "new", "0"])
    assert not _attempt_owner_is_live(owner)


def test_audit_repository_cannot_bypass_the_continuity_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)

    def reject_bypass(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("coordinator required")

    monkeypatch.setattr(AuditContinuityCoordinator, "execute", reject_bypass)
    with pytest.raises(RuntimeError, match="coordinator required"):
        audit.ensure_archive_authority(now_ms=1)


def test_audit_repository_replays_a_prepared_mutation_with_its_original_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("crash after prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
    with pytest.raises(RuntimeError, match="crash after prepare"):
        audit.ensure_archive_authority(now_ms=123, archive_instance_id="archive:replayed")
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT archive_instance_id, created_at_ms FROM archive_authority").fetchone() == (
            "archive:replayed",
            123,
        )


def test_reconcile_waits_for_an_inflight_continuity_write(tmp_path: Path) -> None:
    """Recovery cannot promote another writer's prepared command out from under it."""

    _audit(tmp_path)
    prepared = threading.Event()
    release_writer = threading.Event()
    reconcile_started = threading.Event()
    applied_by: list[str] = []

    def pause_after_prepare(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            prepared.set()
            if not release_writer.wait(timeout=10):
                raise TimeoutError("writer was not released")

    writer = AuditContinuityCoordinator(tmp_path, phase_hook=pause_after_prepare)
    recovery = AuditContinuityCoordinator(tmp_path)
    mutation = AuditMutation("test_serialized_recovery", "mutation:writer", 1, {})

    def execute_writer() -> None:
        writer.execute(mutation, lambda _connection, _mutation: applied_by.append("writer"))

    def reconcile() -> None:
        reconcile_started.set()
        recovery.reconcile(lambda _connection, _mutation: applied_by.append("recovery"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer_result = pool.submit(execute_writer)
        assert prepared.wait(timeout=10)
        reconcile_result = pool.submit(reconcile)
        assert reconcile_started.wait(timeout=10)
        with pytest.raises(FuturesTimeoutError):
            reconcile_result.result(timeout=0.1)
        release_writer.set()
        writer_result.result(timeout=10)
        reconcile_result.result(timeout=10)

    assert applied_by == ["writer"]
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256, pending_mutation_id "
            "FROM audit_continuity_control WHERE singleton = 1"
        ).fetchone()
        audit_head = audit.execute(
            "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
        ).fetchone()
    assert source_head == (audit_head[0], audit_head[1], None)
    assert audit_head[2] == mutation.mutation_id


def test_mark_preview_stale_advances_and_replays_durable_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "stale-continuity-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:stale-continuity",
        archive_identity_digest="identity:stale-continuity",
        parameter_digest="params:stale-continuity",
    )
    executor.authorize_bound(_binding(actuator), preview, _principal())
    with sqlite3.connect(tmp_path / "source.db") as source:
        before_generation = int(
            source.execute("SELECT committed_generation FROM audit_continuity_control").fetchone()[0]
        )

    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "mark_preview_stale" and phase == "after_source_prepare":
            raise RuntimeError("crash after stale-preview prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
    with pytest.raises(RuntimeError, match="stale-preview prepare"):
        audit.mark_preview_stale(preview)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()

    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "audit.db") as audit_db:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256, pending_mutation_id FROM audit_continuity_control"
        ).fetchone()
        audit_head = audit_db.execute("SELECT generation, head_sha256 FROM audit_continuity_head").fetchone()
        assert source_head == (before_generation + 1, audit_head[1], None)
        assert audit_head[0] == before_generation + 1
        assert audit_db.execute(
            "SELECT state FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
        ).fetchone() == ("stale",)
        assert audit_db.execute(
            "SELECT state FROM operation_authorizations WHERE preview_id = ?", (preview.preview_ref,)
        ).fetchone() == ("revoked",)


def test_replayed_start_keeps_the_crashed_owner_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery never adopts an actuator-less pre-effect attempt into its own process."""

    initialize_active_archive_root(tmp_path)
    crashed_owner = "pid:999999999:0"
    first = AuditRepository.for_archive_root(tmp_path, attempt_owner_id=crashed_owner)
    executor = OperationExecutor(audit=first, token_factory=lambda: "replayed-owner-token")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:replayed-owner",
        archive_identity_digest="identity:replayed-owner",
        parameter_digest="params:replayed-owner",
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_start(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "consume_authorization_and_start" and phase == "after_source_prepare":
            raise RuntimeError("crash after start prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_start)
    with pytest.raises(RuntimeError, match="crash after start prepare"):
        first.consume_authorization_and_start(preview, authorization)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    recovery = AuditRepository.for_archive_root(tmp_path, attempt_owner_id="pid:12345:recovery")
    recovery.reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        operation_id = str(conn.execute("SELECT operation_id FROM operation_runs").fetchone()[0])
        assert conn.execute(
            "SELECT worker_id FROM operation_attempts WHERE operation_id = ?", (operation_id,)
        ).fetchone() == (crashed_owner,)

    assert recovery.recover_abandoned_attempts() == (operation_id,)
    assert recovery.get_operation(operation_id)["status"] == "interrupted"  # type: ignore[index]


def test_optional_archive_authority_id_replays_without_changing_existing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An omitted authority id remains omitted across a pre-commit crash."""

    audit = _audit(tmp_path)
    assert audit.ensure_archive_authority(now_ms=1, archive_instance_id="archive:existing") == "archive:existing"
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_after_prepare(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "ensure_archive_authority" and phase == "after_source_prepare":
            raise RuntimeError("crash after optional authority prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_after_prepare)
    with pytest.raises(RuntimeError, match="optional authority prepare"):
        audit.ensure_archive_authority(now_ms=2)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT archive_instance_id, created_at_ms FROM archive_authority").fetchone() == (
            "archive:existing",
            1,
        )


def test_typed_domain_receipt_replays_after_source_prepare_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real executor route persists and replays typed receipt values as JSON."""

    audit = _audit(tmp_path)
    actuator = _TypedReceiptActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "typed-receipt-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:typed-receipt",
        archive_identity_digest="identity:typed-receipt",
        parameter_digest="params:typed-receipt",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_finalize(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "finalize_attempt" and phase == "after_source_prepare":
            raise RuntimeError("crash after typed receipt prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_finalize)
    with pytest.raises(AuditFinalizationError, match="not reported completed"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    with sqlite3.connect(tmp_path / "source.db") as source:
        pending_payload = str(source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0])
    assert "private-cache" not in pending_payload
    assert "annotation-batch:typed" not in pending_payload
    assert '"domain_receipt"' not in pending_payload
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    AuditRepository.for_archive_root(tmp_path).reconcile_continuity()
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT status FROM operation_runs").fetchone() == ("completed",)
        receipt_json = str(
            conn.execute("SELECT detail_json FROM operation_events WHERE event_type = 'attempt_finalized'").fetchone()[
                0
            ]
        )
    assert "private-cache" not in receipt_json
    detail = json.loads(receipt_json)
    assert detail["affected_count"] == 1
    assert "domain_receipt" not in detail
    assert "annotation-batch:typed" not in receipt_json
    with sqlite3.connect(tmp_path / "source.db") as source:
        command = source.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0]
    assert command is None


def test_recovery_disposition_replays_after_source_prepare_crash_at_daemon_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daemon startup completes a prepared recovery classification command.

    Anti-vacuity: omitting target dispositions from the write-ahead command
    makes this restart raise before it can clear the pending source command.
    """

    actuator = _Actuator()
    audit, operation_id = _dead_nonterminal_operation(tmp_path, actuator)
    original_phase = AuditContinuityCoordinator._phase

    def interrupt_disposition(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "record_recovery_disposition" and phase == "after_source_prepare":
            raise RuntimeError("crash after recovery disposition prepare")
        original_phase(self, phase, mutation)

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_disposition)
    disposition = RecoveryDisposition(
        "confirmed-not-applied",
        "retry-exact",
        "inspector proved the target was retained",
        (RecoveryTargetDisposition("session:fixture", "not-applied", "retry-exact"),),
    )
    with pytest.raises(RuntimeError, match="recovery disposition prepare"):
        audit.record_recovery_disposition(operation_id, disposition)
    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", original_phase)

    recover_interrupted_operations(tmp_path)

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "recovered_not_applied")
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed",)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)


def test_atomic_batch_finalization_marks_every_target_and_terminates_run(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator(target_refs=("session:first", "session:second"))
    executor = OperationExecutor(audit=audit, token_factory=lambda: "batch-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:batch",
        archive_identity_digest="identity:batch",
        parameter_digest="params:batch",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())

    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, affected_count, unknown_count FROM operation_runs WHERE operation_id = ?",
            (receipt.operation_id,),
        ).fetchone() == ("completed", 2, 0)
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
            (receipt.operation_id,),
        ).fetchall() == [("applied",), ("applied",)]


def test_zero_target_finalization_completes_a_successful_noop(tmp_path: Path) -> None:
    """The real start/finalize route terminalizes a successful empty target set."""

    audit = _audit(tmp_path)
    actuator = _Actuator(target_refs=())
    executor = OperationExecutor(audit=audit, token_factory=lambda: "zero-target-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:zero-target",
        archive_identity_digest="identity:zero-target",
        parameter_digest="params:zero-target",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())

    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason, affected_count FROM operation_runs WHERE operation_id = ?",
            (receipt.operation_id,),
        ).fetchone() == ("completed", None, 0)


def test_expired_authorization_is_durably_marked_before_execute_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound execution route commits expiry instead of rolling it back with the refusal."""

    audit = _audit(tmp_path)
    clock = [1_000]
    executor = OperationExecutor(audit=audit, now_ms=lambda: clock[0], token_factory=lambda: "expired-token")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:expired",
        archive_identity_digest="identity:expired",
        parameter_digest="params:expired",
        expires_at_ms=61_000,
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())
    clock[0] = 61_000
    monkeypatch.setattr("polylogue.operations.audit.time.time", lambda: 61.0)

    with pytest.raises(TokenExpiredError):
        executor.execute_bound(_binding(_Actuator()), preview, authorization, object())

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT state FROM operation_authorizations").fetchone() == ("expired",)


def test_authorization_expiry_is_canonicalized_from_the_durable_preview(tmp_path: Path) -> None:
    """A caller cannot issue a longer-lived bearer than the preview authorizes.

    Anti-vacuity: storing ``authorization.expires_at_ms`` accepts the forged
    expiry and leaves an authorization row that outlives its durable preview.
    """

    audit = _audit(tmp_path)
    clock = [1_000]
    executor = OperationExecutor(audit=audit, now_ms=lambda: clock[0], token_factory=lambda: "canonical-expiry")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:canonical-expiry",
        archive_identity_digest="identity:canonical-expiry",
        parameter_digest="params:canonical-expiry",
        expires_at_ms=2_000,
    )
    authorization = executor.authorize_bound(_binding(_Actuator()), preview, _principal())

    with pytest.raises(ValueError, match="evidence differs"):
        audit.issue_authorization(
            preview,
            _principal(),
            replace(authorization, token="forged-expiry", expires_at_ms=3_000),
            issued_at_ms=1_100,
        )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT expires_at_ms FROM operation_previews").fetchone() == (2_000,)
        assert conn.execute("SELECT expires_at_ms FROM operation_authorizations").fetchone() == (2_000,)


def test_authorization_consumption_uses_durable_actor_and_capability_evidence(tmp_path: Path) -> None:
    """Execution refuses reconstructed authority that differs from durable rows.

    Anti-vacuity: persisting run fields from the caller object records this
    substituted actor/capability instead of the issued authorization evidence.
    """

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "durable-evidence")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:durable-evidence",
        archive_identity_digest="identity:durable-evidence",
        parameter_digest="params:durable-evidence",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    forged = replace(
        authorization,
        actor="actor:substituted",
        role="administrator",
        capability="archive.substituted.write",
        capabilities=("archive.substituted.write",),
    )

    with pytest.raises(AuthorizationMismatchError, match="principal mismatch"):
        audit.consume_authorization_and_start(preview, forged)

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone() == (0,)
        assert conn.execute("SELECT state FROM operation_authorizations").fetchone() == ("active",)
        assert conn.execute("SELECT actor_ref FROM operation_authorizations").fetchone() == ("actor:test",)
        assert conn.execute("SELECT capability FROM operation_authorization_capabilities").fetchone() == (
            "archive.fixture.write",
        )


def test_authorization_replay_preserves_the_prepared_expiry_and_issue_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried WAL authorization reuses its prepared durable evidence verbatim.

    Anti-vacuity: omitting ``issued_at_ms`` during replay silently substitutes
    the recovery clock and can make an authorization valid longer than the
    original prepared command proved.
    """

    audit = _audit(tmp_path)
    clock = [1_000]
    executor = OperationExecutor(audit=audit, now_ms=lambda: clock[0], token_factory=lambda: "replayed-expiry")
    preview = executor.prepare_bound(
        _binding(_Actuator()),
        object(),
        _principal(),
        archive_instance_id="archive:replayed-expiry",
        archive_identity_digest="identity:replayed-expiry",
        parameter_digest="params:replayed-expiry",
        expires_at_ms=2_000,
    )
    original_abort = AuditContinuityCoordinator._abort_prepared

    def interrupt_issue(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "issue_authorization" and phase == "after_source_prepare":
            raise RuntimeError("crash after authorization prepare")

    monkeypatch.setattr(AuditContinuityCoordinator, "_phase", interrupt_issue)
    monkeypatch.setattr(AuditContinuityCoordinator, "_abort_prepared", lambda _self, _prepared: None)
    with pytest.raises(RuntimeError, match="authorization prepare"):
        executor.authorize_bound(_binding(_Actuator()), preview, _principal())

    monkeypatch.setattr(AuditContinuityCoordinator, "_abort_prepared", original_abort)
    audit.reconcile_continuity()

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT issued_at_ms, expires_at_ms FROM operation_authorizations").fetchone() == (
            1_000,
            2_000,
        )


def test_already_satisfied_receipt_preserves_target_state_and_zero_affected_count(tmp_path: Path) -> None:
    """A nonempty idempotent success remains distinct from an applied domain effect."""

    audit = _audit(tmp_path)
    actuator = _AlreadySatisfiedActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "already-satisfied-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:already-satisfied",
        archive_identity_digest="identity:already-satisfied",
        parameter_digest="params:already-satisfied",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == ("already_satisfied",)
        assert conn.execute(
            "SELECT status, affected_count FROM operation_runs WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == (
            "completed",
            0,
        )


def test_failed_receipt_marks_the_audit_attempt_failed(tmp_path: Path) -> None:
    """A domain-declared failure cannot leave an applied attempt receipt.

    Anti-vacuity: restoring the rejected-only attempt mapping makes the target
    failed while this attempt row incorrectly returns applied.
    """

    audit = _audit(tmp_path)
    actuator = _FailedReceiptActuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "failed-receipt-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:failed-receipt",
        archive_identity_digest="identity:failed-receipt",
        parameter_digest="params:failed-receipt",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    receipt = executor.execute_bound(_binding(actuator), preview, authorization, object())

    assert receipt.operation_id is not None
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_attempts WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == ("failed",)
        assert conn.execute(
            "SELECT status FROM operation_runs WHERE operation_id = ?", (receipt.operation_id,)
        ).fetchone() == ("failed",)


def test_tampered_preview_payload_refuses_before_audit_intent(tmp_path: Path) -> None:
    """Execution cannot journal targets substituted into a reconstructed preview.

    Anti-vacuity: deleting the typed hash validation consumes the token and
    creates an audit run whose target set comes from the altered preview.
    """

    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "tampered-preview-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:tampered-preview",
        archive_identity_digest="identity:tampered-preview",
        parameter_digest="params:tampered-preview",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    tampered_preview = replace(preview, plan=replace(preview.plan, targets=()))

    with pytest.raises(AuthorizationMismatchError, match="authority hash"):
        executor.execute_bound(_binding(actuator), tampered_preview, authorization, object())

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone() == (0,)
        assert conn.execute("SELECT state FROM operation_authorizations").fetchone() == ("active",)


def test_blocked_finalization_rejects_targets_and_fails_parent_run(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "blocked-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:blocked",
        archive_identity_digest="identity:blocked",
        parameter_digest="params:blocked",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)

    audit.finalize_attempt(operation_id, status="blocked")

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, terminal_reason, rejected_count FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "target_rejected", 1)
        assert conn.execute(
            "SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("rejected",)


def test_authorized_adjudication_resolves_the_full_unknown_atomic_batch(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator(target_refs=("session:first", "session:second"))
    executor = OperationExecutor(audit=audit, token_factory=lambda: "reconcile-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:reconcile",
        archive_identity_digest="identity:reconcile",
        parameter_digest="params:reconcile",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    operation_id = audit.consume_authorization_and_start(preview, authorization)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    audit.recover_abandoned_attempts()

    audit.adjudicate_recovery(
        operation_id,
        target_outcomes={"session:first": "applied", "session:second": "applied"},
        reason="operator inspected durable target evidence",
    )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT status, affected_count, unknown_count FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("completed", 2, 0)
        assert conn.execute(
            "SELECT state, domain_receipt_ref FROM operation_targets WHERE operation_id = ? ORDER BY ordinal",
            (operation_id,),
        ).fetchall() == [("applied", None), ("applied", None)]


def test_invalid_capability_and_stale_preview_refuse_before_apply(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    with pytest.raises(CapabilityDeniedError):
        executor.authorize_bound(
            _binding(actuator),
            preview,
            MutationPrincipal("actor:bad", frozenset(), "internal"),
        )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    actuator.changed = True
    with pytest.raises(PlanStaleError):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    assert actuator.calls == 0
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute(
            "SELECT state FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
        ).fetchone() == ("stale",)
        assert conn.execute(
            "SELECT state FROM operation_authorizations WHERE authorization_id = ?",
            (authorization.authorization_id,),
        ).fetchone() == ("revoked",)
    actuator.changed = False
    with pytest.raises(TokenConsumedError):
        executor.execute_bound(_binding(actuator), preview, authorization, object())


def test_new_authorization_revokes_every_older_token_for_preview(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    tokens = iter(("first-token", "second-token"))
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: next(tokens))
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )

    first = executor.authorize_bound(_binding(actuator), preview, _principal())
    second = executor.authorize_bound(_binding(actuator), preview, _principal())

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        states = conn.execute(
            "SELECT authorization_id, state FROM operation_authorizations WHERE preview_id = ? ORDER BY issued_at_ms, authorization_id",
            (preview.preview_ref,),
        ).fetchall()
    assert {state for _authorization_id, state in states} == {"active", "revoked"}
    assert sum(state == "active" for _authorization_id, state in states) == 1
    with pytest.raises(TokenConsumedError):
        executor.execute_bound(_binding(actuator), preview, first, object())
    receipt = executor.execute_bound(_binding(actuator), preview, second, object())
    assert receipt.status == "applied"


def test_crash_after_intent_is_queryable_unknown_and_never_completed(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator(crash=True)
    executor = OperationExecutor(audit=audit, token_factory=lambda: "crash-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())
    with pytest.raises(RuntimeError, match="simulated"):
        executor.execute_bound(_binding(actuator), preview, authorization, object())
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        operation_id = str(conn.execute("SELECT operation_id FROM operation_runs").fetchone()[0])
        status = str(
            conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()[0]
        )
        target_state = str(
            conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,)).fetchone()[0]
        )
    finally:
        conn.close()
    assert status == "interrupted"
    assert target_state == "unknown"
    assert audit.list_events(operation_id)[-1]["event_type"] == "attempt_unknown"


def test_token_consumption_and_initial_attempt_roll_back_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit(tmp_path)
    actuator = _Actuator()
    executor = OperationExecutor(audit=audit, token_factory=lambda: "rollback-token")
    preview = executor.prepare_bound(
        _binding(actuator),
        object(),
        _principal(),
        archive_instance_id="archive:test",
        archive_identity_digest="identity:test",
        parameter_digest="params:test",
    )
    authorization = executor.authorize_bound(_binding(actuator), preview, _principal())

    def fail_event(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected audit transaction failure")

    monkeypatch.setattr(AuditRepository, "_append_event", staticmethod(fail_event))
    with pytest.raises(RuntimeError, match="injected"):
        audit.consume_authorization_and_start(preview, authorization)

    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        auth_state = str(
            conn.execute(
                "SELECT state FROM operation_authorizations WHERE preview_id = ?", (preview.preview_ref,)
            ).fetchone()[0]
        )
        run_count = int(conn.execute("SELECT COUNT(*) FROM operation_runs").fetchone()[0])
        preview_state = str(
            conn.execute(
                "SELECT state FROM operation_previews WHERE preview_id = ?", (preview.preview_ref,)
            ).fetchone()[0]
        )
    finally:
        conn.close()
    assert auth_state == "active"
    assert preview_state == "prepared"
    assert run_count == 0
