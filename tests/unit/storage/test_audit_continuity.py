"""Crash-window and rollback proofs for the source-backed audit head."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import pytest

from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceInput, FrozenSourceManifest
from polylogue.storage.sqlite.audit_continuity import (
    AuditContinuityCoordinator,
    AuditContinuityError,
    AuditMutation,
    CanonicalAuditLiteral,
)

if TYPE_CHECKING:
    from polylogue.operations.audit import AuditRepository
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

_SourceCompletionControl: TypeAlias = tuple[Path, "AuditRepository", dict[str, Any], Callable[..., Any]]


def _mutation(number: int) -> AuditMutation:
    return AuditMutation(
        kind="test-audit-write",
        mutation_id=f"mutation:{number}",
        created_at_ms=number,
        payload={"number": number},
    )


def _apply(conn: sqlite3.Connection, mutation: AuditMutation) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, 1)",
        (f"archive:{mutation.mutation_id}", mutation.created_at_ms),
    )
    return mutation.mutation_id


def test_ingest_abort_rolls_back_the_manifest_and_clears_the_wal(tmp_path: Path) -> None:
    """A failed audit apply rolls the whole accept_ingest prepare back.

    polylogue-2kbrl: this previously asserted the opposite -- that the prepared
    manifest and its WAL entry were both *retained*, because the prepare phase
    published durable source rows that an abort could not undo. That retention
    is what wedged the audit tier: a non-transient failure left the pending
    entry forever, refusing every later audit mutation and every operation
    read. The manifest is now published during promotion instead, so the abort
    is a true rollback and nothing durable is stranded.
    """
    from polylogue.storage.sqlite.write_lease import write_lease

    initialize_active_archive_root(tmp_path)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    with write_lease("test.continuity-source-publication", archive_root=tmp_path):
        blob_hash, _ = publisher.write_from_bytes(b"synthetic retained input")
        publisher.flush()
    receipt_id = publisher.receipt_id(blob_hash)
    assert receipt_id is not None
    manifest = FrozenSourceManifest(
        "ingest-generation",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", blob_hash, receipt_id),),
    )
    mutation = AuditMutation("accept_ingest", "ingest-request", 1, {"manifest": manifest.to_dict()})

    def fail_apply(_conn: sqlite3.Connection, _mutation: AuditMutation) -> str:
        raise sqlite3.OperationalError("injected audit failure")

    coordinator = AuditContinuityCoordinator(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match="injected audit failure"):
        coordinator.execute(mutation, fail_apply)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 0
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        # The reservation was never spent, so the same manifest stays acceptable.
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1
    coordinator.execute(mutation, _apply)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 1
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


def test_ingest_prepare_missing_receipt_rolls_back_manifest_and_wal(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    manifest = FrozenSourceManifest(
        "ingest-generation",
        "d" * 64,
        (FrozenSourceInput("input.json", "/synthetic/input.json", "a" * 64, "missing"),),
    )
    with pytest.raises(ValueError, match="reservation is missing"):
        AuditContinuityCoordinator(tmp_path).execute(
            AuditMutation("accept_ingest", "ingest-request", 1, {"manifest": manifest.to_dict()}),
            _apply,
        )
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 0


def test_same_inode_stale_audit_copy_is_rejected(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    coordinator = AuditContinuityCoordinator(tmp_path)
    coordinator.execute(_mutation(1), _apply)
    stale_bytes = (tmp_path / "audit.db").read_bytes()
    coordinator.execute(_mutation(2), _apply)

    # Deliberately overwrite rather than replace the path. The inode remains
    # stable, so this proves the source/audit head catches the rollback that
    # the former st_dev/st_ino receipt accepted.
    audit_path = tmp_path / "audit.db"
    inode = audit_path.stat().st_ino
    audit_path.write_bytes(stale_bytes)
    assert audit_path.stat().st_ino == inode

    with pytest.raises(AuditContinuityError, match="regressed|replaced"):
        coordinator.reconcile(_apply)


def test_crash_before_source_prepare_leaves_no_command(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "before_source_prepare":
            raise RuntimeError("crash before source prepare")

    with pytest.raises(RuntimeError, match="crash before source"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)


def test_pending_command_replays_after_audit_rollback(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("crash before audit commit")

    with pytest.raises(RuntimeError, match="crash before"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM archive_authority").fetchone()[0] == 1


def test_pending_command_promotes_after_audit_commit(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_audit_commit":
            raise RuntimeError("crash before source promotion")

    with pytest.raises(RuntimeError, match="crash before"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)

    AuditContinuityCoordinator(tmp_path).reconcile(_apply)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is None


@pytest.mark.parametrize(
    ("dropped_table", "error"),
    [
        ("audit_continuity_control", "current source schema"),
        ("audit_continuity_head", "current audit schema"),
    ],
)
def test_current_schema_missing_a_continuity_table_is_damage(tmp_path: Path, dropped_table: str, error: str) -> None:
    """A present tier without its continuity half is damage, never standby.

    Fresh v1 stamps every durable tier at 1, so the old version-gated check
    (source >= 32, audit >= 2) read this exact archive as a legitimate
    pre-continuity window and returned False (polylogue-h6yuj).

    Anti-vacuity: restore the ``user_version`` gate and this stops raising.
    """
    initialize_active_archive_root(tmp_path)
    path = tmp_path / ("source.db" if dropped_table.endswith("control") else "audit.db")
    with sqlite3.connect(path) as connection:
        connection.execute(f"DROP TABLE {dropped_table}")
        connection.commit()

    with pytest.raises(AuditContinuityError, match=error):
        AuditContinuityCoordinator(tmp_path).is_available()


@pytest.mark.parametrize(
    ("path_name", "entry_kind"),
    [
        ("source.db", "external_symlink"),
        ("audit.db", "directory"),
        ("audit.db", "dangling_symlink"),
    ],
)
def test_invalid_present_continuity_tier_never_enters_standby(tmp_path: Path, path_name: str, entry_kind: str) -> None:
    """Only literal absence can disable continuity; invalid entries fail closed."""

    initialize_active_archive_root(tmp_path)
    path = tmp_path / path_name
    if entry_kind == "external_symlink":
        external = tmp_path.parent / "external-source.db"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
    elif entry_kind == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.unlink()
        path.symlink_to(tmp_path / "missing-audit.db")

    with pytest.raises(AuditContinuityError, match="regular file|safely"):
        AuditContinuityCoordinator(tmp_path).is_available()


def test_empty_fresh_archive_can_use_the_genesis_continuity_head(tmp_path: Path) -> None:
    """Genesis is valid only when it describes an empty freshly-created audit journal."""

    initialize_active_archive_root(tmp_path)

    assert AuditContinuityCoordinator(tmp_path).is_available()


def test_second_mutation_refuses_while_first_command_is_pending(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    def interrupt(phase: str, _mutation: AuditMutation) -> None:
        if phase == "after_source_prepare":
            raise RuntimeError("leave pending")

    with pytest.raises(RuntimeError, match="leave pending"):
        AuditContinuityCoordinator(tmp_path, phase_hook=interrupt).execute(_mutation(1), _apply)
    with pytest.raises(AuditContinuityError, match="already pending"):
        AuditContinuityCoordinator(tmp_path).execute(_mutation(2), _apply)


def test_rejected_audit_transaction_aborts_its_prepared_command(tmp_path: Path) -> None:
    """A deterministic reject cannot leave the source WAL blocking later work."""

    initialize_active_archive_root(tmp_path)

    def reject(_conn: sqlite3.Connection, _mutation: AuditMutation) -> object:
        raise ValueError("already consumed")

    coordinator = AuditContinuityCoordinator(tmp_path)
    with pytest.raises(ValueError, match="already consumed"):
        coordinator.execute(_mutation(1), reject)

    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)
    assert coordinator.execute(_mutation(2), _apply) == "mutation:2"
    with closing(sqlite3.connect(tmp_path / "audit.db")) as audit:
        assert audit.execute("SELECT generation, mutation_id FROM audit_continuity_head").fetchone() == (
            1,
            "mutation:2",
        )


def test_continuity_mutation_is_refused_without_the_write_lease(tmp_path: Path) -> None:
    """A durable source.db continuity write outside the lease is refused.

    polylogue-xh6cz: ``_open_source_write_connection`` reached
    ``open_verified_sqlite_write_connection`` with neither the lease nor the
    flock, so ``BEGIN IMMEDIATE`` + ``UPDATE audit_continuity_control`` against
    the durable raw-bytes tier was an unserialized in-process writer.

    Anti-vacuity: delete the ``require_write_lease`` call from
    ``open_verified_sqlite_write_connection`` and this passes -- the
    unserialized writer commits.
    """
    from polylogue.storage.sqlite.write_lease import UnleasedWriteError, arm_write_lease_enforcement

    initialize_active_archive_root(tmp_path)
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError, match="write lease"):
            AuditContinuityCoordinator(tmp_path).execute(_mutation(1), _apply)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute("SELECT committed_generation FROM audit_continuity_control").fetchone() == (0,)


def test_audit_leaf_write_is_refused_without_the_write_lease(tmp_path: Path) -> None:
    """The audit.db leaf factory is on the same gate as the source.db one.

    Anti-vacuity: drop ``require_write_lease`` from
    ``open_verified_audit_connection`` and the unleased writer opens.
    """
    from polylogue.storage.sqlite.audit_leaf import open_verified_audit_connection
    from polylogue.storage.sqlite.write_lease import UnleasedWriteError, arm_write_lease_enforcement

    initialize_active_archive_root(tmp_path)
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError, match="write lease"):
            with open_verified_audit_connection(tmp_path / "audit.db"):
                pass


def test_continuity_mutation_commits_under_a_held_lease(tmp_path: Path) -> None:
    """The gate serializes the route; it does not close it.

    The whole prepare/promote sequence nests inside one held lease, and
    ``require_write_lease`` only asserts -- it never re-acquires -- so the
    repeated write-connection opens inside a single mutation cannot deadlock.

    Anti-vacuity: make ``require_write_lease`` acquire rather than assert and
    this hangs or raises instead of committing generation 1.
    """
    from polylogue.storage.sqlite.write_lease import arm_write_lease_enforcement, write_lease

    initialize_active_archive_root(tmp_path)
    with arm_write_lease_enforcement(), write_lease("test.audit.continuity", archive_root=tmp_path):
        AuditContinuityCoordinator(tmp_path).execute(_mutation(1), _apply)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT committed_generation, pending_mutation_id FROM audit_continuity_control"
        ).fetchone() == (1, None)


@pytest.fixture
def source_completion_control(tmp_path: Path) -> _SourceCompletionControl:
    from tests.infra.audit_completion import make_source_completion_control

    control: _SourceCompletionControl = make_source_completion_control(tmp_path)
    return control


@pytest.mark.parametrize("audit_committed", [False, True])
def test_native_completion_recovery_preserves_exact_receipts_once(
    source_completion_control: _SourceCompletionControl, audit_committed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed Source command survives both remaining continuity windows."""
    from polylogue.storage.sqlite.write_lease import write_lease

    root, repository, payload, install = source_completion_control
    prepared = install(payload)
    original_loads = json.loads

    def scalar_only(value: str | bytes, *args: Any, **kwargs: Any) -> Any:
        assert value[:1] not in (b"{", "{"), "whole completion hydrated"
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", scalar_only)
    with write_lease("test.continuity-completion-recover", archive_root=root):
        if audit_committed:
            with repository._continuity._pending() as pending:
                assert pending is not None
                repository._continuity._apply_prepared(pending, repository._replay_pending_mutation)
        repository.reconcile_continuity()
        repository.reconcile_continuity()
    with closing(sqlite3.connect(root / "audit.db")) as audit:
        events = audit.execute(
            "SELECT detail_json FROM operation_events WHERE event_type='excision_source_committed'"
        ).fetchall()
        assert len(events) == 1
        assert events[0][0] == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with closing(sqlite3.connect(root / "source.db")) as source:
        assert source.execute(
            "SELECT committed_generation,committed_head_sha256,pending_mutation_id FROM audit_continuity_control"
        ).fetchone() == (prepared.next_generation, prepared.next_head_sha256, None)


@pytest.mark.parametrize("wrong", ["operation", "attempt", "plan", "missing", "extra", "duplicate", "overlap", "stale"])
def test_native_completion_rejects_wrong_closure_and_keeps_source_command(
    source_completion_control: _SourceCompletionControl, wrong: str
) -> None:
    from polylogue.storage.sqlite.write_lease import write_lease

    root, repository, payload, install = source_completion_control
    if wrong == "operation":
        payload["operation_id"] = "operation:" + "x" * 24
    elif wrong == "attempt":
        payload["attempt_id"] = "attempt:" + "x" * 24
    elif wrong == "plan":
        payload["plan_hash"] = "0" * 64
    elif wrong == "missing":
        payload["targets"] = []
    elif wrong == "extra":
        payload["targets"] = [*payload["targets"], {**payload["targets"][0], "session_id": "codex:another"}]
    elif wrong == "duplicate":
        payload["targets"] = [*payload["targets"], payload["targets"][0]]
    elif wrong == "overlap":
        payload["targets"][0]["shared_blob_hashes"] = ["a" * 64]
    install(payload)
    if wrong == "stale":
        with closing(sqlite3.connect(root / "audit.db")) as audit:
            audit.execute("UPDATE operation_attempts SET state='unknown' WHERE attempt_id=?", (payload["attempt_id"],))
            audit.commit()
    with write_lease("test.continuity-completion-refuse", archive_root=root):
        with pytest.raises(AuditContinuityError):
            repository.reconcile_continuity()
    with closing(sqlite3.connect(root / "source.db")) as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone()[0] is not None
    with closing(sqlite3.connect(root / "audit.db")) as audit:
        assert audit.execute(
            "SELECT count(*) FROM operation_events WHERE event_type='excision_source_committed'"
        ).fetchone() == (0,)


def test_native_completion_rollback_leaves_no_receipt_or_command(
    source_completion_control: _SourceCompletionControl,
) -> None:
    from polylogue.storage.sqlite.write_lease import write_lease

    root, repository, payload, install = source_completion_control
    install(payload, rollback=True)
    with write_lease("test.continuity-completion-rollback", archive_root=root):
        repository.reconcile_continuity()
    with closing(sqlite3.connect(root / "source.db")) as source:
        assert source.execute("SELECT pending_mutation_id FROM audit_continuity_control").fetchone() == (None,)
    with closing(sqlite3.connect(root / "audit.db")) as audit:
        assert audit.execute(
            "SELECT count(*) FROM operation_events WHERE event_type='excision_source_committed'"
        ).fetchone() == (0,)


@pytest.mark.parametrize("corruption", ["space", "escape", "hash", "count", "head"])
def test_native_pending_refuses_rehashed_noncanonical_or_malformed_command(
    source_completion_control: _SourceCompletionControl, corruption: str
) -> None:
    from polylogue.storage.sqlite.audit_leaf import open_verified_sqlite_write_connection
    from polylogue.storage.sqlite.write_lease import write_lease

    root, repository, payload, install = source_completion_control
    prepared = install(payload)
    raw = b"".join(prepared.chunks())
    if corruption == "space":
        raw = raw.replace(b'"targets":[', b'"targets": [')
    elif corruption == "escape":
        session_id = payload["targets"][0]["session_id"]
        assert session_id.startswith("c")
        raw = raw.replace(json.dumps(session_id).encode(), b'"\\u0063' + session_id[1:].encode() + b'"')
    elif corruption == "hash":
        raw = raw.replace(b'"' + b"a" * 64 + b'"', b'"' + b"a" * 65 + b'"')
    elif corruption == "count":
        raw = raw.replace(b'"source_blob_refs":0', b'"source_blob_refs":true')
    else:
        raw = raw.replace(prepared.next_head_sha256.encode(), b"f" * 64)
    assert raw != b"".join(prepared.chunks())
    with write_lease("test.continuity-completion-corrupt", archive_root=root):
        with open_verified_sqlite_write_connection(root / "source.db") as source:
            source.execute(
                "UPDATE audit_continuity_control SET pending_payload_json=?,pending_payload_sha256=?",
                (raw.decode(), hashlib.sha256(raw).hexdigest()),
            )
            source.commit()
        with pytest.raises(AuditContinuityError):
            repository.reconcile_continuity()


@pytest.mark.parametrize("native_completion", [False, True])
@pytest.mark.parametrize("replacement_point", ["after_fetch", "before_audit_commit"])
def test_pending_replacement_cannot_publish_audit_from_retired_source(
    source_completion_control: _SourceCompletionControl,
    native_completion: bool,
    replacement_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A byte-identical new Source inode cannot authorize an Audit commit."""
    import shutil
    from contextlib import contextmanager

    from polylogue.storage.sqlite.write_lease import write_lease

    root, repository, payload, install = source_completion_control
    coordinator = repository._continuity
    if native_completion:
        install(payload)
    else:
        with write_lease("test.continuity-source-replacement-prepare", archive_root=root):
            coordinator._prepare(_mutation(42))
    with closing(sqlite3.connect(root / "audit.db")) as audit:
        prior_head = audit.execute("SELECT generation,head_sha256,mutation_id FROM audit_continuity_head").fetchone()
        prior_events = audit.execute("SELECT count(*) FROM operation_events").fetchone()
        prior_authorities = audit.execute("SELECT count(*) FROM archive_authority").fetchone()
    replacements: list[int] = []

    def replace_source() -> None:
        assert not replacements
        source_path = root / "source.db"
        original_inode = source_path.stat().st_ino
        original_bytes = source_path.read_bytes()
        copy_path = root / "replacement-source.db"
        shutil.copyfile(source_path, copy_path)
        assert copy_path.read_bytes() == original_bytes
        copy_path.replace(source_path)
        assert source_path.stat().st_ino != original_inode
        replacements.append(original_inode)

    from polylogue.storage.io_phase_metrics import connection_cursor as original_cursor

    @contextmanager
    def replace_after_fetch(*args: Any, **kwargs: Any) -> Iterator[object]:
        with original_cursor(*args, **kwargs) as cursor:
            if replacement_point == "after_fetch" and args[1].startswith("SELECT rowid,committed_generation,"):

                class FetchedPending:
                    def fetchone(self) -> object:
                        row = cursor.fetchone()
                        assert row is not None and row[3] is not None
                        replace_source()
                        return row

                yield FetchedPending()
            else:
                yield cursor

    monkeypatch.setattr("polylogue.storage.sqlite.audit_continuity.connection_cursor", replace_after_fetch)

    def apply(connection: sqlite3.Connection, mutation: AuditMutation) -> object:
        result = (
            repository._replay_pending_mutation(connection, mutation)
            if native_completion
            else _apply(connection, mutation)
        )
        if replacement_point == "before_audit_commit":
            replace_source()
        return result

    with write_lease("test.continuity-source-replacement-reconcile", archive_root=root):
        with pytest.raises(AuditContinuityError):
            coordinator.reconcile(apply)
    assert len(replacements) == 1
    with closing(sqlite3.connect(root / "audit.db")) as audit:
        assert (
            audit.execute("SELECT generation,head_sha256,mutation_id FROM audit_continuity_head").fetchone()
            == prior_head
        )
        assert audit.execute("SELECT count(*) FROM operation_events").fetchone() == prior_events
        assert audit.execute("SELECT count(*) FROM archive_authority").fetchone() == prior_authorities


def test_native_pending_literal_cannot_outlive_its_original_snapshot(
    source_completion_control: _SourceCompletionControl,
) -> None:
    root, repository, payload, install = source_completion_control
    install(payload)
    with repository._continuity._pending() as pending:
        assert pending is not None
        literal = pending.mutation.payload
        assert isinstance(literal, CanonicalAuditLiteral)
        assert sum(len(chunk) for chunk in literal.verified_chunks()) == literal.byte_length
    with pytest.raises(AuditContinuityError, match="snapshot"):
        list(literal.verified_chunks())


def test_native_pending_blob_close_failure_retains_the_actual_creator(
    source_completion_control: _SourceCompletionControl, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        NativeSQLCustodyOwner,
        retained_native_sql_owners_on_current_thread,
    )

    _root, repository, payload, install = source_completion_control
    install(payload)
    close = NativeSQLCustodyOwner.close_incremental_blob
    captured: list[NativeSQLCustodyOwner] = []

    def fail_close(owner: NativeSQLCustodyOwner, blob: sqlite3.Blob) -> None:
        if not captured:
            captured.append(owner)
        if owner is captured[0]:
            raise sqlite3.OperationalError("synthetic pending Blob close failure")
        close(owner, blob)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(NativeSQLCustodyOwner, "close_incremental_blob", fail_close)
            with pytest.raises(NativeConnectionSettlementError) as failed:
                with repository._continuity._pending():
                    pytest.fail("unsettled pending reader escaped")
        assert captured and failed.value.owner is captured[0]
        assert any(owner is captured[0] for owner in retained_native_sql_owners_on_current_thread())
        assert captured[0].connection is not None and captured[0]._incremental_blobs
    finally:
        if captured:
            captured[0].close()
    assert not any(owner is captured[0] for owner in retained_native_sql_owners_on_current_thread())


def test_native_receipt_visitor_streams_exact_large_session_literal(
    source_completion_control: _SourceCompletionControl, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.audit_continuity import scan_excision_source_completion
    from polylogue.storage.sqlite.literal_cells import LITERAL_CHUNK_BYTES

    _root, _repository, payload, install = source_completion_control
    target = payload["targets"][0]
    target["session_id"] = "synthetic:" + "x" * (LITERAL_CHUNK_BYTES + 1) + 'é\\"\x00'
    original_loads = json.loads

    def bounded_scalar(value: str | bytes, *args: Any, **kwargs: Any) -> Any:
        assert len(value) < LITERAL_CHUNK_BYTES, "session literal materialized"
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", bounded_scalar)
    prepared = install(payload)

    class Visitor:
        def __init__(self) -> None:
            self.ordinals: list[int] = []
            self.counts: dict[str, int] = {}
            self.hashes: list[tuple[str, bool]] = []
            self.session_digest = hashlib.sha256()
            self.max_transfer = 0
            self.ended = 0

        def begin_embedding_intent(self) -> None:
            pass

        def embedding_incarnation(self, value: tuple[int, int] | None) -> None:
            assert value is None

        def embedding_namespace_header(self, value: tuple[int, int, int] | None) -> None:
            assert value is None

        def embedding_namespace_link_chunk(self, chunk: bytes) -> None:
            pytest.fail("absent namespace carried a link")

        def embedding_output(self, meta_present: bool, retire: bool, vector_hash: str, vector_present: bool) -> None:
            pytest.fail("absent tier carried an output")

        def embedding_presence(self, present: bool) -> None:
            assert not present

        def begin_embedding_row(self, ordinal: int) -> None:
            pytest.fail("absent tier carried a row")

        def begin_embedding_cell(self, byte_length: int) -> None:
            pytest.fail("absent tier carried a cell")

        def embedding_literal_hex_chunk(self, chunk: bytes) -> None:
            pytest.fail("absent tier carried literal bytes")

        def embedding_cell_storage_class(self, storage_class: str) -> None:
            pytest.fail("absent tier carried a storage class")

        def end_embedding_cell(self) -> None:
            pytest.fail("absent tier ended a cell")

        def embedding_row_identity(self, table: str, row_address: int | str) -> None:
            pytest.fail("absent tier carried a row identity")

        def end_embedding_row(self) -> None:
            pytest.fail("absent tier ended a row")

        def embedding_schema_version(self, version: int | None) -> None:
            assert version is None

        def end_embedding_intent(self) -> None:
            pass

        def begin_target(self, ordinal: int) -> None:
            self.ordinals.append(ordinal)

        def source_count(self, key: str, value: int) -> None:
            self.counts[key] = value

        def blob_hash(self, value: str, *, removed: bool) -> None:
            self.hashes.append((value, removed))

        def session_literal_chunk(self, chunk: bytes) -> None:
            self.max_transfer = max(self.max_transfer, len(chunk))
            self.session_digest.update(chunk)

        def end_target(self) -> None:
            self.ended += 1

    visitor = Visitor()
    assert isinstance(prepared.mutation.payload, CanonicalAuditLiteral)
    assert scan_excision_source_completion(prepared.mutation.payload, visitor) == (
        payload["operation_id"],
        payload["attempt_id"],
        payload["plan_hash"],
    )
    assert visitor.ordinals == [0] and visitor.ended == 1
    assert visitor.counts == target["counts"]
    assert visitor.hashes == [("a" * 64, True), ("b" * 64, False)]
    assert visitor.max_transfer <= LITERAL_CHUNK_BYTES
    assert (
        visitor.session_digest.hexdigest()
        == hashlib.sha256(
            json.dumps(target["session_id"], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "wrong",
    [
        "hex_length",
        "hex_case",
        "fixed_length",
        "storage_class",
        "table",
        "cell_count",
        "negative_zero",
        "output_duplicate",
        "row_duplicate",
        "absent_evidence",
        "missing_incarnation",
        "schema",
        "old_rowid",
        "vector_address_mismatch",
        "vector_address_duplicate",
    ],
)
def test_native_embedding_intent_refuses_malformed_original_literal_rows(
    source_completion_control: _SourceCompletionControl, wrong: str
) -> None:
    from tests.infra.audit_completion import present_embeddings_intent_example

    _root, _repository, payload, install = source_completion_control
    intent: dict[str, Any] = present_embeddings_intent_example()
    row = intent["rows"][0]
    first = row["cells"][0]
    if wrong == "hex_length":
        first["byte_length"] += 1
    elif wrong == "hex_case":
        first["literal_hex"] = "AF" * first["byte_length"]
    elif wrong == "fixed_length":
        first["storage_class"] = "integer"
    elif wrong == "storage_class":
        first["storage_class"] = "unknown"
    elif wrong == "table":
        row["table"] = "unowned_relation"
    elif wrong == "cell_count":
        row["cells"] = row["cells"][:-1]
    elif wrong == "negative_zero":
        # Its canonical producer sees integer zero; replace the retained
        # literal spelling below to exercise recovery grammar as well.
        row["row_address"]["physical_rowid"] = 0
    elif wrong == "output_duplicate":
        intent["outputs"] *= 2
    elif wrong == "row_duplicate":
        intent["rows"] *= 2
    elif wrong == "absent_evidence":
        intent["present"] = False
    elif wrong == "missing_incarnation":
        intent["incarnation"] = None
    elif wrong == "schema":
        intent["schema_version"] += 1
    elif wrong == "old_rowid":
        vector = intent["rows"][1]
        vector["physical_rowid"] = 1
        del vector["row_address"]
    elif wrong == "vector_address_mismatch":
        intent["rows"][1]["row_address"]["vector_derivation_hash"] = "b" * 64
    else:
        intent["rows"].append(intent["rows"][1])
    payload["embeddings_intent"] = intent
    if wrong == "negative_zero":
        from polylogue.storage.sqlite.audit_continuity import excision_completion_identity

        raw = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            .encode()
            .replace(b'"physical_rowid":0', b'"physical_rowid":-0')
        )

        def chunks() -> Generator[bytes, None, None]:
            yield raw

        literal = CanonicalAuditLiteral(len(raw), hashlib.sha256(raw).hexdigest(), chunks)
        with pytest.raises(AuditContinuityError):
            excision_completion_identity(literal)
    else:
        with pytest.raises(AuditContinuityError):
            install(payload)


def test_native_embedding_intent_preserves_present_typed_bytes_and_signed_rowid(
    source_completion_control: _SourceCompletionControl,
) -> None:
    from tests.infra.audit_completion import present_embeddings_intent_example

    _root, _repository, payload, install = source_completion_control
    payload["embeddings_intent"] = present_embeddings_intent_example()
    prepared = install(payload)
    assert isinstance(prepared.mutation.payload, CanonicalAuditLiteral)
    assert (
        b"".join(prepared.mutation.payload.verified_chunks())
        == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
