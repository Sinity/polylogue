"""Begun Audit provenance permits preparation, never physical removal."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.core.types import ContentHash
from polylogue.operations.audit import AuditRepository
from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs
from polylogue.operations.mutation_transaction import (
    StartedBoundMutation,
    _authorized_removal_apply,
)
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError, ReferenceSealStaleError
from polylogue.storage.sqlite.write_lease import authorized_session_removal, write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.excision_execution import begin_excision_control
from tests.infra.storage_records import SessionBuilder


def _begin(root: Path) -> tuple[StartedBoundMutation, SessionExcisionArgs]:
    root.mkdir(parents=True, exist_ok=True)
    with write_lease("test.excision-provenance-begin", archive_root=root):
        bootstrap_archive_root(root)
        builder = SessionBuilder(root / "index.db", "excision-provenance").provider("codex").add_message(text="Neutral")
        builder.save()
        return begin_excision_control(
            root,
            builder.native_session_id(),
            reason="synthetic",
            actor="synthetic:operator",
        )


def test_begun_original_proof_is_distinct_from_physical_apply_authority(tmp_path: Path) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        attempt_id = seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        assert attempt_id.startswith("attempt:")
        with pytest.raises(ReferenceSealError):
            seal._require_begun_excision_apply()
        with _authorized_removal_apply(started.plan, tmp_path, SessionExcisionActuator(), args):
            seal._require_begun_excision_apply()
            seal._require_begun_excision_apply()
        with pytest.raises(ReferenceSealError):
            seal._require_begun_excision_apply()


@pytest.mark.parametrize("wrong", ["operation", "plan", "target"])
def test_excision_preparation_refuses_unbound_operation_plan_or_target(tmp_path: Path, wrong: str) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        with pytest.raises(ReferenceSealError):
            seal.bind_begun_excision(
                "operation:unrelated" if wrong == "operation" else started.operation_id,
                "0" * 64 if wrong == "plan" else started.plan.plan_hash,
                ("codex:unrelated" if wrong == "target" else args.session_id,),
            )
        assert seal._begun_excision is None


def test_source_only_cannot_acquire_user_removal_preparation(tmp_path: Path) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
        with pytest.raises(ReferenceSealError):
            seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        assert set(seal._observers) == {"source"}


def test_begun_preparation_refuses_foreign_plan_even_under_actual_custody(tmp_path: Path) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        with (
            write_lease("test.foreign-excision-frame", archive_root=tmp_path),
            authorized_session_removal(
                archive_root=tmp_path, plan_hash="0" * 64, session_ids=(args.session_id,), excise_assertions=True
            ),
        ):
            with pytest.raises(ReferenceSealError):
                seal._require_begun_excision_apply()


def test_original_audit_change_refuses_before_preparation_provenance(tmp_path: Path) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        with write_lease("test.excision-audit-currency", archive_root=tmp_path):
            AuditRepository.for_archive_root(tmp_path).recover_abandoned_attempts()
        with pytest.raises(ReferenceSealStaleError):
            seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        assert seal._begun_excision is None


def test_begun_plan_refuses_changed_session_revision_before_capture(tmp_path: Path) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with write_lease("test.excision-changed-revision", archive_root=tmp_path):
        changed = SessionBuilder(tmp_path / "index.db", "excision-provenance").provider("codex")
        changed.conv = changed.conv.model_copy(update={"content_hash": ContentHash("f" * 64)})
        changed.add_message(text="Neutral").add_message(text="New synthetic revision").save()
        with closing(
            open_isolated_write_connection(
                tmp_path / "index.db", purpose="test.excision-changed-revision-read", archive_root=tmp_path
            )
        ) as index:
            with connection_cursor(
                index, "SELECT lower(hex(content_hash)) FROM sessions WHERE session_id=?", (args.session_id,)
            ) as rows:
                assert rows.fetchone()[0] == "f" * 64
        assert changed.native_session_id() == args.session_id
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        with pytest.raises(ReferenceSealStaleError):
            seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        assert seal._begun_excision is None
        assert seal._pending_tier_permits == {}


def test_begun_plan_refuses_new_user_assertion_before_effect_preparation(tmp_path: Path) -> None:
    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    with write_lease("test.excision-changed-user-population", archive_root=tmp_path):
        with closing(
            open_isolated_write_connection(
                tmp_path / "user.db",
                purpose="test.excision-changed-user-population",
                archive_root=tmp_path,
            )
        ) as user:
            upsert_assertion(
                user,
                assertion_id="new-unrelated-user-obligation",
                target_ref="user:local",
                kind=AssertionKind.EXCISION_RECORD,
                value={"synthetic": "survives"},
                now_ms=1,
            )
            user.commit()
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        with pytest.raises(ReferenceSealStaleError):
            seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        assert seal._begun_excision is None
        assert seal._pending_tier_permits == {}


@pytest.mark.parametrize("coordinate", ["raw", "hook", "container", "material"])
def test_user_preparation_refuses_substituted_frozen_source_coordinate(tmp_path: Path, coordinate: str) -> None:
    from dataclasses import replace

    from polylogue.security.excision import (
        ContainerDisposition,
        ContainerItem,
        ExcisionRawTarget,
        ExcisionReceipt,
        _stage_excision_user_receipt,
        excision_target_from_replay,
    )

    started, args = _begin(tmp_path)
    assert started.operation_id is not None
    targets = started.plan.context["targets"]
    assert isinstance(targets, list) and len(targets) == 1
    original = excision_target_from_replay(targets[0])
    if coordinate == "raw":
        substitute = replace(original, raw_targets=(ExcisionRawTarget("foreign-raw", b"x" * 32, "foreign/path"),))
    elif coordinate == "hook":
        substitute = replace(original, hook_event_ids=("foreign-hook",))
    elif coordinate == "container":
        substitute = replace(
            original,
            containers=ContainerDisposition(
                removable_items=(ContainerItem("foreign-generation", "foreign-item", None),)
            ),
        )
    else:
        substitute = replace(original, material_ids=("foreign-material",), material_blob_hashes=(b"y" * 32,))
    receipt = ExcisionReceipt(session_id=args.session_id, found=True, reason="synthetic", excised_at_ms=2)
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        attempt_id = seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (args.session_id,))
        with pytest.raises(ReferenceSealError):
            seal.original_excision_target(args.session_id)
        with seal.original_read_snapshot(), seal.user_producer():
            assert seal.original_excision_target(args.session_id) == original
            with pytest.raises(ReferenceSealError):
                _stage_excision_user_receipt(
                    seal,
                    substitute,
                    receipt,
                    operation_id=started.operation_id,
                    attempt_id=attempt_id,
                    plan_hash=started.plan.plan_hash,
                )
            with seal.user_rows("SELECT count(*) FROM assertions") as rows:
                assert rows.fetchone()[0] == 0
            with connection_cursor(seal._scratch, "SELECT count(*) FROM temp.known_tier_effects") as rows:
                assert rows.fetchone()[0] == 0
            assert seal._pending_tier_permits == {}
            with pytest.raises(ReferenceSealError):
                seal.original_excision_target("codex:foreign")


@pytest.mark.parametrize("shared", [False, True])
def test_selected_source_liveness_reads_surviving_postimage_without_resurrecting_target(
    tmp_path: Path, shared: bool
) -> None:
    from polylogue.security.excision import (
        _load_excision_source_target,
        _PreparedExcisionBlobSourceRead,
        _stage_excision_raw_delete,
    )
    from polylogue.storage.blob_liveness import LivenessState, inspect_session_blob_references
    from polylogue.storage.blob_store import BlobStore

    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    with write_lease("test.selected-excision-liveness-seed", archive_root=root):
        bootstrap_archive_root(root)
        builder = (
            SessionBuilder(root / "index.db", "selected-source-liveness").provider("codex").add_message(text="Neutral")
        )
        builder.save()
        session_id = builder.native_session_id()
        _origin, _, native_id = session_id.partition(":")
        digest, size = BlobStore(root / "blob").write_from_bytes(b"Neutral retained raw")
        blob_hash = bytes.fromhex(digest)
        with closing(sqlite3.connect(root / "source.db")) as source:
            for raw_id, raw_native in (
                ("owned-raw", native_id),
                *((("surviving-raw", "unrelated"),) if shared else ()),
            ):
                with connection_cursor(
                    source,
                    "INSERT INTO raw_sessions(raw_id,origin,native_id,source_path,blob_hash,blob_size,acquired_at_ms) "
                    "VALUES (?,'codex',?,?,?,?,1)",
                    (raw_id, raw_native, f"synthetic/{raw_id}", blob_hash, size),
                ):
                    pass
            source.commit()
        with closing(sqlite3.connect(root / "index.db")) as index:
            with connection_cursor(index, "UPDATE sessions SET raw_id='owned-raw' WHERE session_id=?", (session_id,)):
                pass
            index.commit()
    started, args = begin_excision_control(root, session_id, reason="synthetic")
    assert started.operation_id is not None
    with PreparedIndexMutation(root / "index.db", archive_root=root) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (session_id,))
        with seal.original_read_snapshot(), seal.source_producer():
            target = seal.original_excision_target(session_id)
            assert tuple(raw.raw_id for raw in target.raw_targets) == ("owned-raw",)
            _load_excision_source_target(seal, target)
            candidates = seal.excision_source_blob_page()
            assert len(candidates) == 1 and candidates[0][0] == blob_hash
            assert seal._literal_scalar_equal(candidates[0][1], "owned-raw")
            assert _stage_excision_raw_delete(seal, "owned-raw") == 1
            source_read = _PreparedExcisionBlobSourceRead(seal)
            actual = inspect_session_blob_references(
                source_read,
                (blob_hash,),
                index_conn=seal.observer("index"),
                excluding_session_ids=frozenset({session_id}),
            )[blob_hash]
            assert actual.state is (LivenessState.LIVE if shared else LivenessState.UNREFERENCED)
            assert actual.surfaces == (("source.db.raw_sessions",) if shared else ())
            with seal.source_rows("SELECT raw_id FROM raw_sessions ORDER BY raw_id") as rows:
                assert [row[0] for row in rows] == (["surviving-raw"] if shared else [])
            seal.record_excision_blob_disposition(blob_hash, removed=not shared)
            assert tuple(seal.excision_source_target_hashes(session_id, removed=not shared)) == (blob_hash,)
            assert seal.excision_source_blob_page(after=blob_hash) == ()
            with pytest.raises(ReferenceSealError):
                _stage_excision_raw_delete(seal, "owned-raw")
            with pytest.raises(ReferenceSealError):
                seal.record_excision_source_blob(session_id, seal.retain_literal_scalar(blob_hash), candidates[0][1])
        with closing(sqlite3.connect(root / "source.db")) as source:
            with connection_cursor(source, "SELECT raw_id FROM raw_sessions ORDER BY raw_id") as rows:
                assert [row[0] for row in rows] == (["owned-raw", "surviving-raw"] if shared else ["owned-raw"])


def test_frozen_cascade_blob_exclusion_exceeds_actual_index_variable_limit(tmp_path: Path) -> None:
    """A frozen closure uses membership rather than one SQL variable per target."""
    from polylogue.security.excision import _PreparedExcisionSessionClosure
    from polylogue.storage.blob_liveness import (
        ConnectionSessionBlobLivenessRead,
        LivenessState,
        inspect_session_blob_references,
    )

    root = tmp_path
    excluded_hash, shared_hash = b"e" * 32, b"s" * 32
    with write_lease("test.excision-large-frozen-closure", archive_root=root):
        bootstrap_archive_root(root)
        builders = [
            SessionBuilder(root / "index.db", f"cascade-{ordinal}").provider("codex").add_message(text="Neutral")
            for ordinal in range(24)
        ]
        survivor = SessionBuilder(root / "index.db", "survivor").provider("codex").add_message(text="Neutral")
        for builder in (*builders, survivor):
            builder.save()
        parent_id = builders[0].native_session_id()
        with closing(sqlite3.connect(root / "index.db")) as index:
            with connection_cursor(index, "SELECT message_id FROM messages WHERE session_id=?", (parent_id,)) as rows:
                branch_point = rows.fetchone()[0]
            for builder in builders[1:]:
                with connection_cursor(
                    index,
                    "INSERT INTO session_links(src_session_id,dst_origin,dst_native_id,link_type,"
                    "resolved_dst_session_id,branch_point_message_id,inheritance,confidence,evidence_json,observed_at_ms) "
                    "VALUES (?,'codex-session','cascade-0','branch',?,?,'prefix-sharing',1.0,'[]',1)",
                    (builder.native_session_id(), parent_id, branch_point),
                ):
                    pass
            for attachment_id, blob_hash in (
                ("excluded-attachment", excluded_hash),
                ("shared-attachment", shared_hash),
            ):
                with connection_cursor(
                    index,
                    "INSERT INTO attachments(attachment_id,blob_hash,acquisition_status) VALUES (?,?,'acquired')",
                    (attachment_id, blob_hash),
                ):
                    pass
            for builder in (*builders, survivor):
                session_id = builder.native_session_id()
                with connection_cursor(
                    index, "SELECT message_id FROM messages WHERE session_id=?", (session_id,)
                ) as rows:
                    message_id = rows.fetchone()[0]
                for position, attachment_id in enumerate(
                    ("shared-attachment",) if builder is survivor else ("excluded-attachment", "shared-attachment")
                ):
                    with connection_cursor(
                        index,
                        "INSERT INTO attachment_refs(attachment_id,session_id,message_id,position,native_identity) VALUES (?,?,?,?,?)",
                        (attachment_id, session_id, message_id, position, attachment_id.encode().hex()),
                    ):
                        pass
            index.commit()
    started, _ = begin_excision_control(root, parent_id, reason="synthetic", cascade_lineage=True)
    assert started.operation_id is not None
    session_ids = tuple(ref.removeprefix("session:") for ref in started.plan.target_refs)
    assert len(session_ids) == 24
    with PreparedIndexMutation(root / "index.db", archive_root=root) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, session_ids)
        with seal.original_read_snapshot(), seal.source_producer():
            index = seal.observer("index")
            source = ConnectionSessionBlobLivenessRead(seal.observer("source"))
            hashes = (excluded_hash, shared_hash)
            baseline = inspect_session_blob_references(
                source, hashes, index_conn=index, excluding_session_ids=frozenset(session_ids)
            )
            old_limit = index.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
            try:
                closure = _PreparedExcisionSessionClosure(seal)
                assert len(closure) > index.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
                actual = inspect_session_blob_references(
                    source, hashes, index_conn=index, excluding_session_ids=closure
                )
                assert actual == baseline
                assert actual[excluded_hash].state is LivenessState.UNREFERENCED
                assert actual[excluded_hash].surfaces == ()
                assert actual[shared_hash].state is LivenessState.LIVE
                assert actual[shared_hash].surfaces == ("index.db.attachment_refs",)
                assert survivor.native_session_id() not in closure
            finally:
                index.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old_limit)
