"""Atomic raw/source-item admission laws."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind, raw_receipt_order_sql
from polylogue.core.enums import Origin, Provider
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    PendingPreParseRawAdmissionRequest,
    RawAdmissionPlan,
    execute_source_item_admission,
    plan_raw_admission,
)
from polylogue.storage.sqlite.archive_tiers.source_items import SourceItemAdmission, publish_source_generation
from polylogue.storage.sqlite.archive_tiers.source_write import bind_source_raw_revision
from polylogue.storage.sqlite.write_lease import write_lease

_PAYLOAD = b'{"synthetic":"source-item"}\n'
_BLOB_HASH = hashlib.sha256(_PAYLOAD).digest()


def _archive(
    tmp_path: Path, *, payload: bytes = _PAYLOAD, coordinate: str = "capture.json"
) -> tuple[sqlite3.Connection, str, str, RawAdmissionPlan]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    blob_hash = hashlib.sha256(payload).digest()
    BlobStore(root / "blob").write_from_bytes(payload)
    conn = sqlite3.connect(root / "source.db")
    conn.execute("PRAGMA foreign_keys = ON")
    (item_id,) = publish_source_generation(
        conn,
        source_generation_id="synthetic-generation",
        manifest_digest="a" * 64,
        addressing_mode="physical-file-v1",
        coordinates=(coordinate,),
        input_blob_hashes={coordinate: blob_hash},
        enumeration_fingerprint="b" * 64,
        observed_at_ms=1,
    )
    request = PendingPreParseRawAdmissionRequest(
        origin=Origin.CLAUDE_CODE_SESSION,
        capture_mode=Provider.CLAUDE_CODE,
        source_path=f"/synthetic/{coordinate}",
        canonical_source_path=f"/synthetic/{coordinate}",
        source_index=0,
        blob_hash=blob_hash,
        blob_size=len(payload),
        acquired_at_ms=2,
    )
    return conn, "synthetic-generation", item_id, plan_raw_admission(request)


def _member(item_id: str, *, coordinate: str = "record:0", entry_ordinal: int | None = None) -> SourceItemAdmission:
    return SourceItemAdmission(
        source_generation_id="synthetic-generation",
        source_item_id=item_id,
        record_coordinate=coordinate,
        entry_ordinal=entry_ordinal,
    )


def _captured_zip_archive(
    tmp_path: Path, *, content_identity: str | None = None
) -> tuple[sqlite3.Connection, RawAdmissionPlan, SourceItemAdmission]:
    """A real accepted ZIP input with its captured member coordinate."""
    import json
    import zipfile

    from polylogue.core.raw_coordinates import (
        CapturedZipMemberCoordinate,
        MemberAddressingMode,
        captured_zip_member_raw_id,
    )
    from polylogue.sources.source_staging import bind_source_input

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    source = tmp_path / "capture.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("capture.json", _PAYLOAD)
    store = BlobStore(root / "blob")
    container = source.read_bytes()
    container_hash = hashlib.sha256(container).digest()
    store.write_from_bytes(container)
    store.write_from_bytes(_PAYLOAD)
    with bind_source_input(source) as binding:
        identity = binding.captured_identity
    profile = identity.member_profile_identity("capture.json")
    assert profile is not None
    coordinate = CapturedZipMemberCoordinate(
        canonical_container=identity.canonical_source_path,
        declared_container=identity.semantic_source_path,
        member_name="capture.json",
        entry_ordinal=0,
        split_index=0,
        addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
        container_blob_hash=container_hash.hex(),
        decoder_fingerprint="b" * 64,
        profile_namespace=str(profile[0]),
    )
    conn = sqlite3.connect(root / "source.db")
    conn.execute("PRAGMA foreign_keys = ON")
    (item_id,) = publish_source_generation(
        conn,
        source_generation_id="accepted-zip",
        manifest_digest="a" * 64,
        addressing_mode="physical-file-v1",
        coordinates=(str(source),),
        source_paths={str(source): str(source)},
        input_blob_hashes={str(source): container_hash},
        captured_input_identities={str(source): identity},
        enumeration_fingerprint="b" * 64,
        observed_at_ms=1,
    )
    request = PendingPreParseRawAdmissionRequest(
        origin=Origin.CLAUDE_CODE_SESSION,
        capture_mode=Provider.CLAUDE_CODE,
        source_path=coordinate.declared_member,
        canonical_source_path=coordinate.canonical_member,
        source_index=coordinate.source_index,
        blob_hash=_BLOB_HASH,
        blob_size=len(_PAYLOAD),
        acquired_at_ms=2,
        captured_zip_coordinate=coordinate,
        addressing_mode=coordinate.addressing_mode.value,
        content_identity=content_identity,
        raw_id=captured_zip_member_raw_id(coordinate, _BLOB_HASH.hex()),
    )
    member = SourceItemAdmission(
        source_generation_id="accepted-zip",
        source_item_id=item_id,
        record_coordinate=json.dumps(["zip-v2", 0, 0, "whole_member"], separators=(",", ":")),
        entry_ordinal=0,
        split_index=0,
        addressing_mode="whole_member",
        content_identity=content_identity,
    )
    return conn, plan_raw_admission(request), member


def test_late_membership_error_rolls_back_raw_and_membership_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the raw and membership rows exist rolls both back.

    Receipt consumption is the last effect of one admission; failing it must
    leave neither the raw nor its membership edge behind.
    """
    from polylogue.storage import blob_publication

    conn, _generation, item_id, plan = _archive(tmp_path)

    consume = blob_publication.consume_blob_publication_receipt
    failed: list[bool] = []

    def late_failure(*args: Any, **kwargs: Any) -> None:
        # The raw's blob reference consumes its own receipt earlier; the
        # admission's final consumption is the one after its membership edge.
        if conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (0,):
            consume(*args, **kwargs)
            return
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        failed.append(True)
        raise ValueError("late receipt consumption failure")

    monkeypatch.setattr(blob_publication, "consume_blob_publication_receipt", late_failure)
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="late receipt consumption failure"):
        execute_source_item_admission(conn, plan, _member(item_id))

    assert failed == [True]
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (0,)
    assert conn.in_transaction
    conn.rollback()
    conn.close()


def test_zip_member_without_captured_receipt_is_refused_before_any_effect(tmp_path: Path) -> None:
    from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError

    conn, _generation, item_id, plan = _archive(tmp_path)
    conn.execute("BEGIN")
    with pytest.raises(RetainedZipMembershipUnprovedError, match="captured member receipt"):
        execute_source_item_admission(conn, plan, _member(item_id, entry_ordinal=3))

    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (0,)
    assert conn.in_transaction
    conn.rollback()
    conn.close()


def test_duplicate_admission_is_one_raw_row_and_one_membership_edge(tmp_path: Path) -> None:
    conn, _generation, item_id, plan = _archive(tmp_path)
    conn.execute("BEGIN")
    first = execute_source_item_admission(conn, plan, _member(item_id))
    duplicate = execute_source_item_admission(conn, plan, _member(item_id))
    conn.commit()

    assert first.raw_id == duplicate.raw_id == plan.raw_id
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (1,)
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (
        plan.raw_id,
        _BLOB_HASH,
    )
    conn.close()


def test_existing_member_refuses_detached_explicit_raw_and_preserves_publication(tmp_path: Path) -> None:
    conn, _generation, item_id, plan = _archive(tmp_path)
    with conn:
        conn.execute("BEGIN")
        execute_source_item_admission(conn, plan, _member(item_id))
    root = tmp_path / "archive"
    with write_lease("test.source-item.publication", archive_root=root):
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        blob_hash, _size = publisher.write_from_bytes(_PAYLOAD)
        publisher.flush()
        receipt = publisher.receipt_id(blob_hash)
    assert receipt is not None
    other = plan_raw_admission(replace(plan.request, raw_id="explicit-other-raw", blob_publication_receipt_id=receipt))
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="member raw identity changed"):
        execute_source_item_admission(conn, other, _member(item_id))
    assert conn.in_transaction
    conn.commit()  # A caught refusal cannot commit a detached raw or spend custody.
    assert conn.execute("SELECT raw_id FROM raw_sessions").fetchall() == [(plan.raw_id,)]
    assert conn.execute("SELECT raw_id FROM source_item_raw_members").fetchall() == [(plan.raw_id,)]
    assert conn.execute(
        "SELECT COUNT(*) FROM blob_publication_reservations WHERE publication_id=?",
        (receipt,),
    ).fetchone() == (1,)
    conn.close()


def test_duplicate_member_keeps_refined_origin_and_renews_exact_raw_receipt(tmp_path: Path) -> None:
    conn, _generation, item_id, initial = _archive(tmp_path)
    pending = plan_raw_admission(replace(initial.request, origin=Origin.UNKNOWN_EXPORT, raw_id=initial.raw_id))
    with conn:
        conn.execute("BEGIN")
        execute_source_item_admission(conn, pending, _member(item_id))
    with conn:
        conn.execute("BEGIN")
        execute_source_item_admission(conn, initial, _member(item_id))
    assert conn.execute("SELECT origin FROM raw_sessions").fetchone() == (Origin.CLAUDE_CODE_SESSION.value,)
    previous_receipt = conn.execute(f"SELECT {raw_receipt_order_sql()} FROM raw_sessions r").fetchone()[0]
    with conn:
        conn.execute("BEGIN")
        result = execute_source_item_admission(conn, pending, _member(item_id))
    assert result.raw_id == initial.raw_id
    assert conn.execute("SELECT origin FROM raw_sessions").fetchone() == (Origin.CLAUDE_CODE_SESSION.value,)
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
    assert conn.execute("SELECT raw_id FROM source_item_raw_members").fetchone() == (initial.raw_id,)
    assert conn.execute(f"SELECT {raw_receipt_order_sql()} FROM raw_sessions r").fetchone()[0] > previous_receipt
    conn.close()


def test_duplicate_member_preserves_revision_refined_by_actual_retained_parser(tmp_path: Path) -> None:
    from tests.infra.raw_owner_routes import converge_pending_raws_with_owner

    payload = (
        b'{"sessionId":"accepted-session","uuid":"native-message","type":"user",'
        b'"cwd":"/synthetic","message":{"role":"user","content":"retained text"}}\n'
    )
    # A Claude Code transcript is a JSONL record stream; at its declared
    # suffix the census binds it as its own session's singleton revision.
    conn, _generation, item_id, plan = _archive(tmp_path, payload=payload, coordinate="capture.jsonl")
    with conn:
        conn.execute("BEGIN")
        execute_source_item_admission(conn, plan, _member(item_id))
    conn.close()
    archive_root = tmp_path / "archive"
    report = converge_pending_raws_with_owner(archive_root, limit=32)
    assert report.failed == 0
    with sqlite3.connect(tmp_path / "archive" / "source.db") as source:
        before = source.execute(
            "SELECT logical_source_key,revision_kind,source_revision,revision_authority FROM raw_sessions WHERE raw_id=?",
            (plan.raw_id,),
        ).fetchone()
        assert before is not None
        assert before[0] == "claude-code-session:accepted-session"
        assert before[0] != plan.revision.logical_source_key
        source.execute("BEGIN")
        repeated = execute_source_item_admission(source, plan, _member(item_id))
        assert repeated.raw_id == plan.raw_id
        after = source.execute(
            "SELECT logical_source_key,revision_kind,source_revision,revision_authority FROM raw_sessions WHERE raw_id=?",
            (plan.raw_id,),
        ).fetchone()
        assert after == before


def test_zip_member_content_identity_reaches_retained_coordinate(tmp_path: Path) -> None:
    """Admission retains the digest needed to reacquire a reserialized member.

    Anti-vacuity: omit ``content_identity`` from the admission request or its
    call to ``record_raw_container_coordinate`` and the retained value is NULL.
    """
    conn, plan, member = _captured_zip_archive(tmp_path, content_identity="d" * 64)
    conn.execute("BEGIN")
    result = execute_source_item_admission(conn, plan, member)
    conn.commit()
    assert conn.execute(
        "SELECT content_identity FROM raw_container_coordinates WHERE raw_id=?", (result.raw_id,)
    ).fetchone() == ("d" * 64,)
    conn.close()


def test_retired_raw_membership_cannot_be_readmitted(tmp_path: Path) -> None:
    conn, _generation, item_id, plan = _archive(tmp_path)
    conn.execute("BEGIN")
    execute_source_item_admission(conn, plan, _member(item_id))
    conn.commit()
    conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (plan.raw_id,))
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (
        None,
        _BLOB_HASH,
    )
    conn.commit()
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="retired; readmission is forbidden"):
        execute_source_item_admission(conn, plan, _member(item_id))
    assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
    assert conn.execute("SELECT raw_id, raw_blob_hash FROM source_item_raw_members").fetchone() == (
        None,
        _BLOB_HASH,
    )
    conn.rollback()
    conn.close()


def test_record_retry_preserves_a_later_canonical_revision_binding(tmp_path: Path) -> None:
    """An exact admitted edge prevents replay of the original pending envelope."""
    conn, _generation, item_id, plan = _archive(tmp_path)
    with conn:
        conn.execute("BEGIN")
        execute_source_item_admission(conn, plan, _member(item_id))
    bind_source_raw_revision(
        conn,
        plan.raw_id,
        RawRevisionEnvelope(
            logical_source_key="claude-code-session:parsed-session",
            kind=RawRevisionKind.FULL,
            source_revision="parsed-revision",
            acquisition_generation=0,
        ),
    )
    with conn:
        conn.execute("BEGIN")
        result = execute_source_item_admission(conn, plan, _member(item_id))
    assert result.raw_id == plan.raw_id
    assert conn.execute("SELECT logical_source_key, source_revision FROM raw_sessions").fetchone() == (
        "claude-code-session:parsed-session",
        "parsed-revision",
    )
    conn.close()


@pytest.mark.parametrize("decoder_fingerprint", ["b" * 64, "c" * 64])
def test_captured_zip_member_requires_the_accepted_decoder(tmp_path: Path, decoder_fingerprint: str) -> None:
    """Correct container evidence cannot authorize a different decoder's raw identity."""
    import json
    import zipfile

    from polylogue.core.raw_coordinates import (
        CapturedZipMemberCoordinate,
        MemberAddressingMode,
        captured_zip_member_raw_id,
    )
    from polylogue.sources.source_staging import bind_source_input

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    source = tmp_path / "capture.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("capture.json", _PAYLOAD)
    store = BlobStore(root / "blob")
    container = source.read_bytes()
    container_hash = hashlib.sha256(container).digest()
    store.write_from_bytes(container)
    store.write_from_bytes(_PAYLOAD)
    with bind_source_input(source) as binding:
        identity = binding.captured_identity
    profile = identity.member_profile_identity("capture.json")
    assert profile is not None
    coordinate = CapturedZipMemberCoordinate(
        canonical_container=identity.canonical_source_path,
        declared_container=identity.semantic_source_path,
        member_name="capture.json",
        entry_ordinal=0,
        split_index=0,
        addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
        container_blob_hash=container_hash.hex(),
        decoder_fingerprint=decoder_fingerprint,
        profile_namespace=str(profile[0]),
    )
    with sqlite3.connect(root / "source.db") as conn:
        (item_id,) = publish_source_generation(
            conn,
            source_generation_id="accepted-zip",
            manifest_digest="a" * 64,
            addressing_mode="physical-file-v1",
            coordinates=(str(source),),
            source_paths={str(source): str(source)},
            input_blob_hashes={str(source): container_hash},
            captured_input_identities={str(source): identity},
            enumeration_fingerprint="b" * 64,
            observed_at_ms=1,
        )
        request = PendingPreParseRawAdmissionRequest(
            origin=Origin.CLAUDE_CODE_SESSION,
            capture_mode=Provider.CLAUDE_CODE,
            source_path=coordinate.declared_member,
            canonical_source_path=coordinate.canonical_member,
            source_index=coordinate.source_index,
            blob_hash=_BLOB_HASH,
            blob_size=len(_PAYLOAD),
            acquired_at_ms=2,
            captured_zip_coordinate=coordinate,
            addressing_mode=coordinate.addressing_mode.value,
            raw_id=captured_zip_member_raw_id(coordinate, _BLOB_HASH.hex()),
        )
        plan = plan_raw_admission(request)
        member = SourceItemAdmission(
            source_generation_id="accepted-zip",
            source_item_id=item_id,
            record_coordinate=json.dumps(["zip-v2", 0, 0, "whole_member"], separators=(",", ":")),
            entry_ordinal=0,
            split_index=0,
            addressing_mode="whole_member",
        )
        conn.execute("BEGIN")
        if decoder_fingerprint != "b" * 64:
            with pytest.raises(ValueError, match="exact accepted physical input"):
                execute_source_item_admission(conn, plan, member)
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
            assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (0,)
        else:
            from dataclasses import replace

            wrong_address = replace(member, record_coordinate='["zip-v2",1,0,"whole_member"]')
            with pytest.raises(ValueError, match="captured ZIP coordinate"):
                execute_source_item_admission(conn, plan, wrong_address)
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (0,)
            assert execute_source_item_admission(conn, plan, member).raw_id == plan.raw_id
            assert conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (1,)


@pytest.mark.parametrize("second_decoder", ["b" * 64, "c" * 64])
def test_completed_group_equivalence_requires_the_same_decoder(tmp_path: Path, second_decoder: str) -> None:
    """Identical member sets do not erase the accepted enumeration authority."""
    from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError
    from polylogue.storage.sqlite.archive_tiers.source_items import (
        ConnectionCompletedSourceItemRead,
        complete_source_item_enumeration,
        retained_completed_source_item_for_raw,
    )

    conn, generation, item_id, plan = _archive(tmp_path)
    try:
        conn.execute("BEGIN")
        execute_source_item_admission(conn, plan, _member(item_id))
        complete_source_item_enumeration(
            conn,
            source_generation_id=generation,
            source_item_id=item_id,
            enumeration_fingerprint="b" * 64,
            record_coordinates=iter(("record:0",)),
            enumerated_at_ms=3,
        )
        (second_item,) = publish_source_generation(
            conn,
            source_generation_id="another-generation",
            manifest_digest="a" * 64,
            addressing_mode="physical-file-v1",
            coordinates=("capture.json",),
            input_blob_hashes={"capture.json": _BLOB_HASH},
            enumeration_fingerprint=second_decoder,
            observed_at_ms=1,
            commit=False,
        )
        execute_source_item_admission(
            conn,
            plan,
            SourceItemAdmission(
                source_generation_id="another-generation",
                source_item_id=second_item,
                record_coordinate="record:0",
            ),
        )
        complete_source_item_enumeration(
            conn,
            source_generation_id="another-generation",
            source_item_id=second_item,
            enumeration_fingerprint=second_decoder,
            record_coordinates=iter(("record:0",)),
            enumerated_at_ms=3,
        )
        if second_decoder != "b" * 64:
            with pytest.raises(RetainedZipMembershipUnprovedError):
                retained_completed_source_item_for_raw(ConnectionCompletedSourceItemRead(conn), plan.raw_id)
        else:
            assert retained_completed_source_item_for_raw(ConnectionCompletedSourceItemRead(conn), plan.raw_id) in {
                (generation, item_id),
                ("another-generation", second_item),
            }
    finally:
        conn.rollback()
        conn.close()
