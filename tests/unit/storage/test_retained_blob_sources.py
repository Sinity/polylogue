"""One owner decides which source window proves a retained blob.

Backup recoverability and raw derivation's blob restoration both read their
candidate windows from ``retained_blob_sources`` (polylogue-ihq32), so for
every window kind a raw one route can prove is a raw the other can restore,
and a source that no longer holds the bytes is refused by both. Raw
derivation restores an absent ZIP-member blob from its container through
acquisition's ZIP admission (polylogue-0y17g). The same owner re-anchors a
recorded path at the archive root in force (polylogue-u5hs1) and provides
first-retention and latest-reference append hypotheses that consumers verify
by exact hash (polylogue-ojfkc).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes
from polylogue.core.raw_coordinates import relocated_source_path
from polylogue.operations import archive_backup
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import record_raw_container_coordinate
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.durable_tier_fixtures import seed_durable_tier

_RECORD = b'{"type":"session_meta","payload":{"id":"window"}}\n'
_EARLIER = b'{"type":"session_meta","payload":{"id":"earlier"}}\n'
_LATER = b'{"type":"event_msg","payload":{"type":"later"}}\n'


@dataclass(frozen=True)
class _Case:
    kind: str
    blob: bytes
    source: bytes
    revision_kind: str
    source_index: int
    offsets: tuple[int, int] | None = None
    predecessor_size: int | None = None
    zip_member: bool = False


_CASES = {
    "direct_file": _Case("direct_file_sha256", _RECORD, _RECORD, "unknown", 1),
    "historical_snapshot_prefix": _Case("historical_snapshot_prefix_sha256", _RECORD, _RECORD + _LATER, "full", 0),
    "append_window": _Case(
        "live_append_segment_sha256",
        _RECORD,
        _EARLIER + _RECORD + _LATER,
        "append",
        -1,
        offsets=(len(_EARLIER), len(_EARLIER) + len(_RECORD)),
    ),
    "append_observed_prefix": _Case(
        "live_append_segment_sha256",
        _EARLIER + _RECORD,
        _EARLIER + _RECORD + _LATER,
        "append",
        -1,
        offsets=(len(_EARLIER), len(_EARLIER) + len(_RECORD)),
    ),
    "legacy_append": _Case(
        "historical_append_segment_sha256",
        _RECORD,
        _EARLIER + _RECORD,
        "unknown",
        -1,
        predecessor_size=len(_EARLIER),
    ),
    "zip_member": _Case(
        "zip_reacquired_payload",
        dumps_bytes({"id": "zipped", "mapping": {"node": {"message": {"author": {"role": "user"}}}}}),
        b"",
        "unknown",
        0,
        zip_member=True,
    ),
}


def _insert_raw(
    conn: sqlite3.Connection,
    raw_id: str,
    *,
    origin: str,
    capture_mode: str,
    source_path: str,
    source_index: int,
    blob: bytes,
    acquired_at_ms: int,
    revision_kind: str,
    offsets: tuple[int, int] | None = None,
) -> None:
    """One raw row and its ``raw_payload`` receipt, as admission writes them.

    The raw-session rowid is immutable first-retention order; rows are received in call order.
    """
    blob_hash = hashlib.sha256(blob).digest()
    conn.execute(
        """INSERT INTO raw_sessions (
            raw_id, origin, capture_mode, source_path, source_index, blob_hash, blob_size,
            acquired_at_ms, revision_kind, append_start_offset, append_end_offset, validation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'passed')""",
        (
            raw_id,
            origin,
            capture_mode,
            source_path,
            source_index,
            blob_hash,
            len(blob),
            acquired_at_ms,
            revision_kind,
            offsets[0] if offsets else None,
            offsets[1] if offsets else None,
        ),
    )
    conn.execute(
        """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
        VALUES (?, ?, 'raw_payload', ?, ?, ?)""",
        (blob_hash, raw_id, source_path, len(blob), acquired_at_ms),
    )


def _seed(
    root: Path,
    case: _Case,
    *,
    recorded_root: Path | None = None,
    predecessor_acquired_at_ms: int = 1,
) -> tuple[str, str, str]:
    """Seed one raw whose blob is absent and whose source holds it.

    ``recorded_root`` is the archive root the path was recorded under; the
    source file itself is always written under ``root``, so a different
    recorded root is an archive that moved after acquisition.
    """
    bootstrap_archive_root(root)
    sources = root / "inbox"
    sources.mkdir()
    recorded_sources = (recorded_root or root) / "inbox"
    if case.zip_member:
        container = sources / "export.zip"
        with zipfile.ZipFile(container, "w") as archive:
            archive.writestr("conversation.json", case.blob)
        source_path = f"{recorded_sources / 'export.zip'}:conversation.json"
        capture_mode, origin = "chatgpt", "chatgpt-export"
    else:
        path = sources / "rollout.jsonl"
        path.write_bytes(case.source)
        source_path = str(recorded_sources / "rollout.jsonl")
        capture_mode, origin = "codex", "codex-session"
    blob_hash = hashlib.sha256(case.blob).hexdigest()
    with seed_durable_tier(root / "source.db") as conn:
        if case.predecessor_size is not None:
            _insert_raw(
                conn,
                "predecessor",
                origin=origin,
                capture_mode=capture_mode,
                source_path=source_path,
                source_index=0,
                blob=_EARLIER,
                acquired_at_ms=predecessor_acquired_at_ms,
                revision_kind="full",
            )
        _insert_raw(
            conn,
            "retained",
            origin=origin,
            capture_mode=capture_mode,
            source_path=source_path,
            source_index=case.source_index,
            blob=case.blob,
            acquired_at_ms=2,
            revision_kind=case.revision_kind,
            offsets=case.offsets,
        )
        if case.zip_member:
            record_raw_container_coordinate(
                conn,
                "retained",
                coordinate_format="zip-v2",
                entry_ordinal=0,
                split_index=0,
                addressing_mode="whole_member",
                manage_transaction=False,
            )
    return "retained", blob_hash, source_path


def _backup_proof(root: Path, blob_hash: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    unproven: list[dict[str, str]] = []
    proofs = archive_backup._source_recoverability_proofs(
        root / "source.db",
        root=root,
        missing_hashes={blob_hash},
        unproven=unproven,
        immutable=False,
    )
    return proofs, unproven


def _raw_restoration(root: Path, raw_id: str, blob_hash: str) -> tuple[bool, str | None]:
    store = BlobStore(root / "blob")
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        prepared, reason = RawObservationDerivation(root)._stage_blob_from_recorded_source(
            archive, store, raw_id, blob_hash=blob_hash
        )
    if prepared is not None:
        assert prepared.hash_hex == blob_hash
        store.discard_prepared(prepared)
    return prepared is not None, reason


@pytest.mark.parametrize("name", sorted(_CASES))
def test_both_routes_prove_and_restore_the_same_window(tmp_path: Path, name: str) -> None:
    """Anti-vacuity: each case is a window kind only one route read before the
    owner existed -- raw derivation read neither the legacy append window nor
    a ZIP member, and backup's direct-file proof read only a whole unchanged
    file -- so dropping any candidate kind from the owner fails one route here.
    """
    case = _CASES[name]
    raw_id, blob_hash, _source_path = _seed(tmp_path, case)

    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    restored, reason = _raw_restoration(tmp_path, raw_id, blob_hash)

    assert [proof["kind"] for proof in proofs] == [case.kind], unproven
    assert (restored, reason) == (True, None)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_both_routes_refuse_a_source_that_no_longer_holds_the_bytes(tmp_path: Path, name: str) -> None:
    case = _CASES[name]
    raw_id, blob_hash, source_path = _seed(tmp_path, case)
    if case.zip_member:
        container = Path(source_path.rsplit(":", 1)[0])
        with zipfile.ZipFile(container, "w") as archive:
            archive.writestr("conversation.json", dumps_bytes({"id": "rewritten", "mapping": {}}))
    else:
        Path(source_path).write_bytes(case.source.replace(b"window", b"rewrite"))

    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    restored, reason = _raw_restoration(tmp_path, raw_id, blob_hash)

    assert proofs == [] and [row["blob_hash"] for row in unproven] == [blob_hash]
    assert restored is False and reason is not None


@pytest.mark.parametrize("name", ["zip_member", "direct_file", "legacy_append"])
def test_both_routes_restore_from_a_source_that_moved_with_the_archive_root(tmp_path: Path, name: str) -> None:
    """A path recorded under the archive's ``inbox`` is read where the archive now is.

    Anti-vacuity: raw derivation read the recorded path, so it refused with
    ``source_missing`` or ``container_coordinate_missing`` while backup
    re-anchored the same path and proved it (polylogue-u5hs1).
    """
    case = _CASES[name]
    root = tmp_path / "archive-now"
    raw_id, blob_hash, source_path = _seed(root, case, recorded_root=tmp_path / "archive-then")
    assert not Path(source_path.split(":", 1)[0] if case.zip_member else source_path).exists()

    proofs, unproven = _backup_proof(root, blob_hash)
    restored, reason = _raw_restoration(root, raw_id, blob_hash)

    assert [proof["kind"] for proof in proofs] == [case.kind], unproven
    assert proofs[0]["source_path"].startswith(str(root / "inbox"))
    assert (restored, reason) == (True, None)


def test_existing_external_source_wins_over_same_tail_under_archive_root(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    external = tmp_path / "external" / "inbox" / "rollout.jsonl"
    external.parent.mkdir(parents=True)
    external.write_bytes(_RECORD)
    (root / "inbox").mkdir()
    (root / "inbox" / "rollout.jsonl").write_bytes(_RECORD.replace(b"window", b"unrelated"))
    blob_hash = hashlib.sha256(_RECORD).hexdigest()
    with seed_durable_tier(root / "source.db") as conn:
        _insert_raw(
            conn,
            "external-raw",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(external),
            source_index=1,
            blob=_RECORD,
            acquired_at_ms=1,
            revision_kind="unknown",
        )

    proofs, unproven = _backup_proof(root, blob_hash)
    restored, reason = _raw_restoration(root, "external-raw", blob_hash)

    assert [proof["source_path"] for proof in proofs] == [str(external)], unproven
    assert (restored, reason) == (True, None)


def test_inaccessible_literal_source_is_not_replaced_by_same_tail_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    literal = tmp_path / "external" / "inbox" / "rollout.jsonl"
    candidate = root / "inbox" / "rollout.jsonl"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(_RECORD)
    literal.parent.mkdir(parents=True)
    literal.write_bytes(_RECORD)
    original_stat = Path.stat

    def denied_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == literal:
            raise PermissionError("synthetic inaccessible literal")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", denied_stat)

    with pytest.raises(PermissionError, match="synthetic inaccessible literal"):
        relocated_source_path(literal, root)


def test_a_window_less_append_finds_its_predecessor_in_first_retention_order(tmp_path: Path) -> None:
    """The predecessor is the full observation received before the append, whatever the clock said.

    The predecessor's ``acquired_at_ms`` is later than the append's (the clock
    stepped back between them), and a second full observation received
    after the append has a larger size and the earliest clock.

    Anti-vacuity: ordering by ``acquired_at_ms`` finds no earlier full
    observation, so both routes refuse with ``legacy_append_window_missing``
    / ``no_source_window``; picking the later retained row replays the wrong window.
    """
    case = _CASES["legacy_append"]
    raw_id, blob_hash, source_path = _seed(tmp_path, case, predecessor_acquired_at_ms=50)
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "received-later",
            origin="codex-session",
            capture_mode="codex",
            source_path=source_path,
            source_index=0,
            blob=_EARLIER + _RECORD + _LATER,
            acquired_at_ms=0,
            revision_kind="full",
        )

    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    restored, reason = _raw_restoration(tmp_path, raw_id, blob_hash)

    assert [proof["kind"] for proof in proofs] == ["historical_append_segment_sha256"], unproven
    assert (proofs[0]["append_start_offset"], proofs[0]["append_end_offset"]) == (
        str(len(_EARLIER)),
        str(len(_EARLIER) + len(_RECORD)),
    )
    assert (restored, reason) == (True, None)


def test_replacing_an_earlier_raw_payload_reference_does_not_reorder_append_history(
    tmp_path: Path,
) -> None:
    """A mutable blob-ref receipt cannot move a full anchor past its append.

    Re-observation replaces the content-addressed ``blob_refs`` row and moves
    its rowid. The first retained ``raw_sessions`` row remains the chronology
    evidence, and the append's exact hash proves the inferred window.
    """
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "rollout.jsonl"
    source.parent.mkdir()
    source.write_bytes(_EARLIER + _RECORD)
    append_hash = hashlib.sha256(_RECORD).hexdigest()
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "full-a",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=0,
            blob=_EARLIER,
            acquired_at_ms=1,
            revision_kind="full",
        )
        _insert_raw(
            conn,
            "append-b",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=_RECORD,
            acquired_at_ms=2,
            revision_kind="unknown",
        )
        conn.execute(
            """INSERT OR REPLACE INTO blob_refs
               (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
               SELECT blob_hash, raw_id, 'raw_payload', source_path, blob_size, 3
               FROM raw_sessions WHERE raw_id = 'full-a'"""
        )

    proofs, unproven = _backup_proof(tmp_path, append_hash)
    restored, reason = _raw_restoration(tmp_path, "append-b", append_hash)

    assert [proof["kind"] for proof in proofs] == ["historical_append_segment_sha256"], unproven
    assert (proofs[0]["append_start_offset"], proofs[0]["append_end_offset"]) == (
        str(len(_EARLIER)),
        str(len(_EARLIER + _RECORD)),
    )
    assert (restored, reason) == (True, None)


def test_reobserved_append_after_a_new_full_anchor_offers_hash_proven_windows(
    tmp_path: Path,
) -> None:
    """Both immutable and latest-reference histories are hypotheses, not authority.

    The same window-less append B is retained before full D, then observed
    again after D. Its replaceable blob reference cannot represent both
    observations. The latest-receipt hypothesis finds C after D+B; exact
    digest and size checks select that candidate while rejecting the
    first-retention hypothesis at the end of D.
    """
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "repeated-rollout.jsonl"
    source.parent.mkdir()
    full_d = _EARLIER + _RECORD + _LATER
    append_c = b'{"type":"event_msg","payload":{"type":"new"}}\n'
    source.write_bytes(full_d + _RECORD + append_c)
    append_hash = hashlib.sha256(append_c).hexdigest()
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "full-a",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=0,
            blob=_EARLIER,
            acquired_at_ms=1,
            revision_kind="full",
        )
        _insert_raw(
            conn,
            "append-b",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=_RECORD,
            acquired_at_ms=2,
            revision_kind="unknown",
        )
        _insert_raw(
            conn,
            "full-d",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=0,
            blob=full_d,
            acquired_at_ms=3,
            revision_kind="full",
        )
        conn.execute(
            """INSERT OR REPLACE INTO blob_refs
               (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
               SELECT blob_hash, raw_id, 'raw_payload', source_path, blob_size, 4
               FROM raw_sessions WHERE raw_id = 'append-b'"""
        )
        _insert_raw(
            conn,
            "append-c",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=append_c,
            acquired_at_ms=5,
            revision_kind="unknown",
        )

    from polylogue.storage.source_blob_restoration import (
        SourceByteWindow,
        read_raw_source_evidence,
        retained_blob_sources_many,
    )

    with seed_durable_tier(tmp_path / "source.db") as conn:
        row = read_raw_source_evidence(conn, "append-c")
        assert row is not None
        (candidate_source,) = retained_blob_sources_many(conn, (row,), root=tmp_path).values()
    candidate_windows = [candidate.window for candidate in candidate_source.candidates]
    assert candidate_windows == [
        SourceByteWindow(len(full_d), len(full_d) + len(append_c)),
        SourceByteWindow(
            len(full_d) + len(_RECORD),
            len(full_d) + len(_RECORD) + len(append_c),
        ),
    ]

    proofs, unproven = _backup_proof(tmp_path, append_hash)
    restored, reason = _raw_restoration(tmp_path, "append-c", append_hash)

    assert [proof["kind"] for proof in proofs] == ["historical_append_segment_sha256"], unproven
    assert (proofs[0]["append_start_offset"], proofs[0]["append_end_offset"]) == (
        str(len(full_d) + len(_RECORD)),
        str(len(full_d) + len(_RECORD) + len(append_c)),
    )
    assert (restored, reason) == (True, None)


def test_reobserved_anchors_do_not_authorize_an_unproven_legacy_window(tmp_path: Path) -> None:
    """First/latest receipt orderings are hypotheses, never chronology authority."""
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "ambiguous-rollout.jsonl"
    source.parent.mkdir()
    full_d = b"full-D-prefix-longer-than-A\n"
    append_c = b"new-tail-after-reused-append\n"
    source.write_bytes(full_d + _RECORD + append_c)
    append_hash = hashlib.sha256(append_c).hexdigest()
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "full-a",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=0,
            blob=_EARLIER,
            acquired_at_ms=1,
            revision_kind="full",
        )
        _insert_raw(
            conn,
            "append-b",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=_RECORD,
            acquired_at_ms=2,
            revision_kind="unknown",
        )
        _insert_raw(
            conn,
            "full-d",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=0,
            blob=full_d,
            acquired_at_ms=3,
            revision_kind="full",
        )
        conn.execute(
            """INSERT OR REPLACE INTO blob_refs
               (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
               SELECT blob_hash, raw_id, 'raw_payload', source_path, blob_size, 4
               FROM raw_sessions WHERE raw_id = 'append-b'"""
        )
        _insert_raw(
            conn,
            "append-c",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=append_c,
            acquired_at_ms=5,
            revision_kind="unknown",
        )
        # Re-observing both anchors after C moves their mutable receipts after
        # C. Neither first-retention nor latest-reference order then proves C.
        for raw_id, stamp in (("full-a", 6), ("full-d", 7)):
            conn.execute(
                """INSERT OR REPLACE INTO blob_refs
                   (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
                   SELECT blob_hash, raw_id, 'raw_payload', source_path, blob_size, ?
                   FROM raw_sessions WHERE raw_id = ?""",
                (stamp, raw_id),
            )

    proofs, unproven = _backup_proof(tmp_path, append_hash)
    restored, reason = _raw_restoration(tmp_path, "append-c", append_hash)

    assert proofs == []
    assert unproven[0]["kind"] == "legacy_append_coordinates_unproven"
    assert unproven[0]["reason"] == "legacy_append_coordinates_unproven"
    assert restored is False and reason == "legacy_append_coordinates_unproven"

    adapter = RawObservationDerivation(tmp_path)
    frame = raw_observation_frame(tmp_path)
    replacement = adapter.compute(frame, "append-c")
    assert replacement.missing_source_coordinate_refusal is not None
    assert adapter.publish(frame, replacement)
    assert adapter.inspect(raw_observation_frame(tmp_path), ("append-c",))["append-c"] == "valid"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        refusal = conn.execute(
            "SELECT artifact_kind, support_status FROM raw_artifacts WHERE raw_id = 'append-c'"
        ).fetchone()
    assert refusal == ("terminal_missing_source_coordinates", "unknown")
    from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle

    lifecycle = read_raw_failure_lifecycle(tmp_path / "source.db")
    assert lifecycle.missing_source_coordinates == 1
    assert lifecycle.state == "degraded"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        from polylogue.operations.status_workload import raw_failure_status_from_connection

        status = raw_failure_status_from_connection(conn)
    assert status["raw_missing_source_coordinates"] == 1
    assert status["raw_failure_lifecycle_state"] == "degraded"


def test_coordinate_refusal_retirement_invalidates_only_stale_non_session_census(tmp_path: Path) -> None:
    """Exact append coordinates invalidate both refusal and its derived census."""
    from polylogue.storage.sqlite.archive_tiers.source_write import retire_missing_source_coordinate_refusal

    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "append.jsonl"
    source.parent.mkdir()
    source.write_bytes(_RECORD)
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "append-refused",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=_RECORD,
            acquired_at_ms=1,
            revision_kind="unknown",
        )
        conn.execute(
            """INSERT INTO raw_artifacts
               (artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                classification_reason, support_status, parse_as_session, schema_eligible,
                first_observed_at_ms, last_observed_at_ms)
               VALUES ('refusal', 'append-refused', 'codex-session', ?, -1,
                       'terminal_missing_source_coordinates', 'reason', 'unknown', 0, 0, 1, 1)""",
            (str(source),),
        )
        conn.execute(
            """INSERT INTO raw_membership_census
               (raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail, revision_authority)
               VALUES ('append-refused', 'old-parser', 'non_session', 0, 1, NULL, NULL)"""
        )
        conn.commit()
        retire_missing_source_coordinate_refusal(conn, "append-refused")
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE raw_id = 'append-refused' "
            "AND artifact_kind = 'terminal_missing_source_coordinates'"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_membership_census WHERE raw_id = 'append-refused'"
        ).fetchone() == (0,)


def test_cancel_after_staging_first_blob_discards_owned_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancellation after one staged blob cannot orphan its temporary file."""
    import asyncio

    bootstrap_archive_root(tmp_path)
    source_a = tmp_path / "inbox" / "a.jsonl"
    source_b = tmp_path / "inbox" / "b.jsonl"
    source_a.parent.mkdir()
    source_a.write_bytes(_RECORD)
    source_b.write_bytes(_LATER)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_a = archive.write_raw_payload(
            provider=Provider.CODEX, payload=_RECORD, source_path=str(source_a), acquired_at_ms=1
        )
        raw_b = archive.write_raw_payload(
            provider=Provider.CODEX, payload=_LATER, source_path=str(source_b), acquired_at_ms=2
        )
        descriptors = {raw_id: archive.raw_revision_descriptor(raw_id) for raw_id in (raw_a, raw_b)}
    store = BlobStore(tmp_path / "blob")
    for _raw_id, blob_hash, _path, _kind, _size in descriptors.values():
        store.blob_path(blob_hash).unlink()

    adapter = RawObservationDerivation(tmp_path)
    original = adapter._stage_blob_from_recorded_source
    calls = 0

    def stage_then_cancel(*args: object, **kwargs: object) -> tuple[PreparedBlob | None, str | None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(adapter, "_stage_blob_from_recorded_source", stage_then_cancel)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        with pytest.raises(asyncio.CancelledError):
            adapter._stage_absent_blob_restorations(archive, (raw_a, raw_b), descriptors)
    assert not any(store.staging_root.iterdir())


def test_inaccessible_windowless_source_stays_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "unanchored.jsonl"
    source.parent.mkdir()
    payload = b'{"type":"event_msg"}\n'
    source.write_bytes(payload)
    payload_hash = hashlib.sha256(payload).hexdigest()
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "unanchored-append",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=-1,
            blob=payload,
            acquired_at_ms=1,
            revision_kind="unknown",
        )

    original_open = Path.open

    def denied_source_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == source:
            raise PermissionError("synthetic source read denial")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_source_open)
    with pytest.raises(PermissionError, match="synthetic source read denial"):
        _raw_restoration(tmp_path, "unanchored-append", payload_hash)
    proofs, unproven = _backup_proof(tmp_path, payload_hash)
    assert proofs == []
    assert unproven[0]["kind"] == "replay_error"


def test_a_second_window_less_append_starts_where_the_first_ends(tmp_path: Path) -> None:
    """Window-less appends are contiguous: the second one's window follows the first.

    Anti-vacuity: starting every window-less append at the end of the full
    observation replays ``_RECORD``'s bytes for ``_LATER`` and both routes
    refuse with a hash mismatch.
    """
    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "rollout.jsonl"
    source.parent.mkdir()
    source.write_bytes(_EARLIER + _RECORD + _LATER)
    blob_hash = hashlib.sha256(_LATER).hexdigest()
    with seed_durable_tier(tmp_path / "source.db") as conn:
        for raw_id, blob, source_index, revision_kind in (
            ("full", _EARLIER, 0, "full"),
            ("first-append", _RECORD, -1, "unknown"),
            ("second-append", _LATER, -1, "unknown"),
        ):
            _insert_raw(
                conn,
                raw_id,
                origin="codex-session",
                capture_mode="codex",
                source_path=str(source),
                source_index=source_index,
                blob=blob,
                acquired_at_ms=1,
                revision_kind=revision_kind,
            )

    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    restored, reason = _raw_restoration(tmp_path, "second-append", blob_hash)

    assert [(proof["append_start_offset"], proof["append_end_offset"]) for proof in proofs] == [
        (str(len(_EARLIER + _RECORD)), str(len(_EARLIER + _RECORD + _LATER)))
    ], unproven
    assert (restored, reason) == (True, None)


def test_many_window_less_appends_share_one_first_retention_order_scan(tmp_path: Path) -> None:
    from polylogue.storage.source_blob_restoration import read_raw_source_evidence, retained_blob_sources_many

    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "many-rollouts.jsonl"
    source.parent.mkdir(exist_ok=True)
    prefix = _EARLIER
    appends = [f'{{"seq":{index}}}\n'.encode() for index in range(48)]
    source.write_bytes(prefix + b"".join(appends))
    with seed_durable_tier(tmp_path / "source.db") as conn:
        _insert_raw(
            conn,
            "many-full",
            origin="codex-session",
            capture_mode="codex",
            source_path=str(source),
            source_index=0,
            blob=prefix,
            acquired_at_ms=1,
            revision_kind="full",
        )
        for index, blob in enumerate(appends):
            _insert_raw(
                conn,
                f"many-append-{index}",
                origin="codex-session",
                capture_mode="codex",
                source_path=str(source),
                source_index=-1,
                blob=blob,
                acquired_at_ms=index + 2,
                revision_kind="unknown",
            )
        rows = tuple(
            row
            for index in range(len(appends))
            if (row := read_raw_source_evidence(conn, f"many-append-{index}")) is not None
        )
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        sources = retained_blob_sources_many(conn, rows, root=tmp_path)

        scans = [statement for statement in statements if "WITH raw_cohort AS MATERIALIZED" in statement]
        assert len(scans) == 2
        assert len(sources) == len(appends)
        offset = len(prefix)
        for index, blob in enumerate(appends):
            candidates = sources[f"many-append-{index}"].candidates
            assert candidates[0].window is not None
            assert candidates[0].window.start == offset
            assert candidates[0].window.end == offset + len(blob)
            offset += len(blob)


def test_distinct_legacy_paths_restrict_history_scans_to_their_source_cohort(tmp_path: Path) -> None:
    from polylogue.storage.source_blob_restoration import read_raw_source_evidence, retained_blob_sources_many

    bootstrap_archive_root(tmp_path)
    path_count = 24
    rows = []
    with seed_durable_tier(tmp_path / "source.db") as conn:
        for index in range(path_count):
            source = tmp_path / "inbox" / f"source-{index}.jsonl"
            source.parent.mkdir(exist_ok=True)
            prefix = f'{{"anchor":{index}}}\n'.encode()
            append = f'{{"append":{index}}}\n'.encode()
            source.write_bytes(prefix + append)
            _insert_raw(
                conn,
                f"distinct-full-{index}",
                origin="codex-session",
                capture_mode="codex",
                source_path=str(source),
                source_index=0,
                blob=prefix,
                acquired_at_ms=index * 2 + 1,
                revision_kind="full",
            )
            _insert_raw(
                conn,
                f"distinct-append-{index}",
                origin="codex-session",
                capture_mode="codex",
                source_path=str(source),
                source_index=-1,
                blob=append,
                acquired_at_ms=index * 2 + 2,
                revision_kind="unknown",
            )
            for decoy in range(4):
                _insert_raw(
                    conn,
                    f"unrelated-{index}-{decoy}",
                    origin="codex-session",
                    capture_mode="codex",
                    source_path=str(tmp_path / "unrelated" / f"{index}-{decoy}.jsonl"),
                    source_index=0,
                    blob=b'{"unrelated":true}\n',
                    acquired_at_ms=path_count * 2 + index * 4 + decoy,
                    revision_kind="full",
                )
            row = read_raw_source_evidence(conn, f"distinct-append-{index}")
            assert row is not None
            rows.append(row)

        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        sources = retained_blob_sources_many(conn, tuple(rows), root=tmp_path)

        scans = [statement for statement in statements if "WITH raw_cohort AS MATERIALIZED" in statement]
        assert len(scans) == path_count * 2
        assert all("SELECT rowid AS first_retained_order" in statement for statement in scans)
        plan = conn.execute(f"EXPLAIN QUERY PLAN {scans[0]}").fetchall()
        plan_details = [str(row[-1]) for row in plan]
        assert any("SEARCH raw_sessions USING INDEX idx_raw_sessions_source_path" in detail for detail in plan_details)
        assert any("SEARCH blob_refs USING INDEX idx_blob_refs_ref_id" in detail for detail in plan_details)
        assert not any("SCAN raw_sessions" in detail or "SCAN blob_refs" in detail for detail in plan_details)
        assert len(sources) == path_count
        for index in range(path_count):
            (candidate,) = sources[f"distinct-append-{index}"].candidates
            assert candidate.window is not None
            assert candidate.window.start == len(f'{{"anchor":{index}}}\n'.encode())


def _chatgpt_conversation(name: str) -> dict[str, object]:
    return {
        "id": name,
        "title": name,
        "create_time": 1,
        "current_node": "m",
        "mapping": {
            "m": {
                "id": "m",
                "parent": None,
                "children": [],
                "message": {
                    "id": "m",
                    "author": {"role": "user"},
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": [name]},
                },
            },
        },
    }


def test_an_absent_zip_member_blob_is_restored_and_materialized(tmp_path: Path) -> None:
    """The production raw route restores a lost ZIP-member blob from its container.

    While the container no longer admits the member the raw is a named,
    retryable refusal and nothing is staged. Once the container holds the
    byte-identical member again, the writer restores the blob and the next
    pass materializes the session.

    Anti-vacuity: the former route refused every container member with
    ``container_member`` and the session never materialized.
    """
    from polylogue.daemon.derivation import DerivationRegistry, DerivationReport, converge
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError

    bootstrap_archive_root(tmp_path)
    member = dumps_bytes(_chatgpt_conversation("zipped"))
    container = tmp_path / "exports" / "export.zip"
    container.parent.mkdir()
    source_path = f"{container}:conversations.json"
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=member, source_path=source_path, acquired_at_ms=1
        )
        record_raw_container_coordinate(
            archive.source_connection,
            raw_id,
            coordinate_format="zip-v2",
            entry_ordinal=0,
            split_index=0,
            addressing_mode="whole_member",
            manage_transaction=True,
        )
    store = BlobStore(tmp_path / "blob")
    blob_path = store.blob_path(hashlib.sha256(member).hexdigest())
    blob_path.unlink()

    with zipfile.ZipFile(container, "w") as bundle:
        bundle.writestr("notes.txt", b"no conversation member")
    with pytest.raises(RetainedPreparationRetryableError, match=r"not restorable from its source \(\w+\)"):
        RawObservationDerivation(tmp_path).compute(raw_observation_frame(tmp_path), raw_id)
    assert not blob_path.exists()
    assert not any(store.staging_root.iterdir())

    with zipfile.ZipFile(container, "w") as bundle:
        bundle.writestr("conversations.json", member)

    def run() -> DerivationReport:
        return converge(DerivationRegistry((RawObservationDerivation(tmp_path),)), raw_observation_frame(tmp_path))

    restoring = run()
    assert restoring.failed == 0
    assert blob_path.read_bytes() == member
    assert restoring.done + run().done == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)
