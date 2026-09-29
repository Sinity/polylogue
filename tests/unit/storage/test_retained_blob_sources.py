"""One owner decides which source window proves a retained blob.

Backup recoverability and raw derivation's blob restoration both read their
candidate windows from ``retained_blob_sources`` (polylogue-ihq32), so for
every window kind a raw one route can prove is a raw the other can restore,
and a source that no longer holds the bytes is refused by both. Raw
derivation restores an absent ZIP-member blob from its container through
acquisition's ZIP admission (polylogue-0y17g). The same owner re-anchors a
recorded path at the archive root in force (polylogue-u5hs1) and orders a
window-less append's predecessor by receipt, not wall clock
(polylogue-ojfkc).
"""

from __future__ import annotations

import hashlib
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes
from polylogue.operations import archive_backup
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage.blob_store import BlobStore
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

    The receipt's rowid is the observation order; rows are received in call order.
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


def test_a_window_less_append_finds_its_predecessor_in_receipt_order(tmp_path: Path) -> None:
    """The predecessor is the full observation received before the append, whatever the clock said.

    The predecessor's ``acquired_at_ms`` is later than the append's (the clock
    stepped back between them), and a second full observation received
    after the append has a larger size and the earliest clock.

    Anti-vacuity: ordering by ``acquired_at_ms`` finds no earlier full
    observation, so both routes refuse with ``legacy_append_window_missing``
    / ``no_source_window``; picking the later receipt replays the wrong window.
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
