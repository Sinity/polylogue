"""One owner decides which source window proves a retained blob.

Backup recoverability and raw derivation's blob restoration both read their
candidate windows from ``retained_blob_sources_many``
(polylogue-ihq32), so for every window kind a raw one route can prove is a
raw the other can restore, and a source that no longer holds the bytes is
refused by both. Raw derivation restores an absent ZIP-member blob from its
container through acquisition's ZIP admission (polylogue-0y17g).
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
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode, captured_zip_member_raw_id
from polylogue.sources.source_acquisition_components import zip_acquisition_fingerprint
from polylogue.storage import backup_package as archive_backup
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.source_blob_restoration import (
    read_raw_source_evidence,
    retained_blob_sources,
    retained_source_location,
    stage_blob_from_sources,
)
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


def _seed(
    root: Path, case: _Case, *, sources: Path | None = None, filename: str = "rollout.jsonl"
) -> tuple[str, str, str]:
    """Seed one raw whose blob is absent and whose recorded source holds it."""
    bootstrap_archive_root(root)
    sources = sources if sources is not None else root / "sources"
    sources.mkdir(parents=True)
    container: Path | None = None
    if case.zip_member:
        container = sources / "export.zip"
        with zipfile.ZipFile(container, "w") as archive:
            archive.writestr("conversation.json", case.blob)
        source_path = f"{container}:conversation.json"
        capture_mode, origin = "chatgpt", "chatgpt-export"
    else:
        path = sources / filename
        path.write_bytes(case.source)
        source_path = str(path)
        capture_mode, origin = "codex", "codex-session"
    blob_hash = hashlib.sha256(case.blob).hexdigest()
    with seed_durable_tier(root / "source.db") as conn:
        if case.predecessor_size is not None:
            conn.execute(
                """INSERT INTO raw_sessions (
                    raw_id, origin, capture_mode, source_path, source_index, blob_hash, blob_size,
                    acquired_at_ms, revision_kind, validation_status
                ) VALUES ('predecessor', ?, ?, ?, 0, ?, ?, 1, 'full', 'passed')""",
                (origin, capture_mode, source_path, hashlib.sha256(_EARLIER).digest(), case.predecessor_size),
            )
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, revision_kind, append_start_offset, append_end_offset, validation_status
            ) VALUES ('retained', ?, ?, ?, ?, ?, ?, 2, ?, ?, ?, 'passed')""",
            (
                origin,
                capture_mode,
                source_path,
                case.source_index,
                bytes.fromhex(blob_hash),
                len(case.blob),
                case.revision_kind,
                case.offsets[0] if case.offsets else None,
                case.offsets[1] if case.offsets else None,
            ),
        )
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
               SELECT blob_hash, raw_id, 'raw_payload', source_path, blob_size, acquired_at_ms
               FROM raw_sessions ORDER BY rowid"""
        )
        if case.zip_member:
            assert container is not None
            record_raw_container_coordinate(
                conn,
                "retained",
                coordinate_format="zip-v2",
                entry_ordinal=0,
                split_index=0,
                addressing_mode="whole_member",
                captured_coordinate=CapturedZipMemberCoordinate(
                    str(container.resolve()),
                    str(container.resolve()),
                    "conversation.json",
                    0,
                    0,
                    MemberAddressingMode.WHOLE_MEMBER,
                    hashlib.sha256(container.read_bytes()).hexdigest(),
                    zip_acquisition_fingerprint(Provider.CHATGPT),
                ),
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


def _raw_restoration(root: Path, raw_id: str, blob_hash: str, source_path: str) -> tuple[bool, str | None]:
    store = BlobStore(root / "blob")
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        row = read_raw_source_evidence(archive.source_connection, raw_id)
        assert row is not None
        prepared, reason = stage_blob_from_sources(
            row,
            store,
            retained_blob_sources(archive.source_connection, row, root=root),
            blob_hash=blob_hash,
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
    raw_id, blob_hash, source_path = _seed(tmp_path, case)

    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    restored, reason = _raw_restoration(tmp_path, raw_id, blob_hash, source_path)

    assert [proof["kind"] for proof in proofs] == [case.kind], unproven
    assert (restored, reason) == (True, None)


@pytest.mark.parametrize("directory", ["inbox", "browser-capture", "hooks"])
@pytest.mark.parametrize("name", ["direct_file", "zip_member"])
@pytest.mark.parametrize("original_present", [True, False])
def test_recorded_source_precedence_and_absent_source_relocation(
    tmp_path: Path, directory: str, name: str, original_present: bool
) -> None:
    """Both consumers must recover exact bytes from the authoritative location.

    Restoring relocation-first selection fails the present-original cases;
    removing relocation fails the absent-original cases. A loose ZIP-looking
    colon filename must stay literal without a captured member receipt.
    """
    root = tmp_path / "archive"
    case = _CASES[name]
    raw_id, blob_hash, source_path = _seed(
        root, case, sources=tmp_path / "original" / directory, filename="loose.zip:conversation.json"
    )
    original = tmp_path / "original" / directory / ("export.zip" if case.zip_member else "loose.zip:conversation.json")
    relocated = root / directory / original.name
    relocated.parent.mkdir()
    original_bytes = original.read_bytes()
    if original_present:
        if case.zip_member:
            with zipfile.ZipFile(relocated, "w") as bundle:
                bundle.writestr("conversation.json", dumps_bytes({"id": "wrong", "mapping": {}}))
        else:
            relocated.write_bytes(_EARLIER)
    else:
        original.replace(relocated)
    proofs, unproven = _backup_proof(root, blob_hash)
    assert [proof["kind"] for proof in proofs] == [case.kind], unproven
    assert _raw_restoration(root, raw_id, blob_hash, source_path) == (True, None)
    assert (original if original_present else relocated).read_bytes() == original_bytes


@pytest.mark.parametrize("fault", [PermissionError(13, "denied"), ValueError("invalid path")])
def test_recorded_source_stat_fault_cannot_authorize_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: Exception
) -> None:
    original = tmp_path / "original" / "inbox" / "input.jsonl"
    root = tmp_path / "archive"
    relocated = root / "inbox" / original.name
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(_RECORD)
    stat = Path.stat

    def fail_original(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == original:
            raise fault
        return stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_original)
    with pytest.raises(type(fault)) as refused:
        retained_source_location({"source_path": str(original)}, root)
    assert refused.value is fault


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
    restored, reason = _raw_restoration(tmp_path, raw_id, blob_hash, source_path)

    assert proofs == [] and [row["blob_hash"] for row in unproven] == [blob_hash]
    assert restored is False and reason is not None


def test_windowless_append_uses_receipt_order_despite_inverted_clocks(tmp_path: Path) -> None:
    """A later full receipt and clock rollback cannot redefine the prior window."""
    bootstrap_archive_root(tmp_path)
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(_EARLIER + _RECORD)
    source_path = str(path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:

        def observe(payload: bytes, clock: int, *, source_index: int = 0) -> str:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=clock,
                source_index=source_index,
            )

        predecessor = observe(_EARLIER, 9000)
        later = observe(_LATER, 8000)
        assert observe(_EARLIER, 1000) == predecessor
        raw_id = observe(_RECORD, 500, source_index=-1)
        assert observe(_LATER, 0) == later
        archive.commit()
    blob_hash = hashlib.sha256(_RECORD).hexdigest()
    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    assert [proof["kind"] for proof in proofs] == ["historical_append_segment_sha256"], unproven
    assert _raw_restoration(tmp_path, raw_id, blob_hash, source_path) == (True, None)


def test_coordinate_less_zip_is_not_inferred_after_root_relocation(tmp_path: Path) -> None:
    raw_id, blob_hash, source_path = _seed(tmp_path, _CASES["zip_member"])
    original = Path(source_path.rsplit(":", 1)[0])
    relocated = tmp_path / "inbox" / original.name
    relocated.parent.mkdir()
    original.replace(relocated)
    recorded_path = f"/absent/archive/inbox/{original.name}:conversation.json"
    with seed_durable_tier(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_container_coordinates WHERE raw_id = ?", (raw_id,))
        conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = ?", (recorded_path, raw_id))
    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    assert proofs == []
    assert [row["blob_hash"] for row in unproven] == [blob_hash]
    assert _raw_restoration(tmp_path, raw_id, blob_hash, recorded_path)[0] is False


def test_windowless_append_without_a_receipt_cannot_infer_a_predecessor(tmp_path: Path) -> None:
    raw_id, blob_hash, source_path = _seed(tmp_path, _CASES["legacy_append"])
    with seed_durable_tier(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM blob_refs WHERE ref_id = ? AND ref_type = 'raw_payload'", (raw_id,))
    proofs, unproven = _backup_proof(tmp_path, blob_hash)
    assert proofs == []
    assert [row["blob_hash"] for row in unproven] == [blob_hash]
    assert _raw_restoration(tmp_path, raw_id, blob_hash, source_path) == (False, "no_source_window")


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


@pytest.mark.asyncio
async def test_an_absent_zip_member_blob_is_restored_and_materialized(tmp_path: Path) -> None:
    """Real retained replay restores exact ZIP bytes and settles its Index receipt."""
    from contextlib import closing

    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    member = dumps_bytes(_chatgpt_conversation("zipped"))
    container = tmp_path / "exports" / "export.zip"

    def seed() -> str:
        bootstrap_archive_root(tmp_path)
        container.parent.mkdir()
        with zipfile.ZipFile(container, "w") as bundle:
            bundle.writestr("conversations.json", member)
        coordinate = CapturedZipMemberCoordinate(
            str(container.resolve()),
            str(container.resolve()),
            "conversations.json",
            0,
            0,
            MemberAddressingMode.WHOLE_MEMBER,
            hashlib.sha256(container.read_bytes()).hexdigest(),
            zip_acquisition_fingerprint(Provider.CHATGPT),
        )
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=member,
                source_path=coordinate.declared_member,
                canonical_source_path=coordinate.canonical_member,
                source_index=coordinate.source_index,
                addressing_mode=coordinate.addressing_mode.value,
                raw_id=captured_zip_member_raw_id(coordinate, hashlib.sha256(member).hexdigest()),
                acquired_at_ms=1,
                captured_zip_coordinate=coordinate,
                capture_mode=Provider.CHATGPT,
                post_parse=True,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)
    store = BlobStore(tmp_path / "blob")
    blob_path = store.blob_path(hashlib.sha256(member).hexdigest())
    with closing(sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True)) as conn:
        assert conn.execute(
            "SELECT source_path, blob_hash, blob_size FROM raw_sessions WHERE raw_id=?", (raw_id,)
        ).fetchone() == (f"{container.resolve()}:conversations.json", hashlib.sha256(member).digest(), len(member))
        assert conn.execute(
            "SELECT entry_ordinal, split_index, addressing_mode, captured_coordinate IS NOT NULL "
            "FROM raw_container_coordinates WHERE raw_id=?",
            (raw_id,),
        ).fetchone() == (0, 0, "whole_member", 1)
    blob_path.unlink()
    with zipfile.ZipFile(container, "w") as bundle:
        bundle.writestr("notes.txt", b"no conversation member")

    async with prepared_live_convergence_owner(tmp_path) as owner:
        with pytest.raises(RetainedPreparationRetryableError, match=r"not restorable from its source \(\w+\)"):
            (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
        assert not blob_path.exists()
        assert not any(store.staging_root.iterdir())
        with zipfile.ZipFile(container, "w") as bundle:
            bundle.writestr("conversations.json", member)
        restoring = await owner.converge_raw_id(raw_id)
        assert restoring.failed == 0, restoring.outcomes
        assert blob_path.read_bytes() == member
        receipts = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
        assert sum(receipt.replayed_logical_sources for receipt in receipts) == 1

    with closing(sqlite3.connect(f"file:{tmp_path / 'index.db'}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (1,)


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
    restored, reason = _raw_restoration(tmp_path, "append-b", append_hash, str(source))

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
    restored, reason = _raw_restoration(tmp_path, "append-c", append_hash, str(source))

    assert [proof["kind"] for proof in proofs] == ["historical_append_segment_sha256"], unproven
    assert (proofs[0]["append_start_offset"], proofs[0]["append_end_offset"]) == (
        str(len(full_d) + len(_RECORD)),
        str(len(full_d) + len(_RECORD) + len(append_c)),
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
    restored, reason = _raw_restoration(tmp_path, "second-append", blob_hash, str(source))

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


def test_history_resolution_stops_when_the_compute_owner_cancels(tmp_path: Path) -> None:
    import threading

    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.storage.source_blob_restoration import retained_blob_sources_many

    raw_id, _hash, _path = _seed(tmp_path, _CASES["legacy_append"])
    with seed_durable_tier(tmp_path / "source.db") as conn:
        row = read_raw_source_evidence(conn, raw_id)
        assert row is not None
        cancelled = threading.Event()
        scans: list[str] = []

        def cancel_during_history(statement: str) -> None:
            if "WITH raw_cohort AS MATERIALIZED" in statement:
                scans.append(statement)
                cancelled.set()

        token = compute_cancel.set(cancelled)
        conn.set_trace_callback(cancel_during_history)
        try:
            with pytest.raises(DaemonOperationCancelled):
                retained_blob_sources_many(conn, (row,), root=tmp_path)
        finally:
            conn.set_trace_callback(None)
            compute_cancel.reset(token)
        assert len(scans) == 1
        assert retained_blob_sources_many(conn, (row,), root=tmp_path)[raw_id].candidates


def test_raw_restoration_resolves_one_history_for_a_shared_append_path(tmp_path: Path) -> None:
    from polylogue.storage.derived.raw import RawObservationDerivation

    bootstrap_archive_root(tmp_path)
    source = tmp_path / "inbox" / "rollout.jsonl"
    source.parent.mkdir()
    source.write_bytes(_EARLIER + _RECORD + _LATER)
    blobs = {"full": _EARLIER, "first": _RECORD, "second": _LATER}
    with seed_durable_tier(tmp_path / "source.db") as conn:
        for index, (raw_id, blob) in enumerate(blobs.items()):
            _insert_raw(
                conn,
                raw_id,
                origin="codex-session",
                capture_mode="codex",
                source_path=str(source),
                source_index=0 if index == 0 else -1,
                blob=blob,
                acquired_at_ms=index,
                revision_kind="full" if index == 0 else "unknown",
            )
    descriptors = {
        raw_id: (Provider.CODEX, hashlib.sha256(blob).hexdigest(), str(source), "unknown", len(blob))
        for raw_id, blob in blobs.items()
        if raw_id != "full"
    }
    import asyncio

    from polylogue.core.compute import BoundedComputeAdapter
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def restore(compute: BoundedComputeAdapter) -> None:
        statements: list[str] = []
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            archive.source_connection.set_trace_callback(statements.append)
            restorations = adapter._stage_absent_blob_restorations(archive, ("first", "second"), descriptors)
            assert restorations is not None
            try:
                assert {raw_id: prepared.temporary_path.read_bytes() for raw_id, prepared in restorations.staged} == {
                    "first": _RECORD,
                    "second": _LATER,
                }
                assert len([s for s in statements if "WITH raw_cohort AS MATERIALIZED" in s]) == 2
            finally:
                restorations.discard()

    async def exercise() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            await owner.run_convergence_sync("test.raw-observation.law", restore, owner._compute_adapter)

    asyncio.run(exercise())
