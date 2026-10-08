"""One owner decides which source window proves a retained blob.

Backup recoverability and raw derivation's blob restoration both read their
candidate windows from ``retained_blob_source_candidates``
(polylogue-ihq32), so for every window kind a raw one route can prove is a
raw the other can restore, and a source that no longer holds the bytes is
refused by both. Raw derivation restores an absent ZIP-member blob from its
container through acquisition's ZIP admission (polylogue-0y17g).
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
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode, captured_zip_member_raw_id
from polylogue.sources.source_acquisition_components import zip_acquisition_fingerprint
from polylogue.storage import backup_package as archive_backup
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.source_blob_restoration import stage_blob_from_recorded_source
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


def _seed(root: Path, case: _Case) -> tuple[str, str, str]:
    """Seed one raw whose blob is absent and whose recorded source holds it."""
    bootstrap_archive_root(root)
    sources = root / "sources"
    sources.mkdir()
    container: Path | None = None
    if case.zip_member:
        container = sources / "export.zip"
        with zipfile.ZipFile(container, "w") as archive:
            archive.writestr("conversation.json", case.blob)
        source_path = f"{container}:conversation.json"
        capture_mode, origin = "chatgpt", "chatgpt-export"
    else:
        path = sources / "rollout.jsonl"
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
        prepared, reason = stage_blob_from_recorded_source(
            archive.source_connection, root, store, raw_id, blob_hash=blob_hash, source_path=source_path
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
