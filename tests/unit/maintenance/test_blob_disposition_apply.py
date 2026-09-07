"""Fault matrix for consuming an accepted blob disposition plan.

Apply restores sole copies and then unlinks what no durable row names, so
every test here names the mutation that would make it red: deleting before
the spool holds the material, deleting something a row still references,
letting an unreferenced orphan survive because nothing proved it, trusting a
stale plan, or converting a blocked member into a silent success.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.browser_capture.receiver import write_capture_envelope_bytes
from polylogue.maintenance.blob_disposition import (
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionPlan,
    build_disposition_context,
    compile_disposition_plan,
)
from polylogue.maintenance.blob_disposition_apply import (
    DIRECT_UNLINK_DETAIL,
    INVALID_ENTRY_COHORT,
    DispositionApplyReceipt,
    MemberOutcome,
    RestorationOutcome,
    apply_disposition_plan,
    restore_plan_members,
    write_receipt,
)
from polylogue.sources.hooks import read_hook_spool_record
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

# A fixed mtime well in the past: the GC seam inherits production GC's defense
# against clock skew, and reading the wall clock to clear it would only make
# the test time-dependent.
_AGED_MTIME = 1_700_000_000


def _hook_envelope(event_id: str = "event-1", *, text: str = "ran a tool") -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "PreToolUse",
        "session_id": "session-1",
        "timestamp": "2026-07-15T02:15:39Z",
        "provider": "claude-code",
        "payload": {"tool_name": "Bash", "detail": text},
    }


def _stored_bytes(envelope: dict[str, object], tmp_path: Path) -> bytes:
    scratch = tmp_path / f"scratch-{envelope['event_id']}.json"
    scratch.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    return json.dumps(read_hook_spool_record(scratch), ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")


def _capture_bytes(
    session_id: str = "conv-123",
    *,
    captured_at: str = "2026-04-24T00:00:00+00:00",
    turns: list[dict[str, str]] | None = None,
) -> bytes:
    envelope = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": f"https://chatgpt.com/c/{session_id}",
            "page_title": "ChatGPT - Work plan",
            "captured_at": captured_at,
            "adapter_name": "chatgpt-dom-v1",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": session_id,
            "title": "Work plan",
            "turns": turns
            or [
                {"provider_turn_id": "u1", "role": "user", "text": "Draft"},
                {"provider_turn_id": "a1", "role": "assistant", "text": "Here"},
            ],
        },
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _revision_pair(session_id: str) -> tuple[bytes, bytes]:
    """One session captured twice: a later capture that also carries more turns.

    Both halves are needed. The dedup fingerprint deliberately excludes
    provenance, so two captures that differ only in ``captured_at`` are one
    content revision; only the extra turn makes the spool choose between them.
    """
    earlier = _capture_bytes(session_id, captured_at="2026-04-24T00:00:00+00:00")
    later = _capture_bytes(
        session_id,
        captured_at="2026-04-24T09:00:00+00:00",
        turns=[
            {"provider_turn_id": "u1", "role": "user", "text": "Draft"},
            {"provider_turn_id": "a1", "role": "assistant", "text": "Here"},
            {"provider_turn_id": "u2", "role": "user", "text": "And the risks?"},
        ],
    )
    return earlier, later


_EARLIER_CAPTURE, _LATER_CAPTURE = _revision_pair("conv-revised")


def _write_spool_file(root: Path, envelope: dict[str, object]) -> Path:
    target = root / "pending" / "2026-07-15"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{envelope['event_id']}.json"
    path.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=4), encoding="utf-8")
    return path


def _stub_archive(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """An archive with only the relations planning reads.

    Enough for classification, restoration and every refusal; the GC seam
    reports its own missing durable schema, so nothing here can be unlinked.
    """
    archive_root = tmp_path / "archive"
    blob_root = archive_root / "blob"
    blob_root.mkdir(parents=True)
    hooks_root = archive_root / "hooks"
    hooks_root.mkdir()
    capture_spool = archive_root / "browser-capture"
    capture_spool.mkdir()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("CREATE TABLE blob_refs (blob_hash BLOB, ref_type TEXT)")
        conn.execute(
            "CREATE TABLE raw_sessions (raw_id TEXT, origin TEXT, native_id TEXT, blob_hash BLOB, "
            "blob_size INTEGER, source_path TEXT, append_start_offset INTEGER)"
        )
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT)")
    return archive_root, blob_root, hooks_root, capture_spool


def _real_archive(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A real archive: the GC seam only unlinks against its durable schema."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_active_archive_root(archive_root)
    hooks_root = archive_root / "hooks"
    hooks_root.mkdir(exist_ok=True)
    capture_spool = archive_root / "browser-capture"
    capture_spool.mkdir(exist_ok=True)
    return archive_root, archive_root / "blob", hooks_root, capture_spool


def _archive_without_gc_generation_ledger(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A supported source tier before durable GC member intents."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DROP TABLE gc_generation_members")
    return archive_root, blob_root, hooks_root, capture_spool


def _store_aged(blob_root: Path, payload: bytes) -> tuple[str, Path]:
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(payload)
    path = store.blob_path(blob_hash)
    os.utime(path, (_AGED_MTIME, _AGED_MTIME))
    return blob_hash, path


def _reference(archive_root: Path, blob_hash: str) -> None:
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(blob_hash), f"raw:{blob_hash[:8]}", "raw_payload", 0, 0),
        )


def _stub_reference(source_db: Path, *blob_hashes: str) -> None:
    """Name blobs in a durable relation so liveness, not silence, decides."""
    with sqlite3.connect(source_db) as conn:
        conn.executemany(
            "INSERT INTO blob_refs (blob_hash, ref_type) VALUES (?, 'raw_payload')",
            [(bytes.fromhex(blob_hash),) for blob_hash in blob_hashes],
        )


def _plan_and_context(
    archive_root: Path,
    blob_root: Path,
    *,
    legacy_root: Path | None = None,
    capture_spool: Path,
) -> tuple[BlobDispositionPlan, BlobDispositionContext]:
    hook_sources = (("legacy-hook-spool-0", legacy_root),) if legacy_root is not None else ()
    context = build_disposition_context(
        archive_root=archive_root,
        blob_root=blob_root,
        source_db=archive_root / "source.db",
        hook_spool_sources=hook_sources,
        browser_capture_spool=capture_spool,
    )
    plan = compile_disposition_plan(
        archive_root=archive_root,
        blob_root=blob_root,
        source_db=archive_root / "source.db",
        context=context,
    )
    return plan, context


def _apply(
    plan: BlobDispositionPlan,
    context: BlobDispositionContext,
    archive_root: Path,
    hooks_root: Path,
    capture_spool: Path,
    *,
    dry_run: bool,
    authorized_digest: str | None = None,
    writer_block_reason: str | None = None,
) -> DispositionApplyReceipt:
    return apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest() if authorized_digest is None else authorized_digest,
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        writer_block_reason=writer_block_reason,
        dry_run=dry_run,
    )


def test_restoration_publishes_into_the_ordinary_spool_and_keeps_the_carrier(tmp_path: Path) -> None:
    """Anti-vacuity: deleting the carrier during restoration makes this red."""
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    store = BlobStore(blob_root)
    envelope = _hook_envelope("sole-copy")
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.RESTORE_REQUIRED

    (result,) = restore_plan_members(
        plan,
        context=context,
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=False,
    )

    assert result.outcome is RestorationOutcome.RESTORED
    restored = Path(result.spool_path)
    assert restored.is_file()
    assert read_hook_spool_record(restored) == json.loads(store.blob_path(blob_hash).read_bytes())
    assert store.blob_path(blob_hash).is_file()


def test_restoration_is_idempotent_by_logical_identity(tmp_path: Path) -> None:
    """Anti-vacuity: matching only today's day shard double-delivers a retry.

    ``enqueue_hook_event`` refuses a collision inside the current day's shard
    only, so a resident event spooled on any other day must be found by
    identity or the retry writes a second carrier of the same event.
    """
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    store = BlobStore(blob_root)
    envelope = _hook_envelope("sole-copy")
    store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    (first,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )
    # Relocate the restored carrier into another day's shard: the retry must
    # still recognize it rather than publish a second copy.
    relocated = hooks_root / "pending" / "2026-07-15"
    relocated.mkdir(parents=True, exist_ok=True)
    Path(first.spool_path).rename(relocated / "sole-copy.json")

    (second,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert first.outcome is RestorationOutcome.RESTORED
    assert second.outcome is RestorationOutcome.RESTORATION_ALREADY_PRESENT
    assert [path.name for path in hooks_root.rglob("*.json")] == ["sole-copy.json"]


def test_an_acknowledged_receipt_does_not_count_as_a_restored_copy(tmp_path: Path) -> None:
    """Anti-vacuity: an rglob over the whole spool root reports this already present.

    The event would then be recorded as restored while living only in
    ``acknowledged/``, which no drain and no watcher reads -- and the carrier
    becomes deletable on that report.
    """
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    store = BlobStore(blob_root)
    envelope = _hook_envelope("acknowledged-only")
    store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    receipt = hooks_root / "acknowledged" / "2026-07-15"
    receipt.mkdir(parents=True)
    (receipt / "acknowledged-only.json").write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    (result,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert result.outcome is RestorationOutcome.RESTORED
    restored = Path(result.spool_path)
    assert restored.is_relative_to(hooks_root / "pending")
    assert read_hook_spool_record(restored) == read_hook_spool_record(receipt / "acknowledged-only.json")
    assert (receipt / "acknowledged-only.json").is_file()


def test_restoration_blocks_on_a_hostile_collision(tmp_path: Path) -> None:
    """Anti-vacuity: overwriting on identity collision loses the resident event."""
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    store = BlobStore(blob_root)
    store.write_from_bytes(_stored_bytes(_hook_envelope("collide", text="the stored call"), tmp_path))
    resident = hooks_root / "pending" / "2026-07-15"
    resident.mkdir(parents=True)
    (resident / "collide.json").write_text(
        json.dumps(_hook_envelope("collide", text="a different call"), sort_keys=True), encoding="utf-8"
    )
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    (result,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert result.outcome is RestorationOutcome.BLOCKED
    assert "different event" in result.detail
    assert json.loads((resident / "collide.json").read_text())["payload"]["detail"] == "a different call"


def test_material_a_configured_source_already_holds_is_not_published_twice(tmp_path: Path) -> None:
    """Anti-vacuity: publishing unconditionally delivers the event a second time.

    A carrier whose material a configured source proves — a legacy spool, or
    the very spool an earlier pass restored it into — is already resident.
    That is the residency this step exists to establish, so it publishes
    nothing and the member stays eligible.
    """
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    legacy_root.mkdir()
    store = BlobStore(blob_root)
    envelope = _hook_envelope("late-arrival")
    store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, _ = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.RESTORE_REQUIRED

    resident = _write_spool_file(legacy_root, envelope)
    _, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    (result,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert result.outcome is RestorationOutcome.RESTORATION_ALREADY_PRESENT
    assert Path(result.spool_path) == resident
    assert list(hooks_root.rglob("*.json")) == []


def test_restoration_proceeds_while_other_members_are_unresolved(tmp_path: Path) -> None:
    """Anti-vacuity: gating restoration on plan acceptance strands sole copies.

    Restoration never removes a carrier, so an unrelated unexplained object
    must not delay preserving the only copy of wanted material.
    """
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    store = BlobStore(blob_root)
    sole_hash, _ = store.write_from_bytes(_stored_bytes(_hook_envelope("sole-copy"), tmp_path))
    mystery_hash, _ = store.write_from_bytes(b"%PDF-1.5\nunexplained\n")
    _stub_reference(archive_root / "source.db", mystery_hash)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert not plan.accepted

    results = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert [result.outcome for result in results] == [RestorationOutcome.RESTORED]
    assert store.blob_path(sole_hash).is_file()


def test_a_stale_authorized_digest_refuses_before_any_effect(tmp_path: Path) -> None:
    """Anti-vacuity: applying without digest binding consumes an edited plan."""
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False, authorized_digest="0" * 64)

    assert not receipt.ok
    assert any("does not match plan digest" in blocker for blocker in receipt.blockers)
    assert receipt.results == ()
    assert store.blob_path(blob_hash).is_file()


def test_an_active_writer_refuses_an_active_apply(tmp_path: Path) -> None:
    """Anti-vacuity: unserialized apply races the archive's single writer."""
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = _apply(
        plan,
        context,
        archive_root,
        hooks_root,
        capture_spool,
        dry_run=False,
        writer_block_reason="live pidfile PID 4242 is running",
    )

    assert not receipt.ok
    assert any("writer is active" in blocker for blocker in receipt.blockers)
    assert store.blob_path(blob_hash).is_file()


def test_a_blocked_restoration_keeps_its_carrier_in_the_namespace(tmp_path: Path) -> None:
    """Anti-vacuity: unlinking a carrier whose restoration failed loses the copy.

    Restoration runs first precisely so this ordering is observable: the only
    verified copy of the material must be in a spool before the blob may go.
    """
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    blob_hash, blob_path = _store_aged(blob_root, _stored_bytes(_hook_envelope("collide", text="stored"), tmp_path))
    resident = hooks_root / "pending" / "2026-07-15"
    resident.mkdir(parents=True)
    (resident / "collide.json").write_text(
        json.dumps(_hook_envelope("collide", text="a different call"), sort_keys=True), encoding="utf-8"
    )
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.RESTORE_REQUIRED

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert not receipt.ok
    (result,) = receipt.results
    assert result.outcome is MemberOutcome.BLOCKED
    assert "not resident in a spool" in result.detail
    assert blob_path.is_file()
    assert receipt.deleted_count == 0
    assert receipt.namespace_after.blob_count == 1
    assert blob_hash in {restoration.blob_hash for restoration in receipt.restorations}


def test_a_sole_copy_is_restored_before_its_carrier_is_deleted(tmp_path: Path) -> None:
    """Anti-vacuity: deleting first, or without reading the spool file back,
    destroys the only copy. The restored capture is byte-identical to the
    carrier and the carrier is gone; reversing the order makes both red."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    payload = _capture_bytes()
    blob_hash, blob_path = _store_aged(blob_root, payload)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.RESTORE_REQUIRED

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (restoration,) = receipt.restorations
    assert restoration.outcome is RestorationOutcome.RESTORED
    assert Path(restoration.spool_path).read_bytes() == payload
    (result,) = receipt.results
    assert result.outcome is MemberOutcome.DELETED
    assert result.restored_to == restoration.spool_path
    assert not blob_path.exists()
    assert receipt.deleted_count == 1 and receipt.deleted_bytes == len(payload)
    assert receipt.namespace_after.blob_count == 0
    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute(
            "SELECT outcome FROM gc_generation_members WHERE blob_hash = ?", (bytes.fromhex(blob_hash),)
        ).fetchall()
    assert rows == [("removed",)]


def test_an_older_revision_of_a_spooled_session_is_superseded_not_blocked(tmp_path: Path) -> None:
    """Anti-vacuity: requiring the destination to hold an equal capture blocks
    every carrier of a session the extension recaptured — 292 of 300 on the
    live archive. Reinstating that equality check makes this red, and so does
    letting the older revision overwrite the later one."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    resident = write_capture_envelope_bytes(_LATER_CAPTURE, spool_path=capture_spool).path
    _, blob_path = _store_aged(blob_root, _EARLIER_CAPTURE)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.RESTORE_REQUIRED

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (restoration,) = receipt.restorations
    assert restoration.outcome is RestorationOutcome.RESTORATION_SUPERSEDED
    assert Path(restoration.spool_path) == resident
    assert resident.read_bytes() == _LATER_CAPTURE
    (result,) = receipt.results
    assert result.outcome is MemberOutcome.DELETED
    assert not blob_path.exists()


def test_a_newer_revision_replaces_the_spooled_capture(tmp_path: Path) -> None:
    """Anti-vacuity: reporting every collision as superseded would drop the
    revision the spool wants, leaving the stale capture resident."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    resident = write_capture_envelope_bytes(_EARLIER_CAPTURE, spool_path=capture_spool).path
    _store_aged(blob_root, _LATER_CAPTURE)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (restoration,) = receipt.restorations
    assert restoration.outcome is RestorationOutcome.RESTORED
    assert resident.read_bytes() == _LATER_CAPTURE
    assert len(list(capture_spool.rglob("*.json"))) == 1


def test_two_carriers_of_one_session_converge_on_one_artifact_holding_the_newer(tmp_path: Path) -> None:
    """Anti-vacuity: blocking on the collision strands both carriers, and
    publishing them independently leaves two artifacts for one session."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    _, earlier_path = _store_aged(blob_root, _EARLIER_CAPTURE)
    _, later_path = _store_aged(blob_root, _LATER_CAPTURE)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    assert receipt.restoration_counts[RestorationOutcome.BLOCKED.value] == 0
    (artifact,) = list(capture_spool.rglob("*.json"))
    assert artifact.read_bytes() == _LATER_CAPTURE
    assert [result.outcome for result in receipt.results] == [MemberOutcome.DELETED, MemberOutcome.DELETED]
    assert not earlier_path.exists() and not later_path.exists()


def test_a_rehearsal_predicts_the_restoration_counts_its_apply_produces(tmp_path: Path) -> None:
    """Anti-vacuity: a dry arm that returns before resolving the destination
    reports every carrier restorable — 880 restorable and 0 blocked in front
    of an apply that restored 580 and blocked 300. Returning early, or
    dropping the rehearsal's memory of what it already published, makes the
    two count sets differ."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    stale_carrier, spooled_revision = _revision_pair("conv-stale")
    write_capture_envelope_bytes(_capture_bytes("conv-resident"), spool_path=capture_spool)
    write_capture_envelope_bytes(spooled_revision, spool_path=capture_spool)
    spooled = {path: path.read_bytes() for path in capture_spool.rglob("*.json")}
    for payload in (
        _EARLIER_CAPTURE,
        _LATER_CAPTURE,
        _capture_bytes("conv-resident"),
        stale_carrier,
        _capture_bytes("conv-fresh"),
    ):
        _store_aged(blob_root, payload)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    rehearsal = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=True)

    assert {path: path.read_bytes() for path in capture_spool.rglob("*.json")} == spooled

    active = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert rehearsal.ok and active.ok, (rehearsal.blockers, active.blockers)
    assert rehearsal.restoration_counts == active.restoration_counts
    assert rehearsal.counts == active.counts
    assert [restoration.outcome for restoration in rehearsal.restorations] == [
        restoration.outcome for restoration in active.restorations
    ]
    assert active.restoration_counts[RestorationOutcome.RESTORATION_SUPERSEDED.value] == 1
    assert active.restoration_counts[RestorationOutcome.BLOCKED.value] == 0
    assert len(list(capture_spool.rglob("*.json"))) == 4


def test_a_capture_carrier_of_a_different_session_never_overwrites_the_artifact_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: converging on artifact name rather than session identity
    would let one session's capture replace another's."""
    import polylogue.browser_capture.receiver as receiver

    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    collision = capture_spool / "chatgpt" / "shared-name.json"
    monkeypatch.setattr(receiver, "capture_artifact_path", lambda envelope, spool_path=None: collision)
    write_capture_envelope_bytes(_capture_bytes("conv-resident"), spool_path=capture_spool)
    _, blob_path = _store_aged(blob_root, _capture_bytes("conv-other"))
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    (restoration,) = receipt.restorations
    assert restoration.outcome is RestorationOutcome.BLOCKED
    assert "name collision" in restoration.detail
    assert collision.read_bytes() == _capture_bytes("conv-resident")
    assert blob_path.is_file()


def test_a_referenced_member_is_never_deleted(tmp_path: Path) -> None:
    """Anti-vacuity: deleting on disposition rather than on referencedness
    unlinks an object a durable row still names, and the row's payload with
    it. Its content being proven at a source changes nothing."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    blob_hash, blob_path = _store_aged(blob_root, _stored_bytes(envelope, tmp_path))
    _reference(archive_root, blob_hash)
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.SOURCE_PRESENT and plan.members[0].referenced

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (result,) = receipt.results
    assert result.outcome is MemberOutcome.RETAINED_REFERENCED
    assert result.referenced
    assert blob_path.is_file()
    assert receipt.deleted_count == 0
    assert receipt.cohorts[BlobDisposition.SOURCE_PRESENT.value]["retained_bytes"] == result.size_bytes


def test_an_unreferenced_orphan_is_deleted_when_live_eligibility_holds(tmp_path: Path) -> None:
    """An unnamed object is GC-eligible even when no source prover explains it.

    The plan records the positive ``unreferenced`` disposition, while apply
    rechecks the live relation before deletion. A newly added reference must
    therefore retain the object instead of trusting the stale plan.
    """
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    orphan_hash, orphan_path = _store_aged(blob_root, b"%PDF-1.5\nunexplained\n")
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.UNREFERENCED
    assert not plan.members[0].referenced
    assert plan.accepted

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (result,) = receipt.results
    assert result.outcome is MemberOutcome.DELETED
    assert result.cohort == BlobDisposition.UNREFERENCED.value
    assert not result.referenced
    assert not orphan_path.exists()
    assert receipt.deleted_count == 1
    payload = receipt.to_dict()
    assert orphan_hash in {member["blob_hash"] for member in cast(list[dict[str, Any]], payload["results"])}
    assert cast(list[str], payload["reference_relations"])[0] == "blob_refs.blob_hash"


def test_a_referenced_unresolved_member_stays_while_the_rest_is_applied(tmp_path: Path) -> None:
    """Anti-vacuity: restoring the zero-unresolved gate applies nothing at all,
    and deleting the referenced mystery destroys the residue the maneuver
    exists to expose."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    proven_hash, proven_path = _store_aged(blob_root, _stored_bytes(envelope, tmp_path))
    mystery_hash, mystery_path = _store_aged(blob_root, b"%PDF-1.5\nunexplained\n")
    _reference(archive_root, mystery_hash)
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    assert plan.unresolved_count == 1 and not plan.accepted

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    outcomes = {result.blob_hash: result.outcome for result in receipt.results}
    assert outcomes[proven_hash] is MemberOutcome.DELETED
    assert outcomes[mystery_hash] is MemberOutcome.RETAINED_REFERENCED
    assert not proven_path.exists()
    assert mystery_path.is_file()
    unresolved = receipt.cohorts[BlobDisposition.UNRESOLVED.value]
    assert unresolved["members"] == 1
    assert unresolved["retained_referenced"] == 1
    assert unresolved["retained_bytes"] == mystery_path.stat().st_size
    assert receipt.namespace_after.blob_count == 1


def test_invalid_namespace_entries_are_removed(tmp_path: Path) -> None:
    """Anti-vacuity: leaving SQLite siblings behind keeps the namespace unclean.

    A ``-wal`` beside a content-addressed object is a byproduct of something
    having opened that object as a database; the object's own bytes are its
    identity, so the sidecar has no owner and no plan member.
    """
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    shard = blob_root / "ab"
    shard.mkdir()
    stray = shard / "index.db-wal"
    stray.write_bytes(b"stale write-ahead log\n")
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    (invalid_entry,) = plan.denominator.invalid_namespace_entries
    assert invalid_entry.relative_path == "ab/index.db-wal"
    assert invalid_entry.issue == "invalid_leaf_name"
    assert not invalid_entry.explained

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (result,) = receipt.results
    assert result.cohort == INVALID_ENTRY_COHORT
    assert result.outcome is MemberOutcome.DELETED
    assert result.from_path == str(stray)
    assert not stray.exists()
    assert receipt.invalid_entries_deleted == 1
    assert receipt.invalid_entry_bytes_deleted == len(b"stale write-ahead log\n")
    assert receipt.namespace_after.invalid_entry_count == 0


def test_a_namespace_entry_that_is_not_a_regular_file_is_refused(tmp_path: Path) -> None:
    """Anti-vacuity: unlinking whatever a record names follows a symlink out of
    the namespace and removes something the archive does not own."""
    archive_root, blob_root, hooks_root, capture_spool = _stub_archive(tmp_path)
    outside = tmp_path / "not-ours.db"
    outside.write_bytes(b"someone else's database\n")
    shard = blob_root / "ab"
    shard.mkdir()
    (shard / "index.db-wal").symlink_to(outside)
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert not receipt.ok
    (result,) = receipt.results
    assert result.outcome is MemberOutcome.BLOCKED
    assert result.detail == "namespace entry is not a regular file"
    assert outside.is_file()


def test_a_dry_rehearsal_is_inert_and_reports_the_active_totals(tmp_path: Path) -> None:
    """Anti-vacuity: a rehearsal that wrote, that spooled, or that reported
    different totals from the run it rehearses would make the review
    meaningless."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    proven = _hook_envelope("proven")
    _write_spool_file(legacy_root, proven)
    proven_hash, proven_path = _store_aged(blob_root, _stored_bytes(proven, tmp_path))
    sole_hash, sole_path = _store_aged(blob_root, _stored_bytes(_hook_envelope("sole-copy"), tmp_path))
    stray = blob_root / "ab"
    stray.mkdir(exist_ok=True)
    (stray / "index.db-wal").write_bytes(b"stale write-ahead log\n")
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    rehearsal = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=True)

    assert rehearsal.ok and rehearsal.dry_run
    assert proven_path.is_file() and sole_path.is_file()
    assert (stray / "index.db-wal").is_file()
    assert list(hooks_root.rglob("*.json")) == []

    active = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert active.ok, active.blockers
    assert active.to_dict()["totals"] == rehearsal.to_dict()["totals"]
    assert active.counts == rehearsal.counts
    assert active.cohorts == rehearsal.cohorts
    assert not proven_path.exists() and not sole_path.exists()
    assert {proven_hash, sole_hash} == {
        result.blob_hash for result in active.results if result.is_blob and result.outcome is MemberOutcome.DELETED
    }


def test_a_second_pass_reports_what_a_previous_pass_already_deleted(tmp_path: Path) -> None:
    """Anti-vacuity: reading an absent object as drift would refuse a resume.

    A pass over 65,873 members that cannot be resumed after an interruption is
    a pass that has to start over.
    """
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    _store_aged(blob_root, _stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    first = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)
    second = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert first.ok and second.ok, (first.blockers, second.blockers)
    assert [result.outcome for result in second.results] == [MemberOutcome.RETAINED_ABSENT]
    assert second.deleted_count == 0
    assert second.namespace_after.blob_count == 0


def test_receipt_totals_and_cohorts_derive_from_member_outcomes(tmp_path: Path) -> None:
    """Anti-vacuity: a summary counter maintained beside the members can drift."""
    archive_root, blob_root, hooks_root, capture_spool = _real_archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    _write_spool_file(legacy_root, _hook_envelope("proven"))
    _store_aged(blob_root, _stored_bytes(_hook_envelope("proven"), tmp_path))
    _store_aged(blob_root, _stored_bytes(_hook_envelope("sole-copy"), tmp_path))
    mystery_hash, mystery_path = _store_aged(blob_root, b"%PDF-1.5\nunexplained\n")
    _reference(archive_root, mystery_hash)
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=True)

    assert sum(receipt.counts.values()) == len(receipt.results) == len(plan.members)
    cohorts = receipt.cohorts
    assert sum(cohort["members"] for cohort in cohorts.values()) == len(plan.members)
    assert cohorts[BlobDisposition.SOURCE_PRESENT.value]["deleted"] == 1
    assert cohorts[BlobDisposition.RESTORE_REQUIRED.value]["deleted"] == 1
    assert cohorts[BlobDisposition.UNRESOLVED.value]["retained_referenced"] == 1
    totals = cast(dict[str, Any], receipt.to_dict()["totals"])
    assert totals["before"] == {
        "blob_count": 3,
        "blob_bytes": receipt.namespace_before.blob_bytes,
        "invalid_entry_count": 0,
    }
    assert totals["after"]["blob_count"] == 1
    assert totals["after"]["blob_bytes"] == mystery_path.stat().st_size
    assert totals["deleted_bytes"] == receipt.namespace_before.blob_bytes - totals["after"]["blob_bytes"]
    destination = tmp_path / "receipts" / "disposition.json"
    write_receipt(destination, receipt)
    published = json.loads(destination.read_text())
    assert published["counts"] == receipt.counts
    assert published["restorations"]["counts"] == receipt.restoration_counts


def test_a_source_tier_without_the_gc_ledger_unlinks_unreferenced_candidates_directly(tmp_path: Path) -> None:
    """Anti-vacuity: deleting the direct-unlink fallback leaves the proven
    object on disk with a blocked receipt; deleting the reference recheck
    would unlink the referenced object too."""
    archive_root, blob_root, hooks_root, capture_spool = _archive_without_gc_generation_ledger(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    blob_hash, blob_path = _store_aged(blob_root, _stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert receipt.ok, receipt.blockers
    (result,) = receipt.results
    assert result.blob_hash == blob_hash
    assert result.outcome is MemberOutcome.DELETED
    assert result.detail == DIRECT_UNLINK_DETAIL
    assert not blob_path.exists()
    assert receipt.deleted_count == 1
    assert receipt.namespace_after.blob_count == 0


def test_legacy_fallback_keeps_an_index_only_attachment_in_dry_and_active_runs(tmp_path: Path) -> None:
    """Anti-vacuity: source-only rechecks delete this attachment in both runs."""
    archive_root, blob_root, hooks_root, capture_spool = _archive_without_gc_generation_ledger(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    blob_hash, blob_path = _store_aged(blob_root, _stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute(
            "INSERT INTO attachments(attachment_id, blob_hash) VALUES (?, ?)", ("index-only", bytes.fromhex(blob_hash))
        )

    rehearsal = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=True)
    active = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert rehearsal.ok and active.ok
    assert rehearsal.results[0].outcome is active.results[0].outcome is MemberOutcome.RETAINED_REFERENCED
    assert blob_path.is_file()
    assert rehearsal.deleted_count == active.deleted_count == 0


def test_legacy_fallback_refuses_when_index_liveness_evidence_is_unavailable(tmp_path: Path) -> None:
    """Anti-vacuity: treating an unavailable index as empty unlinks this blob."""
    archive_root, blob_root, hooks_root, capture_spool = _archive_without_gc_generation_ledger(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    blob_hash, blob_path = _store_aged(blob_root, _stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("DROP TABLE attachments")

    rehearsal = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=True)
    active = _apply(plan, context, archive_root, hooks_root, capture_spool, dry_run=False)

    assert not rehearsal.ok and not active.ok
    assert rehearsal.results[0].outcome is active.results[0].outcome is MemberOutcome.BLOCKED
    assert any("index.attachments is missing" in blocker for blocker in active.blockers)
    assert blob_path.is_file()
