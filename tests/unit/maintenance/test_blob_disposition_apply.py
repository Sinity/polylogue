"""Fault matrix for consuming an accepted blob disposition plan.

The apply boundary is irreversible, so every test here names the mutation
that would make it red: deleting before restoring, trusting a stale plan,
accepting a changed source or denominator, or converting a blocked member
into a silent success.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.maintenance.blob_disposition import (
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionPlan,
    build_disposition_context,
    compile_disposition_plan,
)
from polylogue.maintenance.blob_disposition_apply import (
    MemberOutcome,
    apply_disposition_plan,
    restore_plan_members,
    write_receipt,
)
from polylogue.sources.hooks import read_hook_spool_record
from polylogue.storage.blob_store import BlobStore


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


def _write_spool_file(root: Path, envelope: dict[str, object]) -> Path:
    target = root / "pending" / "2026-07-15"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{envelope['event_id']}.json"
    path.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=4), encoding="utf-8")
    return path


def _archive(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    archive_root = tmp_path / "archive"
    blob_root = archive_root / "blob"
    blob_root.mkdir(parents=True)
    hooks_root = archive_root / "hooks"
    hooks_root.mkdir()
    capture_spool = archive_root / "browser-capture"
    capture_spool.mkdir()
    source_db = archive_root / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE blob_refs (blob_hash BLOB, ref_type TEXT)")
        conn.execute(
            "CREATE TABLE raw_sessions (raw_id TEXT, origin TEXT, native_id TEXT, blob_hash BLOB, "
            "blob_size INTEGER, source_path TEXT, append_start_offset INTEGER)"
        )
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT)")
    return archive_root, blob_root, hooks_root, capture_spool


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


def test_restoration_publishes_into_the_ordinary_spool_and_keeps_the_carrier(tmp_path: Path) -> None:
    """Anti-vacuity: deleting the carrier during restoration makes this red."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
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

    assert result.outcome is MemberOutcome.RESTORED
    restored = Path(result.detail)
    assert restored.is_file()
    assert read_hook_spool_record(restored) == json.loads(store.blob_path(blob_hash).read_bytes())
    assert store.blob_path(blob_hash).is_file()


def test_restoration_is_idempotent_by_logical_identity(tmp_path: Path) -> None:
    """Anti-vacuity: matching only today's day shard double-delivers a retry.

    ``enqueue_hook_event`` refuses a collision inside the current day's shard
    only, so a resident event spooled on any other day must be found by
    identity or the retry writes a second carrier of the same event.
    """
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
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
    Path(first.detail).rename(relocated / "sole-copy.json")

    (second,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert first.outcome is MemberOutcome.RESTORED
    assert second.outcome is MemberOutcome.RESTORATION_ALREADY_PRESENT
    assert [path.name for path in hooks_root.rglob("*.json")] == ["sole-copy.json"]


def test_restoration_blocks_on_a_hostile_collision(tmp_path: Path) -> None:
    """Anti-vacuity: overwriting on identity collision loses the resident event."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
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

    assert result.outcome is MemberOutcome.BLOCKED
    assert "different event" in result.detail
    assert json.loads((resident / "collide.json").read_text())["payload"]["detail"] == "a different call"


def test_a_source_proof_appearing_after_planning_blocks_restoration(tmp_path: Path) -> None:
    """Anti-vacuity: skipping revalidation restores material already at its source."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    legacy_root.mkdir()
    store = BlobStore(blob_root)
    envelope = _hook_envelope("late-arrival")
    store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, _ = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    assert plan.members[0].disposition is BlobDisposition.RESTORE_REQUIRED

    _write_spool_file(legacy_root, envelope)
    _, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    (result,) = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert result.outcome is MemberOutcome.BLOCKED
    assert "no longer justified" in result.detail


def test_a_stale_authorized_digest_refuses_before_any_effect(tmp_path: Path) -> None:
    """Anti-vacuity: applying without digest binding consumes an edited plan."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest="0" * 64,
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=False,
    )

    assert not receipt.ok
    assert any("does not match plan digest" in blocker for blocker in receipt.blockers)
    assert store.blob_path(blob_hash).is_file()


def test_an_unresolved_member_refuses_the_whole_plan(tmp_path: Path) -> None:
    """Anti-vacuity: applying a partially explained plan deletes beside a mystery."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    proven_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    mystery_hash, _ = store.write_from_bytes(b"%PDF-1.5\nunexplained\n")
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    assert plan.unresolved_count == 1

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=False,
    )

    assert not receipt.ok
    assert any("not acceptable" in blocker for blocker in receipt.blockers)
    assert store.blob_path(proven_hash).is_file()
    assert store.blob_path(mystery_hash).is_file()


def test_an_active_writer_refuses_an_active_apply(tmp_path: Path) -> None:
    """Anti-vacuity: unserialized apply races the archive's single writer."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        writer_block_reason="live pidfile PID 4242 is running",
        dry_run=False,
    )

    assert not receipt.ok
    assert any("writer is active" in blocker for blocker in receipt.blockers)
    assert store.blob_path(blob_hash).is_file()


def test_a_changed_source_invalidates_the_member_before_deletion(tmp_path: Path) -> None:
    """Anti-vacuity: trusting the planning-time proof deletes divergent material."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    spool_file = _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    plan, _ = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)
    assert plan.accepted

    spool_file.write_text(
        json.dumps(_hook_envelope("proven", text="rewritten at the source"), sort_keys=True), encoding="utf-8"
    )
    _, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=False,
    )

    assert not receipt.ok
    assert store.blob_path(blob_hash).is_file()
    assert any(result.outcome is MemberOutcome.BLOCKED for result in receipt.results)


def test_a_referenced_object_is_retained_not_deleted(tmp_path: Path) -> None:
    """Anti-vacuity: deleting a proven-but-referenced object breaks a live row."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    envelope = _hook_envelope("proven")
    _write_spool_file(legacy_root, envelope)
    store = BlobStore(blob_root)
    blob_hash, _ = store.write_from_bytes(_stored_bytes(envelope, tmp_path))
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("INSERT INTO blob_refs (blob_hash, ref_type) VALUES (?, ?)", (bytes.fromhex(blob_hash), "raw"))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=False,
    )

    assert receipt.ok
    assert [result.outcome for result in receipt.results] == [MemberOutcome.RETAINED_REFERENCED]
    assert store.blob_path(blob_hash).is_file()


def test_a_dry_rehearsal_touches_nothing(tmp_path: Path) -> None:
    """Anti-vacuity: a rehearsal that wrote would make the review meaningless."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    proven = _hook_envelope("proven")
    _write_spool_file(legacy_root, proven)
    store = BlobStore(blob_root)
    proven_hash, _ = store.write_from_bytes(_stored_bytes(proven, tmp_path))
    sole_hash, _ = store.write_from_bytes(_stored_bytes(_hook_envelope("sole-copy"), tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=True,
    )

    assert receipt.ok and receipt.dry_run
    assert store.blob_path(proven_hash).is_file()
    assert store.blob_path(sole_hash).is_file()
    assert list(hooks_root.rglob("*.json")) == []


def test_receipt_totals_derive_from_member_outcomes(tmp_path: Path) -> None:
    """Anti-vacuity: a summary counter maintained beside the members can drift."""
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    legacy_root = tmp_path / "legacy-hooks"
    _write_spool_file(legacy_root, _hook_envelope("proven"))
    store = BlobStore(blob_root)
    store.write_from_bytes(_stored_bytes(_hook_envelope("proven"), tmp_path))
    store.write_from_bytes(_stored_bytes(_hook_envelope("sole-copy"), tmp_path))
    plan, context = _plan_and_context(archive_root, blob_root, legacy_root=legacy_root, capture_spool=capture_spool)

    receipt = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=True,
    )

    assert sum(receipt.counts.values()) == len(receipt.results) == len(plan.members)
    destination = tmp_path / "receipts" / "disposition.json"
    write_receipt(destination, receipt)
    assert json.loads(destination.read_text())["counts"] == receipt.counts


def test_restoration_proceeds_while_other_members_are_unresolved(tmp_path: Path) -> None:
    """Anti-vacuity: gating restoration on plan acceptance strands sole copies.

    Restoration never removes a carrier, so an unrelated unexplained object
    must not delay preserving the only copy of wanted material.
    """
    archive_root, blob_root, hooks_root, capture_spool = _archive(tmp_path)
    store = BlobStore(blob_root)
    sole_hash, _ = store.write_from_bytes(_stored_bytes(_hook_envelope("sole-copy"), tmp_path))
    store.write_from_bytes(b"%PDF-1.5\nunexplained\n")
    plan, context = _plan_and_context(archive_root, blob_root, capture_spool=capture_spool)
    assert not plan.accepted

    results = restore_plan_members(
        plan, context=context, hook_spool_root=hooks_root, browser_capture_spool=capture_spool, dry_run=False
    )

    assert [result.outcome for result in results] == [MemberOutcome.RESTORED]
    assert store.blob_path(sole_hash).is_file()

    refused = apply_disposition_plan(
        plan,
        context=context,
        authorized_digest=plan.digest(),
        source_db=archive_root / "source.db",
        index_db=archive_root / "index.db",
        hook_spool_root=hooks_root,
        browser_capture_spool=capture_spool,
        dry_run=False,
    )
    assert not refused.ok
