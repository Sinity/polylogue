"""Sealed maintenance parts use the existing session-profile owner.

The maintenance surface receives these neutral, uncapped receipts, then owns
its own protocol/audit translation. These tests keep that surface from
rediscovering archive work or borrowing a writer outside the normal owner.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.daemon.convergence import (
    DaemonConverger,
    SelectedSessionTarget,
    SessionProfileConvergenceOwner,
    make_session_profile_derivation,
    make_session_profile_frame,
)
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.storage.derived.session import derivation as session_derivation
from tests.infra.convergence_harness import (
    build_converged_archive,
    rich_convergence_sources,
    seed_partial_convergence_archive,
    session_materialization_facts,
)


async def _owner_for(
    index_db: Path, archive_root: Path
) -> tuple[
    SessionProfileConvergenceOwner,
    BoundedComputeAdapter,
    DaemonWriteCoordinator,
]:
    adapter = make_session_profile_derivation(index_db, archive_root=archive_root, now=lambda: 0.0)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    return (
        SessionProfileConvergenceOwner(
            DaemonConverger((), derivations=[adapter]),
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        ),
        compute,
        coordinator,
    )


async def _shutdown(compute: BoundedComputeAdapter, coordinator: DaemonWriteCoordinator) -> None:
    compute.shutdown(wait=True)
    await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_selected_required_part_publishes_once_and_returns_full_uncapped_receipt(tmp_path: Path) -> None:
    """A selected required target cannot widen into a no-hint archive sweep.

    Anti-vacuity: route this through ``owner.converge`` or return the generic
    capped report and the cold sibling is discovered or receipt fields needed
    by maintenance are unavailable.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    owner, compute, coordinator = await _owner_for(recovered.index_db, recovered.root)
    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    target = SelectedSessionTarget(recovered.target_session_id, "required")
    try:
        first = await owner.converge_selected(
            frame,
            targets=(target,),
            expected_generation=frame.source_revision,
            expected_recipe=frame.recipe_version("session_profile"),
            stop_requested=lambda: None,
        )
        assert len(first) == 1
        receipt = first[0]
        assert receipt.session_id == recovered.target_session_id
        assert receipt.state == "published"
        assert receipt.input_binding is not None
        assert receipt.output_binding == receipt.input_binding
        assert receipt.certified_counts.profiles == 1
        assert receipt.publication_known_committed is True
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile
            is not None
        )
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id).profile is None
        )

        repeat = await owner.converge_selected(
            frame,
            targets=(target,),
            expected_generation=frame.source_revision,
            expected_recipe=frame.recipe_version("session_profile"),
            stop_requested=lambda: None,
        )
        assert len(repeat) == 1
        assert repeat[0].state == "already_satisfied"
        assert repeat[0].output_binding == receipt.output_binding
        assert repeat[0].certified_counts == receipt.certified_counts
        assert repeat[0].publication_known_committed is False
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_required_part_certifies_nonzero_work_event_and_phase_counts(tmp_path: Path) -> None:
    """A receipt reports actual sibling rows from the real profile family.

    Anti-vacuity: infer the receipt from a profile-only row or report fixed
    zero sibling counts and accepted maintenance cannot certify its full
    selected partition.
    """
    archive = build_converged_archive(tmp_path / "archive", rich_convergence_sources(), session_order=(0,))
    target_id = archive.session_ids[0]
    before = session_materialization_facts(archive.root / "index.db", session_id=target_id)
    assert before.work_events and before.phases
    with sqlite3.connect(archive.root / "index.db") as conn:
        for table in ("session_work_events", "session_phases", "session_latency_profiles", "session_profiles"):
            conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (target_id,))
        conn.commit()

    owner, compute, coordinator = await _owner_for(archive.root / "index.db", archive.root)
    frame = make_session_profile_frame(archive.root / "index.db", archive_root=archive.root, scope=(target_id,))
    try:
        receipt = (
            await owner.converge_selected(
                frame,
                targets=(SelectedSessionTarget(target_id, "required"),),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert receipt.state == "published"
        assert receipt.certified_counts.profiles == 1
        assert receipt.certified_counts.work_events == len(before.work_events) > 0
        assert receipt.certified_counts.phases == len(before.phases) > 0
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_excess_part_retires_only_the_sealed_orphan(tmp_path: Path) -> None:
    """An explicit excess target is certified absent without source discovery.

    Anti-vacuity: use required/excess paging and the owner may enumerate
    unrelated archive rows; rebuild an excess key and its profile persists.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    owner, compute, coordinator = await _owner_for(recovered.index_db, recovered.root)
    required_frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    required = SelectedSessionTarget(recovered.target_session_id, "required")
    try:
        assert (
            await owner.converge_selected(
                required_frame,
                targets=(required,),
                expected_generation=required_frame.source_revision,
                expected_recipe=required_frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0].state == "published"
        with sqlite3.connect(recovered.index_db) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (recovered.target_session_id,))
            conn.commit()

        excess_frame = make_session_profile_frame(
            recovered.index_db,
            archive_root=recovered.root,
            scope=(recovered.target_session_id,),
        )
        receipt = (
            await owner.converge_selected(
                excess_frame,
                targets=(SelectedSessionTarget(recovered.target_session_id, "excess"),),
                expected_generation=excess_frame.source_revision,
                expected_recipe=excess_frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert receipt.state == "published"
        assert receipt.input_binding is None
        assert receipt.output_binding is None
        assert receipt.certified_counts.profiles == 0
        assert receipt.certified_counts.work_events == 0
        assert receipt.certified_counts.phases == 0
        assert receipt.publication_known_committed is True
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id).profile is None
        )
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_part_rejects_stale_contract_or_stop_without_writer_work(tmp_path: Path) -> None:
    """A stopped or stale part yields no unchecked publication claim.

    Anti-vacuity: compare only target ids or check stop after bridge admission
    and this either publishes under the wrong recipe/generation or creates an
    outcome for a target the caller withdrew before computation.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    owner, compute, coordinator = await _owner_for(recovered.index_db, recovered.root)
    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    target = SelectedSessionTarget(recovered.target_session_id, "required")
    try:
        stale = await owner.converge_selected(
            frame,
            targets=(target,),
            expected_generation="index-generation:retired",
            expected_recipe=frame.recipe_version("session_profile"),
            stop_requested=lambda: None,
        )
        assert stale[0].state == "stale"
        assert stale[0].publication_known_committed is False
        assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile is None

        stale_recipe = await owner.converge_selected(
            frame,
            targets=(target,),
            expected_generation=frame.source_revision,
            expected_recipe="session-profile",
            stop_requested=lambda: None,
        )
        assert stale_recipe[0].state == "stale"
        assert stale_recipe[0].publication_known_committed is False
        assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile is None

        unsealed_frame = make_session_profile_frame(recovered.index_db, archive_root=recovered.root, scope=())
        with pytest.raises(ValueError, match="exactly match"):
            await owner.converge_selected(
                unsealed_frame,
                targets=(target,),
                expected_generation=unsealed_frame.source_revision,
                expected_recipe=unsealed_frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile is None

        stopped = await owner.converge_selected(
            frame,
            targets=(target,),
            expected_generation=frame.source_revision,
            expected_recipe=frame.recipe_version("session_profile"),
            stop_requested=lambda: "operator-stop",
        )
        assert stopped == ()
        assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile is None
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_part_retries_only_the_same_binding_moved_target(tmp_path: Path) -> None:
    """A selected binding race gets one bounded retry without widening scope.

    Anti-vacuity: reuse the generic paging kernel or rediscover after a moved
    binding and this path can publish/retry an unsealed sibling instead of
    returning a bounded pending receipt for the accepted target.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    adapter = make_session_profile_derivation(recovered.index_db, archive_root=recovered.root, now=lambda: 0.0)
    original_compute = adapter.compute
    compute_attempts = 0

    def move_target_binding(frame: object, session_id: str) -> object:
        nonlocal compute_attempts
        replacement = original_compute(frame, session_id)
        compute_attempts += 1
        with sqlite3.connect(recovered.index_db) as conn:
            conn.execute(
                "UPDATE messages SET model_name = ? WHERE session_id = ?",
                (f"binding-moved-{compute_attempts}", session_id),
            )
            conn.commit()
        return replacement

    adapter.compute = move_target_binding  # type: ignore[method-assign]
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        DaemonConverger((), derivations=[adapter]),
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    try:
        receipt = (
            await owner.converge_selected(
                frame,
                targets=(SelectedSessionTarget(recovered.target_session_id, "required"),),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert compute_attempts == 2
        assert receipt.state == "pending"
        assert receipt.reason == "binding_moved"
        assert receipt.publication_known_committed is False
        assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile is None
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id).profile is None
        )
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_part_retains_a_committed_effect_when_post_certification_is_lost(tmp_path: Path) -> None:
    """A committed bridge result remains visible when later certification fails.

    Anti-vacuity: clear the committed flag when the post-write read fails and
    maintenance/audit cannot distinguish an unobserved non-write from a real
    index mutation whose current validity is unknown.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    adapter = make_session_profile_derivation(recovered.index_db, archive_root=recovered.root, now=lambda: 0.0)
    original_facts = adapter.selected_part_facts
    observations = 0

    def facts_missing_after_publish(frame: object, session_id: str) -> object:
        nonlocal observations
        observations += 1
        if observations == 3:
            raise RuntimeError("post-publication read unavailable")
        return original_facts(frame, session_id)

    adapter.selected_part_facts = facts_missing_after_publish  # type: ignore[method-assign]
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        DaemonConverger((), derivations=[adapter]),
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    try:
        receipt = (
            await owner.converge_selected(
                frame,
                targets=(SelectedSessionTarget(recovered.target_session_id, "required"),),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert receipt.state == "unknown"
        assert receipt.publication_known_committed is True
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile
            is not None
        )
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_marker_failure_preserves_committed_index_effect_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-tier marker failure cannot erase a committed index-family fact.

    Anti-vacuity: treat a post-index marker exception like a failed index
    publication and maintenance records no effect/counts even though the
    derived partition is already committed and marker recovery remains due.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    with sqlite3.connect(recovered.index_db) as conn:
        conn.execute(
            "UPDATE blocks SET text = ? WHERE message_id = (SELECT message_id FROM messages WHERE session_id = ?)",
            ("::finding: marker-lowering must retry", recovered.target_session_id),
        )
        conn.commit()
    adapter = make_session_profile_derivation(recovered.index_db, archive_root=recovered.root, now=lambda: 0.0)
    original_lower = session_derivation._lower_prepared_markers

    def fail_after_index_commit(marker_write_connection: object, prepared: object) -> None:
        del marker_write_connection, prepared
        raise RuntimeError("synthetic marker-tier failure")

    monkeypatch.setattr(session_derivation, "_lower_prepared_markers", fail_after_index_commit)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        DaemonConverger((), derivations=[adapter]),
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    target = SelectedSessionTarget(recovered.target_session_id, "required")
    try:
        failed = (
            await owner.converge_selected(
                frame,
                targets=(target,),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert failed.state == "failed"
        assert failed.publication_known_committed is True
        assert failed.certified_counts.profiles == 1
        with sqlite3.connect(recovered.index_db) as conn:
            expected_counts = tuple(
                int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (recovered.target_session_id,)
                    ).fetchone()[0]
                )
                for table in ("session_profiles", "session_work_events", "session_phases")
            )
        assert failed.certified_counts == type(failed.certified_counts)(*expected_counts)
        assert adapter.inspect(frame, (recovered.target_session_id,))[recovered.target_session_id] == "stale"

        monkeypatch.setattr(session_derivation, "_lower_prepared_markers", original_lower)
        recovered_marker = (
            await owner.converge_selected(
                frame,
                targets=(target,),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert recovered_marker.state == "published"
        assert recovered_marker.publication_known_committed is True
        assert adapter.inspect(frame, (recovered.target_session_id,))[recovered.target_session_id] == "valid"

        # A later marker-only retry has a valid, pre-existing index family.
        # Its failed user-tier attempt must retain exact counts but cannot
        # claim that this invocation committed another index replacement.
        with sqlite3.connect(recovered.root / "user.db") as conn:
            conn.execute("DELETE FROM assertions")
            conn.commit()
        monkeypatch.setattr(session_derivation, "_lower_prepared_markers", fail_after_index_commit)
        marker_only_failure = (
            await owner.converge_selected(
                frame,
                targets=(target,),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert marker_only_failure.state == "failed"
        assert marker_only_failure.publication_known_committed is False
        assert marker_only_failure.certified_counts == failed.certified_counts
        assert adapter.inspect(frame, (recovered.target_session_id,))[recovered.target_session_id] == "stale"

        monkeypatch.setattr(session_derivation, "_lower_prepared_markers", original_lower)
        final_marker_recovery = (
            await owner.converge_selected(
                frame,
                targets=(target,),
                expected_generation=frame.source_revision,
                expected_recipe=frame.recipe_version("session_profile"),
                stop_requested=lambda: None,
            )
        )[0]
        assert final_marker_recovery.state == "published"
        assert final_marker_recovery.publication_known_committed is True
        assert adapter.inspect(frame, (recovered.target_session_id,))[recovered.target_session_id] == "valid"
    finally:
        await _shutdown(compute, coordinator)


@pytest.mark.asyncio
async def test_selected_part_cancellation_waits_for_the_bridged_publication(tmp_path: Path) -> None:
    """Cancelling maintenance cannot detach its admitted session writer.

    Anti-vacuity: return from selected-part cancellation before the bridge
    settles and shutdown can release archive exclusion while this publication
    still mutates its exact target.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    adapter = make_session_profile_derivation(recovered.index_db, archive_root=recovered.root, now=lambda: 0.0)
    started = threading.Event()
    release = threading.Event()
    original_publish = adapter.publish

    def blocking_publish(frame: object, replacement: object) -> bool:
        started.set()
        assert release.wait(timeout=2.0)
        return original_publish(frame, replacement)

    adapter.publish = blocking_publish  # type: ignore[method-assign]
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        DaemonConverger((), derivations=[adapter]),
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    task = asyncio.create_task(
        owner.converge_selected(
            frame,
            targets=(SelectedSessionTarget(recovered.target_session_id, "required"),),
            expected_generation=frame.source_revision,
            expected_recipe=frame.recipe_version("session_profile"),
            stop_requested=lambda: None,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 1.0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert coordinator.snapshot().active_actor == "derivation.session_profile"

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert coordinator.snapshot().active_actor is None
    finally:
        release.set()
        if not task.done():
            with pytest.raises(asyncio.CancelledError):
                await task
        await _shutdown(compute, coordinator)
