"""Sealed maintenance parts use the existing session-profile owner.

The maintenance surface receives these neutral, uncapped receipts, then owns
its own protocol/audit translation. These tests keep that surface from
rediscovering archive work or borrowing a writer outside the normal owner.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from polylogue.daemon.convergence import (
    DaemonConverger,
    SelectedSessionTarget,
    SessionProfileConvergenceOwner,
)
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.session_profile_convergence import (
    make_session_profile_derivation,
    make_session_profile_frame,
    make_session_summary_derivation,
    make_session_usage_rollup_derivation,
)
from polylogue.storage.derived.session import derivation as session_derivation
from polylogue.storage.derived.session.derivation import SessionProfilePartFacts, SessionProfileReplacement
from tests.infra.convergence_harness import (
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
            _converger_for(index_db, archive_root, adapter),
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        ),
        compute,
        coordinator,
    )


def _converger_for(index_db: Path, archive_root: Path, adapter: object) -> DaemonConverger:
    """Keep selected-profile tests on the production summary/profile graph."""
    return DaemonConverger(
        (),
        derivations=(
            make_session_summary_derivation(index_db, archive_root=archive_root),
            make_session_usage_rollup_derivation(index_db, archive_root=archive_root, now=lambda: 0.0),
            adapter,
        ),
    )


async def _shutdown(compute: BoundedComputeAdapter, coordinator: DaemonWriteCoordinator) -> None:
    compute.shutdown(wait=True)
    await coordinator.shutdown(timeout=1.0)


def test_session_inspection_cannot_certify_a_partition_from_mixed_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the adapter's read snapshot falsely certifies this partition.

    The old output disagrees with its inputs before the competing commit;
    afterwards it is missing a sibling. No committed state is valid, although
    the old output and the new input binding would appear to agree.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    adapter = make_session_profile_derivation(recovered.index_db, archive_root=recovered.root, now=lambda: 0.0)
    frame = make_session_profile_frame(
        recovered.index_db, archive_root=recovered.root, scope=(recovered.target_session_id,)
    )
    converger = _converger_for(recovered.index_db, recovered.root, adapter)
    # Two: this session's canonical usage rollup, then the profile prepared
    # from it. The rollup is the profile's declared prerequisite key.
    assert converger.converge_derivations(frame).done == 2
    with sqlite3.connect(recovered.index_db) as conn:
        title = conn.execute(
            "SELECT title FROM sessions WHERE session_id = ?", (recovered.target_session_id,)
        ).fetchone()[0]
        conn.execute("UPDATE sessions SET title = 'Changed input' WHERE session_id = ?", (recovered.target_session_id,))
        conn.commit()
    assert adapter.inspect(frame, (recovered.target_session_id,)) == {recovered.target_session_id: "stale"}

    stored_partitions = session_derivation._stored_partitions
    commits = 0

    def commit_between_output_and_input_reads(
        conn: sqlite3.Connection, session_ids: Sequence[str]
    ) -> Mapping[str, session_derivation._StoredPartition]:
        nonlocal commits
        stored = stored_partitions(conn, session_ids)
        with sqlite3.connect(recovered.index_db) as writer:
            writer.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, recovered.target_session_id))
            removed = writer.execute(
                "DELETE FROM session_latency_profiles WHERE session_id = ?", (recovered.target_session_id,)
            ).rowcount
            writer.commit()
        assert removed == 1
        commits += 1
        return stored

    with monkeypatch.context() as race:
        race.setattr(session_derivation, "_stored_partitions", commit_between_output_and_input_reads)
        assert adapter.inspect(frame, (recovered.target_session_id,)) == {recovered.target_session_id: "stale"}
    assert commits == 1
    assert adapter.inspect(frame, (recovered.target_session_id,)) == {recovered.target_session_id: "stale"}

    restarted = _converger_for(recovered.index_db, recovered.root, adapter)
    assert restarted.converge_derivations(frame).done == 1
    assert adapter.inspect(frame, (recovered.target_session_id,)) == {recovered.target_session_id: "valid"}
    assert restarted.converge_derivations(frame).made_no_publication_attempts


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

    def move_target_binding(frame: object, session_id: str) -> SessionProfileReplacement:
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

    adapter.compute = move_target_binding  # type: ignore[method-assign, assignment]
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        _converger_for(recovered.index_db, recovered.root, adapter),
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

    def facts_missing_after_publish(frame: object, session_id: str) -> SessionProfilePartFacts:
        nonlocal observations
        observations += 1
        if observations == 3:
            raise RuntimeError("post-publication read unavailable")
        return original_facts(frame, session_id)

    adapter.selected_part_facts = facts_missing_after_publish  # type: ignore[method-assign]
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        _converger_for(recovered.index_db, recovered.root, adapter),
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
        _converger_for(recovered.index_db, recovered.root, adapter),
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
