"""The composed daemon owner registers the counter prerequisite it consumes."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import aiosqlite
import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.daemon.derivation import DerivationFrame, DerivationReport, Outcome
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.session_profile_convergence import make_session_profile_frame
from polylogue.storage.derived.session.derivation import SESSION_PROFILE_DOMAIN, SESSION_PROFILE_RECIPE_VERSION
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.derived.session.marker_domain import SESSION_MARKER_DOMAIN, SESSION_MARKER_RECIPE_VERSION
from polylogue.storage.derived.session.summary import SESSION_SUMMARY_DOMAIN, SESSION_SUMMARY_RECIPE_VERSION
from polylogue.storage.derived.session.usage_rollup import (
    SESSION_USAGE_ROLLUP_DOMAIN,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.io_phase_metrics import connect_measured
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.convergence_harness import _seed_raw_source_session, seed_partial_convergence_archive


@pytest.mark.asyncio
async def test_working_dir_edits_refresh_profile_through_composed_demand(tmp_path: Path) -> None:
    """A path-only edit moves the input fence and the published cwd evidence."""
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    session_id = recovered.target_session_id
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        assert (await composed.callback((session_id,))).done
        with sqlite3.connect(recovered.index_db) as conn:
            original_session = conn.execute(
                "SELECT content_hash, updated_at_ms, message_count FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            original_messages = conn.execute(
                "SELECT message_id, content_hash, input_tokens FROM messages WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
            previous_binding = session_input_bindings(conn, (session_id,))[session_id]

        edits = (
            (
                "INSERT INTO session_working_dirs(session_id, path, position) VALUES (?, ?, 0)",
                (session_id, "/work/one"),
                "/work/one",
            ),
            ("UPDATE session_working_dirs SET path = ? WHERE session_id = ?", ("/work/two", session_id), "/work/two"),
            ("DELETE FROM session_working_dirs WHERE session_id = ?", (session_id,), None),
        )
        for sql, params, expected_path in edits:
            with sqlite3.connect(recovered.index_db) as conn:
                conn.execute(sql, params)
                conn.commit()
                changed_binding = session_input_bindings(conn, (session_id,))[session_id]
                assert changed_binding != previous_binding
                assert conn.execute(
                    "SELECT input_content_hash FROM session_profiles WHERE session_id = ?", (session_id,)
                ).fetchone() == (None,)
                assert (
                    conn.execute("SELECT 1 FROM session_profile_demand WHERE session_id = ?", (session_id,)).fetchone()
                    is not None
                )
            report = await composed.callback((session_id,))
            assert any(
                item.key.domain == SESSION_PROFILE_DOMAIN and item.outcome is Outcome.DONE for item in report.outcomes
            )
            with sqlite3.connect(recovered.index_db) as conn:
                stored_binding, evidence_json = conn.execute(
                    "SELECT input_content_hash, evidence_payload_json FROM session_profiles WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                assert stored_binding == changed_binding
                cwd_paths = json.loads(evidence_json)["cwd_paths"]
                assert (expected_path in cwd_paths) if expected_path is not None else not cwd_paths
                assert (
                    conn.execute(
                        "SELECT content_hash, updated_at_ms, message_count FROM sessions WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    == original_session
                )
                assert (
                    conn.execute(
                        "SELECT message_id, content_hash, input_tokens FROM messages WHERE session_id = ? ORDER BY position",
                        (session_id,),
                    ).fetchall()
                    == original_messages
                )
            previous_binding = changed_binding
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_composed_callback_repairs_summary_before_counter_dependent_profile(tmp_path: Path) -> None:
    """The composed adapter repairs a damaged counter before profile publication.

    Anti-vacuity: omitting the summary adapter from composition leaves the
    profile's declared prerequisite unresolved; omitting the prerequisite lets
    the profile consume the corrupted session counter.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    with sqlite3.connect(recovered.index_db) as conn:
        conn.execute(
            "UPDATE sessions SET word_count = 0 WHERE session_id = ?",
            (recovered.target_session_id,),
        )
        conn.commit()

    frame = make_session_profile_frame(
        recovered.index_db,
        archive_root=recovered.root,
        scope=(recovered.target_session_id,),
    )
    # Every domain the composed owner drives, in the order it drives them.
    # Omitting one here made this assertion pass vacuously against a frame that
    # already carried the usage rollup (inherited red at 9ef655cb8, repaired
    # rather than re-narrowed).
    assert frame.recipe_versions == {
        SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
        SESSION_USAGE_ROLLUP_DOMAIN: session_usage_rollup_recipe_version(),
        SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        SESSION_MARKER_DOMAIN: SESSION_MARKER_RECIPE_VERSION,
    }

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        report = await composed.callback((recovered.target_session_id,))
        # The declared run order, all of it. Naming only two domains let this
        # pass while a third already ran (inherited red at 9ef655cb8).
        assert [(item.key.domain, item.outcome) for item in report.outcomes] == [
            (SESSION_SUMMARY_DOMAIN, Outcome.DONE),
            (SESSION_USAGE_ROLLUP_DOMAIN, Outcome.DONE),
            (SESSION_PROFILE_DOMAIN, Outcome.DONE),
        ]
        with sqlite3.connect(recovered.index_db) as conn:
            stored_word_count = conn.execute(
                "SELECT word_count FROM sessions WHERE session_id = ?",
                (recovered.target_session_id,),
            ).fetchone()
            expected_word_count = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) FROM messages WHERE session_id = ?",
                (recovered.target_session_id,),
            ).fetchone()
            profile_count = conn.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?",
                (recovered.target_session_id,),
            ).fetchone()
        assert stored_word_count is not None and expected_word_count is not None
        assert stored_word_count[0] == expected_word_count[0]
        assert profile_count == (1,)

        scoped_unchanged = await composed.callback((recovered.target_session_id,))
        assert scoped_unchanged.made_no_publication_attempts
        assert scoped_unchanged.work.inspected == 0

        # The periodic archive sweep advances through prerequisite domains.
        periodic = [await composed.callback(None) for _ in range(3)]
        assert any(item.key.key == recovered.unrelated_session_id for report in periodic for item in report.outcomes)
        periodic_unchanged = await composed.callback(None)
        assert periodic_unchanged.made_no_publication_attempts

        with sqlite3.connect(recovered.index_db) as conn:
            conn.execute(
                "UPDATE messages SET input_tokens = COALESCE(input_tokens, 0) + 1 WHERE session_id = ?",
                (recovered.target_session_id,),
            )
            conn.commit()
        changed = await composed.callback((recovered.target_session_id,))
        assert any(
            item.key.domain == SESSION_PROFILE_DOMAIN and item.outcome is Outcome.DONE for item in changed.outcomes
        )
        assert changed.work.inspected >= 3
        with sqlite3.connect(recovered.index_db) as conn:
            assert (
                conn.execute(
                    "SELECT 1 FROM session_profile_demand WHERE session_id = ?",
                    (recovered.target_session_id,),
                ).fetchone()
                is None
            )
        assert (await composed.callback(None)).work.inspected == 0
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_promoted_generation_starts_a_bounded_profile_pass_from_new_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.daemon.convergence import SessionProfileConvergenceOwner

    # The promoted pass visits every audit domain, and its return value is the
    # last domain's report; keep each pass's report to find the profile's.
    passes: list[DerivationReport] = []
    real_converge = SessionProfileConvergenceOwner.converge

    async def recording_converge(self: object, *args: object, **kwargs: object) -> DerivationReport:
        report = await real_converge(self, *args, **kwargs)  # type: ignore[arg-type]
        passes.append(report)
        return report

    monkeypatch.setattr(SessionProfileConvergenceOwner, "converge", recording_converge)
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        await composed.callback(None)
        with sqlite3.connect(recovered.index_db) as conn:
            conn.execute("UPDATE sessions SET word_count = 0 WHERE session_id = ?", (recovered.target_session_id,))
            conn.commit()
            assert conn.execute(
                "SELECT 1 FROM session_profile_demand WHERE session_id = ?", (recovered.target_session_id,)
            ).fetchone()

        passes.clear()
        await composed.converge_promoted()
        assert any(
            item.key.domain == SESSION_PROFILE_DOMAIN
            and item.key.key == recovered.target_session_id
            and item.outcome is Outcome.DONE
            for report in passes
            for item in report.outcomes
        )
        assert all(report.work.discovered <= 128 and report.work.published <= 64 for report in passes)
        with sqlite3.connect(recovered.index_db) as conn:
            assert conn.execute(
                "SELECT 1 FROM session_profiles WHERE session_id = ?", (recovered.target_session_id,)
            ).fetchone()
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("audit_pass takes an absolute time.monotonic() deadline")
async def test_periodic_sweep_reaches_more_than_one_budget_of_profiles_without_demand(tmp_path: Path) -> None:
    """A quiet archive tail must survive bounded prerequisite passes.

    The archive-wide audit advances only in bounded slices after demand
    (polylogue-6remh), so each iteration here is one ``audit_pass`` slice.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("{}\n")

    def seed() -> int:
        conn = connect_measured(recovered.index_db)
        try:
            for number in range(129):
                _seed_raw_source_session(conn, session_id=f"sweep-{number:03d}", source_path=source_path)
            conn.execute("DELETE FROM session_profile_demand")
            conn.commit()
            count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
            assert count > 128
            assert conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == 0
            return count
        finally:
            conn.close()

    expected = run_off_event_loop(seed)

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        prerequisite_ticks = 0
        assert composed.audit_pass is not None
        for _ in range(16):
            sliced = await composed.audit_pass(time.monotonic() + 600.0)
            assert sliced is not None
            report = sliced
            prerequisite_ticks += 1
            assert report.work.discovered <= 128
            assert report.work.published <= 64
            if report.cursor.position(SESSION_USAGE_ROLLUP_DOMAIN).swept:
                break
        else:
            pytest.fail("bounded prerequisite sweep did not finish")
        with sqlite3.connect(recovered.index_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == 0
            # Prerequisite publication can create transaction-owned hints.
            # Remove them here so the profile phase must discover the tail.
            conn.execute("DELETE FROM session_profile_demand")
            conn.commit()

        profile_ticks = 0
        for _ in range(16):
            sliced = await composed.audit_pass(time.monotonic() + 600.0)
            if sliced is None:
                break
            report = sliced
            profile_ticks += 1
            assert report.work.discovered <= 128
            assert report.work.published <= 64
            with sqlite3.connect(recovered.index_db) as conn:
                if conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == expected:
                    break
        assert prerequisite_ticks > 1
        assert profile_ticks > 1
        with sqlite3.connect(recovered.index_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == expected
            assert conn.execute("SELECT COUNT(*) FROM session_profile_demand").fetchone()[0] == 0
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_fresh_owner_resweeps_missing_profile_without_demand(tmp_path: Path) -> None:
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    with sqlite3.connect(recovered.index_db) as conn:
        conn.execute("DELETE FROM session_profile_demand")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == 0

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        # The startup audit, not a demand tick, finds profiles no demand names.
        await composed.converge_backlog(600.0)
        with sqlite3.connect(recovered.index_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == 2
            conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (recovered.target_session_id,))
            conn.execute("DELETE FROM session_profile_demand")
            conn.commit()

        # A daemon restart loses the process cursor and every demand hint.
        restarted = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        await restarted.converge_backlog(600.0)
        with sqlite3.connect(recovered.index_db) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", (recovered.target_session_id,)
            ).fetchone() == (1,)
            assert conn.execute("SELECT COUNT(*) FROM session_profile_demand").fetchone() == (0,)
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_marker_lowering_sees_a_user_db_created_after_composition(tmp_path: Path) -> None:
    """A ``user.db`` that appears after composition consumes retained inputs.

    Anti-vacuity: binding marker availability at construction time (the
    ``user_db.exists()`` check this replaces) leaves both marker connections
    ``None`` for the owner's lifetime, so no assertion is ever lowered and the
    final assertion count stays zero.
    """
    from polylogue.markers import candidates_for_block
    from polylogue.storage.accepted_marker_inputs import append_accepted_marker_input, prepare_accepted_marker_input
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    with sqlite3.connect(recovered.index_db) as conn:
        message_id, block_id = conn.execute(
            "SELECT message_id, block_id FROM blocks WHERE session_id = ? ORDER BY block_id LIMIT 1",
            (recovered.target_session_id,),
        ).fetchone()
        assert message_id is not None and block_id is not None
        conn.execute(
            "UPDATE blocks SET text = ? WHERE block_id = ?",
            ("current index text cannot become marker input\n", block_id),
        )
        conn.commit()

    candidate = candidates_for_block(str(message_id), str(block_id), "::finding: marker survives a late user tier\n")[0]
    candidate_record = asdict(candidate)
    candidate_record["assertion_kind"] = (
        candidate.assertion_kind.value if candidate.assertion_kind is not None else None
    )
    async with aiosqlite.connect(recovered.root / "source.db") as source:
        batch = prepare_accepted_marker_input(
            "late-user-tier-marker",
            [{"session_id": recovered.target_session_id, "candidates": [candidate_record]}],
        )
        await append_accepted_marker_input(source, batch)
        await source.commit()

    user_db = recovered.root / "user.db"
    user_db.unlink()

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        # The durable user tier is created only after the owner exists, exactly
        # as the daemon's own startup order does.
        with sqlite3.connect(user_db) as conn:
            initialize_archive_tier(conn, ArchiveTier.USER)
            conn.commit()

        report = await composed.callback((recovered.target_session_id,))
        assert report.outcomes
        assert all(item.outcome is Outcome.DONE for item in report.outcomes), report.outcomes

        with sqlite3.connect(user_db) as conn:
            lowered = conn.execute("SELECT COUNT(*) FROM assertions").fetchone()
        assert lowered is not None and lowered[0] > 0
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_backlog_call_sweeps_every_domain_in_bounded_passes(tmp_path: Path) -> None:
    """One periodic tick drains the audit sweep instead of one pass of it.

    Anti-vacuity: a backlog call that runs a single bounded pass (the old
    periodic tick) publishes at most 64 keys of one domain, so fewer than all
    129+ profiles exist afterwards and the audit is still pending.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("{}\n")

    def seed() -> int:
        conn = connect_measured(recovered.index_db)
        try:
            for number in range(129):
                _seed_raw_source_session(conn, session_id=f"backlog-{number:03d}", source_path=source_path)
            conn.execute("DELETE FROM session_profile_demand")
            conn.commit()
            count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
            return count
        finally:
            conn.close()

    expected = run_off_event_loop(seed)

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        passes = 0
        real_pass = composed.audit_pass
        assert real_pass is not None

        async def counting(deadline_at: float) -> DerivationReport | None:
            nonlocal passes
            passes += 1
            report = await real_pass(deadline_at)
            assert report is None or report.work.published <= 64
            return report

        # A pass whose instant already passed (e.g. spent waiting for the
        # owner) does not start.
        assert await real_pass(0.0) is None
        counted = replace(composed, audit_pass=counting)
        await counted.converge_backlog(600.0)

        assert passes > 1
        assert not composed.audit_pending()
        with sqlite3.connect(recovered.index_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == expected
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


# An absolute monotonic instant no test run reaches; the fake owner ignores it.
_FAR_DEADLINE = 1e18


@pytest.mark.asyncio
async def test_a_persistently_pending_domain_does_not_starve_later_audit_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary key that stays blocked leaves usage, profile and markers their turns.

    Anti-vacuity: restart the pending domain in place instead of rotating and
    every pass re-runs the summary domain, so the marker domain is never
    visited; drop the re-owe step and the domains after a late-completing
    summary are not revisited.
    """
    from polylogue.daemon.convergence import SessionProfileConvergenceOwner
    from polylogue.daemon.derivation import DiscoveryPhase, DomainCursor, PassCursor

    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    visited: list[str] = []
    summary_blocked = True

    async def fake_converge(self: object, frame: object, **kwargs: object) -> DerivationReport:
        (domain,) = cast(tuple[str, ...], kwargs["domains"])
        visited.append(domain)
        pending = domain == SESSION_SUMMARY_DOMAIN and summary_blocked
        return DerivationReport(
            frame=cast(DerivationFrame, frame),
            counts={Outcome.PENDING: 1} if pending else {Outcome.DONE: 1},
            cursor=PassCursor({domain: DomainCursor(phase=DiscoveryPhase.DONE)}),
            cursor_unsettled_domains=frozenset({domain}) if pending else frozenset(),
        )

    monkeypatch.setattr(SessionProfileConvergenceOwner, "converge", fake_converge)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        assert composed.audit_pass is not None
        for _ in range(6):
            await composed.audit_pass(_FAR_DEADLINE)
        assert visited == [
            SESSION_SUMMARY_DOMAIN,
            SESSION_USAGE_ROLLUP_DOMAIN,
            SESSION_PROFILE_DOMAIN,
            SESSION_MARKER_DOMAIN,
            SESSION_SUMMARY_DOMAIN,
            SESSION_SUMMARY_DOMAIN,
        ]
        assert composed.audit_pending()

        summary_blocked = False
        visited.clear()
        while composed.audit_pending():
            await composed.audit_pass(_FAR_DEADLINE)
        assert visited == [
            SESSION_SUMMARY_DOMAIN,
            SESSION_USAGE_ROLLUP_DOMAIN,
            SESSION_PROFILE_DOMAIN,
            SESSION_MARKER_DOMAIN,
        ]
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_unreadable_bootstrap_audit_returns_one_fault_and_retries_after_schema_exists(tmp_path: Path) -> None:
    """Actual SQLite discovery failure cannot be retried inside one backlog call."""
    archive_root = tmp_path / "fresh"
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        real_pass = composed.audit_pass
        assert real_pass is not None
        audit_calls = 0

        async def observe(deadline: float) -> DerivationReport | None:
            nonlocal audit_calls
            audit_calls += 1
            assert audit_calls == 1, "unchanged unavailable output was acquired again in the same tick"
            return await real_pass(deadline)

        report = await replace(composed, audit_pass=observe).converge_backlog(600.0)
        assert report.failed == 1
        assert report.outcomes[0].key.domain == SESSION_SUMMARY_DOMAIN
        assert report.outcomes[0].key.key == "*"
        assert composed.audit_pending()
        assert not archive_root.exists(), "failed read discovery must not create an archive"

        recovered = seed_partial_convergence_archive(archive_root, target_hot=False)
        await composed.converge_backlog(600.0)
        assert not composed.audit_pending()
        with closing(sqlite3.connect(recovered.index_db)) as connection:
            assert connection.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] > 0
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["discovery", "swept", "pending"])
async def test_faulted_or_unchanged_audit_retains_owed_domains_and_serves_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Cursor progress, rather than changing counts, bounds same-tick retries."""
    from polylogue.daemon.convergence import SessionProfileConvergenceOwner
    from polylogue.daemon.derivation import DiscoveryPhase, DomainCursor, PassCursor

    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    visited: list[str] = []
    blocked = True

    async def converge(self: object, frame: DerivationFrame, **kwargs: object) -> DerivationReport:
        if frame.profile_demand_only:
            assert "domains" not in kwargs
            return DerivationReport(frame)
        (domain,) = cast(tuple[str, ...], kwargs["domains"])
        visited.append(domain)
        # This refuses an immediate repeat, so the old loop fails without a
        # sleep or a performance threshold. Count changes cannot justify it.
        assert len(visited) <= 5
        faulted = blocked and domain == SESSION_SUMMARY_DOMAIN
        outcome = Outcome.PENDING if faulted and failure == "pending" else Outcome.FAILED if faulted else Outcome.DONE
        phase = DiscoveryPhase.REQUIRED if faulted and failure == "discovery" else DiscoveryPhase.DONE
        return DerivationReport(
            frame,
            counts={outcome: len(visited)},
            cursor=PassCursor({domain: DomainCursor(phase)}),
            cursor_unsettled_domains=frozenset({domain}) if faulted and phase is DiscoveryPhase.DONE else frozenset(),
        )

    monkeypatch.setattr(SessionProfileConvergenceOwner, "converge", converge)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        await composed.converge_backlog(600.0)
        assert visited == [SESSION_SUMMARY_DOMAIN]
        await composed.converge_backlog(600.0)
        assert visited == [
            SESSION_SUMMARY_DOMAIN,
            SESSION_USAGE_ROLLUP_DOMAIN,
            SESSION_PROFILE_DOMAIN,
            SESSION_MARKER_DOMAIN,
            SESSION_SUMMARY_DOMAIN,
        ]
        assert composed.audit_pending()
        blocked = False
        visited.clear()
        await composed.converge_backlog(600.0)
        assert not composed.audit_pending()
        assert set(visited) == {
            SESSION_SUMMARY_DOMAIN,
            SESSION_USAGE_ROLLUP_DOMAIN,
            SESSION_PROFILE_DOMAIN,
            SESSION_MARKER_DOMAIN,
        }
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_backlog_cursor_progress_is_bound_to_the_generation_it_walked() -> None:
    """An equal cursor on a new declared generation is new work, not an unchanged retry."""
    from polylogue.daemon.derivation import DiscoveryPhase, DomainCursor, PassCursor
    from polylogue.daemon.session_profile_composition import ComposedSessionProfiles

    old_frame = DerivationFrame("/synthetic/archive", "index-generation:old")
    cursor = PassCursor({SESSION_SUMMARY_DOMAIN: DomainCursor(DiscoveryPhase.DONE)})
    initial = DerivationReport(old_frame)
    reports = [
        DerivationReport(old_frame, counts={Outcome.PENDING: 1}, cursor=cursor),
        DerivationReport(
            replace(old_frame, source_revision="index-generation:new"), counts={Outcome.PENDING: 1}, cursor=cursor
        ),
    ]
    consumed: list[str] = []

    async def demand(_scope: object) -> DerivationReport:
        return initial

    async def promoted() -> DerivationReport:
        return initial

    async def audit(_deadline: float) -> DerivationReport | None:
        if not reports:
            consumed.append("drained")
            return None
        report = reports.pop(0)
        consumed.append(report.frame.source_revision)
        return report

    profiles = ComposedSessionProfiles(demand, promoted, cast(Any, None), audit_pass=audit)
    await profiles.converge_backlog(600.0)
    assert consumed == ["index-generation:old", "index-generation:new", "drained"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fault", "pending"])
async def test_rotating_unsettled_sweeps_reach_every_domain_tail_and_retry_the_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """The real kernel keeps all four advancing cursors across bounded rotations."""
    from tests.infra.audit_derivation import install_audit_derivations

    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    adapters = install_audit_derivations(monkeypatch, failure)
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        reports: list[DerivationReport] = []
        real_pass = composed.audit_pass
        assert real_pass is not None

        async def observe(deadline: float) -> DerivationReport | None:
            report = await real_pass(deadline)
            if report is not None:
                reports.append(report)
            return report

        composed = replace(composed, audit_pass=observe)
        # Each faulted prefix rotates once; each pending sweep yields at its
        # terminal tail. Eight scheduled ticks cover four prefixes and tails.
        for _ in range(8):
            await composed.converge_backlog(600.0)
        assert all(adapter.keys[-1] in adapter.inspected for adapter in adapters)
        assert composed.audit_pending(), "a healthy tail must not certify its unsettled prefix"
        assert any(report.failed if failure == "fault" else report.pending for report in reports)
        before = [adapter.pages for adapter in adapters]
        for adapter in adapters:
            adapter.failure = None
        await composed.converge_backlog(600.0)
        assert not composed.audit_pending()
        assert all(adapter.pages > previous for adapter, previous in zip(adapters, before, strict=True))
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_new_generation_resets_unsettled_sweep_facts_and_restarts_the_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old partial fault cannot keep a new generation owed or skip its prefix."""
    from polylogue.daemon import session_profile_composition as composition
    from tests.infra.audit_derivation import install_audit_derivations

    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    adapters = install_audit_derivations(monkeypatch, "fault")
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=recovered.root)
    try:
        composed = compose_session_profile_callback(
            recovered.root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        report = await composed.converge_backlog(600.0)
        assert report.failed == 1
        assert not report.cursor.position(SESSION_SUMMARY_DOMAIN).swept
        first = adapters[0].prefix_inspections
        for adapter in adapters:
            adapter.failure = None
        real_frame = make_session_profile_frame

        def new_frame(*args: Any, **kwargs: Any) -> DerivationFrame:
            frame = real_frame(*args, **kwargs)
            return replace(frame, source_revision=frame.source_revision + ":new-generation")

        monkeypatch.setattr(composition, "make_session_profile_frame", new_frame)
        await composed.converge_backlog(600.0)
        assert not composed.audit_pending()
        assert adapters[0].prefix_inspections > first
        assert all(adapter.keys[-1] in adapter.inspected for adapter in adapters)
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_promotion_report_retains_each_domains_consumed_unsettled_evidence() -> None:
    from polylogue.daemon.session_profile_composition import _merge_reports

    frame = DerivationFrame("/synthetic/archive", "generation")
    first = DerivationReport(frame, cursor_unsettled_domains=frozenset({SESSION_SUMMARY_DOMAIN}))
    second = DerivationReport(frame, cursor_unsettled_domains=frozenset({SESSION_MARKER_DOMAIN}))
    assert _merge_reports(first, second).cursor_unsettled_domains == frozenset(
        {SESSION_SUMMARY_DOMAIN, SESSION_MARKER_DOMAIN}
    )
