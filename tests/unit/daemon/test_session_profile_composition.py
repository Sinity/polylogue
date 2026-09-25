"""The composed daemon owner registers the counter prerequisite it consumes."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import aiosqlite
import pytest

from polylogue.daemon.derivation import Outcome
from polylogue.daemon.execution import BoundedComputeAdapter
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
from tests.infra.convergence_harness import seed_partial_convergence_archive


@pytest.mark.asyncio
async def test_working_dir_edits_refresh_profile_through_composed_demand(tmp_path: Path) -> None:
    """A path-only edit moves the input fence and the published cwd evidence."""
    recovered = seed_partial_convergence_archive(tmp_path / "archive", target_hot=False)
    session_id = recovered.target_session_id
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
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
    coordinator = DaemonWriteCoordinator()
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

        # The periodic pass still owns another seed obligation. It may settle
        # that session once, then its next unchanged pass has no index work.
        periodic = await composed.callback(None)
        assert any(item.key.key == recovered.unrelated_session_id for item in periodic.outcomes)
        periodic_unchanged = await composed.callback(None)
        assert periodic_unchanged.made_no_publication_attempts
        assert periodic_unchanged.work.inspected == 0

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
    coordinator = DaemonWriteCoordinator()
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
