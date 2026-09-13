"""The composed daemon owner registers the counter prerequisite it consumes."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon.derivation import Outcome
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.session_profile_convergence import make_session_profile_frame
from polylogue.storage.derived.session.derivation import SESSION_PROFILE_DOMAIN, SESSION_PROFILE_RECIPE_VERSION
from polylogue.storage.derived.session.summary import SESSION_SUMMARY_DOMAIN, SESSION_SUMMARY_RECIPE_VERSION
from tests.infra.convergence_harness import seed_partial_convergence_archive


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
    assert frame.recipe_versions == {
        SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
        SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
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
        assert [(item.key.domain, item.outcome) for item in report.outcomes] == [
            (SESSION_SUMMARY_DOMAIN, Outcome.DONE),
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

        assert (await composed.callback((recovered.target_session_id,))).wrote_nothing
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)
