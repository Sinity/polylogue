"""Bounded cold-start evidence through the ordinary daemon CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.daemon_cold_start import qualify, write_fixture

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.load_sensitive,
    pytest.mark.uses_real_clock("The owned daemon process and HTTP requests use real wall time."),
    pytest.mark.timeout(145),
]


@pytest.mark.parametrize("rejected", [600, 4096])
def test_empty_archive_discovers_rejected_prefix_and_publishes_exact_sessions(
    rejected: int,
    one_shot_workspace_env: dict[str, Path],
) -> None:
    workspace = one_shot_workspace_env["archive_root"].parent
    source = workspace / f"cold-source-{rejected}"
    archive = workspace / f"cold-archive-{rejected}"
    digest = write_fixture(source, rejected=rejected)
    receipt = qualify(
        archive=archive,
        source=source,
        artifacts=workspace / f"cold-artifacts-{rejected}",
        rejected=rejected,
        digest=digest,
    )
    assert receipt["outcome"] == "success"
    assert len(receipt["verified_sessions"]) == 3
    assert "public_search_all" in receipt["milestones_upper_bound_s"]
    assert receipt["diagnostics"]["state"] == "measured"


def test_held_first_directory_walk_keeps_status_and_metrics_responsive(
    one_shot_workspace_env: dict[str, Path],
) -> None:
    workspace = one_shot_workspace_env["archive_root"].parent
    source = workspace / "held-source"
    archive = workspace / "held-archive"
    digest = write_fixture(source, rejected=600)
    receipt = qualify(
        archive=archive, source=source, artifacts=workspace / "held-artifacts", rejected=600, digest=digest, held=True
    )
    assert receipt["outcome"] == "success"
    assert "discovering" in receipt["phase_coverage"]
    assert "discovery_first" in receipt["milestones_upper_bound_s"]


def test_malformed_last_session_cannot_produce_success_receipt(
    one_shot_workspace_env: dict[str, Path],
) -> None:
    workspace = one_shot_workspace_env["archive_root"].parent
    source = workspace / "malformed-source"
    archive = workspace / "malformed-archive"
    digest = write_fixture(source, rejected=600, malformed_last=True)
    receipt = qualify(
        archive=archive,
        source=source,
        artifacts=workspace / "malformed-artifacts",
        rejected=600,
        digest=digest,
        malformed_last=True,
    )
    assert receipt["outcome"] == "incomplete_population"
    assert receipt["verified_sessions"] == []
    assert receipt["candidate_sessions_unpublished"] == 2
    assert receipt["parse_refusal"]["source"] == "nested/z-session-2.jsonl"
    assert receipt["parse_refusal"]["error"]
    assert receipt["error"]
    assert receipt["last_log_records"]
