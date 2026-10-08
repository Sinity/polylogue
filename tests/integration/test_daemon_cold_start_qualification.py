"""Bounded cold-start evidence through the ordinary daemon CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.infra.daemon_cold_start import (
    FIXTURE_NESTED_SESSION,
    SESSION_IDS,
    qualify,
    write_fixture,
    write_retained_measurement_receipt,
)


def _assert_owned_process_tree_stopped(receipt: dict[str, object]) -> None:
    assert receipt["process_tree_survivors"] == []
    expected = "clear" if receipt["process_tree_rss_available"] else "unavailable"
    assert receipt["process_tree_survivor_check"] == expected


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.load_sensitive,
    pytest.mark.uses_real_clock("The owned daemon process and HTTP requests use real wall time."),
    # The 120-second active qualification deadline leaves up to 60 seconds for
    # a final bounded request iteration, child shutdown/reap, survivor checks,
    # fixture digesting, and failure-receipt serialization.
    pytest.mark.timeout(180),
]


@pytest.mark.parametrize("rejected", [600, 4096])
def test_empty_archive_discovers_rejected_prefix_and_publishes_exact_sessions(
    rejected: int,
    one_shot_workspace_env: dict[str, Path],
    request: pytest.FixtureRequest,
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
        measure_discovery=rejected == 4096,
    )
    assert receipt["schema_validation_mode"] == "advisory"
    _assert_owned_process_tree_stopped(receipt)
    if rejected == 4096:
        report_file = request.config.getoption("polylogue_report_file", default=None)
        retained_path = write_retained_measurement_receipt(
            receipt,
            Path(report_file) if report_file is not None else None,
        )
        if report_file is not None:
            assert retained_path is not None and retained_path.is_file()
            retained = json.loads(retained_path.read_text(encoding="utf-8"))
            assert retained["candidate_sha"] == receipt["candidate"]["sha"]
            assert retained["discovery"] == receipt["discovery_measurement"]
            assert retained["process_tree_rss"]["sampled_peak_bytes"] == receipt["process_tree_rss_bytes"]
            assert retained["process_tree_rss"]["limits"]["task_ids_per_sample"] > 0
            assert retained["schema_validation_mode"] == "advisory"
            expected_survivor_check = "clear" if receipt["process_tree_rss_available"] else "unavailable"
            assert retained["process_tree_survivor_check"] == expected_survivor_check
            assert retained["process_tree_survivor_count"] == 0
            if expected_survivor_check == "clear":
                assert retained["process_tree_survivor_check_missing_reason"] is None
            else:
                assert retained["process_tree_survivor_check_missing_reason"]
    assert receipt["outcome"] == "success"
    assert len(receipt["verified_sessions"]) == 3
    assert "public_search_all" in receipt["milestones_upper_bound_s"]
    assert receipt["intake_counts"]["offered_bytes"] > 0
    assert receipt["intake_counts"]["succeeded"] > 0
    assert isinstance(receipt["intake_counts"]["failed"], int)
    assert isinstance(receipt["intake_counts"]["deferred"], int)
    assert receipt["fixture"]["unchanged_after_run"] is True
    assert receipt["diagnostics"]["state"] == "measured"
    if rejected == 4096:
        measurement = receipt["discovery_measurement"]
        assert measurement["missing_events"] == []
        assert measurement["intervals_s"]["root_listing_and_entry_inspection"] >= 0
        assert measurement["intervals_s"]["root_sort_after_listing"] >= 0
        assert (
            measurement["timestamps_elapsed_s"]["first_yielded_entry"]
            >= measurement["timestamps_elapsed_s"]["root_sort_end"]
        )
        assert "first_publication_upper_bound" in measurement["timestamps_elapsed_s"]
        assert measurement["intervals_s"]["first_yield_to_first_publication_upper_bound"] > 0
        if receipt["process_tree_rss_available"]:
            assert receipt["process_tree_rss_bytes"] > 0
            assert receipt["process_tree_rss_sample_count"] > 0
            assert receipt["process_tree_rss_task_count_at_peak"] >= receipt["process_tree_rss_process_count_at_peak"]
            assert receipt["process_tree_rss_peak_sample_truncated"] is False
        else:
            assert receipt["process_tree_rss_bytes"] is None
            assert receipt["process_tree_rss_missing_reason"] == "proc_children_unavailable"


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
    assert receipt["schema_validation_mode"] == "advisory"
    _assert_owned_process_tree_stopped(receipt)
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
    assert receipt["schema_validation_mode"] == "advisory"
    _assert_owned_process_tree_stopped(receipt)
    assert receipt["outcome"] == "incomplete_population"
    # The malformed third is a settled parse refusal, so the cold generation
    # promotes the two sound sessions and never a third.
    assert receipt["verified_sessions"] == sorted(SESSION_IDS[:2])
    assert receipt["parse_refusal"]["source"] == FIXTURE_NESTED_SESSION
    assert receipt["parse_refusal"]["error"]
    assert receipt["error"]
    assert receipt["last_log_records"]
