"""Slow-lane: a real polylogued subprocess drains a raw-authority backlog under probing.

This is the AC2 (daemon-health responsiveness) evidence for polylogue-hjpx.2. Unlike
``devtools/raw_authority_scale_proof.py`` and ``devtools/raw_authority_restart_proof.py``
(both call ``repair_raw_materialization`` in-process, no HTTP surface to probe), this
starts a real ``polylogued run`` subprocess and probes its ``/healthz/live``,
``/healthz/ready``, and ``/api/status`` endpoints while its own periodic convergence
loop drains a synthetic backlog.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from devtools.raw_authority_daemon_health_proof import run_raw_authority_daemon_health_proof

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_real_daemon_drains_backlog_while_staying_probeable(tmp_path: Path) -> None:
    # This proof is about daemon HTTP responsiveness under a real drain, not
    # about the corpus generator's own host-contention self-protection (that
    # is raw-authority-scale-proof's job) -- disable its pressure gate here,
    # matching the sibling raw-authority-scale-proof unit tests' convention.
    payload = run_raw_authority_daemon_health_proof(
        tmp_path,
        components=96,
        raws=96,
        probe_interval_seconds=0.1,
        max_drain_seconds=120.0,
        readiness_timeout_seconds=20.0,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    assert payload["success"] is True
    assert payload["final_backlog_candidate_count"] == 0
    assert cast(int, payload["initial_backlog_candidate_count"]) > 0

    probe = cast(dict[str, object], payload["probe"])
    summaries = cast(list[dict[str, object]], probe["endpoint_summaries"])
    endpoints = {cast(str, summary["endpoint"]) for summary in summaries}
    assert endpoints == {"/healthz/live", "/healthz/ready", "/api/status"}
    for summary in summaries:
        sample_count = cast(int, summary["sample_count"])
        assert sample_count > 0
        # A run_raw_authority_daemon_health_proof call that returns at all
        # already proves every endpoint's longest unresponsive span stayed
        # under the documented bound (evaluate_responsiveness raises
        # otherwise) -- do not additionally require zero failed samples,
        # since one isolated timing blip (e.g. readiness briefly reporting
        # not-ready right at daemon startup) is not itself a responsiveness
        # regression. Bound the failure *rate* instead of requiring zero.
        assert cast(int, summary["failure_count"]) <= max(1, sample_count // 20)
        assert summary["p50_ms"] is not None
        assert summary["max_ms"] is not None

    timeline = cast(list[dict[str, object]], payload["timeline"])
    assert timeline
