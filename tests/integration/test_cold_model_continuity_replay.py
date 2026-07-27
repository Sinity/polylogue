"""End-to-end proof of z9gh gap #3: paged/cancelled/cold-model MCP replay.

Runs the real ``devtools.cold_model_continuity_replay`` harness against real
MCP stdio JSON-RPC: a plan formulated at runtime from live-discovered tool
schemas plus the query-discovery catalog (never the scripted t8t
``ContinuityRouteStep`` declarations), executed to exhaustion with full
pagination-invariant verification, and a real server-confirmed mid-flight
cancellation of that same self-formulated plan -- against a freshly seeded
synthetic archive sized to force genuine multi-page pagination.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from devtools.cold_model_continuity_replay import (
    DEFAULT_COLD_MODEL_GOAL,
    main,
    run_cold_model_continuity_proof,
)


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    return dict(value)


def _int(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _str(value: object) -> str:
    assert isinstance(value, str)
    return value


@pytest.mark.asyncio
async def test_cold_model_continuity_proof_formulates_pages_and_cancels_over_real_mcp_stdio() -> None:
    report = await run_cold_model_continuity_proof()

    properties = _dict(report["properties"])

    plan_lane = _dict(properties["cold_model_plan_formulation"])
    assert plan_lane["status"] == "pass", plan_lane
    plan = _dict(plan_lane["plan"])
    assert plan["tool"] == "query"
    plan_arguments = _dict(plan["arguments"])
    assert "coldmodelpagingproof" in _str(plan_arguments["expression"])

    pagination_lane = _dict(properties["lossless_pagination"])
    assert pagination_lane["status"] == "pass", pagination_lane
    proof = _dict(pagination_lane["proof"])
    # The default goal seeds 27 rows at a forced page limit of 8 -- this
    # must genuinely round-trip more than one page, not merely claim to.
    assert _int(proof["page_count"]) > 1
    assert _int(proof["enumerated_item_count"]) == DEFAULT_COLD_MODEL_GOAL.corpus_row_count
    assert _int(proof["unique_identity_count"]) == DEFAULT_COLD_MODEL_GOAL.corpus_row_count
    assert _int(proof["exact_count"]) == DEFAULT_COLD_MODEL_GOAL.corpus_row_count
    assert proof["exact_enumeration_verified"] is True

    cancellation_lane = _dict(properties["midflight_cancellation"])
    assert cancellation_lane["status"] == "pass", cancellation_lane
    receipt = _dict(cancellation_lane["receipt"])
    assert receipt["confirmed"] is True
    assert receipt["outcome"] == "cancelled_confirmed"

    # The catalog-wire-fetch lane is a real, separately tracked defect
    # (polylogue-3k30): explain(subject="result") overflows the MCP response
    # budget with a non-narrowing continuation, so this lane is expected to
    # report a failure with an explicit diagnostic, not silently pass.
    catalog_lane = _dict(properties["cold_model_catalog_wire_fetch"])
    assert catalog_lane["status"] == "fail"
    assert "polylogue-3k30" in _str(catalog_lane["detail"])
    catalog_fetch = _dict(catalog_lane["catalog_fetch"])
    assert catalog_fetch["source"] == "in-process-registry-fallback"
    assert _int(catalog_fetch["example_count"]) > 0

    assert report["diagnostics"]
    # Given the known catalog-fetch defect, overall status is honestly "fail"
    # even though plan formulation, pagination, and cancellation all pass.
    assert report["status"] == "fail"

    json.dumps(report)


@pytest.mark.asyncio
async def test_cold_model_continuity_proof_forces_multiple_real_pagination_round_trips() -> None:
    goal = DEFAULT_COLD_MODEL_GOAL
    assert goal.corpus_row_count > goal.forced_page_limit * 2, (
        "the default goal must force at least three pagination round trips, "
        "not merely two, to rule out an off-by-one that only happens to work"
    )

    report = await run_cold_model_continuity_proof(goal=goal)

    properties = _dict(report["properties"])
    pagination_lane = _dict(properties["lossless_pagination"])
    proof = _dict(pagination_lane["proof"])
    assert _int(proof["page_count"]) >= 3


def test_main_cli_writes_json_output_and_reports_the_known_catalog_defect(tmp_path: Path) -> None:
    output_path = tmp_path / "cold-model-report.json"

    exit_code = main(["--output", str(output_path)])

    # Honest exit code: the known catalog-wire-fetch defect makes this "fail"
    # overall even though the other two properties pass -- see the AC-honesty
    # note in the harness's own report and polylogue-3k30.
    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["properties"]["lossless_pagination"]["status"] == "pass"
    assert payload["properties"]["midflight_cancellation"]["status"] == "pass"
    assert payload["properties"]["cold_model_plan_formulation"]["status"] == "pass"
