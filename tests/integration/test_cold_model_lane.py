"""The cold model lane over the real MCP stdio wire.

The production replay stays the execution oracle; what is proved here is that
discovery is genuinely wire-only and registry-complete, and that a server
which regresses the closed polylogue-3k30 continuation defect makes the lane
fail rather than fall back to an in-process registry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from devtools.continuity_cold_model import (
    ColdModelLaneError,
    ScriptedColdModelBackend,
    capture_wire_discovery,
    run_cold_model_lane,
)
from devtools.continuity_replay import CancellationExerciseReceipt
from devtools.continuity_scenarios import CONTINUITY_SCENARIOS, continuity_scenario
from polylogue.core.json import JSONDocument, JSONValue
from tests.infra.continuity import load_continuity_catalog, seed_continuity_archive


def _doc(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def _rows(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def _int(value: JSONValue) -> int:
    assert isinstance(value, int)
    return value


@pytest.fixture(scope="module")
def cold_corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, JSONDocument]:
    archive_root = tmp_path_factory.mktemp("cold-model-lane") / "archive"
    catalog = load_continuity_catalog()
    seed_continuity_archive(archive_root, catalog=catalog)
    return archive_root, catalog


def _synthetic_backend() -> ScriptedColdModelBackend:
    answers = {
        scenario.sparse_prompt: json.dumps(
            {
                "steps": [
                    {"tool": step.tool, "arguments": step.argument_dict(), "paginate": step.paginate}
                    for step in scenario.route_steps
                ],
                "stop_conditions": ["all continuations are exhausted"],
                "citation_fields": [
                    next(
                        segment
                        for segment in reversed(projection.path)
                        if isinstance(segment, str) and segment not in {"*", "items"}
                    )
                    for projection in scenario.evidence_projections
                ],
                "uncertainty": "medium",
            }
        )
        for scenario in CONTINUITY_SCENARIOS
    }
    return ScriptedColdModelBackend(answers=answers)


class _StubRoute:
    """A server whose explain continuation never advances (the 3k30 shape)."""

    def __init__(self, *, advance: bool) -> None:
        self.advance = advance
        self.calls = 0

    @property
    def transport_name(self) -> str:
        return "stub"

    @property
    def discovery(self) -> dict[str, Any]:
        return {"protocol_version": "test", "tools": {"explain": {"input_schema": {}}}}

    async def invoke(self, tool: str, arguments: Mapping[str, object]) -> str:
        self.calls += 1
        offset = arguments.get("offset", 0)
        assert isinstance(offset, int)
        if self.advance and offset >= 2:
            return json.dumps({"examples": [{"key": f"k{offset}"}]})
        return json.dumps(
            {
                "budget_exceeded": True,
                "page": {"examples": [{"key": f"k{offset}"}]},
                "continuation": {
                    "tool": "explain",
                    "arguments": {
                        "subject": arguments["subject"],
                        "offset": offset + 1 if self.advance else 0,
                    },
                },
            }
        )

    async def exercise_cancellation(
        self, tool: str, arguments: Mapping[str, object], *, grace_ms: int
    ) -> CancellationExerciseReceipt:  # pragma: no cover - unused here
        raise NotImplementedError


class TestWireDiscovery:
    """A non-advancing continuation fails; there is no registry fallback."""

    @pytest.mark.asyncio
    async def test_non_advancing_fails(self) -> None:
        """Anti-vacuity: the pre-3k30 server looped forever instead of erroring."""
        with pytest.raises(ColdModelLaneError) as excinfo:
            await capture_wire_discovery(_StubRoute(advance=False), subjects=("result",), max_hops=5)
        assert excinfo.value.kind == "discovery_continuation_not_advancing"
        assert excinfo.value.axis == "discovery"

    @pytest.mark.asyncio
    async def test_advancing_exhausts(self) -> None:
        capture = await capture_wire_discovery(_StubRoute(advance=True), subjects=("result",), max_hops=5)
        assert {example["key"] for example in capture.examples} == {"k0", "k1", "k2"}


@pytest.mark.slow
class TestColdLane:
    """Every registry scenario is graded against real wire discovery."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(900)
    async def test_registry_covered_on_wire(self, cold_corpus: tuple[Path, JSONDocument]) -> None:
        """Anti-vacuity: a lane that skipped a scenario, or that read discovery
        from the in-process catalogue instead of the wire, would still report a
        green status here."""
        archive_root, catalog = cold_corpus
        report = await run_cold_model_lane(archive_root, catalog, _synthetic_backend())

        assert report["coverage_errors"] == []
        assert report["scenario_count"] == len(CONTINUITY_SCENARIOS)
        assert report["status"] == "pass"

        receipt = _doc(report["discovery_receipt"])
        assert receipt["transport"] == "mcp-stdio-json-rpc"
        assert receipt["protocol_version"] != "in-process-registration"
        assert _int(receipt["continuation_hops"]) > 2
        assert _int(receipt["example_count"]) > 100
        assert _int(receipt["tool_count"]) >= 4

        for row in _rows(report["results"]):
            result = _doc(row)
            assert result["disposition"] == "pass"
            assert result["product_status"] == "pass"
            assert result["model_execution_status"] == "pass"
            assert _int(result["prompt_bytes"]) > 1000

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_wrong_plan_fails_grade(self, cold_corpus: tuple[Path, JSONDocument]) -> None:
        """A wrong model answer fails formulation while the production route
        still passes -- the two are graded on separate axes."""
        archive_root, catalog = cold_corpus
        scenario = continuity_scenario("resume")
        wrong = json.dumps(
            {
                "steps": [{"tool": "query", "arguments": {"expression": "sessions where nothing"}}],
                "stop_conditions": ["stop"],
                "citation_fields": ["message_id"],
                "uncertainty": "low",
            }
        )
        backend = ScriptedColdModelBackend(answers={scenario.sparse_prompt: wrong})

        report = await run_cold_model_lane(archive_root, catalog, backend, scenario_names=("resume",))

        result = _doc(_rows(report["results"])[0])
        assert result["disposition"] == "fail"
        assert result["product_status"] == "pass"
        attempt = _doc(_rows(result["attempts"])[0])
        grades = [_doc(grade) for grade in _rows(attempt["grades"])]
        failed = {grade["axis"] for grade in grades if grade["status"] == "fail"}
        assert failed == {"plan_equivalence"}

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_same_signature_wrong_arguments_fail_model_execution(
        self, cold_corpus: tuple[Path, JSONDocument]
    ) -> None:
        """A correct tool family cannot stand in for the model's actual query."""
        archive_root, catalog = cold_corpus
        scenario = continuity_scenario("resume")
        wrong = json.dumps(
            {
                "steps": [
                    {
                        "tool": "query",
                        "arguments": {"expression": "messages where text:absent", "limit": 2},
                        "paginate": True,
                    }
                ],
                "stop_conditions": ["continuation exhausted"],
                "citation_fields": ["message_id"],
                "uncertainty": "medium",
            }
        )
        backend = ScriptedColdModelBackend(answers={scenario.sparse_prompt: wrong})

        report = await run_cold_model_lane(archive_root, catalog, backend, scenario_names=("resume",))

        result = _doc(_rows(report["results"])[0])
        assert result["product_status"] == "pass"
        assert result["model_execution_status"] == "fail"
        assert result["disposition"] == "fail"
        attempt = _doc(_rows(result["attempts"])[0])
        grades = [_doc(grade) for grade in _rows(attempt["grades"])]
        assert all(grade["status"] == "pass" for grade in grades)
        assert _rows(result["model_execution_diagnostics"])
