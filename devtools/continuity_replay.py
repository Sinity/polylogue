"""Run a privacy-safe continuity oracle against recorded route observations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polylogue.product.continuity_scenarios import continuity_scenario

_LIVE_MCP_SCENARIO = "mcp-query-transaction"
_LIVE_MCP_EXPRESSION = "actions where tool:Workflow"
_LIVE_MCP_PAGE_SIZE = 2


def _seed_live_mcp_archive(archive_root: Path) -> None:
    """Create the deterministic, independently-censused replay corpus."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        for number in range(1, 5):
            native_id = f"continuity-{number:02d}"
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    title=f"continuity workflow {number:02d}",
                    messages=[
                        ParsedMessage(
                            provider_message_id=f"{native_id}-message",
                            role=Role.ASSISTANT,
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TOOL_USE,
                                    tool_name="Workflow",
                                    tool_id=f"workflow-{number:02d}",
                                    tool_input={"step": number},
                                )
                            ],
                        )
                    ],
                )
            )


def _mcp_server_parameters(archive_root: Path) -> Any:
    """Return a fresh stdio server process; no continuation state is shared."""
    from mcp import StdioServerParameters

    environment = dict(os.environ)
    environment["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from polylogue.mcp.cli import main; main()"],
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
    )


def _tool_text(result: Any) -> str:
    if result.isError:
        raise RuntimeError(f"MCP query_units failed: {result.content!r}")
    text = "".join(item.text for item in result.content if hasattr(item, "text"))
    if not text:
        raise RuntimeError("MCP query_units returned no text payload")
    return text


async def _mcp_query_page(
    archive_root: Path, arguments: dict[str, object], *, discover: bool
) -> tuple[dict[str, object], dict[str, object]]:
    """Make one real MCP stdio client/server call and return its transcript row."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from pydantic import AnyUrl

    started = time.monotonic()
    async with (
        stdio_client(_mcp_server_parameters(archive_root)) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as client,
    ):
        await client.initialize()
        discovery: dict[str, object] = {}
        if discover:
            tools = await client.list_tools()
            resources = await client.list_resources()
            tool_names = {tool.name for tool in tools.tools}
            resource_uris = {str(resource.uri) for resource in resources.resources}
            if "query_units" not in tool_names or "polylogue://capabilities/query" not in resource_uris:
                raise RuntimeError("MCP discovery did not expose the canonical query transaction")
            capability = await client.read_resource(AnyUrl("polylogue://capabilities/query"))
            discovery = {
                "query_units_discovered": True,
                "capability_resource_read": bool(capability.contents),
            }
        result = await client.call_tool("query_units", arguments)
    payload = json.loads(_tool_text(result))
    if not isinstance(payload, dict):
        raise RuntimeError("MCP query_units payload must be an object")
    elapsed = time.monotonic() - started
    return payload, {
        "tool": "query_units",
        "arguments": arguments,
        "elapsed_seconds": elapsed,
        "page_bytes": len(json.dumps(payload, sort_keys=True).encode("utf-8")),
        **discovery,
    }


async def run_live_mcp_replay(scenario_name: str, fixture: dict[str, object]) -> dict[str, object]:
    """Run a cold-server MCP continuity walk over a local synthetic archive.

    The first page is intentionally the cancellation point. Its opaque
    continuation is recorded, the client and server are torn down, and every
    remaining page is fetched through a new stdio server process. This proves
    that progress is carried by transaction evidence rather than process memory.
    """
    if scenario_name != _LIVE_MCP_SCENARIO:
        raise ValueError(f"live MCP replay only supports {_LIVE_MCP_SCENARIO!r}")
    with tempfile.TemporaryDirectory(prefix="polylogue-continuity-") as temporary:
        archive_root = Path(temporary) / "archive"
        _seed_live_mcp_archive(archive_root)
        first, first_transcript = await _mcp_query_page(
            archive_root,
            {"expression": _LIVE_MCP_EXPRESSION, "limit": _LIVE_MCP_PAGE_SIZE},
            discover=True,
        )
        continuation = first.get("continuation")
        if not isinstance(continuation, str):
            raise RuntimeError("first MCP page did not issue an advancing continuation")
        transcript: list[dict[str, object]] = [
            {"phase": "initial-page", **first_transcript, "query_ref": first.get("query_ref")},
            {
                "phase": "cancelled-walk",
                "continuation": continuation,
                "reason": "client stopped after bounded page",
            },
        ]
        pages = [first]
        while isinstance(continuation, str):
            page, row = await _mcp_query_page(
                archive_root,
                {"expression": "actions where tool:ignored", "continuation": continuation},
                discover=False,
            )
            if page.get("query_ref") != first.get("query_ref") or page.get("result_ref") != first.get("result_ref"):
                raise RuntimeError("cold MCP resume changed transaction identity")
            transcript.append({"phase": "cold-resume-page", **row, "query_ref": page.get("query_ref")})
            pages.append(page)
            next_continuation = page.get("continuation")
            if next_continuation == continuation:
                raise RuntimeError("MCP continuation did not advance")
            continuation = next_continuation if isinstance(next_continuation, str) else None
        refs: list[str] = []
        for page in pages:
            items = page.get("items", [])
            if not isinstance(items, list):
                raise RuntimeError("MCP query_units page has invalid items")
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("session_id"), str):
                    raise RuntimeError("MCP action row has no session identity")
                refs.append(f"session:{item['session_id']}")
        page_sizes: list[int] = []
        for row in transcript:
            page_bytes = row.get("page_bytes")
            if isinstance(page_bytes, int):
                page_sizes.append(page_bytes)
        observation = {
            "refs": refs,
            "calls": len(pages),
            "page_bytes": max(page_sizes, default=0),
            "cancel_seconds": 0.0,
            "continuation_state_lost": False,
            "non_progressing_continuation": False,
        }
        evaluated = evaluate_replay(scenario_name, fixture, observation)
        return {
            **evaluated,
            "receipt": {
                "kind": "mcp-continuity-replay-v1",
                "scenario": scenario_name,
                "query_ref": first.get("query_ref"),
                "result_ref": first.get("result_ref"),
                "cancelled_after_pages": 1,
                "resumed_with_fresh_server_process": True,
                "transcript": transcript,
            },
        }


def evaluate_replay(
    scenario_name: str,
    fixture: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    """Classify an observed black-box walk against a fixture-derived oracle."""
    scenario = continuity_scenario(scenario_name)
    refs = observation.get("refs", ())
    if not isinstance(refs, list):
        raise ValueError("observation.refs must be a JSON list")
    observed_answer = observation.get("answer", {})
    if not isinstance(observed_answer, dict):
        raise ValueError("observation.answer must be a JSON object")
    observed_mutations = observation.get("mutations", {})
    if not isinstance(observed_mutations, dict):
        raise ValueError("observation.mutations must be a JSON object")
    calls = observation.get("calls", 0)
    if not isinstance(calls, int):
        raise ValueError("observation.calls must be an integer")
    failure = observation.get("failure")
    failure_value = str(failure) if failure is not None else None
    page_bytes = observation.get("page_bytes")
    if page_bytes is not None and not isinstance(page_bytes, int):
        raise ValueError("observation.page_bytes must be an integer")
    cancel_seconds = observation.get("cancel_seconds")
    if cancel_seconds is not None and not isinstance(cancel_seconds, (int, float)):
        raise ValueError("observation.cancel_seconds must be numeric")
    result = scenario.classify(
        fixture,
        observed_refs=[str(ref) for ref in refs],
        observed_answer=observed_answer,
        observed_mutations=observed_mutations,
        calls=calls,
        observed_failure=failure_value if failure_value in scenario.failure_taxonomy else None,  # type: ignore[arg-type]
        page_bytes=page_bytes,
        cancel_seconds=float(cancel_seconds) if cancel_seconds is not None else None,
        continuation_state_lost=bool(observation.get("continuation_state_lost", False)),
        non_progressing_continuation=bool(observation.get("non_progressing_continuation", False)),
    )
    return {
        "scenario": scenario.scenario_id,
        "classification": result,
        "expected_refs": list(scenario.oracle(fixture)),
        "expected_answer": scenario.oracle_answer(fixture),
        "expected_mutations": dict(scenario.oracle_mutations(fixture)),
        "observed_answer": observed_answer,
        "observed_mutations": observed_mutations,
        "observed_refs": refs,
        "calls": calls,
        "budget": {
            "max_calls": scenario.budget.max_calls,
            "max_page_bytes": scenario.budget.max_page_bytes,
            "max_cancel_seconds": scenario.budget.max_cancel_seconds,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("observation", type=Path, nargs="?")
    parser.add_argument("--live-mcp", action="store_true", help="execute the deterministic stdio MCP replay")
    parser.add_argument("--receipt", type=Path, help="write the replay receipt JSON to this path")
    args = parser.parse_args(argv)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if args.live_mcp:
        if args.observation is not None:
            parser.error("observation is not accepted with --live-mcp")
        result = asyncio.run(run_live_mcp_replay(args.scenario, fixture))
    else:
        if args.observation is None:
            parser.error("observation is required unless --live-mcp is used")
        observation = json.loads(args.observation.read_text(encoding="utf-8"))
        result = evaluate_replay(args.scenario, fixture, observation)
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_replay", "main", "run_live_mcp_replay"]
