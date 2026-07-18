"""Re-run the basic-usage demo suite's commands and assert output shape.

Guards `.agent/demos/basic-usage/` against silent regressions: each of the
eight walkthroughs documented there (find, read, search, resume, cost,
lineage, MCP, status/health) is re-executed against a fresh seeded demo
archive and checked for expected *shape* (non-empty results, expected JSON
keys, expected session refs) -- never exact volatile counts, since the demo
corpus's construct coverage can grow without invalidating any walkthrough.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


class DemoCheckError(AssertionError):
    """One basic-usage walkthrough did not reproduce its documented shape."""


def _invoke(archive_root: Path, args: list[str]) -> str:
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    runner = CliRunner()
    result = runner.invoke(cli, args, catch_exceptions=False)
    if result.exit_code != 0:
        raise DemoCheckError(f"command {args!r} exited {result.exit_code}: {result.output}")
    return result.output


def check_find_query(archive_root: Path) -> None:
    fielded = _invoke(
        archive_root,
        ["--origin", "codex-session", "find", "sessions where origin:codex-session", "then", "select", "--json"],
    )
    rows = json.loads(fielded)
    if not rows:
        raise DemoCheckError("fielded query returned no sessions")
    if not all(row["origin"] == "codex-session" for row in rows):
        raise DemoCheckError("fielded query returned a non-codex-session row")

    aggregate = _invoke(archive_root, ["find", "actions where tool:bash | group by origin | count"])
    if "count=" not in aggregate:
        raise DemoCheckError(f"pipeline aggregate produced no count rows: {aggregate!r}")


def check_read(archive_root: Path) -> None:
    output = _invoke(archive_root, ["find", "id:codex-session:demo-receipts", "then", "read", "--view", "transcript"])
    if "codex-session:demo-receipts" not in output:
        raise DemoCheckError("transcript read did not include the session ref")
    if "## user" not in output or "## assistant" not in output:
        raise DemoCheckError("transcript read did not include user/assistant turns")


def check_search(archive_root: Path) -> None:
    output = _invoke(archive_root, ["find", "clock", "then", "select", "--json"])
    rows = json.loads(output)
    if not rows:
        raise DemoCheckError("free-text search returned no hits")
    if len({row["origin"] for row in rows}) < 2:
        raise DemoCheckError("free-text search did not span multiple origins as the demo documents")


def check_resume(archive_root: Path) -> None:
    output = _invoke(archive_root, ["find", "id:codex-session:demo-receipts", "then", "continue"])
    if "resume" not in output.lower() or "demo-receipts" not in output:
        raise DemoCheckError(f"resume command generation did not reference the session: {output!r}")


def check_cost(archive_root: Path) -> None:
    output = _invoke(archive_root, ["analyze", "usage", "--format", "json"])
    payload = json.loads(output)
    lanes = payload.get("logical_pricing_lanes")
    if not lanes:
        raise DemoCheckError("usage rollup produced no logical_pricing_lanes")
    for lane in lanes:
        usage = lane.get("usage", {})
        lanes_present = {"input_tokens", "output_tokens", "cached_input_tokens"}
        if not lanes_present.issubset(usage):
            raise DemoCheckError(f"pricing lane usage is missing disjoint token lanes: {usage!r}")


def check_lineage(archive_root: Path) -> None:
    fork_output = _invoke(
        archive_root, ["find", "id:codex-session:demo-lineage-fork", "then", "read", "--view", "transcript"]
    )
    parent_output = _invoke(
        archive_root, ["find", "id:codex-session:demo-lineage-parent", "then", "read", "--view", "transcript"]
    )
    parent_line = "I have the base context and can branch the analysis."
    fork_only_line = "The fork diverges into demo corpus construct checks."
    if parent_line not in fork_output:
        raise DemoCheckError("forked session's transcript did not compose the parent's prefix")
    if fork_only_line not in fork_output:
        raise DemoCheckError("forked session's transcript did not include its own divergent tail")
    if parent_line not in parent_output:
        raise DemoCheckError("parent session's own transcript regressed")


def check_mcp_roundtrip(archive_root: Path) -> None:
    from devtools.continuity_replay import StdioMCPContinuityRoute

    async def _run() -> None:
        async with StdioMCPContinuityRoute(archive_root) as route:
            search_text = await route.invoke("search", {"query": "clock", "limit": 3})
            search_payload = json.loads(search_text)
            hits = search_payload.get("hits", [])
            if not hits:
                raise DemoCheckError("MCP search returned no hits")
            session_id = hits[0]["session"]["id"]
            summary_text = await route.invoke("get_session_summary", {"id": session_id})
            summary_payload = json.loads(summary_text)
            if summary_payload.get("id") != session_id:
                raise DemoCheckError("MCP get_session_summary did not resolve the session search returned")

    asyncio.run(_run())


def check_status_health(archive_root: Path) -> None:
    output = _invoke(archive_root, ["status", "--daemon-url", "http://127.0.0.1:1"])
    if "Sessions:" not in output:
        raise DemoCheckError("status fallback did not report a session count")
    if "daemon not running" not in output.lower():
        raise DemoCheckError("status did not report the expected daemon-unreachable fallback state")


_CHECKS: tuple[tuple[str, object], ...] = (
    ("find-query", check_find_query),
    ("read", check_read),
    ("search", check_search),
    ("resume", check_resume),
    ("cost", check_cost),
    ("lineage", check_lineage),
    ("mcp-roundtrip", check_mcp_roundtrip),
    ("status-health", check_status_health),
)


def run_checks(archive_root: Path) -> list[tuple[str, str | None]]:
    """Run every walkthrough check; return ``(name, error)`` pairs (``error`` is ``None`` on success)."""
    results: list[tuple[str, str | None]] = []
    for name, check in _CHECKS:
        try:
            check(archive_root)  # type: ignore[operator]
        except DemoCheckError as exc:
            results.append((name, str(exc)))
        else:
            results.append((name, None))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools workspace basic-usage-demo-check",
        description="Re-run the basic-usage demo suite's commands and assert output shape.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="A freshly seeded demo archive root (polylogue demo seed --with-overlays).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON results to stdout.")
    args = parser.parse_args(argv)

    results = run_checks(args.archive_root)
    failures = [(name, error) for name, error in results if error is not None]
    if args.json:
        sys.stdout.write(
            json.dumps({"results": [{"check": n, "error": e} for n, e in results]}, indent=2, sort_keys=True) + "\n"
        )
    else:
        for name, error in results:
            print(f"{'FAIL' if error else 'ok'}: {name}" + (f" -- {error}" if error else ""))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
