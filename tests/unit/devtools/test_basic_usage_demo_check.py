"""Integration coverage for the basic-usage demo suite's regression check.

Seeds a real deterministic demo archive and runs every documented
walkthrough (CLI find/read/search/resume/cost/lineage, a real stdio MCP
round-trip, and the daemon-unreachable status fallback) exactly as
`.agent/demos/basic-usage/README.md` documents them. This is intentionally
an integration test (spawns a real MCP subprocess, seeds a real archive) --
if any of the eight walkthroughs stop reproducing their documented shape,
this test fails loudly instead of the regression going unnoticed until an
external cold reader hits it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from devtools.basic_usage_demo_check import DemoCheckError, check_lineage, run_checks


@pytest.fixture
def seeded_demo_archive(tmp_path: Path) -> Path:
    from polylogue.demo import seed_demo_archive

    archive_root = tmp_path / "archive"
    asyncio.run(seed_demo_archive(archive_root, force=True, with_overlays=True))
    return archive_root


def test_all_basic_usage_walkthroughs_reproduce_their_documented_shape(seeded_demo_archive: Path) -> None:
    results = run_checks(seeded_demo_archive)
    failures = {name: error for name, error in results if error is not None}
    assert failures == {}
    assert {name for name, _ in results} == {
        "find-query",
        "read",
        "search",
        "resume",
        "cost",
        "lineage",
        "mcp-roundtrip",
        "status-health",
    }


def test_check_lineage_fails_loudly_when_composition_breaks(
    seeded_demo_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: the check must actually be capable of failing, not just always pass."""
    import devtools.basic_usage_demo_check as module

    real_invoke = module._invoke

    def _mismatched_invoke(archive_root: Path, args: list[str]) -> str:
        output = real_invoke(archive_root, args)
        if args[:2] == ["find", "id:codex-session:demo-lineage-fork"]:
            return output.replace("I have the base context and can branch the analysis.", "")
        return output

    monkeypatch.setattr(module, "_invoke", _mismatched_invoke)
    with pytest.raises(DemoCheckError, match="compose the parent's prefix"):
        check_lineage(seeded_demo_archive)
