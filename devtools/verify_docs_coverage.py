"""Verify every public CLI command, MCP tool, config key, and stable daemon
route is reachable (named) from the docs tree.

This is the reverse direction of ``devtools verify doc-commands`` (which
checks that every command *mentioned* in the docs resolves to a real
command). This lint checks that every real public surface is *mentioned*
somewhere in the docs tree, so a new command/tool/config-key/route shipped
without a matching doc update fails a gate instead of silently rotting into
an undocumented surface (polylogue-3tl.9).

Four typed inventories, each already maintained for a different purpose and
reused here rather than re-derived:

- CLI commands: ``polylogue.cli.command_inventory.iter_command_paths`` over
  the live Click tree.
- MCP tools: ``tests.infra.mcp.EXPECTED_TOOL_NAMES``.
- Config keys: ``polylogue.config.config_inventory_by_key``.
- Daemon routes: ``polylogue.daemon.route_contracts.ROUTE_CONTRACTS``,
  restricted to ``stable``/``shell_supported`` routes (``operational`` and
  ``private`` routes are internal-only by design and are not expected to
  have reader-facing docs).

"Reachable from the docs tree" is a coarse but honest check: the surface's
identifying token (command display name, tool name, config key, or route
pattern) appears verbatim somewhere in ``README.md`` or ``docs/**/*.md``.
This will not catch a doc that mentions a command name in an unrelated
sentence, but it does catch the failure mode this bead exists for: a surface
added with zero doc footprint.

Baseline
--------

The repo already carries pre-existing undocumented surfaces (discovered by
running this lint against HEAD when it was introduced). Those are recorded
in ``docs/plans/docs-coverage-baseline.yaml`` as tracked debt so introducing
the gate does not require writing ~100 doc entries in the same change. The
baseline is a ratchet, not a target: it must never grow, and CI should trend
it toward empty. A baseline entry whose surface has since been documented is
reported as "stale" (should be removed) but does not fail the lane.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard repo dep
    yaml = None  # type: ignore[assignment]

from devtools import repo_root as _get_root

ROOT = _get_root()
DOCS_ROOT = ROOT / "docs"
BASELINE_PATH = ROOT / "docs" / "plans" / "docs-coverage-baseline.yaml"

Surface = str  # "cli" | "mcp" | "config" | "route"


@dataclass(frozen=True, slots=True)
class CoverageGap:
    surface: Surface
    name: str


def _docs_corpus_text() -> str:
    """Concatenated text of every reader-facing Markdown page."""
    parts = [(ROOT / "README.md").read_text(encoding="utf-8")]
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _cli_inventory() -> tuple[str, ...]:
    from polylogue.cli.click_app import cli
    from polylogue.cli.command_inventory import iter_command_paths

    return tuple(sorted({path.display_name for path in iter_command_paths(cli)}))


def _mcp_inventory() -> tuple[str, ...]:
    from tests.infra.mcp import EXPECTED_TOOL_NAMES

    return tuple(sorted(EXPECTED_TOOL_NAMES))


def _config_inventory() -> tuple[str, ...]:
    from polylogue.config import config_inventory_by_key

    return tuple(sorted(config_inventory_by_key()))


def _route_inventory() -> tuple[str, ...]:
    from polylogue.daemon.route_contracts import ROUTE_CONTRACTS

    return tuple(
        sorted({route.pattern for route in ROUTE_CONTRACTS if route.stability in ("stable", "shell_supported")})
    )


def _all_inventories() -> dict[Surface, tuple[str, ...]]:
    return {
        "cli": _cli_inventory(),
        "mcp": _mcp_inventory(),
        "config": _config_inventory(),
        "route": _route_inventory(),
    }


def _load_baseline() -> dict[Surface, dict[str, str]]:
    """Return ``{surface: {name: reason}}`` from the baseline YAML."""
    if not BASELINE_PATH.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the docs coverage baseline")
    data = yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8")) or {}
    out: dict[Surface, dict[str, str]] = {}
    for surface, entries in (data.get("gaps") or {}).items():
        surface_map: dict[str, str] = {}
        for entry in entries or ():
            if isinstance(entry, str):
                surface_map[entry] = ""
            elif isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str):
                    surface_map[name] = str(entry.get("reason", ""))
        out[surface] = surface_map
    return out


@dataclass(frozen=True, slots=True)
class CoverageReport:
    gaps: tuple[CoverageGap, ...]
    stale_baseline: tuple[CoverageGap, ...]
    inventory_sizes: dict[Surface, int]

    @property
    def ok(self) -> bool:
        return not self.gaps


def collect_coverage(*, docs_text: str | None = None) -> CoverageReport:
    text = docs_text if docs_text is not None else _docs_corpus_text()
    baseline = _load_baseline()
    inventories = _all_inventories()

    gaps: list[CoverageGap] = []
    stale: list[CoverageGap] = []
    for surface, names in inventories.items():
        surface_baseline = baseline.get(surface, {})
        for name in names:
            documented = name in text
            baselined = name in surface_baseline
            if not documented and not baselined:
                gaps.append(CoverageGap(surface=surface, name=name))
            elif documented and baselined:
                stale.append(CoverageGap(surface=surface, name=name))

    return CoverageReport(
        gaps=tuple(gaps),
        stale_baseline=tuple(stale),
        inventory_sizes={surface: len(names) for surface, names in inventories.items()},
    )


def _format_report(report: CoverageReport) -> str:
    if report.ok:
        lines = ["docs-coverage: every public CLI command, MCP tool, config key, and stable route is reachable"]
        if report.stale_baseline:
            lines.append("")
            lines.append(f"Stale baseline entries (now documented, remove from {BASELINE_PATH.name}):")
            for gap in report.stale_baseline:
                lines.append(f"  {gap.surface}: {gap.name}")
        return "\n".join(lines)

    lines = [f"docs-coverage: {len(report.gaps)} undocumented public surface(s), not in the tracked baseline:"]
    by_surface: dict[str, list[str]] = {}
    for gap in report.gaps:
        by_surface.setdefault(gap.surface, []).append(gap.name)
    for surface, names in sorted(by_surface.items()):
        lines.append(f"  {surface} ({len(names)}):")
        for name in sorted(names):
            lines.append(f"    - {name}")
    lines.append("")
    lines.append(
        f"Fix: document the surface (README.md or docs/**/*.md), or, for pre-existing debt only, "
        f"add it to {BASELINE_PATH.relative_to(ROOT)} with a reason."
    )
    if report.stale_baseline:
        lines.append("")
        lines.append(f"Stale baseline entries (now documented, remove from {BASELINE_PATH.name}):")
        for gap in report.stale_baseline:
            lines.append(f"  {gap.surface}: {gap.name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = collect_coverage()

    if args.json:
        payload = {
            "ok": report.ok,
            "gaps": [{"surface": g.surface, "name": g.name} for g in report.gaps],
            "stale_baseline": [{"surface": g.surface, "name": g.name} for g in report.stale_baseline],
            "inventory_sizes": report.inventory_sizes,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
