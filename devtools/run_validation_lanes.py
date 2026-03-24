"""Run named validation lanes for the remaining operator frontier.

Usage:
    python -m devtools.run_validation_lanes --list
    python -m devtools.run_validation_lanes --lane machine-contract
    python -m devtools.run_validation_lanes --lane frontier-local --dry-run
    python -m devtools.run_validation_lanes --lane archive-intelligence --dry-run
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class LaneConfig:
    """Configuration for a validation lane."""

    name: str
    description: str
    timeout_s: int
    command: list[str] | None = None
    sub_lanes: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        return bool(self.sub_lanes)


LANES: dict[str, LaneConfig] = {
    "machine-contract": LaneConfig(
        name="machine-contract",
        description="Root CLI JSON success/failure envelopes and runtime-health machine surfaces",
        timeout_s=180,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "machine_contract"],
    ),
    "query-routing": LaneConfig(
        name="query-routing",
        description="Query-first CLI route planning, integration, and streamed read-surface proofs",
        timeout_s=240,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "query_routing"],
    ),
    "semantic-stack": LaneConfig(
        name="semantic-stack",
        description="Unified harmonization, semantic facts/profile convergence, proof, and contract inventory coverage",
        timeout_s=360,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/unit/core/test_semantic_facts.py",
            "tests/unit/core/test_filters_schemas.py",
            "tests/unit/sources/test_unified_semantic_laws.py",
            "tests/integration/test_extraction_db.py",
            "tests/unit/core/test_semantic_proof.py",
            "tests/unit/cli/test_check.py",
            "tests/unit/showcase/test_qa_runner.py",
            "tests/unit/showcase/test_report.py",
            "tests/unit/core/test_conversation_semantics.py",
        ],
    ),
    "source-provider-fidelity": LaneConfig(
        name="source-provider-fidelity",
        description="Source traversal, Drive/runtime source boundaries, parser decoding, and provider-ingest fidelity",
        timeout_s=420,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/unit/sources/test_drive_ops.py",
            "tests/unit/sources/test_source_laws.py",
            "tests/unit/sources/test_acquisition_encoding.py",
            "tests/unit/storage/test_parse_tracking.py",
            "tests/unit/pipeline/test_ingestion_chaos.py",
            "tests/integration/test_security.py",
        ],
    ),
    "maintenance-control-plane": LaneConfig(
        name="maintenance-control-plane",
        description="Health, maintenance selection, cache/live provenance, publication maintenance summaries, and machine output",
        timeout_s=480,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/unit/core/test_health_core.py",
            "tests/unit/storage/test_fts5.py",
            "tests/unit/cli/test_check.py",
            "tests/unit/cli/test_source_selection_helpers.py",
            "tests/unit/cli/test_deterministic_output.py",
            "tests/unit/mcp/test_tool_contracts.py",
            "tests/unit/site/test_builder.py",
            "tests/unit/cli/test_site.py",
            "tests/integration/test_health.py",
        ],
    ),
    "archive-data-products": LaneConfig(
        name="archive-data-products",
        description="Durable archive products, external consumer contracts, product-aware grouped stats, and health/governance surfaces",
        timeout_s=600,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/unit/cli/test_products.py",
            "tests/unit/core/test_facade_api.py",
            "tests/unit/mcp/test_tool_contracts.py",
            "tests/unit/cli/test_query_exec.py",
            "tests/unit/cli/test_check.py",
            "tests/unit/cli/test_click_app.py",
            "tests/unit/core/test_health_core.py",
            "tests/integration/test_health.py",
        ],
    ),
    "retrieval-dogfood": LaneConfig(
        name="retrieval-dogfood",
        description="Action-aware query truth, grouped retrieval stats, archive health, and MCP retrieval payload coverage",
        timeout_s=480,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/unit/cli/test_query_exec.py",
            "tests/unit/cli/test_query_exec_laws.py",
            "tests/unit/storage/test_store_ops.py",
            "tests/unit/core/test_filters_props.py",
            "tests/unit/core/test_health_core.py",
            "tests/unit/mcp/test_tool_contracts.py",
            "tests/unit/cli/test_source_selection_helpers.py",
        ],
    ),
    "embeddings-coverage": LaneConfig(
        name="embeddings-coverage",
        description="Embedding coverage/readiness stats, health exposure, and embed command contracts",
        timeout_s=300,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/unit/cli/test_embed.py",
            "tests/unit/storage/test_embedding_stats.py",
            "tests/unit/core/test_health_core.py",
            "tests/unit/cli/test_source_selection_helpers.py",
            "tests/unit/mcp/test_tool_contracts.py",
        ],
    ),
    "schema-roundtrip": LaneConfig(
        name="schema-roundtrip",
        description="Synthetic schema-and-evidence roundtrip proof lane and operator/report contracts",
        timeout_s=600,
        command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "tests/integration/test_schema_evidence_roundtrip_lane.py",
            "tests/unit/cli/test_check.py",
            "tests/unit/showcase/test_qa_runner.py",
            "tests/unit/showcase/test_report.py",
        ],
    ),
    "tui": LaneConfig(
        name="tui",
        description="Textual Mission Control screens and interaction-state coverage",
        timeout_s=240,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "tui"],
    ),
    "chaos": LaneConfig(
        name="chaos",
        description="Hostility, interruption, and chronology integration coverage",
        timeout_s=900,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "chaos"],
    ),
    "scale-fast": LaneConfig(
        name="scale-fast",
        description="Fast storage scale budgets",
        timeout_s=120,
        command=[sys.executable, "-m", "devtools.run_scale_lanes", "--lane", "fast"],
    ),
    "scale-slow": LaneConfig(
        name="scale-slow",
        description="Slow local storage scale budgets",
        timeout_s=360,
        command=[sys.executable, "-m", "devtools.run_scale_lanes", "--lane", "slow"],
    ),
    "long-haul-small": LaneConfig(
        name="long-haul-small",
        description="Small reproducible benchmark/long-haul campaign",
        timeout_s=1800,
        command=[sys.executable, "-m", "devtools.run_campaign", "--scale", "small"],
    ),
    "live-exercises": LaneConfig(
        name="live-exercises",
        description="Operator-run live archive showcase/QA exercise lane",
        timeout_s=1800,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "qa",
            "--live",
            "--only",
            "exercises",
            "--tier",
            "0",
            "--json",
        ],
    ),
    "live-embed-stats": LaneConfig(
        name="live-embed-stats",
        description="Live archive embedding readiness/readiness JSON surface",
        timeout_s=120,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "embed",
            "--stats",
            "--json",
        ],
    ),
    "live-retrieval-dogfood": LaneConfig(
        name="live-retrieval-dogfood",
        description="Live archive action-aware grouped retrieval stats on a bounded semantic slice",
        timeout_s=180,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "--provider",
            "claude-code",
            "--since",
            "2026-01-01",
            "--stats-by",
            "action",
            "--format",
            "json",
            "--limit",
            "50",
        ],
    ),
    "live-products-status": LaneConfig(
        name="live-products-status",
        description="Live archive durable product readiness surface",
        timeout_s=180,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "products",
            "status",
            "--json",
        ],
    ),
    "live-products-tags": LaneConfig(
        name="live-products-tags",
        description="Live archive durable tag-rollup product surface",
        timeout_s=180,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "products",
            "tags",
            "--limit",
            "20",
            "--json",
        ],
    ),
    "live-project-stats": LaneConfig(
        name="live-project-stats",
        description="Live archive project-grouped stats over durable session products",
        timeout_s=180,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "--provider",
            "claude-code",
            "--since",
            "2026-01-01",
            "--stats-by",
            "project",
            "--format",
            "json",
            "--limit",
            "50",
        ],
    ),
    "live-health-json": LaneConfig(
        name="live-health-json",
        description="Live archive machine-readable health/proof surface",
        timeout_s=180,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "check",
            "--json",
        ],
    ),
    "live-maintenance-preview": LaneConfig(
        name="live-maintenance-preview",
        description="Live archive machine-readable maintenance preview for safe repairs and destructive cleanup",
        timeout_s=240,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "check",
            "--json",
            "--repair",
            "--cleanup",
            "--preview",
        ],
    ),
    "memory-budget": LaneConfig(
        name="memory-budget",
        description="Live archive grouped retrieval command under an explicit RSS budget",
        timeout_s=240,
        command=[
            sys.executable,
            "-m",
            "devtools.query_memory_budget",
            "--max-rss-mb",
            "1536",
            "--",
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "--provider",
            "claude-code",
            "--since",
            "2026-01-01",
            "--stats-by",
            "action",
            "--format",
            "json",
            "--limit",
            "50",
        ],
    ),
    "maintenance-memory-budget": LaneConfig(
        name="maintenance-memory-budget",
        description="Live archive maintenance preview under an explicit RSS budget",
        timeout_s=240,
        command=[
            sys.executable,
            "-m",
            "devtools.query_memory_budget",
            "--max-rss-mb",
            "1024",
            "--",
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "check",
            "--json",
            "--repair",
            "--cleanup",
            "--preview",
        ],
    ),
    "source-runtime-governance": LaneConfig(
        name="source-runtime-governance",
        description="Local source/provider fidelity plus runtime maintenance/control-plane convergence",
        timeout_s=1800,
        sub_lanes=("source-provider-fidelity", "maintenance-control-plane"),
    ),
    "live-archive-small": LaneConfig(
        name="live-archive-small",
        description="Bounded live archive retrieval/readiness/health dogfood lane",
        timeout_s=480,
        sub_lanes=("live-embed-stats", "live-retrieval-dogfood", "live-products-status", "live-health-json"),
    ),
    "live-products-small": LaneConfig(
        name="live-products-small",
        description="Bounded live archive durable-product and grouped-stats dogfood lane",
        timeout_s=480,
        sub_lanes=("live-products-status", "live-products-tags", "live-project-stats"),
    ),
    "live-governance-small": LaneConfig(
        name="live-governance-small",
        description="Bounded live archive governance lane for health, maintenance preview, and maintenance memory budget",
        timeout_s=720,
        sub_lanes=("live-health-json", "live-maintenance-preview", "maintenance-memory-budget"),
    ),
    "live-archive-slow": LaneConfig(
        name="live-archive-slow",
        description="Broader live archive dogfood lane including retrieval/readiness and live QA exercises",
        timeout_s=2400,
        sub_lanes=("live-archive-small", "live-exercises"),
    ),
    "archive-intelligence": LaneConfig(
        name="archive-intelligence",
        description="Local archive-intelligence closure lane for retrieval, embedding readiness, and schema roundtrip",
        timeout_s=1800,
        sub_lanes=("retrieval-dogfood", "embeddings-coverage", "schema-roundtrip"),
    ),
    "archive-data-products-live": LaneConfig(
        name="archive-data-products-live",
        description="Local durable-product contract lane plus bounded live archive product dogfooding",
        timeout_s=1800,
        sub_lanes=("archive-data-products", "live-products-small"),
    ),
    "runtime-substrate-contracts": LaneConfig(
        name="runtime-substrate-contracts",
        description="Local runtime-substrate closure lane across query, semantic proof, durable products, and maintenance control-plane contracts",
        timeout_s=2400,
        sub_lanes=(
            "query-routing",
            "semantic-stack",
            "maintenance-control-plane",
            "archive-data-products",
        ),
    ),
    "runtime-substrate-live": LaneConfig(
        name="runtime-substrate-live",
        description="Bounded live archive lane for runtime-substrate dogfooding, governance, and memory budgets",
        timeout_s=1800,
        sub_lanes=("live-archive-small", "live-governance-small", "memory-budget"),
    ),
    "runtime-substrate-hardening": LaneConfig(
        name="runtime-substrate-hardening",
        description="Full runtime-substrate validation lane covering local contracts plus bounded live archive checks",
        timeout_s=3600,
        sub_lanes=("runtime-substrate-contracts", "runtime-substrate-live"),
    ),
    "frontier-local": LaneConfig(
        name="frontier-local",
        description="Non-live local closure lane for machine/query/semantic/TUI/chaos validation",
        timeout_s=1500,
        sub_lanes=("machine-contract", "query-routing", "semantic-stack", "tui", "chaos"),
    ),
    "frontier-extended": LaneConfig(
        name="frontier-extended",
        description="Local closure lane plus fast scale and small long-haul campaign",
        timeout_s=3600,
        sub_lanes=("frontier-local", "scale-fast", "long-haul-small"),
    ),
}

VALID_LANES = frozenset(LANES)


def parse_lane(lane_name: str) -> LaneConfig:
    """Parse and validate a lane name."""
    if lane_name not in LANES:
        raise ValueError(
            f"Invalid lane: {lane_name!r}. Valid lanes: {', '.join(sorted(VALID_LANES))}"
        )
    return LANES[lane_name]


def build_lane_command(lane: LaneConfig) -> list[str]:
    """Build the concrete subprocess command for a non-composite lane."""
    if lane.command is None:
        raise ValueError(f"Lane {lane.name!r} is composite and has no direct command")
    return lane.command


def _print_lane(lane: LaneConfig, *, indent: str = "") -> None:
    print(f"{indent}{lane.name}: {lane.description}")
    if lane.is_composite:
        for child_name in lane.sub_lanes:
            _print_lane(parse_lane(child_name), indent=indent + "  ")
    else:
        print(f"{indent}  command: {' '.join(build_lane_command(lane))}")
        print(f"{indent}  timeout: {lane.timeout_s}s")


def run_lane(lane: LaneConfig) -> int:
    """Execute a validation lane."""
    if lane.is_composite:
        print(f"Validation lane: {lane.name} — {lane.description}")
        for child_name in lane.sub_lanes:
            exit_code = run_lane(parse_lane(child_name))
            if exit_code != 0:
                return exit_code
        return 0

    cmd = build_lane_command(lane)
    print(f"Validation lane: {lane.name} — {lane.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {lane.timeout_s}s")
    print()

    try:
        result = subprocess.run(cmd, timeout=lane.timeout_s)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"\nLane {lane.name!r} timed out after {lane.timeout_s}s")
        return 2


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lane",
        choices=sorted(VALID_LANES),
        help="Validation lane to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_lanes",
        help="List available lanes and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected lane command(s) without running them",
    )
    args = parser.parse_args(argv)

    if args.list_lanes:
        print("Available validation lanes:")
        for lane_name in sorted(VALID_LANES):
            lane = parse_lane(lane_name)
            print(f"  {lane.name}: {lane.description}")
        return 0

    if not args.lane:
        parser.error("--lane is required unless --list is used")

    lane = parse_lane(args.lane)
    if args.dry_run:
        _print_lane(lane)
        return 0

    return run_lane(lane)


if __name__ == "__main__":
    sys.exit(main())
