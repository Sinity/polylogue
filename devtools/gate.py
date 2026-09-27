"""Named invariant checks and the table the verifier runs them from.

A gate is one check with a PASS/FAIL verdict. ``devtools gate <name>`` runs one.
Each gate declares the tier that runs it:

- ``quick``: every ``devtools verify --quick``, which is the hosted
  ``quick-gate`` check on every pull request. A gate is here when it is cheap
  or regularly catches real defects before merge.
- ``periodic``: ``devtools verify --periodic`` (the scheduled static run) and
  ``devtools verify --all``. A gate is here when its invariant rarely regresses
  and a regression caught on the next scheduled run is cheap to fix. Over
  968 quick runs from 2026-08-31 to 2026-09-27, none of these gates caught
  more than one real defect.
- ``manual``: only ``devtools gate <name>``.

A gate marked ``blocking=False`` reports its verdict and is recorded in the
receipt, but does not decide the verifier's exit code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from devtools.toolchain import venv_bin, venv_python

ROOT = Path(__file__).resolve().parents[1]

#: How a gate's argv is built. ``tool`` resolves the first token in the
#: checkout venv's bin directory; ``module`` runs ``python -m <module>``;
#: ``devtools`` runs ``python -m devtools <args>``.
GateKind = str
GateTier = Literal["quick", "periodic", "manual"]


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    description: str
    kind: GateKind
    args: tuple[str, ...]
    label: str
    tier: GateTier = "manual"
    blocking: bool = True

    def command(self, *, root: Path = ROOT) -> list[str]:
        if self.kind == "mypy":
            return mypy_command(root=root)
        if self.kind == "tool":
            return [venv_bin(self.args[0], root=root), *self.args[1:]]
        if self.kind == "module":
            return [venv_python(root=root), "-m", *self.args]
        if self.kind == "devtools":
            return [venv_python(root=root), "-m", "devtools", *self.args]
        raise ValueError(f"unknown gate kind {self.kind!r}")


def mypy_command(*, root: Path = ROOT) -> list[str]:
    """Run the checker on the checkout's cache, seeded from its siblings.

    Batch lanes have independent trees but type-check the same repository. The
    wrapper gives each checkout its own incremental cache, seeded from the one
    its siblings last published, so warm lanes check side by side; only a cold
    repository's first scan is serialized (``devtools/mypy_gate.py``).
    """
    return [venv_python(root=root), "-m", "devtools.mypy_gate"]


GATES: tuple[Gate, ...] = (
    Gate(
        "format",
        "Check source formatting with ruff.",
        "tool",
        ("ruff", "format", "--check", "polylogue/", "tests/", "devtools/"),
        label="gate format",
        tier="quick",
    ),
    Gate(
        "lint",
        "Lint sources with ruff.",
        "tool",
        ("ruff", "check", "polylogue/", "tests/", "devtools/"),
        label="gate lint",
        tier="quick",
    ),
    Gate(
        "mypy",
        "Type-check the repository.",
        "mypy",
        (),
        label="gate mypy",
        tier="quick",
    ),
    Gate(
        "generated-surfaces",
        "Check every generated repository surface against its sources.",
        "devtools",
        ("render", "all", "--check"),
        label="gate generated-surfaces",
        tier="quick",
    ),
    Gate(
        "layering",
        "Check inter-package imports against docs/plans/layering.yaml.",
        "module",
        ("devtools.verify_layering", "--json"),
        label="gate layering",
        tier="quick",
    ),
    Gate(
        "rebuild-routes",
        "Census every route reaching a derived-tier rebuild entrypoint against docs/plans/rebuild-route-census.yaml.",
        "module",
        ("devtools.verify_rebuild_routes", "--json"),
        label="gate rebuild-routes",
        tier="periodic",
    ),
    Gate(
        "controlled-read",
        "Census every direct ArchiveStore.open_existing against docs/plans/controlled-read-census.yaml.",
        "module",
        ("devtools.verify_controlled_read", "--json"),
        label="gate controlled-read",
        tier="periodic",
    ),
    Gate(
        "patterns",
        "Enforce AST-shape defect-family rules with shrinking grandfathered baselines.",
        "module",
        ("devtools.verify_patterns", "--json"),
        label="gate patterns",
        tier="quick",
    ),
    Gate(
        "api-parity",
        "Check CLI/MCP/Python semantic-operation parity and docs/library-api.md against the live facade.",
        "module",
        ("devtools.verify_api_parity", "--check"),
        label="gate api-parity",
        tier="periodic",
    ),
    Gate(
        "declaration-bindings",
        "Resolve every declared handler, owner path, output, and example in the live declaration registries.",
        "module",
        ("devtools.verify_declaration_bindings",),
        label="gate declaration-bindings",
        tier="periodic",
    ),
    Gate(
        "doc-commands",
        "Validate executable documentation examples against live command inventories.",
        "module",
        ("devtools.verify_doc_commands",),
        label="gate doc-commands",
        tier="periodic",
    ),
    Gate(
        "schema-manifest",
        "Require durable archive schema changes to use a complete migration chain.",
        "module",
        ("devtools.verify_schema_manifest", "--check-evolution"),
        label="gate schema-manifest",
        tier="quick",
    ),
    Gate(
        "oracle-integrity",
        "Verify tests certify production-reachable code and never read ambient user paths.",
        "module",
        ("devtools.verify_oracle_integrity",),
        label="gate oracle-integrity",
        tier="periodic",
    ),
    Gate(
        "testmon-selection",
        "Prove a generated fixture uses a small affected selection with the managed worker default.",
        "module",
        ("devtools.verify_testmon_selection",),
        label="gate testmon-selection",
        tier="periodic",
    ),
    Gate(
        "consumer-reachability",
        "Report newly added modules, tables, and tools without production consumers.",
        "module",
        ("devtools.consumer_reachability", "--json"),
        label="gate consumer-reachability",
        # Manual: it reports on the diff against the merge base, which a
        # scheduled run on master does not have, and as a report-only step in
        # the quick tier it cost every run 13 s without deciding any of them.
        tier="manual",
        # Report-only: the incremental base/head diff it reasons over is not
        # stable enough across rebases to decide a verifier's exit code.
        blocking=False,
    ),
    Gate(
        "test-packages",
        "Verify every directory holding collectible test modules is a package.",
        "module",
        ("devtools.verify_test_packages",),
        label="gate test-packages",
        tier="quick",
    ),
    Gate(
        "root-topology",
        "Verify no non-kernel module sits at the polylogue/ package root.",
        "module",
        ("devtools.verify_root_topology",),
        label="gate root-topology",
        tier="quick",
    ),
    Gate(
        "timestamp-doctrine",
        "Verify durable-tier DDL never stores a timestamp column as TEXT.",
        "module",
        ("devtools.verify_timestamp_doctrine",),
        label="gate timestamp-doctrine",
        tier="quick",
    ),
    Gate(
        "durable-enum-checks",
        "Verify durable-tier DDL carries no enum-derived membership CHECK.",
        "module",
        ("devtools.verify_durable_enum_checks",),
        label="gate durable-enum-checks",
        tier="quick",
    ),
    Gate(
        "schema-privacy",
        "Verify the committed-schema privacy registry.",
        "module",
        ("devtools.verify_schema_privacy",),
        label="gate schema-privacy",
        tier="periodic",
    ),
    Gate(
        "schema-provider-identity",
        "Verify no committed schema package contains an element whose $id names another subject.",
        "module",
        ("devtools.verify_schema_provider_identity",),
        label="gate schema-provider-identity",
        tier="quick",
    ),
    Gate(
        "schema-closure",
        "Ratchet the derived schema identity closure: it may shrink, never grow.",
        "module",
        ("devtools.verify_schema_closure", "--json"),
        label="gate schema-closure",
        tier="quick",
    ),
    Gate(
        "test-collection",
        "Collect the declared test corpus without running test bodies.",
        "module",
        ("devtools.verify_test_collection", "--json"),
        label="gate test-collection",
        tier="manual",
    ),
    Gate(
        "schema-audit",
        "Run committed provider schema package quality checks.",
        "module",
        ("devtools.schema_audit",),
        label="gate schema-audit",
    ),
    Gate(
        "schema-roundtrip",
        "Verify committed provider schema packages reload and roundtrip cleanly.",
        "module",
        ("devtools.verify_schema_roundtrip",),
        label="gate schema-roundtrip",
    ),
    Gate(
        "population-coverage",
        "Verify every origin, detector route, and artifact kind in the source inventory is declared and witnessed.",
        "module",
        ("devtools.verify_population_coverage",),
        label="gate population-coverage",
        tier="quick",
    ),
    Gate(
        "agent-integration",
        "Verify manual compilation, parser examples, continuation, native delivery, and packaging.",
        "module",
        ("devtools.verify_agent_integration",),
        label="gate agent-integration",
    ),
    Gate(
        "distribution",
        "Verify wheel/sdist installed artifacts expose only supported runtime entrypoints.",
        "module",
        ("devtools.verify_distribution_surface",),
        label="gate distribution",
    ),
    Gate(
        "webui",
        "Run the declared typed WebUI generation, contract, unit, and build checks.",
        "module",
        ("devtools.verify_webui",),
        label="gate webui",
    ),
)

GATES_BY_NAME: dict[str, Gate] = {gate.name: gate for gate in GATES}
GATE_NAMES: tuple[str, ...] = tuple(gate.name for gate in GATES)


def quick_gates() -> tuple[Gate, ...]:
    """Gates of the ``quick`` tier."""
    return tuple(gate for gate in GATES if gate.tier == "quick")


def periodic_gates() -> tuple[Gate, ...]:
    """Every static gate a scheduled or complete run owes: ``quick`` plus ``periodic``."""
    return tuple(gate for gate in GATES if gate.tier in {"quick", "periodic"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools gate",
        description="Run one named invariant check.",
    )
    parser.add_argument("name", nargs="?", choices=GATE_NAMES, help="the gate to run")
    parser.add_argument("--list", action="store_true", help="list the declared gates and exit")
    args, passthrough = parser.parse_known_args(list(argv or []))
    if args.list or args.name is None:
        for gate in GATES:
            marks: str = "" if gate.tier == "manual" else gate.tier
            if marks and not gate.blocking:
                marks += ", report-only"
            suffix = f"  [{marks}]" if marks else ""
            print(f"{gate.name:<24} {gate.description}{suffix}")
        return 0 if args.list else 2
    gate = GATES_BY_NAME[args.name]
    completed = subprocess.run([*gate.command(), *passthrough], cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
