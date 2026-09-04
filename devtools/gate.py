"""Named invariant checks and the table the verifier runs them from.

A gate is one check with a PASS/FAIL verdict. ``devtools gate <name>`` runs one;
``devtools verify --quick`` runs every gate marked ``in_quick``.

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

from devtools.toolchain import venv_bin, venv_python

ROOT = Path(__file__).resolve().parents[1]

#: How a gate's argv is built. ``tool`` resolves the first token in the
#: checkout venv's bin directory; ``module`` runs ``python -m <module>``;
#: ``devtools`` runs ``python -m devtools <args>``.
GateKind = str


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    description: str
    kind: GateKind
    args: tuple[str, ...]
    label: str
    in_quick: bool = False
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
    """Use the checkout-local foreground checker owned by the verify task."""
    return [venv_bin("mypy", root=root)]


GATES: tuple[Gate, ...] = (
    Gate(
        "format",
        "Check source formatting with ruff.",
        "tool",
        ("ruff", "format", "--check", "polylogue/", "tests/", "devtools/"),
        label="gate format",
        in_quick=True,
    ),
    Gate(
        "lint",
        "Lint sources with ruff.",
        "tool",
        ("ruff", "check", "polylogue/", "tests/", "devtools/"),
        label="gate lint",
        in_quick=True,
    ),
    Gate(
        "mypy",
        "Type-check the repository.",
        "mypy",
        (),
        label="gate mypy",
        in_quick=True,
    ),
    Gate(
        "generated-surfaces",
        "Check every generated repository surface against its sources.",
        "devtools",
        ("render", "all", "--check"),
        label="gate generated-surfaces",
        in_quick=True,
    ),
    Gate(
        "layering",
        "Check inter-package imports against docs/plans/layering.yaml.",
        "module",
        ("devtools.verify_layering", "--json"),
        label="gate layering",
        in_quick=True,
    ),
    Gate(
        "patterns",
        "Enforce AST-shape defect-family rules with shrinking grandfathered baselines.",
        "module",
        ("devtools.verify_patterns", "--json"),
        label="gate patterns",
        in_quick=True,
    ),
    Gate(
        "doc-commands",
        "Validate executable documentation examples against live command inventories.",
        "module",
        ("devtools.verify_doc_commands",),
        label="gate doc-commands",
        in_quick=True,
    ),
    Gate(
        "schema-versioning",
        "Verify durable-tier migration and derived-tier rebuild boundaries.",
        "module",
        ("devtools.verify_schema_upgrade_lane",),
        label="gate schema-versioning",
        in_quick=True,
    ),
    Gate(
        "oracle-integrity",
        "Verify tests certify production-reachable code and never read ambient user paths.",
        "module",
        ("devtools.verify_oracle_integrity",),
        label="gate oracle-integrity",
        in_quick=True,
    ),
    Gate(
        "testmon-selection",
        "Prove a generated fixture uses a small affected selection with the managed worker default.",
        "module",
        ("devtools.verify_testmon_selection",),
        label="gate testmon-selection",
        in_quick=True,
    ),
    Gate(
        "consumer-reachability",
        "Report newly added modules, tables, and tools without production consumers.",
        "module",
        ("devtools.consumer_reachability", "--json"),
        label="gate consumer-reachability",
        in_quick=True,
        # Report-only: the incremental base/head diff it reasons over is not
        # stable enough across rebases to decide a verifier's exit code.
        blocking=False,
    ),
    Gate(
        "timestamp-doctrine",
        "Verify durable-tier DDL never stores a timestamp column as TEXT.",
        "module",
        ("devtools.verify_timestamp_doctrine",),
        label="gate timestamp-doctrine",
        in_quick=True,
    ),
    Gate(
        "schema-privacy",
        "Verify the committed-schema privacy registry.",
        "module",
        ("devtools.verify_schema_privacy",),
        label="gate schema-privacy",
        in_quick=True,
    ),
    Gate(
        "atlas",
        "Check atlas citation anchors and verification-commit freshness.",
        "module",
        ("devtools.verify_atlas",),
        label="gate atlas",
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
        "schema-inference-gate",
        "Run the read-only schema-inference prerequisite and persist a PASS/FAIL receipt.",
        "module",
        ("devtools.schema_inference_gate",),
        label="gate schema-inference-gate",
    ),
    Gate(
        "blob-conservation",
        "Verify both directions of blob/reference conservation without mutation.",
        "module",
        ("polylogue.maintenance.blob_conservation",),
        label="gate blob-conservation",
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
    Gate(
        "js",
        "Run the JavaScript test suites of the browser-extension and webui packages.",
        "module",
        ("devtools.verify_js_tests",),
        label="gate js",
    ),
)

GATES_BY_NAME: dict[str, Gate] = {gate.name: gate for gate in GATES}
GATE_NAMES: tuple[str, ...] = tuple(gate.name for gate in GATES)


def quick_gates() -> tuple[Gate, ...]:
    return tuple(gate for gate in GATES if gate.in_quick)


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
            marks = "quick" if gate.in_quick else ""
            if gate.in_quick and not gate.blocking:
                marks = "quick, report-only"
            suffix = f"  [{marks}]" if marks else ""
            print(f"{gate.name:<24} {gate.description}{suffix}")
        return 0 if args.list else 2
    gate = GATES_BY_NAME[args.name]
    completed = subprocess.run([*gate.command(), *passthrough], cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
