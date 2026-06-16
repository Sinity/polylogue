"""Validate the externally-presentable release gate definition (#1827)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GATE_DOC = ROOT / "docs" / "plans" / "release-readiness-gate.md"


@dataclass(frozen=True, slots=True)
class GateCommand:
    argv: tuple[str, ...]
    required: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "required": self.required, "reason": self.reason}


REQUIRED_COMMANDS: tuple[GateCommand, ...] = (
    GateCommand(("devtools", "release-readiness"), True, "release gate definition"),
    GateCommand(("devtools", "verify", "--quick"), True, "static/generated baseline"),
    GateCommand(("devtools", "verify", "--lab"), True, "verification-lab baseline"),
    GateCommand(("devtools", "build-package"), True, "wheel/sdist/Nix package smoke"),
    GateCommand(("devtools", "render-pages"), True, "documentation site build"),
    GateCommand(("devtools", "verify-doc-commands"), True, "README/docs command examples"),
)

FOCUSED_COMMANDS: tuple[GateCommand, ...] = (
    GateCommand(
        ("devtools", "test", "tests/unit/cli/test_query_verbs_runtime.py"),
        False,
        "command/read surface changed",
    ),
    GateCommand(
        ("devtools", "test", "tests/unit/storage/test_blackboard_facade.py"),
        False,
        "user assertion/KV surface changed",
    ),
    GateCommand(("devtools", "lab-scenario", "verify-baselines"), False, "showcase/demo baselines changed"),
    GateCommand(("nix", "flake", "check"), False, "packaging, Nix, or dependency metadata changed"),
)

REQUIRED_DOC_HEADINGS: tuple[str, ...] = (
    "## Gate Rule",
    "## Required Local Commands",
    "## Automated Gate Matrix",
    "## Manual Release Review",
    "## Current Status",
    "## Release PR Body Requirements",
)

REQUIRED_RELEASE_BODY_FIELDS: tuple[str, ...] = (
    "- Command floor:",
    "- Machine output:",
    "- README/demo:",
    "- Import/demo fixture:",
    "- Recovery/digest:",
    "- Web/API scope:",
    "- Packaging:",
    "- Known caveats scoped out:",
)


def build_report(*, gate_doc: Path = GATE_DOC) -> dict[str, Any]:
    """Return a JSON-serializable release-gate definition report."""
    from devtools.command_catalog import COMMANDS

    errors: list[str] = []
    text = gate_doc.read_text(encoding="utf-8") if gate_doc.exists() else ""
    if not text:
        errors.append(f"missing gate document: {gate_doc}")

    for heading in REQUIRED_DOC_HEADINGS:
        if heading not in text:
            errors.append(f"gate document missing heading: {heading}")
    for field in REQUIRED_RELEASE_BODY_FIELDS:
        if field not in text:
            errors.append(f"release PR template missing field: {field}")

    for command in (*REQUIRED_COMMANDS, *FOCUSED_COMMANDS):
        command_text = " ".join(command.argv)
        if command.required and command_text not in text:
            errors.append(f"required command missing from gate document: {command_text}")
        if command.argv[0] == "devtools" and command.argv[1] not in COMMANDS:
            errors.append(f"unknown devtools command in release gate: {command_text}")

    if "Satisfied:" not in text:
        errors.append("gate document missing satisfied release-status list")
    if "Still blocking external release claims:" not in text:
        errors.append("gate document missing blocking release-status list")

    return {
        "ok": not errors,
        "gate_doc": str(gate_doc.relative_to(ROOT) if gate_doc.is_relative_to(ROOT) else gate_doc),
        "required_commands": [command.to_dict() for command in REQUIRED_COMMANDS],
        "focused_commands": [command.to_dict() for command in FOCUSED_COMMANDS],
        "required_release_body_fields": list(REQUIRED_RELEASE_BODY_FIELDS),
        "errors": errors,
    }


def _print_human(report: dict[str, Any]) -> None:
    status = "ok" if report["ok"] else "FAIL"
    print(f"release-readiness: {status}")
    print(f"gate doc: {report['gate_doc']}")
    print("required commands:")
    for command in report["required_commands"]:
        print(f"  {' '.join(command['argv'])}  # {command['reason']}")
    print("focused commands:")
    for command in report["focused_commands"]:
        print(f"  {' '.join(command['argv'])}  # {command['reason']}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the gate-definition report as JSON.")
    args = parser.parse_args(argv)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
