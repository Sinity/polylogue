"""Render the executable MCP algebra inventory and compatibility map."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.mcp.declarations.models import MCPRole, MCPTransactionDeclaration
from polylogue.mcp.declarations.registry import (
    MCP_TOOL_DECLARATIONS,
    PRIVILEGED_ALGEBRA,
    TARGET_DEFAULT_READ_ALGEBRA,
    TARGET_PROMPTS,
    TARGET_RESOURCES,
)

DEFAULT_OUTPUT_PATH = Path("docs/generated/mcp-equivalence.json")
SCHEMA_VERSION = 1


def _transaction_payload(item: MCPTransactionDeclaration) -> dict[str, object]:
    return {
        "name": item.name,
        "verb": item.verb.value,
        "minimum_role": item.minimum_role,
        "object_kinds": list(item.object_kinds),
        "result_semantics": [value.value for value in item.result_semantics],
        "purpose": item.purpose,
        "migration_owner": item.migration_owner,
    }


def build_equivalence_payload() -> dict[str, object]:
    """Return the stable generated map consumed by drift checks and operators."""

    role_counts = Counter(declaration.minimum_role for declaration in MCP_TOOL_DECLARATIONS)
    read_retirements = sorted(
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.retirement_owner == "polylogue-t46.8.2"
    )
    privileged_retirements = sorted(
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.retirement_owner == "polylogue-t46.8.3"
    )
    bound_python = sorted(
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.python_parity.binding is not None
    )
    governed_absences = sorted(
        declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.python_parity.binding is None
    )

    role_order: tuple[MCPRole, ...] = ("read", "write", "review", "admin")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": control_plane_command("render mcp-equivalence"),
        "authority": {
            "declarations": "polylogue/mcp/declarations/registry.py",
            "live_registration": "polylogue/mcp/declarations/adapter.py",
            "independent_name_baseline": "tests/infra/mcp.py::MCP_TOOL_NAME_BASELINE",
            "independent_output_baseline": "tests/unit/mcp/test_envelope_contracts.py::TOOL_CONTRACT",
            "migration_authority": {
                "read": "polylogue-t46.8.2",
                "privileged": "polylogue-t46.8.3",
                "python_parity": "polylogue-s1kr",
            },
        },
        "compatibility_surface": {
            "tool_count": len(MCP_TOOL_DECLARATIONS),
            "minimum_role_counts": {role: role_counts.get(role, 0) for role in role_order},
            "python_binding_count": len(bound_python),
            "governed_python_absence_count": len(governed_absences),
            "tool_names": [declaration.name for declaration in MCP_TOOL_DECLARATIONS],
        },
        "target_algebra": {
            "default_read_transaction_count": len(TARGET_DEFAULT_READ_ALGEBRA),
            "default_read_transactions": [_transaction_payload(item) for item in TARGET_DEFAULT_READ_ALGEBRA],
            "privileged_transactions": [_transaction_payload(item) for item in PRIVILEGED_ALGEBRA],
            "resources": [asdict(item) for item in TARGET_RESOURCES],
            "prompts": [asdict(item) for item in TARGET_PROMPTS],
        },
        "migration_groups": {
            "polylogue-t46.8.2": read_retirements,
            "polylogue-t46.8.3": privileged_retirements,
        },
        "python_parity": {
            "bound_tools": bound_python,
            "governed_absences": governed_absences,
        },
        "tools": [declaration.to_dict() for declaration in MCP_TOOL_DECLARATIONS],
    }


def render_output() -> str:
    return json.dumps(build_equivalence_payload(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the generated MCP algebra equivalence map.")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"target JSON artifact (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--check", action="store_true", help="Exit non-zero when the artifact is out of sync.")
    args = parser.parse_args(argv)
    output_path = Path(args.output_path)
    rendered = render_output()

    if args.check:
        current = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if current != rendered:
            print("render mcp-equivalence: out of sync:", file=sys.stderr)
            print(f"  - {output_path}", file=sys.stderr)
            print(
                f"render mcp-equivalence: run: {control_plane_command('render mcp-equivalence')}",
                file=sys.stderr,
            )
            return 1
        print("render mcp-equivalence: sync OK")
        return 0

    write_if_changed(output_path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
