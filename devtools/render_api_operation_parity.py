"""Render and verify the semantic-operation parity map for the Python API."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.api.operation_parity import (
    API_EXCLUSIONS,
    API_OPERATIONS,
    API_PARITY_AUTHORITY,
    ApiOperation,
    SurfaceBinding,
    facade_callable_records,
    validate_live_facade,
)

DEFAULT_OUTPUT_PATH = Path("docs/generated/api-operation-parity.json")
DEFAULT_LIBRARY_API_PATH = Path("docs/library-api.md")
SCHEMA_VERSION = 1
BEGIN_MARKER = "<!-- BEGIN GENERATED API OPERATION PARITY -->"
END_MARKER = "<!-- END GENERATED API OPERATION PARITY -->"


def _binding_payload(binding: SurfaceBinding) -> dict[str, object]:
    names = binding.names
    absence = binding.intentional_absence_authority
    return {"names": list(names), "intentional_absence_authority": absence}


def build_parity_payload() -> dict[str, object]:
    """Build the committed machine-readable operation matrix."""

    validate_live_facade()
    records = {
        binding: {"signature": signature, "async": is_async}
        for binding, signature, is_async in facade_callable_records()
    }
    operations: list[dict[str, object]] = []
    for operation in API_OPERATIONS:
        operations.append(
            {
                "operation_id": operation.operation_id,
                "section": operation.section,
                "summary": operation.summary,
                "route_class": operation.route_class,
                "python": [{"binding": binding, **records.get(binding, {})} for binding in operation.python_bindings],
                "cli": _binding_payload(operation.cli),
                "mcp": _binding_payload(operation.mcp),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": control_plane_command("render api-operation-parity"),
        "authority": {
            "operation_declarations": "polylogue/api/operation_parity.py",
            "drift_owner": API_PARITY_AUTHORITY,
            "facade": "polylogue.api.Polylogue",
            "documentation": "docs/library-api.md",
        },
        "operation_count": len(operations),
        "operations": operations,
        "exclusions": [asdict(exclusion) for exclusion in API_EXCLUSIONS],
    }


def render_parity_output() -> str:
    return json.dumps(build_parity_payload(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _display_binding(binding: SurfaceBinding) -> str:
    names = binding.names
    absence = binding.intentional_absence_authority
    return ", ".join(f"`{name}`" for name in names) if names else f"Intentional absence: `{absence}`"


def render_library_api_section() -> str:
    """Render the signature- and asyncness-aware section in library-api.md."""

    validate_live_facade()
    records = {binding: (signature, is_async) for binding, signature, is_async in facade_callable_records()}
    by_section: defaultdict[str, list[ApiOperation]] = defaultdict(list)
    for operation in API_OPERATIONS:
        by_section[operation.section].append(operation)

    lines = [
        BEGIN_MARKER,
        "",
        "## Generated facade operation index",
        "",
        "This reference is generated from `polylogue/api/operation_parity.py`. Each live public facade callable is bound to a stable semantic operation ID; exported data models and adapter helpers are listed as intentional exclusions in the committed [machine-readable matrix](generated/api-operation-parity.json).",
        "",
    ]
    for section, operations in by_section.items():
        lines.extend([f"### {section}", ""])
        for operation in operations:
            lines.extend(
                [
                    f"#### `{operation.operation_id}`",
                    "",
                    operation.summary,
                    "",
                    f"Route/tier class: `{operation.route_class}`. CLI: {_display_binding(operation.cli)}. MCP: {_display_binding(operation.mcp)}.",
                    "",
                ]
            )
            lines.extend(["| Python callable | Signature |", "|---|---|"])
            for binding in operation.python_bindings:
                if binding not in records:
                    lines.append(f"| `{binding}` | Constructed facade builder |")
                    continue
                signature, is_async = records[binding]
                prefix = "async " if is_async else ""
                lines.append(f"| `{binding}` | `{prefix}{signature}` |")
            lines.append("")
    lines.extend(["### Intentional exclusions", "", "| Export | Reason | Authority |", "|---|---|---|"])
    for exclusion in API_EXCLUSIONS:
        lines.append(f"| `{exclusion.binding}` | {exclusion.reason} | `{exclusion.authority}` |")
    lines.extend(["", END_MARKER])
    return "\n".join(lines)


def replace_library_api_section(current: str, section: str) -> str:
    """Replace only the generated block, rejecting an unmarked duplicate surface."""

    if current.count(BEGIN_MARKER) != 1 or current.count(END_MARKER) != 1:
        raise ValueError("docs/library-api.md must contain exactly one API parity marker pair")
    start = current.index(BEGIN_MARKER)
    end = current.index(END_MARKER, start) + len(END_MARKER)
    return current[:start] + section + current[end:]


def render_library_api_output(path: Path) -> str:
    return replace_library_api_section(path.read_text(encoding="utf-8"), render_library_api_section())


def validate_library_api_section(contents: str) -> None:
    """Reject a missing, mis-sectioned, stale-signature API reference."""

    if contents.count(BEGIN_MARKER) != 1 or contents.count(END_MARKER) != 1:
        raise ValueError("docs/library-api.md must contain exactly one API parity marker pair")
    start = contents.index(BEGIN_MARKER)
    end = contents.index(END_MARKER, start) + len(END_MARKER)
    if contents[start:end] != render_library_api_section():
        raise ValueError("docs/library-api.md generated API parity section does not match live facade signatures")


def _check(path: Path, expected: str, label: str) -> bool:
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    if actual == expected:
        print(f"render api-operation-parity: sync OK: {label}")
        return True
    print(f"render api-operation-parity: out of sync: {label}", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the semantic-operation parity matrix for the Python API.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--library-api-path", type=Path, default=DEFAULT_LIBRARY_API_PATH)
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero when the artifact or API reference is out of sync."
    )
    args = parser.parse_args(argv)
    try:
        parity = render_parity_output()
        library_api = render_library_api_output(args.library_api_path)
        validate_library_api_section(library_api)
    except (ValueError, OSError) as exc:
        print(f"render api-operation-parity: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return (
            0
            if _check(args.output_path, parity, str(args.output_path))
            and _check(args.library_api_path, library_api, str(args.library_api_path))
            else 1
        )
    write_if_changed(args.output_path, parity)
    write_if_changed(args.library_api_path, library_api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
