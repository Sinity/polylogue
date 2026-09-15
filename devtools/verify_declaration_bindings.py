"""Resolve every declared binding in the live declaration registries.

A declaration that names a handler, owner path, output target, or example that
no longer exists used to fail far downstream, as an opaque registration or
discovery error. This gate resolves those bindings directly and reports one
actionable line per break: the owning path, the code, and the declaration's own
repair command.

Anti-vacuity: rename or delete a declared handler function, point a declaration
at a file that does not exist, or register a declaration with no example,
output, or completeness edge, and this exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.declarations import DeclarationRegistryProtocol
from polylogue.declarations.diagnostics import diagnose_registry, format_diagnostic

ROOT = _get_root()


def _mcp() -> DeclarationRegistryProtocol:
    from polylogue.mcp.declarations.registry import MCP_KERNEL_REGISTRY

    return MCP_KERNEL_REGISTRY


def _daemon() -> DeclarationRegistryProtocol:
    from polylogue.daemon.route_contracts import DAEMON_ROUTE_REGISTRY

    return DAEMON_ROUTE_REGISTRY


#: ``structural`` says whether the registry's domain has already declared the
#: examples and completeness edges ``validate_declaration`` requires. Both
#: registries are enforced fully. The structural check reads the declaration
#: only, so it cannot tell a real request shape from an invented one; the
#: daemon-route examples are additionally replayed against the production
#: handlers by ``TestDeclaredRouteExamples`` in
#: ``tests/unit/daemon/test_web_reader.py``.
REGISTRIES: dict[str, tuple[Callable[[], DeclarationRegistryProtocol], bool]] = {
    "mcp": (_mcp, True),
    "daemon-route": (_daemon, True),
}


def run(*, root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """Return actionable diagnostic lines per registry, deterministically."""

    return {
        name: tuple(
            format_diagnostic(item) for item in diagnose_registry(factory(), root=root, include_structural=structural)
        )
        for name, (factory, structural) in sorted(REGISTRIES.items())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve declared bindings in the live declaration registries.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    report = run()
    failures = sum(len(lines) for lines in report.values())
    if args.json:
        print(json.dumps({name: list(lines) for name, lines in report.items()}, indent=2, sort_keys=True))
    else:
        for name, lines in report.items():
            if not lines:
                print(f"declaration-bindings: {name}: OK")
                continue
            for line in lines:
                print(f"declaration-bindings: {name}: {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
