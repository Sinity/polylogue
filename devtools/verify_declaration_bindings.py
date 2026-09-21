"""Resolve every declared binding in the live declaration registries.

A declaration that names a handler, owner path, output target, or example that
no longer exists used to fail far downstream, as an opaque registration or
discovery error. This gate resolves those bindings directly and reports one
actionable line per break: the owning path, the code, and the declaration's own
repair command.

Anti-vacuity: rename or delete a declared handler function, point a declaration
at a file that does not exist, or register a declaration with no example,
output, or completeness edge, and this exits non-zero. For the maintenance
family's campaign-actuator retirements, record a retirement condition as met
while its actuator is still defined -- or delete an actuator whose condition is
not met -- and this exits non-zero naming that actuator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.declarations import DeclarationRegistryProtocol, Diagnostic
from polylogue.declarations.diagnostics import diagnose_registry, format_diagnostic

ROOT = _get_root()


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    """One live declaration registry and how much of it is enforced.

    ``structural`` says whether the registry's domain has already declared the
    examples and completeness edges ``validate_declaration`` requires. Every
    registry is enforced fully today. The structural check reads the
    declaration only, so it cannot tell a real request shape from an invented
    one; the daemon-route examples are additionally replayed against the
    production handlers by ``TestDeclaredRouteExamples`` in
    ``tests/unit/daemon/test_web_reader.py``.

    ``domain`` is the family's own validation hook, for bindings the shared
    kernel cannot resolve (a ``spec_attr``, an ``ArchiveStore`` executor
    method). Domain-specific validation stays with the domain; only its
    reporting shape is shared.
    """

    factory: Callable[[], DeclarationRegistryProtocol]
    structural: bool
    domain: Callable[[], tuple[Diagnostic, ...]] | None = None


def _mcp() -> DeclarationRegistryProtocol:
    from polylogue.mcp.declarations.registry import MCP_KERNEL_REGISTRY

    return MCP_KERNEL_REGISTRY


def _daemon() -> DeclarationRegistryProtocol:
    from polylogue.daemon.route_contracts import DAEMON_ROUTE_REGISTRY

    return DAEMON_ROUTE_REGISTRY


def _query() -> DeclarationRegistryProtocol:
    from polylogue.archive.query.declarations import QUERY_KERNEL_REGISTRY

    return QUERY_KERNEL_REGISTRY


def _marker() -> DeclarationRegistryProtocol:
    from polylogue.markers.declarations import MARKER_KERNEL_REGISTRY

    return MARKER_KERNEL_REGISTRY


def _maintenance() -> DeclarationRegistryProtocol:
    from polylogue.maintenance.declarations import MAINTENANCE_KERNEL_REGISTRY

    return MAINTENANCE_KERNEL_REGISTRY


def _query_domain_diagnostics() -> tuple[Diagnostic, ...]:
    """Resolve the query family's own domain bindings."""

    from polylogue.archive.query.declarations import query_binding_diagnostics

    return query_binding_diagnostics()


def _maintenance_domain_diagnostics() -> tuple[Diagnostic, ...]:
    """Resolve the maintenance family's campaign-actuator retirements.

    Two of the family's declared commands are backed by finite durable-source
    actuators that exist for one campaign. The family declares what each one
    serves and the condition that retires it; this resolves that declaration
    against the tree in both directions -- an actuator that outlived a met
    condition, and a declaration whose actuator vanished while the work it
    named is still owed.
    """

    from polylogue.maintenance.declarations import retirement_diagnostics

    return retirement_diagnostics(root=ROOT)


REGISTRIES: dict[str, _RegistryEntry] = {
    "mcp": _RegistryEntry(_mcp, structural=True),
    "daemon-route": _RegistryEntry(_daemon, structural=True),
    "query": _RegistryEntry(_query, structural=True, domain=_query_domain_diagnostics),
    "marker": _RegistryEntry(_marker, structural=True),
    "maintenance": _RegistryEntry(_maintenance, structural=True, domain=_maintenance_domain_diagnostics),
}


def run(*, root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """Return actionable diagnostic lines per registry, deterministically."""

    report: dict[str, tuple[str, ...]] = {}
    for name, entry in sorted(REGISTRIES.items()):
        diagnostics = list(diagnose_registry(entry.factory(), root=root, include_structural=entry.structural))
        if entry.domain is not None:
            diagnostics.extend(entry.domain())
        report[name] = tuple(
            format_diagnostic(item)
            for item in sorted(diagnostics, key=lambda item: (item.declaration_id, item.code, item.message))
        )
    return report


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
