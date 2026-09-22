"""Census every route that reaches a derived-tier rebuild entrypoint.

Gate classification: **blocking architectural boundary check**.

The campaign invariant is that a derived tier is rebuilt only by ordinary
daemon convergence, and the recovery design in ``CLAUDE.md`` is reconvergence
through the production daemon.  That invariant was asserted in prose and cited
``devtools gate layering`` as its check.  The citation was false, and the hole
was measured on 2026-09-21: an orchestration-only rebuild route added under
``polylogue/maintenance/`` and awaiting
``polylogue.pipeline.services.indexing.rebuild_index`` entirely outside
``DaemonConverger`` left the layering gate green.  The layering gate's writer
census answers a different question -- *which file executes its own DML* -- so
a route that orchestrates existing tier writers is invisible to it, and so is
new DML added inside an already-censused module.

This gate answers the question the invariant actually needs: *who can reach a
rebuild entrypoint at all*.  It is the same shape as the writer-module census
it sits beside: an exact, checked-in declaration that may shrink but never
grows without a deliberate edit.

What is observed
----------------

``docs/plans/rebuild-route-census.yaml`` declares the rebuild entrypoints.  For
each one the gate builds a module-and-function call graph over ``polylogue/``
and reports two populations:

``direct_caller``
    a function that references a declared entrypoint itself.

``entry_root``
    a function in the upward closure of the entrypoints that no other closure
    member calls -- the point at which a rebuild becomes reachable from outside
    the rebuild stack.  Roots are what make the census a *repo-wide* criterion
    rather than a per-file assertion: a new module that reaches a rebuild
    through three intermediate helpers still surfaces as a new root, and an
    orchestration-only module that executes no SQL of its own surfaces as both
    a new direct caller and a new root.

Every function in either population must carry a declaration with a ``route``
from :data:`ROUTE_VOCABULARY` and a non-empty ``reason``.  Fixture-only seeding
is therefore *recorded*, not special-cased in code.

Why AST, not an import graph
----------------------------

``devtools/verify_layering.py`` documents that grimp silently drops modules in
namespace packages with no ``__init__.py`` -- 85 of them in this checkout,
including every parser under ``sources/parsers/`` -- and so uses AST as the
primary extractor with grimp only as a bounded cross-check.  A caller census
built on an import graph would report those modules as unable to reach
anything, which is exactly the vacuity this gate exists to remove.  The whole
analysis here is AST over the file tree, so a namespace-package module is
inspected like any other.

Known limits, stated so nobody mistakes the gate for more than it is: a call
reached through ``getattr``, ``importlib``, or a value stored in a container is
not resolved.  Registry dispatch is not resolved either, which is why an
actuator invoked by a registry appears as a root rather than as a callee of its
dispatcher -- the declaration records the dispatch evidence in its ``reason``.

Usage:
  devtools gate rebuild-routes
  devtools gate rebuild-routes --json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from devtools import repo_root as _get_root
from devtools.required_gate import evidence_gate_result

#: The declaration this gate reads. Policy (which functions are rebuild
#: entrypoints) and census (who reaches them, and under what route) live
#: together because both are deliberate edits reviewed as one change.
DECLARATION_PATH = "docs/plans/rebuild-route-census.yaml"

#: Every route classification a census entry may claim, with what claiming it
#: asserts. ``daemon_convergence`` is the only token the gate proves on its
#: own; the others record a human judgement that review has to accept.
ROUTE_VOCABULARY: dict[str, str] = {
    "daemon_convergence": (
        "reached from the daemon convergence stage list; the gate proves this "
        "structurally and refuses the claim it cannot prove"
    ),
    "daemon_operation": (
        "reached only as a declared daemon operation handler, inside the daemon process, under its authority and audit"
    ),
    "fixture_seeding": ("demo or fixture seeding of a throwaway archive, not a route over an operator archive"),
    "unreached": (
        "defined and exported but reached by no production caller at this head; "
        "a retirement candidate, recorded so it cannot quietly acquire one"
    ),
}

MODULE_SCOPE = "<module>"


@dataclass(frozen=True)
class CensusEntry:
    """One declared route into the rebuild stack."""

    function: str
    kinds: tuple[str, ...]
    entrypoints: tuple[str, ...]
    route: str
    reason: str


@dataclass
class CallGraph:
    """Function-grain call/reference edges over one package tree."""

    #: qualified name -> repository-relative file that defines it
    files: dict[str, str] = field(default_factory=dict)
    #: every function/class qualified name defined in the tree
    defined: set[str] = field(default_factory=set)
    #: callee qualified name -> callers that reference it
    callers: dict[str, set[str]] = field(default_factory=dict)
    #: caller qualified name -> callees it references
    callees: dict[str, set[str]] = field(default_factory=dict)
    #: module qualified name -> repository-relative file
    modules: dict[str, str] = field(default_factory=dict)

    def file_for(self, qualname: str) -> str:
        """Best-effort source file for a qualified name.

        A nested function or a ``<module>`` scope has no entry of its own, so
        the longest defining module prefix is used.
        """
        if qualname in self.files:
            return self.files[qualname]
        parts = qualname.split(".")
        for stop in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:stop])
            if candidate in self.modules:
                return self.modules[candidate]
        return qualname


def _module_name(path: Path, *, repo_root: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _scope_walk(tree: ast.Module) -> list[tuple[ast.AST, tuple[str, ...], str | None]]:
    """Every node paired with its enclosing scope path and enclosing class."""
    found: list[tuple[ast.AST, tuple[str, ...], str | None]] = []

    def walk(node: ast.AST, stack: tuple[str, ...], enclosing_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, (*stack, child.name), child.name)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*stack, child.name), enclosing_class)
            else:
                found.append((child, stack, enclosing_class))
                walk(child, stack, enclosing_class)

    walk(tree, (), None)
    return found


def _definitions(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()

    def walk(node: ast.AST, stack: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                names.add(".".join((module, *stack, child.name)))
                walk(child, (*stack, child.name))

    walk(tree, ())
    return names


#: One lexical scope's ``(symbol aliases, module aliases)``.
ScopeImports = tuple[dict[str, str], dict[str, str]]


def _import_maps(tree: ast.Module, *, module: str, is_package_init: bool) -> dict[tuple[str, ...], ScopeImports]:
    """Return each lexical scope's ``(symbol aliases, module aliases)``.

    Imports are collected from the whole tree, not only module scope, because
    a rebuild route is routinely imported inside the function that uses it
    (``polylogue/operations/mutation_actuators.py`` does exactly that). They
    are kept PER SCOPE rather than merged, because merging lets one function's
    local import overwrite another's binding of the same name: a later
    ``def audit(): from polylogue.report import rebuild_index`` silently
    replaced the module-wide mapping, and the function that really calls the
    rebuild entrypoint disappeared from the graph -- so the census reported no
    new route for a call that bypasses daemon convergence.

    The mapping is keyed by the scope path :func:`_scope_walk` reports, so a
    reference resolves against its own scope first and then each enclosing one.
    """
    scopes: dict[tuple[str, ...], ScopeImports] = {(): ({}, {})}

    def record(node: ast.AST, scope: tuple[str, ...]) -> None:
        symbols, modules = scopes.setdefault(scope, ({}, {}))
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = module.split(".")
                keep = len(parts) - node.level + (1 if is_package_init else 0)
                prefix = ".".join(parts[: max(0, keep)])
                base = f"{prefix}.{node.module}" if node.module else prefix
            else:
                base = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                symbols[alias.asname or alias.name] = f"{base}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    modules[head] = head

    def walk(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*scope, child.name))
            else:
                if isinstance(child, ast.Import | ast.ImportFrom):
                    record(child, scope)
                walk(child, scope)

    walk(tree, ())
    return scopes


def _merged_imports(scopes: Mapping[tuple[str, ...], ScopeImports]) -> ScopeImports:
    """Every scope's bindings in one map, for chasing module re-exports.

    A re-export chain is a property of the module, not of a reference site, so
    it reads the union. Order is by scope depth so an outer binding wins over
    a deeper one with the same name.
    """
    symbols: dict[str, str] = {}
    modules: dict[str, str] = {}
    for scope in sorted(scopes, key=len, reverse=True):
        scope_symbols, scope_modules = scopes[scope]
        symbols.update(scope_symbols)
        modules.update(scope_modules)
    return symbols, modules


def _visible_imports(scopes: Mapping[tuple[str, ...], ScopeImports], scope: tuple[str, ...]) -> ScopeImports:
    """The bindings visible at ``scope``: its own, then each enclosing one."""
    symbols: dict[str, str] = {}
    modules: dict[str, str] = {}
    for depth in range(len(scope) + 1):
        enclosing = scopes.get(scope[:depth])
        if enclosing is None:
            continue
        symbols.update(enclosing[0])
        modules.update(enclosing[1])
    return symbols, modules


def _dotted_name(node: ast.AST) -> str | None:
    """``a.b.c`` for an attribute chain rooted at a plain name, else ``None``."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def build_call_graph(package_root: Path, *, repo_root: Path) -> CallGraph:
    """Build the function-grain reference graph for one package tree."""
    graph = CallGraph()
    trees: dict[str, tuple[Path, ast.Module]] = {}
    for path in sorted(package_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module = _module_name(path, repo_root=repo_root)
        relative = path.relative_to(repo_root).as_posix()
        trees[module] = (path, tree)
        graph.modules[module] = relative
        for name in _definitions(tree, module):
            graph.defined.add(name)
            graph.files[name] = relative

    scoped_aliases: dict[str, dict[tuple[str, ...], ScopeImports]] = {
        module: _import_maps(tree, module=module, is_package_init=path.name == "__init__.py")
        for module, (path, tree) in trees.items()
    }
    aliases: dict[str, ScopeImports] = {module: _merged_imports(scopes) for module, scopes in scoped_aliases.items()}

    def resolve(qualname: str) -> str:
        """Follow re-exports until the name lands on a definition.

        ``polylogue/cli/commands/demo.py`` imports ``seed_demo_archive`` from
        ``polylogue.demo``, which imports it from ``polylogue.demo.seed``.
        Without this chase the CLI would look like it reaches nothing.
        """
        seen: set[str] = set()
        while qualname and qualname not in graph.defined and qualname not in seen:
            seen.add(qualname)
            owner, _, leaf = qualname.rpartition(".")
            symbols = aliases.get(owner, ({}, {}))[0]
            if leaf not in symbols:
                break
            qualname = symbols[leaf]
        return qualname

    for module, (_path, tree) in trees.items():
        scopes = scoped_aliases[module]
        visible: dict[tuple[str, ...], ScopeImports] = {}
        for node, stack, enclosing_class in _scope_walk(tree):
            if stack not in visible:
                visible[stack] = _visible_imports(scopes, stack)
            symbols, module_aliases = visible[stack]
            target: str | None = None
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in symbols:
                    target = symbols[node.id]
                elif f"{module}.{node.id}" in graph.defined:
                    target = f"{module}.{node.id}"
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name) and node.value.id == "self" and enclosing_class is not None:
                    target = f"{module}.{enclosing_class}.{node.attr}"
                else:
                    # ``polylogue.pipeline.services.indexing.rebuild_index(...)``
                    # after ``import polylogue.pipeline.services.indexing`` is
                    # ordinary syntax, and a chain of ``ast.Attribute`` nodes.
                    # Reading only the one whose value is an ``ast.Name`` saw
                    # the intermediate module and never the function, so a new
                    # non-daemon rebuild route written this way left the
                    # blocking gate green.
                    dotted = _dotted_name(node)
                    if dotted is not None:
                        head, _, rest = dotted.partition(".")
                        if head in module_aliases:
                            target = f"{module_aliases[head]}.{rest}" if rest else module_aliases[head]
                        elif head in symbols:
                            target = f"{symbols[head]}.{rest}" if rest else symbols[head]
            if target is None:
                continue
            target = resolve(target)
            if not target.startswith(f"{package_root.name}."):
                continue
            caller = ".".join((module, *stack)) if stack else f"{module}.{MODULE_SCOPE}"
            if target == caller:
                continue
            graph.callers.setdefault(target, set()).add(caller)
            graph.callees.setdefault(caller, set()).add(target)
    return graph


@dataclass(frozen=True)
class Observation:
    """What the tree says about reachability of the declared entrypoints."""

    #: function -> entrypoints it can reach (itself included for entrypoints)
    reaches: dict[str, frozenset[str]]
    direct_callers: frozenset[str]
    entry_roots: frozenset[str]
    unresolved_entrypoints: tuple[str, ...]
    daemon_reachable: frozenset[str]

    @property
    def population(self) -> frozenset[str]:
        return self.direct_callers | self.entry_roots


def observe(graph: CallGraph, *, entrypoints: Iterable[str], convergence_seeds: Iterable[str]) -> Observation:
    """Compute the rebuild-reaching closure, its direct callers, and its roots."""
    declared = tuple(dict.fromkeys(entrypoints))
    unresolved = tuple(name for name in declared if name not in graph.defined)
    live = [name for name in declared if name in graph.defined]

    reaches: dict[str, set[str]] = {name: {name} for name in live}
    queue: deque[str] = deque(live)
    while queue:
        node = queue.popleft()
        payload = frozenset(reaches[node])
        for caller in graph.callers.get(node, ()):
            current = reaches.setdefault(caller, set())
            if not payload <= current:
                current |= payload
                queue.append(caller)

    closure = set(reaches)
    direct = {caller for name in live for caller in graph.callers.get(name, ())}
    roots = {name for name in closure if not (graph.callers.get(name, set()) & closure)}

    forward: set[str] = set()
    pending = deque(seed for seed in convergence_seeds if seed in graph.defined)
    forward.update(pending)
    while pending:
        for callee in graph.callees.get(pending.popleft(), ()):
            if callee not in forward:
                forward.add(callee)
                pending.append(callee)

    return Observation(
        reaches={name: frozenset(value) for name, value in reaches.items()},
        direct_callers=frozenset(direct),
        entry_roots=frozenset(roots),
        unresolved_entrypoints=unresolved,
        daemon_reachable=frozenset(closure & forward),
    )


def _as_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


@dataclass(frozen=True)
class Declaration:
    entrypoints: tuple[str, ...]
    convergence_seeds: tuple[str, ...]
    package: str
    entries: dict[str, CensusEntry]
    malformed: tuple[str, ...]


def load_declaration(path: Path) -> Declaration:
    """Parse the census declaration, keeping malformed rows visible."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data: Mapping[str, object] = raw if isinstance(raw, dict) else {}
    entries: dict[str, CensusEntry] = {}
    malformed: list[str] = []
    raw_routes = data.get("routes")
    rows: list[object] = list(raw_routes) if isinstance(raw_routes, list) else []
    for index, item in enumerate(rows):
        if not isinstance(item, dict) or not isinstance(item.get("function"), str):
            malformed.append(f"routes[{index}]")
            continue
        function = str(item["function"])
        entries[function] = CensusEntry(
            function=function,
            kinds=tuple(sorted(_as_strings(item.get("kinds")))),
            entrypoints=tuple(sorted(_as_strings(item.get("entrypoints")))),
            route=str(item.get("route") or ""),
            reason=str(item.get("reason") or "").strip(),
        )
    return Declaration(
        entrypoints=_as_strings(data.get("entrypoints")),
        convergence_seeds=_as_strings(data.get("convergence_stage_seeds")),
        package=str(data.get("package") or "polylogue"),
        entries=entries,
        malformed=tuple(malformed),
    )


def _kinds_for(function: str, observation: Observation) -> tuple[str, ...]:
    kinds = []
    if function in observation.direct_callers:
        kinds.append("direct_caller")
    if function in observation.entry_roots:
        kinds.append("entry_root")
    return tuple(kinds)


def collect_violations(
    *, repo_root: Path, declaration_path: Path | None = None
) -> tuple[list[dict[str, object]], Observation, Declaration, CallGraph]:
    """Run the census and return every finding, with the evidence behind it."""
    path = declaration_path or (repo_root / DECLARATION_PATH)
    declaration = load_declaration(path)
    graph = build_call_graph(repo_root / declaration.package, repo_root=repo_root)
    observation = observe(
        graph,
        entrypoints=declaration.entrypoints,
        convergence_seeds=declaration.convergence_seeds,
    )

    violations: list[dict[str, object]] = []
    for name in declaration.malformed:
        violations.append({"function": name, "rule": "rebuild_census_row_malformed"})

    # A renamed or deleted entrypoint would silently empty the census and leave
    # the gate green over nothing. Refuse the declaration instead.
    for name in observation.unresolved_entrypoints:
        violations.append(
            {
                "function": name,
                "rule": "rebuild_entrypoint_unresolved",
                "detail": "declared entrypoint resolves to no definition under the package",
            }
        )

    observed = observation.population
    for function in sorted(observed - set(declaration.entries)):
        violations.append(
            {
                "function": function,
                "file": graph.file_for(function),
                "rule": "rebuild_route_undeclared",
                "kinds": list(_kinds_for(function, observation)),
                "entrypoints": sorted(observation.reaches.get(function, frozenset())),
            }
        )
    for function in sorted(set(declaration.entries) - observed):
        violations.append(
            {
                "function": function,
                "rule": "rebuild_route_census_stale",
                "detail": "declared route no longer reaches a rebuild entrypoint -- drop the entry",
            }
        )
    for function in sorted(observed & set(declaration.entries)):
        entry = declaration.entries[function]
        kinds = _kinds_for(function, observation)
        reached = tuple(sorted(observation.reaches.get(function, frozenset())))
        if entry.kinds != kinds:
            violations.append(
                {
                    "function": function,
                    "file": graph.file_for(function),
                    "rule": "rebuild_route_kind_drift",
                    "declared": list(entry.kinds),
                    "observed": list(kinds),
                }
            )
        if entry.entrypoints != reached:
            violations.append(
                {
                    "function": function,
                    "file": graph.file_for(function),
                    "rule": "rebuild_route_entrypoint_drift",
                    "declared": list(entry.entrypoints),
                    "observed": list(reached),
                }
            )
        if entry.route not in ROUTE_VOCABULARY:
            violations.append(
                {
                    "function": function,
                    "rule": "rebuild_route_unknown_classification",
                    "declared": entry.route,
                    "allowed": sorted(ROUTE_VOCABULARY),
                }
            )
        elif entry.route == "daemon_convergence" and function not in observation.daemon_reachable:
            # The token the whole invariant rests on is the one token the gate
            # refuses to take on trust.
            violations.append(
                {
                    "function": function,
                    "file": graph.file_for(function),
                    "rule": "rebuild_route_daemon_claim_unproven",
                    "detail": "not reachable from any declared convergence stage seed",
                }
            )
        if not entry.reason:
            violations.append({"function": function, "rule": "rebuild_route_reason_missing"})
    return violations, observation, declaration, graph


def _format_violation(violation: Mapping[str, object]) -> str:
    rule = str(violation.get("rule"))
    location = str(violation.get("file") or "")
    prefix = f"  {location}: " if location else "  "
    if rule == "rebuild_route_undeclared":
        return (
            f"{prefix}{violation['function']} reaches "
            f"{', '.join(_as_strings(violation.get('entrypoints')))} "
            f"as {'/'.join(_as_strings(violation.get('kinds')))} "
            f"({rule}; declare it in {DECLARATION_PATH} with a route and a reason)"
        )
    if rule in {"rebuild_route_kind_drift", "rebuild_route_entrypoint_drift"}:
        return (
            f"{prefix}{violation['function']}: {rule} declared="
            f"{violation.get('declared')} observed={violation.get('observed')}"
        )
    detail = violation.get("detail") or violation.get("declared") or ""
    suffix = f" ({detail})" if detail else ""
    return f"{prefix}{violation['function']}: {rule}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _get_root()
    declaration_path = repo_root / DECLARATION_PATH
    if not declaration_path.is_file():
        gate = evidence_gate_result(
            gate="rebuild-routes",
            executable=sys.executable,
            executable_available=True,
            required_count=1,
            inspected_count=0,
            missing_count=1,
            details=(str(declaration_path),),
        )
        if args.json:
            print(json.dumps({"ok": False, "required_gate": gate.to_payload()}, indent=2))
        else:
            print(f"error: {declaration_path} not found", file=sys.stderr)
        return 1

    violations, observation, declaration, graph = collect_violations(repo_root=repo_root)
    gate = evidence_gate_result(
        gate="rebuild-routes",
        executable=sys.executable,
        executable_available=True,
        required_count=len(declaration.entrypoints),
        inspected_count=len(declaration.entrypoints) - len(observation.unresolved_entrypoints),
        missing_count=len(observation.unresolved_entrypoints),
        semantic_violation_count=len([item for item in violations if item["rule"] != "rebuild_entrypoint_unresolved"]),
        details=tuple(str(item.get("function")) for item in violations[:8]),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "violations": violations,
                    "count": len(violations),
                    "entrypoints": list(declaration.entrypoints),
                    "closure_size": len(observation.reaches),
                    "direct_callers": sorted(observation.direct_callers),
                    "entry_roots": sorted(observation.entry_roots),
                    "daemon_stage_reachable": sorted(observation.daemon_reachable),
                    "required_gate": gate.to_payload(),
                },
                indent=2,
            )
        )
    else:
        for violation in violations:
            print(_format_violation(violation))
        if not violations:
            print(
                f"rebuild-routes: {len(declaration.entrypoints)} entrypoint(s), "
                f"{len(observation.reaches)} reaching function(s), "
                f"{len(observation.direct_callers)} direct caller(s) and "
                f"{len(observation.entry_roots)} entry root(s), all declared."
            )
            if not observation.daemon_reachable:
                print(
                    "  note: no reaching function is provably reached from the declared "
                    "convergence stage seeds -- see the declaration's route classifications."
                )
        del graph
    return 1 if violations or not gate.ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
