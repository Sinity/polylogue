"""Structured production-route reachability checks for proof tests.

The check is deliberately based on Python's import and call structure. A test
may name the exact production symbol it exercises, but that symbol must also
be reachable from the declared production entrypoint. This prevents a test
that directly calls an orphaned helper from certifying a production route.
"""

from __future__ import annotations

import ast
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProductionSeamSpec:
    """Machine-readable contract between one test and a production route."""

    test_path: str
    test_function: str
    production_entrypoint: str
    tested_symbols: tuple[str, ...]
    required_symbols: tuple[str, ...] = ()
    production_namespace: str = "polylogue"

    def to_dict(self) -> dict[str, object]:
        return {
            "test_path": self.test_path,
            "test_function": self.test_function,
            "production_entrypoint": self.production_entrypoint,
            "tested_symbols": list(self.tested_symbols),
            "required_symbols": list(self.required_symbols),
            "production_namespace": self.production_namespace,
        }


@dataclass(frozen=True, slots=True)
class ProductionReachabilityViolation:
    """One structured failure from a production seam check."""

    code: str
    symbol: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "symbol": self.symbol}


@dataclass(frozen=True, slots=True)
class ProductionReachabilityReport:
    """Machine-readable result for one :class:`ProductionSeamSpec`."""

    spec: ProductionSeamSpec
    violations: tuple[ProductionReachabilityViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "ok": self.ok,
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class _FunctionNode:
    qualified_name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    module: str


@dataclass(frozen=True, slots=True)
class _ParsedModule:
    name: str
    tree: ast.Module
    path: Path


class _CallGraph:
    def __init__(self, modules: Iterable[_ParsedModule]) -> None:
        self.nodes: dict[str, _FunctionNode] = {}
        self.edges: dict[str, frozenset[str]] = {}
        parsed = tuple(modules)
        for module in parsed:
            self._index_functions(module)
        for module in parsed:
            self._index_edges(module)

    def _index_functions(self, module: _ParsedModule) -> None:
        for statement in module.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified_name = f"{module.name}.{statement.name}"
                self.nodes[qualified_name] = _FunctionNode(qualified_name, statement, module.name)
            elif isinstance(statement, ast.ClassDef):
                for member in statement.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualified_name = f"{module.name}.{statement.name}.{member.name}"
                        self.nodes[qualified_name] = _FunctionNode(qualified_name, member, module.name)

    def _index_edges(self, module: _ParsedModule) -> None:
        module_imports = _imports_from_nodes(module.tree.body, module.name)
        local_functions = {
            name: f"{module.name}.{name}"
            for name in (
                node.name for node in module.tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        }
        for function in tuple(node for node in self.nodes.values() if node.module == module.name):
            bindings = {**module_imports, **local_functions, **_imports_from_nodes(function.node.body, module.name)}
            targets: set[str] = set()
            for call in ast.walk(function.node):
                if not isinstance(call, ast.Call):
                    continue
                references = [call.func, *call.args, *(keyword.value for keyword in call.keywords)]
                for reference in references:
                    for expression in ast.walk(reference):
                        target = _resolve_call_target(expression, bindings, self.nodes)
                        if target is not None:
                            targets.add(target)
            self.edges[function.qualified_name] = frozenset(targets)

    def reachable_from(self, root: str) -> frozenset[str]:
        seen: set[str] = set()
        pending = deque([root])
        while pending:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self.edges.get(current, ()))
        return frozenset(seen)


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root)
    relative = relative.parent if relative.name == "__init__.py" else relative.with_suffix("")
    return ".".join(relative.parts)


def _source_files(roots: Iterable[Path]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            paths.add(root)
        elif root.is_dir():
            paths.update(path for path in root.rglob("*.py") if path.is_file())
    return tuple(sorted(paths))


def _parse_modules(source_root: Path, roots: Iterable[Path]) -> tuple[_ParsedModule, ...]:
    modules: list[_ParsedModule] = []
    for path in _source_files(roots):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"cannot parse {path}: {exc}") from exc
        modules.append(_ParsedModule(_module_name(path, source_root), tree, path))
    return tuple(modules)


def _resolve_relative_module(module: str, imported: str | None, level: int) -> str:
    if level == 0:
        return imported or ""
    base = module.split(".")[:-level]
    return ".".join((*base, *(imported.split(".") if imported else ())))


def _imports_from_nodes(nodes: Iterable[ast.AST], module: str) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_relative_module(module, node.module, node.level)
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                bindings[local_name] = f"{imported_module}.{alias.name}"
    return bindings


def _attribute_parts(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_parts(node.value)
        return None if parent is None else (*parent, node.attr)
    return None


def _resolve_call_target(function: ast.AST, bindings: dict[str, str], nodes: dict[str, _FunctionNode]) -> str | None:
    parts = _attribute_parts(function)
    if parts is None or not parts:
        return None
    bound = bindings.get(parts[0])
    if bound is None:
        return None
    target = ".".join((bound, *parts[1:]))
    if target in nodes:
        return target
    if bound in nodes and len(parts) == 2:
        method = f"{bound}.{parts[1]}"
        if method in nodes:
            return method
    return None


def _test_function_name(spec: ProductionSeamSpec, source_root: Path) -> str:
    test_path = Path(spec.test_path)
    absolute_path = test_path if test_path.is_absolute() else source_root / test_path
    return f"{_module_name(absolute_path, source_root)}.{spec.test_function}"


@lru_cache(maxsize=8)
def _call_graph(source_root: Path, test_path: Path) -> _CallGraph:
    production_root = source_root / "polylogue"
    graph_roots = (production_root if production_root.is_dir() else source_root, test_path)
    return _CallGraph(_parse_modules(source_root, graph_roots))


def check_production_seam(spec: ProductionSeamSpec, *, source_root: Path) -> ProductionReachabilityReport:
    """Check a test's direct call and its production entrypoint's call graph."""

    test_path = Path(spec.test_path)
    absolute_test_path = test_path if test_path.is_absolute() else source_root / test_path
    graph = _call_graph(source_root.resolve(), absolute_test_path.resolve())
    violations: list[ProductionReachabilityViolation] = []
    entrypoint = spec.production_entrypoint
    test_function = _test_function_name(spec, source_root)
    if not entrypoint.startswith(f"{spec.production_namespace}."):
        violations.append(ProductionReachabilityViolation("entrypoint_outside_production", entrypoint))
    if entrypoint not in graph.nodes:
        violations.append(ProductionReachabilityViolation("missing_production_entrypoint", entrypoint))
    if test_function not in graph.nodes:
        violations.append(ProductionReachabilityViolation("missing_test_function", test_function))

    test_targets = graph.edges.get(test_function, frozenset())
    reachable = graph.reachable_from(entrypoint)
    for symbol in spec.tested_symbols:
        if symbol not in graph.nodes:
            violations.append(ProductionReachabilityViolation("missing_tested_symbol", symbol))
        elif symbol not in test_targets:
            violations.append(ProductionReachabilityViolation("test_symbol_not_called", symbol))
        elif symbol not in reachable:
            violations.append(ProductionReachabilityViolation("tested_symbol_unreachable", symbol))
    for symbol in spec.required_symbols:
        if symbol not in graph.nodes:
            violations.append(ProductionReachabilityViolation("missing_required_symbol", symbol))
        elif symbol not in reachable:
            violations.append(ProductionReachabilityViolation("required_symbol_unreachable", symbol))
    return ProductionReachabilityReport(spec, tuple(violations))


def assert_production_seam(spec: ProductionSeamSpec, *, source_root: Path) -> None:
    """Raise with the structured report when a seam contract is violated."""

    report = check_production_seam(spec, source_root=source_root)
    if not report.ok:
        raise AssertionError(report.to_json())


__all__ = [
    "ProductionReachabilityReport",
    "ProductionReachabilityViolation",
    "ProductionSeamSpec",
    "assert_production_seam",
    "check_production_seam",
]
