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
from collections.abc import Callable, Iterable
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
    # Fixture names that establish this seam's hermetic filesystem boundary.
    # Keeping them in the contract prevents a proof test from silently
    # changing its isolation setup (for example, from ``workspace_env`` to a
    # raw ambient path).
    fixture_boundary: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "test_path": self.test_path,
            "test_function": self.test_function,
            "production_entrypoint": self.production_entrypoint,
            "tested_symbols": list(self.tested_symbols),
            "required_symbols": list(self.required_symbols),
            "production_namespace": self.production_namespace,
            "fixture_boundary": list(self.fixture_boundary),
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
        self.add_modules(modules)

    def add_modules(self, modules: Iterable[_ParsedModule]) -> None:
        parsed = tuple(modules)
        self.module_names = {module.name for module in parsed}
        for module in parsed:
            self._index_functions(module)
        self._index_prefixes()
        for module in parsed:
            self._index_edges(module)

    def _index_prefixes(self) -> None:
        """Map every dotted prefix to the node names beneath it.

        Resolution asks repeatedly whether any node lives under a candidate
        name. Answering that by scanning every node is quadratic in the graph:
        it cost 1.2 billion string comparisons on this repository. The prefixes
        are known once the nodes are, so the question becomes a lookup.
        """
        descendants: dict[str, set[str]] = {}
        for name in self.nodes:
            parts = name.split(".")
            for end in range(1, len(parts)):
                descendants.setdefault(".".join(parts[:end]), set()).add(name)
        self._descendants = descendants

    def descendants_of(self, prefix: str) -> frozenset[str]:
        return frozenset(self._descendants.get(prefix, ()))

    def has_descendants(self, prefix: str) -> bool:
        return prefix in self._descendants

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
        module_imports = _imports_from_nodes(
            module.tree.body, module.name, is_package=module.path.name == "__init__.py"
        )

        def imported_targets(bindings: dict[str, str], *, expand_nodes: bool) -> set[str]:
            targets: set[str] = set()
            for imported in bindings.values():
                parts = imported.split(".")
                for end in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:end])
                    if candidate in self.module_names:
                        targets.add(candidate)
                    if expand_nodes and (candidate in self.nodes or self.has_descendants(candidate)):
                        targets.add(candidate)
                        targets.update(self.descendants_of(candidate))
            return targets

        # Package roots are declared production entrypoints. Follow their
        # imports so a route exposed through a facade package is not mistaken
        # for an orphan merely because the facade's callable is a class
        # method. This remains fail-closed for callable seams: imported code
        # still needs an actual call edge from the selected production
        # function in ``check_production_seam``.
        self.edges[module.name] = frozenset(imported_targets(module_imports, expand_nodes=True))
        local_functions = {
            name: f"{module.name}.{name}"
            for name in (
                node.name for node in module.tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        }
        local_classes = {
            node.name: f"{module.name}.{node.name}" for node in module.tree.body if isinstance(node, ast.ClassDef)
        }
        for function in tuple(node for node in self.nodes.values() if node.module == module.name):
            bindings = {
                **module_imports,
                **local_functions,
                **local_classes,
                **_imports_from_nodes(function.node.body, module.name, is_package=module.path.name == "__init__.py"),
            }
            shadowed = _shadowed_names(function.node)
            for name in shadowed:
                bindings.pop(name, None)
            qualified_parts = function.qualified_name.split(".")
            if len(qualified_parts) >= 3:
                bindings["self"] = ".".join(qualified_parts[:-1])
            # A local import executes as part of the production callable. Its
            # module is therefore a real route edge even when the imported
            # object is a class or the call is hidden behind a constructor.
            # Keep this production-only: test seam edges remain call-based so
            # importing a symbol cannot satisfy ``test_symbol_not_called``.
            local_imports = _imports_in_function(
                function.node, module.name, is_package=module.path.name == "__init__.py"
            )
            targets: set[str] = (
                imported_targets(local_imports, expand_nodes=False) if module.name.startswith("polylogue.") else set()
            )
            for call in _calls_in_function(function.node):
                target = _resolve_call_target(call.func, bindings, self.nodes, self.has_descendants)
                if target is not None:
                    targets.add(target)
            self.edges[function.qualified_name] = frozenset(targets)

    def reachable_from(self, root: str) -> frozenset[str]:
        seen: set[str] = set()
        # Consumer-reachability declares entrypoints by module (for example a
        # console script's ``module:function`` target is normalized to its
        # module). Seed that module's top-level functions so traversal follows
        # the callable production routes exported by the declared entrypoint.
        pending = deque(
            [
                root,
                *(
                    node.qualified_name
                    for node in self.nodes.values()
                    if node.module == root and node.qualified_name.count(".") == root.count(".") + 1
                ),
            ]
        )
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


def _resolve_relative_module(module: str, imported: str | None, level: int, *, is_package: bool = False) -> str:
    if level == 0:
        return imported or ""
    parts = module.split(".")
    base = parts[: len(parts) - level + 1] if is_package else parts[:-level]
    return ".".join((*base, *(imported.split(".") if imported else ())))


def _imports_from_nodes(nodes: Iterable[ast.AST], module: str, *, is_package: bool = False) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_relative_module(module, node.module, node.level, is_package=is_package)
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                bindings[local_name] = f"{imported_module}.{alias.name}"
    return bindings


def _imports_in_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef, module: str, *, is_package: bool = False
) -> dict[str, str]:
    """Return imports executed by a function, including conditional imports."""

    class ImportScanner(ast.NodeVisitor):
        def __init__(self) -> None:
            self.nodes: list[ast.AST] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            del node

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            del node

        def visit_Import(self, node: ast.Import) -> None:
            self.nodes.append(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            self.nodes.append(node)

    scanner = ImportScanner()
    scanner.visit(function)
    return _imports_from_nodes(scanner.nodes, module, is_package=is_package)


def _attribute_parts(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_parts(node.value)
        return None if parent is None else (*parent, node.attr)
    return None


def _resolve_call_target(
    function: ast.AST,
    bindings: dict[str, str],
    nodes: dict[str, _FunctionNode],
    has_descendants: Callable[[str], bool] | None = None,
) -> str | None:
    if has_descendants is None:

        def has_descendants(prefix: str) -> bool:
            return any(name.startswith(f"{prefix}.") for name in nodes)

    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Call):
        constructor = _resolve_call_target(function.value.func, bindings, nodes, has_descendants)
        if constructor is not None:
            candidate = f"{constructor}.{function.attr}"
            if candidate in nodes:
                return candidate
    parts = _attribute_parts(function)
    if parts is None or not parts:
        return None
    bound = bindings.get(parts[0])
    if bound is None:
        return None
    target = ".".join((bound, *parts[1:]))
    if target in nodes:
        return target
    if bound in nodes or has_descendants(bound):
        return bound
    return None


def _test_function_name(spec: ProductionSeamSpec, source_root: Path) -> str:
    test_path = Path(spec.test_path)
    absolute_path = test_path if test_path.is_absolute() else source_root / test_path
    return f"{_module_name(absolute_path, source_root)}.{spec.test_function}"


def _test_fixture_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    return frozenset(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )


def _shadowed_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {argument.arg for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
    if function.args.vararg is not None:
        names.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        names.add(function.args.kwarg.arg)

    class ShadowScanner(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                self.generic_visit(node)
            else:
                names.add(node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is function:
                self.generic_visit(node)
            else:
                names.add(node.name)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            names.add(node.name)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                names.add(node.id)

    scanner = ShadowScanner()
    scanner.visit(function)
    return names


class _CallScanner(ast.NodeVisitor):
    def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.root = root
        self.calls: list[ast.Call] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            # A function's decorators, annotations, defaults, and type
            # parameters execute in the defining scope, not when the
            # function body runs.  The reachability contract describes the
            # production route executed by the callable, so scan only its
            # statements.  Nested callable bodies are intentionally skipped.
            for statement in node.body:
                self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)


def _calls_in_function(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.Call, ...]:
    scanner = _CallScanner(function)
    scanner.visit(function)
    return tuple(scanner.calls)


@lru_cache(maxsize=8)
def _production_modules(source_root: Path, signature: tuple[tuple[str, int, int], ...]) -> tuple[_ParsedModule, ...]:
    del signature
    production_root = source_root / "polylogue"
    return _parse_modules(source_root, (production_root if production_root.is_dir() else source_root,))


def _source_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in _source_files((root,)))


def _call_graph(source_root: Path, test_path: Path) -> _CallGraph:
    production_root = source_root / "polylogue"
    production_modules = _production_modules(source_root, _source_signature(production_root))
    production_graph = _CallGraph(production_modules)
    graph = _CallGraph(())
    graph.nodes = dict(production_graph.nodes)
    graph.edges = dict(production_graph.edges)
    graph.add_modules(_parse_modules(source_root, (test_path,)))
    return graph


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
    elif spec.fixture_boundary:
        test_node = graph.nodes[test_function]
        declared = _test_fixture_names(test_node.node)
        for fixture_name in spec.fixture_boundary:
            if not fixture_name.isidentifier():
                violations.append(ProductionReachabilityViolation("fixture_boundary_invalid", fixture_name))
            elif fixture_name not in declared:
                violations.append(ProductionReachabilityViolation("fixture_boundary_not_declared", fixture_name))

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
