"""Oracle-integrity lint: tests must certify live code, hermetically.

Two failure classes, one family -- "the test exercises something other than
what it claims" (polylogue-4v2d3, polylogue-jdesf):

``reachability``
    A test module whose ENTIRE production target set is unreachable from any
    production entrypoint certifies dead code. It stays green forever
    regardless of live behaviour, which is the opposite of what a test is for.

``hermeticity``
    A test that reads a real user/archive path (``~/.codex``, ``~/.claude``,
    ``/realm/state``, ``/realm/db``) is not testing the fixture it claims to
    test; it is reading whatever the developer's machine happens to contain.

Why a static lint rather than grep. Reachability by grep LIES in this
repository, and has already produced four recorded wrong deletions. Several
registries dispatch by string key or deferred import, so a symbol with no
inbound import edge may still be live:

* ``REPAIR_HANDLERS`` / ``PREVIEW_HANDLERS`` (``storage/repair.py``) and
  ``ARCHIVE_VERIFICATION_CHECKS`` (``maintenance/archive_verification.py``)
  hold DIRECT function references inside a dict/tuple literal, then dispatch
  through ``TABLE[runtime_name](...)``. The callee expression is an
  ``ast.Subscript``, which an AST call-graph walker cannot resolve.
* ``INSIGHT_REGISTRY`` (``insights/registry.py``) stores
  ``operations_method_name`` as a plain string resolved by ``getattr`` at call
  time -- no import edge exists at all.
* Click's ``_LazyCommand``/``_LazyGroup``
  (``cli/click_command_registration.py``) resolve subcommands through
  ``importlib.import_module`` + ``getattr`` on a COMPUTED dotted string, so
  ``polylogue/cli/commands/*`` has no static inbound edge from the root app.

This module therefore seeds the reachability closure from those registries
explicitly (:func:`_registry_roots`) in addition to the packaging entrypoints,
rather than pretending import edges are the whole graph. Four seeded classes,
each one a channel that produced a measured false "dead code" verdict:

* Click lazy commands (``cli/commands/*`` has no inbound edge at all).
* ``python -m``-invoked modules -- ``if __name__ == "__main__":`` is a real
  external entrypoint. ``security/precommit_scan`` (run from the pre-commit
  hook) and ``schemas/promotion_audit`` (run from ``devtools/verify.py``) were
  both reported dead without it.
* Ancestor packages -- importing ``pkg.sub`` executes ``pkg/__init__.py``.
  Omitting this reported 53 live parents, ``polylogue.core`` among them.
* Literal-container registries. Note the walk must RECURSE into call
  arguments: ``ARCHIVE_VERIFICATION_CHECKS`` is a tuple of ``_registry_spec(...)``
  calls, so a non-recursive pass saw only ``ast.Call`` nodes and contributed
  exactly zero roots.

A symbol re-exported through an unreachable facade is resolved per SYMBOL to
its defining module, so a pass-through never reads as dead -- deliberately not
per module, since "this module imports something reachable" would upgrade
nearly every dead module in the tree.

TYPE_CHECKING is reported, never guessed. An import under ``if TYPE_CHECKING:``
is erased at runtime, so it cannot make a symbol execute in production --
treating it as a reachability edge would hide genuinely dead code. But calling
such a symbol "unreachable" overstates what a static pass knows, because the
annotation may accompany a real runtime relationship expressed some other way.
So the verdict is three-valued (:class:`ReachabilityVerdict`): ``reachable``,
``type_only``, ``unreachable``. Only ``unreachable`` fails. ``type_only`` is
reported for a human to adjudicate and is exactly the ambiguity
polylogue-4v2d3's own notes recorded against
``storage/sqlite/queries/session_links.py``. A deletion sweep consuming this
lint's output must treat ``type_only`` as "needs a human", not as "safe".
"""

from __future__ import annotations

import ast
import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import tomllib

from devtools.production_reachability import (
    _imports_from_nodes,
    _module_name,
    _parse_modules,
)

PRODUCTION_PACKAGE = "polylogue"

#: Entrypoints that are not console scripts but are still ways in from outside
#: the process: the Python API facade, the MCP server, and the daemon runner.
#: Omitting these was the first calibration bug -- every ``tests/unit/api``
#: module reported "certifies dead code" because nothing in
#: ``[project.scripts]`` reaches ``polylogue.api`` directly.
PUBLIC_SURFACE_ROOTS: tuple[str, ...] = (
    "polylogue.api",
    "polylogue.mcp",
    "polylogue.daemon",
    "polylogue.hooks",
)

#: Ambient paths a hermetic test must never read. These are real user and live
#: archive locations; a test touching one is reading the developer's machine.
AMBIENT_PATH_MARKERS: tuple[str, ...] = (
    "~/.codex",
    "~/.claude",
    "~/.config/polylogue",
    "/realm/state",
    "/realm/db",
    "/realm/data",
)

#: Call expressions that resolve an ambient location at runtime. ``Path.home``
#: and ``expanduser`` are the two ways to reach ``$HOME`` without writing a
#: literal, so a literal-only scan would miss them entirely.
AMBIENT_CALL_MARKERS: tuple[str, ...] = (
    "Path.home",
    "os.path.expanduser",
    "os.path.expandvars",
    "pathlib.Path.home",
)


class ReachabilityVerdict(StrEnum):
    """Three-valued reachability outcome for one test module."""

    REACHABLE = "reachable"
    TYPE_ONLY = "type_only"
    UNREACHABLE = "unreachable"
    NO_TARGETS = "no_targets"


@dataclass(frozen=True, slots=True)
class OracleAllowlistEntry:
    """One structured exemption. A bare name is not accepted anywhere."""

    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class OracleFinding:
    """One machine-readable lint failure."""

    code: str
    path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class OracleIntegrityReport:
    findings: tuple[OracleFinding, ...]
    reachable_modules: int = 0
    type_only_modules: tuple[str, ...] = ()
    scanned_modules: int = 0
    roots: tuple[str, ...] = field(default=())
    baselined: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "scanned_modules": self.scanned_modules,
            "reachable_modules": self.reachable_modules,
            "type_only_modules": list(self.type_only_modules),
            "root_count": len(self.roots),
            "baselined": self.baselined,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# Reachability roots
# ---------------------------------------------------------------------------


def _packaging_entrypoints(source_root: Path) -> tuple[str, ...]:
    """Resolve ``[project.scripts]`` into ``module.attr`` reachability roots.

    These are the only true process entrypoints: everything a user can run
    starts at one of them. Reading them from ``pyproject.toml`` rather than
    hardcoding a list means a new console script cannot silently fall outside
    the closure.
    """
    manifest = source_root / "pyproject.toml"
    if not manifest.is_file():
        return ()
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    roots: set[str] = set()
    for target in scripts.values():
        if not isinstance(target, str) or ":" not in target:
            continue
        module, _, attribute = target.partition(":")
        if module.startswith(PRODUCTION_PACKAGE):
            roots.add(f"{module}.{attribute}")
    return tuple(sorted(roots))


def _main_module_roots(source_root: Path) -> tuple[str, ...]:
    """Modules invoked as ``python -m pkg.mod`` are entrypoints too.

    ``if __name__ == "__main__":`` is a real, externally-reachable entry that no
    import edge records. Two live cases were reported DEAD without this:
    ``security/precommit_scan`` (invoked from ``.beads-hooks/pre-commit``) and
    ``schemas/promotion_audit`` (invoked from ``devtools/verify.py`` -- three
    lines below where this very gate inserts itself). Both are exactly the
    census-incident class where a tool is live but has no importer.
    """
    production_root = source_root / PRODUCTION_PACKAGE
    roots: set[str] = set()
    for path in sorted(production_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Name):
                continue
            if test.left.id != "__name__":
                continue
            if any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators):
                roots.add(_module_name(path, source_root))
                break
    return tuple(sorted(roots))


def _lazy_attribute_map_roots(source_root: Path) -> tuple[str, ...]:
    """``polylogue/__init__.py``'s lazy ``__getattr__`` name -> module map.

    Another string registry: the package exposes attributes by importing a
    module named in a dict at access time, so the mapped modules have no static
    inbound edge.
    """
    init_path = source_root / PRODUCTION_PACKAGE / "__init__.py"
    if not init_path.is_file():
        return ()
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = node.value
            if (
                candidate.startswith(f"{PRODUCTION_PACKAGE}.")
                and (source_root / (candidate.replace(".", "/") + ".py")).is_file()
            ):
                roots.add(candidate)
    return tuple(sorted(roots))


def _literal_container_roots(modules: Iterable[ast.Module], module_names: Sequence[str]) -> tuple[str, ...]:
    """Names held as DIRECT references inside module-level dict/tuple literals.

    ``REPAIR_HANDLERS = {"empty_sessions": repair_empty_sessions, ...}`` makes
    ``repair_empty_sessions`` live, but the dispatch site is
    ``REPAIR_HANDLERS[name](...)`` -- an ``ast.Subscript`` callee the call-graph
    walker cannot resolve. Treating the literal's values as roots restores the
    edge without pretending to evaluate the subscript.
    """
    roots: set[str] = set()
    for tree, module_name in zip(modules, module_names, strict=True):
        bindings = _imports_from_nodes(tree.body, module_name)
        local = {
            node.name: f"{module_name}.{node.name}"
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None:
                continue
            for element in _walk_literal_references(value):
                resolved = local.get(element) or bindings.get(element)
                if resolved is not None and resolved.startswith(f"{PRODUCTION_PACKAGE}."):
                    roots.add(resolved)
    return tuple(sorted(roots))


def _walk_literal_references(value: ast.expr) -> tuple[str, ...]:
    """Bare names referenced anywhere inside a literal container.

    Recurses through nested containers AND call arguments, because the
    registry shape that matters is a TUPLE OF CALLS --
    ``ARCHIVE_VERIFICATION_CHECKS = (_registry_spec("name", "desc", _check_fn,
    ...), ...)``. A non-recursive pass saw only the ``ast.Call`` elements and
    never reached ``_check_fn``, so the registry contributed nothing.
    """
    names: list[str] = []
    for node in ast.walk(value):
        if isinstance(node, ast.Name):
            names.append(node.id)
    return tuple(names)


def _click_lazy_roots(source_root: Path) -> tuple[str, ...]:
    """Resolve Click's deferred ``(module, attr)`` pairs into roots.

    ``cli/commands/*`` has no static inbound edge: ``_LazyCommand._resolve``
    calls ``importlib.import_module`` on a computed string. Enumerating the
    command directory and applying the same ``_GROUP_ATTRS``/``_COMMAND_ATTRS``
    fallback the registration module uses reproduces that mapping without
    importing (and therefore executing) the CLI.
    """
    registration = source_root / PRODUCTION_PACKAGE / "cli" / "click_command_registration.py"
    commands_dir = source_root / PRODUCTION_PACKAGE / "cli" / "commands"
    if not registration.is_file() or not commands_dir.is_dir():
        return ()
    tree = ast.parse(registration.read_text(encoding="utf-8"))
    overrides: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        if statement.target.id not in {"_GROUP_ATTRS", "_COMMAND_ATTRS"}:
            continue
        if isinstance(statement.value, ast.Dict):
            for key, item in zip(statement.value.keys, statement.value.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(item, ast.Constant):
                    overrides[str(key.value)] = str(item.value)
    roots: set[str] = set()
    for path in sorted(commands_dir.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        stem = path.stem
        attribute = overrides.get(stem, f"{stem}_command")
        module = _module_name(path, source_root)
        roots.add(f"{module}.{attribute}")
    return tuple(sorted(roots))


def _registry_roots(source_root: Path, trees: Sequence[ast.Module], names: Sequence[str]) -> tuple[str, ...]:
    """Every root that static import edges alone would miss."""
    return tuple(
        sorted(
            {
                *_literal_container_roots(trees, names),
                *_click_lazy_roots(source_root),
                *_main_module_roots(source_root),
                *_lazy_attribute_map_roots(source_root),
            }
        )
    )


# ---------------------------------------------------------------------------
# Reachability sweep
# ---------------------------------------------------------------------------


def _is_type_checking_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def module_import_edges(
    tree: ast.Module, module_name: str, *, is_package: bool = False
) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(runtime_modules, type_checking_only_modules)`` this module imports.

    Known blind spot, deliberately not guessed at: a test that reaches
    production through ``importlib.import_module("polylogue.x")`` has no static
    import node, so its target set is empty and it classifies as
    ``NO_TARGETS`` -- silent, never a violation. Resolving a dynamic import
    string would mean evaluating an arbitrary expression, and a lint whose
    product is a deletion worklist must not guess. 6 test modules currently
    use ``importlib.import_module``; none are exempted, they are simply not
    audited for reachability.

    Reachability is computed at MODULE granularity, not symbol granularity.
    That is the coarser and therefore safer choice, and it is what
    polylogue-4v2d3 actually prescribes ("intersect grep-derived test imports
    with the layering/topology import graph devtools already builds").

    A symbol-level call graph was tried first and rejected: it under-approximates
    badly, because real reachability runs through class instantiation, decorator
    registration, and module-level execution that an AST call walker does not
    model. Measured on this corpus it produced 127 "certifies dead code"
    verdicts out of 1128 modules -- roughly one test in nine, which is not
    credible. A lint that cries dead code at that rate would either be disabled
    or, worse, believed by a deletion sweep.
    """
    runtime: set[str] = set()
    type_only: set[str] = set()
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for child in ast.walk(node):
                guarded.add(id(child))
    for node in ast.walk(tree):
        target = type_only if id(node) in guarded else runtime
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PRODUCTION_PACKAGE):
                    target.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = node.module or ""
            if node.level:
                # ``_module_name`` already strips ``__init__`` from a package,
                # so ``polylogue/ui/__init__.py`` is named ``polylogue.ui``.
                # For that module ``from .facade import X`` (level 1) means
                # ``polylogue.ui.facade``, NOT ``polylogue.facade`` -- the
                # package is its own level-1 anchor. Getting this wrong made
                # every module reached only through a package __init__ look
                # unreachable.
                parts = module_name.split(".")
                keep = len(parts) - node.level + 1 if is_package else len(parts) - node.level
                base = parts[: max(0, keep)]
                resolved = ".".join((*base, *(resolved.split(".") if resolved else ())))
            if resolved.startswith(PRODUCTION_PACKAGE):
                target.add(resolved)
                # ``from polylogue.pkg import mod`` also reaches ``pkg.mod``.
                for alias in node.names:
                    if alias.name != "*":
                        target.add(f"{resolved}.{alias.name}")
    return frozenset(runtime), frozenset(type_only - runtime)


def _reexport_bindings(tree: ast.Module, module_name: str, *, is_package: bool = False) -> dict[str, str]:
    """``{imported name: defining production module}`` for one module."""
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        resolved = node.module or ""
        if node.level:
            parts = module_name.split(".")
            keep = len(parts) - node.level + 1 if is_package else len(parts) - node.level
            resolved = ".".join((*parts[: max(0, keep)], *(resolved.split(".") if resolved else ())))
        if not resolved.startswith(PRODUCTION_PACKAGE):
            continue
        for alias in node.names:
            if alias.name != "*":
                bindings[alias.asname or alias.name] = resolved
    return bindings


@dataclass(frozen=True, slots=True)
class ImportClosure:
    """Module-level reachability over the production import graph."""

    reachable: frozenset[str]
    type_only_reachable: frozenset[str]
    known_modules: frozenset[str]
    roots: tuple[str, ...]

    #: ``module -> {names it re-exports}`` resolved to their defining modules.
    #: ``module -> {exported name -> defining module}`` for ``from X import n``
    #: bindings, so a facade's pass-through resolves per SYMBOL.
    reexport_bindings: dict[str, dict[str, str]] = field(default_factory=dict)

    def reexports_symbol_from_reachable(self, module: str, symbol: str) -> bool:
        """True when ``module`` binds ``symbol`` from a module that is live.

        Deliberately symbol-precise rather than module-precise. "This module
        imports something reachable" would upgrade almost every dead module in
        the tree -- nearly all import ``polylogue.core.enums`` -- trading the
        lint's false positives for false NEGATIVES, and a false negative here
        means a deletion sweep never hears about real dead code.
        """
        source = self.reexport_bindings.get(module, {}).get(symbol)
        return source is not None and source in self.reachable

    def status(self, module: str) -> ReachabilityVerdict:
        if module in self.reachable:
            return ReachabilityVerdict.REACHABLE
        if module in self.type_only_reachable:
            return ReachabilityVerdict.TYPE_ONLY
        return ReachabilityVerdict.UNREACHABLE


def build_import_closure(source_root: Path, roots: Sequence[str]) -> ImportClosure:
    """BFS the production import graph from the declared roots."""
    production_root = source_root / PRODUCTION_PACKAGE
    runtime_edges: dict[str, frozenset[str]] = {}
    type_edges: dict[str, frozenset[str]] = {}
    known: set[str] = set()
    reexport_bindings: dict[str, dict[str, str]] = {}
    for path in sorted(production_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        name = _module_name(path, source_root)
        known.add(name)
        runtime_edges[name], type_edges[name] = module_import_edges(tree, name, is_package=path.name == "__init__.py")
        reexport_bindings[name] = _reexport_bindings(tree, name, is_package=path.name == "__init__.py")

    def _walk(edge_map: dict[str, frozenset[str]], seeds: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        pending = deque(seeds)
        while pending:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            for candidate in edge_map.get(current, frozenset()):
                # An imported NAME resolves to its owning module when the name
                # itself is not a module (``from x.y import fn`` -> ``x.y``).
                pending.append(candidate if candidate in edge_map else candidate.rsplit(".", 1)[0])
        return seen

    def _with_ancestors(modules: set[str]) -> set[str]:
        """A reachable ``pkg.sub`` makes every ancestor ``__init__`` reachable.

        Importing ``polylogue.storage.sqlite.queries.x`` executes every
        ancestor package's ``__init__.py`` -- that is Python's import
        semantics, not an approximation. Omitting this reported 53 live parent
        packages (including ``polylogue.core``) as dead.
        """
        expanded = set(modules)
        for module in modules:
            parts = module.split(".")
            for depth in range(1, len(parts)):
                ancestor = ".".join(parts[:depth])
                if ancestor in known:
                    expanded.add(ancestor)
        return expanded

    seed_modules = {root.rsplit(".", 1)[0] if root not in runtime_edges else root for root in roots}
    reachable = _with_ancestors(_walk(runtime_edges, seed_modules & known))
    combined = {name: runtime_edges[name] | type_edges[name] for name in runtime_edges}
    type_reachable = _with_ancestors(_walk(combined, seed_modules & known)) - reachable
    return ImportClosure(
        reexport_bindings=reexport_bindings,
        reachable=frozenset(reachable),
        type_only_reachable=frozenset(type_reachable),
        known_modules=frozenset(known),
        roots=tuple(sorted(roots)),
    )


def _imported_symbol_names(tree: ast.Module) -> frozenset[str]:
    """Bare names a module imported from the production package."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PRODUCTION_PACKAGE):
            names.update(alias.name for alias in node.names if alias.name != "*")
    return frozenset(names)


def classify_test_module(
    tree: ast.Module,
    module_name: str,
    *,
    closure: ImportClosure,
) -> tuple[ReachabilityVerdict, tuple[str, ...]]:
    """Classify one test module's production target set."""
    runtime, type_only = module_import_edges(tree, module_name)
    targets = {
        candidate if candidate in closure.known_modules else candidate.rsplit(".", 1)[0]
        for candidate in (runtime | type_only)
    } & closure.known_modules
    if not targets:
        return ReachabilityVerdict.NO_TARGETS, ()

    statuses = {target: closure.status(target) for target in targets}
    # Root class (c): re-export facades. A test importing a LIVE symbol through
    # an unreachable facade module is not certifying dead code -- the facade is
    # a pass-through, and the definition it re-exports runs in production. So
    # an unreachable target is re-checked at symbol granularity: if the module
    # binds the imported name via ``from <reachable> import <name>``, the test
    # exercises live code and the verdict upgrades.
    for target, status in list(statuses.items()):
        if status is ReachabilityVerdict.REACHABLE:
            continue
        if any(closure.reexports_symbol_from_reachable(target, s) for s in _imported_symbol_names(tree)):
            statuses[target] = ReachabilityVerdict.REACHABLE
    if any(status is ReachabilityVerdict.REACHABLE for status in statuses.values()):
        return ReachabilityVerdict.REACHABLE, ()
    if any(status is ReachabilityVerdict.TYPE_ONLY for status in statuses.values()):
        return ReachabilityVerdict.TYPE_ONLY, tuple(
            sorted(name for name, status in statuses.items() if status is ReachabilityVerdict.TYPE_ONLY)
        )
    return ReachabilityVerdict.UNREACHABLE, tuple(sorted(statuses))


# ---------------------------------------------------------------------------
# Hermeticity sweep
# ---------------------------------------------------------------------------


def _docstring_nodes(tree: ast.Module) -> frozenset[int]:
    """Ids of every docstring constant, which are prose rather than reads."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            found.add(id(first.value))
    return frozenset(found)


def scan_hermeticity(path: Path, tree: ast.Module) -> tuple[OracleFinding, ...]:
    """Flag ambient user/archive reads in a test module.

    This enforces an ESCAPE, it does not re-implement isolation. The sanctioned
    boundary is ``tests/conftest.py``'s ``workspace_env`` /
    ``_clear_polylogue_env``, which point XDG and ``POLYLOGUE_ARCHIVE_ROOT`` at
    ``tmp_path``. A test that names an ambient path in source has stepped
    outside that boundary no matter what fixtures it also requests, so the
    check is a source scan rather than a runtime sandbox.
    """
    findings: list[OracleFinding] = []
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                # A module/class/function docstring naming ``~/.codex`` is
                # documentation, not a filesystem read. Flagging it made
                # ``tests/unit/core/test_paths.py`` a violation for explaining
                # the precedence ladder it tests.
                continue
            for marker in AMBIENT_PATH_MARKERS:
                if marker in node.value:
                    findings.append(
                        OracleFinding(
                            "ambient_path_literal",
                            str(path),
                            f"line {node.lineno}: reads ambient path {marker!r}",
                        )
                    )
        elif isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted is not None and dotted in AMBIENT_CALL_MARKERS:
                findings.append(
                    OracleFinding(
                        "ambient_path_call",
                        str(path),
                        f"line {node.lineno}: resolves an ambient location via {dotted}()",
                    )
                )
    return tuple(findings)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

#: Structured exemptions. Every entry carries a reason -- a bare path is not
#: accepted, so an exemption cannot accumulate without someone stating why.
REACHABILITY_ALLOWLIST: tuple[OracleAllowlistEntry, ...] = (
    OracleAllowlistEntry(
        "tests/infra",
        "Shared harness code (SessionBuilder, corpus fixtures). It is the test "
        "substrate itself, not a certification of production behaviour.",
    ),
    OracleAllowlistEntry(
        "tests/unit/devtools",
        "devtools is repo tooling, not the polylogue production package, so its "
        "tests have no polylogue entrypoint to be reachable from by construction.",
    ),
    OracleAllowlistEntry(
        "tests/fixtures",
        "Static fixture payloads and the synthetic modules the reachability checker itself parses.",
    ),
    OracleAllowlistEntry(
        "tests/conftest.py",
        "Fixture definitions; imports production symbols to build isolation, not to assert production behaviour.",
    ),
)

HERMETICITY_ALLOWLIST: tuple[OracleAllowlistEntry, ...] = (
    OracleAllowlistEntry(
        "tests/unit/devtools",
        "devtools tests assert on path-policy machinery itself (basetemp "
        "selection, checkout guard), so ambient path names are the subject "
        "under test rather than an escape.",
    ),
    OracleAllowlistEntry(
        "tests/unit/config",
        "Config resolution tests assert the XDG/archive-root ladder's own "
        "behaviour and must name the paths that ladder resolves.",
    ),
)


def _allowlisted(path: Path, source_root: Path, allowlist: Sequence[OracleAllowlistEntry]) -> bool:
    try:
        relative = path.relative_to(source_root).as_posix()
    except ValueError:
        relative = path.as_posix()
    return any(relative == entry.path or relative.startswith(f"{entry.path}/") for entry in allowlist)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


#: Findings that already existed when the lint was introduced. This is a
#: RATCHET, exactly like ``docs/plans/layering-surface-baseline.json``: a
#: pre-existing violation is exempted so the gate can be adopted green today,
#: while any NEW violation fails. Baseline entries are a worklist, not an
#: endorsement -- WS-F's deletion sweeps consume this file directly.
BASELINE_PATH = Path("docs/plans/oracle-integrity-baseline.json")


def load_baseline(source_root: Path, *, path: Path | None = None) -> frozenset[tuple[str, str, str]]:
    """Load ``(code, path, detail)`` triples that are exempt because they pre-existed.

    Keyed on the DETAIL as well as the file, per-entry, the way the layering
    baseline is. Keying on ``(code, path)`` alone exempted an entire FILE once
    any finding in it was baselined -- a reviewer appended a brand-new
    ``~/.codex`` read to an already-baselined file and the gate stayed green
    while ``baselined`` silently rose 34 -> 35. A ratchet that can be widened
    by editing an exempted file is not a ratchet.
    """
    baseline_path = path or (source_root / BASELINE_PATH)
    if not baseline_path.is_file():
        return frozenset()
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    triples: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        code, path_value, detail = entry.get("code"), entry.get("path"), entry.get("detail")
        if isinstance(code, str) and isinstance(path_value, str) and isinstance(detail, str):
            triples.add((code, path_value, detail))
    return frozenset(triples)


def check_oracle_integrity(
    source_root: Path,
    *,
    test_root: Path | None = None,
    reachability_allowlist: Sequence[OracleAllowlistEntry] = REACHABILITY_ALLOWLIST,
    hermeticity_allowlist: Sequence[OracleAllowlistEntry] = HERMETICITY_ALLOWLIST,
    baseline: frozenset[tuple[str, str, str]] | None = None,
) -> OracleIntegrityReport:
    """Run both oracle-integrity sweeps over the test corpus."""
    source_root = source_root.resolve()
    tests_dir = (test_root or source_root / "tests").resolve()
    production_root = source_root / PRODUCTION_PACKAGE

    production_modules = _parse_modules(source_root, (production_root,))

    roots = list(_packaging_entrypoints(source_root))
    roots.extend(PUBLIC_SURFACE_ROOTS)
    roots.extend(
        _registry_roots(
            source_root,
            [module.tree for module in production_modules],
            [module.name for module in production_modules],
        )
    )
    closure = build_import_closure(source_root, roots)

    findings: list[OracleFinding] = []
    type_only_modules: list[str] = []
    reachable_count = 0
    scanned = 0

    for path in sorted(tests_dir.rglob("test_*.py")):
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(OracleFinding("unparseable_test", str(path), str(exc)))
            continue
        module_name = _module_name(path, source_root)

        if not _allowlisted(path, source_root, hermeticity_allowlist):
            findings.extend(scan_hermeticity(path.relative_to(source_root), tree))

        if _allowlisted(path, source_root, reachability_allowlist):
            continue
        verdict, symbols = classify_test_module(tree, module_name, closure=closure)
        if verdict is ReachabilityVerdict.REACHABLE:
            reachable_count += 1
        elif verdict is ReachabilityVerdict.TYPE_ONLY:
            type_only_modules.append(path.relative_to(source_root).as_posix())
        elif verdict is ReachabilityVerdict.UNREACHABLE:
            findings.append(
                OracleFinding(
                    "certifies_dead_code",
                    path.relative_to(source_root).as_posix(),
                    "every production symbol this module imports is unreachable from a "
                    f"production entrypoint: {', '.join(symbols)}",
                )
            )

    exempt = load_baseline(source_root) if baseline is None else baseline
    retained = tuple(finding for finding in findings if (finding.code, finding.path, finding.detail) not in exempt)
    return OracleIntegrityReport(
        findings=retained,
        baselined=len(findings) - len(retained),
        reachable_modules=reachable_count,
        type_only_modules=tuple(sorted(type_only_modules)),
        scanned_modules=scanned,
        roots=tuple(sorted(roots)),
    )


__all__ = [
    "AMBIENT_CALL_MARKERS",
    "BASELINE_PATH",
    "AMBIENT_PATH_MARKERS",
    "HERMETICITY_ALLOWLIST",
    "REACHABILITY_ALLOWLIST",
    "OracleAllowlistEntry",
    "OracleFinding",
    "OracleIntegrityReport",
    "ReachabilityVerdict",
    "check_oracle_integrity",
    "load_baseline",
    "classify_test_module",
    "scan_hermeticity",
]
