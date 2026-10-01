"""Census of hand-written ``except sqlite3`` degradation policy in read paths.

Each such handler decides, at its own call site, what an unavailable store
looks like to the caller -- usually a zero or an empty list that is
indistinguishable from a real answer. The typed route
(``polylogue.core.evidence`` plus ``polylogue.storage.tier_access``) classifies
that once at the seam instead, so this census exists to ratchet the improvised
sites down. Handlers that preserve an explicit failure boundary by raising are
not degradation sites. Imported canonical operation envelopes returning only
``failed``/``rejected`` with an explicit error and no result are also visible
failure boundaries; handlers that return a value (including an empty or
otherwise fabricated projection) remain in the census.

The census is **content-anchored**, the same mechanism
``devtools/verify_patterns.py`` uses: a site is identified by
``(file, sha1(normalized handler source))``, not by how many sites its file
carries. A per-file integer ceiling is measured against whichever baseline the
branch started from, so two independently green branches that each add a
different handler to the same file both pass and their merge sums to red on
master. Anchors do not sum: each branch's addition is an anchor the baseline
does not contain, so each branch fails on its own.

Identical handler text repeated inside one file collapses to one anchor with a
count, exactly as in ``verify_patterns``; that residual count is per-anchor, not
per-file, so it only aggregates sites that are literally the same text.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

from devtools.ast_cache import parse_source, walk_module

__all__ = [
    "DegradationAnchor",
    "anchor_text",
    "census_sqlite_degradation_anchors",
    "load_sqlite_degradation_baseline",
    "module_degradation_anchors",
    "normalized_handler_digest",
]

#: ``(repo-relative file, sha1 of the normalized handler source)``.
DegradationAnchor: TypeAlias = tuple[str, str]


def _handles_sqlite(node: ast.ExceptHandler) -> bool:
    if node.type is None:
        return False
    candidates = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
    for candidate in candidates:
        # ``sqlite3.Error``/``sqlite3.OperationalError``/... -- the attribute
        # form is the only one production code uses, and matching on the
        # module name keeps aliased re-exports out of the census.
        if (
            isinstance(candidate, ast.Attribute)
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == "sqlite3"
        ):
            return True
    return False


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Inspect one lexical scope without importing nested definitions' bindings."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        child = pending.pop()
        yield child
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            pending.extend(ast.iter_child_nodes(child))


def _canonical_envelope_name(node: ast.Return, parents: dict[ast.AST, ast.AST], name: str) -> bool:
    scope = parents.get(node)
    while scope is not None:
        if isinstance(scope, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
            bindings: list[bool] = []
            if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
                arguments = [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]
                if scope.args.vararg is not None:
                    arguments.append(scope.args.vararg)
                if scope.args.kwarg is not None:
                    arguments.append(scope.args.kwarg)
                if any(argument.arg == name for argument in arguments):
                    bindings.append(False)
            for child in _scope_nodes(scope):
                if isinstance(child, ast.ImportFrom):
                    for alias in child.names:
                        if (alias.asname or alias.name) == name:
                            bindings.append(
                                child.module == "polylogue.operations.daemon_execution"
                                and child.level == 0
                                and alias.name == "operation_envelope"
                                and child.lineno < node.lineno
                            )
                elif isinstance(child, ast.Import):
                    if any((alias.asname or alias.name.split(".")[0]) == name for alias in child.names):
                        bindings.append(False)
                elif (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Store | ast.Del)
                    and child.id == name
                    or isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                    and child.name == name
                    or isinstance(child, ast.ExceptHandler)
                    and child.name == name
                ):
                    bindings.append(False)
            if bindings:
                return all(bindings)
        scope = parents.get(scope)
    return False


def _explicit_failure_return(node: ast.Return, parents: dict[ast.AST, ast.AST]) -> bool:
    call = node.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return False
    if not _canonical_envelope_name(node, parents, call.func.id) or len(call.args) != 2:
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if None in keywords or "result" in keywords or "error" not in keywords:
        return False
    error = keywords["error"]
    if isinstance(error, ast.Constant) and error.value is None:
        return False

    def failure_only(value: ast.AST | None) -> bool:
        if isinstance(value, ast.Constant):
            return value.value in {"failed", "rejected"}
        if isinstance(value, ast.IfExp):
            return failure_only(value.body) and failure_only(value.orelse)
        return False

    return failure_only(keywords.get("outcome"))


def _returns_value(node: ast.ExceptHandler, parents: dict[ast.AST, ast.AST]) -> bool:
    """Include every handler except one whose failure boundary is explicit.

    ``continue``, ``break``, ``pass`` and fallback assignments all degrade
    even when no return appears in the handler itself. A bare re-raise or an
    explicit exception raise preserves the boundary. Canonical operation envelopes
    with failure-only outcomes and an explicit error preserve it as well.
    """
    children = list(ast.walk(node))

    terminal_failure = bool(
        node.body and isinstance(node.body[-1], ast.Return) and _explicit_failure_return(node.body[-1], parents)
    )

    def escapes_handler_loop(child: ast.AST) -> bool:
        if not terminal_failure:
            return True
        ancestor = parents.get(child)
        while ancestor is not None and ancestor is not node:
            if isinstance(ancestor, ast.For | ast.AsyncFor | ast.While):
                return False
            ancestor = parents.get(ancestor)
        return True

    if any(
        isinstance(child, ast.Pass)
        or (isinstance(child, ast.Continue | ast.Break) and escapes_handler_loop(child))
        or (isinstance(child, ast.Return) and not _explicit_failure_return(child, parents))
        for child in children
    ):
        return True
    assigned = {
        target.id
        for child in children
        if isinstance(child, ast.Assign | ast.AnnAssign | ast.NamedExpr)
        for target in (child.targets if isinstance(child, ast.Assign) else [child.target])
        if isinstance(target, ast.Name)
    }
    scope = parents.get(node)
    while scope is not None and not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        scope = parents.get(scope)
    if not assigned or scope is None:
        return False
    return any(
        isinstance(child, ast.Return)
        and child.value is not None
        and not _explicit_failure_return(child, parents)
        and any(isinstance(value, ast.Name) and value.id in assigned for value in ast.walk(child.value))
        for child in ast.walk(scope)
        if child is not node
    )


def normalized_handler_digest(source: str, node: ast.ExceptHandler) -> str:
    """Return the content anchor digest for one handler.

    The handler's own source text is normalized on whitespace so that
    reindentation, line wrapping, or an unrelated edit elsewhere in the file
    leaves the anchor intact, and hashed with sha1 (matching
    ``devtools/verify_patterns.py``). Two textually different handlers -- the
    case the count form cannot separate -- get two different anchors.
    """

    segment = ast.get_source_segment(source, node)
    if segment is None:  # pragma: no cover - only for sources without positions
        raise ValueError("cannot extract handler source segment")
    return hashlib.sha1(" ".join(segment.split()).encode("utf-8")).hexdigest()


def anchor_text(anchor: DegradationAnchor, count: int = 1) -> str:
    """Render one anchor for human-readable gate output."""
    file_name, digest = anchor
    suffix = f":{count}" if count != 1 else ""
    return f"{file_name}:{digest}{suffix}"


def census_sqlite_degradation_anchors(repo_root: Path, roots: tuple[str, ...]) -> Counter[DegradationAnchor]:
    """Return the observed degradation-site anchors under ``roots``."""
    anchors: Counter[DegradationAnchor] = Counter()
    for root in roots:
        root_path = repo_root / root
        if not root_path.is_dir():
            continue
        for py_file in sorted(root_path.rglob("*.py")):
            try:
                source, tree = parse_source(py_file)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            anchors.update(module_degradation_anchors(source, tree, file_rel=py_file.relative_to(repo_root).as_posix()))
    return anchors


def module_degradation_anchors(source: str, tree: ast.Module, *, file_rel: str) -> list[DegradationAnchor]:
    """Return one module's degradation-site anchors, one per site."""
    parents = {child: parent for parent in walk_module(tree) for child in ast.iter_child_nodes(parent)}
    return [
        (file_rel, normalized_handler_digest(source, node))
        for node in walk_module(tree)
        if isinstance(node, ast.ExceptHandler) and _handles_sqlite(node) and _returns_value(node, parents)
    ]


def load_sqlite_degradation_baseline(baseline_path: Path) -> Counter[DegradationAnchor]:
    """Load the checked-in anchor baseline; a missing file means no ceiling."""
    if not baseline_path.exists():
        return Counter()
    with open(baseline_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        return Counter()
    entries = raw.get("anchors")
    if not isinstance(entries, list):
        return Counter()
    anchors: Counter[DegradationAnchor] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid baseline entry in {baseline_path}: {entry!r}")
        file_name = entry.get("file")
        digest = entry.get("digest")
        count = entry.get("count", 1)
        if (
            not isinstance(file_name, str)
            or not file_name
            or not isinstance(digest, str)
            or len(digest) != hashlib.sha1().digest_size * 2
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
        ):
            raise ValueError(f"invalid baseline entry in {baseline_path}: {entry!r}")
        anchors[(file_name, digest)] += count
    return anchors
