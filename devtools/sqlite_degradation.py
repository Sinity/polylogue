"""Census of hand-written ``except sqlite3`` degradation policy in read paths.

Each such handler decides, at its own call site, what an unavailable store
looks like to the caller -- usually a zero or an empty list that is
indistinguishable from a real answer. The typed route
(``polylogue.core.evidence`` plus ``polylogue.storage.tier_access``) classifies
that once at the seam instead, so this census exists to ratchet the improvised
sites down. Handlers that preserve an explicit failure boundary by raising are
not degradation sites; handlers that return a value (including an empty or
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
from pathlib import Path
from typing import TypeAlias

from devtools.ast_cache import parse_source, walk_module

__all__ = [
    "DegradationAnchor",
    "anchor_text",
    "census_sqlite_degradation_anchors",
    "load_sqlite_degradation_baseline",
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


def _returns_value(node: ast.ExceptHandler, parents: dict[ast.AST, ast.AST]) -> bool:
    """Include every handler except one whose failure boundary is explicit.

    ``continue``, ``break``, ``pass`` and fallback assignments all degrade
    even when no return appears in the handler itself. A bare re-raise or an
    explicit exception raise preserves the boundary.
    """
    children = list(ast.walk(node))
    if any(isinstance(child, ast.Return | ast.Continue | ast.Break | ast.Pass) for child in children):
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
            file_rel = py_file.relative_to(repo_root).as_posix()
            parents = {child: parent for parent in walk_module(tree) for child in ast.iter_child_nodes(parent)}
            for node in walk_module(tree):
                if isinstance(node, ast.ExceptHandler) and _handles_sqlite(node) and _returns_value(node, parents):
                    anchors[(file_rel, normalized_handler_digest(source, node))] += 1
    return anchors


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
