"""Census of hand-written ``except sqlite3`` degradation policy in read paths.

Each such handler decides, at its own call site, what an unavailable store
looks like to the caller -- usually a zero or an empty list that is
indistinguishable from a real answer. The typed route
(``polylogue.core.evidence`` plus ``polylogue.storage.tier_access``) classifies
that once at the seam instead, so this census exists to ratchet the improvised
sites down: the baseline records what each file carries today, and
``devtools gate layering`` fails when a file grows past it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

__all__ = ["census_sqlite_degradation_sites", "load_sqlite_degradation_baseline"]


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


def census_sqlite_degradation_sites(repo_root: Path, roots: tuple[str, ...]) -> dict[str, int]:
    """Return repo-relative file -> count of ``except sqlite3.*`` handlers."""
    counts: dict[str, int] = {}
    for root in roots:
        root_path = repo_root / root
        if not root_path.is_dir():
            continue
        for py_file in sorted(root_path.rglob("*.py")):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and _handles_sqlite(node))
            if count:
                counts[py_file.relative_to(repo_root).as_posix()] = count
    return counts


def load_sqlite_degradation_baseline(baseline_path: Path) -> dict[str, int]:
    """Load the checked-in per-file ceiling; a missing file means no ceiling."""
    if not baseline_path.exists():
        return {}
    with open(baseline_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        return {}
    sites = raw.get("sites")
    if not isinstance(sites, dict):
        return {}
    return {str(key): int(value) for key, value in sites.items() if isinstance(value, int)}
