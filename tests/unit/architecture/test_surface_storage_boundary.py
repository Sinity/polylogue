"""Surface-to-storage import boundary check (#860).

The CLI, MCP, and Python API surfaces must route archive reads and
writes through ``polylogue.operations`` / ``polylogue.facade`` rather
than reaching into the storage repository directly. This static check
walks every Python module under ``polylogue/cli/``, ``polylogue/mcp/``,
and ``polylogue/api/`` and asserts that none of them import from
``polylogue.storage.repository`` (or its submodules).

Allow-list:

* ``polylogue/api/__init__.py`` — exposes ``Polylogue`` and must wire
  ``SessionRepository`` as a typed property (the API IS the
  archive seam).
* ``polylogue/api/archive.py`` — declares a ``SessionRepository``
  type alias inside a ``TYPE_CHECKING`` block for the mixin protocol.
* ``polylogue/api/ingest.py`` — typed property for ingest-path callers.
* ``polylogue/cli/shared/types.py`` — typed property exposed to CLI
  commands that legitimately need the repository handle while the
  remaining bypasses are moved.

This list should shrink over time; do not add new entries without
filing a follow-up issue.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_ROOTS = (
    REPO_ROOT / "polylogue" / "cli",
    REPO_ROOT / "polylogue" / "mcp",
    REPO_ROOT / "polylogue" / "api",
)
FORBIDDEN_PREFIX = "polylogue.storage.repository"

ALLOWED: frozenset[Path] = frozenset(
    {
        REPO_ROOT / "polylogue" / "api" / "__init__.py",
        REPO_ROOT / "polylogue" / "api" / "archive.py",
        REPO_ROOT / "polylogue" / "api" / "ingest.py",
        REPO_ROOT / "polylogue" / "cli" / "shared" / "types.py",
    }
)


def _iter_surface_modules() -> list[Path]:
    files: list[Path] = []
    for root in SURFACE_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def _imports_from_storage_repository(source: str) -> list[str]:
    """Return the module strings that import from ``polylogue.storage.repository``."""
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == FORBIDDEN_PREFIX or node.module.startswith(FORBIDDEN_PREFIX + "."):
                hits.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FORBIDDEN_PREFIX or alias.name.startswith(FORBIDDEN_PREFIX + "."):
                    hits.append(alias.name)
    return hits


@pytest.mark.parametrize("path", _iter_surface_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_surface_module_does_not_import_storage_repository(path: Path) -> None:
    """No surface module imports from ``polylogue.storage.repository`` outside the allow-list."""
    hits = _imports_from_storage_repository(path.read_text(encoding="utf-8"))
    if not hits:
        return
    if path in ALLOWED:
        return
    relative = path.relative_to(REPO_ROOT)
    pytest.fail(
        f"{relative} imports from {FORBIDDEN_PREFIX}: {sorted(set(hits))}. "
        "Route through the Polylogue facade "
        "instead, or add an explicit allow-list entry with a follow-up issue."
    )


def test_allow_list_does_not_grow_silently() -> None:
    """The allow-list should not contain stale entries; every allowed file must still import storage.repository."""
    stale = []
    for path in ALLOWED:
        assert path.exists(), f"allow-list references missing file: {path}"
        if not _imports_from_storage_repository(path.read_text(encoding="utf-8")):
            stale.append(path.relative_to(REPO_ROOT))
    assert not stale, f"allow-list entries no longer import storage.repository and should be removed: {stale}"
