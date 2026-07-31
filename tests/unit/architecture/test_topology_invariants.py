"""Durable topology invariants for the realized package layout."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


KERNEL_ROOT_FILES = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "version.py",
        "errors.py",
        "types.py",
        "protocols.py",
        "config.py",
        "logging.py",
        "services.py",
        "assets.py",
        "py.typed",
        # Must import before anything else touches `sqlite3` (it swaps in a
        # modern bundled build when the system one predates FTS5's
        # `contentless_delete`, #3070) -- `polylogue/__init__.py`'s first
        # statement, so it belongs at the same kernel root level as the
        # other always-imported-first modules above.
        "_sqlite_compat.py",
    }
)


def root_files() -> set[str]:
    return {p.name for p in (ROOT / "polylogue").glob("*.py")}


def test_polylogue_root_matches_kernel_rule() -> None:
    files = root_files()
    extra = files - KERNEL_ROOT_FILES
    assert not extra, f"non-kernel files at polylogue/ root: {sorted(extra)}"
