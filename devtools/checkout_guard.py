"""Assert that ``import polylogue`` resolves inside the invoking checkout.

Every guarded entry point shares this one path-aware check. The installed CLI
uses :func:`find_git_worktree_root` to identify a Polylogue checkout from cwd;
outside one, the guard intentionally does nothing.
"""

from __future__ import annotations

from pathlib import Path

import tomllib


class CheckoutImportMismatchError(RuntimeError):
    """``import polylogue`` resolved to a package outside the invoking checkout."""


def resolved_polylogue_path() -> Path:
    """Import ``polylogue`` at call time and return its resolved package path."""
    import polylogue

    return Path(polylogue.__file__).resolve()


def _is_polylogue_checkout_root(candidate: Path) -> bool:
    """Return whether ``candidate`` is plausibly the root of a Polylogue checkout."""
    pyproject = candidate / "pyproject.toml"
    try:
        raw = pyproject.read_bytes()
    except OSError:
        raw = None
    if raw is not None:
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            data = {}
        if data.get("project", {}).get("name") == "polylogue":
            return True
    try:
        return (candidate / "polylogue" / "cli" / "click_app.py").is_file()
    except OSError:
        return False


def find_git_worktree_root(start: Path) -> Path | None:
    """Find the enclosing Polylogue Git checkout, if any.

    The first Git boundary is authoritative. An unrelated repository and a
    directory with no Git ancestor both return ``None`` so installed CLI use
    outside a Polylogue checkout does not invoke the guard.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        try:
            has_git = (candidate / ".git").exists()
        except OSError:
            has_git = False
        if has_git:
            return candidate if _is_polylogue_checkout_root(candidate) else None
    return None


def assert_polylogue_matches_checkout(repo_root: Path, *, context: str) -> Path:
    """Raise unless the resolved package path is contained by ``repo_root``."""
    resolved_root = repo_root.resolve()
    resolved_pkg = resolved_polylogue_path()
    try:
        resolved_pkg.relative_to(resolved_root)
    except ValueError:
        raise CheckoutImportMismatchError(
            f"{context}: `import polylogue` resolved OUTSIDE this checkout.\n"
            f"  invoking checkout : {resolved_root}\n"
            f"  resolved package  : {resolved_pkg}\n"
            "\n"
            "Use an environment whose `polylogue` import resolves from this checkout.\n"
        ) from None
    return resolved_pkg
