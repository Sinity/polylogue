"""Assert that ``import polylogue`` resolves inside the invoking checkout.

Every guarded entry point shares this one path-aware check. The installed CLI
uses :func:`find_git_worktree_root` to identify a Polylogue checkout from cwd;
outside one, the guard intentionally does nothing.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
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


def git_ceiling_directories(env: Mapping[str, str] | None = None) -> frozenset[Path]:
    """Directories ``GIT_CEILING_DIRECTORIES`` forbids discovery from entering.

    Same contract as git: the listed directories and everything above them are
    never examined; entries are resolved, and an empty entry is ignored.
    """
    raw = (os.environ if env is None else env).get("GIT_CEILING_DIRECTORIES", "")
    ceilings: set[Path] = set()
    for entry in raw.split(":"):
        entry = entry.strip()
        if not entry:
            continue
        # A leading ``::`` disables symlink resolution for the rest of the list.
        with contextlib.suppress(OSError):
            ceilings.add(Path(entry.removeprefix(":")).resolve())
    return frozenset(ceilings)


def find_git_worktree_root(start: Path) -> Path | None:
    """Find the enclosing Polylogue Git checkout, if any.

    The first Git boundary is authoritative. An unrelated repository and a
    directory with no Git ancestor both return ``None`` so installed CLI use
    outside a Polylogue checkout does not invoke the guard. Discovery honours
    ``GIT_CEILING_DIRECTORIES`` exactly as git does.
    """
    current = start.resolve()
    ceilings = git_ceiling_directories()
    for candidate in (current, *current.parents):
        if candidate in ceilings:
            return None
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
