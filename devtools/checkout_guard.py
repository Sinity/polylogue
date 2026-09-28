"""Assert that ``import polylogue`` resolves inside the invoking checkout.

Every guarded entry point shares this one path-aware check. The installed CLI
uses :func:`find_git_worktree_root` to identify a Polylogue checkout from cwd;
outside one, the guard intentionally does nothing.
"""

from __future__ import annotations

import contextlib
import os
import sys
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


#: Variables that name a checkout. An agent shell started in the primary
#: checkout carries them into every later command, including ones aimed at a
#: worktree, so a tool launched from the worktree would resolve the primary
#: checkout's code or interpreter and report its result as the worktree's.
_CHECKOUT_ROOT_VARIABLES = ("POLYLOGUE_REPO_ROOT", "POLYLOGUE_ROOT")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _foreign_checkout(path_text: str, root: Path) -> bool:
    """Whether ``path_text`` lies in a Polylogue checkout other than ``root``.

    A relative entry is resolved against the working directory, exactly as the
    process that consults it would resolve it.
    """
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    # The nearest owning checkout decides, so a clone nested inside ``root``
    # is as foreign as a sibling.
    owner = _polylogue_checkout_ancestor(path)
    return owner is not None and owner != root.resolve()


def _polylogue_checkout_ancestor(path: Path) -> Path | None:
    """The Polylogue checkout containing ``path``, ignoring Git ceilings.

    Ownership of an environment entry is a containment question, not
    repository discovery: a ceiling that stops ``git`` from searching above a
    directory does not make that directory's venv this checkout's.
    """
    try:
        current = path.resolve()
    except OSError:
        return None
    for candidate in (current, *current.parents):
        try:
            has_git = (candidate / ".git").exists()
        except OSError:
            has_git = False
        if has_git:
            return candidate if _is_polylogue_checkout_root(candidate) else None
    return None


class ForeignInterpreterError(RuntimeError):
    """Devtools is running on an interpreter that belongs to another checkout."""


def assert_interpreter_belongs_to(root: Path, *, context: str) -> None:
    """Refuse a Python whose environment is another Polylogue checkout's.

    Rewriting ``PATH`` cannot change the interpreter already running, and
    commands launch children through ``sys.executable``, so a foreign one would
    run this checkout's code on the other checkout's dependencies.
    """
    prefix = Path(sys.prefix)
    owner = _polylogue_checkout_ancestor(prefix)
    if owner is None or owner == root.resolve():
        return
    raise ForeignInterpreterError(
        f"{context}: running on another checkout's interpreter.\n"
        f"  invoking checkout : {root.resolve()}\n"
        f"  interpreter prefix: {prefix}\n"
        "\n"
        "Provision this checkout's environment (enter its devshell once) and rerun.\n"
    )


def normalize_checkout_environment(root: Path, environ: dict[str, str] | None = None) -> list[str]:
    """Rebind checkout-naming variables to ``root``; return what was corrected.

    ``root`` is the checkout this devtools code belongs to, which the wrapper
    derives from the working directory's Git root. Its own ``.venv`` is
    authoritative: a ``VIRTUAL_ENV``, ``PATH`` or ``PYTHONPATH`` entry in a
    different checkout is replaced or dropped, and ``POLYLOGUE_REPO_ROOT`` /
    ``POLYLOGUE_ROOT`` are rebound, so every subprocess resolves this checkout.
    """
    env = os.environ if environ is None else environ
    resolved_root = root.resolve()
    corrected: list[str] = []
    for name in _CHECKOUT_ROOT_VARIABLES:
        value = env.get(name)
        if value and Path(value).resolve() != resolved_root:
            env[name] = str(resolved_root)
            corrected.append(f"{name}={value}")
    own_venv = resolved_root / ".venv"
    virtual_env = env.get("VIRTUAL_ENV")
    if virtual_env and _polylogue_checkout_ancestor(Path(virtual_env)) != resolved_root:
        corrected.append(f"VIRTUAL_ENV={virtual_env}")
        if own_venv.is_dir():
            env["VIRTUAL_ENV"] = str(own_venv)
        else:
            env.pop("VIRTUAL_ENV", None)
    for name in ("PATH", "PYTHONPATH"):
        value = env.get(name)
        if not value:
            continue
        entries = value.split(os.pathsep)
        kept = [entry for entry in entries if not _foreign_checkout(entry, resolved_root)]
        if name == "PATH" and (own_venv / "bin").is_dir():
            # First, not merely present: an earlier directory with the same
            # tool would otherwise still win.
            kept = [str(own_venv / "bin"), *(entry for entry in kept if entry != str(own_venv / "bin"))]
        if kept != entries:
            dropped = [entry for entry in entries if entry not in kept]
            if dropped:
                corrected.append(f"{name} entries {dropped}")
            env[name] = os.pathsep.join(kept)
    if environ is None:
        # ``PYTHONPATH`` was copied into ``sys.path`` when this interpreter
        # started; dropping it from the environment protects children only.
        foreign = [entry for entry in sys.path if entry and _foreign_checkout(entry, resolved_root)]
        for entry in foreign:
            sys.path.remove(entry)
        if foreign:
            corrected.append(f"sys.path entries {foreign}")
    return corrected


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
