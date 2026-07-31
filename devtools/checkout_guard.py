"""Refuse to run against a `polylogue` import that resolves outside the invoking checkout.

The hazard (2026-07-30/31, four corrupted lanes in one day): the shared
``.venv`` at the main checkout has an editable install (``*.pth``) pointing at
that checkout's ``polylogue/`` tree. When a tool runs from a *linked git
worktree* using that same venv, a plain ``import polylogue`` with nothing else
on ``sys.path`` silently resolves to the **main checkout's** package instead
of the worktree's own source — not the worktree the invoking process actually
lives in. The failure is silent: no ImportError, no warning, just wrong code
answering every question (schema version compared, benchmark measured, CLI
behavior read) while looking exactly like a genuine result. It cost real time
four separate times in one day before anyone happened to check
``polylogue.__file__``.

This module is the **single shared resolver** for the check, mirroring the
shape of ``devtools.verify_runs.resolve_pytest_basetemp_root`` (PR #3449 fixed
a structurally identical bug — two independent basetemp-placement
implementations silently disagreeing — with one shared resolver, headroom
checked, failing loudly with every candidate named). Every entry point that
can plausibly run against the wrong tree calls
:func:`assert_polylogue_matches_checkout` instead of hand-rolling its own
comparison, so there is exactly one definition of "matches" and one message
shape.

Wired into:

- ``devtools/__main__.py`` — the CLI entry point, right after the ``sys.path``
  fixup and before any command dispatches.
- ``devtools/click_dispatch.py:main()`` — the programmatic entry point (tests
  and any caller that imports the dispatcher directly instead of going
  through ``__main__.py``).
- ``devtools/verify.py:main()`` / ``devtools/run_tests.py:main()`` — the
  ``devtools verify`` / ``devtools test`` preflight, so a wrong-tree run
  refuses before any step (ruff, mypy, pytest) executes, and prints the
  resolved path as part of the run's receipt.
- ``tests/conftest.py:pytest_configure`` — so even a bare ``pytest`` /
  ``python -m pytest`` invocation that bypasses ``devtools`` entirely still
  refuses, because ``tests/conftest.py`` is always collected for any pytest
  run rooted at this repo.

What this does **not** close: an ad hoc one-off script
(``python3 /realm/tmp/scratch.py``) that never imports ``devtools`` or
``pytest`` has no hook to run this check from. That residual gap is
unavoidable without a machine-wide interpreter customization (see the
CLAUDE.md note next to this hazard for the environmental-layer cost/benefit
call); a script that wants the guarantee can import and call
:func:`assert_polylogue_matches_checkout` itself in one line.
"""

from __future__ import annotations

from pathlib import Path


class CheckoutImportMismatchError(RuntimeError):
    """``import polylogue`` resolved to a package outside the invoking checkout."""


def resolved_polylogue_path() -> Path:
    """Import ``polylogue`` (idempotent) and return its resolved package path.

    A function-local import, not a module-level one: this call site *is* the
    resolution being checked, so it must run at call time against whatever
    ``sys.path`` looks like right now, not whatever it looked like when this
    module was first imported.
    """
    import polylogue

    return Path(polylogue.__file__).resolve()


def assert_polylogue_matches_checkout(repo_root: Path, *, context: str) -> Path:
    """Raise loudly when the imported ``polylogue`` package lives outside ``repo_root``.

    Returns the resolved package path on success, so callers can also use it
    as the "print the resolved path in the receipt" observability hook
    (requirement (3) of the worktree-import hazard fix) without importing
    ``polylogue`` a second time.
    """
    resolved_root = repo_root.resolve()
    resolved_pkg = resolved_polylogue_path()
    try:
        resolved_pkg.relative_to(resolved_root)
    except ValueError:
        raise CheckoutImportMismatchError(
            f"{context}: `import polylogue` resolved OUTSIDE this checkout — every "
            "result from this process is untrustworthy until this is fixed.\n"
            f"  invoking checkout : {resolved_root}\n"
            f"  resolved package  : {resolved_pkg}\n"
            "\n"
            "Cause: the active virtualenv's editable install (a `.pth` file in "
            "site-packages) points at a different checkout than the one this process "
            "is running from — typically a linked git worktree reusing the main "
            "checkout's shared `.venv` on PATH instead of a worktree-local one. Fix "
            "one of:\n"
            f"  - give this checkout its own venv, e.g. `cd {resolved_root} && "
            "direnv allow` (the flake devShell creates + syncs a local `.venv` "
            "automatically), or\n"
            f"  - re-install editable from THIS checkout so its own `.pth` entry "
            f"wins: `uv pip install -e {resolved_root}`.\n"
        ) from None
    return resolved_pkg
