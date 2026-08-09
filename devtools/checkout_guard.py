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
- ``polylogue/cli/click_app.py:main()`` — the installed ``polylogue``/``plg``
  console scripts. This one differs from the others: those callers compute
  "the invoking checkout" from their *own* ``__file__`` (trustworthy, because
  ``devtools/__main__.py`` and ``tests/conftest.py`` are only ever reached via
  a real filesystem path on ``sys.path``, never via the shared venv's
  editable ``.pth``). ``polylogue/cli/click_app.py`` cannot do that: if the
  hazard has already occurred, *this file itself* resolved from the wrong
  checkout, so its own ``__file__`` is exactly as compromised as
  ``polylogue.__file__`` and proves nothing. The trustworthy anchor there is
  the process's current working directory instead — see
  :func:`find_git_worktree_root`.

What this does **not** close: an ad hoc one-off script
(``python3 /realm/tmp/scratch.py``) that never imports ``devtools`` or
``pytest`` has no hook to run this check from. That residual gap is
unavoidable without a machine-wide interpreter customization (see the
CLAUDE.md note next to this hazard for the environmental-layer cost/benefit
call); a script that wants the guarantee can import and call
:func:`assert_polylogue_matches_checkout` itself in one line.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomllib

from devtools.testmon_state import validate_stamp


class CheckoutImportMismatchError(RuntimeError):
    """``import polylogue`` resolved to a package outside the invoking checkout."""


class CheckoutEnvironmentMismatchError(CheckoutImportMismatchError):
    """The checkout contains environment state that makes results untrustworthy."""


@dataclass(frozen=True, slots=True)
class EnvironmentArtifact:
    """One checkout-local artifact that can poison a linked-worktree run."""

    kind: str
    path: Path
    detail: str
    remediation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "detail": self.detail,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class CheckoutEnvironmentFingerprint:
    """Provenance and contamination evidence attached to verification receipts."""

    checkout_root: Path
    polylogue_import_path: Path
    python_executable: Path
    python_environment_root: Path | None
    linked_worktree: bool
    testmon_state_origin: Path | None
    verify_state_origin: Path | None
    artifacts: tuple[EnvironmentArtifact, ...]

    @property
    def clean(self) -> bool:
        return not self.artifacts

    def as_dict(self) -> dict[str, object]:
        return {
            "checkout_root": str(self.checkout_root),
            "polylogue_import_path": str(self.polylogue_import_path),
            "python_executable": str(self.python_executable),
            "python_environment_root": (
                str(self.python_environment_root) if self.python_environment_root is not None else None
            ),
            "linked_worktree": self.linked_worktree,
            "testmon_state_origin": str(self.testmon_state_origin) if self.testmon_state_origin else None,
            "verify_state_origin": str(self.verify_state_origin) if self.verify_state_origin else None,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


_TESTMON_STATE_DIR = Path(".cache/testmon")
_TESTMON_STATE_MARKER = _TESTMON_STATE_DIR / "seed.json"
_TESTMON_SEED_ATTEMPT = _TESTMON_STATE_DIR / "seed-attempt.json"
_TESTMON_SEED_PROTOCOL_VERSION = 4
_VERIFY_STATE_DIR = Path(".cache/verify")
_VERIFY_STATE_MARKER = _VERIFY_STATE_DIR / "current-run.json"


def resolved_polylogue_path() -> Path:
    """Import ``polylogue`` (idempotent) and return its resolved package path.

    A function-local import, not a module-level one: this call site *is* the
    resolution being checked, so it must run at call time against whatever
    ``sys.path`` looks like right now, not whatever it looked like when this
    module was first imported.
    """
    import polylogue

    return Path(polylogue.__file__).resolve()


def _is_polylogue_checkout_root(candidate: Path) -> bool:
    """Return whether ``candidate`` is plausibly the root of a Polylogue checkout.

    Cheap, filesystem-only markers (no subprocess): either a
    ``pyproject.toml`` whose ``[project].name`` is literally ``"polylogue"``,
    or (fallback, for the rare case a checkout's ``pyproject.toml`` is
    missing/unparseable but the source tree is intact) the package's own
    entry-point file, ``polylogue/cli/click_app.py``, existing under
    ``candidate``. Either marker is present in every Polylogue checkout and
    worktree, and absent from an unrelated repository.
    """
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
        project_name = data.get("project", {}).get("name")
        if project_name == "polylogue":
            return True
    try:
        return (candidate / "polylogue" / "cli" / "click_app.py").is_file()
    except OSError:
        return False


def find_git_worktree_root(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for the enclosing Polylogue checkout.

    Pure filesystem ``stat``/``exists`` calls — no subprocess — so this is
    cheap enough to run unconditionally on every CLI invocation (a handful of
    directory levels at most, not a ``git rev-parse`` fork+exec). Matches both
    a plain repo's ``.git`` directory and a linked worktree's ``.git`` *file*
    (which points at the shared ``.git/worktrees/<name>`` gitdir elsewhere).

    Returns the first ``.git``-containing ancestor directory, but **only**
    when that directory also looks like a Polylogue checkout
    (:func:`_is_polylogue_checkout_root`). Two ``None`` cases, deliberately
    not distinguished by the caller:

    - No ``.git`` entry anywhere in the ancestry — e.g. a real installed
      ``polylogue`` invocation with no dev checkout in its cwd ancestry at
      all.
    - A ``.git`` entry *is* found, but it belongs to some other, unrelated
      repository (``cd ~/some-other-project && polylogue --version`` — an
      ordinary, common invocation). The walk stops at that repo's root
      rather than climbing past it into parent directories: a git
      repository boundary is exactly the boundary of "the checkout the
      process is running from", so a non-Polylogue repo there means cwd is
      not inside any Polylogue checkout, full stop — climbing further up
      could otherwise find an unrelated Polylogue checkout that merely
      happens to be an ancestor directory and misfire the guard against it.

    Both cases must no-op identically (2026-08-01 regression, polylogue-373yt):
    the guard exists to catch a linked *Polylogue* worktree resolving a
    *different* Polylogue checkout's package, not to flag every invocation
    whose cwd happens to have some ``.git`` ancestor.
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


def _is_linked_worktree(repo_root: Path) -> bool:
    """Return whether ``repo_root`` has Git's linked-worktree ``.git`` file."""
    try:
        return (repo_root / ".git").is_file()
    except OSError:
        return False


def _python_environment_root(executable: Path) -> Path | None:
    """Return the checkout owning a ``.venv`` executable, when identifiable."""
    # Keep the lexical ``.venv`` component. Nix/direnv Python launchers are
    # symlinks into the Nix store, so resolving first would erase the checkout
    # boundary and falsely classify a valid lane-local interpreter as foreign.
    candidate = executable if executable.is_absolute() else Path.cwd() / executable
    for parent in (candidate, *candidate.parents):
        if parent.name == ".venv":
            return parent.parent.resolve()
    return None


def _marker_origin(marker: Path) -> Path | None:
    """Read a marker's checkout origin, returning ``None`` for legacy state."""
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("checkout_root")
    if raw is None:
        binding = payload.get("binding")
        if isinstance(binding, Mapping):
            raw = binding.get("checkout_root")
    if raw is None:
        fingerprint = payload.get("environment_fingerprint")
        if isinstance(fingerprint, Mapping):
            raw = fingerprint.get("checkout_root")
    if not isinstance(raw, str) or not raw:
        return None
    return Path(raw).resolve()


def _is_valid_in_progress_testmon_seed_attempt(attempt: Path) -> bool:
    """Recognize the live seed ledger before its completion marker exists.

    ``verify --seed-testmon`` writes this receipt before pytest starts and
    rewrites it after the run. It is the only unmarked testmon state that a
    linked worktree may trust. Keep this contract local to the guard so a
    random JSON file cannot turn an inherited cache into accepted state.
    """
    try:
        payload = json.loads(attempt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping) or payload.get("status") not in {"running", "incomplete", "reusable"}:
        return False
    protocol_version = payload.get("protocol_version")
    if not isinstance(protocol_version, int) or isinstance(protocol_version, bool) or protocol_version <= 0:
        return False
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        return False
    if not isinstance(identity.get("worktree_fingerprint"), str) or not identity["worktree_fingerprint"]:
        return False
    if not isinstance(identity.get("python"), str) or not identity["python"]:
        return False
    if not isinstance(identity.get("skip_slow"), bool) or not isinstance(identity.get("lab"), bool):
        return False
    git_head = identity.get("git_head")
    if git_head is not None and (not isinstance(git_head, str) or not git_head):
        return False
    if "checkout_root" in payload:
        return False
    expected_nodeids = payload.get("expected_nodeids")
    if not isinstance(expected_nodeids, list) or any(
        not isinstance(nodeid, str) or not nodeid for nodeid in expected_nodeids
    ):
        return False
    expected_count = payload.get("expected_count")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count != len(expected_nodeids)
    ):
        return False
    if not isinstance(payload.get("resume"), bool):
        return False
    for key in ("started_at", "run_id", "artifact_dir", "testmon_data_before"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            return False
    artifact_dir = Path(payload["artifact_dir"])
    return not artifact_dir.is_absolute() and artifact_dir.parts[:3] == (".cache", "verify", "runs")


def _cache_artifact(
    *,
    repo_root: Path,
    state_dir: Path,
    marker: Path,
    kind: str,
    remediation: str,
) -> tuple[Path | None, EnvironmentArtifact | None]:
    """Classify one cache directory by its explicit checkout-root marker."""
    state_path = repo_root / state_dir
    if not state_path.exists():
        return None, None
    try:
        if state_path.is_dir() and not any(state_path.iterdir()):
            return None, None
    except OSError:
        pass
    marker_path = repo_root / marker
    origin = _marker_origin(marker_path)
    if origin == repo_root:
        if (
            state_dir == _TESTMON_STATE_DIR
            and validate_stamp(
                marker_path,
                state_path / "testmondata",
                checkout_root=repo_root,
                protocol_version=_TESTMON_SEED_PROTOCOL_VERSION,
            )
            is None
        ):
            return (
                origin,
                EnvironmentArtifact(
                    kind="invalid_testmon_seed",
                    path=marker_path,
                    detail="testmon seed marker is stale, malformed, or its SQLite graph is incomplete",
                    remediation=(
                        f"remove {state_path} and run `devtools verify --seed-testmon` "
                        "to rebuild the typed testmon state"
                    ),
                ),
            )
        return origin, None
    if (
        origin is None
        and not marker_path.exists()
        and state_dir == _TESTMON_STATE_DIR
        and _is_valid_in_progress_testmon_seed_attempt(repo_root / _TESTMON_SEED_ATTEMPT)
    ):
        return None, None
    if origin is None:
        detail = f"{state_path} has no verifiable checkout-root marker"
    else:
        detail = f"{state_path} was produced by checkout {origin}, not {repo_root}"
    return (
        origin,
        EnvironmentArtifact(
            kind=kind,
            path=repo_root / marker if (repo_root / marker).exists() else state_path,
            detail=detail,
            remediation=remediation,
        ),
    )


def checkout_environment_fingerprint(
    repo_root: Path,
    *,
    polylogue_import_path: Path | None = None,
    python_executable: Path | None = None,
) -> CheckoutEnvironmentFingerprint:
    """Collect cheap, deterministic provenance and contamination evidence.

    Artifact checks are intentionally scoped to linked worktrees. The main
    checkout owns the shared development environment and its caches; a lane
    worktree must either have a lane-local interpreter/cache marker or refuse
    before verification starts.
    """
    resolved_root = repo_root.resolve()
    resolved_pkg = (polylogue_import_path or resolved_polylogue_path()).resolve()
    executable_input = python_executable or Path(sys.executable)
    environment_root = _python_environment_root(executable_input)
    executable = executable_input.resolve()
    linked = _is_linked_worktree(resolved_root)
    artifacts: list[EnvironmentArtifact] = []
    testmon_origin: Path | None = None
    verify_origin: Path | None = None

    if linked:
        if environment_root is not None and environment_root != resolved_root:
            artifacts.append(
                EnvironmentArtifact(
                    kind="interpreter_from_other_checkout",
                    path=executable,
                    detail=(
                        f"interpreter environment {environment_root} belongs to another checkout; "
                        f"this run executes code from {resolved_root}"
                    ),
                    remediation=(f"run `cd {resolved_root} && direnv allow` to provision this worktree's venv"),
                )
            )
        local_venv = resolved_root / ".venv"
        if local_venv.exists() and environment_root != resolved_root:
            artifacts.append(
                EnvironmentArtifact(
                    kind="stray_venv",
                    path=local_venv,
                    detail="checkout-local .venv is present but is not the active interpreter environment",
                    remediation=(
                        f"remove {local_venv} and run `cd {resolved_root} && direnv allow`, "
                        "or activate its own venv before verifying"
                    ),
                )
            )
        node_modules = resolved_root / "node_modules"
        if node_modules.exists():
            artifacts.append(
                EnvironmentArtifact(
                    kind="stray_node_modules",
                    path=node_modules,
                    detail="root-level node_modules is untracked lane state and can be inherited from another checkout",
                    remediation=f"remove {node_modules} before running the lane verification",
                )
            )
        testmon_origin, testmon_artifact = _cache_artifact(
            repo_root=resolved_root,
            state_dir=_TESTMON_STATE_DIR,
            marker=_TESTMON_STATE_MARKER,
            kind="inherited_testmon_cache",
            remediation=(
                f"remove {resolved_root / _TESTMON_STATE_DIR} and let `devtools verify --seed-testmon` "
                "or the managed bootstrap recreate it"
            ),
        )
        verify_origin, verify_artifact = _cache_artifact(
            repo_root=resolved_root,
            state_dir=_VERIFY_STATE_DIR,
            marker=_VERIFY_STATE_MARKER,
            kind="inherited_verify_cache",
            remediation=f"remove {resolved_root / _VERIFY_STATE_DIR} and rerun the managed devtools command",
        )
        if testmon_artifact is not None:
            artifacts.append(testmon_artifact)
        if verify_artifact is not None:
            artifacts.append(verify_artifact)

    return CheckoutEnvironmentFingerprint(
        checkout_root=resolved_root,
        polylogue_import_path=resolved_pkg,
        python_executable=executable,
        python_environment_root=environment_root,
        linked_worktree=linked,
        testmon_state_origin=testmon_origin,
        verify_state_origin=verify_origin,
        artifacts=tuple(artifacts),
    )


def _raise_environment_mismatch(context: str, fingerprint: CheckoutEnvironmentFingerprint) -> None:
    if fingerprint.clean:
        return
    lines = [
        f"{context}: checkout environment is untrustworthy; refusing to run before it is repaired.",
    ]
    for artifact in fingerprint.artifacts:
        lines.extend(
            (
                f"  offending path : {artifact.path}",
                f"  reason         : {artifact.detail}",
                f"  remediation    : {artifact.remediation}",
            )
        )
    raise CheckoutEnvironmentMismatchError("\n".join(lines))


def assert_polylogue_matches_checkout(
    repo_root: Path,
    *,
    context: str,
    python_executable: Path | None = None,
) -> CheckoutEnvironmentFingerprint:
    """Raise loudly when import or environment provenance does not match ``repo_root``.

    Returns the validated environment fingerprint on success, so callers can
    persist the exact provenance the guard checked without reading the
    filesystem a second time.
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
    fingerprint = checkout_environment_fingerprint(
        repo_root,
        polylogue_import_path=resolved_pkg,
        python_executable=python_executable,
    )
    _raise_environment_mismatch(context, fingerprint)
    return fingerprint
