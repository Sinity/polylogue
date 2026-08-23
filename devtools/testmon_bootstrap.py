"""Prepare checkout-local pytest-testmon state for plain ``devtools verify``.

Pytest-testmon already owns interrupted-run recovery, failing/new test
selection, node deletion, and dependency replacement.  This module therefore
does only the checkout boundary work that the plugin cannot do itself:

* derive the native ``--testmon-env`` name from collection semantics;
* validate that the local SQLite database contains that environment;
* require changed executable modules to occur in its dependency graph;
* remove only an invalid checkout-owned database and its SQLite sidecars;
* optionally copy a matching main-checkout database into a linked worktree by
  SQLite online backup plus atomic rename.

There are no seed markers, completion stamps, shard ledgers, or release grants.
An absent graph is reported to the caller.  Only an explicitly requested
complete-corpus run may build a new environment; plain verification reuses a
compatible graph or refuses before pytest starts.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

TESTMON_DATA_RELPATH = Path(".cache/testmon/testmondata")
TESTMON_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
NATIVE_TESTMON_LIFECYCLE_LOCK_TIMEOUT_S = 60.0
_DESCRIPTOR_FD_ROOTS = (Path("/proc/self/fd"), Path("/dev/fd"))

NativeStateStatus = Literal["absent", "valid", "invalid", "incomplete"]
NativeSelectionMode = Literal["all", "bootstrap", "affected"]
ASTClassification = Literal["declaration-only", "executable", "source-unreadable"]

_ENVIRONMENT_INPUTS = (
    "uv.lock",
    "pyproject.toml",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    # Keep the conventional repository-root hook as an absent-path sentinel:
    # creating it changes collection even though it was not present when the
    # previous environment was named.
    "conftest.py",
    # NOT devtools/verify.py. That orchestrator is ~3,500 lines of flag parsing,
    # output formatting, receipt bookkeeping and retention, none of which
    # changes what pytest collects -- yet hashing it meant a COMMENT there
    # discarded every graph and forced a full-corpus bootstrap (~9.5x a warm
    # run). On the branch that introduced this split, 16 of 132 commits touched
    # a digest input, most of them fixes to the verification harness itself.
    # The collection-affecting values it used to carry now live in
    # pytest_collection_contract, which IS hashed, so a genuine change to
    # markers, plugins, ini overrides or collection roots still invalidates.
    # verify.py's behaviour remains covered the ordinary way: the tests that
    # import it carry real testmon edges to it.
    "devtools/pytest_collection_contract.py",
)
_PYTEST_ENVIRONMENT_KEYS = (
    "HYPOTHESIS_PROFILE",
    "POLYLOGUE_CI",
)


def _is_ignored_native_testmon_path(relative: str) -> bool:
    return relative == "tests/benchmarks" or relative.startswith("tests/benchmarks/")


@dataclass(frozen=True, slots=True)
class NativeTestmonEnvironment:
    name: str
    corpus_count: int
    corpus_digest: str
    nodeids: tuple[str, ...]
    fingerprinted_files: frozenset[str]


@dataclass(frozen=True, slots=True)
class NativeTestmonState:
    status: NativeStateStatus
    reason: str
    environment: NativeTestmonEnvironment | None = None
    missing_executable_paths: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.environment is not None

    @property
    def resumable(self) -> bool:
        """A graph that holds real recorded work but does not yet cover every changed module.

        Distinct from ``invalid``: the database is structurally sound and its
        environment row is unambiguous, it simply has not fingerprinted every
        executable path yet -- which is the normal state of a bootstrap that
        was interrupted. Such a graph must not drive affected selection (that
        would silently skip tests), but discarding it throws away every test
        the interrupted run already recorded and guarantees the next
        invocation starts from zero again.
        """
        return self.status == "incomplete" and self.environment is not None


def native_testmon_fallback_allowed(*states: NativeTestmonState) -> bool:
    """Allow a candidate fallback only when no observed state is invalid."""
    return all(state.status != "invalid" for state in states)


@dataclass(frozen=True, slots=True)
class NativeTestmonPreparation:
    environment_name: str
    selection_mode: NativeSelectionMode
    local_state: NativeTestmonState
    copied_from: Path | None
    removed_paths: tuple[Path, ...]
    linked_worktree: bool
    main_checkout: Path | None
    fallback_allowed: bool


@dataclass(frozen=True, slots=True)
class NativeTestmonSourceBinding:
    """A private SQLite filename family retained across validation and copy."""

    directory_descriptor: int
    descriptor: int
    sidecar_descriptors: tuple[tuple[str, int], ...]
    data_path: Path
    private_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeTestmonChangeImpact:
    """Changed inputs that native Python tracing can and cannot select."""

    executable_paths: tuple[str, ...]
    runtime_data_paths: tuple[str, ...]


class NativeTestmonRepairError(RuntimeError):
    """The exact derived testmon state could not be repaired safely."""


class NativeTestmonDeadlineError(NativeTestmonRepairError):
    """The verify invocation deadline expired during native-state preparation."""


class NativeTestmonLifecycleLockTimeoutError(NativeTestmonRepairError):
    """The shared native testmon lifecycle lock did not become available."""


@contextlib.contextmanager
def native_testmon_lifecycle_lock(
    repo_root: Path,
    *,
    timeout_s: float = NATIVE_TESTMON_LIFECYCLE_LOCK_TIMEOUT_S,
    waiter_label: str = "testmon",
) -> Iterator[None]:
    """Serialize native testmon state changes for one checkout across processes."""
    cache = repo_root.resolve() / ".cache"
    try:
        mode = cache.lstat().st_mode
    except FileNotFoundError:
        cache.mkdir(exist_ok=True)
        mode = cache.lstat().st_mode
    except OSError as exc:
        raise NativeTestmonRepairError(f"cannot inspect native testmon lock parent {cache}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise NativeTestmonRepairError(f"native testmon lock parent is not an owned directory: {cache}")
    lock_path = cache / "native-testmon-lifecycle.lock"
    directory_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        directory_descriptor = os.open(
            cache,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_directory = os.fstat(directory_descriptor)
        current_directory = cache.lstat()
        if not stat.S_ISDIR(opened_directory.st_mode) or (opened_directory.st_dev, opened_directory.st_ino) != (
            current_directory.st_dev,
            current_directory.st_ino,
        ):
            raise NativeTestmonRepairError(f"native testmon lock parent changed while binding: {cache}")
        lock_descriptor = os.open(
            lock_path.name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        opened_lock = os.fstat(lock_descriptor)
        current_lock = os.stat(lock_path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened_lock.st_mode)
            or not stat.S_ISREG(current_lock.st_mode)
            or opened_lock.st_nlink != 1
            or current_lock.st_nlink != 1
            or (opened_lock.st_dev, opened_lock.st_ino) != (current_lock.st_dev, current_lock.st_ino)
        ):
            raise NativeTestmonRepairError(
                f"native testmon lifecycle lock is not an owned single-link regular file: {lock_path}"
            )
    except OSError as exc:
        if lock_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(lock_descriptor)
            lock_descriptor = None
        if directory_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(directory_descriptor)
            directory_descriptor = None
        raise NativeTestmonRepairError(f"cannot bind native testmon lifecycle lock {lock_path}: {exc}") from exc
    try:
        assert lock_descriptor is not None
        with os.fdopen(lock_descriptor, "r+", encoding="utf-8") as handle:
            lock_descriptor = None
            deadline = time.monotonic() + max(0.0, timeout_s)
            announced_wait = False
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise NativeTestmonLifecycleLockTimeoutError(
                            f"timed out waiting for native testmon lifecycle lock after {timeout_s:.1f}s"
                        ) from exc
                    if not announced_wait:
                        handle.seek(0)
                        holder = handle.read().strip() or "another testmon lifecycle"
                        print(
                            f"{waiter_label}: waiting for native testmon lifecycle lock ({holder})",
                            file=sys.stderr,
                            flush=True,
                        )
                        announced_wait = True
                    time.sleep(min(0.05, remaining))
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}")
            handle.flush()
            try:
                yield
            finally:
                handle.seek(0)
                handle.truncate()
    finally:
        if lock_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(lock_descriptor)
        if directory_descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(directory_descriptor)


@contextlib.contextmanager
def native_testmon_lifecycle_locks(
    repo_roots: Iterable[Path],
    *,
    timeout_s: float = NATIVE_TESTMON_LIFECYCLE_LOCK_TIMEOUT_S,
    waiter_label: str = "testmon",
) -> Iterator[None]:
    """Acquire several checkout locks in path order to avoid cross-lane deadlocks."""
    roots = sorted({path.resolve() for path in repo_roots}, key=str)
    with contextlib.ExitStack() as stack:
        for root in roots:
            stack.enter_context(native_testmon_lifecycle_lock(root, timeout_s=timeout_s, waiter_label=waiter_label))
        yield


@contextlib.contextmanager
def _optional_native_testmon_source_lock(
    source_lock: contextlib.AbstractContextManager[bool],
) -> Iterator[bool]:
    """Treat an unavailable linked source as absent without leaking its lock."""
    try:
        source_available = source_lock.__enter__()
    except NativeTestmonRepairError:
        yield False
        return
    try:
        yield source_available
    except BaseException as exc:
        source_lock.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        source_lock.__exit__(None, None, None)


def _ensure_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise NativeTestmonDeadlineError("verify invocation deadline expired during native testmon preparation")


def _remaining_timeout(deadline_monotonic: float | None, maximum: float) -> float:
    _ensure_deadline(deadline_monotonic)
    if deadline_monotonic is None:
        return maximum
    return max(0.001, min(maximum, deadline_monotonic - time.monotonic()))


def _fingerprint_inputs(
    root: Path,
    relative_paths: Sequence[str],
    *,
    deadline_monotonic: float | None = None,
) -> str:
    digest = hashlib.sha256()
    for relative in relative_paths:
        _ensure_deadline(deadline_monotonic)
        digest.update(relative.encode())
        digest.update(b"\0")
        try:
            with (root / relative).open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    _ensure_deadline(deadline_monotonic)
        except OSError:
            digest.update(b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def _pytest_plugins_assignment(node: ast.stmt) -> ast.expr | None:
    match node:
        case ast.Assign(targets=targets, value=value) if any(
            isinstance(target, ast.Name) and target.id == "pytest_plugins" for target in targets
        ):
            return value
        case ast.AnnAssign(target=ast.Name(id="pytest_plugins"), value=value):
            return value
        case _:
            return None


def _indirect_pytest_plugins_declaration(node: ast.stmt) -> bool:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Constant)
            and child.slice.value == "pytest_plugins"
        ):
            return True
        if isinstance(child, ast.Call):
            if (
                isinstance(child.func, ast.Name)
                and child.func.id == "setattr"
                and len(child.args) >= 2
                and isinstance(child.args[1], ast.Constant)
                and child.args[1].value == "pytest_plugins"
            ):
                return True
            if any(keyword.arg == "pytest_plugins" for keyword in child.keywords):
                return True
    return False


def _declared_pytest_plugin_names(
    root: Path,
    *,
    deadline_monotonic: float | None = None,
) -> set[str]:
    """Read static local plugin declarations that pytest loads at collection."""
    names: set[str] = set()
    candidates: set[Path] = set()
    candidates.add(root / "conftest.py")
    for path in root.glob("tests/**/*.py"):
        _ensure_deadline(deadline_monotonic)
        if _is_ignored_native_testmon_path(path.relative_to(root).as_posix()):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "pytest_plugins" not in source:
            continue
        candidates.add(path)
    for path in root.glob("tests/**/conftest.py"):
        _ensure_deadline(deadline_monotonic)
        if _is_ignored_native_testmon_path(path.relative_to(root).as_posix()):
            continue
        candidates.add(path)
    for path in sorted(candidates):
        _ensure_deadline(deadline_monotonic)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        declaration_count = 0
        for node in tree.body:
            value = _pytest_plugins_assignment(node)
            if value is None:
                dynamic_reference = (
                    any(isinstance(child, ast.Name) and child.id == "pytest_plugins" for child in ast.walk(node))
                    or any(
                        alias.name == "pytest_plugins" or alias.asname == "pytest_plugins"
                        for child in ast.walk(node)
                        if isinstance(child, ast.Import | ast.ImportFrom)
                        for alias in child.names
                    )
                    or _indirect_pytest_plugins_declaration(node)
                )
                if dynamic_reference:
                    raise NativeTestmonRepairError(
                        f"repository pytest_plugins declaration must be one literal assignment: {path}"
                    )
                continue
            declaration_count += 1
            if declaration_count != 1:
                raise NativeTestmonRepairError(
                    f"repository pytest_plugins declaration must be one literal assignment: {path}"
                )
            try:
                declared = ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise NativeTestmonRepairError(
                    f"repository pytest_plugins declaration must be a literal string/list/tuple: {path}"
                ) from exc
            if isinstance(declared, str):
                names.add(declared)
            elif isinstance(declared, tuple | list) and all(isinstance(name, str) for name in declared):
                names.update(declared)
            else:
                raise NativeTestmonRepairError(
                    f"repository pytest_plugins declaration must contain only literal plugin names: {path}"
                )
    return names


def _active_local_pytest_plugin_paths(
    root: Path,
    *,
    deadline_monotonic: float | None = None,
) -> set[str]:
    """Resolve collection-active local pytest plugins regardless of filename."""
    paths: set[str] = set()
    plugin_names = _declared_pytest_plugin_names(root, deadline_monotonic=deadline_monotonic)
    for raw_name in plugin_names:
        _ensure_deadline(deadline_monotonic)
        module_name = raw_name.strip()
        if not module_name or any(part in {"", ".", ".."} for part in module_name.split(".")):
            continue
        module_path = Path(*module_name.split("."))
        module_file = root / module_path.with_suffix(".py")
        if module_file.is_file() and not _is_ignored_native_testmon_path(module_file.relative_to(root).as_posix()):
            paths.add(module_file.relative_to(root).as_posix())
        package = root / module_path
        if (package / "__init__.py").is_file():
            for path in package.rglob("*.py"):
                _ensure_deadline(deadline_monotonic)
                if path.is_file() and not _is_ignored_native_testmon_path(path.relative_to(root).as_posix()):
                    paths.add(path.relative_to(root).as_posix())
    return paths


def _environment_input_paths(
    root: Path,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[str, ...]:
    """Discover collection and managed-pytest harness inputs."""
    paths = set(_ENVIRONMENT_INPUTS)
    # NOT tests/**/conftest.py. Every collected test executes conftest, so
    # testmon holds an edge to it from every test and selects precisely when it
    # changes -- hashing it here is the same double-counting that made editing
    # the orchestrator discard the graph. Keeping it also made the transparent
    # worktree graph copy unreachable: that copy requires the main checkout's
    # graph to be valid under the LANE's digest, and a lane exists precisely
    # because it is on a different branch, so any harness file differing between
    # the two branches permanently defeated it. Measured on this repository: 5 of
    # 22 digest inputs differed between the main checkout and both live lanes,
    # and every one of the five was harness implementation rather than
    # collection semantics.
    patterns = ("devtools/pytest*.py",)
    for pattern in patterns:
        for path in root.glob(pattern):
            _ensure_deadline(deadline_monotonic)
            if path.is_file() and not _is_ignored_native_testmon_path(path.relative_to(root).as_posix()):
                paths.add(path.relative_to(root).as_posix())
    paths.update(_active_local_pytest_plugin_paths(root, deadline_monotonic=deadline_monotonic))
    return tuple(sorted(paths))


def _installed_distributions() -> tuple[tuple[str, str], ...]:
    distributions: list[tuple[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        version = distribution.version
        if not name or not version:
            raise NativeTestmonRepairError("active Python distributions are not fully identifiable")
        distributions.append((name.casefold(), version))
    return tuple(sorted(distributions))


def testmon_environment_digest(
    repo_root: Path,
    *,
    pytest_profile: str = "default",
    pytest_environment: Mapping[str, str | None] | None = None,
    deadline_monotonic: float | None = None,
) -> str:
    """Return the native testmon environment name for collection semantics."""
    root = repo_root.resolve()
    _ensure_deadline(deadline_monotonic)
    payload = {
        "protocol": 1,
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": platform.python_version(),
            "abi_flags": getattr(sys, "abiflags", ""),
            "platform": platform.platform(),
        },
        "distributions": _installed_distributions(),
        "inputs": _fingerprint_inputs(
            root,
            _environment_input_paths(root, deadline_monotonic=deadline_monotonic),
            deadline_monotonic=deadline_monotonic,
        ),
        "pytest_environment": {
            key: (os.environ.get(key) if pytest_environment is None else pytest_environment.get(key))
            for key in _PYTEST_ENVIRONMENT_KEYS
        },
        "pytest_profile": pytest_profile,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"polylogue-{hashlib.sha256(encoded).hexdigest()}"


def _is_docstring(node: ast.stmt, *, first: bool) -> bool:
    return (
        first
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
            and node.attr == "TYPE_CHECKING"
        )
    )


def _is_literal_all_assignment(node: ast.Assign) -> bool:
    """Recognize an export declaration that coverage cannot fingerprint."""
    if not all(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
        return False
    return isinstance(node.value, (ast.List, ast.Tuple)) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.value.elts
    )


def _body_is_executable(body: list[ast.stmt]) -> bool:
    for index, node in enumerate(body):
        if _is_docstring(node, first=index == 0):
            continue
        if isinstance(node, ast.Pass):
            continue
        if isinstance(node, ast.ImportFrom) and node.module in {"typing", "collections.abc"}:
            # Type vocabulary has no traceable behavior on its own. A module
            # whose remaining body is declarations stays outside testmon's
            # coverage graph, just like a pure enum declaration.
            continue
        if isinstance(node, ast.Import) and all(alias.name == "builtins" for alias in node.names):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and all(alias.name == "annotations" for alias in node.names)
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "enum"
            and all(alias.name in {"Enum", "IntEnum", "StrEnum"} for alias in node.names)
        ):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and (isinstance(node.value.value, str) or node.value.value is Ellipsis)
        ):
            continue
        if isinstance(node, ast.AnnAssign) and node.value is None:
            continue
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            # The guarded body is deliberately invisible at runtime. An else
            # branch does execute and therefore retains ordinary classification.
            if _body_is_executable(node.orelse):
                return True
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list or node.args.defaults or any(value is not None for value in node.args.kw_defaults):
                return True
            if _body_is_executable(node.body):
                return True
            continue
        if isinstance(node, ast.ClassDef):
            if _is_pure_enum_declaration(node) or _is_pure_protocol_declaration(node):
                continue
            if node.decorator_list or node.bases or node.keywords or _body_is_executable(node.body):
                return True
            continue
        if isinstance(node, ast.Assign):
            if _is_literal_all_assignment(node):
                continue
            return True
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                return True
            continue
        return True
    return False


def _is_pure_enum_declaration(node: ast.ClassDef) -> bool:
    """Recognize enum value declarations that tracing cannot observe usefully."""
    bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
    if not bases.intersection({"Enum", "IntEnum", "StrEnum"}):
        return False
    for member in node.body:
        if (
            isinstance(member, ast.Expr)
            and isinstance(member.value, ast.Constant)
            and isinstance(member.value.value, str)
        ):
            continue
        if isinstance(member, (ast.Assign, ast.AnnAssign, ast.Pass)):
            continue
        return False
    return True


def _is_pure_protocol_declaration(node: ast.ClassDef) -> bool:
    """Recognize runtime-checkable Protocol shapes that coverage cannot fingerprint."""
    bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
    if "Protocol" not in bases:
        return False
    for decorator in node.decorator_list:
        if not (isinstance(decorator, ast.Name) and decorator.id == "runtime_checkable"):
            return False
    return not _body_is_executable(node.body)


def classify_source_ast(source_path: Path) -> ASTClassification:
    """Classify whether a module contains executable runtime behavior."""
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except OSError:
        return "source-unreadable"
    except (SyntaxError, UnicodeDecodeError):
        return "executable"
    return "executable" if _body_is_executable(tree.body) else "declaration-only"


def _safe_relative_path(raw: str) -> str | None:
    normalized = PurePosixPath(raw.replace("\\", "/"))
    if normalized.is_absolute() or not normalized.parts or ".." in normalized.parts:
        return None
    return str(normalized)


def executable_python_paths(repo_root: Path, paths: Iterable[str]) -> tuple[str, ...]:
    """Return changed Python paths whose runtime behavior needs graph edges.

    Deleted modules are excluded: an edge for a nonexistent file is
    unrecordable, so requiring one makes ``incomplete`` permanent -- no
    rebuild, however complete, can ever clear it (observed 2026-08-18: a
    branch that renamed a test module could not produce a valid graph even
    after a full 20,506-test bootstrap). Selection soundness does not need
    the requirement -- pytest-testmon itself detects a deleted dependency
    when a recorded fingerprint no longer matches, and selects the
    dependent tests.
    """
    root = repo_root.resolve()
    executable: list[str] = []
    for raw in sorted(set(paths)):
        relative = _safe_relative_path(raw)
        if relative is None or not relative.endswith(".py"):
            continue
        source = root / relative
        if not source.exists() or not source.is_file():
            continue
        if classify_source_ast(source) == "declaration-only":
            continue
        executable.append(relative)
    return tuple(executable)


def classify_native_testmon_changes(repo_root: Path, paths: Iterable[str]) -> NativeTestmonChangeImpact:
    """Classify changed product inputs against native testmon's trace boundary.

    Python tracing does not observe non-Python runtime data. Non-Python files
    under the shipped ``polylogue`` package or the test runtime tree are
    therefore outside the native graph and cannot safely use affected
    selection. The caller must run the complete native corpus for those
    changes. This convention covers additions, deletions, and all data formats
    without a filename registry.
    """
    normalized = tuple(relative for raw in sorted(set(paths)) if (relative := _safe_relative_path(raw)) is not None)
    native_paths = tuple(relative for relative in normalized if not _is_ignored_native_testmon_path(relative))
    runtime_data = tuple(
        relative
        for relative in native_paths
        if (not relative.endswith(".py") and relative.startswith(("polylogue/", "tests/")))
        or (
            relative.endswith(".py")
            and relative.startswith(("devtools/", "polylogue/", "tests/"))
            and classify_source_ast(repo_root / relative) == "declaration-only"
        )
        or relative.startswith("packaging/")
    )
    return NativeTestmonChangeImpact(
        executable_paths=executable_python_paths(
            repo_root,
            (relative for relative in native_paths if relative not in runtime_data),
        ),
        runtime_data_paths=runtime_data,
    )


def _readonly_uri(path: Path) -> str:
    return f"{path.absolute().as_uri()}?mode=ro"


def _testmon_schema_version() -> int:
    module = importlib.import_module("testmon.db")
    value = getattr(module, "DATA_VERSION", None)
    if not isinstance(value, int):
        raise NativeTestmonRepairError("pytest-testmon does not expose an integer database version")
    return value


def _digest_nodeids(nodeids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(nodeids).encode()).hexdigest()


def inspect_native_testmon_environment(
    data_path: Path,
    *,
    environment_name: str,
    required_executable_paths: Sequence[str] = (),
    data_fd: int | None = None,
    bound_data_path: Path | None = None,
    bound_sidecar_fds: Mapping[str, int] | None = None,
    deadline_monotonic: float | None = None,
) -> NativeTestmonState:
    """Validate one native environment without interpreting plugin internals."""
    _ensure_deadline(deadline_monotonic)
    sqlite_data_path = bound_data_path or (_descriptor_bound_path(data_fd) if data_fd is not None else data_path)
    sidecars = tuple(Path(f"{sqlite_data_path}{suffix}") for suffix in TESTMON_SIDECAR_SUFFIXES)
    if data_fd is None and not data_path.exists():
        if any(path.exists() or path.is_symlink() for path in sidecars):
            return NativeTestmonState("invalid", "SQLite sidecars exist without the owned database")
        return NativeTestmonState("absent", "native testmon database is absent")
    try:
        state = os.fstat(data_fd) if data_fd is not None else data_path.lstat()
        bound_state = sqlite_data_path.lstat() if bound_data_path is not None else state
    except OSError as exc:
        return NativeTestmonState("invalid", f"cannot inspect native testmon database: {exc}")
    if not stat.S_ISREG(state.st_mode) or (data_fd is None and state.st_nlink != 1) or state.st_nlink < 1:
        return NativeTestmonState("invalid", "native testmon database is not a single-link regular file")
    if bound_data_path is not None and (bound_state.st_dev, bound_state.st_ino) != (state.st_dev, state.st_ino):
        return NativeTestmonState("invalid", "native testmon database changed while binding")
    bound_sidecar_fds = bound_sidecar_fds or {}
    for suffix, sidecar in zip(TESTMON_SIDECAR_SUFFIXES, sidecars, strict=True):
        descriptor = bound_sidecar_fds.get(suffix)
        try:
            sidecar_state = sidecar.lstat()
        except FileNotFoundError:
            if descriptor is not None:
                return NativeTestmonState("invalid", f"bound native testmon sidecar disappeared: {sidecar}")
            continue
        except OSError as exc:
            return NativeTestmonState("invalid", f"cannot inspect native testmon sidecar {sidecar}: {exc}")
        if descriptor is not None:
            retained = os.fstat(descriptor)
            if not stat.S_ISREG(retained.st_mode) or (sidecar_state.st_dev, sidecar_state.st_ino) != (
                retained.st_dev,
                retained.st_ino,
            ):
                return NativeTestmonState("invalid", f"bound native testmon sidecar changed: {sidecar}")
        elif bound_data_path is not None:
            if not stat.S_ISREG(sidecar_state.st_mode) or sidecar_state.st_nlink != 1:
                return NativeTestmonState("invalid", f"unexpected native testmon sidecar: {sidecar}")
        elif not stat.S_ISREG(sidecar_state.st_mode) or sidecar_state.st_nlink != 1:
            return NativeTestmonState("invalid", f"native testmon sidecar is not a single-link regular file: {sidecar}")
    # A killed writer (supervisor SIGTERM, operator interrupt) leaves a WAL
    # that only a read-WRITE connection can replay -- the read-only URI below
    # fails with SQLITE_READONLY_RECOVERY, the state is judged invalid, and
    # removal then destroys EVERY environment's accumulated graph in the
    # shared file (observed 2026-08-18: one killed bootstrap cost the warm
    # master graph and a 20,500-test rebuild). Checkpoint first so ordinary
    # crash recovery happens instead.
    if any(path.exists() for path in sidecars):
        try:
            # Recover the public filename family before reopening the retained
            # descriptor alias.  SQLite derives WAL/SHM names from the opened
            # basename, so checkpointing only the alias can leave the public
            # WAL untouched and make a sound graph appear invalid later.
            recovery_path = sqlite_data_path
            recovery = sqlite3.connect(recovery_path, timeout=_remaining_timeout(deadline_monotonic, 10))
            try:
                checkpoint = recovery.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            finally:
                recovery.close()
        except (sqlite3.Error, OSError) as exc:
            return NativeTestmonState("invalid", f"cannot recover native testmon sidecars: {exc}")
        if checkpoint is None or checkpoint[0] != 0:
            return NativeTestmonState("invalid", f"native testmon sidecar checkpoint failed: {checkpoint}")
    try:
        with (
            contextlib.closing(
                sqlite3.connect(
                    _readonly_uri(sqlite_data_path),
                    uri=True,
                    timeout=_remaining_timeout(deadline_monotonic, 10),
                )
            ) as connection,
            connection,
        ):
            if deadline_monotonic is not None:
                connection.set_progress_handler(lambda: int(time.monotonic() >= deadline_monotonic), 1_000)
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            _ensure_deadline(deadline_monotonic)
            if quick_check is None or quick_check[0] != "ok":
                return NativeTestmonState("invalid", "SQLite quick_check failed")
            version_row = connection.execute("PRAGMA user_version").fetchone()
            if version_row is None or version_row[0] != _testmon_schema_version():
                return NativeTestmonState("invalid", "pytest-testmon database schema version changed")
            environment_rows = connection.execute(
                "SELECT id FROM environment WHERE environment_name = ? ORDER BY id DESC",
                (environment_name,),
            ).fetchall()
            _ensure_deadline(deadline_monotonic)
            if len(environment_rows) != 1:
                if not environment_rows:
                    # A sound database that simply does not carry this
                    # environment is not damaged state. The file is shared by
                    # every environment name (the hypothesis-profile fallback
                    # probes two in a single invocation), so reporting "invalid"
                    # here invites the caller to delete another environment's
                    # graph on a routine miss.
                    return NativeTestmonState("absent", f"native environment {environment_name!r} is absent")
                return NativeTestmonState("invalid", "native environment is ambiguous")
            environment_id = int(environment_rows[0][0])
            nodeids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT test_name FROM test_execution WHERE environment_id = ? ORDER BY test_name",
                    (environment_id,),
                ).fetchall()
                if isinstance(row[0], str) and row[0]
            )
            _ensure_deadline(deadline_monotonic)
            if not nodeids:
                # An environment row with no recorded executions is EMPTY, not
                # damaged. pytest creates the row at startup, so this is exactly
                # what a bootstrap interrupted before its first test completes
                # leaves behind -- and calling it "invalid" makes the caller
                # delete the whole shared SQLite file, taking every OTHER
                # environment's graph with it. That is the loop no number of
                # retries escapes: kill a bootstrap once, and the next run starts
                # from zero, and so does the one after.
                #
                # Reported as absent instead: there is nothing here to reuse, so
                # this environment bootstraps, while graphs belonging to other
                # environment names survive untouched.
                return NativeTestmonState("absent", f"native environment {environment_name!r} has no recorded tests")
            if len(nodeids) != len(set(nodeids)):
                return NativeTestmonState("invalid", "native environment has no unique collected corpus")
            uncovered = connection.execute(
                """
                SELECT COUNT(*)
                FROM test_execution AS execution
                LEFT JOIN test_execution_file_fp AS edge ON edge.test_execution_id = execution.id
                WHERE execution.environment_id = ? AND edge.test_execution_id IS NULL
                """,
                (environment_id,),
            ).fetchone()
            _ensure_deadline(deadline_monotonic)
            if uncovered is None or int(uncovered[0]) != 0:
                return NativeTestmonState("invalid", "native environment has tests without dependency placeholders")
            raw_files = connection.execute(
                """
                SELECT DISTINCT fingerprint.filename
                FROM test_execution AS execution
                JOIN test_execution_file_fp AS edge ON edge.test_execution_id = execution.id
                JOIN file_fp AS fingerprint ON fingerprint.id = edge.fingerprint_id
                WHERE execution.environment_id = ?
                """,
                (environment_id,),
            ).fetchall()
            _ensure_deadline(deadline_monotonic)
    except NativeTestmonDeadlineError:
        raise
    except (NativeTestmonRepairError, OSError, sqlite3.Error, TypeError, ValueError) as exc:
        _ensure_deadline(deadline_monotonic)
        return NativeTestmonState("invalid", f"native testmon database is unreadable: {exc}")
    fingerprinted = frozenset(
        relative
        for row in raw_files
        if row and isinstance(row[0], str)
        if (relative := _safe_relative_path(row[0])) is not None
    )
    required = tuple(sorted(set(required_executable_paths)))
    missing = tuple(path for path in required if path not in fingerprinted)
    environment = NativeTestmonEnvironment(
        name=environment_name,
        corpus_count=len(nodeids),
        corpus_digest=_digest_nodeids(nodeids),
        nodeids=nodeids,
        fingerprinted_files=fingerprinted,
    )
    if missing:
        return NativeTestmonState(
            "incomplete",
            "changed executable modules are absent from the native dependency graph",
            environment,
            missing,
        )
    return NativeTestmonState("valid", "native environment is current", environment)


def canonical_test_nodeid(nodeid: str) -> str:
    """One id per test across lane shapes.

    xdist loadgroup runs record ``path::test[param]@group`` while single-process
    lanes record ``path::test[param]``; graph rows, certificates, and coverage
    math must compare on ONE form or every cross-shape comparison invents
    phantom missing/attested tests (509 of them, observed 2026-08-18). Only a
    trailing ``@group`` outside any param bracket is stripped, so params that
    themselves contain ``@`` survive.
    """
    base, sep, suffix = nodeid.rpartition("@")
    if sep and suffix and "]" not in suffix and ":" not in suffix and "/" not in suffix:
        return base
    return nodeid


_CERTIFIED_CORPUS_TABLE = "polylogue_certified_corpus"


def read_certified_corpus(data_path: Path, environment_name: str) -> frozenset[str] | None:
    """The corpus a completed covered run certified for this environment.

    ``None`` means no completed run has certified the environment yet -- an
    interrupted bootstrap leaves a graph whose recorded rows LOOK like a
    complete corpus (the corpus is derived from the graph, so it is blind to
    its own losses). Attestation is sound only against a certificate.
    """
    if not data_path.exists():
        return None
    try:
        with contextlib.closing(sqlite3.connect(_readonly_uri(data_path), uri=True)) as connection:
            row = connection.execute(
                f"SELECT nodeids_json FROM {_CERTIFIED_CORPUS_TABLE} WHERE environment_name = ?",
                (environment_name,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None or not isinstance(row[0], str):
        return None
    try:
        nodeids = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(nodeids, list):
        return None
    return frozenset(str(nodeid) for nodeid in nodeids)


def write_certified_corpus(repo_root: Path, environment_name: str, nodeids: Iterable[str]) -> bool:
    """Record the corpus of a completed covered run alongside the graph."""
    data_path = repo_root.resolve() / TESTMON_DATA_RELPATH
    if not data_path.exists():
        return False
    payload = json.dumps(sorted({canonical_test_nodeid(nodeid) for nodeid in nodeids}))
    try:
        with contextlib.closing(sqlite3.connect(data_path)) as connection:
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {_CERTIFIED_CORPUS_TABLE} ("
                "environment_name TEXT PRIMARY KEY,"
                "nodeids_json TEXT NOT NULL,"
                "certified_at TEXT NOT NULL)"
            )
            connection.execute(
                f"INSERT INTO {_CERTIFIED_CORPUS_TABLE} (environment_name, nodeids_json, certified_at)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(environment_name) DO UPDATE SET"
                " nodeids_json = excluded.nodeids_json, certified_at = excluded.certified_at",
                (environment_name, payload, datetime.now(UTC).isoformat()),
            )
            connection.commit()
    except sqlite3.Error:
        return False
    return True


def certified_attestation_violation(
    repo_root: Path,
    *,
    environment_name: str,
    current_nodeids: Sequence[str],
    certificate_data_path: Path | None = None,
) -> str | None:
    """Why this graph may NOT attest unexecuted tests, or None when it may.

    A graph row set is blind to its own losses (an interrupted serial lane or
    damaged rows shrink the corpus AND the coverage claim together), so
    attestation is checked against the certificate the last completed covered
    run wrote. Tests whose test file no longer exists are legitimately gone
    and do not violate the certificate.
    """
    root = repo_root.resolve()
    certified = read_certified_corpus(
        certificate_data_path if certificate_data_path is not None else root / TESTMON_DATA_RELPATH,
        environment_name,
    )
    if certified is None:
        return "no completed covered run has certified this environment's corpus"
    current = {canonical_test_nodeid(nodeid) for nodeid in current_nodeids}
    lost = {
        nodeid
        for nodeid in {canonical_test_nodeid(entry) for entry in certified} - current
        if (root / nodeid.split("::", 1)[0]).is_file()
    }
    if lost:
        sample = ", ".join(sorted(lost)[:5])
        return f"{len(lost)} certified test(s) are missing from the graph (e.g. {sample})"
    return None


def _owned_paths(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root.resolve()
    _validate_owned_state_parents(root)
    data = root / TESTMON_DATA_RELPATH
    return (data, *(Path(f"{data}{suffix}") for suffix in TESTMON_SIDECAR_SUFFIXES))


def _validate_owned_state_parents(repo_root: Path) -> None:
    """Reject state paths that escape the checkout through a symlink parent."""
    parent = repo_root.resolve()
    for part in TESTMON_DATA_RELPATH.parent.parts:
        parent /= part
        try:
            mode = parent.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise NativeTestmonRepairError(f"cannot inspect owned testmon parent {parent}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise NativeTestmonRepairError(f"refusing symlinked owned testmon parent {parent}")
        if not stat.S_ISDIR(mode):
            raise NativeTestmonRepairError(f"owned testmon parent is not a directory: {parent}")


def validate_native_testmon_state_ownership(repo_root: Path) -> None:
    """Reject parent or file replacement before managed SQLite access."""
    for path in _owned_paths(repo_root):
        try:
            state = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NativeTestmonRepairError(f"cannot inspect owned testmon path {path}: {exc}") from exc
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise NativeTestmonRepairError(f"owned testmon path is not a single-link regular file: {path}")


def remove_invalid_native_testmon_state(repo_root: Path) -> tuple[Path, ...]:
    """Remove only the exact checkout-owned SQLite file and known sidecars."""
    removed: list[Path] = []
    for path in _owned_paths(repo_root):
        try:
            state = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NativeTestmonRepairError(f"cannot inspect owned testmon path {path}: {exc}") from exc
        if stat.S_ISDIR(state.st_mode):
            raise NativeTestmonRepairError(f"refusing to remove directory at owned SQLite path {path}")
        if stat.S_ISREG(state.st_mode) and state.st_nlink != 1:
            raise NativeTestmonRepairError(f"refusing to remove hard-linked owned SQLite path {path}")
        try:
            path.unlink()
        except OSError as exc:
            raise NativeTestmonRepairError(f"cannot remove invalid owned testmon path {path}: {exc}") from exc
        removed.append(path)
    return tuple(removed)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_owned_testmon_directory(repo_root: Path, *, create: bool) -> int:
    """Open the owned testmon directory without following any path component."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise NativeTestmonRepairError("safe testmon directory operations require O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow
    current = os.open(repo_root.resolve(), flags)
    try:
        for part in TESTMON_DATA_RELPATH.parent.parts:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o755, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_owned_testmon_child(
    directory_fd: int,
    name: str,
    *,
    private_name: str | None = None,
) -> tuple[int, Path, str]:
    """Open and retain the private bound child before exposing its descriptor path."""
    child_fd: int | None = None
    bound_fd: int | None = None
    try:
        child_fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        opened = os.fstat(child_fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or opened.st_nlink != 1
            or current.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise NativeTestmonRepairError(f"owned testmon child changed while binding: {name}")
        private_name = private_name or f".{name}.bound-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        bound_child = _descriptor_bound_path(child_fd)
        os.link(bound_child, private_name, dst_dir_fd=directory_fd, follow_symlinks=True)
        # Open the link before validating it, then use the retained descriptor
        # below. Reopening ``private_name`` after validation would let a
        # replacement of the .bound-* entry redirect SQLite to another inode.
        bound_fd = os.open(
            private_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        private_identity = os.fstat(bound_fd)
        if not stat.S_ISREG(private_identity.st_mode) or (opened.st_dev, opened.st_ino) != (
            private_identity.st_dev,
            private_identity.st_ino,
        ):
            raise NativeTestmonRepairError(f"owned testmon child link changed while binding: {name}")
        os.close(child_fd)
        child_fd = None
        return bound_fd, _descriptor_bound_path(bound_fd), private_name
    except NativeTestmonRepairError:
        if private_name is not None:
            _unlink_bound_entry_if_owned(
                directory_fd,
                private_name,
                child_fd,
            )
        if bound_fd is not None:
            os.close(bound_fd)
        if child_fd is not None:
            os.close(child_fd)
        raise
    except OSError as exc:
        if private_name is not None:
            _unlink_bound_entry_if_owned(
                directory_fd,
                private_name,
                child_fd,
            )
        if bound_fd is not None:
            os.close(bound_fd)
        if child_fd is not None:
            os.close(child_fd)
        raise NativeTestmonRepairError(f"cannot bind owned testmon child {name}: {exc}") from exc


def _descriptor_bound_path(file_descriptor: int, *, owner_pid: int | None = None) -> Path:
    """Return a path that reopens the exact held descriptor, or fail closed."""
    roots = (Path("/proc") / str(owner_pid) / "fd",) if owner_pid is not None else _DESCRIPTOR_FD_ROOTS
    for root in roots:
        candidate = root / str(file_descriptor)
        try:
            os.stat(candidate)
        except OSError:
            continue
        return candidate
    raise NativeTestmonRepairError(
        f"no descriptor-bound filesystem namespace is available for file descriptor {file_descriptor}"
    )


def _unlink_bound_entry_if_owned(directory_fd: int, private_name: str, retained_fd: int | None) -> None:
    """Remove a private binding only while its directory entry names the retained inode."""
    if retained_fd is None:
        return
    try:
        retained = os.fstat(retained_fd)
        current = os.stat(private_name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return
    if (current.st_dev, current.st_ino) != (retained.st_dev, retained.st_ino):
        return
    with contextlib.suppress(FileNotFoundError):
        os.unlink(private_name, dir_fd=directory_fd)


def _reclaim_stale_native_testmon_bindings(directory_fd: int, name: str) -> None:
    """Remove crash-left aliases only when they still reference this source."""
    try:
        source = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        entries = os.listdir(directory_fd)
    except OSError as exc:
        raise NativeTestmonRepairError(f"cannot inspect native testmon bindings: {exc}") from exc
    private_name = re.compile(rf"\A\.{re.escape(name)}\.bound-[0-9]+-[0-9a-f]{{32}}\.tmp\Z")
    for entry in entries:
        if private_name.fullmatch(entry) is None:
            continue
        try:
            candidate = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(candidate.st_mode) and (candidate.st_dev, candidate.st_ino) == (source.st_dev, source.st_ino):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(entry, dir_fd=directory_fd)


@contextlib.contextmanager
def native_testmon_source_binding(data_path: Path) -> Iterator[NativeTestmonSourceBinding | None]:
    """Retain a private SQLite filename family for validation and copying."""
    directory_fd: int | None = None
    child_fd: int | None = None
    private_names: list[str] = []
    sidecar_descriptors: list[tuple[str, int]] = []
    try:
        try:
            present = data_path.is_file()
        except OSError as exc:
            raise NativeTestmonRepairError(f"cannot inspect native testmon source {data_path}: {exc}") from exc
        if not present:
            yield None
            return
        directory_fd = _open_owned_testmon_directory(data_path.parent.parent.parent, create=False)
        _reclaim_stale_native_testmon_bindings(directory_fd, data_path.name)
        child_fd, _bound_path, private_name = _open_owned_testmon_child(directory_fd, data_path.name)
        private_names.append(private_name)
        for suffix in TESTMON_SIDECAR_SUFFIXES:
            public_name = f"{data_path.name}{suffix}"
            try:
                os.stat(public_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            sidecar_fd, _sidecar_path, sidecar_private_name = _open_owned_testmon_child(
                directory_fd,
                public_name,
                private_name=f"{private_name}{suffix}",
            )
            sidecar_descriptors.append((suffix, sidecar_fd))
            private_names.append(sidecar_private_name)
        private_base = _descriptor_bound_path(directory_fd) / private_name
        yield NativeTestmonSourceBinding(
            directory_descriptor=directory_fd,
            descriptor=child_fd,
            sidecar_descriptors=tuple(sidecar_descriptors),
            data_path=private_base,
            private_names=tuple(private_names),
        )
    except NativeTestmonRepairError:
        raise
    except OSError as exc:
        raise NativeTestmonRepairError(f"cannot bind native testmon source {data_path}: {exc}") from exc
    finally:
        try:
            if directory_fd is not None:
                for suffix, descriptor in reversed(sidecar_descriptors):
                    _unlink_bound_entry_if_owned(directory_fd, f"{private_names[0]}{suffix}", descriptor)
                if child_fd is not None and private_names:
                    _unlink_bound_entry_if_owned(directory_fd, private_names[0], child_fd)
        finally:
            try:
                if directory_fd is not None:
                    os.close(directory_fd)
            finally:
                for _suffix, descriptor in sidecar_descriptors:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
                if child_fd is not None:
                    os.close(child_fd)


def _publish_validated_testmon_database(
    *,
    destination_directory_fd: int,
    destination_name: str,
    temporary_fd: int,
    publication_name: str,
    deadline_monotonic: float | None,
) -> None:
    """Publish the validated inode and recover from a mutable source race.

    ``rename`` is atomic, but its source is still a directory entry.  Prove
    the destination inode after the rename as well as the publication entry
    before it; if a concurrent replacement won that narrow window, retry from
    the still-open validated inode.  Every attempt is an atomic rename and the
    caller's cleanup removes both scratch names on success or failure.
    """
    validated_identity = os.fstat(temporary_fd)
    expected_identity = (validated_identity.st_dev, validated_identity.st_ino)
    for attempt in range(3):
        _ensure_deadline(deadline_monotonic)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(publication_name, dir_fd=destination_directory_fd)
        os.link(
            str(_descriptor_bound_path(temporary_fd)),
            publication_name,
            dst_dir_fd=destination_directory_fd,
            follow_symlinks=True,
        )
        publication_identity = os.stat(
            publication_name,
            dir_fd=destination_directory_fd,
            follow_symlinks=False,
        )
        if (publication_identity.st_dev, publication_identity.st_ino) != expected_identity:
            raise NativeTestmonRepairError("validated SQLite publication inode changed before rename")
        os.replace(
            publication_name,
            destination_name,
            src_dir_fd=destination_directory_fd,
            dst_dir_fd=destination_directory_fd,
        )
        published_identity = os.stat(
            destination_name,
            dir_fd=destination_directory_fd,
            follow_symlinks=False,
        )
        if (published_identity.st_dev, published_identity.st_ino) == expected_identity:
            return
        if attempt == 2:
            raise NativeTestmonRepairError("published SQLite database inode changed after atomic rename")


def _atomic_copy_sqlite_database(
    source: Path,
    destination: Path,
    *,
    environment_name: str,
    required_executable_paths: Sequence[str],
    deadline_monotonic: float | None,
    source_fd: int | None = None,
    bound_source_path: Path | None = None,
) -> None:
    _ensure_deadline(deadline_monotonic)
    source_directory_fd: int | None = None
    source_child_fd: int | None = None
    source_private_name: str | None = None
    destination_directory_fd: int | None = None
    temporary_fd: int | None = None
    source_path: Path | None = None
    temporary_name = f".{destination.name}.copy-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    publication_name = f".{destination.name}.publish-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        if source_fd is None:
            source_directory_fd = _open_owned_testmon_directory(source.parent.parent.parent, create=False)
        else:
            # A bound SQLite filename family retains the source generation and
            # its paired WAL without reopening the mutable public basename.
            source_path = bound_source_path or _descriptor_bound_path(source_fd)
        destination_directory_fd = _open_owned_testmon_directory(destination.parent.parent.parent, create=True)
        if source_fd is None:
            # Retain the source inode before SQLite opens it.  The returned
            # path is bound to the private hard-link descriptor, so replacing
            # the public source name cannot redirect the copy; replacing the
            # .bound-* entry makes the descriptor namespace fail closed.
            assert source_directory_fd is not None
            source_child_fd, source_path, source_private_name = _open_owned_testmon_child(
                source_directory_fd, source.name
            )
        assert source_path is not None
        if source_fd is not None and bound_source_path is not None:
            retained_source = os.fstat(source_fd)
            current_source = source_path.stat()
            if (current_source.st_dev, current_source.st_ino) != (retained_source.st_dev, retained_source.st_ino):
                raise NativeTestmonRepairError("bound source testmon database changed before SQLite backup")
        destination_name = destination.name
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_directory_fd,
        )
        # O_EXCL binds this descriptor to the new inode.  Opening the child
        # name again would let a replacement of that entry redirect SQLite.
        temporary_sqlite_path = _descriptor_bound_path(temporary_fd)
        with (
            contextlib.closing(
                sqlite3.connect(
                    _readonly_uri(source_path),
                    uri=True,
                    timeout=_remaining_timeout(deadline_monotonic, 60),
                )
            ) as source_connection,
            contextlib.closing(
                sqlite3.connect(temporary_sqlite_path, timeout=_remaining_timeout(deadline_monotonic, 60))
            ) as destination_connection,
            source_connection,
            destination_connection,
        ):
            source_connection.backup(
                destination_connection,
                pages=256,
                progress=lambda _status, _remaining, _total: _ensure_deadline(deadline_monotonic),
                sleep=0.05,
            )
        if source_fd is not None and bound_source_path is not None:
            retained_source = os.fstat(source_fd)
            current_source = source_path.stat()
            if (current_source.st_dev, current_source.st_ino) != (retained_source.st_dev, retained_source.st_ino):
                raise NativeTestmonRepairError("bound source testmon database changed during SQLite backup")
        _ensure_deadline(deadline_monotonic)
        # Validate through the held descriptor.  Reopening ``temporary`` by
        # name here would allow a replacement of that directory entry to
        # substitute a different database between the copy and validation.
        copied = inspect_native_testmon_environment(
            temporary_sqlite_path,
            environment_name=environment_name,
            required_executable_paths=required_executable_paths,
            data_fd=temporary_fd,
            deadline_monotonic=deadline_monotonic,
        )
        if not copied.valid:
            raise NativeTestmonRepairError(f"copied main-checkout database failed validation: {copied.reason}")
        os.fsync(temporary_fd)
        # Remove the DESTINATION's sidecars before the replace: a stale -wal
        # from a previous database under the same name is read as this new
        # database's write-ahead log, and quick_check then fails with page
        # corruption ("Rowid out of order" -- observed 2026-08-18 in a lane
        # worktree whose fresh copy sat beside a -wal from an earlier run).
        for suffix in TESTMON_SIDECAR_SUFFIXES:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(f"{destination_name}{suffix}", dir_fd=destination_directory_fd)
        _publish_validated_testmon_database(
            destination_directory_fd=destination_directory_fd,
            destination_name=destination_name,
            temporary_fd=temporary_fd,
            publication_name=publication_name,
            deadline_monotonic=deadline_monotonic,
        )
        os.fsync(destination_directory_fd)
        _ensure_deadline(deadline_monotonic)
    except NativeTestmonDeadlineError:
        raise
    except (OSError, sqlite3.Error, NativeTestmonRepairError) as exc:
        raise NativeTestmonRepairError(f"SQLite online backup failed: {exc}") from exc
    finally:
        if destination_directory_fd is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=destination_directory_fd)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(publication_name, dir_fd=destination_directory_fd)
            for suffix in TESTMON_SIDECAR_SUFFIXES:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(f"{temporary_name}{suffix}", dir_fd=destination_directory_fd)
        if temporary_fd is not None:
            os.close(temporary_fd)
        if destination_directory_fd is not None:
            os.close(destination_directory_fd)
        if source_directory_fd is not None:
            if source_private_name is not None:
                _unlink_bound_entry_if_owned(source_directory_fd, source_private_name, source_child_fd)
            os.close(source_directory_fd)
        if source_child_fd is not None:
            os.close(source_child_fd)


def linked_worktree_info(
    repo_root: Path,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[bool, Path] | None:
    """Return linked-worktree status and the main checkout path."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--absolute-git-dir", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=_remaining_timeout(deadline_monotonic, 10),
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired:
        _ensure_deadline(deadline_monotonic)
        return None
    except OSError:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return None
    git_dir = Path(lines[0]).resolve()
    raw_common = Path(lines[1])
    common_dir = raw_common.resolve() if raw_common.is_absolute() else (repo_root / raw_common).resolve()
    return git_dir != common_dir, common_dir.parent


def prepare_native_testmon_environment(
    repo_root: Path,
    *,
    required_executable_paths: Sequence[str] = (),
    pytest_profile: str = "default",
    pytest_environment: Mapping[str, str | None] | None = None,
    deadline_monotonic: float | None = None,
    source_lock_factory: Callable[[Path], contextlib.AbstractContextManager[bool]] | None = None,
) -> NativeTestmonPreparation:
    """Repair derived local state and optionally reuse a matching main graph."""
    root = repo_root.resolve()
    _validate_owned_state_parents(root)
    environment_name = testmon_environment_digest(
        root,
        pytest_profile=pytest_profile,
        pytest_environment=pytest_environment,
        deadline_monotonic=deadline_monotonic,
    )
    local_data = root / TESTMON_DATA_RELPATH
    local_data.parent.mkdir(parents=True, exist_ok=True)
    _validate_owned_state_parents(root)
    local = inspect_native_testmon_environment(
        local_data,
        environment_name=environment_name,
        required_executable_paths=required_executable_paths,
        deadline_monotonic=deadline_monotonic,
    )
    missing_checkout_paths = tuple(
        path for path in sorted(set(required_executable_paths)) if not (root / path).is_file()
    )
    if local.valid and missing_checkout_paths:
        local = NativeTestmonState(
            "invalid",
            "changed executable modules are absent from the current checkout",
            local.environment,
            missing_checkout_paths,
        )
    fallback_allowed = native_testmon_fallback_allowed(local)
    info = linked_worktree_info(root, deadline_monotonic=deadline_monotonic)
    linked = bool(info and info[0])
    main_checkout = info[1] if linked and info is not None else None
    if local.valid:
        violation = certified_attestation_violation(
            root,
            environment_name=environment_name,
            current_nodeids=local.environment.nodeids if local.environment is not None else (),
        )
        if violation is not None:
            # A coverage certificate governs a stronger release-attestation
            # claim, not testmon's sound dependency selection. Keep selecting
            # from the compatible graph; only an explicit --all run renews a
            # stale certificate.
            local = NativeTestmonState(
                local.status,
                f"attestation unavailable: {violation}",
                local.environment,
                local.missing_executable_paths,
            )
        return NativeTestmonPreparation(
            environment_name,
            "affected",
            local,
            None,
            (),
            linked,
            main_checkout,
            fallback_allowed,
        )

    # Retain a merely incomplete graph. An interrupted bootstrap leaves a
    # sound database that simply has not fingerprinted every changed module
    # yet; removing it discards every test the interrupted run recorded, so
    # the next invocation bootstraps from zero and is interrupted at the same
    # point -- a loop no number of retries escapes. Only genuinely unusable
    # state (corrupt, schema-incompatible, ambiguous environment, or replaced
    # on disk) is removed.
    # Only genuinely damaged state is removed. A database that is merely
    # missing this environment, or holds an interrupted bootstrap's partial
    # graph, is left alone: deleting it discards work that belongs to another
    # environment or to the run that was interrupted.
    removed: tuple[Path, ...] = ()
    if local.status == "invalid":
        removed = remove_invalid_native_testmon_state(root)
    _ensure_deadline(deadline_monotonic)
    copied_from: Path | None = None
    if main_checkout is not None and main_checkout != root and not missing_checkout_paths:
        source_lock = (
            source_lock_factory(main_checkout) if source_lock_factory is not None else contextlib.nullcontext(True)
        )
        main = NativeTestmonState("absent", "optional main native testmon state unavailable")
        with _optional_native_testmon_source_lock(source_lock) as source_available:
            if source_available:
                try:
                    _validate_owned_state_parents(main_checkout)
                except NativeTestmonRepairError as exc:
                    main = NativeTestmonState("absent", f"optional main native testmon state unavailable: {exc}")
                else:
                    main_data = main_checkout / TESTMON_DATA_RELPATH
                    main_binding_context = native_testmon_source_binding(main_data)
                    try:
                        main_binding = main_binding_context.__enter__()
                    except NativeTestmonRepairError as exc:
                        # The linked checkout is optional. A stale private hard link or
                        # another source-only ownership violation must not turn an
                        # otherwise safe local bootstrap into a rejected verify.
                        main_binding = None
                        main = NativeTestmonState("absent", f"optional main native testmon state unavailable: {exc}")
                    else:
                        try:
                            if main_binding is None:
                                # The public source path may appear immediately after
                                # binding checked it. Without a retained descriptor it
                                # is not safe to inspect or copy that mutable inode.
                                main = NativeTestmonState(
                                    "absent",
                                    "optional main native testmon state unavailable before descriptor binding",
                                )
                            else:
                                main = inspect_native_testmon_environment(
                                    main_data,
                                    environment_name=environment_name,
                                    required_executable_paths=required_executable_paths,
                                    data_fd=main_binding.descriptor,
                                    bound_data_path=main_binding.data_path,
                                    bound_sidecar_fds=dict(main_binding.sidecar_descriptors),
                                    deadline_monotonic=deadline_monotonic,
                                )
                                if main.valid:
                                    _atomic_copy_sqlite_database(
                                        main_data,
                                        local_data,
                                        environment_name=environment_name,
                                        required_executable_paths=required_executable_paths,
                                        deadline_monotonic=deadline_monotonic,
                                        source_fd=main_binding.descriptor,
                                        bound_source_path=main_binding.data_path,
                                    )
                                    copied_from = main_data
                        finally:
                            main_binding_context.__exit__(None, None, None)
        fallback_allowed = native_testmon_fallback_allowed(local, main)

        if copied_from is not None:
            local = inspect_native_testmon_environment(
                local_data,
                environment_name=environment_name,
                required_executable_paths=required_executable_paths,
                deadline_monotonic=deadline_monotonic,
            )
            if not local.valid:
                raise NativeTestmonRepairError(f"published native testmon copy is invalid: {local.reason}")
            violation = certified_attestation_violation(
                root,
                environment_name=environment_name,
                current_nodeids=local.environment.nodeids if local.environment is not None else (),
            )
            if violation is not None:
                local = NativeTestmonState(
                    local.status,
                    f"attestation unavailable: {violation}",
                    local.environment,
                    local.missing_executable_paths,
                )
            return NativeTestmonPreparation(
                environment_name,
                "affected",
                local,
                copied_from,
                removed,
                linked,
                main_checkout,
                fallback_allowed,
            )

    if local.resumable:
        # OPERATOR DECISION 2026-08-18: prefer the hazard to the standstill.
        # A resumable graph is structurally sound and merely lacks edges for some
        # changed modules. The previous rule discarded it and ran the complete
        # corpus, which is ~9.5x a warm run; measured against the recorded run
        # history, 5.1 of 5.65 hours of testmon-tier time went to runs that
        # selected nothing and ran everything. The residual risk is precise and
        # bounded: tests whose only dependency is an un-fingerprinted module may
        # not be selected on THIS run. They are selected on the next one, because
        # the run still records edges for everything it executes. The uncovered
        # paths are named in the receipt rather than paid for every time.
        return NativeTestmonPreparation(
            environment_name, "affected", local, copied_from, removed, linked, main_checkout, fallback_allowed
        )
    return NativeTestmonPreparation(
        environment_name,
        "bootstrap",
        local,
        copied_from,
        removed,
        linked,
        main_checkout,
        fallback_allowed,
    )


__all__ = [
    "ASTClassification",
    "NativeTestmonEnvironment",
    "NativeTestmonDeadlineError",
    "NativeTestmonChangeImpact",
    "NativeTestmonLifecycleLockTimeoutError",
    "NativeTestmonPreparation",
    "NativeTestmonRepairError",
    "NativeTestmonSourceBinding",
    "NativeTestmonState",
    "TESTMON_DATA_RELPATH",
    "classify_source_ast",
    "classify_native_testmon_changes",
    "executable_python_paths",
    "inspect_native_testmon_environment",
    "linked_worktree_info",
    "native_testmon_fallback_allowed",
    "native_testmon_lifecycle_lock",
    "native_testmon_lifecycle_locks",
    "native_testmon_source_binding",
    "prepare_native_testmon_environment",
    "remove_invalid_native_testmon_state",
    "testmon_environment_digest",
    "validate_native_testmon_state_ownership",
]
