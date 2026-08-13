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
An absent main database is normal.  The next plain verify invocation builds
the current environment by running the ordinary correctness corpus.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import stat
import subprocess
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

TESTMON_DATA_RELPATH = Path(".cache/testmon/testmondata")
TESTMON_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

NativeStateStatus = Literal["absent", "valid", "invalid"]
NativeSelectionMode = Literal["bootstrap", "affected"]
ASTClassification = Literal["declaration-only", "executable", "source-unreadable"]

_ENVIRONMENT_INPUTS = (
    "uv.lock",
    "pyproject.toml",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    "devtools/checkout_guard.py",
    "devtools/testmon_bootstrap.py",
    "devtools/verify.py",
    "devtools/verify_runs.py",
)
_PYTEST_ENVIRONMENT_KEYS = (
    "HYPOTHESIS_PROFILE",
    "POLYLOGUE_CI",
    "PYTEST_ADDOPTS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "PYTEST_PLUGINS",
)


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


@dataclass(frozen=True, slots=True)
class NativeTestmonPreparation:
    environment_name: str
    selection_mode: NativeSelectionMode
    local_state: NativeTestmonState
    copied_from: Path | None
    removed_paths: tuple[Path, ...]
    linked_worktree: bool
    main_checkout: Path | None


class NativeTestmonRepairError(RuntimeError):
    """The exact derived testmon state could not be repaired safely."""


def _fingerprint_inputs(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in relative_paths:
        digest.update(relative.encode())
        digest.update(b"\0")
        try:
            contents = (root / relative).read_bytes()
        except OSError:
            digest.update(b"missing")
        else:
            digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def _environment_input_paths(root: Path) -> tuple[str, ...]:
    """Discover collection and managed-pytest harness inputs."""
    paths = set(_ENVIRONMENT_INPUTS)
    patterns = (
        "devtools/pytest*.py",
        "tests/**/conftest.py",
        "tests/infra/**/*.py",
    )
    for pattern in patterns:
        paths.update(path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file())
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


def testmon_environment_digest(repo_root: Path, *, pytest_profile: str = "default") -> str:
    """Return the native testmon environment name for collection semantics."""
    root = repo_root.resolve()
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
        "inputs": _fingerprint_inputs(root, _environment_input_paths(root)),
        "pytest_environment": {key: os.environ.get(key) for key in _PYTEST_ENVIRONMENT_KEYS},
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


def _body_is_executable(body: list[ast.stmt]) -> bool:
    for index, node in enumerate(body):
        if _is_docstring(node, first=index == 0):
            continue
        if isinstance(node, (ast.Pass, ast.Import, ast.ImportFrom)):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and (isinstance(node.value.value, str) or node.value.value is Ellipsis)
        ):
            continue
        if isinstance(node, ast.AnnAssign) and node.value is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list or node.args.defaults or any(value is not None for value in node.args.kw_defaults):
                return True
            if _body_is_executable(node.body):
                return True
            continue
        if isinstance(node, ast.ClassDef):
            if node.decorator_list or node.bases or node.keywords or _body_is_executable(node.body):
                return True
            continue
        if isinstance(node, ast.Assign):
            return True
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                return True
            continue
        return True
    return False


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
    """Return changed Python paths whose runtime behavior needs graph edges."""
    root = repo_root.resolve()
    executable: list[str] = []
    for raw in sorted(set(paths)):
        relative = _safe_relative_path(raw)
        if relative is None or not relative.endswith(".py"):
            continue
        source = root / relative
        if not source.exists():
            continue
        if not source.is_file() or classify_source_ast(source) != "declaration-only":
            executable.append(relative)
    return tuple(executable)


def _readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


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
) -> NativeTestmonState:
    """Validate one native environment without interpreting plugin internals."""
    sidecars = tuple(Path(f"{data_path}{suffix}") for suffix in TESTMON_SIDECAR_SUFFIXES)
    if not data_path.exists():
        if any(path.exists() or path.is_symlink() for path in sidecars):
            return NativeTestmonState("invalid", "SQLite sidecars exist without the owned database")
        return NativeTestmonState("absent", "native testmon database is absent")
    try:
        mode = data_path.lstat().st_mode
    except OSError as exc:
        return NativeTestmonState("invalid", f"cannot inspect native testmon database: {exc}")
    if not stat.S_ISREG(mode):
        return NativeTestmonState("invalid", "native testmon database is not a regular file")
    try:
        with sqlite3.connect(_readonly_uri(data_path), uri=True, timeout=10) as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                return NativeTestmonState("invalid", "SQLite quick_check failed")
            version_row = connection.execute("PRAGMA user_version").fetchone()
            if version_row is None or version_row[0] != _testmon_schema_version():
                return NativeTestmonState("invalid", "pytest-testmon database schema version changed")
            environment_rows = connection.execute(
                "SELECT id FROM environment WHERE environment_name = ? ORDER BY id DESC",
                (environment_name,),
            ).fetchall()
            if len(environment_rows) != 1:
                reason = "native environment is absent" if not environment_rows else "native environment is ambiguous"
                return NativeTestmonState("invalid", reason)
            environment_id = int(environment_rows[0][0])
            nodeids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT test_name FROM test_execution WHERE environment_id = ? ORDER BY test_name",
                    (environment_id,),
                ).fetchall()
                if isinstance(row[0], str) and row[0]
            )
            if not nodeids or len(nodeids) != len(set(nodeids)):
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
    except (NativeTestmonRepairError, OSError, sqlite3.Error, TypeError, ValueError) as exc:
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
            "invalid",
            "changed executable modules are absent from the native dependency graph",
            environment,
            missing,
        )
    return NativeTestmonState("valid", "native environment is current", environment)


def _owned_paths(repo_root: Path) -> tuple[Path, ...]:
    data = repo_root.resolve() / TESTMON_DATA_RELPATH
    return (data, *(Path(f"{data}{suffix}") for suffix in TESTMON_SIDECAR_SUFFIXES))


def remove_invalid_native_testmon_state(repo_root: Path) -> tuple[Path, ...]:
    """Remove only the exact checkout-owned SQLite file and known sidecars."""
    removed: list[Path] = []
    for path in _owned_paths(repo_root):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NativeTestmonRepairError(f"cannot inspect owned testmon path {path}: {exc}") from exc
        if stat.S_ISDIR(mode):
            raise NativeTestmonRepairError(f"refusing to remove directory at owned SQLite path {path}")
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


def _atomic_copy_sqlite_database(
    source: Path,
    destination: Path,
    *,
    environment_name: str,
    required_executable_paths: Sequence[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copy-{os.getpid()}-{uuid.uuid4().hex}.tmp")
    try:
        with (
            sqlite3.connect(_readonly_uri(source), uri=True, timeout=60) as source_connection,
            sqlite3.connect(temporary, timeout=60) as destination_connection,
        ):
            source_connection.backup(destination_connection)
        copied = inspect_native_testmon_environment(
            temporary,
            environment_name=environment_name,
            required_executable_paths=required_executable_paths,
        )
        if not copied.valid:
            raise NativeTestmonRepairError(f"copied main-checkout database failed validation: {copied.reason}")
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except (OSError, sqlite3.Error) as exc:
        raise NativeTestmonRepairError(f"SQLite online backup failed: {exc}") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        for suffix in TESTMON_SIDECAR_SUFFIXES:
            with contextlib.suppress(FileNotFoundError):
                Path(f"{temporary}{suffix}").unlink()


def linked_worktree_info(repo_root: Path) -> tuple[bool, Path] | None:
    """Return linked-worktree status and the main checkout path."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--absolute-git-dir", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
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
) -> NativeTestmonPreparation:
    """Repair derived local state and optionally reuse a matching main graph."""
    root = repo_root.resolve()
    environment_name = testmon_environment_digest(root, pytest_profile=pytest_profile)
    local_data = root / TESTMON_DATA_RELPATH
    local = inspect_native_testmon_environment(
        local_data,
        environment_name=environment_name,
        required_executable_paths=required_executable_paths,
    )
    info = linked_worktree_info(root)
    linked = bool(info and info[0])
    main_checkout = info[1] if linked and info is not None else None
    if local.valid:
        return NativeTestmonPreparation(environment_name, "affected", local, None, (), linked, main_checkout)

    removed = remove_invalid_native_testmon_state(root)
    copied_from: Path | None = None
    if main_checkout is not None and main_checkout != root:
        main_data = main_checkout / TESTMON_DATA_RELPATH
        main = inspect_native_testmon_environment(
            main_data,
            environment_name=environment_name,
            required_executable_paths=required_executable_paths,
        )
        if main.valid:
            _atomic_copy_sqlite_database(
                main_data,
                local_data,
                environment_name=environment_name,
                required_executable_paths=required_executable_paths,
            )
            copied_from = main_data
            local = inspect_native_testmon_environment(
                local_data,
                environment_name=environment_name,
                required_executable_paths=required_executable_paths,
            )
            if not local.valid:
                raise NativeTestmonRepairError(f"published native testmon copy is invalid: {local.reason}")
            return NativeTestmonPreparation(
                environment_name,
                "affected",
                local,
                copied_from,
                removed,
                linked,
                main_checkout,
            )

    return NativeTestmonPreparation(environment_name, "bootstrap", local, copied_from, removed, linked, main_checkout)


__all__ = [
    "ASTClassification",
    "NativeTestmonEnvironment",
    "NativeTestmonPreparation",
    "NativeTestmonRepairError",
    "NativeTestmonState",
    "TESTMON_DATA_RELPATH",
    "classify_source_ast",
    "executable_python_paths",
    "inspect_native_testmon_environment",
    "linked_worktree_info",
    "prepare_native_testmon_environment",
    "remove_invalid_native_testmon_state",
    "testmon_environment_digest",
]
