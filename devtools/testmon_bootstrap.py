"""Prepare checkout-local pytest-testmon state for plain ``devtools verify``.

Pytest-testmon already owns interrupted-run recovery, failing/new test
selection, node deletion, and dependency replacement.  This module therefore
does only the checkout boundary work that the plugin cannot do itself:

* derive the native ``--testmon-env`` name from collection semantics;
* validate that the local SQLite database contains that environment;
* require changed executable modules to occur in its dependency graph;
* remove only an invalid checkout-owned database and its SQLite sidecars;
* report the current checkout's graph state without borrowing mutable state
  from a sibling worktree.

The graph stays local to this checkout. A linked worktree without a graph
takes a snapshot of the main checkout's graph when that graph is valid for
the same environment; otherwise an absent graph is reported to the caller. Only an explicitly requested complete-corpus run may build a new
environment; plain verification reuses a compatible graph or refuses before
pytest starts.
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
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

TESTMON_DATA_RELPATH = Path(".cache/testmon/testmondata")
TESTMON_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

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


@dataclass(frozen=True, slots=True)
class NativeTestmonPreparation:
    environment_name: str
    selection_mode: NativeSelectionMode
    local_state: NativeTestmonState
    removed_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class NativeTestmonChangeImpact:
    """Changed inputs that native Python tracing can and cannot select."""

    executable_paths: tuple[str, ...]
    runtime_data_paths: tuple[str, ...]


class NativeTestmonRepairError(RuntimeError):
    """The exact derived testmon state could not be repaired safely."""


class NativeTestmonDeadlineError(NativeTestmonRepairError):
    """The verify invocation deadline expired during native-state preparation."""


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
    deadline_monotonic: float | None = None,
) -> str:
    """Return the native testmon environment name for collection semantics.

    Only what changes collection is hashed. Process environment such as the
    hypothesis profile changes example budgets, not the collected corpus, and
    hashing it gave every shell and the daemon a different graph.
    """
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
    deadline_monotonic: float | None = None,
) -> NativeTestmonState:
    """Validate one native environment without interpreting plugin internals."""
    _ensure_deadline(deadline_monotonic)
    sidecars = tuple(Path(f"{data_path}{suffix}") for suffix in TESTMON_SIDECAR_SUFFIXES)
    if not data_path.exists():
        if any(path.exists() or path.is_symlink() for path in sidecars):
            return NativeTestmonState("invalid", "SQLite sidecars exist without the owned database")
        return NativeTestmonState("absent", "native testmon database is absent")
    try:
        state = data_path.lstat()
    except OSError as exc:
        return NativeTestmonState("invalid", f"cannot inspect native testmon database: {exc}")
    if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
        return NativeTestmonState("invalid", "native testmon database is not a single-link regular file")
    for _suffix, sidecar in zip(TESTMON_SIDECAR_SUFFIXES, sidecars, strict=True):
        try:
            sidecar_state = sidecar.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            return NativeTestmonState("invalid", f"cannot inspect native testmon sidecar {sidecar}: {exc}")
        if not stat.S_ISREG(sidecar_state.st_mode) or sidecar_state.st_nlink != 1:
            return NativeTestmonState("invalid", f"native testmon sidecar is not a single-link regular file: {sidecar}")
    # A killed writer can leave a WAL that needs ordinary SQLite recovery before
    # the read-only validation connection can inspect the graph.
    if any(path.exists() for path in sidecars):
        try:
            recovery = sqlite3.connect(data_path, timeout=_remaining_timeout(deadline_monotonic, 10))
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
                    _readonly_uri(data_path),
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
                    # every environment name, so reporting "invalid" here
                    # invites the caller to delete another environment's graph
                    # on a routine miss.
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
    lanes record ``path::test[param]``; graph rows and coverage
    math must compare on ONE form or every cross-shape comparison invents
    phantom missing tests (509 of them, observed 2026-08-18). Only a
    trailing ``@group`` outside any param bracket is stripped, so params that
    themselves contain ``@`` survive.
    """
    base, sep, suffix = nodeid.rpartition("@")
    if sep and suffix and "]" not in suffix and ":" not in suffix and "/" not in suffix:
        return base
    return nodeid


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


def main_checkout_root(repo_root: Path) -> Path | None:
    """Return the main checkout of a linked worktree, or None for the main checkout itself."""
    root = repo_root.resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir", "--git-common-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        return None
    git_dir, common = ((root / line.strip()).resolve() for line in lines)
    if git_dir == common:
        return None
    main = common.parent
    if main == root or not (main / ".git").exists():
        return None
    return main


def seed_native_testmon_from_main_checkout(
    repo_root: Path,
    *,
    environment_name: str,
    required_executable_paths: Sequence[str] = (),
    deadline_monotonic: float | None = None,
) -> NativeTestmonState | None:
    """Copy the main checkout's graph into a linked worktree lacking this environment.

    Provisioning copies the graph once, when the worktree is created; a graph
    the nightly corpus run publishes later never reaches an older worktree.
    The copy is a consistent SQLite snapshot, so the worktree still owns its
    own state afterwards. Returns None for the main checkout itself, and an
    absent state naming the main graph's condition when there is nothing
    usable to copy. The caller decides that the local graph lacks it.
    """
    root = repo_root.resolve()
    local_data = root / TESTMON_DATA_RELPATH
    main = main_checkout_root(root)
    if main is None:
        return None
    main_data = main / TESTMON_DATA_RELPATH
    main_state = inspect_native_testmon_environment(
        main_data,
        environment_name=environment_name,
        required_executable_paths=required_executable_paths,
        deadline_monotonic=deadline_monotonic,
    )
    if not (main_state.valid or main_state.resumable):
        # A main graph that merely lacks edges for modules this lane added
        # is still the corpus every other test depends on; copied, it is
        # resumable here exactly as it would be there. Anything else is
        # reported so the refusal names the main graph, not just this one.
        return NativeTestmonState(
            "absent",
            f"native testmon database is absent; main checkout {main} graph: {main_state.reason}",
        )
    _ensure_deadline(deadline_monotonic)
    _validate_owned_state_parents(root)
    local_data.parent.mkdir(parents=True, exist_ok=True)
    # A local database that lacks this environment (another digest's graph,
    # or the empty row an interrupted bootstrap leaves) is replaced whole:
    # the snapshot carries every environment the main checkout holds. Its
    # sidecars go first, so a stale WAL is never replayed onto the snapshot.
    validate_native_testmon_state_ownership(root)
    for sidecar in (Path(f"{local_data}{suffix}") for suffix in TESTMON_SIDECAR_SUFFIXES):
        with contextlib.suppress(FileNotFoundError):
            sidecar.unlink()
    staging = local_data.with_name(f"{local_data.name}.seed-{uuid.uuid4().hex}.tmp")
    created = False
    try:
        descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
        created = True
        state = staging.lstat()
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise NativeTestmonRepairError(f"staging path is not a fresh regular file: {staging}")
        source = sqlite3.connect(_readonly_uri(main_data), uri=True, timeout=_remaining_timeout(deadline_monotonic, 10))
        try:
            target = sqlite3.connect(staging)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        os.replace(staging, local_data)
    except (sqlite3.Error, OSError) as exc:
        if created:
            with contextlib.suppress(OSError):
                staging.unlink()
        raise NativeTestmonRepairError(f"cannot seed native testmon graph from {main_data}: {exc}") from exc
    return inspect_native_testmon_environment(
        local_data,
        environment_name=environment_name,
        required_executable_paths=required_executable_paths,
        deadline_monotonic=deadline_monotonic,
    )


def prepare_native_testmon_environment(
    repo_root: Path,
    *,
    required_executable_paths: Sequence[str] = (),
    deadline_monotonic: float | None = None,
) -> NativeTestmonPreparation:
    """Inspect and repair derived state owned by this checkout only."""
    root = repo_root.resolve()
    _validate_owned_state_parents(root)
    environment_name = testmon_environment_digest(root, deadline_monotonic=deadline_monotonic)
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
    if local.valid:
        return NativeTestmonPreparation(environment_name, "affected", local, ())

    if local.status == "absent":
        seeded = seed_native_testmon_from_main_checkout(
            root,
            environment_name=environment_name,
            required_executable_paths=required_executable_paths,
            deadline_monotonic=deadline_monotonic,
        )
        if seeded is not None:
            local = seeded
            if local.valid and not missing_checkout_paths:
                return NativeTestmonPreparation(environment_name, "affected", local, ())

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
        invalid = local
        removed = remove_invalid_native_testmon_state(root)
        # What remains after the repair is an empty slot, and an empty slot
        # seeds: orphan sidecars must not cost a lane its selection. The
        # original diagnosis survives unless a replacement graph arrives.
        repaired = inspect_native_testmon_environment(
            local_data,
            environment_name=environment_name,
            required_executable_paths=required_executable_paths,
            deadline_monotonic=deadline_monotonic,
        )
        seeded = (
            seed_native_testmon_from_main_checkout(
                root,
                environment_name=environment_name,
                required_executable_paths=required_executable_paths,
                deadline_monotonic=deadline_monotonic,
            )
            if repaired.status == "absent"
            else None
        )
        if seeded is not None and seeded.environment is not None:
            local = seeded
        else:
            local = NativeTestmonState(
                "invalid",
                invalid.reason + ("; " + seeded.reason if seeded is not None else "; removed"),
            )
        if local.valid and not missing_checkout_paths:
            return NativeTestmonPreparation(environment_name, "affected", local, removed)
    if missing_checkout_paths and local.environment is not None:
        # A seeded or resumable graph is still no graph for a module that
        # left the checkout underneath preparation.
        return NativeTestmonPreparation(
            environment_name,
            "bootstrap",
            NativeTestmonState(
                "invalid",
                "changed executable modules are absent from the current checkout",
                local.environment,
                missing_checkout_paths,
            ),
            removed,
        )
    _ensure_deadline(deadline_monotonic)

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
        return NativeTestmonPreparation(environment_name, "affected", local, removed)
    return NativeTestmonPreparation(environment_name, "bootstrap", local, removed)


__all__ = [
    "ASTClassification",
    "NativeTestmonEnvironment",
    "NativeTestmonDeadlineError",
    "NativeTestmonChangeImpact",
    "NativeTestmonPreparation",
    "NativeTestmonRepairError",
    "NativeTestmonState",
    "TESTMON_DATA_RELPATH",
    "classify_source_ast",
    "classify_native_testmon_changes",
    "executable_python_paths",
    "inspect_native_testmon_environment",
    "main_checkout_root",
    "prepare_native_testmon_environment",
    "seed_native_testmon_from_main_checkout",
    "remove_invalid_native_testmon_state",
    "testmon_environment_digest",
    "validate_native_testmon_state_ownership",
]
