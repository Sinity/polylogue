from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from devtools import testmon_bootstrap
from devtools.testmon_bootstrap import (
    TESTMON_DATA_RELPATH,
    NativeTestmonDeadlineError,
    NativeTestmonRepairError,
    _atomic_copy_sqlite_database,
    _testmon_schema_version,
    canonical_test_nodeid,
    classify_native_testmon_changes,
    classify_source_ast,
    executable_python_paths,
    inspect_native_testmon_environment,
    native_testmon_source_binding,
    prepare_native_testmon_environment,
    remove_invalid_native_testmon_state,
    validate_native_testmon_state_ownership,
)
from devtools.testmon_bootstrap import (
    testmon_environment_digest as _testmon_environment_digest,
)


def test_ast_classification_distinguishes_declarations_from_execution(tmp_path: Path) -> None:
    declarations = tmp_path / "types.py"
    declarations.write_text(
        '"""Types only."""\nname: str\n\nclass Record:\n    identifier: int\n\n    def label(self) -> str: ...\n',
        encoding="utf-8",
    )
    executable = tmp_path / "runtime.py"
    executable.write_text("VALUE: int = 3\n", encoding="utf-8")

    assert classify_source_ast(declarations) == "declaration-only"
    assert classify_source_ast(executable) == "executable"


def test_ast_classification_treats_type_checking_guards_as_declarations(tmp_path: Path) -> None:
    declarations = tmp_path / "protocols.py"
    declarations.write_text(
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from polylogue.archive.models import Session\n\n"
        "class SessionReader:\n"
        "    session: 'Session'\n"
        "    def read(self) -> 'Session': ...\n",
        encoding="utf-8",
    )

    assert classify_source_ast(declarations) == "declaration-only"

    declarations.write_text(declarations.read_text(encoding="utf-8") + "\nVALUE = build_runtime_value()\n")

    assert classify_source_ast(declarations) == "executable"


def test_ast_classification_treats_ordinary_imports_as_executable(tmp_path: Path) -> None:
    """Removing ordinary-import execution from the classifier makes this fail."""
    module = tmp_path / "runtime.py"
    module.write_text("from package.runtime import value\n", encoding="utf-8")

    assert classify_source_ast(module) == "executable"


def test_pure_enum_contracts_are_non_traceable_runtime_inputs(tmp_path: Path) -> None:
    module = tmp_path / "devtools" / "verification_contracts.py"
    module.parent.mkdir()
    module.write_text(
        "from enum import StrEnum\n\n"
        "class VerificationScope(StrEnum):\n"
        "    AFFECTED = 'affected'\n"
        "    RELEASE_BASELINE = 'release-baseline'\n",
        encoding="utf-8",
    )

    assert classify_source_ast(module) == "declaration-only"
    impact = classify_native_testmon_changes(tmp_path, ("devtools/verification_contracts.py",))

    assert impact.executable_paths == ()
    assert impact.runtime_data_paths == ("devtools/verification_contracts.py",)


def test_runtime_protocol_contracts_are_non_traceable_runtime_inputs(tmp_path: Path) -> None:
    module = tmp_path / "polylogue" / "core" / "protocols.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING, Protocol, runtime_checkable\n\n"
        "if TYPE_CHECKING:\n"
        "    from polylogue.storage.models import Record\n\n"
        "@runtime_checkable\n"
        "class RecordStore(Protocol):\n"
        "    async def get(self, record_id: str) -> Record | None: ...\n\n"
        "__all__ = ('RecordStore',)\n",
        encoding="utf-8",
    )

    assert classify_source_ast(module) == "declaration-only"
    impact = classify_native_testmon_changes(tmp_path, ("polylogue/core/protocols.py",))

    assert impact.executable_paths == ()
    assert impact.runtime_data_paths == ("polylogue/core/protocols.py",)


def test_executable_paths_require_current_runtime_modules_not_deleted_ones(tmp_path: Path) -> None:
    module = tmp_path / "polylogue" / "runtime.py"
    module.parent.mkdir()
    module.write_text("VALUE = factory()\n", encoding="utf-8")
    malformed = module.with_name("malformed.py")
    malformed.write_text("def broken(:\n", encoding="utf-8")

    # A deleted module is excluded: no rebuild can ever record an edge for a
    # file that does not exist, so requiring one made ``incomplete``
    # permanent. pytest-testmon detects the deletion itself through the
    # dependents' stale recorded fingerprints.
    assert executable_python_paths(
        tmp_path,
        ("polylogue/runtime.py", "polylogue/malformed.py", "polylogue/deleted.py"),
    ) == (
        "polylogue/malformed.py",
        "polylogue/runtime.py",
    )


def test_package_runtime_data_changes_force_full_native_selection(tmp_path: Path) -> None:
    runtime = tmp_path / "polylogue" / "archive" / "semantic" / "data" / "prices.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("{}\n", encoding="utf-8")
    declaration = runtime.with_name("types.pyi")
    declaration.write_text("VALUE: int\n", encoding="utf-8")

    impact = classify_native_testmon_changes(
        tmp_path,
        (
            "polylogue/archive/semantic/data/prices.json",
            "polylogue/archive/semantic/data/deleted.json",
            "polylogue/archive/semantic/data/types.pyi",
            "docs/prices.json",
        ),
    )

    assert impact.executable_paths == ()
    assert impact.runtime_data_paths == (
        "polylogue/archive/semantic/data/deleted.json",
        "polylogue/archive/semantic/data/prices.json",
        "polylogue/archive/semantic/data/types.pyi",
    )


def test_test_runtime_data_changes_force_full_native_selection(tmp_path: Path) -> None:
    runtime = tmp_path / "tests" / "data" / "payload.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("{}\n", encoding="utf-8")

    impact = classify_native_testmon_changes(
        tmp_path,
        (
            "tests/data/payload.json",
            "tests/data/deleted.json",
            "docs/payload.json",
        ),
    )

    assert impact.executable_paths == ()
    assert impact.runtime_data_paths == (
        "tests/data/deleted.json",
        "tests/data/payload.json",
    )


def test_benchmarks_are_not_required_graph_paths_and_packaging_is_untraceable(tmp_path: Path) -> None:
    """Restoring benchmark graph edges or dropping packaging inputs makes this fail."""
    benchmark = tmp_path / "tests" / "benchmarks" / "test_scale.py"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("def test_scale(): pass\n", encoding="utf-8")
    packaging_python = tmp_path / "packaging" / "hatch_build.py"
    packaging_python.parent.mkdir()
    packaging_python.write_text("def build_hook(): pass\n", encoding="utf-8")

    impact = classify_native_testmon_changes(
        tmp_path,
        (
            "tests/benchmarks/test_scale.py",
            "tests/benchmarks/deleted.py",
            "packaging/polylogue.nix",
            "packaging/hatch_build.py",
            "docs/release.md",
        ),
    )

    assert impact.executable_paths == ()
    assert impact.runtime_data_paths == ("packaging/hatch_build.py", "packaging/polylogue.nix")


def test_testmon_schema_matches_the_tested_dependency_contract(tmp_path: Path) -> None:
    """Changing the pinned testmon schema or required columns makes this fail."""
    import testmon.db

    database = tmp_path / "testmondata"
    db = testmon.db.DB(str(database))
    try:
        assert _testmon_schema_version() == 14
        assert tuple(db.con.execute("PRAGMA user_version").fetchone()) == (14,)
        for table, expected in {
            "environment": {"id", "environment_name", "system_packages", "python_version"},
            "file_fp": {"id", "filename", "method_checksums", "mtime", "fsha"},
            "test_execution": {"id", "environment_id", "test_name", "duration", "failed", "forced"},
            "test_execution_file_fp": {"test_execution_id", "fingerprint_id"},
        }.items():
            columns = {row[1] for row in db.con.execute(f"PRAGMA table_info({table})")}
            assert expected <= columns
    finally:
        db.con.close()


def test_environment_digest_changes_with_collection_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-q'\n", encoding="utf-8")
    initial = _testmon_environment_digest(tmp_path, pytest_profile="slow=include")

    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-ra'\n", encoding="utf-8")
    config_changed = _testmon_environment_digest(tmp_path, pytest_profile="slow=include")
    monkeypatch.setattr("devtools.testmon_bootstrap._installed_distributions", lambda: (("pytest", "changed"),))
    distributions_changed = _testmon_environment_digest(tmp_path, pytest_profile="slow=include")
    monkeypatch.setenv("POLYLOGUE_CI", "testmon-digest-contract")
    managed_environment_changed = _testmon_environment_digest(tmp_path, pytest_profile="slow=include")
    profile_changed = _testmon_environment_digest(tmp_path, pytest_profile="slow=exclude")

    assert (
        len(
            {
                initial,
                config_changed,
                distributions_changed,
                managed_environment_changed,
                profile_changed,
            }
        )
        == 5
    )


def test_environment_digest_changes_when_root_conftest_is_added(tmp_path: Path) -> None:
    initial = _testmon_environment_digest(tmp_path)

    (tmp_path / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items.reverse()\n",
        encoding="utf-8",
    )

    assert _testmon_environment_digest(tmp_path) != initial


def test_environment_digest_ignores_benchmark_conftest_and_its_plugin_declaration(tmp_path: Path) -> None:
    benchmark_conftest = tmp_path / "tests" / "benchmarks" / "conftest.py"
    benchmark_conftest.parent.mkdir(parents=True)
    benchmark_conftest.write_text("pytest_plugins = dynamic_plugin_names\n", encoding="utf-8")

    initial = _testmon_environment_digest(tmp_path)
    benchmark_conftest.write_text("pytest_plugins = other_dynamic_plugin_names\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) == initial


def test_inactive_runtime_helper_does_not_force_fresh_environment(tmp_path: Path) -> None:
    helper = tmp_path / "tests" / "infra" / "runtime_helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("def answer() -> int:\n    return 41\n", encoding="utf-8")
    initial = _testmon_environment_digest(tmp_path)

    helper.write_text("def answer() -> int:\n    return 42\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) == initial


def test_declared_local_fixture_plugin_changes_environment(tmp_path: Path) -> None:
    conftest = tmp_path / "tests" / "conftest.py"
    plugin = tmp_path / "tests" / "infra" / "fixture_plugin.py"
    plugin.parent.mkdir(parents=True)
    conftest.write_text('pytest_plugins = ("tests.infra.fixture_plugin",)\n', encoding="utf-8")
    plugin.write_text("import pytest\n\n@pytest.fixture\ndef value():\n    return 1\n", encoding="utf-8")
    initial = _testmon_environment_digest(tmp_path)

    plugin.write_text("import pytest\n\n@pytest.fixture\ndef value():\n    return 2\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) != initial


def test_dynamic_pytest_plugin_declaration_fails_closed(tmp_path: Path) -> None:
    conftest = tmp_path / "tests" / "conftest.py"
    conftest.parent.mkdir(parents=True)
    conftest.write_text(
        "from plugin_config import plugin_names\n\npytest_plugins = plugin_names\n",
        encoding="utf-8",
    )

    with pytest.raises(NativeTestmonRepairError, match="must be a literal string/list/tuple"):
        _testmon_environment_digest(tmp_path)

    conftest.write_text(
        'pytest_plugins = []\npytest_plugins.append("tests.infra.fixture_plugin")\n',
        encoding="utf-8",
    )
    with pytest.raises(NativeTestmonRepairError, match="must be one literal assignment"):
        _testmon_environment_digest(tmp_path)

    conftest.write_text(
        'globals()["pytest_plugins"] = ("tests.infra.fixture_plugin",)\n',
        encoding="utf-8",
    )
    with pytest.raises(NativeTestmonRepairError, match="must be one literal assignment"):
        _testmon_environment_digest(tmp_path)


def test_environment_digest_ignores_neutralized_pytest_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "local_plugin.py"
    plugin.write_text("VALUE = 'v1'\n", encoding="utf-8")
    monkeypatch.setenv("PYTEST_PLUGINS", "local_plugin")

    initial = _testmon_environment_digest(tmp_path)
    plugin.write_text("VALUE = 'v2'\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) == initial


@pytest.mark.parametrize("addopts", ["-p local_plugin", "-p=local_plugin"])
def test_environment_digest_ignores_plugins_from_neutralized_pytest_addopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    addopts: str,
) -> None:
    plugin = tmp_path / "local_plugin.py"
    plugin.write_text("VALUE = 'v1'\n", encoding="utf-8")
    monkeypatch.setenv("PYTEST_ADDOPTS", addopts)

    initial = _testmon_environment_digest(tmp_path)
    plugin.write_text("VALUE = 'v2'\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) == initial


def test_environment_digest_ignores_neutralized_pytest_addopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _testmon_environment_digest(tmp_path)

    monkeypatch.setenv("PYTEST_ADDOPTS", "--setup-only --ignore-glob=tests/**")

    assert _testmon_environment_digest(tmp_path) == initial


def test_environment_digest_stops_at_invocation_deadline(tmp_path: Path) -> None:
    with pytest.raises(NativeTestmonDeadlineError, match="invocation deadline"):
        _testmon_environment_digest(tmp_path, deadline_monotonic=0.0)


def test_plugin_declaration_discovery_stops_at_invocation_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_declaration = tmp_path / "tests" / "test_plugins.py"
    plugin_declaration.parent.mkdir()
    plugin_declaration.write_text('pytest_plugins = ("fixture_plugin",)\n', encoding="utf-8")
    clock = {"value": 0.0}
    original_read_text = Path.read_text

    def expire_after_plugin_discovery(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        source = original_read_text(path, encoding=encoding, errors=errors)
        if path == plugin_declaration:
            clock["value"] = 1.0
        return source

    monkeypatch.setattr("devtools.testmon_bootstrap.time.monotonic", lambda: clock["value"])
    monkeypatch.setattr(Path, "read_text", expire_after_plugin_discovery)

    with pytest.raises(NativeTestmonDeadlineError, match="invocation deadline"):
        _testmon_environment_digest(tmp_path, deadline_monotonic=0.5)


def test_invalid_cleanup_removes_only_owned_sqlite_and_sidecars(tmp_path: Path) -> None:
    state_dir = tmp_path / ".cache" / "testmon"
    state_dir.mkdir(parents=True)
    owned = [state_dir / "testmondata", state_dir / "testmondata-wal", state_dir / "testmondata-shm"]
    unrelated = state_dir / "keep.txt"
    for path in (*owned, unrelated):
        path.write_text(path.name, encoding="utf-8")

    removed = remove_invalid_native_testmon_state(tmp_path)

    assert set(removed) == set(owned)
    assert unrelated.read_text(encoding="utf-8") == "keep.txt"


def test_invalid_cleanup_refuses_directory_at_database_path(tmp_path: Path) -> None:
    (tmp_path / ".cache" / "testmon" / "testmondata").mkdir(parents=True)

    with pytest.raises(NativeTestmonRepairError, match="refusing to remove directory"):
        remove_invalid_native_testmon_state(tmp_path)


@pytest.mark.parametrize("symlinked_parent", [".cache", ".cache/testmon"])
def test_invalid_cleanup_refuses_symlinked_state_parents(
    tmp_path: Path,
    symlinked_parent: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "testmon" / "testmondata" if symlinked_parent == ".cache" else outside / "testmondata"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("external state", encoding="utf-8")
    parent = tmp_path / symlinked_parent
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(NativeTestmonRepairError, match="symlinked owned testmon parent"):
        remove_invalid_native_testmon_state(tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "external state"


def test_native_preparation_rejects_symlinked_state_parent_before_inspection(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "testmon").symlink_to(outside, target_is_directory=True)

    with pytest.raises(NativeTestmonRepairError, match="symlinked owned testmon parent"):
        prepare_native_testmon_environment(tmp_path)


def test_native_inspection_rejects_symlinked_database_before_sqlite_open(tmp_path: Path) -> None:
    outside = tmp_path / "outside.db"
    outside.write_text("external state", encoding="utf-8")
    data = tmp_path / "testmondata"
    data.symlink_to(outside)

    state = inspect_native_testmon_environment(data, environment_name="owned-environment")

    assert state.status == "invalid"
    assert state.reason == "native testmon database is not a single-link regular file"
    assert outside.read_text(encoding="utf-8") == "external state"


def test_native_inspection_rejects_symlinked_sidecar_before_sqlite_open(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    data.write_text("not opened", encoding="utf-8")
    outside = tmp_path / "outside-wal"
    outside.write_text("external sidecar", encoding="utf-8")
    sidecar = Path(f"{data}-wal")
    sidecar.symlink_to(outside)

    state = inspect_native_testmon_environment(data, environment_name="owned-environment")

    assert state.status == "invalid"
    assert state.reason == f"native testmon sidecar is not a single-link regular file: {sidecar}"
    assert outside.read_text(encoding="utf-8") == "external sidecar"


@pytest.mark.parametrize("suffix", ("", "-wal"))
def test_native_testmon_ownership_rejects_hardlinked_database_and_sidecars(tmp_path: Path, suffix: str) -> None:
    state_dir = tmp_path / ".cache" / "testmon"
    state_dir.mkdir(parents=True)
    outside = tmp_path / f"outside{suffix}"
    outside.write_text("external state", encoding="utf-8")
    owned = state_dir / f"testmondata{suffix}"
    os.link(outside, owned)

    with pytest.raises(NativeTestmonRepairError, match="single-link regular file"):
        validate_native_testmon_state_ownership(tmp_path)

    assert outside.read_text(encoding="utf-8") == "external state"


@pytest.mark.parametrize("suffix", ("", "-wal"))
def test_native_inspection_rejects_hardlinked_database_and_sidecars(tmp_path: Path, suffix: str) -> None:
    data = tmp_path / "testmondata"
    if suffix:
        data.write_text("not opened", encoding="utf-8")
    outside = tmp_path / f"outside{suffix}"
    outside.write_text("external state", encoding="utf-8")
    owned = Path(f"{data}{suffix}")
    os.link(outside, owned)

    state = inspect_native_testmon_environment(data, environment_name="owned-environment")

    assert state.status == "invalid"
    subject = "database" if not suffix else "sidecar"
    assert state.reason == f"native testmon {subject} is not a single-link regular file" + (
        "" if not suffix else f": {owned}"
    )
    assert outside.read_text(encoding="utf-8") == "external state"


@pytest.mark.parametrize("alias_suffix", ("foreign.tmp", "123-not-a-uuid.tmp"))
def test_source_binding_preserves_foreign_or_malformed_same_inode_alias(
    tmp_path: Path,
    alias_suffix: str,
) -> None:
    """Reclamation must not delete arbitrary same-inode private-looking names."""
    source_root = tmp_path / "source"
    (source_root / "tests").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    alias = source_data.with_name(f".{source_data.name}.bound-{alias_suffix}")
    os.link(source_data, alias)

    with pytest.raises(NativeTestmonRepairError, match="child changed while binding"):
        with native_testmon_source_binding(source_data):
            pytest.fail("a foreign or malformed alias must reject the source bind")

    assert alias.exists(), "a foreign or malformed alias was reclaimed by broad prefix matching"
    assert source_data.stat().st_nlink == 2


def test_source_binding_reclaims_exactly_shaped_stale_same_inode_alias(tmp_path: Path) -> None:
    """A crash-left alias produced by this module remains reclaimable."""
    source_root = tmp_path / "source"
    (source_root / "tests").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    alias = source_data.with_name(f".{source_data.name}.bound-123-{'a' * 32}.tmp")
    os.link(source_data, alias)

    with native_testmon_source_binding(source_data) as binding:
        assert binding is not None

    assert not alias.exists()
    assert source_data.stat().st_nlink == 1


def test_native_inspection_recovers_sidecars_through_retained_source_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sidecar recovery cannot reopen a replacement of the public source path."""
    import sqlite3

    source_root = tmp_path / "source"
    replacement_root = tmp_path / "replacement"
    (source_root / "tests").mkdir(parents=True)
    (replacement_root / "tests").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_original.py",
        recorded_test_name="tests/test_original.py::test_original",
    )
    replacement_data = _seed_partial_native_graph(
        replacement_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_twin.py",
        recorded_test_name="tests/test_twin.py::test_twin",
    )
    original_identity = (source_data.stat().st_dev, source_data.stat().st_ino)
    Path(f"{source_data}-wal").write_bytes(b"stale sidecar")
    original_connect = sqlite3.connect
    recovery_databases: list[Path] = []
    executed_sql: list[str] = []

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        database_path = Path(database)
        recovery_databases.append(database_path)
        if database_path != source_data and replacement_data.exists():
            os.replace(replacement_data, source_data)
        connection = cast(sqlite3.Connection, original_connect(database, *args, **kwargs))
        connection.set_trace_callback(executed_sql.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    with native_testmon_source_binding(source_data) as binding:
        assert binding is not None
        assert Path(f"{binding.data_path}-wal").exists(), binding
        state = inspect_native_testmon_environment(
            source_data,
            environment_name="owned-environment",
            data_fd=binding.descriptor,
            bound_data_path=binding.data_path,
            bound_sidecar_fds=dict(binding.sidecar_descriptors),
        )

    assert state.valid
    assert state.environment is not None
    assert state.environment.nodeids == ("tests/test_original.py::test_original",)
    assert recovery_databases
    assert all(database != source_data for database in recovery_databases)
    assert "PRAGMA wal_checkpoint(PASSIVE)" in executed_sql
    current_identity = (source_data.stat().st_dev, source_data.stat().st_ino)
    assert current_identity != original_identity
    assert not replacement_data.exists()


def _seed_partial_native_graph(
    root: Path,
    *,
    environment_name: str,
    fingerprinted: str,
    recorded_test_name: str = "tests/test_recorded.py::test_recorded",
) -> Path:
    """Write a sound testmon database that covers only one executable path.

    This is the shape an interrupted bootstrap leaves behind: real recorded
    executions, a well-formed environment row, and a dependency graph that has
    simply not reached every changed module yet.
    """
    import testmon.db

    data = root / TESTMON_DATA_RELPATH
    data.parent.mkdir(parents=True, exist_ok=True)
    db = testmon.db.DB(str(data))
    try:
        con = db.con
        environment_id = con.execute(
            "INSERT INTO environment (environment_name, system_packages, python_version) VALUES (?, ?, ?)",
            (environment_name, "", "3.14"),
        ).lastrowid
        execution_id = con.execute(
            "INSERT INTO test_execution (environment_id, test_name, duration, failed, forced) VALUES (?, ?, ?, ?, ?)",
            (environment_id, recorded_test_name, 0.01, 0, 0),
        ).lastrowid
        fingerprint_id = con.execute(
            "INSERT INTO file_fp (filename, method_checksums, mtime, fsha) VALUES (?, ?, ?, ?)",
            (fingerprinted, b"", 0.0, ""),
        ).lastrowid
        con.execute(
            "INSERT INTO test_execution_file_fp (test_execution_id, fingerprint_id) VALUES (?, ?)",
            (execution_id, fingerprint_id),
        )
        con.commit()
    finally:
        db.con.close()
    return data


def test_atomic_copy_uses_dev_fd_when_proc_fd_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The macOS descriptor namespace fallback remains descriptor-bound."""
    probe = tmp_path / "descriptor-probe"
    probe_fd = os.open(probe, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not (Path("/dev/fd") / str(probe_fd)).exists():
            pytest.skip("this platform has no /dev/fd descriptor namespace")
    finally:
        os.close(probe_fd)

    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    (source_root / "tests").mkdir(parents=True)
    (destination_root / ".cache" / "testmon").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    monkeypatch.setattr(
        testmon_bootstrap,
        "_DESCRIPTOR_FD_ROOTS",
        (tmp_path / "missing-proc-fd", Path("/dev/fd")),
    )

    _atomic_copy_sqlite_database(
        source_data,
        destination_data,
        environment_name="owned-environment",
        required_executable_paths=(),
        deadline_monotonic=None,
    )

    assert inspect_native_testmon_environment(
        destination_data,
        environment_name="owned-environment",
    ).valid


def test_descriptor_bound_path_fails_closed_without_a_descriptor_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fd = os.open(tmp_path / "descriptor-probe", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        monkeypatch.setattr(testmon_bootstrap, "_DESCRIPTOR_FD_ROOTS", (tmp_path / "missing",))
        with pytest.raises(NativeTestmonRepairError, match="descriptor-bound filesystem namespace"):
            testmon_bootstrap._descriptor_bound_path(fd)
    finally:
        os.close(fd)


def test_atomic_copy_holds_destination_directory_across_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    (source_root / "tests").mkdir(parents=True)
    (destination_root / ".cache" / "testmon").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("external", encoding="utf-8")
    original_replace = os.replace

    def replace_after_parent_swap(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        owned_cache = destination_root / ".cache"
        owned_cache.rename(destination_root / ".cache-owned")
        owned_cache.symlink_to(external, target_is_directory=True)
        original_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "replace", replace_after_parent_swap)
    _atomic_copy_sqlite_database(
        source_data,
        destination_data,
        environment_name="owned-environment",
        required_executable_paths=(),
        deadline_monotonic=None,
    )

    assert (destination_root / ".cache-owned" / "testmon" / "testmondata").is_file()
    assert sentinel.read_text(encoding="utf-8") == "external"
    assert list(external.iterdir()) == [sentinel]


def test_atomic_copy_does_not_install_a_valid_replacement_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid replacement of the old pathname cannot win publication."""
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    (source_root / "tests").mkdir(parents=True)
    (destination_root / ".cache" / "testmon").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    replacement_root = tmp_path / "replacement"
    replacement_data = _seed_partial_native_graph(
        replacement_root,
        environment_name="different-environment",
        fingerprinted="tests/test_recorded.py",
    )
    original_replace = os.replace
    attacked = False

    def replace_after_validation(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal attacked
        publication_entries = tuple(destination_data.parent.glob(f".{destination_data.name}.publish-*.tmp"))
        if publication_entries and not attacked:
            attacked = True
            original_replace(replacement_data, publication_entries[0])
        original_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "replace", replace_after_validation)
    _atomic_copy_sqlite_database(
        source_data,
        destination_data,
        environment_name="owned-environment",
        required_executable_paths=(),
        deadline_monotonic=None,
    )

    assert attacked
    assert len(tuple(destination_data.parent.glob(f".{destination_data.name}.publish-*.tmp"))) == 0
    copied = inspect_native_testmon_environment(destination_data, environment_name="owned-environment")
    replacement = inspect_native_testmon_environment(destination_data, environment_name="different-environment")
    assert copied.valid
    assert replacement.status == "absent"


def test_atomic_copy_retains_source_child_across_source_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source replacement after inspection cannot redirect the SQLite copy."""
    import sqlite3

    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    replacement_root = tmp_path / "replacement"
    (source_root / "tests").mkdir(parents=True)
    (destination_root / ".cache" / "testmon").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    replacement_data = _seed_partial_native_graph(
        replacement_root,
        environment_name="replacement-environment",
        fingerprinted="tests/test_recorded.py",
    )
    inspected = inspect_native_testmon_environment(source_data, environment_name="owned-environment")
    assert inspected.valid
    original_connect = sqlite3.connect
    replaced = False

    def replace_before_source_open(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement_data, source_data)
        return cast(sqlite3.Connection, original_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", replace_before_source_open)
    _atomic_copy_sqlite_database(
        source_data,
        destination_data,
        environment_name="owned-environment",
        required_executable_paths=(),
        deadline_monotonic=None,
    )

    assert replaced
    assert inspect_native_testmon_environment(destination_data, environment_name="owned-environment").valid
    assert (
        inspect_native_testmon_environment(
            destination_data,
            environment_name="replacement-environment",
        ).status
        == "absent"
    )


def test_atomic_copy_does_not_consume_replaced_private_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the actual .bound-* entry cannot redirect SQLite source open.

    Both databases are valid and carry the same environment name. The node id
    is the distinguishing evidence, so an unsafe replacement cannot pass by
    making the copied database fail a later environment check.
    """
    import sqlite3

    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    replacement_root = tmp_path / "replacement"
    (source_root / "tests").mkdir(parents=True)
    (destination_root / ".cache" / "testmon").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
        recorded_test_name="tests/test_original.py::test_original",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    replacement_data = _seed_partial_native_graph(
        replacement_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
        recorded_test_name="tests/test_twin.py::test_twin",
    )
    source_state = inspect_native_testmon_environment(
        source_data,
        environment_name="owned-environment",
    )
    assert source_state.valid
    assert source_state.environment is not None
    assert source_state.environment.nodeids == ("tests/test_original.py::test_original",)
    replacement_state = inspect_native_testmon_environment(
        replacement_data,
        environment_name="owned-environment",
    )
    assert replacement_state.valid
    assert replacement_state.environment is not None
    assert replacement_state.environment.nodeids == ("tests/test_twin.py::test_twin",)
    original_connect = sqlite3.connect
    replaced = False
    private_entry: Path | None = None

    def replace_private_bound_source(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal private_entry, replaced
        if not replaced:
            private_entries = tuple(source_data.parent.glob(f".{source_data.name}.bound-*.tmp"))
            assert len(private_entries) == 1
            private_entry = private_entries[0]
            replaced = True
            os.replace(replacement_data, private_entry)
        return cast(sqlite3.Connection, original_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", replace_private_bound_source)
    try:
        _atomic_copy_sqlite_database(
            source_data,
            destination_data,
            environment_name="owned-environment",
            required_executable_paths=(),
            deadline_monotonic=None,
        )
    except NativeTestmonRepairError:
        # A platform may fail closed when the retained descriptor namespace
        # cannot safely be reopened after the private entry is replaced.
        assert not destination_data.exists()
    else:
        copied = inspect_native_testmon_environment(
            destination_data,
            environment_name="owned-environment",
        )
        assert copied.valid
        assert copied.environment is not None
        assert copied.environment.nodeids == ("tests/test_original.py::test_original",)

    assert replaced
    assert private_entry is not None
    replacement = inspect_native_testmon_environment(private_entry, environment_name="owned-environment")
    assert replacement.valid
    assert replacement.environment is not None
    assert replacement.environment.nodeids == ("tests/test_twin.py::test_twin",)


def test_source_binding_closes_descriptors_when_bound_entry_becomes_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private-entry cleanup leaves a directory replacement untouched."""
    source_root = tmp_path / "source"
    (source_root / "tests").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_recorded.py",
    )
    directory_fds: list[int] = []
    original_open_directory = testmon_bootstrap._open_owned_testmon_directory

    def capture_directory_fd(*args: Any, **kwargs: Any) -> int:
        descriptor = original_open_directory(*args, **kwargs)
        directory_fds.append(descriptor)
        return descriptor

    monkeypatch.setattr(testmon_bootstrap, "_open_owned_testmon_directory", capture_directory_fd)
    bound_fd: int | None = None
    private_entry: Path | None = None
    with native_testmon_source_binding(source_data) as binding:
        assert binding is not None
        bound_fd = binding.descriptor
        private_entry = next(source_data.parent.glob(f".{source_data.name}.bound-*.tmp"))
        replacement_directory = tmp_path / "replacement-directory"
        replacement_directory.mkdir()
        private_entry.unlink()
        replacement_directory.rename(private_entry)

    assert bound_fd is not None
    assert directory_fds
    for descriptor in (*directory_fds, bound_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert private_entry is not None
    assert private_entry.is_dir()
    private_entry.rmdir()


def test_atomic_copy_carries_validated_source_descriptor_across_public_replacement(
    tmp_path: Path,
) -> None:
    """A public replacement after validation cannot redirect the retained source."""
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    replacement_root = tmp_path / "replacement"
    (source_root / "tests").mkdir(parents=True)
    (destination_root / ".cache" / "testmon").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_original.py",
        recorded_test_name="tests/test_original.py::test_original",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    replacement_data = _seed_partial_native_graph(
        replacement_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_twin.py",
        recorded_test_name="tests/test_twin.py::test_twin",
    )

    with native_testmon_source_binding(source_data) as binding:
        assert binding is not None
        inspected = inspect_native_testmon_environment(
            source_data,
            environment_name="owned-environment",
            data_fd=binding.descriptor,
        )
        assert inspected.valid
        os.replace(replacement_data, source_data)
        _atomic_copy_sqlite_database(
            source_data,
            destination_data,
            environment_name="owned-environment",
            required_executable_paths=(),
            deadline_monotonic=None,
            source_fd=binding.descriptor,
        )

    copied = inspect_native_testmon_environment(destination_data, environment_name="owned-environment")
    assert copied.valid
    assert copied.environment is not None
    assert copied.environment.nodeids == ("tests/test_original.py::test_original",)


def test_atomic_copy_retains_wal_backed_bound_source_after_public_replacement(tmp_path: Path) -> None:
    """A bound SQLite family carries WAL-only graph rows across public replacement."""
    import sqlite3

    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    replacement_root = tmp_path / "replacement"
    for root in (source_root, destination_root, replacement_root):
        (root / "tests").mkdir(parents=True)
    source_data = _seed_partial_native_graph(
        source_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_original.py",
        recorded_test_name="tests/test_original.py::test_original",
    )
    replacement_data = _seed_partial_native_graph(
        replacement_root,
        environment_name="owned-environment",
        fingerprinted="tests/test_twin.py",
        recorded_test_name="tests/test_twin.py::test_twin",
    )
    destination_data = destination_root / TESTMON_DATA_RELPATH
    with sqlite3.connect(source_data) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "UPDATE test_execution SET test_name = ?",
            ("tests/test_wal.py::test_wal",),
        )
        connection.commit()
        assert Path(f"{source_data}-wal").exists()
        with native_testmon_source_binding(source_data) as binding:
            assert binding is not None
            assert "-wal" in dict(binding.sidecar_descriptors)
            os.replace(replacement_data, source_data)
            _atomic_copy_sqlite_database(
                source_data,
                destination_data,
                environment_name="owned-environment",
                required_executable_paths=(),
                deadline_monotonic=None,
                source_fd=binding.descriptor,
                bound_source_path=binding.data_path,
            )

    copied = inspect_native_testmon_environment(destination_data, environment_name="owned-environment")
    assert copied.valid
    assert copied.environment is not None
    assert copied.environment.nodeids == ("tests/test_wal.py::test_wal",)


def test_partial_bootstrap_graph_is_incomplete_rather_than_invalid(tmp_path: Path) -> None:
    """An interrupted bootstrap is resumable state, not corruption."""
    covered, uncovered = "polylogue/covered.py", "polylogue/uncovered.py"
    (tmp_path / "polylogue").mkdir()
    for relative in (covered, uncovered):
        (tmp_path / relative).write_text("value = 1\n", encoding="utf-8")
    environment_name = _testmon_environment_digest(tmp_path)
    data = _seed_partial_native_graph(tmp_path, environment_name=environment_name, fingerprinted=covered)

    state = inspect_native_testmon_environment(
        data,
        environment_name=environment_name,
        required_executable_paths=(covered, uncovered),
    )

    assert state.status == "incomplete"
    assert state.resumable is True
    assert state.valid is False
    assert state.missing_executable_paths == (uncovered,)


def test_a_resumable_graph_drives_affected_selection_rather_than_a_full_corpus(tmp_path: Path) -> None:
    """OPERATOR DECISION 2026-08-18: prefer the bounded hazard to the standstill.

    A resumable graph is structurally sound and merely lacks edges for some
    changed modules. Discarding its selection and running the complete corpus
    costs ~9.5x a warm run, and the recorded history showed 5.1 of 5.65 hours of
    baseline verification going to runs that selected nothing and ran everything.

    The residual risk is bounded and self-correcting: a test whose ONLY
    dependency is an un-fingerprinted module may not be selected on this run, but
    the run still records edges for everything it executes, so it is selected on
    the next one. The uncovered paths are named in the receipt instead of being
    paid for on every invocation.
    """
    covered, uncovered = "polylogue/covered.py", "polylogue/uncovered.py"
    (tmp_path / "polylogue").mkdir()
    for relative in (covered, uncovered):
        (tmp_path / relative).write_text("value = 1\n", encoding="utf-8")
    environment_name = _testmon_environment_digest(tmp_path)
    data = _seed_partial_native_graph(tmp_path, environment_name=environment_name, fingerprinted=covered)
    recorded_bytes = data.read_bytes()

    preparation = prepare_native_testmon_environment(tmp_path, required_executable_paths=(covered, uncovered))

    assert preparation.selection_mode == "affected"
    assert preparation.removed_paths == ()
    assert data.exists(), "an interrupted bootstrap's recorded work must survive into the next invocation"
    assert data.read_bytes() == recorded_bytes
    assert preparation.local_state.missing_executable_paths == (uncovered,), (
        "the receipt must still name what the graph does not cover, so the exposure stays visible"
    )


def test_preparation_still_removes_genuinely_unusable_state(tmp_path: Path) -> None:
    """Corruption is not resumable; only incompleteness is."""
    data = tmp_path / TESTMON_DATA_RELPATH
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(b"not a sqlite database")

    preparation = prepare_native_testmon_environment(tmp_path)

    assert preparation.selection_mode == "bootstrap"
    assert not data.exists()
    assert preparation.removed_paths != ()


def test_probing_an_absent_environment_preserves_another_environments_graph(tmp_path: Path) -> None:
    """A routine environment miss must not delete the shared database.

    One verify invocation prepares twice when the hypothesis-profile fallback
    engages: once for the default-profile digest, then again for the release
    profile. Both names address the same testmon database. Treating "this
    database does not carry my environment" as damaged state made the first
    probe delete the second probe's graph, so the warm path could never engage
    and every run bootstrapped from scratch.
    """
    covered = "polylogue/covered.py"
    (tmp_path / "polylogue").mkdir()
    (tmp_path / covered).write_text("value = 1\n", encoding="utf-8")
    resident = "resident-environment"
    data = _seed_partial_native_graph(tmp_path, environment_name=resident, fingerprinted=covered)

    absent = inspect_native_testmon_environment(data, environment_name="some-other-environment")
    assert absent.status == "absent"
    assert "some-other-environment" in absent.reason

    prepare_native_testmon_environment(
        tmp_path,
        required_executable_paths=(),
        pytest_profile="default",
        pytest_environment={"HYPOTHESIS_PROFILE": "default"},
    )

    assert data.exists(), "probing an absent environment deleted the shared database"
    survivor = inspect_native_testmon_environment(data, environment_name=resident)
    assert survivor.environment is not None
    assert survivor.environment.nodeids == ("tests/test_recorded.py::test_recorded",)


def test_optional_main_source_that_appears_after_binding_stays_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public source appearing after a missed bind cannot become a source."""
    lane = tmp_path / "lane"
    main = tmp_path / "main"
    lane.mkdir()
    main.mkdir()

    monkeypatch.setattr(testmon_bootstrap, "testmon_environment_digest", lambda *_args, **_kwargs: "lane-environment")
    monkeypatch.setattr(
        testmon_bootstrap,
        "linked_worktree_info",
        lambda checkout, **_kwargs: (True, main) if checkout.resolve() == lane.resolve() else None,
    )

    @contextlib.contextmanager
    def source_binding(data_path: Path) -> Iterator[None]:
        assert not data_path.exists()
        _seed_partial_native_graph(
            main,
            environment_name="lane-environment",
            fingerprinted="tests/test_recorded.py",
        )
        yield None

    monkeypatch.setattr(testmon_bootstrap, "native_testmon_source_binding", source_binding)

    original_inspect = testmon_bootstrap.inspect_native_testmon_environment

    def inspect_without_unbound_source_read(
        data_path: Path,
        *,
        data_fd: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if data_path == main / TESTMON_DATA_RELPATH and data_fd is None:
            raise AssertionError("an optional source that appeared after binding must not be read unbound")
        return original_inspect(data_path, data_fd=data_fd, **kwargs)

    monkeypatch.setattr(testmon_bootstrap, "inspect_native_testmon_environment", inspect_without_unbound_source_read)

    preparation = prepare_native_testmon_environment(lane)

    assert preparation.selection_mode == "bootstrap"
    assert preparation.copied_from is None
    assert preparation.local_state.status == "absent"
    assert (main / TESTMON_DATA_RELPATH).is_file()


def test_optional_main_invalid_parent_falls_back_to_lane_local_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsafe optional source parent cannot reject an otherwise safe lane."""
    lane = tmp_path / "lane"
    main = tmp_path / "main"
    lane.mkdir()
    main.mkdir()

    monkeypatch.setattr(testmon_bootstrap, "testmon_environment_digest", lambda *_args, **_kwargs: "lane-environment")
    monkeypatch.setattr(
        testmon_bootstrap,
        "linked_worktree_info",
        lambda checkout, **_kwargs: (True, main) if checkout.resolve() == lane.resolve() else None,
    )
    original_validate = testmon_bootstrap._validate_owned_state_parents

    def reject_main_parent(checkout: Path) -> None:
        if checkout.resolve() == main.resolve():
            raise NativeTestmonRepairError("refusing symlinked owned testmon parent")
        original_validate(checkout)

    monkeypatch.setattr(testmon_bootstrap, "_validate_owned_state_parents", reject_main_parent)

    preparation = prepare_native_testmon_environment(
        lane,
        source_lock_factory=lambda _main_checkout: contextlib.nullcontext(True),
    )

    assert preparation.selection_mode == "bootstrap"
    assert preparation.copied_from is None
    assert preparation.local_state.status == "absent"
    assert preparation.fallback_allowed is True


def test_optional_main_binding_open_error_falls_back_to_lane_local_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional source-open race is typed so bootstrap can fall back."""
    lane = tmp_path / "lane"
    main = tmp_path / "main"
    lane.mkdir()
    (main / "tests").mkdir(parents=True)
    _seed_partial_native_graph(
        main,
        environment_name="lane-environment",
        fingerprinted="tests/test_recorded.py",
    )

    monkeypatch.setattr(testmon_bootstrap, "testmon_environment_digest", lambda *_args, **_kwargs: "lane-environment")
    monkeypatch.setattr(
        testmon_bootstrap,
        "linked_worktree_info",
        lambda checkout, **_kwargs: (True, main) if checkout.resolve() == lane.resolve() else None,
    )

    def binding_open_race(_parent: Path, *, create: bool) -> int:
        assert create is False
        raise OSError("source directory replaced during bind")

    monkeypatch.setattr(testmon_bootstrap, "_open_owned_testmon_directory", binding_open_race)

    preparation = prepare_native_testmon_environment(lane)

    assert preparation.selection_mode == "bootstrap"
    assert preparation.copied_from is None
    assert preparation.local_state.status == "absent"
    assert preparation.fallback_allowed is True


def test_an_interrupted_bootstrap_does_not_delete_every_environments_graph(tmp_path: Path) -> None:
    """A killed bootstrap must not cost the next run its graph -- or anyone else's.

    pytest writes the environment row at startup, so a bootstrap interrupted
    before its first test completes leaves a row with zero recorded executions.
    Classifying that as damaged made the caller delete the whole shared SQLite
    file, and the file is shared by every environment name -- the
    hypothesis-profile fallback alone probes two per invocation. One interrupted
    run therefore reset every graph in the checkout, which is the loop no number
    of retries escapes.
    """
    import sqlite3

    data = tmp_path / TESTMON_DATA_RELPATH
    data.parent.mkdir(parents=True, exist_ok=True)
    environment_name = _testmon_environment_digest(tmp_path)
    import testmon.db

    db = testmon.db.DB(str(data))
    try:
        db.con.execute(
            "INSERT INTO environment (environment_name, system_packages, python_version) VALUES (?, '', '')",
            (environment_name,),
        )
        db.con.commit()
    finally:
        db.con.close()

    state = inspect_native_testmon_environment(data, environment_name=environment_name)

    assert state.status == "absent", "an empty environment is nothing to reuse, not damage to repair"
    assert not state.valid

    preparation = prepare_native_testmon_environment(tmp_path)

    assert preparation.selection_mode == "bootstrap"
    assert preparation.removed_paths == (), "the shared database must survive an interrupted bootstrap"
    assert data.exists()
    with sqlite3.connect(data) as connection:
        rows = connection.execute("SELECT COUNT(*) FROM environment").fetchone()
    assert rows[0] == 1, "other environments' rows must be untouched"


@pytest.mark.parametrize(
    ("nodeid", "canonical"),
    [
        ("tests/test_a.py::test_plain", "tests/test_a.py::test_plain"),
        ("tests/test_a.py::test_p[chatgpt]@chatgpt", "tests/test_a.py::test_p[chatgpt]"),
        ("tests/test_a.py::test_p[user@example.com]", "tests/test_a.py::test_p[user@example.com]"),
        ("tests/test_a.py::test_p[a@b]@group", "tests/test_a.py::test_p[a@b]"),
        ("tests/test_a.py::test_p[x]", "tests/test_a.py::test_p[x]"),
    ],
)
def test_canonical_test_nodeid_strips_only_loadgroup_suffix(nodeid: str, canonical: str) -> None:
    # xdist loadgroup lanes record `id@group`, single-process lanes record `id`;
    # comparing across shapes without one canonical form invented 509 phantom
    # missing/attested tests and a permanent re-execution treadmill.
    assert canonical_test_nodeid(nodeid) == canonical
