from __future__ import annotations

from pathlib import Path

import pytest

from devtools.testmon_bootstrap import (
    NativeTestmonDeadlineError,
    NativeTestmonRepairError,
    classify_native_testmon_changes,
    classify_source_ast,
    executable_python_paths,
    remove_invalid_native_testmon_state,
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


def test_executable_paths_require_current_runtime_modules_but_allow_deletion(tmp_path: Path) -> None:
    module = tmp_path / "polylogue" / "runtime.py"
    module.parent.mkdir()
    module.write_text("VALUE = factory()\n", encoding="utf-8")
    malformed = module.with_name("malformed.py")
    malformed.write_text("def broken(:\n", encoding="utf-8")

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
    monkeypatch.setenv("PYTEST_ADDOPTS", "--strict-markers")
    environment_changed = _testmon_environment_digest(tmp_path, pytest_profile="slow=include")
    profile_changed = _testmon_environment_digest(tmp_path, pytest_profile="slow=exclude")

    assert (
        len(
            {
                initial,
                config_changed,
                distributions_changed,
                environment_changed,
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


def test_environment_digest_hashes_explicit_local_plugin_regardless_of_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "local_plugin.py"
    plugin.write_text("VALUE = 'v1'\n", encoding="utf-8")
    monkeypatch.setenv("PYTEST_PLUGINS", "local_plugin")

    initial = _testmon_environment_digest(tmp_path)
    plugin.write_text("VALUE = 'v2'\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) != initial


@pytest.mark.parametrize("addopts", ["-p local_plugin", "-p=local_plugin"])
def test_environment_digest_hashes_local_plugin_from_pytest_addopts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    addopts: str,
) -> None:
    plugin = tmp_path / "local_plugin.py"
    plugin.write_text("VALUE = 'v1'\n", encoding="utf-8")
    monkeypatch.setenv("PYTEST_ADDOPTS", addopts)

    initial = _testmon_environment_digest(tmp_path)
    plugin.write_text("VALUE = 'v2'\n", encoding="utf-8")

    assert _testmon_environment_digest(tmp_path) != initial


def test_environment_digest_stops_at_invocation_deadline(tmp_path: Path) -> None:
    with pytest.raises(NativeTestmonDeadlineError, match="invocation deadline"):
        _testmon_environment_digest(tmp_path, deadline_monotonic=0.0)


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
