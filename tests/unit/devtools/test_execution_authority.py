"""Production selection refuses incompatible evidence and transient execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_slot, verify
from devtools.testmon_provision import inspect_testmon_graph
from devtools.testmon_provision import testmon_datafile as _testmon_datafile
from devtools.verify_runs import git_worktree_content_sha256
from tests.infra.execution_authority import record_graph as _record
from tests.infra.execution_authority import source_repository as _source_repository

pytestmark = pytest.mark.uses_real_clock


@pytest.fixture
def source_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _source_repository(tmp_path, monkeypatch)


@pytest.mark.parametrize("policy", ["autouse", "warnings", "profile"])
def test_real_graph_cannot_select_under_changed_execution_policy(source_repository: Path, policy: str) -> None:
    """Removing policy from graph identity admits the unrelated test and misses failure."""
    root = source_repository
    if policy == "warnings":
        (root / "tests/nested/test_one.py").write_text(
            "import warnings\nfrom helper import value\ndef test_one():\n    assert value()\n    warnings.warn('synthetic', DeprecationWarning)\n"
        )
    elif policy == "profile":
        (root / "tests/nested/test_one.py").write_text(
            "from hypothesis import given, settings, strategies as st\n"
            "@given(st.booleans())\ndef test_one(value):\n"
            "    if settings.default.max_examples > 10:\n"
            "        from helper import value\n        assert value()\n"
        )
    profile = "verify" if policy == "profile" else "default"
    recorded = _record(root, profile=profile)
    assert recorded[0] == 0, recorded
    assert inspect_testmon_graph(root, profile=profile).usable
    (root / "tests/test_other.py").write_text(
        "from neutral import value\ndef test_other():\n    assert value() is True\n"
    )
    if policy == "autouse":
        (root / "tests/nested/conftest.py").write_text(
            "import pytest\n@pytest.fixture(autouse=True)\ndef new_fixture():\n    raise AssertionError('synthetic')\n"
        )
    elif policy == "warnings":
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ncache_dir = '.cache/pytest'\nfilterwarnings = ['error::DeprecationWarning']\n"
        )
    else:
        (root / "helper.py").write_text("def value():\n    return False\n")
    graph = inspect_testmon_graph(root, profile="default")
    admission = verify._affected_admission(root=root, graph=graph, hypothesis_profile="default")
    assert not admission.admitted
    assert graph.full_rerun_cause
    # The explicit complete route proves the excluded test really fails.
    failed = _record(root)
    assert failed[0] == 1, failed


def test_rename_keeps_removed_python_source_authority(source_repository: Path) -> None:
    root = source_repository
    (root / "docs").mkdir()
    (root / "helper.py").rename(root / "docs/helper.md")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    changed = verify._git_changed_paths(root)
    assert changed is not None and {"helper.py", "docs/helper.md"} <= changed
    assert verify._selection_for_changes(changed) == "affected"


def test_managed_snapshot_executes_admitted_bytes_despite_transient_mmap_restore(
    source_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An endpoint-only boundary executes the temporary mmap value instead."""
    import mmap
    import threading

    root = source_repository
    (root / "helper.py").write_text("value = 1\n")
    (root / "tests/nested/test_one.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "def test_one():\n"
        "    gate = Path(os.environ['SYNTHETIC_GATE'])\n"
        "    gate.with_suffix('.ready').touch()\n"
        "    while not gate.exists():\n        time.sleep(0.01)\n"
        "    import helper\n    assert helper.value == 1\n"
        "    gate.with_suffix('.done').touch()\n"
    )
    gate = root.parent / "gate"
    monkeypatch.setenv("SYNTHETIC_GATE", str(gate))
    before = git_worktree_content_sha256(root)
    errors: list[BaseException] = []

    def mutate() -> None:
        import time

        try:
            while not gate.with_suffix(".ready").exists():
                time.sleep(0.01)
            with (root / "helper.py").open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as mapping:
                mapping[:] = b"value = 2\n"
                mapping.flush()
                gate.touch()
                while not gate.with_suffix(".done").exists():
                    time.sleep(0.01)
                mapping[:] = b"value = 1\n"
                mapping.flush()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=mutate, daemon=True)
    thread.start()
    exit_code, receipt = _record(root)
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert exit_code == 0, receipt
    assert git_worktree_content_sha256(root) == before
    assert receipt["execution_source"]["status"] == "stable"
    assert receipt["execution_source"]["observer"] == "readonly_snapshot"
    assert receipt["worktree_provenance"]["git_worktree_content_sha256"] == before
    assert inspect_testmon_graph(root).usable


@pytest.mark.parametrize("cache_style", ["prefix", "ordinary"])
def test_managed_snapshot_rejects_transient_timestamp_valid_shared_bytecode(
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_style: str,
) -> None:
    """Reading a shared timestamp-valid pyc executes 2 despite copied source 1."""
    import os
    import py_compile
    import sys

    from devtools import execution_source

    root = source_repository
    helper = root / "helper.py"
    helper.write_text("value = 1\n")
    (root / "tests/nested/test_one.py").write_text(
        "import sys, pytest\nfrom pathlib import Path\n"
        "def test_one():\n"
        "    import helper\n    assert helper.value == 1\n"
        "    assert sys.pycache_prefix is not None\n"
        "    with pytest.raises(OSError):\n"
        "        (Path(sys.pycache_prefix) / 'write').write_text('synthetic')\n"
    )
    prefix = root / ".cache/shared-bytecode"
    if cache_style == "prefix":
        monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(prefix))
        cache = prefix / str(root).lstrip("/") / f"helper.{sys.implementation.cache_tag}.pyc"
    else:
        monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
        cache = root / "__pycache__" / f"helper.{sys.implementation.cache_tag}.pyc"
    original = execution_source.ExecutionSourceGuard.__init__
    before = git_worktree_content_sha256(root)

    def start(guard: Any, checkout: Path) -> None:
        original(guard, checkout)
        original_stat = helper.stat()
        # The cached temporary value has the exact timestamp and byte count
        # of the already copied source, then the original bytes are restored.
        timestamp = (guard.copy / "helper.py").stat().st_mtime
        helper.write_text("value = 2\n")
        os.utime(helper, (timestamp, timestamp))
        cache.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(helper), cfile=str(cache), doraise=True)
        helper.write_text("value = 1\n")
        os.utime(helper, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    monkeypatch.setattr(execution_source.ExecutionSourceGuard, "__init__", start)
    exit_code, receipt = _record(root)
    assert exit_code == 0, receipt
    assert git_worktree_content_sha256(root) == before
    assert receipt["execution_source"]["status"] == "stable"
    assert inspect_testmon_graph(root).usable


def test_managed_snapshot_settles_detached_children_before_source_publication(source_repository: Path) -> None:
    """Group-only settlement leaves a setsid writer alive after the receipt."""
    import time

    root = source_repository
    (root / "tests/nested/test_one.py").write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "def test_one():\n"
        "    subprocess.Popen([sys.executable, '-c',\n"
        "        \"import time; from pathlib import Path; p=Path('.cache/heartbeat'); \"\n"
        "        \"exec('while True:\\n p.write_text(str(time.monotonic_ns()))\\n time.sleep(0.01)')\"],\n"
        "        start_new_session=True)\n"
        "    while not Path('.cache/heartbeat').exists():\n        time.sleep(0.01)\n"
    )
    exit_code, receipt = _record(root)
    assert exit_code == 0, receipt
    assert receipt["execution_source"]["pid_namespace_settled"] is True
    heartbeat = root / ".cache/heartbeat"
    terminal = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == terminal
    assert inspect_testmon_graph(root).usable


@pytest.mark.parametrize("mutation", ["write", "replace", "new-directory"])
def test_managed_snapshot_refuses_source_writes(source_repository: Path, mutation: str) -> None:
    root = source_repository
    action = {
        "write": "Path('helper.py').write_text('value = 2')",
        "replace": "Path('.cache/replacement.py').replace('helper.py')",
        "new-directory": "Path('new_sources').mkdir()",
    }[mutation]
    (root / ".cache/replacement.py").write_text("value = 2")
    (root / "tests/nested/test_one.py").write_text(
        "import pytest\nfrom pathlib import Path\ndef test_one():\n"
        "    with pytest.raises(OSError):\n        " + action + "\n"
    )
    assert _record(root)[0] == 0


@pytest.mark.parametrize("race", ["membership", "policy"])
def test_managed_snapshot_rejects_membership_and_policy_admission_races(
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    from devtools import execution_source

    root = source_repository
    original = execution_source.ExecutionSourceGuard.__init__

    def start(guard: Any, checkout: Path) -> None:
        # Policy changes after argv was keyed but before the copy; membership
        # changes after the manifest copy but before provenance identification.
        if race == "policy":
            (root / "tests/nested/conftest.py").write_text("# new policy\n")
        original(guard, checkout)
        if race == "membership":
            (root / "late").mkdir()
            (root / "late/source.py").write_text("value = 1")

    monkeypatch.setattr(execution_source.ExecutionSourceGuard, "__init__", start)
    with pytest.raises(pytest_slot.PytestSlotUnavailableError):
        _record(root)
    assert Path(str(_testmon_datafile(root)) + ".authority-unavailable").exists()


@pytest.mark.parametrize("stage", ["provenance", "launch", "sampler"])
def test_setup_failure_cannot_clear_prior_graph_taint(
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    root = source_repository
    assert _record(root)[0] == 0
    marker = Path(str(_testmon_datafile(root)) + ".authority-unavailable")
    marker.write_text("prior unavailable execution")
    original = subprocess.Popen
    identify = pytest_slot._focused_worktree_provenance
    calls = 0

    def fail_launch(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("process_group") == 0:
            raise OSError("synthetic launch failure")
        return original(*args, **kwargs)

    def fail_provenance(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic provenance failure")
        return identify(*args, **kwargs)

    def fail_sampler(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic sampler failure")

    with monkeypatch.context() as patch:
        if stage == "launch":
            patch.setattr(subprocess, "Popen", fail_launch)
        elif stage == "provenance":
            patch.setattr(pytest_slot, "_focused_worktree_provenance", fail_provenance)
        else:
            patch.setattr(pytest_slot, "ProcessGroupMemorySampler", fail_sampler)
        with pytest.raises(OSError):
            _record(root)
    assert marker.exists()
    assert not inspect_testmon_graph(root).usable
    # A completed subset cannot restore full-graph authority either.
    assert _record(root, subset="tests/test_other.py")[0] == 0
    assert marker.exists()
    assert _record(root)[0] == 0
    assert not marker.exists()
    assert inspect_testmon_graph(root).usable


def test_verify_rerun_uses_the_graph_writers_hypothesis_budget(
    source_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping CLI semantics from verify's rerun clears a full-budget failure."""
    from tests.infra.execution_authority import verify_graph

    root = source_repository
    (root / "tests/nested/test_one.py").write_text(
        "from hypothesis import given, settings, strategies as st\n"
        "@given(st.booleans())\ndef test_one(value):\n"
        "    if settings.default.max_examples > 10:\n"
        "        from helper import value\n        assert value()\n"
    )
    assert _record(root, profile="verify")[0] == 0
    (root / "helper.py").write_text("def value():\n    return False\n")
    monkeypatch.setenv("HYPOTHESIS_PROFILE", "verify")
    exit_code, _elapsed, metadata = verify_graph(root, profile="default")
    assert exit_code == 1
    assert metadata["hypothesis_profile"] == "default"
    assert metadata["rerun"]["still_failed"]
    assert not metadata["rerun"]["flaky"]
