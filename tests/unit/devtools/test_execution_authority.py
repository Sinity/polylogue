"""Production selection refuses incompatible evidence and transient execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools import pytest_slot, verify
from devtools.execution_source import ExecutionSourceGuard
from devtools.testmon_provision import inspect_testmon_graph
from devtools.testmon_provision import testmon_datafile as _testmon_datafile
from devtools.verify_runs import git_worktree_content_sha256
from tests.infra.execution_authority import record_graph as _record
from tests.infra.execution_authority import source_repository as _source_repository


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


def test_managed_transient_execution_voids_receipt_and_retains_graph(source_repository: Path) -> None:
    """Endpoint equality alone accepts executed temporary code and its graph."""
    root = source_repository
    recorded = _record(root)
    assert recorded[0] == 0, recorded
    (root / "tests/nested/test_one.py").write_text(
        "from pathlib import Path\n"
        "def test_one():\n"
        "    path = Path('helper.py')\n    original = path.read_bytes()\n"
        "    try:\n        path.write_text('def value():\\n    return 42\\n')\n"
        "        namespace = {}\n        exec(compile(path.read_bytes(), str(path), 'exec'), namespace)\n"
        "        assert namespace['value']() == 42\n"
        "    finally:\n        path.write_bytes(original)\n"
    )
    before = git_worktree_content_sha256(root)
    exit_code, receipt = _record(root)
    assert git_worktree_content_sha256(root) == before
    assert exit_code == 125
    assert receipt["diagnosis"] == "execution_source_unavailable"
    assert receipt["execution_source"]["status"] == "unavailable"
    assert receipt["worktree_provenance"] is None
    metadata = pytest_slot.termination_metadata(pytest_slot.SlotOutcome(exit_code, "held", receipt=receipt))
    assert metadata["diagnosis"] == "execution_source_unavailable"
    assert metadata["worktree_provenance_unknown"] is True
    assert _testmon_datafile(root).exists()
    assert not inspect_testmon_graph(root).usable


@pytest.mark.parametrize("mutation", ["replace", "new-directory", "watch-loss"])
def test_source_observer_refuses_membership_and_coverage_loss(source_repository: Path, mutation: str) -> None:
    root = source_repository
    guard = ExecutionSourceGuard(root)
    if mutation == "replace":
        temporary = root / "replacement.py"
        temporary.write_bytes((root / "helper.py").read_bytes())
        temporary.replace(root / "helper.py")
    elif mutation == "new-directory":
        directory = root / "new_sources"
        directory.mkdir()
        (directory / "new.py").write_text("value = 1\n")
    else:
        (root / "tests/nested").rename(root / "tests/moved")
    assert guard.finish()["status"] == "unavailable"


@pytest.mark.parametrize("failure", ["overflow", "read-fault"])
def test_source_observer_retains_unavailable_coverage(
    source_repository: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A lost event stream cannot be replaced by equal content samples."""
    import os
    import struct

    guard = ExecutionSourceGuard(source_repository)
    original = os.read
    pending = True

    def read(fd: int, count: int) -> bytes:
        nonlocal pending
        if fd == guard.fd and pending:
            pending = False
            if failure == "read-fault":
                raise OSError("synthetic stream fault")
            return struct.pack("iIII", -1, 0x00004000, 0, 0)
        return original(fd, count)

    monkeypatch.setattr(os, "read", read)
    assert guard.finish()["status"] == "unavailable"


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
