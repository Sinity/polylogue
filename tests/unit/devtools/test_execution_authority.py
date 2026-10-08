"""Production selection refuses incompatible evidence and transient execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from devtools import pytest_memory, pytest_slot, verify
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


def test_managed_snapshot_preserves_actual_host_pid_and_outer_basetemp(
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID isolation makes host basetemp and receipt owners appear dead."""
    import os

    root = source_repository
    parent_namespace = Path("/proc/self/ns/pid").stat().st_ino
    monkeypatch.setenv("SYNTHETIC_PARENT_PID_NAMESPACE", str(parent_namespace))
    monkeypatch.setenv("SYNTHETIC_PARENT_PID", str(os.getpid()))
    outer = root.parent / f"tmp-{os.getpid()}-abcdef"
    outer.mkdir()
    (outer / "active").touch()
    monkeypatch.setenv("SYNTHETIC_OUTER_TEMP", str(outer))
    (root / "tests/nested/test_one.py").write_text(
        "import os\nfrom pathlib import Path\ndef test_one():\n"
        "    pid = os.getpid()\n"
        "    assert int(Path('/proc/self/stat').read_text().split()[0]) == pid\n"
        "    assert int(Path(f'/proc/{pid}/stat').read_text().split()[0]) == pid\n"
        "    assert Path('/proc/self/ns/pid').stat().st_ino == int(os.environ['SYNTHETIC_PARENT_PID_NAMESPACE'])\n"
        "    assert Path('/proc/' + os.environ['SYNTHETIC_PARENT_PID']).exists()\n"
        "    from devtools.pytest_slot import sweep_stale_temp_trees\n"
        "    outer = Path(os.environ['SYNTHETIC_OUTER_TEMP'])\n"
        "    sweep_stale_temp_trees(outer.parent)\n"
        "    assert (outer / 'active').exists()\n"
    )
    exit_code, receipt = _record(root)
    assert exit_code == 0, receipt
    assert receipt["execution_source"]["custody_settled"] is True
    assert (outer / "active").exists()


@pytest.mark.parametrize("fork_on_term", [False, True])
@pytest.mark.parametrize("new_userns", [False, True])
def test_managed_snapshot_settles_detached_children_before_source_publication(
    source_repository: Path, monkeypatch: pytest.MonkeyPatch, fork_on_term: bool, new_userns: bool
) -> None:
    """Group-only settlement leaves a setsid writer alive after the receipt."""
    import time

    root = source_repository
    samplers: list[Any] = []
    if new_userns:
        sampler_type = pytest_memory.ProcessGroupMemorySampler

        def sparse_sampler(*args: Any, **kwargs: Any) -> Any:
            # The initial root sample precedes the nested namespace/detach;
            # this descendant must be discovered from its actual marker.
            kwargs["interval_s"] = 60
            sampler = sampler_type(*args, **kwargs)
            actual_sample = sampler.sample

            def sample_then_release() -> Any:
                sample = actual_sample()
                (root / ".cache/initial-sample").touch()
                return sample

            monkeypatch.setattr(sampler, "sample", sample_then_release)
            samplers.append(sampler)
            return sampler

        monkeypatch.setattr(pytest_slot, "ProcessGroupMemorySampler", sparse_sampler)
    detached = (
        "import ctypes, os, signal, sys, time\nfrom pathlib import Path\n"
        "assert ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) == 0\n"
        "def heartbeat(name):\n"
        "    path = Path('.cache/' + name)\n"
        "    path.with_name(name + '-pid').write_text(str(os.getpid()))\n"
        "    while True:\n        path.write_text(str(time.monotonic_ns()))\n        time.sleep(0.01)\n"
    )
    if fork_on_term:
        detached += (
            "def fork_on_term(*_args):\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    if os.fork() == 0:\n"
            "        os.setsid()\n        signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
            "        heartbeat('fork-heartbeat')\n"
            "    while not Path('.cache/fork-heartbeat').exists():\n        time.sleep(0.001)\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, fork_on_term)\n"
        )
    detached += "heartbeat('heartbeat')\n"
    (root / "tests/nested/test_one.py").write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "def test_one():\n"
        f"    command = [sys.executable, '-c', {detached!r}]\n"
        + (
            "    while not Path('.cache/initial-sample').exists():\n        time.sleep(0.001)\n"
            "    command = ['bwrap', '--unshare-user', '--uid', '0', '--gid', '0', '--bind', '/', '/', '--', *command]\n"
            if new_userns
            else ""
        )
        + "    subprocess.Popen(command, start_new_session=True)\n"
        "    while not Path('.cache/heartbeat').exists():\n        time.sleep(0.01)\n"
    )
    exit_code, receipt = _record(root)
    assert exit_code == 0, receipt
    assert receipt["execution_source"]["custody_settled"] is True
    heartbeat = root / ".cache/heartbeat"
    if new_userns:
        detached_pid = int((root / ".cache/heartbeat-pid").read_text())
        assert detached_pid not in samplers[0]._known_births
    terminal = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == terminal
    if fork_on_term:
        fork = root / ".cache/fork-heartbeat"
        final_fork = fork.read_text()
        time.sleep(0.1)
        assert fork.read_text() == final_fork
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


def test_managed_custody_leaves_an_actual_opaque_foreign_peer_alive(source_repository: Path) -> None:
    """A same-UID/cgroup scan refuses or signals this unrelated opaque birth."""
    import os
    import sys
    import time

    root = source_repository
    ready = root.parent / "peer-ready"
    program = (
        "import ctypes, time\nfrom pathlib import Path\n"
        "assert ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) == 0\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True:\n    time.sleep(0.01)\n"
    )
    peer = subprocess.Popen([sys.executable, "-c", program])
    try:
        while not ready.exists():
            assert peer.poll() is None
            time.sleep(0.001)
        proc = Path("/proc") / str(peer.pid)
        assert proc.stat().st_uid == os.getuid()
        assert "4294967295" not in (proc / "uid_map").read_text()
        with pytest.raises(PermissionError):
            (proc / "environ").read_bytes()
        (root / "tests/nested/test_one.py").write_text(
            "from pathlib import Path\ndef test_one():\n"
            "    Path('.cache/child-cgroup').write_text(Path('/proc/self/cgroup').read_text())\n"
        )
        code, receipt = _record(root)
        assert code == 0, receipt
        assert receipt["execution_source"]["custody_settled"] is True
        assert len(receipt["execution_source"]["attempt_closures"]) == 1
        assert (root / ".cache/child-cgroup").read_text() == (proc / "cgroup").read_text()
        assert peer.poll() is None
    finally:
        peer.terminate()
        peer.wait()


@pytest.mark.parametrize("loss", ["forced-kill", "wrong-token", "missing-closure"])
def test_managed_custody_cannot_publish_without_its_exact_attempt_closure(
    source_repository: Path, monkeypatch: pytest.MonkeyPatch, loss: str
) -> None:
    import os
    import signal
    import time

    from devtools import execution_source

    root = source_repository
    guards: list[Any] = []
    original = execution_source.ExecutionSourceGuard.launched_process
    if loss == "forced-kill":
        (root / "tests/nested/test_one.py").write_text(
            "import time\nfrom pathlib import Path\ndef test_one():\n"
            "    Path('.cache/ready-to-stop').touch()\n    while True:\n        time.sleep(0.01)\n"
        )

    def damage(guard: Any, process: subprocess.Popen[Any]) -> None:
        original(guard, process)
        guards.append(guard)
        if loss == "forced-kill":
            while not (root / ".cache/ready-to-stop").exists():
                assert process.poll() is None
                time.sleep(0.001)
            os.kill(process.pid, signal.SIGKILL)
        elif loss == "wrong-token":
            guard.attempts[-1]["token"] = "wrong-attempt"
        else:
            # The wrapper writes its private record, then this exact receipt
            # disappears before publication; wrapper death alone proves nothing.
            process.wait()
            closure = guard.attempts[-1]["closure"]
            closure.seek(0)
            closure.truncate()

    monkeypatch.setattr(execution_source.ExecutionSourceGuard, "launched_process", damage)
    code, receipt = _record(root)
    assert code == 125, receipt
    assert receipt["execution_source"]["status"] == "unavailable"
    assert receipt["execution_source"]["custody_settled"] is False
    assert guards[0].copy.exists()
    assert not inspect_testmon_graph(root).usable


def test_managed_child_cannot_inherit_or_reopen_the_private_closure_descriptor(
    source_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import os

    root = source_repository
    popen = subprocess.Popen

    def observe(*args: Any, **kwargs: Any) -> Any:
        command = args[0]
        if isinstance(command, list) and len(command) == 5 and command[1:3] == ["-I", "-B"]:
            config = json.loads(command[-1])
            descriptor = config["receipt_fd"]
            info = os.fstat(descriptor)
            kwargs["env"] = {
                **kwargs["env"],
                "SYNTHETIC_CLOSURE_FD": str(descriptor),
                "SYNTHETIC_CLOSURE_INODE": str(info.st_ino),
                "SYNTHETIC_CLOSURE_DEVICE": str(info.st_dev),
            }
        return popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", observe)
    (root / "tests/nested/test_one.py").write_text(
        "import json, os, pytest\nfrom pathlib import Path\ndef test_one():\n"
        "    fd = int(os.environ['SYNTHETIC_CLOSURE_FD'])\n"
        "    try:\n        info = os.fstat(fd)\n"
        "    except OSError:\n        pass\n"
        "    else:\n        assert (info.st_dev, info.st_ino) != (int(os.environ['SYNTHETIC_CLOSURE_DEVICE']), int(os.environ['SYNTHETIC_CLOSURE_INODE']))\n"
        "    pid = os.getppid()\n"
        "    while pid:\n"
        "        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0')\n"
        "        if any(item.startswith(b'{\"command\"') for item in argv):\n"
        "            with pytest.raises(PermissionError):\n                Path(f'/proc/{pid}/fd/{fd}').open('wb')\n"
        "            break\n"
        "        pid = int(Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[1])\n"
        "    else:\n        raise AssertionError('supervisor not found')\n"
    )
    code, receipt = _record(root)
    assert code == 0, receipt
    assert receipt["execution_source"]["custody_settled"] is True


@pytest.mark.parametrize("before_registration", [False, True])
def test_managed_cancel_requests_kernel_settlement_of_opaque_detached_child(
    source_repository: Path, before_registration: bool
) -> None:
    import json
    import os
    import select
    import signal
    import sys
    import time

    root = source_repository
    checkout = Path(verify.__file__).parents[1]
    ready, identity = root / ".cache/cancel-ready", root / ".cache/detached-pid"
    detached = (
        "import ctypes, os, time\nfrom pathlib import Path\n"
        "assert ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) == 0\n"
        "Path('.cache/detached-pid').write_text(str(os.getpid()))\n"
        "while True:\n    time.sleep(0.01)\n"
    )
    program = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {detached!r}], start_new_session=True)\n"
        "while not Path('.cache/detached-pid').exists():\n    time.sleep(0.001)\n"
        "Path('.cache/cancel-ready').touch()\n"
        "while True:\n    time.sleep(0.01)\n"
    )
    launch, log = root / ".cache/cancel-launch.json", root / ".cache/cancel.log"
    environment = {**os.environ, "POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE": "1"}
    environment["TESTMON_DATAFILE"] = str(_testmon_datafile(root))
    launch.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", program],
                "working_directory": str(root),
                "environment": environment,
                "log_path": str(log),
            }
        )
    )
    registration = root / ".cache/registration-ready"
    registration_gate = (
        "import json, time\nfrom devtools.execution_source import ExecutionSourceGuard\n"
        "def gate(guard, process):\n"
        f"    marker = Path({str(registration)!r})\n"
        "    temporary = marker.with_suffix('.tmp')\n"
        "    temporary.write_text(json.dumps({'copy': str(guard.copy), 'pid': process.pid}))\n"
        "    temporary.replace(marker)\n"
        "    while True:\n        time.sleep(0.001)\n"
        "ExecutionSourceGuard.launched_process = gate\n"
        if before_registration
        else ""
    )
    controller = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(checkout)!r})\n"
        "from devtools import pytest_slot\n"
        "pytest_slot.admission_ledger = lambda _env: None\n"
        "pytest_slot.admit_width = lambda argv, **_kwargs: (list(argv), None)\n"
        + registration_gate
        + f"raise SystemExit(pytest_slot._run_launch(Path({str(launch)!r})))\n"
    )
    waiter = subprocess.Popen([sys.executable, "-c", controller])
    observed_pidfd: int | None = None
    observed: dict[str, Any] = {}
    try:
        while not ready.exists() or (before_registration and not registration.exists()):
            assert waiter.poll() is None
            time.sleep(0.001)
        detached_pid = int(identity.read_text())
        if before_registration:
            observed = json.loads(registration.read_text())
            observed_pidfd = os.pidfd_open(observed["pid"])
        waiter.send_signal(signal.SIGTERM)
        assert waiter.wait(timeout=10) == 128 + signal.SIGTERM
        receipt = json.loads(pytest_slot._slot_result_path(log).read_text())
        assert receipt["status"] == "interrupted"
        assert receipt["execution_source"]["status"] == "unavailable"
        assert not inspect_testmon_graph(root).usable
        if before_registration:
            assert receipt["execution_source"]["custody_settled"] is False
            assert receipt["execution_source"]["attempt_closures"] == []
            assert Path(observed["copy"]).exists()
            # Cancellation must settle physical custody before returning,
            # without supplying the missing original attempt binding.
            poller = select.poll()
            assert observed_pidfd is not None
            poller.register(observed_pidfd, select.POLLIN)
            assert any(events & select.POLLIN for _fd, events in poller.poll(0))
        else:
            assert receipt["execution_source"]["custody_settled"] is True
            assert len(receipt["execution_source"]["attempt_closures"]) == 1
        assert not Path(f"/proc/{detached_pid}").exists()
    finally:
        if waiter.poll() is None:
            waiter.kill()
        waiter.wait()
        if observed_pidfd is not None:
            os.close(observed_pidfd)
