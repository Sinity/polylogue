"""Proof tests for the autouse host-clock guard (tests/infra/clock_guard.py).

These assert the guard actually makes the host clock unreachable from
guarded test code — the property the old `test-clock-hygiene` lint could
only detect after the fact. If this file is itself removed from the guard
(e.g. via a stray `uses_real_clock` marker), these tests fail loudly because
the `pytest.raises` blocks would no longer see a raise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest


def test_time_time_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.time()


def test_time_monotonic_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.monotonic()


def test_time_monotonic_ns_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.monotonic_ns()


def test_time_time_ns_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        time.time_ns()


def test_datetime_now_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        datetime.now()


def test_datetime_utcnow_raises_outside_frozen_clock() -> None:
    with pytest.raises(RuntimeError, match="frozen_clock"):
        datetime.utcnow()


def test_frozen_clock_fixture_bypasses_the_guard(frozen_clock: object) -> None:
    # Requesting frozen_clock exempts this test from the raising guard;
    # time.time() should resolve to the frozen clock's controlled value
    # instead of raising.
    assert time.time() == pytest.approx(1700000000.0)


@pytest.mark.uses_real_clock("proves the opt-out marker suppresses the guard")
def test_uses_real_clock_marker_bypasses_the_guard() -> None:
    # Must not raise.
    time.time()
    datetime.now()


@pytest.fixture(scope="module")
def writable_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A writable copy of the executed checkout for nested collection proofs.

    Managed runs execute from a readonly copy of the declared source, so a
    proof module cannot be added to the checkout itself. The copy keeps the
    same Git object store, the exact executed bytes of every declared file and
    its own environment over the original installed dependencies, so the
    nested ``devtools test`` admits it like the real checkout.
    """
    root = Path(__file__).resolve().parents[3]
    destination = tmp_path_factory.mktemp("clock-guard-checkout") / "checkout"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(root), str(destination)],
        check=True,
        capture_output=True,
    )
    # A branch of its own: devtools refuses a run on the clone's default branch.
    subprocess.run(["git", "-C", str(destination), "checkout", "--quiet", "-b", "clock-guard-proof"], check=True)
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    for name in sorted(set(listed.stdout.split(b"\0")) - {b""}):
        source = root / os.fsdecode(name)
        target = destination / os.fsdecode(name)
        if not source.is_file() or source.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copymode(source, target)
    # Devtools refuses an interpreter, or import paths, that resolve into
    # another checkout, so the copy owns its environment: a reflinked copy of
    # the executed checkout's installed dependencies.
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(root / ".venv"), str(destination / ".venv")],
        check=True,
        capture_output=True,
    )
    return destination


def _managed_collection(module: Path, *, root: Path, state: Path) -> subprocess.CompletedProcess[str]:
    """Collect ``module`` through ``devtools test`` without touching operator history.

    The nested run appends its history and evidence rows under ``state`` and
    its receipt tree is removed afterwards, so these proof runs never appear
    in ``devtools why --history`` or count against the checkout's retained
    failure details.
    """
    history = state / "history.jsonl"
    result = subprocess.run(
        [
            str(root / ".venv" / "bin" / "python"),
            "-m",
            "devtools",
            "test",
            "--collect-only",
            "--rootdir",
            str(root),
            str(module),
            "--json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "POLYLOGUE_VERIFY_HISTORY_PATH": str(history),
            "POLYLOGUE_VERIFICATION_EVIDENCE_PATH": str(state / "evidence.jsonl"),
        },
        check=False,
    )
    receipt = json.loads(result.stdout)
    run_id = str(receipt["run_id"])
    shutil.rmtree(root / ".cache" / "verify" / "runs" / run_id)
    # Anti-vacuity: without the overrides the row lands in the operator's
    # shared history and this file does not exist.
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert [row["run_id"] for row in rows] == [run_id]
    return result


@pytest.mark.parametrize(
    "expression",
    ("datetime.now()", "time.time()", "time.time_ns()", "time.monotonic()", "time.monotonic_ns()"),
)
def test_module_level_clock_read_fails_during_managed_collection(
    expression: str, tmp_path: Path, writable_checkout: Path
) -> None:
    """The guard must be armed before pytest imports ordinary test modules."""
    root = writable_checkout
    temporary_root = root / "tests" / f".clock-guard-{uuid4().hex}"
    temporary_root.mkdir()
    violating_module = temporary_root / "test_module_level_clock.py"
    violating_module.write_text(
        "from datetime import datetime\n"
        "import time\n\n"
        f"MODULE_READ = {expression}\n\n"
        "def test_never_collects():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    try:
        result = _managed_collection(violating_module, root=root, state=tmp_path)
    finally:
        shutil.rmtree(temporary_root)

    assert result.returncode != 0
    assert "frozen_clock" in result.stdout + result.stderr


def test_module_level_real_clock_marker_exempts_managed_collection(tmp_path: Path, writable_checkout: Path) -> None:
    root = writable_checkout
    temporary_root = root / "tests" / f".clock-guard-{uuid4().hex}"
    temporary_root.mkdir()
    exempt_module = temporary_root / "test_module_level_clock_exempt.py"
    exempt_module.write_text(
        "import pytest\n"
        "from datetime import datetime\n"
        "import time\n\n"
        "pytestmark = pytest.mark.uses_real_clock('collection benchmark')\n"
        "MODULE_NOW = datetime.now()\n"
        "MODULE_TIME = time.time()\n\n"
        "def test_collects():\n"
        "    assert MODULE_NOW and MODULE_TIME\n",
        encoding="utf-8",
    )
    try:
        result = _managed_collection(exempt_module, root=root, state=tmp_path)
    finally:
        shutil.rmtree(temporary_root)

    assert result.returncode == 0, result.stdout + result.stderr


def _collect_guarded_module(source: str, root: Path) -> subprocess.CompletedProcess[str]:
    directory = root / "tests" / f".clock-guard-{uuid4().hex}"
    directory.mkdir()
    module = directory / "test_guard_regression.py"
    module.write_text(source, encoding="utf-8")
    try:
        return subprocess.run(
            # The checkout's own interpreter: devtools refuses another checkout's.
            [
                str(root / ".venv" / "bin" / "python"),
                "-m",
                "devtools",
                "test",
                "--collect-only",
                "--rootdir",
                str(root),
                str(module),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            check=False,
        )
    finally:
        shutil.rmtree(directory)


@pytest.mark.parametrize(
    "source",
    (
        "import time, pytest\n"
        "CLOCK = time.time()\n"
        "pytestmark = pytest.mark.uses_real_clock('whole module')\n"
        "def test_collected(): pass\n",
        "import time, pytest\n"
        "@pytest.mark.parametrize('epoch', [time.time()])\n"
        "@pytest.mark.uses_real_clock('decorator evaluates a clock')\n"
        "def test_collected(epoch): pass\n",
    ),
    ids=("module-marker-covers-whole-source", "marker-below-clock-reading-decorator"),
)
def test_collection_clock_exemption_covers_declared_source(source: str, writable_checkout: Path) -> None:
    """Both cases fail collection when an exemption starts/ends at the wrong line."""
    result = _collect_guarded_module(source, writable_checkout)
    assert result.returncode == 0, result.stdout + result.stderr


def test_caught_clock_read_cannot_disarm_collection_profile(writable_checkout: Path) -> None:
    """Raising from the profiler lets the first caught read disable the second."""
    result = _collect_guarded_module(
        "import time\n"
        "try:\n"
        "    time.time()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "time.monotonic()\n"
        "def test_collected(): pass\n",
        writable_checkout,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "time.time()" in output and "time.monotonic()" in output


@pytest.mark.uses_real_clock("checks real interpreter profile callbacks on a worker thread")
def test_collection_guard_chains_each_threads_own_previous_profile() -> None:
    from tests.infra import clock_guard

    main_calls: list[int] = []
    worker_calls: list[int] = []

    def probe() -> None:
        pass

    def main_profile(frame: object, event: str, arg: object) -> None:
        if event == "call" and getattr(frame, "f_code", None) is probe.__code__:
            main_calls.append(threading.get_ident())

    def worker_profile(frame: object, event: str, arg: object) -> None:
        if event == "call" and getattr(frame, "f_code", None) is probe.__code__:
            worker_calls.append(threading.get_ident())

    previous_main = sys.getprofile()
    previous_worker = threading.getprofile()
    invoking_thread = threading.get_ident()
    try:
        sys.setprofile(main_profile)
        threading.setprofile(worker_profile)
        clock_guard._install_collection_guard()
        probe()
        worker = threading.Thread(target=probe)
        worker.start()
        worker.join()
    finally:
        clock_guard._remove_collection_guard()
        sys.setprofile(previous_main)
        threading.setprofile(previous_worker)

    assert main_calls == [invoking_thread]
    assert len(worker_calls) == 1 and worker_calls[0] != invoking_thread
