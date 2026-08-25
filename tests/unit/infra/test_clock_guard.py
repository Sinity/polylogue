"""Proof tests for the autouse host-clock guard (tests/infra/clock_guard.py).

These assert the guard actually makes the host clock unreachable from
guarded test code — the property the old `test-clock-hygiene` lint could
only detect after the fact. If this file is itself removed from the guard
(e.g. via a stray `uses_real_clock` marker), these tests fail loudly because
the `pytest.raises` blocks would no longer see a raise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


@pytest.mark.parametrize(
    "expression",
    ("datetime.now()", "time.time()", "time.time_ns()", "time.monotonic()", "time.monotonic_ns()"),
)
def test_module_level_clock_read_fails_during_managed_collection(expression: str) -> None:
    """The guard must be armed before pytest imports ordinary test modules."""
    root = Path(__file__).resolve().parents[3]
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
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "devtools",
                "test",
                "--collect-only",
                "--rootdir",
                str(root),
                str(violating_module),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            check=False,
        )
    finally:
        shutil.rmtree(temporary_root)

    assert result.returncode != 0
    assert "frozen_clock" in result.stdout + result.stderr


def test_module_level_real_clock_marker_exempts_managed_collection() -> None:
    root = Path(__file__).resolve().parents[3]
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
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "devtools",
                "test",
                "--collect-only",
                "--rootdir",
                str(root),
                str(exempt_module),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            check=False,
        )
    finally:
        shutil.rmtree(temporary_root)

    assert result.returncode == 0, result.stdout + result.stderr
