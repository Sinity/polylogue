"""Runtime clock guard (#1300 follow-up): make direct host-clock reads
*unreachable* from test code instead of merely detecting them afterward.

Previously a separate lint (``devtools verify test-clock-hygiene``) AST-
scanned every test file and compared the findings against a hand-kept
``docs/plans/test-clock-allowlist.yaml`` registry. That registry is a rung-1
tell: the capability (calling the host clock directly) stayed fully
available, and the only thing standing between a new violation and a clean
verify run was someone remembering the lint existed.

This module removes the capability instead of scanning for its use:

- An autouse fixture (``_guard_host_clock``) patches ``time.time``,
  ``time.monotonic``, ``time.monotonic_ns``, and ``time.time_ns`` for the
  duration of every guarded test with a wrapper that raises immediately when
  called from test-file code (frame-checked, so production code under test
  is untouched — it still reads the real clock exactly as before).
- The same fixture patches the ``datetime`` symbol inside the *test's own
  module* to a subclass whose ``.now()``/``.utcnow()`` raise, mirroring the
  technique ``frozen_clock`` already uses to pin ``datetime.now`` in
  production modules. (CPython's ``datetime.datetime`` is an immutable C
  type — ``datetime.datetime.now = ...`` raises ``TypeError`` — so the class
  itself can never be patched directly; patching the module-level name
  binding that ``from datetime import datetime`` creates is the mechanism
  ``frozen_clock`` already relies on, applied here to test modules instead
  of production ones.)

A test that genuinely needs the real clock (a timing benchmark, a fuzz
harness measuring real latency, a test that waits on real OS thread/process
state) opts out explicitly with a marker colocated in the same file:

    @pytest.mark.uses_real_clock("timing benchmark measures real latency")
    def test_something(): ...

or, for a whole module::

    pytestmark = pytest.mark.uses_real_clock("...")

There is no external registry to keep in sync — the opt-in lives next to the
code it exempts, so a reviewer sees the marker and the call in the same
diff hunk instead of a separate YAML row that can drift unnoticed.

``tests/infra/`` (where this fixture and its siblings live) and
``conftest.py`` files are exempt for the same reason the old lint exempted
them: they are the harness, not the tests.
"""

from __future__ import annotations

import time as _time_module
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from devtools import repo_root as _get_root

ROOT = _get_root()
TESTS_DIR = ROOT / "tests"
INFRA_DIR = TESTS_DIR / "infra"

_GUIDANCE = (
    "{name}() reads the host clock directly from test code ({path}), which "
    "makes the test nondeterministic. Use the `frozen_clock` fixture "
    "(tests/infra/frozen_clock.py) instead. If this test genuinely needs the "
    "real clock (a timing benchmark, a real-thread/process wait, ...), opt "
    'out explicitly: @pytest.mark.uses_real_clock("why").'
)


def _is_guarded_path(path: Path) -> bool:
    """True when a test-file path is subject to the clock guard."""
    try:
        path.relative_to(INFRA_DIR)
    except ValueError:
        pass
    else:
        return False
    if path.name == "conftest.py":
        return False
    try:
        path.relative_to(TESTS_DIR)
    except ValueError:
        return False
    return True


def _caller_is_guarded(frame_depth: int) -> tuple[bool, Path | None]:
    import sys

    frame = sys._getframe(frame_depth + 1)
    caller_path = Path(frame.f_code.co_filename)
    return _is_guarded_path(caller_path), caller_path


def _time_raiser(name: str, real: Any) -> Any:
    def _wrapped(*args: object, **kwargs: object) -> object:
        guarded, caller_path = _caller_is_guarded(1)
        if guarded:
            raise RuntimeError(_GUIDANCE.format(name=name, path=caller_path))
        return real(*args, **kwargs)

    return _wrapped


class _RaisingDateTime(datetime):
    """``datetime`` subclass whose clock reads always raise.

    Installed into a guarded test module's ``datetime`` symbol so
    ``datetime.now()`` / ``datetime.utcnow()`` called from that module's
    top-level test code fail immediately.
    """

    _module_path: str

    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
        raise RuntimeError(_GUIDANCE.format(name="datetime.now", path=cls._module_path))

    @classmethod
    def utcnow(cls) -> datetime:  # type: ignore[override]
        raise RuntimeError(_GUIDANCE.format(name="datetime.utcnow", path=cls._module_path))


def _make_raising_datetime(module_path: str) -> type[datetime]:
    return type("GuardedDateTime", (_RaisingDateTime,), {"_module_path": module_path})


def _opted_out(request: pytest.FixtureRequest) -> bool:
    return bool(list(request.node.iter_markers(name="uses_real_clock")))


@pytest.fixture(autouse=True)
def _guard_host_clock(request: pytest.FixtureRequest) -> Iterator[None]:
    """Autouse guard: block direct host-clock reads from guarded test files.

    No-op for tests/infra, conftest.py, tests that opt out via
    ``@pytest.mark.uses_real_clock(...)``, or tests that request the
    ``frozen_clock`` fixture themselves (which manages the clock directly).
    """
    node_path = Path(str(request.node.fspath))
    if not _is_guarded_path(node_path):
        yield
        return
    if _opted_out(request):
        yield
        return
    if "frozen_clock" in request.fixturenames:
        yield
        return

    module: ModuleType | None = getattr(request, "module", None)
    real_datetime_symbol = getattr(module, "datetime", None) if module is not None else None
    patch_datetime = module is not None and real_datetime_symbol is datetime

    rel_path = str(node_path.relative_to(ROOT)) if node_path.is_absolute() else str(node_path)

    # ``new=`` replaces the attribute with the wrapper function directly.
    # ``side_effect=`` would instead wrap it in a ``MagicMock``, which
    # inserts extra unittest.mock frames between the call site and our
    # wrapper — breaking the caller-frame check the wrapper relies on to
    # tell test code from production code.
    patches = [
        patch("time.time", new=_time_raiser("time.time", _time_module.time)),
        patch("time.monotonic", new=_time_raiser("time.monotonic", _time_module.monotonic)),
        patch("time.monotonic_ns", new=_time_raiser("time.monotonic_ns", _time_module.monotonic_ns)),
        patch("time.time_ns", new=_time_raiser("time.time_ns", _time_module.time_ns)),
    ]
    if patch_datetime and module is not None:
        patches.append(patch.object(module, "datetime", _make_raising_datetime(rel_path)))

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "uses_real_clock(reason): exempt a test from the autouse host-clock "
        "guard (tests/infra/clock_guard.py) because it genuinely needs the "
        "real wall/monotonic clock.",
    )


__all__ = ["pytest_configure"]
