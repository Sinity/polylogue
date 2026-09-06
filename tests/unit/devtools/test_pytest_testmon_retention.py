"""testmon's controller-side retention is freed as each coverage batch is saved.

``TestmonCollect`` keeps every phase report of the session and attaches the
batch's line-level coverage map to the teardown report that flushes it. Both
are consumed the moment the fingerprints are written and neither is released,
so a corpus run carries the whole session in the one process that also drives
the workers.

Anti-vacuity:
- drop the ``trylast`` marker and
  ``test_the_batch_is_freed_only_after_testmon_has_consumed_it`` goes red, which
  is the case that matters: freeing first destroys the fingerprints;
- pop every nodeid instead of the batch's and
  ``test_a_test_outside_the_saved_batch_keeps_its_reports`` goes red -- under
  xdist the other workers' tests are mid-flight and their reports are what the
  next save reads;
- leave ``nodes_files_lines`` on the report and
  ``test_the_saved_coverage_map_is_dropped_from_the_report`` goes red -- the map
  is the larger of the two structures;
- drop the ``workerinput`` guard and
  ``test_a_worker_registers_nothing`` goes red: the worker is the side that
  attaches the batch, not the side that consumes it.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from devtools.pytest_testmon_retention import (
    BATCH_ATTRIBUTE,
    COLLECT_PLUGIN_NAME,
    PLUGIN_NAME,
    ControllerRetention,
)


class _Report:
    """A teardown report carrying a saved batch, as testmon's worker sends it."""

    def __init__(self, nodeid: str, batch: dict[str, Any] | None = None) -> None:
        self.nodeid = nodeid
        self.when = "teardown"
        if batch is not None:
            setattr(self, BATCH_ATTRIBUTE, batch)


class _Collect:
    """A stand-in for ``TestmonCollect``, holding the map the plugin prunes."""

    def __init__(self, nodeids: list[str]) -> None:
        self.reports = {nodeid: {"setup": object(), "call": object()} for nodeid in nodeids}


def _config(collect: _Collect | None) -> Any:
    return SimpleNamespace(
        pluginmanager=SimpleNamespace(get_plugin=lambda name: collect if name == COLLECT_PLUGIN_NAME else None)
    )


def test_the_saved_batch_is_dropped_from_the_retained_reports() -> None:
    collect = _Collect(["t::a", "t::b", "t::c"])
    plugin = ControllerRetention(_config(collect))

    plugin.pytest_runtest_logreport(_Report("t::c", {"t::a": {}, "t::b": {}, "t::c": {}}))

    assert collect.reports == {}


def test_a_test_outside_the_saved_batch_keeps_its_reports() -> None:
    """Under xdist the batch is one worker's; the others are still running."""
    collect = _Collect(["gw0::a", "gw0::b", "gw1::c", "gw1::d"])
    plugin = ControllerRetention(_config(collect))

    plugin.pytest_runtest_logreport(_Report("gw0::b", {"gw0::a": {}, "gw0::b": {}}))

    assert sorted(collect.reports) == ["gw1::c", "gw1::d"]


def test_the_saved_coverage_map_is_dropped_from_the_report() -> None:
    collect = _Collect(["t::a"])
    plugin = ControllerRetention(_config(collect))
    report = _Report("t::a", {"t::a": {"polylogue/x.py": {1, 2, 3}}})

    plugin.pytest_runtest_logreport(report)

    assert not hasattr(report, BATCH_ATTRIBUTE)


def test_a_report_without_a_saved_batch_frees_nothing() -> None:
    """Most teardowns carry an empty map: their tests belong to the next save."""
    collect = _Collect(["t::a", "t::b"])
    plugin = ControllerRetention(_config(collect))

    plugin.pytest_runtest_logreport(_Report("t::a", {}))
    plugin.pytest_runtest_logreport(_Report("t::b"))

    assert sorted(collect.reports) == ["t::a", "t::b"]


def test_a_run_without_testmon_is_left_alone() -> None:
    """``devtools test`` and the rerun both run with ``-p no:testmon``."""
    plugin = ControllerRetention(_config(None))

    plugin.pytest_runtest_logreport(_Report("t::a", {"t::a": {}}))


def test_a_worker_registers_nothing() -> None:
    from devtools import pytest_testmon_retention

    registered: list[str] = []
    manager = SimpleNamespace(register=lambda _plugin, name: registered.append(name))

    pytest_testmon_retention.pytest_configure(SimpleNamespace(pluginmanager=manager, workerinput={"workerid": "gw0"}))
    assert registered == []

    pytest_testmon_retention.pytest_configure(SimpleNamespace(pluginmanager=manager))
    assert registered == [PLUGIN_NAME]


def test_the_batch_is_freed_only_after_testmon_has_consumed_it() -> None:
    """The ordering against the installed testmon, dispatched by pluggy itself.

    testmon's own ``pytest_runtest_logreport`` is an unmarked hookimpl, so
    ``trylast`` places this plugin after it. Both facts are checked: that the
    real hook carries no ordering marker, and the order pluggy calls them in
    when the retention plugin is registered last -- the position from which an
    unmarked hook would run first.
    """
    testmon_hook = pytest.importorskip("testmon.pytest_testmon").TestmonCollect.pytest_runtest_logreport
    opts = getattr(testmon_hook, "pytest_impl", {})
    assert not (opts.get("tryfirst") or opts.get("trylast") or opts.get("hookwrapper") or opts.get("wrapper")), (
        "testmon marked its report hook; the retention plugin's ordering must be rechecked"
    )

    seen: list[str] = []

    class _Testmon:
        """Consumes the batch where testmon does; red if it was freed first."""

        def pytest_runtest_logreport(self, report: Any) -> None:
            seen.append("testmon")
            assert getattr(report, BATCH_ATTRIBUTE, None), "the batch was freed before testmon read it"

    collect = _Collect(["t::a"])
    manager = pytest.PytestPluginManager()
    manager.register(_Testmon(), COLLECT_PLUGIN_NAME + "-stand-in")
    manager.register(ControllerRetention(_config(collect)), PLUGIN_NAME)

    manager.hook.pytest_runtest_logreport(report=_Report("t::a", {"t::a": {}}))

    assert seen == ["testmon"]
    assert collect.reports == {}


def test_the_plugin_declares_the_name_testmon_registers() -> None:
    """The lookup is by registered name; a rename would silently free nothing."""
    source = inspect.getsource(pytest.importorskip("testmon.pytest_testmon"))

    assert f'"{COLLECT_PLUGIN_NAME}"' in source or f"'{COLLECT_PLUGIN_NAME}'" in source
