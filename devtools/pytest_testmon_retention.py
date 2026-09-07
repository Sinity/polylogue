"""Free testmon's controller-side retention as each coverage batch is saved.

testmon keeps every phase report of the session in ``TestmonCollect.reports``
and attaches the batch's line-level coverage map to the teardown report that
flushes it. Both are consumed the moment the fingerprints are written, and
neither is released: a corpus run therefore carries every batch's coverage map
and every report of the session to the end, in the one process that also drives
the workers.

This plugin runs after testmon's own report hook and drops what testmon has
just consumed. Only the batch that was saved is dropped: the reports of tests
whose batch has not flushed yet are what the next save reads.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

__all__ = ["BATCH_ATTRIBUTE", "COLLECT_PLUGIN_NAME", "PLUGIN_NAME", "ControllerRetention"]

PLUGIN_NAME: Final = "polylogue_testmon_retention"
#: The name testmon registers its collecting plugin under.
COLLECT_PLUGIN_NAME: Final = "TestmonCollect"
#: The batch coverage map testmon attaches to a flushing teardown report.
BATCH_ATTRIBUTE: Final = "nodes_files_lines"


def pytest_configure(config: Any) -> None:
    # A worker attaches the batch for the controller to consume and serializes
    # the report from this same hook; the controller is the only side that has
    # anything to free.
    if hasattr(config, "workerinput"):
        return
    config.pluginmanager.register(ControllerRetention(config), PLUGIN_NAME)


def pytest_unconfigure(config: Any) -> None:
    plugin = config.pluginmanager.get_plugin(PLUGIN_NAME)
    if plugin is not None:
        config.pluginmanager.unregister(plugin)


class ControllerRetention:
    """Drop each saved coverage batch and the reports it was computed from."""

    def __init__(self, config: Any) -> None:
        self._config = config

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logreport(self, report: Any) -> None:
        batch = report.__dict__.pop(BATCH_ATTRIBUTE, None)
        if not batch:
            return
        collect = self._config.pluginmanager.get_plugin(COLLECT_PLUGIN_NAME)
        reports = getattr(collect, "reports", None)
        if reports is None:
            return
        # The flushing test is itself in the batch: testmon adds every name it
        # started to ``batched_test_names`` and gives each one an entry.
        for nodeid in batch:
            reports.pop(nodeid, None)
