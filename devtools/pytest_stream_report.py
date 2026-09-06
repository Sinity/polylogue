"""The managed run's JSON test report, written without holding it in memory.

The controller receives one report per phase per test. Accumulating them until
the session ends costs the corpus run 124 MiB in the one process that must also
drive eight workers, and the payload is not needed until the run is over. Each
test is spooled to disk as its teardown lands and the report is assembled from
the spool at session end, so the controller holds only the tests in flight.

The file keeps the shape its consumers read -- ``tests`` with per-phase
durations, outcomes and failure detail, plus ``summary``, ``exitcode``,
``duration``, ``created`` and ``root``. Collection nodes, captured streams,
captured logs, keywords and warnings are not written: nothing reads them, and
on the corpus they are most of the payload.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Final

__all__ = ["PLUGIN_NAME", "REPORT_FILE_OPTION", "StreamingReport", "report_file_argument", "spool_paths"]

PLUGIN_NAME: Final = "polylogue_stream_report"
REPORT_FILE_OPTION: Final = "--polylogue-report-file"

#: Phase outcomes whose ``longrepr`` is the xdist worker banner rather than a
#: diagnosis. Writing it for every passed phase triples the report's size.
_QUIET_OUTCOMES: Final = frozenset({"passed"})


def report_file_argument(path: object) -> str:
    """The command-line argument that directs a managed run's report to ``path``."""
    return f"{REPORT_FILE_OPTION}={path}"


def spool_paths(path: Path | str) -> tuple[Path, ...]:
    """The intermediate files a report at ``path`` may have left behind.

    A run killed between its first test and its assembly leaves its spool; the
    next run of the same report clears it with the report itself.
    """
    report = Path(path)
    parent = report.parent if str(report.parent) else Path(".")
    return tuple(sorted({*parent.glob(f"{report.name}.*.parts"), *parent.glob(f"{report.name}.*.tmp")}))


def pytest_addoption(parser: Any) -> None:
    parser.getgroup("polylogue").addoption(
        REPORT_FILE_OPTION,
        dest="polylogue_report_file",
        default=None,
        help="write the streamed JSON test report to this path",
    )


def pytest_configure(config: Any) -> None:
    path = config.getoption("polylogue_report_file", None)
    # An xdist worker inherits the whole command line, and the report is the
    # controller's: a worker writing it would truncate the run to its own share.
    if not path or hasattr(config, "workerinput"):
        return
    config.pluginmanager.register(StreamingReport(config, Path(str(path))), PLUGIN_NAME)


def pytest_unconfigure(config: Any) -> None:
    plugin = config.pluginmanager.get_plugin(PLUGIN_NAME)
    if plugin is not None:
        config.pluginmanager.unregister(plugin)


def _file_location(location: Any) -> dict[str, Any]:
    return {"path": location.path, "lineno": location.lineno, "message": location.message}


def _stage(report: Any) -> dict[str, Any]:
    """One setup/call/teardown entry, carrying failure detail only when it exists."""
    outcome = str(getattr(report, "outcome", ""))
    stage: dict[str, Any] = {"duration": float(getattr(report, "duration", 0.0) or 0.0), "outcome": outcome}
    longrepr: Any = getattr(report, "longrepr", None)
    crash = getattr(longrepr, "reprcrash", None)
    if crash is not None:
        stage["crash"] = _file_location(crash)
        with contextlib.suppress(AttributeError):
            stage["traceback"] = [_file_location(entry.reprfileloc) for entry in longrepr.reprtraceback.reprentries]
    if outcome not in _QUIET_OUTCOMES:
        text = str(getattr(report, "longreprtext", "") or "")
        if text:
            stage["longrepr"] = text
    return stage


class StreamingReport:
    """Spool each finished test, then assemble the report from the spool."""

    def __init__(self, config: Any, path: Path) -> None:
        self._config = config
        self._path = path
        self._spool_path = path.with_name(f"{path.name}.{os.getpid()}.parts")
        self._spool: Any = None
        self._open: dict[str, dict[str, Any]] = {}
        self._outcomes: Counter[str] = Counter()
        self._deselected = 0
        self._started_at = time.time()
        self._failure: str | None = None

    def pytest_sessionstart(self, session: Any) -> None:
        del session
        self._started_at = time.time()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._spool = self._spool_path.open("w", encoding="utf-8")
        except OSError as error:  # pragma: no cover - unwritable artifact directory
            self._failure = str(error)

    def pytest_deselected(self, items: list[Any]) -> None:
        self._deselected += len(items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = str(getattr(report, "nodeid", ""))
        entry = self._open.get(nodeid)
        if entry is None:
            location = getattr(report, "location", ("", None, ""))
            entry = {"nodeid": nodeid, "lineno": location[1], "outcome": "passed"}
            self._open[nodeid] = entry
        # The test's outcome is not any one phase's: an erroring fixture, an
        # xfail and a passing call all report ``passed`` somewhere.
        outcome = self._config.hook.pytest_report_teststatus(report=report, config=self._config)[0]
        if outcome not in ("passed", ""):
            entry["outcome"] = outcome
        entry[str(getattr(report, "when", ""))] = _stage(report)
        if getattr(report, "when", "") == "teardown":
            self._flush(nodeid)

    def _flush(self, nodeid: str) -> None:
        entry = self._open.pop(nodeid, None)
        if entry is None:
            return
        self._outcomes[str(entry["outcome"])] += 1
        if self._spool is None:
            return
        try:
            self._spool.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as error:  # pragma: no cover - full or unwritable spool
            self._failure = str(error)
            self._spool = None

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        # A worker lost mid-test leaves a started test without a teardown; it is
        # still a test the run executed, so it reaches the report.
        for nodeid in list(self._open):
            self._flush(nodeid)
        if self._spool is not None:
            self._spool.close()
            self._spool = None
        if self._failure is not None:
            return
        summary: dict[str, Any] = dict(self._outcomes)
        summary["total"] = sum(self._outcomes.values())
        summary["collected"] = int(getattr(session, "testscollected", 0)) + self._deselected
        if self._deselected:
            summary["deselected"] = self._deselected
        header = {
            "created": time.time(),
            "duration": time.time() - self._started_at,
            "exitcode": int(exitstatus),
            "root": str(self._config.rootpath),
            "summary": summary,
        }
        with contextlib.suppress(OSError):
            self._write(header)
        with contextlib.suppress(OSError):
            self._spool_path.unlink()

    def _write(self, header: dict[str, Any]) -> None:
        """Assemble the report, holding one test at a time."""
        destination = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
        with destination.open("w", encoding="utf-8") as handle:
            handle.write("{")
            for key, value in header.items():
                handle.write(f"{json.dumps(key)}: {json.dumps(value)}, ")
            handle.write('"tests": [')
            separator = ""
            if self._spool_path.exists():
                with self._spool_path.open("r", encoding="utf-8") as spool:
                    for line in spool:
                        line = line.strip()
                        if not line:
                            continue
                        handle.write(separator + line)
                        separator = ", "
            handle.write("]}\n")
        destination.replace(self._path)
