"""Session-end retention probe: name what a pytest worker still holds.

Loaded explicitly with ``-p tests.infra.retention_probe``. It is inert unless
``POLYLOGUE_RETENTION_PROBE`` names a directory, so an ordinary managed run
pays nothing for it.

Two questions are answered separately, because they fail differently:

* *How much* -- an RSS curve sampled per test, so the peak reported here is
  the peak of the RUN, taken before the probe's own heap walk allocates
  anything. The walk is deliberately last.
* *Who* -- a whole-heap walk at session end that aggregates by type, then
  names every container above a size floor with the object that refers to it.
  Module globals are enumerated directly; everything else is attributed
  through one batched ``gc.get_referrers`` call.

The walk treats modules, classes, functions and frames as opaque leaves: a
memo key in this repository routinely holds a function object, and recursing
through its ``__globals__`` reaches every object in the process, which would
attribute the entire heap to whichever root was visited first.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import BuiltinFunctionType, FrameType, FunctionType, MethodType, ModuleType
from typing import Any

import pytest

__all__ = ["PROBE_DIR_ENV", "pytest_configure"]

PROBE_DIR_ENV = "POLYLOGUE_RETENTION_PROBE"
_MIB = 1024.0 * 1024.0
#: Containers at or above this many elements are named individually.
_CONTAINER_LEN_FLOOR = 256
#: How many named containers the report carries.
_CONTAINER_REPORT_COUNT = 60
#: How many of those get a referrer attribution (one batched pass for all).
_OWNER_REPORT_COUNT = 40
#: Sample the RSS curve this often, in tests.
_CURVE_EVERY = 50

_OPAQUE = (ModuleType, type, FunctionType, MethodType, BuiltinFunctionType, FrameType)

#: The nine module-level caches polylogue-0e6m3 is about, in file order.
_WIRE_SUPPORT_CACHES = (
    "_IDENTITY_DIGESTS",
    "_KEYWORDS",
    "_KEYWORD_TUPLES",
    "_GENERATED_WITNESSES",
    "_GENERATED_BATCHES",
    "_CONSTRUCT_COVERAGE",
    "_VALIDATIONS",
    "_PARSED_PAYLOADS",
    "_RECEIPTS",
)

_PAGE_KIB = os.sysconf("SC_PAGE_SIZE") // 1024


def _rss_kib() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * _PAGE_KIB
    except (OSError, IndexError, ValueError):
        return 0


def _sizeof(obj: object) -> int:
    try:
        return sys.getsizeof(obj)
    except (TypeError, ValueError):
        return 0


def _deep_size(roots: list[Any], seen: set[int]) -> tuple[int, int]:
    """Bytes and object count reachable from ``roots``, treating code as opaque."""
    total = 0
    count = 0
    stack = list(roots)
    while stack:
        obj = stack.pop()
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        total += _sizeof(obj)
        count += 1
        if isinstance(obj, _OPAQUE):
            continue
        try:
            stack.extend(gc.get_referents(obj))
        except Exception:  # pragma: no cover - a hostile __getattr__
            continue
    return total, count


def _brief(obj: object) -> str:
    try:
        text = repr(obj)
    except Exception:  # pragma: no cover - a hostile __repr__
        return f"<unrepresentable {type(obj).__name__}>"
    return text[:160]


def _type_name(obj: object) -> str:
    cls = type(obj)
    module = getattr(cls, "__module__", "") or ""
    return f"{module}.{cls.__qualname__}" if module and module != "builtins" else cls.__qualname__


def _module_global_index() -> dict[int, str]:
    """``id(module.__dict__)`` -> module name, for owner attribution."""
    index: dict[int, str] = {}
    for name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict):
            index[id(namespace)] = name
    return index


def _heap_by_type() -> tuple[dict[str, dict[str, float]], int, list[Any]]:
    """Aggregate the whole reachable heap by type; return the tracked objects too."""
    tracked = gc.get_objects()
    seen: set[int] = set()
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    stack: list[Any] = list(tracked)
    walked = 0
    while stack:
        obj = stack.pop()
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        size = _sizeof(obj)
        entry = totals[_type_name(obj)]
        entry[0] += size
        entry[1] += 1
        walked += 1
        if isinstance(obj, _OPAQUE):
            continue
        try:
            stack.extend(gc.get_referents(obj))
        except Exception:  # pragma: no cover
            continue
    ranked = {
        name: {"mib": round(values[0] / _MIB, 2), "count": int(values[1])}
        for name, values in sorted(totals.items(), key=lambda item: item[1][0], reverse=True)[:40]
    }
    return ranked, walked, tracked


def _named_containers(tracked: list[Any], module_index: dict[int, str]) -> list[dict[str, Any]]:
    """Every large container in the heap, deep-sized and ranked.

    Sizing is one shared-``seen`` pass in descending-length order, so the whole
    ranking costs a single heap walk and no byte is counted twice: a row's
    ``marginal_mib`` is what it retains that no longer container already
    claimed. The reported rows are then re-sized standalone (``own_mib``) so a
    container whose contents are pooled with a neighbour stays readable.
    """
    candidates: list[Any] = []
    for obj in tracked:
        if isinstance(obj, (dict, list, set, frozenset, tuple)):
            try:
                if len(obj) >= _CONTAINER_LEN_FLOOR:
                    candidates.append(obj)
            except Exception:  # pragma: no cover
                continue
    candidates.sort(key=len, reverse=True)
    shared: set[int] = set()
    rows: list[dict[str, Any]] = []
    for obj in candidates:
        size, count = _deep_size([obj], shared)
        rows.append(
            {
                "type": _type_name(obj),
                "len": len(obj),
                "marginal_mib": round(size / _MIB, 2),
                "marginal_objects": count,
                "module_globals_of": module_index.get(id(obj)),
                "_obj": obj,
            }
        )
    rows.sort(key=lambda row: row["marginal_mib"], reverse=True)
    rows = rows[:_CONTAINER_REPORT_COUNT]
    for row in rows:
        own_seen: set[int] = set()
        own_bytes, _ = _deep_size([row["_obj"]], own_seen)
        row["own_mib"] = round(own_bytes / _MIB, 2)
    return rows


def _attribute_owners(rows: list[dict[str, Any]], module_index: dict[int, str]) -> None:
    """Name what refers to each reported container, in one referrer pass."""
    targets = [row["_obj"] for row in rows[:_OWNER_REPORT_COUNT]]
    if not targets:
        return
    by_id = {id(obj): rows[index] for index, obj in enumerate(targets)}
    try:
        referrers = gc.get_referrers(*targets)
    except Exception:  # pragma: no cover
        return
    owners: dict[int, list[str]] = defaultdict(list)
    for referrer in referrers:
        if isinstance(referrer, FrameType) or referrer is targets or referrer is rows:
            continue
        if isinstance(referrer, dict):
            module_name = module_index.get(id(referrer))
            for key, value in list(referrer.items()):
                marker = id(value)
                if marker in by_id and len(owners[marker]) < 4:
                    owners[marker].append(f"{module_name}.{key}" if module_name else f"<dict>[{key!r}]")
            continue
        described = f"{_type_name(referrer)} {_brief(referrer)}"
        try:
            referents = gc.get_referents(referrer)
        except Exception:  # pragma: no cover
            continue
        for value in referents:
            marker = id(value)
            if marker in by_id and len(owners[marker]) < 4 and described not in owners[marker]:
                owners[marker].append(described)
    for marker, row in by_id.items():
        row["owners"] = owners.get(marker) or ["<unattributed>"]


def _wire_support_report() -> dict[str, Any] | None:
    module = sys.modules.get("tests.infra.wire_support")
    if module is None:
        return None
    shared: set[int] = set()
    caches: dict[str, Any] = {}
    for name in _WIRE_SUPPORT_CACHES:
        cache = getattr(module, name, None)
        if cache is None:
            caches[name] = {"absent": True}
            continue
        own_seen: set[int] = set()
        own_bytes, own_objects = _deep_size([cache], own_seen)
        marginal_bytes, _ = _deep_size([cache], shared)
        caches[name] = {
            "len": len(cache),
            "own_mib": round(own_bytes / _MIB, 3),
            "own_objects": own_objects,
            "marginal_mib": round(marginal_bytes / _MIB, 3),
        }
    return {"caches": caches, "union_objects": len(shared)}


class _RetentionProbe:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.started = time.monotonic()
        self.tests = 0
        self.peak_rss_kib = _rss_kib()
        self.start_rss_kib = self.peak_rss_kib
        self.curve: list[dict[str, int]] = []

    def _write(self, name: str, payload: dict[str, Any]) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / name
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            sys.stderr.write(f"retention-probe: wrote {path}\n")
        except OSError as exc:  # pragma: no cover - diagnostic only
            sys.stderr.write(f"retention-probe: could not write {name}: {exc}\n")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "teardown":
            return
        self.tests += 1
        rss = _rss_kib()
        if rss > self.peak_rss_kib:
            self.peak_rss_kib = rss
        if self.tests % _CURVE_EVERY == 0:
            sample: dict[str, int] = {"tests": self.tests, "rss_kib": rss}
            module = sys.modules.get("tests.infra.wire_support")
            if module is not None:
                for name in _WIRE_SUPPORT_CACHES:
                    cache = getattr(module, name, None)
                    if cache is not None:
                        sample[f"len:{name}"] = len(cache)
            self.curve.append(sample)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        del session, exitstatus
        pre_walk_rss_kib = _rss_kib()
        wall_s = round(time.monotonic() - self.started, 1)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        payload: dict[str, Any] = {
            "worker": worker,
            "pid": os.getpid(),
            "tests": self.tests,
            "wall_s": wall_s,
            "start_rss_mib": round(self.start_rss_kib / 1024, 1),
            "peak_rss_mib": round(self.peak_rss_kib / 1024, 1),
            "pre_walk_rss_mib": round(pre_walk_rss_kib / 1024, 1),
            "rss_curve": self.curve,
        }
        # A probe is diagnostic: a failure in the walk must not decide the run.
        try:
            gc.collect()
            payload["after_gc_rss_mib"] = round(_rss_kib() / 1024, 1)
            payload["wire_support"] = _wire_support_report()
            module_index = _module_global_index()
            by_type, walked, tracked = _heap_by_type()
            payload["heap_objects_walked"] = walked
            payload["heap_by_type"] = by_type
            rows = _named_containers(tracked, module_index)
            _attribute_owners(rows, module_index)
            for row in rows:
                row.pop("_obj", None)
            payload["large_containers"] = rows
            del tracked
            payload["walk_wall_s"] = round(time.monotonic() - self.started - wall_s, 1)
        except BaseException as exc:  # pragma: no cover - diagnostic only
            payload["walk_error"] = f"{type(exc).__name__}: {exc}"
        self._write(f"retention-{worker}-{os.getpid()}.json", payload)


def pytest_configure(config: pytest.Config) -> None:
    directory = os.environ.get(PROBE_DIR_ENV, "").strip()
    if not directory:
        return
    probe = _RetentionProbe(Path(directory))
    probe._write(
        f"configured-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-{os.getpid()}.json",
        {"loaded": True, "start_rss_mib": round(probe.start_rss_kib / 1024, 1)},
    )
    config.pluginmanager.register(probe, "polylogue-retention-probe")
