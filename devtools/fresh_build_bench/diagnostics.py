"""Private built-in diagnostics around an owner's existing production call.

The owner supplies input/recipe/candidate authority and retains all product
resources. This module records no arguments, return values, locals, or SQL.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from devtools.fresh_build_bench.corpus import refuse_inside_checkout

_CAPTURE_LOCK = threading.Lock()
_REQUIRED_BINDINGS = frozenset(
    {
        "candidate",
        "product_tree",
        "runtime",
        "input_sha256",
        "recipe_sha256",
        "schema_inventory_digest",
        "component",
        "runner_sha256",
    }
)


class DiagnosticRefusalError(ValueError):
    """The capture cannot certify its requested diagnostic scope."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class BoundModule:
    """A loaded module's name and exact source hash, supplied by its owner."""

    name: str
    sha256: str


class _ProfileStats(Protocol):
    # pstats publishes these fields; typeshed's Stats omits their annotations.
    stats: Mapping[tuple[str, int, str], tuple[int, int, float, float, object]]
    total_calls: int
    prim_calls: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime() -> dict[str, Any]:
    path = Path(sys.executable).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "version": sys.version,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
    }


def _boundary() -> dict[str, Any]:
    # RSS is a point observation, not the peak within the captured interval.
    rss = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {
        "wall_s": time.perf_counter(),
        "thread_cpu_s": time.thread_time(),
        "process_cpu_s": time.process_time(),
        "rss_bytes": rss,
        "native_tid": threading.get_native_id(),
    }


def _write_json(path: Path, data: object) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def _reserve(path: Path) -> None:
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))


class DiagnosticCapture:
    """Capture state; immutable output references become available after exit."""

    def __init__(
        self,
        mode: str,
        output_directory: Path,
        bindings: Mapping[str, Any],
        module_authority: Mapping[str, BoundModule],
        summary_rows: int,
        traceback_frames: int,
    ) -> None:
        self.mode = mode
        self.output_directory = output_directory
        self.bindings = json.loads(json.dumps(dict(bindings), allow_nan=False))
        self.authority = {str(Path(path).resolve()): module for path, module in module_authority.items()}
        self.summary_rows = summary_rows
        self.traceback_frames = traceback_frames
        self.manifest_path: Path | None = None
        self.manifest_sha256: str | None = None
        self._profiler: Any = None
        self._tracemalloc: Any = None
        self._before_snapshot: Any = None
        self._observations: dict[str, Any] = {}
        self._tool_files: dict[str, Any] = {}
        self._artifacts: list[dict[str, Any]] = []
        self._started: dict[str, Any] = {}

    def _observe_runtime(self, phase: str) -> None:
        observed = _runtime()
        self._observations[phase] = observed
        if observed != self.bindings["runtime"]:
            raise DiagnosticRefusalError("runtime_binding_changed")

    def _verify_modules(self) -> None:
        for path, module in self.authority.items():
            try:
                actual = _sha256(Path(path))
            except OSError as exc:
                raise DiagnosticRefusalError("module_binding_unavailable") from exc
            if actual != module.sha256:
                raise DiagnosticRefusalError("module_binding_changed")

    def _tool(self, module: Any) -> None:
        path = getattr(module, "__file__", None)
        self._tool_files[module.__name__] = (
            {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))} if path else {"builtin": True}
        )

    def _start(self) -> None:
        self._observe_runtime("before_tool_imports")
        self._verify_modules()
        if self.mode == "cprofile":
            import _lsprof
            import cProfile
            import pstats

            self._tool(cProfile)
            self._tool(_lsprof)
            self._tool(pstats)
            if sys.getprofile() is not None:
                raise DiagnosticRefusalError("existing_thread_profiler")
            self._profiler = cProfile.Profile()
        else:
            import _tracemalloc
            import tracemalloc

            self._tool(tracemalloc)
            self._tool(_tracemalloc)
            if tracemalloc.is_tracing():
                raise DiagnosticRefusalError("existing_allocation_tracer")
            self._tracemalloc = tracemalloc
        self._observe_runtime("after_tool_imports")
        if self._profiler is not None:
            self._profiler.enable()
        else:
            self._tracemalloc.start(self.traceback_frames)
            self._before_snapshot = self._tracemalloc.take_snapshot()
            self._tracemalloc.reset_peak()
        self._observe_runtime("after_start")
        self._started = _boundary()

    def _stop(self) -> None:
        if self._profiler is not None:
            self._profiler.disable()
        if self._tracemalloc is not None and self._tracemalloc.is_tracing():
            self._tracemalloc.stop()

    def _location(self, filename: str, line: int, function: str | None = None) -> dict[str, Any]:
        module = self.authority.get(str(Path(filename).resolve()))
        if module is None:
            # Unknown filenames/functions may be constructed from private input.
            return {"unbound_key_sha256": hashlib.sha256(json.dumps([filename, line, function]).encode()).hexdigest()}
        result: dict[str, Any] = {"module": module.name, "source_sha256": module.sha256, "line": line}
        if function is not None:
            result["function"] = function
        return result

    def _artifact(self, filename: str) -> Path:
        path = self.output_directory / filename
        _reserve(path)
        return path

    def _reference(self, path: Path) -> dict[str, Any]:
        return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}

    def _profile_summary(self) -> dict[str, Any]:
        import pstats

        raw = self._artifact("calls.pstats")
        self._profiler.dump_stats(str(raw))
        self._artifacts.append(self._reference(raw))
        stats = cast(_ProfileStats, pstats.Stats(self._profiler))
        rows: list[dict[str, Any]] = []
        module_self: dict[str, float] = {}
        for (filename, line, function), (primitive, calls, self_s, inclusive_s, _callers) in stats.stats.items():
            location = self._location(filename, line, function)
            rows.append(
                {
                    **location,
                    "primitive_calls": primitive,
                    "calls": calls,
                    "self_elapsed_s": self_s,
                    "inclusive_elapsed_s": inclusive_s,
                }
            )
            module = location.get("module", "unbound")
            module_self[module] = module_self.get(module, 0.0) + self_s
        return {
            "coverage": "runtime_cprofile_merged_python_and_exposed_C_calls",
            "timer": "instrumented_reported_call_elapsed",
            "limits": [
                "no_thread_attribution",
                "concurrent_nesting_and_elapsed_unqualified",
                "native_call_internals_unresolved",
                "elapsed_includes_wait",
                "inclusive_rows_overlap",
                "reported_self_totals_not_exclusive_under_concurrency",
                "profiler_overhead_present",
            ],
            "total_calls": stats.total_calls,
            "primitive_calls": stats.prim_calls,
            "function_rows": len(rows),
            "summary_rows_per_ranking": self.summary_rows,
            "omitted_rows_per_ranking": max(0, len(rows) - self.summary_rows),
            "top_self": sorted(rows, key=lambda row: row["self_elapsed_s"], reverse=True)[: self.summary_rows],
            "top_inclusive": sorted(rows, key=lambda row: row["inclusive_elapsed_s"], reverse=True)[
                : self.summary_rows
            ],
            "top_calls": sorted(rows, key=lambda row: row["calls"], reverse=True)[: self.summary_rows],
            "reported_module_self_elapsed_s": module_self,
        }

    def _allocation_summary(self, after: Any, memory: tuple[int, int], tracer_bytes: int) -> dict[str, Any]:
        for filename, snapshot in (
            ("allocations-before.snapshot", self._before_snapshot),
            ("allocations-after.snapshot", after),
        ):
            raw = self._artifact(filename)
            snapshot.dump(str(raw))
            self._artifacts.append(self._reference(raw))
        changes = after.compare_to(self._before_snapshot, "traceback")
        rows = [
            {
                "frames": [self._location(frame.filename, frame.lineno) for frame in change.traceback],
                "net_bytes": change.size_diff,
                "net_blocks": change.count_diff,
            }
            for change in changes
        ]
        return {
            "coverage": "process_wide_traced_Python_allocations",
            "limits": [
                "not_native_memory_or_RSS",
                "prestart_allocations_untraced",
                "net_live_allocation_change_not_total_churn",
                "no_thread_attribution",
                "snapshot_and_helper_overhead_present",
            ],
            "traceback_frames": self.traceback_frames,
            "current_traced_bytes": memory[0],
            "peak_traced_bytes_since_reset": memory[1],
            "tracer_metadata_bytes": tracer_bytes,
            "net_bytes": sum(change.size_diff for change in changes),
            "net_blocks": sum(change.count_diff for change in changes),
            "traceback_rows": len(rows),
            "summary_rows": self.summary_rows,
            "omitted_rows": max(0, len(rows) - self.summary_rows),
            "largest_absolute_net_changes": sorted(rows, key=lambda row: abs(row["net_bytes"]), reverse=True)[
                : self.summary_rows
            ],
        }

    def _finish(self, failure: BaseException | None) -> None:
        ended = _boundary()
        after = memory = tracer_bytes = None
        try:
            if self._tracemalloc is not None:
                memory = self._tracemalloc.get_traced_memory()
                tracer_bytes = self._tracemalloc.get_tracemalloc_memory()
                after = self._tracemalloc.take_snapshot()
        finally:
            self._stop()
        refusal = None
        try:
            self._observe_runtime("after_stop")
            self._verify_modules()
            if ended["native_tid"] != self._started["native_tid"]:
                raise DiagnosticRefusalError("capture_thread_changed")
        except DiagnosticRefusalError as exc:
            refusal = exc
        if self.mode == "cprofile":
            summary = self._profile_summary()
        else:
            assert memory is not None and tracer_bytes is not None
            summary = self._allocation_summary(after, memory, tracer_bytes)
        summary_path = self.output_directory / "summary.json"
        _write_json(summary_path, summary)
        manifest = {
            "format": "polylogue-component-diagnostic-v1",
            "mode": self.mode,
            "diagnostic_outcome": "refused" if refusal else "captured",
            "refusal_reason": refusal.reason if refusal else None,
            "call_outcome": "failed" if failure else "completed",
            "exception_type": f"{type(failure).__module__}.{type(failure).__qualname__}" if failure else None,
            "timing_role": "diagnostic_only_excluded_from_unprofiled_comparison",
            "bindings": self.bindings,
            "module_authority": {
                path: {"name": module.name, "sha256": module.sha256} for path, module in self.authority.items()
            },
            "runtime_observations": self._observations,
            "tool_files": self._tool_files,
            "boundaries": {"start": self._started, "end": ended},
            "boundary_deltas": {
                key: ended[key] - self._started[key] for key in ("wall_s", "thread_cpu_s", "process_cpu_s")
            },
            "boundary_scope": "context_body_with_entry_exit_bookkeeping; serialization_excluded",
            "thread_CPU_scope": "caller_thread_boundary_counter_separate_from_profile_call_statistics",
            "RSS_scope": "two_process_points_not_interval_peak",
            "raw_artifacts": self._artifacts,
            "summary": self._reference(summary_path),
        }
        self.manifest_path = self.output_directory / "manifest.json"
        _write_json(self.manifest_path, manifest)
        self.manifest_sha256 = _sha256(self.manifest_path)
        if refusal:
            raise refusal


@contextmanager
def capture_diagnostic(
    mode: str,
    *,
    output_directory: Path,
    bindings: Mapping[str, Any],
    module_authority: Mapping[str, BoundModule],
    summary_rows: int = 20,
    traceback_frames: int = 1,
) -> Iterator[DiagnosticCapture]:
    """Wrap exactly one owner's callable; preserve its return and exception.

    The new external directory is owner-only. Row limits affect summaries;
    traceback_frames limits raw allocation ancestry. Originals retain hashes.
    """
    if mode not in {"cprofile", "tracemalloc"}:
        raise DiagnosticRefusalError("unsupported_stack_sampling" if mode == "stack" else "unsupported_mode")
    if not bindings.keys() >= _REQUIRED_BINDINGS:
        raise DiagnosticRefusalError("missing_owner_bindings")
    if summary_rows < 1 or traceback_frames < 1:
        raise DiagnosticRefusalError("invalid_diagnostic_configuration")
    output_directory = output_directory.resolve()
    refuse_inside_checkout(output_directory, "diagnostic output")
    if any(
        (parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
        for parent in (output_directory, *output_directory.parents)
    ):
        raise DiagnosticRefusalError("output_inside_checkout")
    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise DiagnosticRefusalError("overlapping_diagnostic_capture")
    capture = None
    failure = None
    try:
        capture = DiagnosticCapture(mode, output_directory, bindings, module_authority, summary_rows, traceback_frames)
        output_directory.mkdir(mode=0o700)
        capture._start()
        try:
            yield capture
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                capture._finish(failure)
            except BaseException as diagnostic_error:
                if failure is None:
                    raise
                failure.add_note(f"diagnostic capture failed: {type(diagnostic_error).__name__}")
    finally:
        if capture is not None:
            capture._stop()
        _CAPTURE_LOCK.release()
