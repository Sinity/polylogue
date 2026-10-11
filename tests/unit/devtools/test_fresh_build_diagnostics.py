"""Actual built-in captures, private artifacts, and owner binding refusals."""

from __future__ import annotations

import hashlib
import json
import pstats
import sys
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from devtools.fresh_build_bench import diagnostics
from devtools.fresh_build_bench.diagnostics import BoundModule, DiagnosticRefusalError, capture_diagnostic


@pytest.fixture
def external_root() -> Iterator[Path]:
    # Managed pytest's tmp_path is inside the checkout; raw captures must stay
    # outside it. This synthetic tree is small and removed on every outcome.
    managed = Path("/realm/tmp/work")
    with tempfile.TemporaryDirectory(
        prefix="polylogue-diagnostics-", dir=managed if managed.is_dir() else None
    ) as directory:
        yield Path(directory)


def _bindings() -> dict[str, Any]:
    return {
        "candidate": {"git_sha": "a" * 40, "dirty": False, "tracked_diff_sha256": "b" * 64},
        "product_tree": "c" * 40,
        "runtime": diagnostics._runtime(),
        "input_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "schema_inventory_digest": "f" * 64,
        "component": "synthetic_exact_callable",
        "runner_sha256": "0" * 64,
    }


def _authority() -> dict[str, BoundModule]:
    path = Path(__file__).resolve()
    return {str(path): BoundModule(__name__, hashlib.sha256(path.read_bytes()).hexdigest())}


def _leaf(value: int) -> int:
    return value + 1


def _worker_only() -> int:
    return sum(range(10))


def _exact_callable() -> object:
    for value in range(9):
        _leaf(value)
    worker = threading.Thread(target=_worker_only)
    worker.start()
    worker.join()
    time.sleep(0.003)
    return _exact_callable


def _read(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return json.loads((directory / "manifest.json").read_text()), json.loads((directory / "summary.json").read_text())


def test_cprofile_wraps_exact_callable_with_private_original_and_merged_scope(external_root: Path) -> None:
    bindings = _bindings()
    output = external_root / "calls"
    with capture_diagnostic(
        "cprofile", output_directory=output, bindings=bindings, module_authority=_authority(), summary_rows=2
    ) as capture:
        result = _exact_callable()
    assert result is _exact_callable
    manifest, summary = _read(output)
    original = cast(diagnostics._ProfileStats, pstats.Stats(str(output / "calls.pstats")))
    leaf = next(row for key, row in original.stats.items() if key[2] == "_leaf")
    assert leaf[:2] == (9, 9)
    assert any(key[2] == "_exact_callable" for key in original.stats)
    assert any(key[2] == "_worker_only" for key in original.stats)
    assert summary["timer"] == "instrumented_reported_call_elapsed"
    assert "inclusive_rows_overlap" in summary["limits"]
    assert "no_thread_attribution" in summary["limits"]
    assert "concurrent_nesting_and_elapsed_unqualified" in summary["limits"]
    assert "reported_self_totals_not_exclusive_under_concurrency" in summary["limits"]
    assert "reported_module_self_elapsed_s" in summary
    assert "disjoint_module_self_elapsed_s" not in summary
    assert summary["omitted_rows_per_ranking"] > 0
    assert len(summary["top_calls"]) == 2
    assert str(Path(__file__).parent) not in (output / "summary.json").read_text()
    assert capture.manifest_path == output / "manifest.json"
    assert capture.manifest_sha256 == hashlib.sha256(capture.manifest_path.read_bytes()).hexdigest()
    assert manifest["bindings"] == bindings
    assert manifest["call_outcome"] == "completed"
    assert manifest["timing_role"] == "diagnostic_only_excluded_from_unprofiled_comparison"
    assert all(observed == bindings["runtime"] for observed in manifest["runtime_observations"].values())
    assert manifest["boundaries"]["start"]["native_tid"] == threading.get_native_id()
    assert manifest["boundary_deltas"]["wall_s"] > 0
    assert output.stat().st_mode & 0o777 == 0o700
    for path in output.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    for reference in [*manifest["raw_artifacts"], manifest["summary"]]:
        path = Path(reference["path"])
        assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert reference["bytes"] == path.stat().st_size


def test_tracemalloc_retains_loadable_snapshots_and_all_thread_allocation_scope(external_root: Path) -> None:
    output = external_root / "allocations"
    held: list[bytearray] = []
    with capture_diagnostic(
        "tracemalloc", output_directory=output, bindings=_bindings(), module_authority=_authority(), traceback_frames=2
    ):
        worker = threading.Thread(target=lambda: held.extend(bytearray(8192) for _ in range(8)))
        worker.start()
        worker.join()
    manifest, summary = _read(output)
    before = tracemalloc.Snapshot.load(str(output / "allocations-before.snapshot"))
    after = tracemalloc.Snapshot.load(str(output / "allocations-after.snapshot"))
    assert sum(change.size_diff for change in after.compare_to(before, "traceback")) == summary["net_bytes"]
    assert summary["net_bytes"] >= 8 * 8192
    assert summary["peak_traced_bytes_since_reset"] >= 8 * 8192
    assert summary["traceback_frames"] == 2
    assert summary["coverage"] == "process_wide_traced_Python_allocations"
    assert "no_thread_attribution" in summary["limits"]
    assert "not_native_memory_or_RSS" in summary["limits"]
    assert manifest["RSS_scope"] == "two_process_points_not_interval_peak"
    assert not tracemalloc.is_tracing()
    assert str(Path(__file__).parent) not in (output / "summary.json").read_text()


@pytest.mark.parametrize("mode", ["cprofile", "tracemalloc"])
def test_failure_preserves_original_exception_without_recording_payload(mode: str, external_root: Path) -> None:
    output = external_root / mode
    original = RuntimeError("private-payload-do-not-export")
    with pytest.raises(RuntimeError) as raised:
        with capture_diagnostic(mode, output_directory=output, bindings=_bindings(), module_authority=_authority()):
            raise original
    assert raised.value is original
    manifest, _ = _read(output)
    assert manifest["call_outcome"] == "failed"
    assert manifest["exception_type"] == "builtins.RuntimeError"
    assert original.args[0] not in (output / "manifest.json").read_text()
    assert sys.getprofile() is None
    assert not tracemalloc.is_tracing()


def test_save_failure_does_not_mask_callable_exception_or_leave_profiler(
    external_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = LookupError("product failure")

    def cannot_write(*_args: object) -> None:
        raise OSError("private disk message")

    monkeypatch.setattr(diagnostics, "_write_json", cannot_write)
    with pytest.raises(LookupError) as raised:
        with capture_diagnostic(
            "cprofile",
            output_directory=external_root / "failed-save",
            bindings=_bindings(),
            module_authority=_authority(),
        ):
            raise original
    assert raised.value is original
    assert original.__notes__ == ["diagnostic capture failed: OSError"]
    assert sys.getprofile() is None


@pytest.mark.parametrize(
    "field,value", [("gil_enabled", "wrong"), ("sha256", "wrong"), ("path", "/unbound"), ("version", "unbound")]
)
def test_runtime_binding_refusal_prevents_callable(field: str, value: str, external_root: Path) -> None:
    bindings = _bindings()
    bindings["runtime"][field] = value
    with pytest.raises(DiagnosticRefusalError, match="runtime_binding_changed"):
        with capture_diagnostic(
            "cprofile", output_directory=external_root / "refused", bindings=bindings, module_authority=_authority()
        ):
            pytest.fail("unbound callable ran")


def test_changed_runtime_at_exit_retains_capture_but_refuses_qualification(
    external_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings()
    changed = {**bindings["runtime"], "gil_enabled": not bindings["runtime"]["gil_enabled"]}
    output = external_root / "changed-gil"
    with pytest.raises(DiagnosticRefusalError, match="runtime_binding_changed"):
        with capture_diagnostic("cprofile", output_directory=output, bindings=bindings, module_authority=_authority()):
            monkeypatch.setattr(diagnostics, "_runtime", lambda: changed)
    manifest, _ = _read(output)
    assert manifest["diagnostic_outcome"] == "refused"
    assert manifest["call_outcome"] == "completed"


def test_module_hash_mismatch_refuses_before_body(external_root: Path) -> None:
    with pytest.raises(DiagnosticRefusalError, match="module_binding_changed"):
        with capture_diagnostic(
            "cprofile",
            output_directory=external_root / "unbound-module",
            bindings=_bindings(),
            module_authority={__file__: BoundModule(__name__, "0" * 64)},
        ):
            pytest.fail("unbound module ran")


def test_existing_tracer_is_not_replaced_or_stopped(external_root: Path) -> None:
    tracemalloc.start()
    try:
        with pytest.raises(DiagnosticRefusalError, match="existing_allocation_tracer"):
            with capture_diagnostic(
                "tracemalloc",
                output_directory=external_root / "existing",
                bindings=_bindings(),
                module_authority=_authority(),
            ):
                pytest.fail("existing tracer was replaced")
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_nested_capture_and_immutable_output_refuse(external_root: Path) -> None:
    output = external_root / "outer"
    with capture_diagnostic("cprofile", output_directory=output, bindings=_bindings(), module_authority=_authority()):
        with pytest.raises(DiagnosticRefusalError, match="overlapping_diagnostic_capture"):
            with capture_diagnostic(
                "cprofile",
                output_directory=external_root / "inner",
                bindings=_bindings(),
                module_authority=_authority(),
            ):
                pytest.fail("nested capture ran")
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        with capture_diagnostic(
            "cprofile", output_directory=output, bindings=_bindings(), module_authority=_authority()
        ):
            pytest.fail("immutable output was replaced")
    assert (output / "manifest.json").read_bytes() == before


def test_unknown_dynamic_filename_and_function_are_opaque(external_root: Path) -> None:
    output = external_root / "opaque"
    namespace: dict[str, Any] = {}
    exec(compile("def private_function_name():\n return 3\n", "/private/transcript/path", "exec"), namespace)
    with capture_diagnostic(
        "cprofile", output_directory=output, bindings=_bindings(), module_authority=_authority(), summary_rows=1000
    ):
        assert namespace["private_function_name"]() == 3
    summary = (output / "summary.json").read_text()
    assert "private_function_name" not in summary
    assert "/private/transcript/path" not in summary
    assert "unbound_key_sha256" in summary


def test_stack_mode_and_checkout_output_refuse(external_root: Path) -> None:
    with pytest.raises(DiagnosticRefusalError, match="unsupported_stack_sampling"):
        with capture_diagnostic(
            "stack", output_directory=external_root / "stack", bindings=_bindings(), module_authority=_authority()
        ):
            pytest.fail("unsupported frame inspection ran")
    with pytest.raises(ValueError, match="outside the checkout"):
        with capture_diagnostic(
            "cprofile",
            output_directory=Path(__file__).parent / "private",
            bindings=_bindings(),
            module_authority=_authority(),
        ):
            pytest.fail("capture was written into public checkout")


def test_empty_git_ledger_directory_is_not_a_checkout(external_root: Path) -> None:
    (external_root / ".git").mkdir()
    with capture_diagnostic(
        "cprofile", output_directory=external_root / "private", bindings=_bindings(), module_authority=_authority()
    ) as capture:
        result = _leaf(1)
    assert result == 2
    assert capture.manifest_path is not None and capture.manifest_path.is_file()


@pytest.mark.parametrize("marker", ["file", "head"])
def test_actual_git_checkout_markers_refuse_output(marker: str, external_root: Path) -> None:
    if marker == "file":
        (external_root / ".git").write_text("gitdir: /synthetic/metadata\n")
    else:
        (external_root / ".git").mkdir()
        (external_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(DiagnosticRefusalError, match="output_inside_checkout"):
        with capture_diagnostic(
            "cprofile", output_directory=external_root / "private", bindings=_bindings(), module_authority=_authority()
        ):
            pytest.fail("actual checkout accepted private capture")
