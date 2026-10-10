"""Native CPU evidence stays distinct from missing Python and off-CPU stacks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from devtools.fresh_build_bench import profile_report, sampler
from devtools.fresh_build_bench.profiler_provider import (
    PerfCpuProfiler,
    ProfilerUnavailableError,
    main,
    parse_perf_report,
    parse_perf_stacks,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cpu_profile"
_REPORT = (_FIXTURES / "native-leaves.txt").read_text()
_SCRIPT = (_FIXTURES / "partial-stacks.txt").read_text()


def _tool(tmp_path: Path, *, report: str = _REPORT, script: str = _SCRIPT) -> str:
    path = tmp_path / "perf"
    path.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, subprocess, sys\n"
        "args=sys.argv[1:]\n"
        "if args[0]=='version': print('perf version synthetic\\ndwarf-unwind: [ on ]')\n"
        "elif args[0]=='record':\n"
        "    pathlib.Path(args[args.index('-o')+1]).write_bytes(b'neutral capture')\n"
        "    raise SystemExit(subprocess.call(args[args.index('--')+1:]))\n"
        "elif args[0]=='inject': pathlib.Path(args[args.index('-o')+1]).write_bytes(b'neutral injected')\n"
        f"elif args[0]=='report': print({report!r})\n"
        f"elif args[0]=='script': print({script!r})\n"
    )
    path.chmod(0o700)
    return str(path)


def _recorded(tmp_path: Path, *, report: str = _REPORT, script: str = _SCRIPT) -> PerfCpuProfiler:
    profiler = PerfCpuProfiler.prepare(
        [sys.executable, "-c", "pass"], tmp_path / "capture", perf=_tool(tmp_path, report=report, script=script)
    )
    (profiler.output_dir / "perf.data").write_bytes(b"neutral capture")
    return profiler


def test_owned_command_plan_has_no_attach_and_never_overwrites_evidence(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "neutral_child", "--option", "value"]
    profiler = PerfCpuProfiler.prepare(command, tmp_path / "capture", perf=_tool(tmp_path), frequency_hz=100)
    boundary = profiler.record_command.index("--")
    assert list(profiler.record_command[boundary + 1 :]) == command
    assert "--pid" not in profiler.record_command[:boundary]
    assert "cpu-clock:u" in profiler.record_command[:boundary]
    assert profiler.environment_delta == {"PYTHON_PERF_JIT_SUPPORT": "1"}
    assert profiler.tool_identity["sha256"]
    (profiler.output_dir / "retained").write_text("evidence")
    with pytest.raises(FileExistsError):
        PerfCpuProfiler.prepare(command, profiler.output_dir, perf=profiler.tool)
    assert (profiler.output_dir / "retained").read_text() == "evidence"


def test_missing_provider_is_unavailable_without_creating_a_recording(tmp_path: Path) -> None:
    with pytest.raises(ProfilerUnavailableError) as caught:
        PerfCpuProfiler.prepare([sys.executable], tmp_path / "capture", perf=str(tmp_path / "missing"))
    assert caught.value.reason == "perf_executable_unavailable"
    assert not (tmp_path / "capture").exists()


def test_native_samples_keep_missing_python_callchains_and_unknown_symbols_visible(tmp_path: Path) -> None:
    profiler = _recorded(tmp_path)
    document = profiler.finish(0, runtime_after_imports={"gil_enabled": False})
    assert document["outcome"] == "degraded"
    assert document["coverage"]["native"] == "leaf_symbols"
    assert document["coverage"]["python"] == "unavailable"
    assert document["coverage"]["off_cpu"] == "unavailable"
    assert document["total_samples"] == 3
    assert document["lost_samples"] == 0
    assert document["unknown_leaf_samples"] == 1
    assert document["empty_stack_samples"] == 1
    assert "python_stack_symbols_unavailable" in document["limitations"]
    assert "unknown_native_leaf_symbols" in document["limitations"]
    assert document["artifacts"]["perf.data"]["sha256"]
    summary = profile_report.summarise(document, top=10, thread_filter=None)
    assert summary["coverage"] == document["coverage"]
    assert summary["native_leaf_samples"][0] == ("worker[42] libsqlite3.so:sqlite3VdbeExec", 2)
    with pytest.raises(profile_report.ProfileRefusedError) as caught:
        profile_report.collapsed(document, weight="cpu", thread_filter=None)
    assert caught.value.reason == "cpu_profile_callchains_unavailable"
    with pytest.raises(FileExistsError):
        profiler.finish(0, runtime_after_imports={"gil_enabled": False})


@pytest.mark.parametrize(
    "runtime,limitation",
    [(None, "runtime_gil_after_imports_unavailable"), ({"gil_enabled": True}, "runtime_gil_enabled")],
)
def test_python_symbol_observation_does_not_hide_incompatible_or_missing_gil_evidence(
    tmp_path: Path, runtime: dict[str, bool] | None, limitation: str
) -> None:
    profiler = _recorded(tmp_path, script=_SCRIPT.replace("sqlite3VdbeExec", "py::known_hot:/neutral.py"))
    document = profiler.finish(0, runtime_after_imports=runtime)
    assert document["python_symbol_samples"] == 2
    assert document["outcome"] == "degraded"
    assert limitation in document["limitations"]


def test_only_unknown_symbols_are_unavailable_even_when_cpu_samples_exist(tmp_path: Path) -> None:
    profiler = _recorded(
        tmp_path, report=_REPORT.replace("libsqlite3.so", "[unknown]").replace("sqlite3VdbeExec", "0x1234")
    )
    document = profiler.finish(0, runtime_after_imports={"gil_enabled": False})
    assert document["total_samples"] == 3
    assert document["outcome"] == "unavailable"
    assert document["reason"] == "cpu_profile_symbols_unavailable"
    with pytest.raises(profile_report.ProfileRefusedError):
        profile_report.summarise(document, top=10, thread_filter=None)


def test_failed_recording_preserves_evidence_without_a_zero_cpu_claim(tmp_path: Path) -> None:
    profiler = _recorded(tmp_path)
    document = profiler.finish(1, runtime_after_imports=None)
    assert document["outcome"] == "unavailable"
    assert document["reason"] == "perf_record_command_failed"
    assert "total_samples" not in document
    assert document["stages"] == {}
    assert document["artifacts"]["perf.data"]["sha256"]


def test_loss_counter_is_unknown_when_the_provider_does_not_supply_it() -> None:
    assert parse_perf_report(_REPORT.replace("# Total Lost Samples: 0\n", ""))["lost_samples"] is None
    assert parse_perf_report(_REPORT.replace("Samples: 0", "Samples: 4"))["lost_samples"] == 4
    assert parse_perf_stacks(_SCRIPT) == {"stack_samples": 3, "python_symbol_samples": 0, "empty_stack_samples": 1}


@pytest.mark.uses_real_clock("owned process wall time is observed by the diagnostic wrapper")
def test_standalone_command_binds_child_runtime_and_renders_native_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "capture"
    code = f"import json, os; from pathlib import Path; assert os.environ['PYTHON_PERF_JIT_SUPPORT']=='1'; Path({str(runtime)!r}).write_text(json.dumps({{'gil_enabled':False}}))"
    assert (
        main(
            [
                "--output-dir",
                str(output),
                "--perf",
                _tool(tmp_path),
                "--runtime-receipt",
                str(runtime),
                "--",
                sys.executable,
                "-c",
                code,
            ]
        )
        == 0
    )
    document = json.loads((output / "profile.json").read_text())
    assert document["runtime_after_imports"]["gil_enabled"] is False
    assert document["budget_qualification"] == "diagnostic_only"
    assert document["record_elapsed_s"] <= document["diagnostic_elapsed_s"]
    assert profile_report.main([str(output / "profile.json")]) == 0
    assert '"native": "leaf_symbols"' in capsys.readouterr().out
    collapsed = tmp_path / "collapsed.txt"
    assert profile_report.main([str(output / "profile.json"), "--collapsed", str(collapsed)]) == 2
    assert not collapsed.exists()


def test_sampler_runtime_receipt_observes_effective_gil_after_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False)
    monkeypatch.setattr(sys, "is_stack_trampoline_active", lambda: True)
    runtime = sampler.runtime_identity()
    assert runtime["gil_enabled"] is False
    assert runtime["perf_trampoline_active"] is True
    assert runtime["executable"] == sys.executable


def test_permission_refusal_identifies_the_missing_kernel_access(tmp_path: Path) -> None:
    profiler = _recorded(tmp_path)
    (profiler.output_dir / "record.stderr").write_text("perf_event_open failed: Operation not permitted")
    document = profiler.finish(1, runtime_after_imports=None)
    assert document["reason"] == "perf_event_permission_denied"
    assert document["coverage"]["native"] == "unavailable"


def test_stale_child_runtime_is_refused_before_an_owned_command_starts(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"gil_enabled": false}')
    output = tmp_path / "capture"
    with pytest.raises(FileExistsError):
        main(["--output-dir", str(output), "--runtime-receipt", str(runtime), "--", sys.executable, "-c", "pass"])
    assert not output.exists()


@pytest.mark.uses_real_clock("owned process wall time is observed by the diagnostic wrapper")
def test_invalid_child_runtime_keeps_recording_and_exposes_a_gap(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "capture"
    code = f"from pathlib import Path; Path({str(runtime)!r}).write_text('invalid json')"
    assert (
        main(
            [
                "--output-dir",
                str(output),
                "--perf",
                _tool(tmp_path),
                "--runtime-receipt",
                str(runtime),
                "--",
                sys.executable,
                "-c",
                code,
            ]
        )
        == 0
    )
    document = json.loads((output / "profile.json").read_text())
    assert document["outcome"] == "degraded"
    assert document["runtime_after_imports"] is None
    assert "runtime_receipt_invalid" in document["limitations"]
    assert document["artifacts"]["perf.data"]["sha256"]
