"""Owned-command Linux perf diagnostics, independent of product dependencies.

The caller supervises ``record_command`` with the same custody and cancellation
rules as its ordinary child, then calls ``finish`` after that child settles.
No attach, privilege escalation or host configuration change is offered here.
Raw perf captures contain stack memory and belong in private diagnostic storage.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "polylogue.cpu-profile.v1"
_LOST = re.compile(r"^# Total Lost Samples:\s*(\d+)", re.MULTILINE)
_SAMPLE_HEADER = re.compile(r"^\s*\S.*\s+\d+\s+[\d.]+:\s+\d+\s+cpu-clock:u:")


class ProfilerUnavailableError(ValueError):
    """The requested provider cannot start, rather than collecting zero CPU."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "resolved_path": str(resolved), "sha256": digest}


def _read_paranoid() -> str | None:
    try:
        return Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
    except OSError:
        return None


def parse_perf_report(text: str) -> dict[str, Any]:
    """Reduce perf's delimited native leaf table without inventing callchains."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 7 or not cells[0].endswith("%"):
            continue
        try:
            samples, period = int(cells[1]), int(cells[2])
            tid = int(cells[4].split(":", 1)[0])
        except ValueError:
            continue
        symbol = re.sub(r"^\[.\]\s*", "", cells[6])
        rows.append(
            {"thread": cells[3], "tid": tid, "dso": cells[5], "symbol": symbol, "samples": samples, "period_ns": period}
        )
    total = sum(row["samples"] for row in rows)
    lost_match = _LOST.search(text)
    unknown = sum(
        row["samples"]
        for row in rows
        if row["dso"] == "[unknown]" or row["symbol"] == "[unknown]" or row["symbol"].startswith("0x")
    )
    return {
        "leaf_samples": rows,
        "total_samples": total,
        "lost_samples": int(lost_match[1]) if lost_match else None,
        "unknown_leaf_samples": unknown,
    }


def parse_perf_stacks(text: str) -> dict[str, int]:
    """Count observed Python symbols and incomplete stacks in perf script."""
    samples = python_samples = empty = 0
    frames: list[str] | None = None

    def account() -> None:
        nonlocal python_samples, empty
        if frames is not None:
            python_samples += any("py::" in frame for frame in frames)
            empty += not frames

    for line in text.splitlines():
        if _SAMPLE_HEADER.match(line):
            account()
            samples += 1
            frames = []
        elif frames is not None and line.startswith("\t"):
            frames.append(line)
    account()
    return {"stack_samples": samples, "python_symbol_samples": python_samples, "empty_stack_samples": empty}


@dataclass(frozen=True, slots=True)
class PerfCpuProfiler:
    """A launch plan and reducer for one owned command, never a live PID."""

    output_dir: Path
    record_command: tuple[str, ...]
    tool: str
    tool_identity: dict[str, Any]
    frequency_hz: float

    @property
    def environment_delta(self) -> dict[str, str]:
        # Python honours this for owned Python descendants as well. Effective
        # activation and GIL mode still require the child's runtime receipt.
        return {"PYTHON_PERF_JIT_SUPPORT": "1"}

    @classmethod
    def prepare(
        cls, command: Sequence[str], output_dir: Path, *, perf: str = "perf", frequency_hz: float = 99
    ) -> PerfCpuProfiler:
        if not command:
            raise ValueError("A profiler requires an owned command")
        if not math.isfinite(frequency_hz) or frequency_hz <= 0 or int(frequency_hz) != frequency_hz:
            raise ValueError("Sampling frequency must be a finite positive integer")
        tool = shutil.which(perf)
        if tool is None:
            raise ProfilerUnavailableError("perf_executable_unavailable")
        output_dir = output_dir.resolve()
        # Existing diagnostics are evidence, never an overwrite target.
        output_dir.mkdir(parents=True, exist_ok=False)
        version = subprocess.run(
            [tool, "version", "--build-options"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        identity: dict[str, Any] = _identity(Path(tool))
        implementation = Path(tool).parent / ".perf-wrapped"
        if implementation.is_file():
            identity["implementation"] = _identity(implementation)
        identity.update(
            {
                "build_options": version.stdout,
                "version_exit_code": version.returncode,
                "perf_event_paranoid": _read_paranoid(),
                "launcher_effective_uid": os.geteuid(),
                "launcher_effective_capabilities": next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in Path("/proc/self/status").read_text().splitlines()
                        if line.startswith("CapEff:")
                    ),
                    None,
                ),
            }
        )
        record = (
            tool,
            "record",
            "-e",
            "cpu-clock:u",
            "-F",
            str(int(frequency_hz)),
            "-g",
            "-k",
            "1",
            "--call-graph",
            "dwarf,8192",
            "-o",
            str(output_dir / "perf.data"),
            "--",
            *command,
        )
        return cls(output_dir, record, tool, identity, frequency_hz)

    def finish(self, record_exit_code: int, *, runtime_after_imports: Mapping[str, Any] | None) -> dict[str, Any]:
        """Decode a settled recording; preserve failures and raw evidence."""
        document: dict[str, Any] = {
            "format": FORMAT,
            "provider": "linux-perf",
            "outcome": "unavailable",
            "reason": None,
            "command": list(self.record_command),
            "environment_delta": self.environment_delta,
            "tool": self.tool_identity,
            "record_exit_code": record_exit_code,
            "runtime_after_imports": dict(runtime_after_imports) if runtime_after_imports is not None else None,
            "frequency_hz": self.frequency_hz,
            "artifacts": {},
            "stages": {},
            "coverage": {
                "measurement": "on_cpu_user_space",
                "native": "unavailable",
                "python": "unavailable",
                "off_cpu": "unavailable",
                "descendants": "inherited_not_independently_qualified",
            },
            "limitations": [],
        }
        data = self.output_dir / "perf.data"
        if record_exit_code != 0:
            document["reason"] = "perf_record_command_failed"
            stderr = self.output_dir / "record.stderr"
            if stderr.is_file() and any(
                message in stderr.read_text(errors="replace")
                for message in ("Permission error", "No permission", "Operation not permitted")
            ):
                document["reason"] = "perf_event_permission_denied"
        elif not data.is_file() or data.stat().st_size == 0:
            document["reason"] = "perf_recording_unavailable"
        else:
            self._reduce(document)
        for path in self.output_dir.iterdir():
            if path.is_file() and path.name != "profile.json":
                document["artifacts"][path.name] = _identity(path)
        # The caller owns receipt persistence so the reducer never rewrites it.
        return document

    def _run_stage(self, document: dict[str, Any], name: str, args: list[str]) -> subprocess.CompletedProcess[str]:
        if (self.output_dir / f"{name}.stdout").exists() or (self.output_dir / f"{name}.stderr").exists():
            raise FileExistsError(f"Profile stage already recorded: {name}")
        # Perf localizes timestamps and numbers. Stabilize only decoding, so
        # recording still inherits the wrapped product's requested locale.
        result = subprocess.run(
            [self.tool, *args],
            capture_output=True,
            text=True,
            cwd=self.output_dir,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        (self.output_dir / f"{name}.stdout").write_text(result.stdout, encoding="utf-8")
        (self.output_dir / f"{name}.stderr").write_text(result.stderr, encoding="utf-8")
        document["stages"][name] = {"command": [self.tool, *args], "exit_code": result.returncode}
        return result

    def _reduce(self, document: dict[str, Any]) -> None:
        injected = self._run_stage(document, "inject", ["inject", "--jit", "-i", "perf.data", "-o", "perf.jit.data"])
        # Native leaf attribution does not require a successful JIT injection.
        source = "perf.jit.data" if injected.returncode == 0 else "perf.data"
        report = self._run_stage(
            document,
            "report",
            [
                "report",
                "-i",
                source,
                "--stdio",
                "--no-children",
                "-g",
                "none",
                "-n",
                "--show-total-period",
                "--sort",
                "comm,pid,dso,symbol",
                "--field-separator",
                "|",
            ],
        )
        if report.returncode:
            document["reason"] = "perf_report_failed"
            return
        document.update(parse_perf_report(report.stdout))
        if not document["total_samples"]:
            document["reason"] = "cpu_profile_samples_unavailable"
            return
        script = self._run_stage(document, "script", ["script", "-i", source])
        document.update(
            parse_perf_stacks(script.stdout)
            if script.returncode == 0
            else {"stack_samples": 0, "python_symbol_samples": 0, "empty_stack_samples": 0}
        )
        document["coverage"]["native"] = (
            "leaf_symbols" if document["unknown_leaf_samples"] < document["total_samples"] else "unavailable"
        )
        document["coverage"]["python"] = (
            "partial_stack_symbols"
            if document["python_symbol_samples"] and document["empty_stack_samples"]
            else "observed_stack_symbols"
            if document["python_symbol_samples"]
            else "unavailable"
        )
        limitations = document["limitations"]
        if injected.returncode:
            limitations.append("perf_jit_injection_failed")
        if script.returncode:
            limitations.append("perf_stack_decode_failed")
        if not document["python_symbol_samples"]:
            limitations.append("python_stack_symbols_unavailable")
        if document["empty_stack_samples"]:
            limitations.append("incomplete_native_callchains")
            if document["python_symbol_samples"]:
                limitations.append("python_stack_coverage_incomplete")
        if document["unknown_leaf_samples"]:
            limitations.append("unknown_native_leaf_symbols")
        if document["lost_samples"] is None:
            limitations.append("lost_sample_count_unavailable")
        elif document["lost_samples"]:
            limitations.append("lost_cpu_samples")
        runtime = document["runtime_after_imports"]
        if runtime is None or runtime.get("gil_enabled") is None:
            limitations.append("runtime_gil_after_imports_unavailable")
        elif runtime["gil_enabled"]:
            limitations.append("runtime_gil_enabled")
        if document["coverage"]["native"] == "unavailable" and document["coverage"]["python"] == "unavailable":
            document["reason"] = "cpu_profile_symbols_unavailable"
            return
        document["outcome"] = "degraded" if limitations else "observed"
        document["reason"] = limitations[0] if limitations else None


def main(argv: list[str] | None = None) -> int:
    """Wrap an owned command inside a declared managed scratch operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perf", default="perf")
    parser.add_argument("--frequency-hz", type=float, default=99)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.runtime_receipt is not None and args.runtime_receipt.exists():
        raise FileExistsError("The child runtime receipt must be new")
    try:
        profiler = PerfCpuProfiler.prepare(command, args.output_dir, perf=args.perf, frequency_hz=args.frequency_hz)
    except ProfilerUnavailableError as exc:
        print(f"profile outcome=unavailable reason={exc.reason}")
        return 2
    env = dict(os.environ)
    env.update(profiler.environment_delta)
    began = time.perf_counter()
    interrupted = False
    previous_handlers = {}
    process = None

    def request_stop(number: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, number)

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[number] = signal.signal(number, request_stop)
    try:
        with (
            (profiler.output_dir / "record.stdout").open("wb") as stdout,
            (profiler.output_dir / "record.stderr").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                profiler.record_command, env=env, stdout=stdout, stderr=stderr, start_new_session=True
            )
            # The managed operation owns detached descendants and its reaper.
            # This wrapper owns the recording process and waits for its exit.
            if interrupted:
                request_stop(signal.SIGTERM, None)
            exit_code = process.wait()
            record_elapsed_s = time.perf_counter() - began
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    runtime = None
    runtime_invalid = False
    if args.runtime_receipt is not None and args.runtime_receipt.is_file():
        try:
            source = json.loads(args.runtime_receipt.read_text())
            runtime = source.get("runtime_after_imports", source) if isinstance(source, dict) else None
            if not isinstance(runtime, dict):
                runtime = None
                runtime_invalid = True
        except (OSError, ValueError):
            runtime_invalid = True
    document = profiler.finish(exit_code, runtime_after_imports=runtime)
    if runtime_invalid:
        document["limitations"].append("runtime_receipt_invalid")
    document.update(
        {
            "record_elapsed_s": record_elapsed_s,
            "diagnostic_elapsed_s": time.perf_counter() - began,
            "interrupted": interrupted,
            "budget_qualification": "diagnostic_only",
        }
    )
    (profiler.output_dir / "profile.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"profile outcome={document['outcome']} reason={document['reason']} receipt={profiler.output_dir / 'profile.json'}"
    )
    return 2 if document["outcome"] == "unavailable" or interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
