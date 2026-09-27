"""In-process wall/CPU stack sampler for the fresh-build benchmark.

``py-spy`` cannot attach to the free-threaded 3.14t interpreter the daemon
runs on, and ``cProfile`` follows one thread at a time. This sampler runs as a
thread inside the measured daemon: every tick it reads ``sys._current_frames``
and each thread's cumulative CPU ticks from ``/proc/self/task/<tid>/stat``. A
sample always counts as wall time for its stack; the CPU ticks the thread
consumed since its previous sample are attributed to the same stack, so an
idle thread parked in ``wait()`` accumulates wall samples but no CPU.

The leaf Python frame's source line is kept so a caller can classify time
spent inside a C call (``execute``, ``commit``, ``fetchall``...) that has no
frame of its own. Output is one JSON document written when the process exits.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import FrameType
from typing import Final

_THREAD_SUFFIX: Final = re.compile(r"(_\d+|-\d+)$")
_MAX_DEPTH: Final = 96
_FLUSH_EVERY_S: Final = 60.0
_CLOCK_TICKS: Final = os.sysconf("SC_CLK_TCK")


def thread_group(name: str) -> str:
    """Collapse per-instance suffixes so pool workers aggregate together."""
    if name.startswith("polylogue-writer:"):
        return name
    previous = None
    while previous != name:
        previous = name
        name = _THREAD_SUFFIX.sub("", name)
    return name


def _thread_cpu_ticks(native_id: int) -> int | None:
    try:
        with open(f"/proc/self/task/{native_id}/stat", "rb") as stream:
            raw = stream.read()
    except OSError:
        return None
    # Fields after the parenthesised command name; utime and stime are the
    # 14th and 15th fields of the whole line.
    tail = raw[raw.rfind(b")") + 2 :].split()
    return int(tail[11]) + int(tail[12])


def _stack(frame: FrameType | None) -> tuple[tuple[str, str, int], ...]:
    frames: list[tuple[str, str, int]] = []
    while frame is not None and len(frames) < _MAX_DEPTH:
        code = frame.f_code
        frames.append((code.co_filename, code.co_qualname, frame.f_lineno or 0))
        frame = frame.f_back
    frames.reverse()
    return tuple(frames)


class StackSampler:
    """Sample every Python thread's stack and CPU at a fixed interval."""

    def __init__(self, out_path: Path, *, interval_s: float) -> None:
        self.out_path = out_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._wall: Counter[tuple[str, tuple[tuple[str, str, int], ...]]] = Counter()
        self._cpu: Counter[tuple[str, tuple[tuple[str, str, int], ...]]] = Counter()
        self._thread_cpu: Counter[str] = Counter()
        self._previous_cpu: dict[int, int] = {}
        self._ticks = 0
        self._sample_seconds = 0.0
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="bench-stack-sampler", daemon=True)
        self._written = False
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def start(self) -> None:
        atexit.register(self.write)
        self._thread.start()

    def _run(self) -> None:
        own = threading.get_ident()
        while not self._stop.wait(self.interval_s):
            began = time.perf_counter()
            frames = sys._current_frames()
            threads = {thread.ident: thread for thread in threading.enumerate()}
            for ident, frame in frames.items():
                if ident == own:
                    continue
                thread = threads.get(ident)
                if thread is None or thread.native_id is None:
                    continue
                group = thread_group(thread.name)
                ticks = _thread_cpu_ticks(thread.native_id)
                delta = 0
                if ticks is not None:
                    delta = max(0, ticks - self._previous_cpu.get(thread.native_id, ticks))
                    self._previous_cpu[thread.native_id] = ticks
                key = (group, _stack(frame))
                self._wall[key] += 1
                if delta:
                    self._cpu[key] += delta
                    self._thread_cpu[group] += delta
            self._ticks += 1
            self._sample_seconds += time.perf_counter() - began
            if time.monotonic() - self._last_flush >= _FLUSH_EVERY_S:
                # A daemon killed by its supervisor never runs atexit;
                # periodic snapshots keep the measured part of the run.
                self._last_flush = time.monotonic()
                self._dump()

    def write(self) -> None:
        if self._written:
            return
        self._written = True
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)
        self._dump()

    def _dump(self) -> None:
        with self._lock:
            self._dump_locked()

    def _dump_locked(self) -> None:
        stacks = [
            {
                "thread": group,
                "stack": [list(frame) for frame in stack],
                "wall_samples": wall,
                "cpu_ticks": self._cpu.get((group, stack), 0),
            }
            for (group, stack), wall in self._wall.items()
        ]
        payload = {
            "format": "polylogue.fresh-build-stack-samples.v1",
            "interval_s": self.interval_s,
            "clock_ticks_per_s": _CLOCK_TICKS,
            "ticks": self._ticks,
            "elapsed_s": time.monotonic() - self._started,
            "sampler_seconds": self._sample_seconds,
            "thread_cpu_ticks": dict(self._thread_cpu),
            "stacks": stacks,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.out_path)


def start_from_environment() -> StackSampler | None:
    """Start a sampler when the benchmark driver asked for one."""
    target = os.environ.get("POLYLOGUE_BENCH_STACK_SAMPLES")
    if not target:
        return None
    interval = float(os.environ.get("POLYLOGUE_BENCH_STACK_INTERVAL_S", "0.01"))
    sampler = StackSampler(Path(target), interval_s=interval)
    sampler.start()
    return sampler
