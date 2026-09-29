"""Summarise a stack-sample document written by :mod:`.sampler`."""

from __future__ import annotations

import argparse
import json
import linecache
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

_SQLITE_CALL: Final = re.compile(
    r"\.(execute|executemany|executescript|commit|rollback|fetchall|fetchone|fetchmany|close|backup|connect)\("
    r"|sqlite3\.connect|__exit__|for .* in (conn|cursor|connection)\b"
)
_WAIT_FUNCTIONS: Final = frozenset({"wait", "_wait_for_tstate_lock", "select", "get", "sleep", "join", "acquire"})


def _short(path: str) -> str:
    marker = "/polylogue/"
    index = path.rfind(marker)
    if index >= 0:
        return path[index + 1 :]
    return path.rsplit("/", 2)[-1] if "/" in path else path


def _frame_label(frame: list[Any]) -> str:
    return f"{_short(frame[0])}:{frame[1]}"


def _leaf_line(frame: list[Any]) -> str:
    return linecache.getline(frame[0], int(frame[2])).strip()


def thread_group_label(thread: str) -> str:
    """Fold every per-actor writer thread into one ``polylogue-writer`` row."""
    return "polylogue-writer" if thread.startswith("polylogue-writer:") else thread


def classify_leaf(frame: list[Any]) -> str:
    """Name the kind of work a leaf frame is doing when it has no deeper frame."""
    text = _leaf_line(frame)
    if _SQLITE_CALL.search(text) or "sqlite" in frame[0]:
        return "sqlite"
    if frame[1].rsplit(".", 1)[-1] in _WAIT_FUNCTIONS:
        return "wait"
    return "python"


def summarise(document: dict[str, Any], *, top: int, thread_filter: str | None) -> dict[str, Any]:
    ticks_per_s = float(document["clock_ticks_per_s"])
    interval = float(document["interval_s"])
    by_thread_wall: Counter[str] = Counter()
    by_thread_cpu: Counter[str] = Counter()
    inclusive_cpu: Counter[str] = Counter()
    inclusive_wall: Counter[str] = Counter()
    leaf_cpu: Counter[str] = Counter()
    kind_cpu: Counter[str] = Counter()
    kind_wall: Counter[str] = Counter()
    for entry in document["stacks"]:
        thread = entry["thread"]
        group = thread_group_label(thread)
        wall = entry["wall_samples"] * interval
        cpu = entry["cpu_ticks"] / ticks_per_s
        by_thread_wall[thread] += wall
        by_thread_cpu[thread] += cpu
        if thread_filter is not None and not group.startswith(thread_filter) and not thread.startswith(thread_filter):
            continue
        stack = entry["stack"]
        if not stack:
            continue
        leaf = stack[-1]
        kind = classify_leaf(leaf)
        kind_cpu[kind] += cpu
        kind_wall[kind] += wall
        leaf_cpu[f"{_frame_label(leaf)}:{leaf[2]}  {_leaf_line(leaf)[:90]}"] += cpu
        seen: set[str] = set()
        for frame in stack:
            label = _frame_label(frame)
            if label in seen:
                continue
            seen.add(label)
            inclusive_cpu[label] += cpu
            inclusive_wall[label] += wall
    return {
        "elapsed_s": document["elapsed_s"],
        "sampler_overhead_s": document["sampler_seconds"],
        "threads_by_cpu_s": by_thread_cpu.most_common(top),
        "threads_by_wall_s": by_thread_wall.most_common(top),
        "leaf_kind_cpu_s": kind_cpu.most_common(),
        "leaf_kind_wall_s": kind_wall.most_common(),
        "inclusive_cpu_s": inclusive_cpu.most_common(top),
        "inclusive_wall_s": inclusive_wall.most_common(top),
        "leaf_cpu_s": leaf_cpu.most_common(top),
    }


def collapsed(document: dict[str, Any], *, weight: str, thread_filter: str | None) -> list[str]:
    """Brendan Gregg collapsed-stack lines for a flame graph renderer."""
    ticks_per_s = float(document["clock_ticks_per_s"])
    interval = float(document["interval_s"])
    lines: Counter[str] = Counter()
    for entry in document["stacks"]:
        thread = entry["thread"]
        if thread_filter is not None and not thread.startswith(thread_filter):
            continue
        value = entry["cpu_ticks"] / ticks_per_s if weight == "cpu" else entry["wall_samples"] * interval
        if value <= 0:
            continue
        frames = ";".join(_frame_label(frame) for frame in entry["stack"])
        lines[f"{thread};{frames}"] += value
    return [f"{stack} {round(value * 1000)}" for stack, value in lines.most_common() if round(value * 1000) > 0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--thread", default=None, help="restrict function tables to a thread-name prefix")
    parser.add_argument("--collapsed", type=Path, default=None, help="also write collapsed stacks here")
    parser.add_argument("--weight", choices=("cpu", "wall"), default="cpu")
    args = parser.parse_args(argv)
    document = json.loads(args.samples.read_text(encoding="utf-8"))
    summary = summarise(document, top=args.top, thread_filter=args.thread)
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"\n== {key}")
            for name, amount in value:
                print(f"{amount:9.2f}  {name}")
        else:
            print(f"{key}: {value:.2f}")
    if args.collapsed is not None:
        args.collapsed.write_text(
            "\n".join(collapsed(document, weight=args.weight, thread_filter=args.thread)) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
