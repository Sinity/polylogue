#!/usr/bin/env python3
"""Measure a real daemon over private copies of an xtask bench sample.

Run through AgentCTL's measure operation. Builds are outside the measured
interval; each size gets a first build and a separate unchanged restart.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import time


def status(archive):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(30)
        connection.connect(str(archive / "daemon.sock"))
        connection.sendall(b'{"op":"status"}\n')
        response = json.loads(connection.makefile("rb").readline())
        if response["result"] != "ok":
            raise RuntimeError(response)
        return response["data"]


def counters(pid):
    root = Path(f"/proc/{pid}")
    io = dict(line.split(":", 1) for line in (root / "io").read_text().splitlines())
    memory = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines())
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "read_bytes": int(io["read_bytes"]),
        "read_call_bytes": int(io["rchar"]),
        "cpu_seconds": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
        "peak_rss_bytes": int(memory["VmHWM"].split()[0]) * 1024,
        "rss_bytes": int(memory["VmRSS"].split()[0]) * 1024,
        "anonymous_bytes": int(memory["RssAnon"].split()[0]) * 1024,
    }


def run(binary, home, archive, log, idle_seconds):
    env = os.environ | {"POLYLOGUE_HOME": str(home), "POLYLOGUE_ARCHIVE": str(archive)}
    env.pop("POLYLOGUE_SINEX_CONFIG", None)
    started = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen([str(binary), "daemon"], env=env, stdout=output, stderr=output)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"daemon exited: {log}")
                if time.monotonic() - started > 1800:
                    raise TimeoutError(f"daemon startup: {log}")
                try:
                    state = status(archive)
                    if state["passes"] > 0 and not state["busy"]:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
            ready_seconds = time.monotonic() - started
            initial = counters(process.pid)
            result = {"ready_seconds": ready_seconds, "startup": initial, "first_pass": state["last_pass"]}
            if state["last_pass"]["errors"] or state["last_pass"]["unreadable_roots"]:
                raise RuntimeError(state["last_pass"])
            if idle_seconds:
                idle_at = time.monotonic()
                time.sleep(idle_seconds)
                final = counters(process.pid)
                elapsed = time.monotonic() - idle_at
                result["idle"] = {
                    "seconds": elapsed,
                    "disk_read_bytes_per_second": (final["read_bytes"] - initial["read_bytes"]) / elapsed,
                    "read_call_bytes_per_second": (final["read_call_bytes"] - initial["read_call_bytes"]) / elapsed,
                    "cpu_percent_of_one_core": 100 * (final["cpu_seconds"] - initial["cpu_seconds"]) / elapsed,
                    "peak_rss_bytes": final["peak_rss_bytes"],
                    "passes": status(archive)["passes"] - state["passes"],
                }
            return result
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            if process.returncode != 0:
                raise RuntimeError(f"daemon stop returned {process.returncode}: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--idle-seconds", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "polylogue"], check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    binary = args.out / "polylogue"
    shutil.copy2(target / "release/polylogue", binary)
    files = sorted(p for p in args.sample.rglob("*.jsonl") if p.stat().st_size <= 32_000_000)
    results = []
    for size in args.sizes_mb:
        directory = args.out / f"{size}mb"
        if directory.exists():
            raise FileExistsError(directory)
        home = directory / "home"
        home.mkdir(parents=True)
        total = 0
        count = 0
        for path in files:
            dest = home / path.relative_to(args.sample)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            total += path.stat().st_size
            count += 1
            if total >= size * 1_000_000:
                break
        # Include SQLite capture so its restart work is measured too.
        db = home / ".codex/state_5.sqlite"
        db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE threads(id TEXT PRIMARY KEY, title TEXT)")
            conn.executemany("INSERT INTO threads VALUES (?,?)", ((f"synthetic-{n}", "synthetic " * 80) for n in range(size * 100)))
        archive = directory / "archive"
        result = {"sample_bytes": total, "sample_files": count, "database_bytes": db.stat().st_size}
        result["first_build"] = run(binary, home, archive, directory / "first.log", 0)
        result["restart"] = run(binary, home, archive, directory / "restart.log", args.idle_seconds if size == max(args.sizes_mb) else 0)
        results.append(result)
        (args.out / "result.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
