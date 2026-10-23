#!/usr/bin/env python3
"""Synthetic command-line controls for an already built scan example."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="scan-controls-") as temporary:
    root = Path(temporary)
    valid = root / "synthetic.jsonl"
    marker = "PRIVATE_BODY_MARKER"
    valid.write_text(json.dumps({"type": "future-synthetic-kind", "text": marker}) + "\n")
    missing = root / "absent.jsonl"
    directory = root / "directory.jsonl"
    directory.mkdir()
    tail = root / "tail.jsonl"
    tail.write_text(valid.read_text() + '{"type":')
    listing = root / "inputs.txt"
    listing.write_text(f"{valid.stat().st_size} {valid}\n{missing}\n{valid}\n")
    empty = root / "empty.txt"
    empty.write_text("\n")

    def run(*args):
        result = subprocess.run([binary, *map(str, args)], capture_output=True, text=True)
        assert marker not in result.stdout + result.stderr
        return result

    for args in [("claude-code",), ("claude-code", "--threads", "0", valid),
                 ("claude-code", "--threads", "no", valid), ("unknown-origin", valid),
                 ("claude-code", "@" + str(empty)), ("claude-code", "@" + str(missing))]:
        assert run(*args).returncode != 0, args

    for threads in [1, 3]:
        result = run("claude-code", "--threads", threads, "@" + str(listing))
        assert result.returncode != 0
        report = json.loads(result.stdout)
        assert report["selected_inputs"] == 3 and len(report["inputs"]) == 3
        assert [entry["input"] for entry in report["inputs"]] == [0, 1, 2]
        assert report["files"] == 2 and report["failed_inputs"] == 1
        assert report["bytes_read"] == 2 * valid.stat().st_size
        assert report["inputs"][1]["outcome"]["stage"] == "open"

    result = run("claude-code", directory)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["inputs"][0]["outcome"]["stage"] == "read"
    assert report["inputs"][0]["outcome"]["error"]["type"] == "io"
    assert report["inputs"][0]["bytes_read"] == 0

    result = run("claude-code", tail)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["incomplete_inputs"] == 1 and report["records"] == 1
    assert report["unknown"] == 1 and report["inputs"][0]["unconsumed_bytes"] == 8

    result = run("claude-code", valid)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["files"] == 1 and report["records"] == report["unknown"] == 1
    assert report["failed_inputs"] == report["incomplete_inputs"] == 0
print("scanner CLI controls passed")
