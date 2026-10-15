#!/usr/bin/env python3
"""Independent synthetic installed-client controls for the registered query subset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def require(ok, message):
    if not ok:
        raise AssertionError(message)


class Run:
    def __init__(self, binary, output):
        self.binary, self.output = binary.resolve(), output.resolve()
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.runtime = tempfile.TemporaryDirectory(prefix="pl-query-", dir="/tmp")
        self.env = dict(os.environ, XDG_RUNTIME_DIR=self.runtime.name,
                        HOME=str(self.output / "home"), NO_COLOR="1")
        self.archive = self.output / "archive"
        self.commands, self.processes, self.checks = [], [], []
        self.child = None
        self.next_import = 1
        self.paths = []

    def cli(self, *args, expected=(5,), archive=None):
        argv = [str(self.binary), "--archive", str(archive or self.archive), *args]
        index = len(self.commands)
        record = {"argv": argv, "expected": list(expected), "exit": None,
                  "stdout": f"{index:03}.stdout", "stderr": f"{index:03}.stderr"}
        self.commands.append(record)
        with (self.output / record["stdout"]).open("wb") as out, (self.output / record["stderr"]).open("wb") as err:
            result = subprocess.run(argv, env=self.env, stdout=out, stderr=err, timeout=45)
        record["exit"] = result.returncode
        text = (self.output / record["stdout"]).read_text()
        require(result.returncode in expected, f"unexpected exit {result.returncode}: {argv}; {record}")
        return json.loads(text) if text else None

    def start(self):
        index = len(self.processes)
        log = self.output / f"daemon-{index}.log"
        argv=[str(self.binary), "--archive", str(self.archive), "daemon"]
        with log.open("wb") as out:
            self.child = subprocess.Popen(argv, env=self.env, stdout=out, stderr=out)
        self.processes.append({"pid": self.child.pid, "argv":argv, "log": log.name, "exit": None})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            require(self.child.poll() is None, f"daemon exited: {log}")
            result = self.cli("status", expected=(0, 6))
            if result and result.get("outcome") == "ok":
                return
            time.sleep(.02)
        raise AssertionError("daemon startup deadline")

    def stop(self, kill=False):
        if self.child:
            self.child.send_signal(signal.SIGKILL if kill else signal.SIGINT)
            self.processes[-1]["exit"] = self.child.wait(timeout=15)
            self.child = None

    def import_(self, native, records, context="claude-main", expected=(0,), timeout="20"):
        number = self.next_import
        self.next_import += 1
        path = self.output / f"input-{number}.jsonl"
        path.write_bytes(records)
        self.paths.append(path)
        declaration = self.output / f"context-{number}.json"
        declaration.write_text(json.dumps({"version": 1, "context": [context, native]}))
        return self.cli("import", str(path), "--format", "claude-code-jsonl", "--context", str(declaration),
                        "--machine", "00112233445566778899aabbccddeeff", "--host", "synthetic-collector",
                        "--request-id", f"01900000-0000-7000-8000-{number:012d}", "--seen-at", "17",
                        "--wait", "--wait-timeout", timeout, expected=expected)

    def query(self, text, count, ids=None, expected=(5,)):
        result = self.cli(text, "--json", expected=expected)
        require(result["unit"] == "sessions" and result["count"] == count, f"wrong count: {result}")
        require(result["next_cursor"] is None, "unexpected incomplete enumeration")
        if ids is not None:
            require([row["id"] for row in result["rows"]] == ids, f"wrong identities: {result}")
        return result

    def checked(self, name):
        self.checks.append(name)


def record(native, message, text, timestamp="2026-10-01T08:00:00.000Z", assistant=False):
    return (json.dumps({"type": "assistant" if assistant else "user", "uuid": message,
                        "sessionId": native, "timestamp": timestamp, "promptSource": "typed",
                        "message": {"role": "assistant" if assistant else "user", "content": text}}) + "\n").encode()


def exercise(run):
    # An unsupported query must fail even when no reader/archive can be opened.
    absent = run.output / "must-not-exist"
    run.cli("near:semantic", "--json", archive=absent, expected=(2,))
    require(not absent.exists(), "unsupported query created state")
    run.start()
    run.query("sessions", 0, [], expected=(1,))
    run.import_("empty-main", b"\n")
    run.query("sessions", 0, [], expected=(1,))
    run.checked("accessible empty differs from unavailable interpretation")
    native, sid = "query:main:oldest", "claude-code:query:main:oldest"
    secret = "gh" + "p_" + "A" * 36
    text = f"olderneedle {secret} Łódź refactor carefully shared phrase ranktoken ranktoken ranktoken ranktoken"
    raw = record(native, "older-message", [{"type":"text","text":text+" boundaryfirst"},{"type":"text","text":"boundarysecond distinctblocktoken"}], "2001-01-01T00:00:00.000Z")
    raw += record(native, "second-message", "separateonlytoken", "2001-01-02T00:00:00.000Z")
    run.import_(native, raw)
    for index in range(110):
        name = f"query-newer-{index:04d}"
        run.import_(name, record(name, f"newer-{index}", "new unrelated ordinary prose ranktoken " + "padding " * 20))
    listed = run.query("origin:claude-code", 111)
    require(listed["rows"][-1]["id"] == sid, "listing did not put the oldest known activity last")
    result = run.query("olderneedle", 1, [sid])
    require(result["searched"]["known_sessions"] == 111, "hidden candidate window")
    require(secret not in json.dumps(result), "hit references disclosed raw prose")
    for query in ["lodz", "łódź", "refact*", '"shared phrase"', 'origin:claude-code olderneedle phrase']:
        run.query(query, 1, [sid])
    run.query("olderneedle separateonlytoken", 0, [])
    run.query("boundaryfirst boundarysecond",1,[sid])
    run.query('"boundaryfirst boundarysecond"',0,[])
    run.checked("oldest-only match across 110 newer nonmatches; words, phrase, prefix, Unicode and same-Message conjunction")
    # Selection happens after complete membership; unavailable relevance is explicit.
    run.query("olderneedle | limit 1", 1, [sid])
    selected=run.query("ranktoken | limit 1",111,["claude-code:query-newer-0109"])
    require(selected["ranking"]["state"]=="unavailable","fabricated canonical relevance")
    before_document=run.cli(sid,"read","--format","json")
    run.cli("olderneedle", "list")
    run.checked("ordinary terminal and JSON listing use the daemon query")
    for query in ["messages olderneedle", "olderneedle OR prose", "sessions | group by origin", "--reveal"]:
        run.cli(query, "--json", expected=(2,))
    run.checked("unsupported shapes refuse before evaluation")
    run.import_(native, b"{not json}\n", expected=(2,))
    run.import_("synthetic-agent", b"{not json}\n", "claude-subagent", expected=(2,))
    incomplete = run.query("missingneedle", 0, [])
    require(incomplete["outcome"] == "degraded" and incomplete["unknown"]["record_errors"] == 1
            and incomplete["unknown"]["unresolved_inputs"] == 1, "lost malformed-only or unresolved dependency")
    run.checked("malformed-only retained input and unresolved subagent qualify even zero matching rows")
    run.import_("pending-main", b'{"unfinished":', expected=(7,), timeout="0")
    pending = run.query("olderneedle", 1, [sid])
    require(pending["unknown"]["pending_inputs"] > 0, "source EOF/progress hid pending population")
    run.checked("unparsed possible contributor remains pending independently of matching rows")
    for path in run.paths:
        path.unlink()
    run.stop(kill=True)
    run.start()
    again = run.query("olderneedle", 1, [sid])
    require(again["rows"] == pending["rows"], "retained search changed after provider loss/restart")
    document = run.cli(sid, "read", "--format", "json")
    require(document["session"]["id"] == sid and document["messages"][0]["id"] == sid + ":older-message",
            f"stable query identity did not open retained Session: {document}")
    require(document["session"]==before_document["session"] and document["messages"]==before_document["messages"], "retained structured Session changed")
    require(secret not in json.dumps(document) and "[credential:" in json.dumps(document),"ordinary read lost full-context credential masking")
    run.checked("stable query ID opens retained structured Session after provider removal and SIGKILL/restart")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = Run(args.binary, args.output)
    candidate=subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    result = {"candidate_sha":candidate,"archive":str(run.archive),"binary": str(run.binary), "binary_sha256": hashlib.sha256(run.binary.read_bytes()).hexdigest(),
              "outcome": "failed", "checks": run.checks, "commands": run.commands, "processes": run.processes}
    try:
        exercise(run)
        result["outcome"] = "passed"
    except Exception as error:
        result["failure"] = str(error)
    finally:
        run.stop()
        run.runtime.cleanup()
        result["runtime_removed"] = not Path(run.runtime.name).exists()
        (run.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"outcome": result["outcome"], "checks": len(run.checks), "result": str(run.output / "result.json"), "failure": result.get("failure")}))
    return 0 if result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
