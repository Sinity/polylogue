"""Run packaged Morrow clients against isolated archives; retain failed runs too.

The fixture preparer and answer checker come from an explicit specification
commit. No provider parser, store connection or installed daemon is used here.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from morrow_report import render_report


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = [
    "polylogue-sinex-joint/realization/demos/prepare.py",
    "polylogue-sinex-joint/realization/demos/check_answer.py",
    "polylogue-spec/examples/archive/claude-code/projects/-home-op-morrow/7f3a2c10-0001-4000-8000-000000000001.jsonl",
    "sinex-spec/examples/kestrel/raw/home/iris/.local/share/atuin/history.db",
]
STEPS = ["package", "fixture", "base", "repeat", "append", "restart", "empty", "malformed"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def qualification(value, record_errors, label):
    require(isinstance(value, dict), f"{label}: missing result qualifier")
    wanted = "degraded" if record_errors else "ok"
    require(value.get("outcome") == wanted, f"{label}: expected {wanted} outcome")
    require(isinstance(value.get("notes"), list)
            and all(isinstance(n, str) and n.strip() for n in value["notes"]),
            f"{label}: result diagnostics are missing or malformed")
    require(isinstance(value.get("unknown"), dict), f"{label}: missing unknown coverage")
    if record_errors:
        require(bool(value["notes"]), f"{label}: retained interpretation failure needs diagnostics")
        unknown = value.get("unknown")
        require(isinstance(unknown, dict) and type(unknown.get("record_errors")) is int
                and unknown["record_errors"] == record_errors,
                f"{label}: wrong retained interpretation failure coverage")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_snapshot(provider, prepared):
    """Change the provider by append only; a later import is still an immutable snapshot."""
    before, after = provider.read_bytes(), prepared.read_bytes()
    require(len(after) > len(before) and after.startswith(before),
            "append fixture must strictly extend the current provider bytes")
    with provider.open('ab') as stream:
        stream.write(after[len(before):])
        stream.flush()
        os.fsync(stream.fileno())
    return {"kind": "append", "path": str(provider), "before_bytes": len(before),
            "after_bytes": len(after), "appended_bytes": len(after) - len(before),
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest()}


class Run:
    def __init__(self, output, control):
        self.output = output
        self.control = control
        self.commands = []
        self.processes = []
        self.steps = {name: {"outcome": "unrun"} for name in STEPS}
        self.current = None
        self.daemon = None
        self.binary = None
        self.control_exercised = False
        self.control_result = {"state": "unrun" if control else "not_requested"}
        self.env = dict(os.environ)
        for key in ["PYTHONPATH", "PYTHONHOME", "XDG_DATA_HOME", "XDG_STATE_HOME"]:
            self.env.pop(key, None)
        home = output / "home"
        home.mkdir(mode=0o700)
        runtime = output / "runtime"
        runtime.mkdir(mode=0o700)
        self.env.update(HOME=str(home), XDG_RUNTIME_DIR=str(runtime), NO_COLOR="1")

    def command(self, argv, label, expected=(0,), timeout=30, env=None):
        index = len(self.commands)
        stdout, stderr = f"{index:03}.stdout", f"{index:03}.stderr"
        record = {"label": label, "argv": list(map(str, argv)), "cwd": str(ROOT),
                  "stdout": stdout, "stderr": stderr, "expected_exits": list(expected), "exit": None}
        self.commands.append(record)
        with (self.output / stdout).open("wb") as out, (self.output / stderr).open("wb") as err:
            try:
                result = subprocess.run(record["argv"], cwd=ROOT, env=env or self.env,
                                        stdout=out, stderr=err, timeout=timeout)
                record["exit"] = result.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                record["failure"] = str(error)
                raise ValueError(f"{label}: {error}") from error
        require(result.returncode in expected,
                f"{label}: exit {result.returncode}, expected {list(expected)}; see {stdout}/{stderr}")
        return (self.output / stdout).read_text()

    def step(self, name, action):
        self.current = name
        self.steps[name]["outcome"] = "running"
        action()
        self.steps[name]["outcome"] = "passed"
        self.current = None

    def client(self, archive, *args, expected=(0,)):
        text = self.command([self.binary, "--archive", archive, *args], "client " + " ".join(args), expected)
        return json.loads(text)

    def start(self, archive):
        require(self.daemon is None, "a daemon is already owned")
        if self.control == "launch-failure":
            archive.write_text("intentional archive-directory collision\n")
            self.control_result.update(state="prepared")
        index = len(self.processes)
        out_name, err_name = f"daemon-{index}.stdout", f"daemon-{index}.stderr"
        argv = [str(self.binary), "--archive", str(archive), "daemon"]
        with (self.output / out_name).open("wb") as out, (self.output / err_name).open("wb") as err:
            self.daemon = subprocess.Popen(argv, cwd=ROOT, env=self.env, stdout=out, stderr=err)
        if self.control == "launch-failure":
            self.control_exercised = True
        self.processes.append({"pid": self.daemon.pid, "argv": argv, "stdout": out_name,
                               "stderr": err_name, "exit": None})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            code = self.daemon.poll()
            if code is not None:
                if self.control == "launch-failure":
                    diagnostic = (self.output / err_name).read_text()
                    expected = code == 4 and str(archive) in diagnostic and f"os error {errno.ENOTDIR}" in diagnostic
                    self.control_result.update(state="launch_refused" if expected else "unrelated_launch_failure",
                                               candidate_exit=code, diagnostic=err_name)
                raise ValueError(f"candidate daemon exited at launch; see {err_name}")
            text = self.command([self.binary, "--archive", archive, "status"], "readiness status", (0, 6))
            if self.commands[-1]["exit"] == 0:
                if self.control == "launch-failure":
                    self.control_result.update(state="unexpectedly_ready")
                require(self.control != "launch-failure", "launch-failure control unexpectedly became ready")
                return
            time.sleep(0.1)
        raise ValueError(f"candidate did not become ready within 10 seconds; see {err_name}")

    def stop(self, check=True):
        if self.daemon is None:
            return
        process, self.daemon = self.daemon, None
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
            self.processes[-1]["forced_kill"] = True
        self.processes[-1]["exit"] = code
        if check:
            require(code == 0 and not self.processes[-1].get("forced_kill"),
                    f"owned daemon did not stop cleanly (exit {code})")

    def admit(self, archive, path, expected, name, checker):
        context = expected.get("import_context")
        require(isinstance(context, dict) and set(context) == {"version", "context"}
                and type(context["version"]) is int and context["version"] == 1
                and isinstance(context["context"], list) and len(context["context"]) == 2
                and context["context"][0] == "claude-main"
                and isinstance(context["context"][1], str) and bool(context["context"][1]),
                "Morrow requires an explicit v1 main-transcript import context")
        context_path = self.output / f"{name}-import-context.json"
        context_path.write_text(json.dumps(context, indent=2) + "\n")
        errors = expected.get("record_errors", 0)
        admitted = self.client(archive, "import", str(path), "--format", "claude-code-jsonl",
                               "--context", str(context_path), "--wait", "--json", expected=(5,) if errors else (0,))
        rows=admitted.get("source_rows")
        require(isinstance(rows,dict) and rows.get("scope")=="imported_source_rows"
                and isinstance(rows.get("basis"),dict)
                and rows["basis"].get("input")==admitted.get("retained",{}).get("id")
                and type(rows["basis"].get("length")) is int
                and rows["basis"]["length"]==admitted.get("retained",{}).get("length"),
                "import --wait needs an explicitly bound source-row settlement")
        if errors:
            qualification(admitted,errors,"import --wait")
            require(rows.get("state")=="unavailable" and rows.get("usable_result") is True,
                    "retained interpretation failure cannot affirm source-row settlement")
        else:
            require(rows.get("state")=="settled", "source-row settlement is still pending")
        # This narrower proof does not satisfy the ordinary Session/read/export
        # assertions below. The full Morrow outcome remains unchanged.
        self.answers(archive, expected, name, checker)

    def answers(self, archive, expected, name, checker):
        sid = expected["session_id"]
        errors = expected.get("record_errors", 0)
        exits = (5,) if errors else (0,)
        listing = self.client(archive, sid, "--json", expected=(1,) if expected["sessions"] == 0 else exits)
        (self.output / f"{name}-query.json").write_text(json.dumps(listing, indent=2) + "\n")
        if expected["sessions"] == 0:
            checker.check("polylogue", expected, listing)
            return
        qualification(listing, errors, "listing")
        require(listing.get("up_to_date") is True,
                "listing is not fresh; freshness does not imply complete evidence")
        require(listing.get("unit") == "sessions" and type(listing.get("count")) is int and listing["count"] == 1
                and [row.get("id") for row in listing.get("rows", [])] == [sid]
                and "next_cursor" in listing and listing["next_cursor"] is None, "wrong exact session selection")
        answer = self.client(archive, sid, "read", "--format", "json", expected=exits)
        qualification(answer.get("result"), errors, "standalone read")
        (self.output / f"{name}-actual.json").write_text(json.dumps(answer, indent=2) + "\n")
        checker.check("polylogue", expected, answer)
        if self.control == "corrupt-answer":
            mutant = deepcopy(answer)
            mutant["messages"] = []
            path = f"{name}-corrupted.json"
            (self.output / path).write_text(json.dumps(mutant, indent=2) + "\n")
            self.control_exercised = True
            self.control_result.update(state="checking_mutant", baseline_valid=True, mutation=path)
            try:
                checker.check("polylogue", expected, mutant)
            except ValueError as error:
                self.control_result.update(state="oracle_rejected", reason=str(error))
                raise ValueError(f"intentional corrupted-answer rejection: {error}") from error
            self.control_result.update(state="oracle_accepted")
            raise ValueError("oracle accepted the planted answer mutation; control failed")
        export = self.output / f"{name}-export"
        self.command([self.binary, "--archive", archive, sid, "export", "--format", "json", "--to", export], "export " + name, expected=exits)
        documents = [p for p in export.rglob("*.json") if p.name != "manifest.json"]
        require(len(documents) == 1, "export did not contain exactly one session document")
        exported = json.loads(documents[0].read_text())
        qualification(exported.get("result"), errors, "exported session")
        checker.check("polylogue", expected, exported)
        def semantic(document):
            return {k: v for k, v in document.items() if k not in {"exported_at", "exported_by"}}
        require(semantic(exported) == semantic(answer), "read/export semantic contents differ")
        manifest_path = export / "manifest.json"
        require(manifest_path.is_file(), "export selection manifest is missing")
        manifest = json.loads(manifest_path.read_text())
        qualification(manifest.get("result"), errors, "export manifest")
        require(manifest.get("format") == "polylogue-export-manifest"
                and type(manifest.get("version")) is int and manifest["version"] == 1
                and type(manifest.get("sessions")) is int and manifest["sessions"] == 1
                and manifest.get("up_to_date") is True, "wrong or incomplete export selection manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-repo", type=Path, required=True)
    parser.add_argument("--spec-revision", required=True, help="explicit committed fixture/oracle revision")
    parser.add_argument("--output", type=Path, required=True, help="new absolute private result directory")
    parser.add_argument("--control", choices=["launch-failure", "corrupt-answer"])
    args = parser.parse_args()
    require(args.output.is_absolute(), "--output must be absolute")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    run = Run(args.output, args.control)
    result = {"success": False, "failure": None, "control": args.control,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "invocation": [sys.executable, *sys.argv], "invocation_cwd": str(ROOT),
              "candidate": {}, "spec": {}, "steps": run.steps, "commands": run.commands,
              "acquisition_mode": "immutable_import_snapshots", "input_changes": [],
              "owned_processes": run.processes,
              "isolated_environment": {key: run.env[key] for key in ["HOME", "XDG_RUNTIME_DIR"]},
              "remaining": ["observed-source same-input/chain append, separator-only progress and restart continuation",
                            "automatic source-signature format recognition without explicit --format",
                            "source accounting and canonical progress binding", "actual provenance reads",
                            "observed retention/admission/publication kills", "provider-loss recovery",
                            "actual MCP transport", "full M0/M1 acceptance"]}
    try:
        head = run.command(["git", "rev-parse", "HEAD"], "candidate revision").strip()
        require(not run.command(["git", "status", "--porcelain"], "candidate cleanliness").strip(), "candidate must be clean")
        result["candidate"]["head"] = head
        def package():
            package = run.command(["nix", "build", "--no-link", ".#default", "--print-out-paths", "--max-jobs", "1"],
                                  "actual candidate package", timeout=900, env=dict(os.environ)).strip()
            require(package.startswith("/nix/store/") and "\n" not in package, "expected exactly one immutable package")
            run.binary = Path(package) / "bin/polylogue"
            result["candidate"].update(package=package, binary_sha256=hashlib.sha256(run.binary.read_bytes()).hexdigest())
        run.step("package", package)
        def fixture():
            revision = run.command(["git", "-C", args.spec_repo, "rev-parse", args.spec_revision + "^{commit}"], "spec revision").strip()
            require(len(revision) == 40, "expected full specification commit identity")
            result["spec"]["revision"] = revision
            hashes = {}
            for relative in FIXTURES:
                data = subprocess.check_output(["git", "-C", str(args.spec_repo), "show", revision + ":" + relative])
                path = args.output / "spec" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                hashes[relative] = hashlib.sha256(data).hexdigest()
            result["spec"]["source_sha256"] = hashes
            demo = args.output / "spec/polylogue-sinex-joint/realization/demos"
            prepared = load(demo / "prepare.py", "morrow_prepare").prepare(args.output / "prepared")
            result["expected"] = prepared["polylogue"]
        run.step("fixture", fixture)
        checker = load(args.output / "spec/polylogue-sinex-joint/realization/demos/check_answer.py", "morrow_checker")
        cases = result["expected"]["cases"]
        archive = args.output / "archive-base"
        provider = args.output / "prepared" / cases["base"]["input"]
        def admit(case, path):
            run.admit(archive, path, cases[case], run.current, checker)
        run.step("base", lambda: (run.start(archive), admit("base", provider)))
        run.step("repeat", lambda: admit("base", provider))
        def append():
            result["input_changes"].append(append_snapshot(
                provider, args.output / "prepared" / cases["append"]["input"]))
            admit("append", provider)
        run.step("append", append)
        run.step("restart", lambda: (run.stop(), run.start(archive), run.answers(archive, cases["append"], "restart", checker)))
        run.stop()
        for case in ["empty", "malformed"]:
            archive = args.output / ("archive-" + case)
            run.step(case, lambda case=case: (run.start(archive), admit(case, args.output / "prepared" / cases[case]["input"])))
            run.stop()
        require(args.control is None, "requested negative control did not exercise its expected failure")
        result["success"] = True
    except Exception as error:
        result["failure"] = str(error)
        if run.current:
            run.steps[run.current].update(outcome="failed", failure=str(error))
    finally:
        try:
            run.stop(check=False)
        except Exception as error:
            result["success"] = False
            result["cleanup_failure"] = str(error)
        result["control_exercised"] = run.control_exercised
        result["control_result"] = run.control_result
        try:
            end = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
            result["candidate"].update(end_head=end, end_dirty=dirty)
            require(end == result["candidate"].get("head") and not dirty, "candidate changed during qualification")
        except Exception as error:
            result["success"] = False
            result["binding_failure"] = str(error)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        try:
            render_report(result, args.output)
        except Exception as error:
            result.update(success=False, report_failure=str(error))
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"success": result["success"], "failure": result["failure"],
                      "result": str(args.output / "result.json"),
                      "report": str(args.output / "report.html") if (args.output / "report.html").is_file() else None}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
