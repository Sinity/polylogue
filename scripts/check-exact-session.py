"""Qualify exact selection and Session reads with the unchanged Morrow oracle.

This bounded stage deliberately stops before the still-unimplemented export
command. The full check-morrow.py driver retains its original acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import importlib.util

spec = importlib.util.spec_from_file_location("morrow_driver", Path(__file__).with_name("check-morrow.py"))
morrow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morrow)


class ExactRun(morrow.Run):
    def answers(self, archive, expected, name, checker):
        sid = expected["session_id"]
        errors = expected.get("record_errors", 0)
        exits = (5,) if errors else (0,)
        listing = self.client(archive, sid, "--json", expected=(1,) if expected["sessions"] == 0 else exits)
        (self.output / f"{name}-query.json").write_text(json.dumps(listing, indent=2) + "\n")
        if expected["sessions"] == 0:
            checker.check("polylogue", expected, listing)
            return
        morrow.qualification(listing, errors, "listing")
        morrow.require(listing.get("up_to_date") is True and listing.get("count") == 1
                       and [row.get("id") for row in listing.get("rows", [])] == [sid], "wrong exact selection")
        answer = self.client(archive, sid, "read", "--format", "json", expected=exits)
        morrow.qualification(answer.get("result"), errors, "standalone read")
        checker.check("polylogue", expected, answer)
        (self.output / f"{name}-actual.json").write_text(json.dumps(answer, indent=2) + "\n")
        session = answer["session"]
        for field, value in {"title": "Project skeleton", "display_title": "Project skeleton",
                             "title_source": "provider", "cwd": "/home/op/morrow", "branch": "main",
                             "harness": "claude-code", "harness_version": "2.1.220"}.items():
            morrow.require(session.get(field) == value, f"wrong Session {field}")
        models = {}
        for fact in expected["message_facts"]:
            if fact.get("model"):
                models[fact["model"]] = models.get(fact["model"], 0) + 1
        morrow.require(session.get("models") == models, "wrong Session model Message counts")
        times = [fact["at"] for fact in expected["message_facts"]]
        morrow.require(session.get("started_at") == min(times)
                       and session.get("last_activity_at") == max(times), "wrong own Session time reduction")
        terminal = self.command([self.binary, "--archive", archive, sid, "read"], "terminal " + name, exits)
        for fact in expected["message_facts"]:
            morrow.require(fact["id"] in terminal, "terminal omitted Message identity")
            for block in fact["blocks"]:
                if "text" in block:
                    morrow.require(block["text"] in terminal, "terminal changed or omitted Message text")
                if block["kind"] == "tool_call":
                    rendered_input = json.dumps(block["input"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                    morrow.require(block["tool"] in terminal and rendered_input in terminal, "terminal omitted tool arguments")
                if block["kind"] == "tool_result":
                    morrow.require("[tool_result " + block["outcome"] + " " + block["call_id"] + "]" in terminal,
                                   "terminal omitted result status or invocation")
        if not errors:
            morrow.require(answer["result"]["settlement"]["state"] == "settled", "read lacks scoped proof")
            for population in ["links", "events", "notes", "runs", "deliveries"]:
                morrow.require(answer.get(population) == [], f"unproved empty {population}")
        # Reject a planted content loss with the original independent oracle.
        if name == "base":
            mutant = dict(answer, messages=[])
            (self.output / "base-corrupted.json").write_text(json.dumps(mutant, indent=2) + "\n")
            try:
                checker.check("polylogue", expected, mutant)
            except ValueError:
                self.corrupt_answer_rejected = True
            else:
                raise ValueError("unchanged Morrow oracle accepted missing Messages")


def controls(run):
    sid = "claude-code:exact-control"
    base = [{"type":"ai-title", "aiTitle":"Synthetic Session"},
            {"type":"user", "uuid":"known-row", "sessionId":"exact-control", "promptSource":"typed",
             "timestamp":"2026-10-01T08:00:00Z", "message":{"role":"user","content":"Synthetic prompt"}}]
    for kind in ["related_pending", "foreign_pending", "malformed_only", "foreign_link", "late_subagent", "required_companion"]:
        archive = run.output / ("control-" + kind)
        run.start(archive)
        def admit(records, name, context, exits):
            path = run.output / (kind + "-" + name + ".jsonl")
            path.write_bytes(records if isinstance(records, bytes) else b"".join((json.dumps(row)+"\n").encode() for row in records))
            ctx = path.with_suffix(".context.json")
            ctx.write_text(json.dumps({"version":1,"context":context}))
            return run.client(archive, "import", str(path), "--format", "claude-code-jsonl", "--context", str(ctx),
                              "--wait", "--wait-timeout", "2", expected=exits)
        admit(base, "base", ["claude-main","exact-control"], (0,))
        before = run.client(archive, sid, "read", "--format", "json")
        morrow.require(before["result"]["outcome"] == "ok", "control baseline must be complete")
        context = ["claude-main", "exact-control"]
        if kind == "foreign_pending":
            context = ["claude-main","foreign-main"]
        if kind in ["related_pending", "foreign_pending"]:
            admit(b"{", "later", context, (7,))
        elif kind == "malformed_only":
            admit(b"{not json}\n", "later", context, (2,))
        elif kind == "foreign_link":
            admit([{"type":"continued-in","continuedInSessionId":"exact-control"}], "later", ["claude-main","foreign-main"], (2,))
        elif kind == "late_subagent":
            admit(b"", "later", ["claude-subagent","ambiguous-agent"], (0,))
        else:
            admit([{"type":"user","uuid":"required-result","sessionId":"exact-control",
                    "parentUuid":"known-row", "timestamp":"2026-10-01T08:01:00Z",
                    "toolUseResult":{"persistedOutputPath":"/synthetic/tool-results/result.txt"},
                    "message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"call", "content":"<persisted-output>\n/synthetic/tool-results/result.txt\n</persisted-output>"}]}}],
                  "later", context, (2,))
        selection = run.client(archive, sid, "--json", expected=(0,) if kind in ["foreign_pending", "foreign_link"] else (5,))
        after = run.client(archive, sid, "read", "--format", "json", expected=(5,))
        morrow.require(after["result"]["outcome"] == "degraded", kind + " lost its qualification")
        morrow.require(after["result"]["settlement"]["settled_through"]["archive_counter"] > before["result"]["settlement"]["settled_through"]["archive_counter"], "later admission did not move the new basis")
        if kind == "malformed_only":
            morrow.require(after["result"]["unknown"].get("record_errors") == 1, "malformed-only input vanished from coverage")
        if kind == "foreign_pending":
            morrow.require(selection["settlement"]["state"] == "settled" and selection["up_to_date"] is False,
                           "independent selection was globally gated or marked fresh")
        morrow.require([message["id"] for message in after["messages"]] == [message["id"] for message in before["messages"]], "later unknown input fabricated or lost known Messages")
        run.stop()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--spec-repo", type=Path, required=True)
    parser.add_argument("--spec-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    morrow.require(args.output.is_absolute(), "output must be absolute")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    run = ExactRun(args.output, None)
    run.binary = args.binary.resolve(strict=True)
    run.corrupt_answer_rejected = False
    run.steps = {name: {"outcome": "unrun"} for name in ["fixture", "base", "repeat", "append", "restart", "empty", "malformed", "dependency_controls"]}
    result = {"success": False, "scope": "exact native Session selection and JSON/terminal read",
              "failure": None, "started_at": datetime.now(timezone.utc).isoformat(),
              "candidate": {}, "spec": {}, "steps": run.steps, "commands": run.commands,
              "owned_processes": run.processes,
              "remaining": ["full unchanged Morrow driver including export", "full M1",
                            "productive late external-result companion acquisition", "general lineage/title/event products",
                            "observed same-input append", "arbitrary queries", "MCP"]}
    try:
        head = run.command(["git", "rev-parse", "HEAD"], "candidate revision").strip()
        morrow.require(not run.command(["git", "status", "--porcelain"], "candidate cleanliness").strip(), "candidate must be clean")
        result["candidate"].update(head=head, binary=str(run.binary), binary_sha256=hashlib.sha256(run.binary.read_bytes()).hexdigest())
        revision = run.command(["git", "-C", args.spec_repo, "rev-parse", args.spec_revision + "^{commit}"], "spec revision").strip()
        result["spec"]["revision"] = revision
        hashes = {}
        def fixture():
            for relative in morrow.FIXTURES:
                data = subprocess.check_output(["git", "-C", str(args.spec_repo), "show", revision + ":" + relative])
                path = args.output / "spec" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                hashes[relative] = hashlib.sha256(data).hexdigest()
            demo = args.output / "spec/polylogue-sinex-joint/realization/demos"
            result["expected"] = morrow.load(demo / "prepare.py", "exact_prepare").prepare(args.output / "prepared")["polylogue"]
            result["spec"]["source_sha256"] = hashes
        run.step("fixture", fixture)
        checker = morrow.load(args.output / "spec/polylogue-sinex-joint/realization/demos/check_answer.py", "exact_checker")
        cases = result["expected"]["cases"]
        archive = args.output / "archive"
        provider = args.output / "prepared" / cases["base"]["input"]
        run.step("base", lambda: (run.start(archive), run.admit(archive, provider, cases["base"], "base", checker)))
        run.step("repeat", lambda: run.admit(archive, provider, cases["base"], "repeat", checker))
        def append():
            result["input_change"] = morrow.append_snapshot(provider, args.output / "prepared" / cases["append"]["input"])
            run.admit(archive, provider, cases["append"], "append", checker)
        run.step("append", append)
        run.stop()
        provider.unlink()
        result["provider_removed"] = str(provider)
        run.step("restart", lambda: (run.start(archive), run.answers(archive, cases["append"], "restart", checker)))
        run.stop()
        for case in ["empty", "malformed"]:
            archive = args.output / ("archive-" + case)
            run.step(case, lambda case=case: (run.start(archive), run.admit(archive, args.output / "prepared" / cases[case]["input"], cases[case], case, checker)))
            run.stop()
        run.step("dependency_controls", lambda: controls(run))
        morrow.require(run.corrupt_answer_rejected, "negative oracle control did not run")
        result["success"] = True
    except Exception as error:
        result["failure"] = str(error)
        if run.current:
            run.steps[run.current].update(outcome="failed", failure=str(error))
    finally:
        try:
            run.stop(check=False)
            run.close_runtime()
        except Exception as error:
            result.update(success=False, cleanup_failure=str(error))
        result["runtime_removed"] = not Path(run.env["XDG_RUNTIME_DIR"]).exists()
        result["corrupt_answer_rejected"] = run.corrupt_answer_rejected
        end = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=morrow.ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=morrow.ROOT))
        result["candidate"].update(end_head=end, end_dirty=dirty)
        if dirty or end != result["candidate"].get("head"):
            result.update(success=False, binding_failure="candidate changed during verification")
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"success": result["success"], "failure": result["failure"], "result": str(args.output / "result.json")}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
