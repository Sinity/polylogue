"""Exercise the landing queue with real Git commits and a simulated forge."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FAKE_GH = r'''#!/usr/bin/env python3
import json, os, re, subprocess, sys
from pathlib import Path

path = Path(os.environ["FORGE_FIXTURE"])
state = json.loads(path.read_text())
args = sys.argv[1:]

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

def save():
    path.write_text(json.dumps(state))

def run_record(index):
    return {
        "id": index + 1,
        "run_attempt": 1,
        "path": ".github/workflows/check.yml",
        "event": "workflow_dispatch",
        "head_repository": {"full_name": "synthetic/landing"},
        "head_branch": state["branches"][index],
        "head_sha": state["revisions"][index],
        "display_title": "Staging " + state["revisions"][index],
        "status": "completed",
        "conclusion": "failure" if index == 0 else "success",
    }

def query(value):
    expression = args[args.index("--jq") + 1]
    subprocess.run(["jq", "-r", expression], input=json.dumps(value), text=True, check=True)

if args[:2] == ["repo", "view"]:
    print("synthetic landing master")
elif args[:2] == ["api", "graphql"]:
    joined = " ".join(args)
    if "pullRequests(states:" in joined:
        print("\n".join(state["heads"]))
    elif "$number: Int!" in joined:
        print("MERGEABLE")
    else:
        for number in re.findall(r"p(\d+): pullRequest", joined):
            fields = [number, "OPEN", "false", "MERGEABLE", "master", state["heads"][number], "-", "0", "1",
                      "false", "-", "check|COMPLETED|SUCCESS|https://example.invalid/check", "-", "test " + number]
            print("\t".join(fields))
elif args[:2] == ["workflow", "run"]:
    staged = next(arg[4:] for arg in args if arg.startswith("rev="))
    branch = args[args.index("--ref") + 1]
    assert branch.startswith(("land-queue-v1-", f"land-queue-{staged}-"))
    assert git("ls-remote", "origin", "refs/heads/" + branch).split()[0] == staged
    assert branch not in state["branches"]
    state["branches"].append(branch)
    state["revisions"].append(staged)
    base = git("ls-remote", "origin", "refs/heads/master").split()[0]
    subjects = git("log", "--reverse", "--format=%s", f"{base}..{staged}")
    state["stages"].append([line.split()[0][1:] for line in subjects.splitlines()])
    count = len(state["stages"])
    if (state["case"] == "red-change" and count == 1) or (state["case"] == "split-change" and count == 2):
        state["heads"]["2"] = state["changed"]
    save()
    if count == 1 and state["case"] in (
        "resume-first", "resume-new-head", "resume-new-base", "resume-foreign-workflow",
        "resume-parallel", "resume-failed-one", "resume-shifted-ref",
        "resume-wrong-input", "resume-legacy-workflow", "resume-reordered",
    ):
        sys.exit(1)  # The forge accepted the dispatch, but its response was lost.
    if state["case"] != "split-current":
        print(f"https://github.com/synthetic/landing/actions/runs/{count}")
elif args[:3] == ["api", "--method", "GET"]:
    assert args[3] == "repos/synthetic/landing/actions/workflows/check.yml/runs"
    branch = next(arg[7:] for arg in args if arg.startswith("branch="))
    revision = next(arg[9:] for arg in args if arg.startswith("head_sha="))
    assert "event=workflow_dispatch" in args
    records = []
    for index, candidate in enumerate(state["branches"]):
        if branch == candidate and revision == state["revisions"][index]:
            record = run_record(index)
            if index == 0:
                if state["case"] == "resume-foreign-workflow":
                    record["path"] = ".github/workflows/other.yml"
                elif state["case"] == "resume-wrong-input":
                    record["display_title"] = "Staging " + state["changed"]
                elif state["case"] == "resume-legacy-workflow":
                    record["display_title"] = "check"
            if state["reruns"]:
                record["run_attempt"] = 2
            records.append(record)
    state["lookups"] += 1
    save()
    query({"workflow_runs": records})
elif args[0] == "api" and "/attempts/" in args[1]:
    match = re.fullmatch(r"repos/synthetic/landing/actions/runs/(\d+)/attempts/(\d+)", args[1])
    index, requested = int(match[1]) - 1, int(match[2])
    reply = run_record(index)
    if state["case"] in ("resume-first", "resume-running"):
        reply["conclusion"] = "success"
        if state["case"] == "resume-running" and not state["restart"]:
            reply.update(status="in_progress", conclusion=None)
    if state["reruns"]:
        state["rerun_polls"] += 1
        state["rerun_attempt_requests"].append(requested)
        save()
        if state["case"] == "rerun-change" or (
            state["case"] not in ("rerun-missing", "resume-attempt-two", "resume-attempt-two-red")
            and state["rerun_polls"] > 8
        ) or state["restart"]:
            passes = state["case"] in ("rerun-passes", "resume-attempt-two", "resume-failed-one")
            reply.update(run_attempt=2, conclusion="success" if passes else "failure")
            if state["rerun_polls"] == 9:
                reply.update(status="in_progress", conclusion=None)
    if state["case"] == "wrong-run":
        reply.update(head_branch="another-branch", head_sha=state["changed"], conclusion="success")
    if state["case"] == "wrong-input":
        reply.update(display_title="Staging " + state["changed"], conclusion="success")
    query(reply)
elif args[:2] == ["run", "view"]:
    assert args[args.index("--json") + 1] == "jobs"
    requested = int(args[args.index("--attempt") + 1])
    print("rerun-check" if requested == 2 else "initial-check")
elif args[:2] == ["run", "download"]:
    pass
elif args[:2] == ["run", "rerun"]:
    state["reruns"] += 1
    assert state["reruns"] == 1
    if state["case"] == "rerun-change":
        state["heads"]["1"] = state["changed"]
    save()
elif args[:2] == ["pr", "comment"]:
    state["comments"].append({
        "pr": args[2], "body": args[args.index("--body") + 1], "rerun_polls": state["rerun_polls"],
    })
    save()
elif args[:2] == ["pr", "merge"]:
    number = args[2]
    head = args[args.index("--match-head-commit") + 1]
    assert state["heads"][number] == head
    if state["case"] == "rerun-passes":
        assert state["rerun_polls"] >= 10
    tip = git("ls-remote", "origin", "refs/heads/master").split()[0]
    tree = git("merge-tree", "--write-tree", "--no-messages", tip, head)
    commit = git("commit-tree", tree, "-p", tip, "-m", "Merge synthetic PR")
    git("push", "-q", "origin", f"{commit}:refs/heads/master")
    state["merged"].append(number)
    save()
else:
    raise RuntimeError(args)
'''

cases = {
    "red-change": ([["1", "2", "3"], ["1", "3"]], ["1", "3"], 0),
    "split-change": ([["1", "2"], ["1"]], ["1"], 0),
    "split-current": ([["1", "2"], ["1"], ["2"]], ["1", "2"], 0),
    "rerun-change": ([["1"]], [], 1),
    "rerun-passes": ([["1"]], ["1"], 1),
    "rerun-fails": ([["1"]], [], 1),
    "rerun-missing": ([["1"]], [], 1),
    "wrong-run": ([["1", "2"]], [], 0),
    "wrong-input": ([["1", "2"]], [], 0),
    "dry-run": ([], [], 0),
    "resume-first": ([["1", "2"]], ["1", "2"], 0),
    "resume-running": ([["1", "2"]], ["1", "2"], 0),
    "resume-failed-one": ([["1"]], ["1"], 1),
    "resume-attempt-two": ([["1"]], ["1"], 1),
    "resume-attempt-two-red": ([["1"]], [], 1),
    "resume-new-head": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-new-base": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-parallel": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-foreign-workflow": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-shifted-ref": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-wrong-input": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-legacy-workflow": ([["1", "2"], ["1", "2"]], ["1", "2"], 0),
    "resume-reordered": ([["1", "2"], ["2", "1"]], ["2", "1"], 0),
}

FAKE_SLEEP = """#!/usr/bin/env python3
import json, os, signal
from pathlib import Path
state = json.loads(Path(os.environ["FORGE_FIXTURE"]).read_text())
if state["case"] == "resume-running" and not state["restart"]:
    os.kill(os.getppid(), signal.SIGKILL)
"""


def exercise(script, case, parent):
    folder = parent / case
    folder.mkdir()
    def run(*args, **kwargs):
        return subprocess.run(args, cwd=folder, capture_output=True, text=True, check=True, **kwargs).stdout.strip()
    run("git", "init", "-q", "-b", "master")
    run("git", "config", "user.name", "Synthetic Test")
    run("git", "config", "user.email", "example@example.invalid")
    run("git", "init", "-q", "--bare", str(folder / "remote"))
    run("git", "remote", "add", "origin", str(folder / "remote"))
    run("git", "commit", "--allow-empty", "-qm", "Base")
    base = run("git", "rev-parse", "HEAD")
    run("git", "push", "-q", "origin", "master")
    singleton = case.startswith(("rerun-", "resume-attempt-", "resume-failed-"))
    count = 3 if case == "red-change" else 1 if singleton else 2
    foreign_branch = "refs/heads/land-queue-foreign"
    run("git", "push", "-q", "origin", f"{base}:{foreign_branch}")
    state = {
        "case": case, "heads": {}, "stages": [], "reruns": 0, "merged": [], "comments": [],
        "branches": [], "revisions": [], "lookups": 0,
        "rerun_polls": 0, "rerun_attempt_requests": [], "restart": False,
    }
    for number in range(1, count + 1):
        run("git", "checkout", "-q", "--detach", base)
        (folder / f"file-{number}").write_text(str(number))
        run("git", "add", f"file-{number}")
        run("git", "commit", "-qm", f"PR {number}")
        state["heads"][str(number)] = run("git", "rev-parse", "HEAD")
        run("git", "push", "-q", "origin", f"HEAD:refs/heads/pr-{number}")
    run("git", "checkout", "-q", "--detach", state["heads"]["1" if count == 1 else "2"])
    run("git", "commit", "--allow-empty", "-qm", "Updated head")
    state["changed"] = run("git", "rev-parse", "HEAD")
    run("git", "push", "-q", "origin", "HEAD:refs/heads/advance")
    fixture = folder / "fixture.json"
    fixture.write_text(json.dumps(state))
    bin_dir = folder / "bin"
    bin_dir.mkdir()
    for name, body in {"gh": FAKE_GH, "sleep": FAKE_SLEEP}.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o700)
    env = {**os.environ, "FORGE_FIXTURE": str(fixture), "PATH": str(bin_dir) + ":" + os.environ["PATH"]}
    command = ["bash", script, "--queue", "--once"]
    if case == "dry-run":
        command.append("--dry-run")
    def controller():
        return subprocess.run(command, cwd=folder, env=env, text=True, capture_output=True, timeout=30)

    previous = None
    if case == "resume-parallel":
        fd = os.open(folder / ".git", os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            previous = controller()
        finally:
            os.close(fd)
    elif case.startswith("resume-"):
        previous = controller()
    if previous is not None:
        first = json.loads(fixture.read_text())
        assert previous.returncode == (-9 if case == "resume-running" else 0), previous
        assert len(first["stages"]) == 1 and not first["merged"], first
        first["restart"] = True
        if case == "resume-new-head":
            first["heads"]["2"] = first["changed"]
        if case == "resume-reordered":
            first["heads"] = dict(reversed(list(first["heads"].items())))
        if case == "resume-new-base":
            newer_base = run("git", "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "Advance master")
            run("git", "push", "-q", "origin", f"{newer_base}:refs/heads/master")
        if case == "resume-shifted-ref":
            ref = "refs/heads/" + first["branches"][0]
            run("git", "push", "-q", f"--force-with-lease={ref}:{first['revisions'][0]}", "origin", f"{base}:{ref}")
        fixture.write_text(json.dumps(first))
    result = controller()
    actual = json.loads(fixture.read_text())
    stages, merged, reruns = cases[case]
    refs = {
        ref: sha
        for sha, ref in (
            row.split() for row in run("git", "ls-remote", "origin", "refs/heads/land-queue-*").splitlines()
        )
    }
    expected_refs = {foreign_branch: base}
    if (case in ("wrong-run", "wrong-input", "rerun-missing") or previous is not None and len(stages) == 2) and actual["branches"]:
        expected_refs["refs/heads/" + actual["branches"][0]] = (
            base if case == "resume-shifted-ref" else actual["revisions"][0]
        )
    red = case in ("rerun-fails", "resume-attempt-two-red")
    passed = (result.returncode == int(red) and actual["stages"] == stages and actual["merged"] == merged
              and actual["reruns"] == reruns and refs == expected_refs)
    if case == "split-current":
        passed &= actual["lookups"] == len(stages)
    if previous is not None:
        resumed = "resuming run https://github.com/synthetic/landing/actions/runs/1 " in result.stdout
        passed &= resumed == (len(stages) == 1)
        if case == "resume-parallel":
            passed &= (actual["branches"][0].startswith(f"land-queue-{actual['revisions'][0]}-")
                       and actual["branches"][1].startswith("land-queue-v1-")
                       and actual["revisions"][0] != actual["revisions"][1])
    if red and len(actual["comments"]) == 1:
        comment = actual["comments"][0]
        passed &= (comment["pr"] == "1" and comment["rerun_polls"] >= 10
                   and "/actions/runs/1/attempts/2" in comment["body"]
                   and "rerun-check" in comment["body"] and "initial-check" not in comment["body"])
    else:
        passed &= not actual["comments"] and not red
    if case in ("rerun-passes", "rerun-fails", "rerun-missing"):
        passed &= actual["rerun_polls"] > 8
        passed &= actual["rerun_attempt_requests"] == [2] * actual["rerun_polls"]
    if case == "dry-run":
        passed &= "would push a land-queue- branch" in result.stdout
    print(f"{case}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(result.stdout, result.stderr)
        print("Observations:", {key: actual[key] for key in ("stages", "merged", "reruns", "comments")})
        print("Remote staging refs:", refs)
    return passed

if __name__ == "__main__":
    script = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).with_name("land").resolve()
    supports_resume = shutil.which("flock") and os.access("/etc/machine-id", os.R_OK)
    if not supports_resume:
        print("Skipping restart scenarios. They need a Linux machine ID and flock.")
    with tempfile.TemporaryDirectory() as folder:
        passed = [
            exercise(str(script), case, Path(folder))
            for case in cases if supports_resume or not case.startswith("resume-")
        ]
    sys.exit(0 if all(passed) else 1)
