"""Exercise the landing queue with real Git commits and a simulated forge."""

import json
import os
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
    assert branch.startswith(f"land-queue-{staged}-")
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
    if state["case"] != "split-current":
        print(f"https://github.com/synthetic/landing/actions/runs/{count}")
elif args[:2] == ["run", "list"]:
    branch = args[args.index("--branch") + 1]
    assert branch == state["branches"][-1]
    query = args[args.index("--jq") + 1]
    assert state["revisions"][-1] in query
    state["lookups"] += 1
    save()
    print(len(state["stages"]))
elif args[:2] == ["run", "watch"]:
    pass
elif args[:2] == ["run", "view"]:
    field = args[args.index("--json") + 1]
    requested = int(args[args.index("--attempt") + 1]) if "--attempt" in args else None
    if field == "jobs":
        print("rerun-check" if requested == 2 else "initial-check")
    elif field in ("status", "headBranch,headSha,status,conclusion", "attempt,headBranch,headSha,status,conclusion"):
        index = int(args[2]) - 1
        reply = {
            "attempt": 1,
            "headBranch": state["branches"][index],
            "headSha": state["revisions"][index],
            "status": "completed",
            "conclusion": "failure" if index == 0 else "success",
        }
        if state["reruns"]:
            state["rerun_polls"] += 1
            if requested is not None:
                state["rerun_attempt_requests"].append(requested)
            save()
            if state["case"] == "rerun-change" or (
                state["case"] != "rerun-missing" and state["rerun_polls"] > 8
            ):
                reply.update(attempt=2, conclusion="success" if state["case"] == "rerun-passes" else "failure")
                if state["rerun_polls"] == 9:
                    reply.update(status="in_progress", conclusion=None)
        if state["case"] == "wrong-run":
            reply.update(headBranch="another-branch", headSha=state["changed"], conclusion="success")
        query = args[args.index("--jq") + 1]
        subprocess.run(["jq", "-r", query], input=json.dumps(reply), text=True, check=True)
    else:
        raise RuntimeError(args)
elif args[:2] == ["run", "download"]:
    pass
elif args[:2] == ["run", "rerun"]:
    state["reruns"] += 1
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
    "dry-run": ([], [], 0),
}

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
    count = 3 if case == "red-change" else 1 if case.startswith("rerun-") else 2
    foreign_branch = "refs/heads/land-queue-foreign"
    run("git", "push", "-q", "origin", f"{base}:{foreign_branch}")
    state = {
        "case": case, "heads": {}, "stages": [], "reruns": 0, "merged": [], "comments": [],
        "branches": [], "revisions": [], "lookups": 0,
        "rerun_polls": 0, "rerun_attempt_requests": [],
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
    for name, body in {"gh": FAKE_GH, "sleep": "#!/usr/bin/env bash\nexit 0\n"}.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o700)
    env = {**os.environ, "FORGE_FIXTURE": str(fixture), "PATH": str(bin_dir) + ":" + os.environ["PATH"]}
    command = ["bash", script, "--queue", "--once"]
    if case == "dry-run":
        command.append("--dry-run")
    result = subprocess.run(command, cwd=folder, env=env, text=True, capture_output=True, timeout=15)
    actual = json.loads(fixture.read_text())
    stages, merged, reruns = cases[case]
    refs = {
        ref: sha
        for sha, ref in (
            row.split() for row in run("git", "ls-remote", "origin", "refs/heads/land-queue-*").splitlines()
        )
    }
    expected_refs = {foreign_branch: base}
    if case in ("wrong-run", "rerun-missing") and actual["branches"]:
        expected_refs["refs/heads/" + actual["branches"][0]] = actual["revisions"][0]
    expected_exit = 1 if case == "rerun-fails" else 0
    passed = (result.returncode == expected_exit and actual["stages"] == stages and actual["merged"] == merged
              and actual["reruns"] == reruns and refs == expected_refs
              and actual["lookups"] == (len(stages) if case == "split-current" else 0))
    if case == "rerun-fails" and len(actual["comments"]) == 1:
        comment = actual["comments"][0]
        passed &= (comment["pr"] == "1" and comment["rerun_polls"] >= 10
                   and "/actions/runs/1/attempts/2" in comment["body"]
                   and "rerun-check" in comment["body"] and "initial-check" not in comment["body"])
    else:
        passed &= not actual["comments"] and case != "rerun-fails"
    if case in ("rerun-passes", "rerun-fails", "rerun-missing"):
        passed &= actual["rerun_polls"] > 8
        passed &= actual["rerun_attempt_requests"] == [2] * actual["rerun_polls"]
    if case == "dry-run":
        passed &= "would push land-queue-" in result.stdout
    print(f"{case}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(result.stdout, result.stderr)
        print("Observations:", {key: actual[key] for key in ("stages", "merged", "reruns", "comments")})
        print("Remote staging refs:", refs)
    return passed

if __name__ == "__main__":
    script = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).with_name("land").resolve()
    with tempfile.TemporaryDirectory() as folder:
        passed = [exercise(str(script), case, Path(folder)) for case in cases]
    sys.exit(0 if all(passed) else 1)
