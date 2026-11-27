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
    base = git("ls-remote", "origin", "refs/heads/master").split()[0]
    subjects = git("log", "--reverse", "--format=%s", f"{base}..{staged}")
    state["stages"].append([line.split()[0][1:] for line in subjects.splitlines()])
    count = len(state["stages"])
    if (state["case"] == "red-change" and count == 1) or (state["case"] == "split-change" and count == 2):
        state["heads"]["2"] = state["changed"]
    save()
elif args[:2] == ["run", "list"]:
    print(len(state["stages"]))
elif args[:2] == ["run", "watch"]:
    pass
elif args[:2] == ["run", "view"]:
    field = args[args.index("--json") + 1]
    if field == "jobs":
        print("check")
    elif field == "status":
        print("in_progress")
    elif field == "conclusion":
        print("failure" if args[2] == "1" else "success")
    else:
        raise RuntimeError(args)
elif args[:2] == ["run", "download"]:
    pass
elif args[:2] == ["run", "rerun"]:
    state["reruns"] += 1
    state["heads"]["1"] = state["changed"]
    save()
elif args[:2] == ["pr", "comment"]:
    state["comments"].append(args[2])
    save()
elif args[:2] == ["pr", "merge"]:
    number = args[2]
    head = args[args.index("--match-head-commit") + 1]
    assert state["heads"][number] == head
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
    count = 3 if case == "red-change" else 1 if case == "rerun-change" else 2
    state = {"case": case, "heads": {}, "stages": [], "reruns": 0, "merged": [], "comments": []}
    for number in range(1, count + 1):
        run("git", "checkout", "-q", "--detach", base)
        (folder / f"file-{number}").write_text(str(number))
        run("git", "add", f"file-{number}")
        run("git", "commit", "-qm", f"PR {number}")
        state["heads"][str(number)] = run("git", "rev-parse", "HEAD")
        run("git", "push", "-q", "origin", f"HEAD:refs/heads/pr-{number}")
    run("git", "checkout", "-q", "--detach", state["heads"]["1" if case == "rerun-change" else "2"])
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
    result = subprocess.run(["bash", script, "--queue", "--once"], cwd=folder, env=env,
                            text=True, capture_output=True, timeout=15)
    actual = json.loads(fixture.read_text())
    stages, merged, reruns = cases[case]
    passed = (result.returncode == 0 and actual["stages"] == stages and actual["merged"] == merged
              and actual["reruns"] == reruns and not actual["comments"])
    print(f"{case}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(result.stdout, result.stderr)
        print("Observations:", {key: actual[key] for key in ("stages", "merged", "reruns", "comments")})
    return passed

if __name__ == "__main__":
    script = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).with_name("land").resolve()
    with tempfile.TemporaryDirectory() as folder:
        passed = [exercise(str(script), case, Path(folder)) for case in cases]
    sys.exit(0 if all(passed) else 1)
