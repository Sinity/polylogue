"""The installed aliases, archive command and MCP reach the owned user unit,
and an installed hook records a repository's identity without the user's git."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

profile, expected_pid = Path(sys.argv[1]), int(sys.argv[2])
expected = json.loads(Path(sys.argv[3]).read_text())
git_fixture = Path(sys.argv[4])
process = Path("/proc") / str(expected_pid)
environment = dict(
    entry.split(b"=", 1)
    for entry in (process / "environ").read_bytes().split(b"\0") if entry
)
for name, path in expected.items():
    if name != "archive":
        assert environment[name.encode()].decode() == path, (name, environment)
argv = (process / "cmdline").read_bytes().split(b"\0")
assert argv[argv.index(b"--archive") + 1].decode() == expected["archive"], argv
assert Path(expected["archive"], "index.db").is_file()
config = Path(os.fsdecode(argv[argv.index(b"--config") + 1]))
assert tomllib.loads(config.read_text())["archive"] == expected["archive"]
for name in ("polylogue", "plg", "plog", "polylogue-main"):
    status = subprocess.run(
        [str(profile / name), "--json", "status"],
        check=True, capture_output=True, timeout=30,
    )
    answer = json.loads(status.stdout)
    assert answer["outcome"] == "ok" and answer["value"]["pid"] == expected_pid, answer
    registration = subprocess.run(
        [str(profile / name)], env=dict(os.environ, COMPLETE="bash"),
        check=True, capture_output=True, timeout=30,
    ).stdout.decode()
    targets = [line for line in registration.splitlines() if line.lstrip().startswith("complete ")]
    assert targets and all(line.endswith(" " + name) for line in targets), registration
request = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "user-service-smoke", "version": "1"},
    },
}
response = subprocess.run(
    [str(profile / "polylogue"), "mcp"],
    input=(json.dumps(request) + "\n").encode(),
    check=True, capture_output=True, timeout=30,
)
answer = json.loads(response.stdout.splitlines()[0])
assert answer["id"] == 1 and answer["result"]["serverInfo"]["name"] == "polylogue", answer


def session_start_records_git_without_git_on_path():
    """The wrapper's fallback git answers when the user's PATH has none."""
    assert shutil.which("git") is None, shutil.which("git")
    repo = Path(expected["HOME"], "work", "morrow")
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", "--no-preserve=mode", str(git_fixture / "repo"), str(repo)], check=True)
    session = "6f1c2a8e-3b4d-4e5f-8a9b-0c1d2e3f4a5b"
    payload = {"session_id": session, "hook_event_name": "SessionStart", "source": "startup", "cwd": str(repo)}
    subprocess.run(
        [str(profile / "polylogue"), "internal", "hook", "SessionStart", "--harness", "claude-code"],
        input=json.dumps(payload).encode(), cwd=repo, check=True, capture_output=True, timeout=30,
    )
    lines = [
        json.loads(line)
        for spool in Path(expected["XDG_STATE_HOME"]).rglob("*.ndjson")
        for line in spool.read_text().splitlines()
    ]
    starts = [line for line in lines if line.get("payload", {}).get("session_id") == session]
    assert len(starts) == 1, lines
    start = starts[0]
    assert "unstated" not in start, start
    assert start["git_identity"]["root_commits"] == [(git_fixture / "root").read_text().strip()], start
    assert start["git_identity"]["branch"] == "main", start
    assert start["repository"] == "https://example.invalid/morrow.git", start


session_start_records_git_without_git_on_path()
