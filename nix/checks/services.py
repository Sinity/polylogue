"""The installed aliases, archive command and MCP reach the owned user unit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

profile, expected_pid = Path(sys.argv[1]), int(sys.argv[2])
expected = json.loads(Path(sys.argv[3]).read_text())
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
