"""The installed aliases, archive command and MCP reach the owned user unit."""
import json
import os
from pathlib import Path
import subprocess
import sys

profile, expected_pid = Path(sys.argv[1]), int(sys.argv[2])
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
