"""Real package transports in a guest OS, with private XDG and archive paths."""
import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

binary, revision = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix="polylogue-package-") as raw:
    root = Path(raw)
    env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "HOME": str(root / "home")}
    for key, name in [("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                      ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache"),
                      ("XDG_RUNTIME_DIR", "runtime")]:
        path = root / name
        path.mkdir(mode=0o700)
        env[key] = str(path)
    Path(env["HOME"]).mkdir()
    archive = root / "archive"
    default_config = Path(env["XDG_CONFIG_HOME"]) / "polylogue/polylogue.toml"
    default_config.parent.mkdir()
    default_config.write_text("invalid = [")
    config = root / "deployment.toml"
    config.write_text("invalid = [")
    command = [binary, "--archive", str(archive), "--config", str(config)]

    def run(args, **kwargs):
        return subprocess.run(command + args, env=env, check=True, capture_output=True, timeout=30, **kwargs)

    version = run(["--version"]).stdout.decode()
    assert revision in version, version
    run(["--help"])
    config.write_text(f'archive = {json.dumps(str(archive))}\n[http]\nlisten = "127.0.0.1:0"\n')
    with (root / "daemon.log").open("wb") as log:
        daemon = subprocess.Popen(command + ["daemon"], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                assert daemon.poll() is None, (root / "daemon.log").read_text()
                status = subprocess.run(command + ["--json", "status"], env=env, capture_output=True, timeout=30)
                if status.returncode == 0:
                    answer = json.loads(status.stdout)
                    assert answer["outcome"] == "ok" and answer["value"]["pid"] == daemon.pid, answer
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError((root / "daemon.log").read_text())
                time.sleep(0.1)

            initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "package-smoke", "version": "1"}}}
            mcp = run(["mcp"], input=(json.dumps(initialize) + "\n").encode())
            reply = json.loads(mcp.stdout.splitlines()[0])
            assert reply["id"] == 1 and reply["result"]["serverInfo"]["name"] == "polylogue", reply

            frames = [
                {"type": "request", "version": 1, "method": "GET", "path": "/v1/status", "headers": {}},
                {"type": "body_end"},
                *[{"type": "response_next"} for _ in range(4)],
            ]
            wire = b""
            for frame in frames:
                encoded = json.dumps(frame).encode()
                wire += struct.pack("=I", len(encoded)) + encoded
            host = subprocess.run(command + ["chrome-extension://abcdefghijklmnopabcdefghijklmnop/"],
                                  env=env, input=wire, check=True, capture_output=True, timeout=30)
            replies = []
            data = host.stdout
            while data:
                assert len(data) >= 4
                size = struct.unpack("=I", data[:4])[0]
                assert len(data) >= size + 4
                replies.append(json.loads(data[4:4 + size]))
                data = data[4 + size:]
            assert replies[0]["type"] == "response" and replies[0]["status"] == 200, replies
            assert replies[-1]["type"] == "response_end", replies
            body = b"".join(base64.b64decode(frame["data"]) for frame in replies if frame["type"] == "response_body")
            assert json.loads(body)["daemon"] == "running", replies
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                daemon.kill()
                daemon.wait()
                raise AssertionError("the owned daemon did not stop") from error
