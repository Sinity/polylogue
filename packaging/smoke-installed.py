"""Smoke installed entrypoints through an isolated installed daemon."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def smoke_installed(*, python: Path, bin_dir: Path, work_dir: Path, suffix: str = "") -> None:
    parent = work_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    work_dir = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("POLYLOGUE_") or name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
            "HERMES_HOME",
        ):
            env.pop(name, None)
    isolated_home = work_dir / "home"
    isolated_home.mkdir(mode=0o700)
    env["HOME"] = str(isolated_home)
    config = work_dir / "polylogue.toml"
    config.touch(mode=0o600)
    for kind in ("config", "data", "cache", "state", "runtime"):
        directory = work_dir / kind
        directory.mkdir(mode=0o700, exist_ok=True)
        env[f"XDG_{kind.upper()}_HOME" if kind != "runtime" else "XDG_RUNTIME_DIR"] = str(directory)
    env.update(
        POLYLOGUE_CONFIG=str(config),
        POLYLOGUE_SITE_CONFIG=str(config),
        POLYLOGUE_ARCHIVE_ROOT=str(work_dir / "archive"),
        POLYLOGUE_CONFIG_DIR=str(work_dir / "config"),
        POLYLOGUE_FORCE_PLAIN="1",
    )
    # Ask the installed package's owner; never duplicate its socket identity.
    path = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            "import os; from polylogue.config import resolve_runtime_config; from polylogue.daemon.socket_path import daemon_socket_path; runtime=resolve_runtime_config(); assert not runtime.sources, 'installed smoke requires zero configured sources'; print(daemon_socket_path(os.environ['POLYLOGUE_ARCHIVE_ROOT']))",
        ],
        cwd=work_dir,
        env=env,
        text=True,
    ).strip()
    print("installed source isolation: zero configured sources", flush=True)
    with socket.socket() as port_reservation:
        port_reservation.bind(("127.0.0.1", 0))
        api_port = port_reservation.getsockname()[1]
    daemon_command = [
        str(bin_dir / f"polylogued{suffix}"),
        "run",
        "--no-watch",
        "--no-source-catchup",
        "--no-browser-capture",
        "--api-port",
        str(api_port),
    ]
    log_path = work_dir / "daemon.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as log:
        daemon = subprocess.Popen(daemon_command, cwd=work_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                status = daemon.poll()
                if status is not None:
                    raise RuntimeError(f"installed daemon exited before socket readiness ({status}); log={log_path}")
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    try:
                        probe.connect(path)
                    except (FileNotFoundError, ConnectionRefusedError):
                        time.sleep(0.05)
                    else:
                        break
            # Transport acceptance precedes archive readiness. Exercise the
            # declared readiness probe before the one-shot installed queries.
            while True:
                status = daemon.poll()
                if status is not None:
                    raise RuntimeError(f"installed daemon exited before archive readiness ({status}); log={log_path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{api_port}/healthz/ready") as response:
                        ready = json.load(response)
                except urllib.error.HTTPError as exc:
                    if exc.code != 503:
                        raise
                    time.sleep(0.05)
                except urllib.error.URLError as exc:
                    if not isinstance(exc.reason, ConnectionRefusedError):
                        raise
                    time.sleep(0.05)
                else:
                    if ready.get("status") != "ready":
                        raise RuntimeError(f"readiness endpoint returned an invalid success: {ready}")
                    break
            print("installed smoke: python -m polylogue --version", flush=True)
            subprocess.run([str(python), "-I", "-m", "polylogue", "--version"], cwd=work_dir, env=env, check=True)
            for script, arguments in (
                ("polylogue", ["--version"]),
                ("polylogue", ["--help"]),
                ("polylogue", ["--plain", "analyze", "--count"]),
                ("polylogue", ["--plain", "ops", "diagnostics", "workload", "--json"]),
                ("polylogue", ["--plain", "ops", "diagnostics", "space", "--json"]),
                ("polylogued", ["--help"]),
                ("polylogue-mcp", ["--help"]),
            ):
                command = [str(bin_dir / f"{script}{suffix}"), *arguments]
                print(f"installed smoke: {script}{suffix} {' '.join(arguments)}", flush=True)
                subprocess.run(command, cwd=work_dir, env=env, check=True)
        finally:
            if daemon.poll() is None:
                daemon.terminate()
            daemon.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    smoke_installed(
        python=args.python.absolute(), bin_dir=args.bin_dir.resolve(), work_dir=args.work_dir, suffix=args.suffix
    )


if __name__ == "__main__":
    main()
