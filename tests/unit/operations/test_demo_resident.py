"""The demo keeps its actual child lifetime outside narrated query timing."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
from builtins import BaseExceptionGroup
from pathlib import Path

import pytest

import polylogue.operations.demo_resident as resident


class Child:
    pid = 321

    def __init__(self) -> None:
        self.status: int | None = None
        self.signals: list[int] = []
        self.waited = False

    def poll(self) -> int | None:
        return self.status

    def send_signal(self, value: int) -> None:
        self.signals.append(value)
        self.status = 0

    def wait(self) -> int:
        self.waited = True
        assert self.status is not None
        return self.status


def test_demo_resident_preserves_cancellation_and_physically_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = Child()
    captured: dict[str, tuple[str, ...]] = {}
    scratch = tmp_path / "private"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(scratch))
    monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "unrelated-synthetic-token")
    monkeypatch.setenv("POLYLOGUE_CONFIG", "unrelated-config")

    def spawn(args: tuple[str, ...], **kwargs: object) -> Child:
        captured["args"] = args
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(resident, "_await_listener", lambda process, path: None)
    cancellation = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as raised:
        with resident.demo_resident(tmp_path / "archive") as env:
            assert "POLYLOGUE_API_AUTH_TOKEN" not in env
            assert env["POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS"] == "0"
            assert Path(env["POLYLOGUE_CONFIG"]).read_text() == ""
            assert env["POLYLOGUE_ARCHIVE_ROOT"] == str(tmp_path / "archive")
            assert "--no-source-catchup" in captured["args"]
            assert "--no-watch" in captured["args"]
            assert "--no-browser-capture" in captured["args"]
            raise cancellation
    assert raised.value is cancellation
    assert child.signals == [signal.SIGINT]
    assert child.waited
    assert not scratch.exists()


def test_demo_resident_listener_rejects_wrong_child(tmp_path: Path) -> None:
    listener = tmp_path / "listener.json"
    listener.write_text(json.dumps({"pid": 322, "listeners": {}}))
    with pytest.raises(RuntimeError, match="different process"):
        resident._await_listener(Child(), listener)  # type: ignore[arg-type]


def test_demo_resident_listener_accepts_only_actual_loopback_bind(tmp_path: Path) -> None:
    listener = tmp_path / "listener.json"
    listener.write_text(
        json.dumps(
            {
                "pid": 321,
                "listeners": {
                    "api": {"host": "127.0.0.1", "port": 45123},
                    "browser_capture": None,
                },
            }
        )
    )
    resident._await_listener(Child(), listener)  # type: ignore[arg-type]
    listener.write_text(
        json.dumps(
            {
                "pid": 321,
                "listeners": {
                    "api": {"host": "127.0.0.1", "port": True},
                    "browser_capture": None,
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match="record is invalid"):
        resident._await_listener(Child(), listener)  # type: ignore[arg-type]


def test_demo_resident_dead_child_is_not_readiness(tmp_path: Path) -> None:
    child = Child()
    child.status = 7
    with pytest.raises(RuntimeError, match="status 7"):
        resident._await_listener(child, tmp_path / "listener.json")  # type: ignore[arg-type]


def test_demo_resident_keeps_primary_and_failed_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "private"
    scratch.mkdir()
    child = Child()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(scratch))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(resident, "_await_listener", lambda process, path: None)
    primary = KeyboardInterrupt()
    cleanup = OSError("synthetic cleanup fault")

    def refuse_cleanup(path: Path) -> None:
        assert path == scratch
        raise cleanup

    monkeypatch.setattr("polylogue.operations.demo_resident.shutil.rmtree", refuse_cleanup)
    with pytest.raises(BaseExceptionGroup) as raised:
        with resident.demo_resident(tmp_path / "archive"):
            raise primary
    assert raised.value.exceptions == (primary, cleanup)
    assert child.waited
    assert scratch.exists()


def test_demo_resident_retries_interrupted_wait_and_keeps_real_child_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_spawn = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []
    interrupted = KeyboardInterrupt()
    primary = KeyboardInterrupt()

    def spawn(args: tuple[str, ...], **kwargs: object) -> subprocess.Popen[bytes]:
        listener = args[-1]
        code = (
            "import os,json,signal,socket,sys; "
            "signal.signal(signal.SIGINT, lambda *args: sys.exit(7)); "
            "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
            "payload={'pid':os.getpid(),'listeners':{'api':{'host':'127.0.0.1',"
            "'port':s.getsockname()[1]},'browser_capture':None}}; "
            "open(sys.argv[1]+'.stage', 'w').write(json.dumps(payload)); "
            "os.replace(sys.argv[1]+'.stage',sys.argv[1]); signal.pause()"
        )
        child = original_spawn((sys.executable, "-c", code, listener))
        children.append(child)
        original_wait = child.wait
        first = True

        def wait(timeout: float | None = None) -> int:
            nonlocal first
            if first:
                first = False
                raise interrupted
            return original_wait(timeout)

        monkeypatch.setattr(child, "wait", wait)
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(BaseExceptionGroup) as raised:
        with resident.demo_resident(tmp_path / "archive"):
            raise primary
    assert raised.value.exceptions[:2] == (primary, interrupted)
    assert isinstance(raised.value.exceptions[2], RuntimeError)
    assert "status 7" in str(raised.value.exceptions[2])
    assert len(children) == 1
    assert children[0].returncode == 7
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(children[0].pid, 0)
