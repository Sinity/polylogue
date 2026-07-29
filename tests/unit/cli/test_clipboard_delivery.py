"""Clipboard delivery: right tool for the session, loud failure otherwise.

`--to clipboard` reported "no clipboard tool found" on a Wayland host that had
wl-clipboard installed, because the candidate list was X11/macOS/Windows only
(xclip/xsel/pbcopy/clip). Worse, the miss was reported to stderr and the
command still exited 0, so a delivery that discarded its entire payload looked
like success.
"""

from __future__ import annotations

import subprocess
from typing import Any

import click
import pytest

from polylogue.cli import query_output


class _RecordingEnv:
    """Minimal AppEnv stand-in capturing what the console was told."""

    def __init__(self) -> None:
        self.printed: list[str] = []
        outer = self

        class _Console:
            def print(self, message: str) -> None:
                outer.printed.append(message)

        class _UI:
            console = _Console()

        self.ui = _UI()


def _spy_run(available: set[str], calls: list[str]) -> Any:
    """Fake ``subprocess.run`` where only ``available`` executables exist."""

    def _run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd[0])
        if cmd[0] not in available:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    return _run


def test_wayland_session_prefers_wl_copy_over_x11_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Wayland, wl-copy is tried FIRST even when an X11 tool also exists.

    XWayland sets DISPLAY too, so xclip can be present and exit 0 while writing
    to the X selection the compositor need not mirror back -- a silent
    wrong-clipboard write. Ordering is the fix, not mere availability.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("DISPLAY", ":0")
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _spy_run({"wl-copy", "xclip"}, calls))

    env = _RecordingEnv()
    query_output.copy_to_clipboard(env, "payload")

    assert calls == ["wl-copy"], f"expected wl-copy first on Wayland, got {calls}"
    assert env.printed == ["Copied to clipboard (wl-copy)."]


def test_x11_session_falls_through_to_xclip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without WAYLAND_DISPLAY the portable tools lead, so X11 is unaffected."""
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _spy_run({"xclip"}, calls))

    query_output.copy_to_clipboard(_RecordingEnv(), "payload")

    assert calls[0] == "xclip"


def test_no_clipboard_tool_raises_instead_of_exiting_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discarded payload must be an error, never a stderr note beside exit 0."""
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    calls: list[str] = []
    monkeypatch.setattr(subprocess, "run", _spy_run(set(), calls))

    with pytest.raises(click.ClickException) as excinfo:
        query_output.copy_to_clipboard(_RecordingEnv(), "payload")

    message = str(excinfo.value)
    assert "wl-clipboard" in message, "must name the Wayland package to install"
    assert "--to file" in message, "must offer a way to actually capture the output"
    assert "wl-copy" in calls, "every candidate should have been attempted"


def test_present_but_failing_tool_reports_which_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool that exists and errors is a different diagnosis from none installed."""
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    def _run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        if cmd[0] == "wl-copy":
            raise subprocess.CalledProcessError(1, cmd)
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", _run)

    with pytest.raises(click.ClickException) as excinfo:
        query_output.copy_to_clipboard(_RecordingEnv(), "payload")

    message = str(excinfo.value)
    assert "wl-copy" in message and "returned an error" in message
    assert "No clipboard tool found" not in message
