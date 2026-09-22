"""Native bootstrap proof: exact browser allowlist and identity-bound replies."""

from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

import pytest

from polylogue.browser_capture import native_host


def test_install_native_host_is_scoped_to_exact_extension_ids(tmp_path: Path) -> None:
    target = tmp_path / "host.json"
    native_host.install_native_host(
        ("z-extension", "a-extension", "a-extension"), executable="/bin/host", destination=target
    )
    manifest = json.loads(target.read_text())
    assert manifest["allowed_origins"] == ["chrome-extension://a-extension/", "chrome-extension://z-extension/"]
    assert manifest["path"] == "/bin/host"
    assert "auth_token" not in target.read_text()


def test_native_host_rejects_missing_browser_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"endpoint": "http://127.0.0.1:8765"}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["host"])
    monkeypatch.setattr(
        sys,
        "stdin",
        type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})(),
    )
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": output})())
    assert native_host.main() == 1
    size = struct.unpack("<I", output.getvalue()[:4])[0]
    assert json.loads(output.getvalue()[4 : 4 + size])["error"] == "native_sender_identity_required"


def test_native_host_binds_expected_receiver_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"endpoint": "http://127.0.0.1:8765", "receiver_id": "rx-other"}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(sys, "argv", ["host", "chrome-extension://good-id/"])
    monkeypatch.setattr(
        sys,
        "stdin",
        type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})(),
    )
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": output})())
    monkeypatch.setattr(native_host, "load_or_mint_receiver_identity", lambda: "rx-actual")
    assert native_host.main() == 1
    size = struct.unpack("<I", output.getvalue()[:4])[0]
    assert json.loads(output.getvalue()[4 : 4 + size])["error"] == "receiver_identity_mismatch"


def test_install_resolves_a_bare_command_to_an_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A native-messaging manifest's `path` must be absolute, not a command name.

    The installer's own `--executable` default is the bare console-script name
    `polylogue-browser-capture-native-host`, and it was written verbatim, so
    `browser-capture native-host install` reported success while installing a
    host neither Chrome nor Firefox can launch on Linux or macOS.

    Anti-vacuity: writing `executable` straight into the record makes the
    manifest `path` the bare name below.
    """
    launcher = tmp_path / "bin" / "polylogue-browser-capture-native-host"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(launcher.parent))

    target = tmp_path / "host.json"
    native_host.install_native_host(
        ("a-extension",), executable="polylogue-browser-capture-native-host", destination=target
    )

    manifest = json.loads(target.read_text())
    assert manifest["path"] == str(launcher)
    assert Path(manifest["path"]).is_absolute()


def test_install_refuses_an_unresolvable_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable launcher is refused, not written as a broken manifest."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    target = tmp_path / "host.json"

    with pytest.raises(ValueError, match="not on PATH"):
        native_host.install_native_host(("a-extension",), executable="no-such-launcher", destination=target)

    assert not target.exists()
