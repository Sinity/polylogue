"""Native bootstrap proof: exact browser allowlist and identity-bound replies."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

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


def test_native_host_rejects_missing_browser_sender(monkeypatch) -> None:
    payload = json.dumps({"endpoint": "http://127.0.0.1:8765"}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(native_host.sys, "argv", ["host"])
    monkeypatch.setattr(
        native_host.sys,
        "stdin",
        type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})(),
    )
    monkeypatch.setattr(native_host.sys, "stdout", type("S", (), {"buffer": output})())
    assert native_host.main() == 1
    size = struct.unpack("<I", output.getvalue()[:4])[0]
    assert json.loads(output.getvalue()[4 : 4 + size])["error"] == "native_sender_identity_required"


def test_native_host_binds_expected_receiver_identity(monkeypatch) -> None:
    payload = json.dumps({"endpoint": "http://127.0.0.1:8765", "receiver_id": "rx-other"}).encode()
    output = io.BytesIO()
    monkeypatch.setattr(native_host.sys, "argv", ["host", "chrome-extension://good-id/"])
    monkeypatch.setattr(
        native_host.sys,
        "stdin",
        type("S", (), {"buffer": io.BytesIO(struct.pack("<I", len(payload)) + payload)})(),
    )
    monkeypatch.setattr(native_host.sys, "stdout", type("S", (), {"buffer": output})())
    monkeypatch.setattr(native_host, "load_or_mint_receiver_identity", lambda: "rx-actual")
    assert native_host.main() == 1
    size = struct.unpack("<I", output.getvalue()[:4])[0]
    assert json.loads(output.getvalue()[4 : 4 + size])["error"] == "receiver_identity_mismatch"
