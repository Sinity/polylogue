"""Installer and native-messaging host for secure browser-capture pairing."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from polylogue.browser_capture.models import BROWSER_CAPTURE_API_SCHEMA
from polylogue.browser_capture.receiver import load_or_mint_receiver_identity, load_or_mint_receiver_token

NATIVE_HOST_NAME = "com.polylogue.browser_capture"


def native_host_manifest_path(*, browser: str = "chrome", home: Path | None = None) -> Path:
    root = home or Path.home()
    if browser == "firefox":
        return root / ".mozilla" / "native-messaging-hosts" / f"{NATIVE_HOST_NAME}.json"
    return root / ".config" / "google-chrome" / "NativeMessagingHosts" / f"{NATIVE_HOST_NAME}.json"


def install_native_host(
    extension_ids: tuple[str, ...], *, executable: str, browser: str = "chrome", destination: Path | None = None
) -> Path:
    ids = tuple(sorted({item.strip() for item in extension_ids if item.strip()}))
    if not ids or any("/" in item or ":" in item for item in ids):
        raise ValueError("at least one valid extension ID is required")
    target = destination or native_host_manifest_path(browser=browser)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": NATIVE_HOST_NAME,
        "description": "Polylogue browser-capture secure credential bootstrap",
        "path": executable,
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{item}/" for item in ids],
    }
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def _read_message() -> dict[str, object] | None:
    header = sys.stdin.buffer.read(4)
    if len(header) != 4:
        return None
    size = struct.unpack("<I", header)[0]
    if size > 64 * 1024:
        return None
    payload = sys.stdin.buffer.read(size)
    value = json.loads(payload) if len(payload) == size else None
    return value if isinstance(value, dict) else None


def _write_message(value: dict[str, object]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)) + payload)
    sys.stdout.buffer.flush()


def main() -> int:
    from polylogue.runtime import require_free_threaded_runtime

    require_free_threaded_runtime(consumer="polylogue browser native host")
    origin = sys.argv[1] if len(sys.argv) > 1 else ""
    extension_id = (
        origin.removeprefix("chrome-extension://").rstrip("/") if origin.startswith("chrome-extension://") else ""
    )
    request = _read_message()
    if not extension_id or request is None:
        _write_message({"ok": False, "error": "native_sender_identity_required"})
        return 1
    endpoint = str(request.get("endpoint") or "")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        _write_message({"ok": False, "error": "loopback_endpoint_required"})
        return 1
    receiver_id = load_or_mint_receiver_identity()
    expected = request.get("receiver_id")
    if expected is not None and expected != receiver_id:
        _write_message({"ok": False, "error": "receiver_identity_mismatch", "receiver_id": receiver_id})
        return 1
    _write_message(
        {
            "ok": True,
            "receiver_id": receiver_id,
            "api_schema": BROWSER_CAPTURE_API_SCHEMA,
            "auth_token": load_or_mint_receiver_token(),
        }
    )
    return 0


__all__ = ["NATIVE_HOST_NAME", "install_native_host", "main", "native_host_manifest_path"]
