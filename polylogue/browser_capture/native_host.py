"""Installer and native-messaging host for secure browser-capture pairing."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import secrets
import sqlite3
import struct
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import ParseResult, urlparse

from ijson.common import JSONError

from polylogue.browser_capture.models import BROWSER_CAPTURE_API_SCHEMA
from polylogue.browser_capture.receiver import (
    load_or_mint_receiver_identity,
    load_or_mint_receiver_token,
    receiver_attestation_proof,
    receiver_status_proof,
)
from polylogue.core.json import JSONValue
from polylogue.schemas.observation_spill import StreamedJSONDocument

NATIVE_HOST_NAME = "com.polylogue.browser_capture"


def native_host_manifest_path(*, browser: str = "chrome", home: Path | None = None) -> Path:
    root = home or Path.home()
    if browser == "firefox":
        return root / ".mozilla" / "native-messaging-hosts" / f"{NATIVE_HOST_NAME}.json"
    return root / ".config" / "google-chrome" / "NativeMessagingHosts" / f"{NATIVE_HOST_NAME}.json"


def resolve_native_host_executable(executable: str) -> str:
    """Return the absolute launcher path a native-messaging manifest requires.

    A manifest's ``path`` is what the browser execs. Chrome and Firefox both
    require an absolute path on Linux and macOS, so writing the bare console
    script name -- which is the installer's own default -- produced a manifest
    the browser cannot launch while the command reported success. A relative
    name is resolved on ``PATH`` here, and an unresolvable one is refused
    rather than written.
    """
    import shutil

    candidate = executable.strip()
    if not candidate:
        raise ValueError("native host executable must be a non-empty path or command name")
    path = Path(candidate)
    if path.is_absolute():
        return str(path)
    resolved = shutil.which(candidate)
    if resolved is None:
        raise ValueError(
            f"native host executable {candidate!r} is not on PATH; "
            "pass --executable with the absolute path to the installed launcher"
        )
    return str(Path(resolved).resolve())


def install_native_host(
    extension_ids: tuple[str, ...], *, executable: str, browser: str = "chrome", destination: Path | None = None
) -> Path:
    ids = tuple(sorted({item.strip() for item in extension_ids if item.strip()}))
    if not ids or any("/" in item or ":" in item for item in ids):
        raise ValueError("at least one valid extension ID is required")
    launcher = resolve_native_host_executable(executable)
    target = destination or native_host_manifest_path(browser=browser)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": NATIVE_HOST_NAME,
        "description": "Polylogue browser-capture secure credential bootstrap",
        "path": launcher,
        "type": "stdio",
        **(
            {"allowed_extensions": list(ids)}
            if browser == "firefox"
            else {"allowed_origins": [f"chrome-extension://{item}/" for item in ids]}
        ),
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


#: Seconds without progress on the loopback attestation exchange before the
#: receiver counts as unauthenticated. It bounds each blocking socket step, not
#: the whole exchange, and matches the extension's receiver health bound: a
#: receiver that answers nothing cannot be told apart from an impostor that
#: holds the port open, and the extension retries on its next health check.
RECEIVER_ATTESTATION_IDLE_TIMEOUT_S = 5.0


def _authenticate_receiver(endpoint: ParseResult, receiver_id: str, secret: str) -> str | None:
    """Challenge the endpoint; ``None`` when it answers with the bearer-keyed proof.

    Otherwise return the refusal code: ``receiver_unreachable`` when nothing
    answered, ``receiver_authentication_failed`` when something else did, and
    ``receiver_observation_storage_failed`` when local response custody could
    not create, write, decode or close its spill. Only
    the challenge crosses the socket, so an impostor listening on the receiver
    port learns no bearer from this exchange. This proof alone does not
    establish endpoint ownership against a relay to another genuine receiver.
    """
    connection = http.client.HTTPConnection(
        endpoint.hostname or "", endpoint.port or 80, timeout=RECEIVER_ATTESTATION_IDLE_TIMEOUT_S
    )
    challenge = secrets.token_urlsafe(32)
    headers = {"Content-Type": "application/json"}
    try:
        connection.request(
            "POST",
            endpoint.path.rstrip("/") + "/v1/receiver/attest",
            body=json.dumps({"challenge": challenge}),
            headers=headers,
        )
        response = connection.getresponse()
        if response.status != 200:
            return "receiver_authentication_failed"
        with _receiver_response_document(response) as body:
            proof = body.get("proof") if isinstance(body, dict) else None
            if (
                isinstance(proof, str)
                and proof.isascii()
                and hmac.compare_digest(proof, receiver_attestation_proof(secret, receiver_id, challenge))
            ):
                return None
    except (OSError, http.client.HTTPException):
        return "receiver_unreachable"
    except ReceiverObservationStorageError:
        return "receiver_observation_storage_failed"
    except (ValueError, JSONError):
        return "receiver_authentication_failed"
    finally:
        connection.close()
    return "receiver_authentication_failed"


class ReceiverResponseAuthenticationError(ValueError):
    """The received status bytes lack the challenge-bound receiver proof."""


class ReceiverObservationStorageError(RuntimeError):
    """The local response spill could not preserve a complete observation."""


@contextmanager
def _receiver_response_document(
    response: http.client.HTTPResponse, *, status_auth: tuple[str, str, str] | None = None
) -> Iterator[JSONValue]:
    """Own a complete receiver JSON response without retaining its wire collection."""
    try:
        scratch = tempfile.TemporaryDirectory(prefix="polylogue-receiver-response-")
    except OSError as exc:
        response.close()
        raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
    try:
        path = Path(scratch.name) / "response.json"
        digest = hashlib.sha256()
        stream = None
        try:
            stream = path.open("wb")
        except OSError as exc:
            raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
        try:
            while chunk := response.read(64 * 1024):
                try:
                    stream.write(chunk)
                except OSError as exc:
                    raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
                digest.update(chunk)
        finally:
            try:
                stream.close()
            except OSError as exc:
                raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
        if status_auth is not None:
            secret, identity, challenge = status_auth
            proof = response.getheader("X-Polylogue-Status-Proof", "")
            expected = receiver_status_proof(secret, identity, challenge, payload_sha256=digest.hexdigest())
            if not proof.isascii() or not hmac.compare_digest(proof, expected):
                raise ReceiverResponseAuthenticationError("receiver_authentication_failed")
        owner = StreamedJSONDocument(path)
        try:
            document = owner.__enter__()
        except (sqlite3.Error, OSError) as exc:
            raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
        try:
            yield document
        finally:
            try:
                owner.__exit__(*sys.exc_info())
            except (sqlite3.Error, OSError) as exc:
                raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
    finally:
        response.close()
        try:
            scratch.cleanup()
        except OSError as exc:
            raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc


def main() -> int:
    from polylogue.runtime import require_free_threaded_runtime

    require_free_threaded_runtime(consumer="polylogue browser native host")
    sender = sys.argv[1] if len(sys.argv) > 1 else ""
    extension_id = (
        sender.removeprefix("chrome-extension://").rstrip("/") if sender.startswith("chrome-extension://") else sender
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
    # Loopback is not identity: whatever process owns the port would receive
    # the bearer, and a fresh profile has no expected receiver id to compare.
    # Release it only to an endpoint that proves it already holds it.
    secret = load_or_mint_receiver_token()
    refusal = _authenticate_receiver(parsed, receiver_id, secret)
    if refusal is not None:
        _write_message({"ok": False, "error": refusal, "receiver_id": receiver_id})
        return 1
    _write_message(
        {
            "ok": True,
            "receiver_id": receiver_id,
            "api_schema": BROWSER_CAPTURE_API_SCHEMA,
            "auth_token": secret,
        }
    )
    return 0


__all__ = ["NATIVE_HOST_NAME", "install_native_host", "main", "native_host_manifest_path"]
