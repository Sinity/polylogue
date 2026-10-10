"""Installer and native-messaging host for secure browser-capture pairing."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import secrets
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ijson.common import JSONError

from polylogue.browser_capture.receiver import (
    receiver_attestation_proof,
    receiver_socket_authority,
    receiver_status_proof,
)
from polylogue.core.json import JSONValue
from polylogue.schemas.observation_spill import StreamedJSONDocument, StreamedJSONReadError
from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError

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
        "description": "Polylogue authenticated browser-capture transport",
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


def _authenticate_receiver(connection: http.client.HTTPConnection, receiver_id: str, secret: str) -> str | None:
    """Authenticate and retain this socket without releasing its credential."""
    challenge = secrets.token_urlsafe(32)
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    try:
        assert connection.sock is not None
        try:
            endpoint = receiver_socket_authority(connection.sock.getpeername())
        except OSError:
            return "receiver_transport_namespace_unavailable"
        # No later step may silently create another, unauthenticated socket.
        connection.auto_open = 0
        connection.request(
            "POST",
            "/v1/receiver/attest",
            body=json.dumps({"challenge": challenge}),
            headers=headers,
        )
        response = connection.getresponse()
        if response.status != 200:
            return "receiver_authentication_failed"
        with _receiver_response_document(response) as body:
            if not isinstance(body, dict):
                return "receiver_authentication_failed"
            proof = body.get("proof")
            if (
                isinstance(proof, str)
                and proof.isascii()
                and body.get("receiver_id") == receiver_id
                and body.get("endpoint") == endpoint
                and hmac.compare_digest(proof, receiver_attestation_proof(secret, receiver_id, challenge, endpoint))
                and connection.sock is not None
                and not response.will_close
            ):
                return None
    except (OSError, http.client.HTTPException):
        return "receiver_unreachable"
    except ReceiverObservationStorageError:
        return "receiver_observation_storage_failed"
    except (ValueError, JSONError):
        return "receiver_authentication_failed"
    return "receiver_authentication_failed"


class ReceiverResponseAuthenticationError(ValueError):
    """The received status bytes lack the challenge-bound receiver proof."""


class ReceiverResponsePayloadError(ValueError):
    """The received JSON bytes could not be decoded into a complete document."""


class ReceiverNetworkReadError(OSError):
    """The peer failed while streaming response bytes, before local decoding."""


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
            while True:
                try:
                    chunk = response.read(64 * 1024)
                except (OSError, http.client.HTTPException) as exc:
                    raise ReceiverNetworkReadError("receiver_unreachable") from exc
                if not chunk:
                    break
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
        except (sqlite3.Error, OSError, NativeConnectionSettlementError) as exc:
            raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
        except (ValueError, JSONError) as exc:
            raise ReceiverResponsePayloadError("receiver_status_invalid_payload") from exc
        try:
            try:
                yield document
            except StreamedJSONReadError as exc:
                raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
        finally:
            try:
                owner.__exit__(*sys.exc_info())
            except (sqlite3.Error, OSError, NativeConnectionSettlementError) as exc:
                raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc
    finally:
        response.close()
        try:
            scratch.cleanup()
        except OSError as exc:
            raise ReceiverObservationStorageError("receiver_observation_storage_failed") from exc


def main() -> int:
    from polylogue.browser_capture.native_transport import _send, serve_native_operation
    from polylogue.runtime import require_free_threaded_runtime

    require_free_threaded_runtime(consumer="polylogue browser native host")
    sender = sys.argv[1] if len(sys.argv) > 1 else ""
    extension_id = (
        sender.removeprefix("chrome-extension://").rstrip("/") if sender.startswith("chrome-extension://") else sender
    )
    if not extension_id:
        _send(sys.stdout.buffer, {"type": "error", "error": "native_sender_identity_required"})
        return 1
    return serve_native_operation(extension_id, input_fd=sys.stdin.buffer.fileno(), output=sys.stdout.buffer)


__all__ = ["NATIVE_HOST_NAME", "install_native_host", "main", "native_host_manifest_path"]
