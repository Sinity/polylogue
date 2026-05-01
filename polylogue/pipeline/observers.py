"""Run observer primitives for progress and post-run notifications."""

from __future__ import annotations

import http.client
import ipaddress
import re
import shlex
import socket
import ssl
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from polylogue.logging import get_logger
from polylogue.pipeline.run_activity import conversation_activity_counts

if TYPE_CHECKING:
    from polylogue.storage.run_state import RunResult

logger = get_logger(__name__)

# Pattern for detecting shell metacharacters that could enable command injection.
_UNSAFE_PATTERN = re.compile(r"[;&|`(){}[\]<>!\\]|\$\(")


def _conversation_activity(result: RunResult) -> tuple[int, int, int]:
    return conversation_activity_counts(result.counts, getattr(result, "drift", None))


class RunObserver:
    """Base observer for pipeline progress and lifecycle notifications."""

    def on_progress(self, amount: int, desc: str | None = None) -> None:
        return None

    def on_completed(self, result: RunResult) -> None:
        return None

    def on_idle(self, result: RunResult) -> None:
        return None

    def on_error(self, exc: Exception) -> None:
        return None


class CompositeObserver(RunObserver):
    """Dispatches observer notifications to multiple observers in order."""

    __slots__ = ("_observers",)

    def __init__(self, observers: list[RunObserver]) -> None:
        self._observers = observers

    def _dispatch(self, method_name: str, *args: object) -> None:
        for observer in self._observers:
            try:
                getattr(observer, method_name)(*args)
            except Exception:
                logger.exception("Observer %s failed during %s", type(observer).__name__, method_name)

    def on_progress(self, amount: int, desc: str | None = None) -> None:
        self._dispatch("on_progress", amount, desc)

    def on_completed(self, result: RunResult) -> None:
        self._dispatch("on_completed", result)

    def on_idle(self, result: RunResult) -> None:
        self._dispatch("on_idle", result)

    def on_error(self, exc: Exception) -> None:
        self._dispatch("on_error", exc)


class NotificationObserver(RunObserver):
    """Desktop notification via ``notify-send`` (or equivalent)."""

    def on_completed(self, result: RunResult) -> None:
        activity_count, new_count, changed_count = _conversation_activity(result)
        if activity_count <= 0:
            return
        try:
            import subprocess

            detail_parts: list[str] = []
            if new_count:
                detail_parts.append(f"{new_count} new")
            if changed_count:
                detail_parts.append(f"{changed_count} changed")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            subprocess.run(
                ["notify-send", "Polylogue", f"Synced {activity_count} conversation change(s){detail}"],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            pass


def _resolve_and_validate(hostname: str, port: int) -> str:
    """Resolve hostname and return the validated IP address.

    All resolved addresses are checked against the SSRF denylist (private,
    loopback, link-local, reserved). The returned IP is what the actual
    connection will use, closing the DNS-rebinding TOCTOU window between
    validation and connect.
    """
    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve webhook hostname {hostname!r}: {exc}") from exc

    if not addr_infos:
        raise ValueError(f"Cannot resolve webhook hostname {hostname!r}: no addresses")

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"Webhook URL resolves to private/reserved address {ip} "
                f"(hostname: {hostname!r}). This is blocked for SSRF protection."
            )

    # Use the first resolved address — same one we'd pin the connection to.
    address = addr_infos[0][4][0]
    return str(address)


def _validate_webhook_url(url: str) -> None:
    """Validate a webhook URL for SSRF protection (URL form + DNS resolution)."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Webhook URL must use http or https scheme, got: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("Webhook URL must have a hostname")

    _resolve_and_validate(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))


def _webhook_request_target(url: str) -> tuple[str, str, str, int, str]:
    """Validate URL and return ``(scheme, hostname, validated_ip, port, path)``."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Webhook URL must use http or https scheme, got: {parsed.scheme!r}")
    if parsed.hostname is None:
        raise ValueError("Webhook URL must have a hostname")
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    validated_ip = _resolve_and_validate(parsed.hostname, port)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.scheme, parsed.hostname, validated_ip, port, path


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that connects to a validated IP while keeping SNI/cert on the hostname.

    Closes the DNS-rebinding window: ``_resolve_and_validate`` resolves the
    hostname once, validates against the SSRF denylist, and the connection
    is made directly to that IP rather than re-resolving at connect time.
    SNI/cert verification still uses the original hostname.
    """

    def __init__(self, hostname: str, ip: str, *, port: int, timeout: float, context: ssl.SSLContext) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=context)
        self._validated_ip = ip
        self._pinned_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._validated_ip, self.port), timeout=self.timeout)
        self.sock = self._pinned_context.wrap_socket(sock, server_hostname=self.host)


def _post_webhook(url: str, data: bytes) -> None:
    scheme, hostname, validated_ip, port, path = _webhook_request_target(url)
    connection: http.client.HTTPConnection
    if scheme == "https":
        connection = _PinnedHTTPSConnection(
            hostname,
            validated_ip,
            port=port,
            timeout=10,
            context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(validated_ip, port=port, timeout=10)
    try:
        headers = {"Content-Type": "application/json"}
        if scheme == "http":
            # Plain HTTP connects to validated IP; restore Host header so the
            # server routes correctly when name-based virtual-hosted.
            headers["Host"] = hostname if port in (80, 443) else f"{hostname}:{port}"
        connection.request("POST", path, body=data, headers=headers)
        response = connection.getresponse()
        response.read()
        response_status = response.status
        if isinstance(response_status, int) and not (200 <= response_status < 300):
            logger.warning(
                "Webhook returned non-2xx status: %s %s",
                response_status,
                response.reason,
            )
    finally:
        connection.close()


class WebhookObserver(RunObserver):
    """POST to webhook URL when a run produces conversation changes."""

    __slots__ = ("_url",)

    def __init__(self, url: str) -> None:
        _validate_webhook_url(url)
        self._url = url

    def on_completed(self, result: RunResult) -> None:
        activity_count, new_count, changed_count = _conversation_activity(result)
        if activity_count <= 0:
            return
        try:
            import json

            data = json.dumps(
                {
                    "event": "sync",
                    "conversation_activity_count": activity_count,
                    "new_conversations": new_count,
                    "changed_conversations": changed_count,
                }
            ).encode()
            _post_webhook(self._url, data)
        except ValueError as exc:
            logger.warning("Webhook blocked for %s: %s", self._url, exc)
        except Exception as exc:
            logger.warning("Webhook failed for %s: %s", self._url, exc)


def _validate_exec_command(command: str) -> list[str]:
    """Validate and parse an exec command string into a safe argv list."""
    if not command or not command.strip():
        raise ValueError("Exec command cannot be empty")

    if _UNSAFE_PATTERN.search(command):
        raise ValueError(
            f"Exec command contains unsafe shell metacharacters: {command!r}. "
            "Use a simple command without shell operators like ;, &, |, $, backticks, etc."
        )

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Cannot parse exec command {command!r}: {exc}") from exc

    if not argv:
        raise ValueError("Exec command parsed to empty argument list")

    return argv


class ExecObserver(RunObserver):
    """Run a command when a run produces conversation changes."""

    __slots__ = ("_argv",)

    def __init__(self, command: str) -> None:
        self._argv = _validate_exec_command(command)

    def on_completed(self, result: RunResult) -> None:
        activity_count, new_count, changed_count = _conversation_activity(result)
        if activity_count <= 0:
            return
        import os
        import subprocess

        env = os.environ.copy()
        env["POLYLOGUE_ACTIVITY_COUNT"] = str(activity_count)
        env["POLYLOGUE_NEW_CONVERSATION_COUNT"] = str(new_count)
        env["POLYLOGUE_CHANGED_CONVERSATION_COUNT"] = str(changed_count)
        subprocess.run(self._argv, env=env, check=False)


__all__ = [
    "CompositeObserver",
    "ExecObserver",
    "NotificationObserver",
    "RunObserver",
    "WebhookObserver",
    "_validate_exec_command",
    "_validate_webhook_url",
]
