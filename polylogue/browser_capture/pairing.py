"""One-time pairing-code bootstrap for the browser-capture receiver.

The receiver's bearer token (:mod:`polylogue.browser_capture.receiver`) is a
long-lived, high-entropy secret; the previous pairing story required the
operator to view it (``token show``) and paste it into the extension popup.
polylogue-gnie's recommended fix is an installer-managed native-messaging
bootstrap, which needs an OS-level host manifest and a stable extension
identity this repo cannot install or exercise headlessly. This module ships
the design's explicit compatibility fallback instead: a short-lived,
single-use pairing *code* (not the token itself) that the operator reads
from a trusted local channel (the ``polylogued browser-capture pairing
start`` CLI output) and types into the extension once. The extension
exchanges the code for the real bearer token over the receiver's own HTTP
surface; the token is never displayed, copied, or typed by the operator.

Threat model (see polylogue-gnie AC #2/#3):

- The exchange endpoint (``POST /v1/pairing/redeem``) is reachable only from
  an allowed extension origin (enforced by the server's existing
  ``_reject_origin`` check, same gate as every other route) or from a
  same-host process sending no ``Origin`` header at all -- an arbitrary
  webpage cannot forge the extension's ``chrome-extension://`` origin.
- The code itself is short-lived (default 180s), single-use (redemption
  clears/consumes it), and rate-limited (5 wrong guesses invalidate it), so
  a local process without the code cannot brute-force it inside its window.
- Only the code's SHA-256 hash is ever persisted to disk; the plaintext code
  exists only in the CLI's stdout and the in-flight HTTPS-equivalent
  loopback request body.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.browser_capture.receiver import load_or_mint_receiver_token
from polylogue.core.json import dumps
from polylogue.core.json import loads as json_loads
from polylogue.paths import browser_capture_pairing_state_path

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Symbols chosen to avoid visually ambiguous characters (0/O, 1/I/L) when an
#: operator reads the code off a terminal and types it into the popup.
PAIRING_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
PAIRING_CODE_LENGTH = 8
PAIRING_CODE_DEFAULT_TTL_SECONDS = 180
PAIRING_CODE_MAX_ATTEMPTS = 5


class PairingCodeError(RuntimeError):
    """Base class for pairing-code redemption failures."""


class PairingCodeInvalidError(PairingCodeError):
    """No pending code, or the supplied code does not match it."""


class PairingCodeExpiredError(PairingCodeError):
    """The pending code's TTL has elapsed."""


class PairingCodeAlreadyUsedError(PairingCodeError):
    """The pending code was already redeemed once."""


class PairingCodeRateLimitedError(PairingCodeError):
    """Too many wrong guesses against the pending code."""


@dataclass(frozen=True, slots=True)
class MintedPairingCode:
    """A freshly minted one-time pairing code."""

    code: str
    expires_at_ms: int
    ttl_seconds: int


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _as_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


@contextmanager
def _pairing_state_lock(target: Path) -> Iterator[None]:
    """Serialize mint/redeem across threads and processes sharing one archive root."""
    import fcntl

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(target.parent / f".{target.name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _read_pairing_state(target: Path) -> dict[str, object] | None:
    if not target.exists():
        return None
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return None
    if not raw.strip():
        return None
    parsed = json_loads(raw)
    return dict(parsed) if isinstance(parsed, dict) else None


def _write_pairing_state(target: Path, record: dict[str, object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(dumps(record))
        os.replace(tmp_path, target)
    except BaseException:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _clear_pairing_state(target: Path) -> None:
    with suppress(FileNotFoundError):
        target.unlink()


def mint_pairing_code(
    *,
    ttl_seconds: int = PAIRING_CODE_DEFAULT_TTL_SECONDS,
    path: Path | None = None,
) -> MintedPairingCode:
    """Mint and persist (hashed) a new one-time pairing code.

    A fresh call always replaces any prior unredeemed code -- only one
    pairing window is live at a time, matching the "run this once, pair
    now" operator flow.
    """
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    target = path if path is not None else browser_capture_pairing_state_path()
    code = "".join(secrets.choice(PAIRING_CODE_ALPHABET) for _ in range(PAIRING_CODE_LENGTH))
    now = _now_ms()
    expires_at_ms = now + ttl_seconds * 1000
    with _pairing_state_lock(target):
        _write_pairing_state(
            target,
            {
                "code_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                "created_at_ms": now,
                "expires_at_ms": expires_at_ms,
                "used_at_ms": None,
                "attempts": 0,
            },
        )
    return MintedPairingCode(code=code, expires_at_ms=expires_at_ms, ttl_seconds=ttl_seconds)


def redeem_pairing_code(
    code: str,
    *,
    path: Path | None = None,
    token_path: Path | None = None,
) -> str:
    """Redeem a one-time pairing code for the receiver's current bearer token.

    Raises one of the :class:`PairingCodeError` subclasses on any failure
    (no pending code, wrong code, expired, already used, or too many wrong
    guesses). A wrong guess still consumes one attempt so a fresh
    ``mint_pairing_code`` call is required after
    :data:`PAIRING_CODE_MAX_ATTEMPTS` misses, even if the correct code has
    not yet expired.
    """
    target = path if path is not None else browser_capture_pairing_state_path()
    normalized = code.strip().upper()
    with _pairing_state_lock(target):
        record = _read_pairing_state(target)
        if record is None:
            raise PairingCodeInvalidError("no pairing code has been started")
        if record.get("used_at_ms") is not None:
            raise PairingCodeAlreadyUsedError("pairing code has already been redeemed")
        attempts = _as_int(record.get("attempts"))
        if attempts >= PAIRING_CODE_MAX_ATTEMPTS:
            raise PairingCodeRateLimitedError("too many pairing attempts; run `pairing start` again")
        if _now_ms() >= _as_int(record.get("expires_at_ms")):
            raise PairingCodeExpiredError("pairing code has expired; run `pairing start` again")
        expected_hash = str(record.get("code_hash") or "")
        candidate_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(candidate_hash, expected_hash):
            record["attempts"] = attempts + 1
            _write_pairing_state(target, record)
            raise PairingCodeInvalidError("pairing code does not match")
        # Single-use: clear the state entirely rather than merely flagging
        # `used_at_ms`, so a stale record can never be replayed even if a
        # caller passes a custom `path` that skips the lock above.
        _clear_pairing_state(target)
    return load_or_mint_receiver_token(token_path)


__all__ = [
    "PAIRING_CODE_DEFAULT_TTL_SECONDS",
    "PAIRING_CODE_MAX_ATTEMPTS",
    "MintedPairingCode",
    "PairingCodeAlreadyUsedError",
    "PairingCodeError",
    "PairingCodeExpiredError",
    "PairingCodeInvalidError",
    "PairingCodeRateLimitedError",
    "mint_pairing_code",
    "redeem_pairing_code",
]
