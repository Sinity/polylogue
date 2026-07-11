"""First-party browser credential lifecycle for daemon web clients.

The daemon bearer token remains the machine-client credential. Browser clients
receive a separate short-lived, origin-bound credential in an HttpOnly cookie;
only a SHA-256 digest is retained by the daemon. The HTTP contract is shared by
the current shell and the v2 generated client, so neither client needs access to
the secret value.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.cookies import CookieError, SimpleCookie
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

WebCredentialScope = Literal["read", "user_state", "events"]
WebCredentialState = Literal[
    "ready",
    "web_credential_missing",
    "web_credential_invalid",
    "web_credential_expired",
    "web_credential_revoked",
    "web_credential_wrong_origin",
    "web_credential_insufficient_scope",
]
WebCredentialFailureState = Literal[
    "web_credential_missing",
    "web_credential_invalid",
    "web_credential_expired",
    "web_credential_revoked",
    "web_credential_wrong_origin",
    "web_credential_insufficient_scope",
]

WEB_CREDENTIAL_COOKIE = "polylogue_web_credential"
WEB_CREDENTIAL_SCOPES: tuple[WebCredentialScope, ...] = ("events", "read", "user_state")
DEFAULT_WEB_CREDENTIAL_TTL_S = 300
DEFAULT_WEB_CREDENTIAL_MAX_RECORDS = 1024
DEFAULT_WEB_CREDENTIAL_MAX_RECORDS_PER_ORIGIN = 256


class _WebCredentialPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WebCredentialReadyPayload(_WebCredentialPayloadModel):
    """Public lifecycle metadata; the credential itself stays in Set-Cookie."""

    state: Literal["ready"] = "ready"
    expires_at: datetime
    scopes: tuple[WebCredentialScope, ...]


class WebCredentialBootstrapPayload(_WebCredentialPayloadModel):
    ok: Literal[True] = True
    credential: WebCredentialReadyPayload


class WebCredentialRevokedPayload(_WebCredentialPayloadModel):
    state: Literal["web_credential_revoked"] = "web_credential_revoked"


class WebCredentialRevocationPayload(_WebCredentialPayloadModel):
    ok: Literal[True] = True
    credential: WebCredentialRevokedPayload


class WebCredentialFailurePayload(_WebCredentialPayloadModel):
    """Typed subset of the daemon error envelope returned by this contract."""

    ok: Literal[False] = False
    error: WebCredentialFailureState
    detail: str | None = None
    field: str | None = None


@dataclass(frozen=True)
class IssuedWebCredential:
    """One newly-issued credential; ``token`` exists only until Set-Cookie."""

    token: str = field(repr=False)
    origin: str
    scopes: tuple[WebCredentialScope, ...]
    expires_at: float

    def public_payload(self) -> WebCredentialReadyPayload:
        return WebCredentialReadyPayload(
            expires_at=datetime.fromtimestamp(self.expires_at, tz=UTC),
            scopes=self.scopes,
        )


@dataclass(frozen=True)
class WebCredentialDecision:
    allowed: bool
    state: WebCredentialState
    expires_at: float | None = None
    scopes: tuple[WebCredentialScope, ...] = ()

    def response_headers(self) -> dict[str, str]:
        return {
            "Cache-Control": "no-store",
            "X-Polylogue-Web-Credential-State": self.state,
        }


@dataclass
class _CredentialRecord:
    origin: str
    scopes: tuple[WebCredentialScope, ...]
    issued_at: float
    expires_at: float
    revoked: bool = False


def _token_digest(token: str) -> bytes | None:
    """Hash generated ASCII credentials; reject hostile cookie values safely."""

    try:
        encoded = token.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        return None
    return hashlib.sha256(encoded).digest()


def _origin(value: str) -> str | None:
    """Return a canonical HTTP origin, rejecting paths and userinfo."""

    if not value or value == "null":
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        return None
    hostname = parsed.hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = 80 if parsed.scheme == "http" else 443
    authority = hostname if port in {None, default_port} else f"{hostname}:{port}"
    return f"{parsed.scheme}://{authority}"


def _host_authority(value: str) -> tuple[str, int | None] | None:
    if not value:
        return None
    try:
        parsed = urlsplit(f"//{value}")
        port = parsed.port
    except ValueError:
        return None
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None
    return parsed.hostname.lower(), port


def same_origin_from_headers(origin_header: str, host_header: str) -> str | None:
    """Return the request origin only when its authority exactly matches Host."""

    canonical = _origin(origin_header)
    host = _host_authority(host_header)
    if canonical is None or host is None:
        return None
    parsed = urlsplit(canonical)
    origin_port = parsed.port or (80 if parsed.scheme == "http" else 443)
    host_port = host[1] or origin_port
    if parsed.hostname != host[0] or origin_port != host_port:
        return None
    return canonical


def exact_origin_allowed(origin_header: str, host_header: str) -> bool:
    """Browser mutations are admitted only from the daemon's exact authority."""

    return not origin_header or same_origin_from_headers(origin_header, host_header) is not None


def read_web_credential_cookie(cookie_header: str) -> str | None:
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except CookieError:
        return None
    morsel = cookie.get(WEB_CREDENTIAL_COOKIE)
    return morsel.value if morsel is not None and morsel.value else None


def credential_cookie(token: str, *, ttl_s: int, secure: bool = False) -> str:
    attributes = [
        f"{WEB_CREDENTIAL_COOKIE}={token}",
        "Path=/",
        "HttpOnly",
        "SameSite=Strict",
        f"Max-Age={max(0, ttl_s)}",
    ]
    if secure:
        attributes.append("Secure")
    return "; ".join(attributes)


def expired_credential_cookie(*, secure: bool = False) -> str:
    attributes = [
        f"{WEB_CREDENTIAL_COOKIE}=",
        "Path=/",
        "HttpOnly",
        "SameSite=Strict",
        "Max-Age=0",
        "Expires=Thu, 01 Jan 1970 00:00:00 GMT",
    ]
    if secure:
        attributes.append("Secure")
    return "; ".join(attributes)


class WebCredentialRegistry:
    """Thread-safe, process-local registry of digest-only web credentials."""

    def __init__(
        self,
        *,
        ttl_s: int = DEFAULT_WEB_CREDENTIAL_TTL_S,
        max_records: int = DEFAULT_WEB_CREDENTIAL_MAX_RECORDS,
        max_records_per_origin: int = DEFAULT_WEB_CREDENTIAL_MAX_RECORDS_PER_ORIGIN,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("web credential TTL must be positive")
        if max_records <= 0 or max_records_per_origin <= 0:
            raise ValueError("web credential record limits must be positive")
        self.ttl_s = ttl_s
        self.max_records = max_records
        self.max_records_per_origin = min(max_records_per_origin, max_records)
        self._clock = clock
        self._records: dict[bytes, _CredentialRecord] = {}
        self._lock = threading.Lock()

    @property
    def retained_record_count(self) -> int:
        with self._lock:
            return len(self._records)

    def issue(
        self,
        origin: str,
        *,
        previous_token: str | None = None,
        scopes: tuple[WebCredentialScope, ...] = WEB_CREDENTIAL_SCOPES,
    ) -> IssuedWebCredential:
        canonical_origin = _origin(origin)
        if canonical_origin is None:
            raise ValueError("credential origin must be an HTTP origin")
        now = self._clock()
        token = secrets.token_urlsafe(32)
        digest = _token_digest(token)
        if digest is None:  # pragma: no cover - token_urlsafe is ASCII by contract
            raise RuntimeError("generated web credential was not ASCII")
        record = _CredentialRecord(
            origin=canonical_origin,
            scopes=tuple(scope for scope in WEB_CREDENTIAL_SCOPES if scope in scopes),
            issued_at=now,
            expires_at=now + self.ttl_s,
        )
        with self._lock:
            self._prune(now)
            if previous_token:
                previous_digest = _token_digest(previous_token)
                previous = self._records.get(previous_digest) if previous_digest is not None else None
                if previous is not None:
                    previous.revoked = True
            self._records[digest] = record
            self._enforce_limits(canonical_origin)
        return IssuedWebCredential(
            token=token,
            origin=record.origin,
            scopes=record.scopes,
            expires_at=record.expires_at,
        )

    def validate(
        self,
        token: str | None,
        *,
        required_scope: WebCredentialScope,
        host_header: str,
        origin_header: str,
        referer_header: str,
        fetch_site: str,
    ) -> WebCredentialDecision:
        if not token:
            return WebCredentialDecision(False, "web_credential_missing")
        digest = _token_digest(token)
        if digest is None:
            return WebCredentialDecision(False, "web_credential_invalid")
        now = self._clock()
        with self._lock:
            self._prune(now)
            record = self._records.get(digest)
            if record is None:
                return WebCredentialDecision(False, "web_credential_invalid")
            if record.revoked:
                return WebCredentialDecision(
                    False,
                    "web_credential_revoked",
                    expires_at=record.expires_at,
                    scopes=record.scopes,
                )
            if now >= record.expires_at:
                return WebCredentialDecision(
                    False,
                    "web_credential_expired",
                    expires_at=record.expires_at,
                    scopes=record.scopes,
                )
            if not self._origin_matches(
                record.origin,
                host_header=host_header,
                origin_header=origin_header,
                referer_header=referer_header,
                fetch_site=fetch_site,
            ):
                return WebCredentialDecision(
                    False,
                    "web_credential_wrong_origin",
                    expires_at=record.expires_at,
                    scopes=record.scopes,
                )
            if required_scope not in record.scopes:
                return WebCredentialDecision(
                    False,
                    "web_credential_insufficient_scope",
                    expires_at=record.expires_at,
                    scopes=record.scopes,
                )
            return WebCredentialDecision(True, "ready", expires_at=record.expires_at, scopes=record.scopes)

    def revoke(self, token: str | None) -> WebCredentialDecision:
        if not token:
            return WebCredentialDecision(False, "web_credential_missing")
        digest = _token_digest(token)
        if digest is None:
            return WebCredentialDecision(False, "web_credential_invalid")
        now = self._clock()
        with self._lock:
            self._prune(now)
            record = self._records.get(digest)
            if record is None:
                return WebCredentialDecision(False, "web_credential_invalid")
            record.revoked = True
            return WebCredentialDecision(
                True,
                "web_credential_revoked",
                expires_at=record.expires_at,
                scopes=record.scopes,
            )

    def _prune(self, now: float) -> None:
        retention_s = max(self.ttl_s, 60)
        stale = [digest for digest, record in self._records.items() if now >= record.expires_at + retention_s]
        for digest in stale:
            del self._records[digest]

    def _enforce_limits(self, origin: str) -> None:
        origin_records = [digest for digest, record in self._records.items() if record.origin == origin]
        for digest in origin_records[: -self.max_records_per_origin]:
            del self._records[digest]
        while len(self._records) > self.max_records:
            del self._records[next(iter(self._records))]

    @staticmethod
    def _origin_matches(
        issued_origin: str,
        *,
        host_header: str,
        origin_header: str,
        referer_header: str,
        fetch_site: str,
    ) -> bool:
        issued = _origin(issued_origin)
        if issued is None:
            return False
        issued_parts = urlsplit(issued)
        issued_port = issued_parts.port or (80 if issued_parts.scheme == "http" else 443)
        host = _host_authority(host_header)
        if host is None:
            return False
        host_port = host[1] or issued_port
        if host[0] != issued_parts.hostname or host_port != issued_port:
            return False
        if fetch_site and fetch_site not in {"same-origin", "none"}:
            return False
        if origin_header:
            return _origin(origin_header) == issued and same_origin_from_headers(origin_header, host_header) == issued
        if referer_header:
            try:
                referer = urlsplit(referer_header)
            except ValueError:
                return False
            referer_origin = _origin(f"{referer.scheme}://{referer.netloc}")
            return referer_origin == issued
        return fetch_site == "same-origin"


__all__ = [
    "DEFAULT_WEB_CREDENTIAL_MAX_RECORDS",
    "DEFAULT_WEB_CREDENTIAL_MAX_RECORDS_PER_ORIGIN",
    "DEFAULT_WEB_CREDENTIAL_TTL_S",
    "WEB_CREDENTIAL_COOKIE",
    "WEB_CREDENTIAL_SCOPES",
    "IssuedWebCredential",
    "WebCredentialBootstrapPayload",
    "WebCredentialDecision",
    "WebCredentialFailurePayload",
    "WebCredentialFailureState",
    "WebCredentialReadyPayload",
    "WebCredentialRegistry",
    "WebCredentialRevocationPayload",
    "WebCredentialRevokedPayload",
    "WebCredentialScope",
    "WebCredentialState",
    "credential_cookie",
    "exact_origin_allowed",
    "expired_credential_cookie",
    "read_web_credential_cookie",
    "same_origin_from_headers",
]
