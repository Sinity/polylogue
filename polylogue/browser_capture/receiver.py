"""Local-only browser-capture receiver helpers."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import orjson

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.lib.hashing import hash_text_short
from polylogue.paths import browser_capture_spool_root

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class BrowserCaptureReceiverConfig:
    """Configuration for the localhost browser-capture receiver."""

    spool_path: Path
    allowed_origins: frozenset[str] = frozenset(
        {
            "https://chatgpt.com",
            "https://claude.ai",
        }
    )

    @classmethod
    def default(cls) -> BrowserCaptureReceiverConfig:
        return cls(spool_path=browser_capture_spool_root())


@dataclass(frozen=True, slots=True)
class BrowserCaptureWriteResult:
    """Result of accepting a browser-capture envelope."""

    provider: str
    provider_session_id: str
    path: Path
    bytes_written: int
    replaced: bool


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN.sub("-", value.strip()).strip(".-")
    return token[:96] if token else "session"


def capture_artifact_path(envelope: BrowserCaptureEnvelope, spool_path: Path | None = None) -> Path:
    """Return the deterministic source artifact path for an envelope."""
    root = spool_path if spool_path is not None else BrowserCaptureReceiverConfig.default().spool_path
    provider = _safe_token(envelope.provider.value)
    session = _safe_token(envelope.provider_session_id)
    suffix = hash_text_short(f"{envelope.provider.value}:{envelope.provider_session_id}", 12)
    return root / provider / f"{session}-{suffix}.json"


def write_capture_envelope(
    envelope: BrowserCaptureEnvelope,
    *,
    spool_path: Path | None = None,
) -> BrowserCaptureWriteResult:
    """Atomically write a browser-capture source artifact into the capture spool."""
    target = capture_artifact_path(envelope, spool_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = envelope.model_dump(mode="json", exclude_none=True)
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    replaced = target.exists()
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(raw)
        handle.write(b"\n")
    temp_path.replace(target)
    return BrowserCaptureWriteResult(
        provider=envelope.provider.value,
        provider_session_id=envelope.provider_session_id,
        path=target,
        bytes_written=target.stat().st_size,
        replaced=replaced,
    )


def receiver_status_payload(config: BrowserCaptureReceiverConfig) -> dict[str, object]:
    """Return JSON status for extension health checks."""
    return {
        "ok": True,
        "receiver": "polylogue-browser-capture",
        "schema_version": 1,
        "spool_path": str(config.spool_path),
        "allowed_origins": sorted(config.allowed_origins),
        "checked_at": datetime.now(UTC).isoformat(),
    }


def existing_capture_state(
    provider: str,
    provider_session_id: str,
    *,
    spool_path: Path | None = None,
) -> dict[str, object]:
    """Return the local capture state visible to the browser extension."""
    envelope = BrowserCaptureEnvelope.model_validate(
        {
            "provenance": {
                "source_url": "about:blank",
                "captured_at": "1970-01-01T00:00:00+00:00",
                "adapter_name": "state-lookup",
            },
            "session": {
                "provider": provider,
                "provider_session_id": provider_session_id,
                "turns": [{"provider_turn_id": "lookup", "role": "system", "text": "lookup"}],
            },
        }
    )
    path = capture_artifact_path(envelope, spool_path)
    state: dict[str, object] = {
        "provider": envelope.provider.value,
        "provider_session_id": envelope.provider_session_id,
        "captured": path.exists(),
        "artifact_path": str(path),
    }
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            state["capture_id"] = payload.get("capture_id")
            state["updated_at"] = payload.get("session", {}).get("updated_at")
        except (OSError, json.JSONDecodeError, AttributeError):
            state["artifact_readable"] = False
    return state


__all__ = [
    "BrowserCaptureReceiverConfig",
    "BrowserCaptureWriteResult",
    "capture_artifact_path",
    "existing_capture_state",
    "receiver_status_payload",
    "write_capture_envelope",
]
