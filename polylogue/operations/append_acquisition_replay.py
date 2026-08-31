"""Read-only replay of live append acquisition units."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import loads as json_loads
from polylogue.sources.live.batch_support import codex_append_payload


def _codex_session_meta_id(source_name: str) -> str | None:
    try:
        with Path(source_name).open("rb") as handle:
            record = json_loads(handle.readline())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(record, dict) or record.get("type") != "session_meta":
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    identity = payload.get("id")
    return identity if isinstance(identity, str) and identity else None


def codex_legacy_header_size(source_name: str) -> int | None:
    """Return the historical Codex append header size for one source file."""
    identity = _codex_session_meta_id(source_name)
    if identity is None:
        return None
    return len(codex_append_payload(b"", identity=identity, legacy_header=True))


def replay_append_acquisition_payload(
    payload: bytes,
    *,
    provider: Provider,
    source_name: str,
    expected_size: int | None = None,
) -> tuple[bytes | None, str | None]:
    """Reproduce the payload stored for a provider append.

    Historical Codex append rows include a compact synthetic ``session_meta``
    header. ``expected_size`` selects that historical shape when the caller
    has the recorded row size; the no-size form is retained for direct replay
    of historical rows and therefore uses the header when the source exposes
    a session identity. New literal-delta rows select the literal shape by
    their recorded size.
    """
    if provider is not Provider.CODEX:
        return payload, None
    identity = _codex_session_meta_id(source_name)
    if identity is None:
        return payload, None
    historical = codex_append_payload(payload, identity=identity, legacy_header=True)
    if expected_size is None or expected_size == len(historical):
        return historical, None
    if expected_size == len(payload):
        return codex_append_payload(payload, identity=identity, legacy_header=False), None
    return None, "append_segment:recorded_size_mismatch"


__all__ = ["codex_legacy_header_size", "replay_append_acquisition_payload"]
