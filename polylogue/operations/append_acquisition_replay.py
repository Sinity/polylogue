"""Read-only replay of normalized live append acquisition units."""

from __future__ import annotations

from io import BytesIO

from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes as json_dumps_bytes
from polylogue.sources.source_acquisition_components import iter_entry_payloads


def replay_append_acquisition_payload(
    payload: bytes,
    *,
    provider: Provider,
    source_name: str,
) -> tuple[bytes | None, str | None]:
    """Reproduce the normalized payload stored for a provider append."""
    if provider is not Provider.CODEX:
        return payload, None
    try:
        records = tuple(iter_entry_payloads(BytesIO(payload), stream_name=source_name, provider_hint=provider))
    except Exception as exc:
        return None, f"append_segment:decode:{exc}"
    if not records:
        return None, "append_segment:decode:no_records"
    return b"".join(json_dumps_bytes(record.payload, append_newline=True) for record in records), None


__all__ = ["replay_append_acquisition_payload"]
