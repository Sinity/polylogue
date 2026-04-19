"""Runtime entry points for provider-meta harmonization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, cast

from pydantic import ValidationError

from polylogue.lib.raw_payload_decode import JSONRecord
from polylogue.schemas.unified_adapters import extract_with_adapter
from polylogue.schemas.unified_fallbacks import extract_fallback_message
from polylogue.schemas.unified_models import HarmonizedMessage
from polylogue.schemas.unified_provider_meta_coercion import _has_extracted_viewports
from polylogue.schemas.unified_provider_meta_harmonize import (
    _harmonize_extracted_provider_meta,
    _overlay_message_context,
)
from polylogue.types import Provider


def _json_record(value: object) -> JSONRecord:
    return cast(JSONRecord, value) if isinstance(value, dict) else {}


def _provider_meta_dict(value: Mapping[str, object]) -> dict[str, Any]:
    return dict(value)


def extract_from_provider_meta(
    provider: Provider | str,
    provider_meta: Mapping[str, object],
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage:
    """Extract HarmonizedMessage from polylogue database format."""
    resolved_provider = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    raw = provider_meta.get("raw")
    if raw is not None:
        raw_record = _json_record(raw)
        try:
            harmonized = extract_with_adapter(resolved_provider, raw_record)
        except (ValidationError, ValueError):
            harmonized = extract_fallback_message(resolved_provider, dict(raw_record))
        return _overlay_message_context(
            harmonized,
            message_id=message_id,
            role=role,
            text=text,
            timestamp=timestamp,
        )

    if _has_extracted_viewports(dict(provider_meta)):
        return _harmonize_extracted_provider_meta(
            resolved_provider,
            _provider_meta_dict(provider_meta),
            message_id=message_id,
            role=role,
            text=text,
            timestamp=timestamp,
        )

    try:
        harmonized = extract_with_adapter(resolved_provider, _json_record(provider_meta))
    except (ValidationError, ValueError, TypeError):
        return _harmonize_extracted_provider_meta(
            resolved_provider,
            _provider_meta_dict(provider_meta),
            message_id=message_id,
            role=role,
            text=text,
            timestamp=timestamp,
        )

    return _overlay_message_context(
        harmonized,
        message_id=message_id,
        role=role,
        text=text,
        timestamp=timestamp,
    )


def is_message_record(provider: Provider | str, raw: Mapping[str, object]) -> bool:
    """Check if a record is an actual message (vs metadata)."""
    resolved_provider = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    if resolved_provider == Provider.CLAUDE_CODE:
        record_type = raw.get("type")
        if record_type is None:
            return True
        return record_type in ("user", "assistant", "system")
    return True


def harmonize_parsed_message(
    provider: str,
    provider_meta: Mapping[str, object] | None,
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage | None:
    """Convert ParsedMessage.provider_meta to HarmonizedMessage."""
    if not provider_meta:
        return None

    raw = provider_meta.get("raw", provider_meta)
    if not is_message_record(provider, _json_record(raw)):
        return None

    return extract_from_provider_meta(
        provider,
        provider_meta,
        message_id=message_id,
        role=role,
        text=text,
        timestamp=timestamp,
    )


def bulk_harmonize(
    provider: str,
    parsed_messages: Iterable[object],
) -> list[HarmonizedMessage]:
    """Bulk convert ParsedMessages to HarmonizedMessages."""
    results: list[HarmonizedMessage] = []
    for parsed_message in parsed_messages:
        meta = getattr(parsed_message, "provider_meta", None)
        if not isinstance(meta, Mapping) or not meta:
            continue
        harmonized = harmonize_parsed_message(
            provider,
            meta,
            message_id=getattr(parsed_message, "provider_message_id", None),
            role=getattr(parsed_message, "role", None),
            text=getattr(parsed_message, "text", None),
            timestamp=getattr(parsed_message, "timestamp", None),
        )
        if harmonized:
            results.append(harmonized)
    return results


__all__ = [
    "bulk_harmonize",
    "extract_from_provider_meta",
    "harmonize_parsed_message",
    "is_message_record",
]
