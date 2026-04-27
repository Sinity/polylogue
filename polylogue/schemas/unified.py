"""Unified provider harmonization — typed adapters with schema-driven fallback.

This module is the public runtime entrypoint for turning provider-native payloads
or extracted provider metadata into a ``HarmonizedMessage``.

Architecture:
- Provider adapters (``unified_adapters.py``) handle per-message extraction
  from already-parsed records (where the parser has extracted messages from
  the wire format)
- Schema extraction (``schemas/extraction.py``) operates at the record level
  and is used by parsers that want generic extraction from raw wire-format data
- Fallback extraction (``unified_fallbacks.py``) handles malformed/partial data
- Provider meta hydration (``unified_provider_meta.py``) handles DB-stored records

The distinction: adapters work on extracted messages (post-parse), schema
extraction works on raw records (pre-parse or during parse).
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from polylogue.lib.json import JSONDocument
from polylogue.lib.provider.semantics import (
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.schemas.unified_adapters import extract_with_adapter
from polylogue.schemas.unified_fallbacks import extract_fallback_message
from polylogue.schemas.unified_models import HarmonizedMessage, _missing_role, extract_token_usage
from polylogue.schemas.unified_provider_meta import (
    _coerce_content_blocks,
    _coerce_reasoning_traces,
    _coerce_tool_calls,
    _extract_generic_cost,
    _extract_generic_tokens,
    _harmonize_extracted_provider_meta,
    _has_extracted_viewports,
    _overlay_message_context,
    bulk_harmonize,
    extract_from_provider_meta,
    harmonize_parsed_message,
    is_message_record,
)
from polylogue.schemas.validator_resolution import resolve_payload_schema
from polylogue.types import Provider

logger = logging.getLogger(__name__)


def extract_harmonized_message(provider: Provider | str, raw: JSONDocument) -> HarmonizedMessage:
    """Extract ``HarmonizedMessage`` from a provider-native message payload.

    This operates on already-extracted message dicts (post-parse), not on
    raw wire-format records. For record-level schema-driven extraction,
    use ``schemas.extraction.extract_message_from_schema`` directly.
    """
    p = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    try:
        return extract_with_adapter(p, raw)
    except (ValidationError, ValueError):
        return extract_fallback_message(p, raw)


def try_schema_extraction(provider: Provider | str, raw: JSONDocument) -> HarmonizedMessage | None:
    """Record-level schema extraction. Returns None if unavailable.

    This is for raw wire-format records (pre-parse). Parsers can call
    this to extract messages from records using schema semantic annotations
    instead of provider-specific Pydantic models.
    """
    p = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    try:
        from polylogue.schemas.extraction import extract_message_from_schema

        _, schema, _ = resolve_payload_schema(p, raw)
        return extract_message_from_schema(raw, schema=schema, provider=p)
    except FileNotFoundError:
        return None
    except Exception:
        return None


__all__ = [
    "HarmonizedMessage",
    "_coerce_content_blocks",
    "_coerce_reasoning_traces",
    "_coerce_tool_calls",
    "_extract_generic_cost",
    "_extract_generic_tokens",
    "_harmonize_extracted_provider_meta",
    "_has_extracted_viewports",
    "_missing_role",
    "_overlay_message_context",
    "bulk_harmonize",
    "extract_chatgpt_text",
    "extract_claude_code_text",
    "extract_content_blocks",
    "extract_from_provider_meta",
    "extract_harmonized_message",
    "extract_reasoning_traces",
    "extract_token_usage",
    "extract_tool_calls",
    "harmonize_parsed_message",
    "is_message_record",
    "try_schema_extraction",
]
