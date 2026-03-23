"""Unified provider harmonization over typed adapters and explicit fallback paths.

This module is the public runtime entrypoint for turning provider-native payloads
or extracted provider metadata into a ``HarmonizedMessage``.

The implementation is intentionally split behind this API:

- ``unified_models`` owns the harmonized runtime model and token helpers
- ``unified_adapters`` owns typed provider-adapter routing
- ``unified_fallbacks`` owns malformed/rawless fallback extraction
- ``unified_provider_meta`` owns DB/provider-meta hydration and parsed-message flow
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from polylogue.lib.provider_semantics import (
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
from polylogue.types import Provider


def extract_harmonized_message(provider: Provider | str, raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract ``HarmonizedMessage`` from raw provider data."""
    p = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    try:
        return extract_with_adapter(p, raw)
    except (ValidationError, ValueError):
        return extract_fallback_message(p, raw)


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
]
