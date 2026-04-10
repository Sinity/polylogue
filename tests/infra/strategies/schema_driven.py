"""Schema-driven Hypothesis strategies for crashlessness testing.

Generates structurally valid but adversarial data from real provider
JSON schemas, stripping x-polylogue-* custom extensions that hypothesis-jsonschema
cannot handle.
"""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema

from polylogue.schemas.registry import SchemaRegistry


def strip_schema_extensions(schema: Any) -> Any:
    """Recursively remove x-polylogue-* keys from a JSON schema."""
    if isinstance(schema, dict):
        return {k: strip_schema_extensions(v) for k, v in schema.items() if not k.startswith("x-polylogue-")}
    if isinstance(schema, list):
        return [strip_schema_extensions(item) for item in schema]
    return schema


@st.composite
def schema_conformant_payload(draw: st.DrawFn, provider: str) -> Any:
    """Generate a payload conformant to a provider's JSON schema.

    Loads the latest schema for the given provider from the registry,
    strips custom extensions, and uses hypothesis-jsonschema to generate
    conformant data.

    Args:
        provider: Provider name (chatgpt, claude-ai, claude-code, codex, gemini)

    Returns:
        A dict/list conformant to the provider's schema.
    """
    registry = SchemaRegistry()
    raw_schema = registry.get_schema(provider, version="latest")
    if raw_schema is None:
        # Fall back — draw a minimal valid dict
        return draw(st.fixed_dictionaries({}))
    cleaned = strip_schema_extensions(copy.deepcopy(raw_schema))
    return draw(from_schema(cleaned))
