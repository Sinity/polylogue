"""Schema-driven Hypothesis strategies for crashlessness testing.

Generates structurally valid but adversarial data from real provider
JSON schemas, stripping test-irrelevant extensions and nested legacy
metaschema declarations that upset ``hypothesis-jsonschema``.
"""

from __future__ import annotations

import copy

from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema

from polylogue.lib.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.registry import SchemaRegistry


def strip_schema_extensions(schema: JSONValue, *, is_root: bool = True) -> JSONValue:
    """Recursively remove generator-hostile keys from a JSON schema."""
    if isinstance(schema, dict):
        cleaned: JSONDocument = {}
        for key, value in schema.items():
            if key.startswith("x-polylogue-"):
                continue
            if key == "$schema":
                if is_root:
                    cleaned[key] = "https://json-schema.org/draft/2020-12/schema"
                continue
            cleaned[key] = strip_schema_extensions(value, is_root=False)
        return cleaned
    if isinstance(schema, list):
        return [strip_schema_extensions(item, is_root=False) for item in schema]
    return schema


@st.composite
def schema_conformant_payload(draw: st.DrawFn, provider: str) -> JSONValue:
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
    cleaned = json_document(strip_schema_extensions(copy.deepcopy(raw_schema)))
    return draw(from_schema(cleaned))
