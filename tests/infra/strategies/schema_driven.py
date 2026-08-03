"""Schema-driven Hypothesis strategies for crashlessness testing.

Two generation modes, both driven by the same registry schemas
(polylogue-amrpx):

- ``"adversarial"`` (default, unchanged behaviour): structurally valid but
  distribution-blind data. All ``x-polylogue-*`` inference annotations are
  stripped except ``x-polylogue-values``, which becomes a JSON Schema
  ``enum`` so at least finite value domains are respected.
- ``"distributional"``: additionally threads the subset of inference
  annotations that are (a) actually emitted by the current schema packages
  and (b) expressible as plain JSON Schema constraints without fabricating
  semantics hypothesis-jsonschema doesn't already support:

  - ``x-polylogue-frequency`` (per-field presence probability): fields
    observed present in >= ``HIGH_FREQUENCY_REQUIRED_THRESHOLD`` of sampled
    records are promoted into the enclosing object's ``required`` list, so
    distributional-mode payloads reliably carry near-universal fields that
    adversarial mode may freely omit.
  - ``x-polylogue-range`` ([min, max] observed numeric range): becomes
    ``minimum``/``maximum`` on numeric properties.
  - ``x-polylogue-array-lengths`` ([min, max] observed array length):
    becomes ``minItems``/``maxItems``.
  - ``x-polylogue-format`` (inference-side format token): mapped to the
    subset of standard JSON Schema ``format`` values hypothesis-jsonschema
    actually generates conformant data for (date-time/uri/email); unknown or
    unsupported tokens (uuid4, mime-type, ...) are left unmapped rather than
    guessed at.

  Annotations seen in committed schema packages but NOT threaded here
  (follow-up material, not fabricated): ``x-polylogue-observed-distribution``
  (hash-bucketed histogram, no retained values to sample from),
  ``x-polylogue-string-lengths`` (attached at the document root as a
  JSONPath list rather than per-node, so consuming it needs a path-matching
  pass), ``x-polylogue-mutually-exclusive`` / ``x-polylogue-foreign-keys``
  (cross-field relational constraints, not single-node schema keywords),
  ``x-polylogue-dynamic-keys``, ``x-polylogue-semantic-role``,
  ``x-polylogue-time-deltas``, ``x-polylogue-exact-structure-ids``.
"""

from __future__ import annotations

import copy
from functools import cache
from typing import Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from hypothesis_jsonschema import from_schema

from polylogue.core.json import JSONDocument, JSONValue, json_document, require_json_value
from polylogue.schemas.registry import SchemaRegistry

GenerationMode = Literal["adversarial", "distributional"]

#: Presence frequency (0..1) above which a field is promoted into the
#: enclosing object's ``required`` list in distributional mode. Chosen high
#: enough that it only fires for fields that are, in practice, near-universal
#: on the sampled corpus -- not a tuned statistical threshold.
HIGH_FREQUENCY_REQUIRED_THRESHOLD = 0.9

#: ``x-polylogue-format`` tokens that map onto a standard JSON Schema
#: ``format`` value hypothesis-jsonschema generates conformant strings for.
#: Tokens with no safe standard equivalent (``uuid4``, ``mime-type``, ...)
#: are deliberately left unmapped -- see module docstring.
_FORMAT_TOKEN_MAP: dict[str, str] = {
    "iso8601": "date-time",
    "url": "uri",
    "email": "email",
}


def strip_schema_extensions(
    schema: JSONValue,
    *,
    is_root: bool = True,
    mode: GenerationMode = "adversarial",
) -> JSONValue:
    """Recursively remove generator-hostile keys from a JSON schema.

    ``x-polylogue-values`` is the exception: it is translated into JSON
    Schema ``enum`` before the extension key is stripped so the same finite
    value-domain annotation constrains Hypothesis generation. In
    ``mode="distributional"``, additional annotations documented in the
    module docstring are threaded into standard JSON Schema keywords.
    """
    if isinstance(schema, dict):
        cleaned: JSONDocument = {}
        for key, value in schema.items():
            if key.startswith("x-polylogue-"):
                continue
            if key == "$schema":
                if is_root:
                    cleaned[key] = "https://json-schema.org/draft/2020-12/schema"
                continue
            cleaned[key] = strip_schema_extensions(value, is_root=False, mode=mode)
        enum_values = _polylogue_value_enum(schema)
        if enum_values and "enum" not in cleaned:
            cleaned["enum"] = enum_values
        if mode == "distributional":
            _apply_distributional_annotations(schema, cleaned)
        return cleaned
    if isinstance(schema, list):
        return [strip_schema_extensions(item, is_root=False, mode=mode) for item in schema]
    return schema


def _polylogue_value_enum(schema: JSONDocument) -> list[JSONValue]:
    values = schema.get("x-polylogue-values")
    if not isinstance(values, list):
        return []
    return [require_json_value(value, context="x-polylogue-values item") for value in values]


def _schema_type_includes(schema: JSONDocument, name: str) -> bool:
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared == name
    if isinstance(declared, list):
        return name in declared
    return False


def _numeric_bound_pair(value: JSONValue | None) -> tuple[float, float] | None:
    """Read a ``[min, max]`` numeric pair, or ``None`` if not one.

    Rejects ``bool`` (a ``JSONValue`` numeric-looking subtype of ``int`` that
    is never a legitimate observed bound) and any out-of-order pair.
    """
    if not isinstance(value, list) or len(value) != 2:
        return None
    low, high = value
    if isinstance(low, bool) or isinstance(high, bool):
        return None
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    if low > high:
        return None
    return float(low), float(high)


def _apply_distributional_annotations(schema: JSONDocument, cleaned: JSONDocument) -> None:
    """Thread real inference annotations from ``schema`` into ``cleaned``.

    Reads from the original (uncleaned) ``schema`` node -- the ``x-polylogue-*``
    keys are already stripped out of ``cleaned`` by the time this runs -- and
    only ever *adds* constraints via ``setdefault``, never overriding an
    explicit schema-declared bound.
    """
    if _schema_type_includes(schema, "number") or _schema_type_includes(schema, "integer"):
        numeric_bounds = _numeric_bound_pair(schema.get("x-polylogue-range"))
        if numeric_bounds is not None:
            low, high = numeric_bounds
            cleaned.setdefault("minimum", low)
            cleaned.setdefault("maximum", high)

    if _schema_type_includes(schema, "array"):
        array_bounds = _numeric_bound_pair(schema.get("x-polylogue-array-lengths"))
        if array_bounds is not None:
            low_len, high_len = array_bounds
            cleaned.setdefault("minItems", int(low_len))
            cleaned.setdefault("maxItems", int(high_len))

    format_token = schema.get("x-polylogue-format")
    if isinstance(format_token, str) and _schema_type_includes(schema, "string"):
        mapped_format = _FORMAT_TOKEN_MAP.get(format_token)
        if mapped_format is not None:
            cleaned.setdefault("format", mapped_format)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        high_frequency_fields: set[str] = set()
        for name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue
            frequency = prop_schema.get("x-polylogue-frequency")
            if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
                continue
            if frequency >= HIGH_FREQUENCY_REQUIRED_THRESHOLD:
                high_frequency_fields.add(name)
        if high_frequency_fields:
            existing_required = cleaned.get("required")
            existing: set[str] = set()
            if isinstance(existing_required, list):
                existing = {item for item in existing_required if isinstance(item, str)}
            cleaned["required"] = cast(JSONValue, sorted(existing | high_frequency_fields))


@st.composite
def schema_conformant_payload(
    draw: st.DrawFn,
    provider: str,
    mode: GenerationMode = "adversarial",
) -> JSONValue:
    """Generate a payload conformant to a provider's JSON schema.

    Loads the latest schema for the given provider from the registry,
    strips custom extensions, and uses hypothesis-jsonschema to generate
    conformant data.

    Args:
        provider: Provider name (chatgpt, claude-ai, claude-code, codex, gemini)
        mode: ``"adversarial"`` (default) generates structurally valid but
            distribution-blind data. ``"distributional"`` additionally
            threads the real inference annotations documented in the module
            docstring, so generated payloads approximate observed archive
            distribution instead of only schema conformance.

    Returns:
        A dict/list conformant to the provider's schema.
    """
    return draw(_schema_strategy(provider, mode))


@cache
def _schema_strategy(provider: str, mode: GenerationMode = "adversarial") -> SearchStrategy[JSONValue]:
    registry = SchemaRegistry()
    raw_schema = registry.get_schema(provider, version="latest")
    if raw_schema is None:
        return st.fixed_dictionaries({})
    cleaned = json_document(strip_schema_extensions(copy.deepcopy(raw_schema), mode=mode))
    return from_schema(cleaned)
