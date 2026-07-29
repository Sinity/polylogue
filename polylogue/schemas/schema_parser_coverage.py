"""Which committed-schema fields does no parser module read (polylogue-2qx.3)?

The committed schema packages under ``polylogue/schemas/providers`` record
every field a provider actually emits. The parsers decide what becomes
archive content. This module answers, per provider, "which schema-declared
field names does nothing in that provider's parser modules reference?" --
the static join the drift sentinel's fourth classification needs to tell
"schema knows this field, the parser ignores it" apart from genuine no-drift.

DELIBERATELY APPROXIMATE, same as any name-based static check: a field
counts as "referenced" if its name appears as a string constant anywhere in
the provider's parser modules (a subscript, a ``.get()``, a comparison, a
membership test). That over-counts (a name can appear without being read
into archive content, or resolve by coincidence via an unrelated string
literal sharing the name) and can under-count (a key read through a
variable, a shared helper, or captured inside a verbatim-stored parent
object is missed). Treat the output as evidence for triage, not a verdict --
this mirrors ``devtools/schema_parser_diff.py``'s own posture towards the
same join (both are stated goals of polylogue-2qx.3; keep their
``PROVIDER_PARSERS`` mappings in sync).
"""

from __future__ import annotations

import ast
import gzip
import json
from collections.abc import Collection
from functools import cache
from pathlib import Path

_SCHEMA_ROOT = Path(__file__).resolve().parent / "providers"
_PARSER_ROOT = Path(__file__).resolve().parents[1] / "sources" / "parsers"

#: Provider schema directory -> the parser modules that own its wire format.
#: A key referenced in any listed module counts as referenced.
#:
#: ``local_agent.py`` is shared: it defines both ``parse_gemini_cli`` and
#: ``parse_hermes`` (hermes's mainstream JSON-snapshot shape is parsed there
#: too, alongside the dedicated ``hermes_*.py`` modules), so it is listed for
#: both providers -- mapping it to gemini-cli alone produced false-positive
#: "unread" rows for hermes, including ``last_updated`` (hermes triage lane,
#: 2026-07-29).
PROVIDER_PARSERS: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude/code_parser.py", "claude/common.py", "claude/history.py", "claude/orchestration.py"),
    "claude-ai": ("claude/ai_parser.py", "claude/common.py"),
    "codex": ("codex.py",),
    "chatgpt": ("chatgpt.py",),
    "gemini-cli": ("local_agent.py",),
    "gemini": ("drive.py", "drive_support.py", "drive_support_blocks.py", "drive_support_text.py"),
    "hermes": ("hermes_state.py", "hermes_spans.py", "hermes_lifecycle.py", "hermes_verification.py", "local_agent.py"),
    "antigravity": ("antigravity.py",),
    "browser-capture": ("browser_capture.py",),
}


def _schema_property_names(schema: object, *, names: set[str]) -> None:
    """Collect every named property anywhere in a JSON Schema document."""
    if not isinstance(schema, dict):
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            names.add(str(name))
            _schema_property_names(child, names=names)
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                _schema_property_names(branch, names=names)
    items = schema.get("items")
    if isinstance(items, dict):
        _schema_property_names(items, names=names)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _schema_property_names(additional, names=names)


def _schema_documents(provider_dir: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for path in sorted(provider_dir.glob("versions/*/elements/*.schema.json.gz")):
        try:
            payload = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            documents.append(payload)
    return documents


def _referenced_strings(paths: list[Path]) -> frozenset[str]:
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
    return frozenset(names)


@cache
def schema_known_field_names(provider: str, *, schema_root: Path | None = None) -> frozenset[str]:
    """Every field name the committed schema packages declare for ``provider``."""
    provider_dir = (schema_root or _SCHEMA_ROOT) / provider
    if not provider_dir.exists():
        return frozenset()
    names: set[str] = set()
    for schema in _schema_documents(provider_dir):
        _schema_property_names(schema, names=names)
    return frozenset(names)


@cache
def parser_referenced_field_names(provider: str, *, parser_root: Path | None = None) -> frozenset[str]:
    """Every string constant appearing in ``provider``'s parser modules."""
    root = parser_root or _PARSER_ROOT
    modules = [root / module for module in PROVIDER_PARSERS.get(provider, ())]
    return _referenced_strings(modules)


def unread_field_names(
    provider: str,
    *,
    schema_root: Path | None = None,
    parser_root: Path | None = None,
) -> frozenset[str]:
    """Schema-declared field names for ``provider`` that no parser module reads."""
    known = schema_known_field_names(provider, schema_root=schema_root)
    if not known:
        return frozenset()
    referenced = parser_referenced_field_names(provider, parser_root=parser_root)
    return known - referenced


def payload_unread_field_names(
    provider: str,
    payload_field_names: Collection[object],
    *,
    schema_root: Path | None = None,
    parser_root: Path | None = None,
) -> list[str]:
    """Intersect one payload's own field names with ``provider``'s unread set.

    ``payload_field_names`` is typically the top-level (or otherwise
    shallow) keys of one already-in-memory validated sample -- cheap to
    compute at ingest time, unlike re-deriving the schema/parser sets
    themselves (which are filesystem reads, cached per provider via
    ``functools.cache``).
    """
    if not payload_field_names:
        return []
    unread = unread_field_names(provider, schema_root=schema_root, parser_root=parser_root)
    if not unread:
        return []
    return sorted(str(name) for name in payload_field_names if str(name) in unread)


def clear_cache() -> None:
    """Reset cached per-provider schema/parser lookups (tests, hot-reload)."""
    schema_known_field_names.cache_clear()
    parser_referenced_field_names.cache_clear()


__all__ = [
    "PROVIDER_PARSERS",
    "clear_cache",
    "parser_referenced_field_names",
    "payload_unread_field_names",
    "schema_known_field_names",
    "unread_field_names",
]
