"""Schema generation from provider data.

This module infers JSON schemas from real data samples, which can be used for:
1. Validation of new imports (detect malformed data)
2. Drift detection (warn when provider format changes)
3. Property-based test generation via hypothesis-jsonschema

Can be used as:
- Module: `from polylogue.schemas.schema_inference import generate_provider_schema`
- CLI: `polylogue schema generate --provider chatgpt`
"""

from __future__ import annotations

import contextlib
import gzip
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from genson import SchemaBuilder
    GENSON_AVAILABLE = True
except ImportError:
    GENSON_AVAILABLE = False


from polylogue.lib.raw_payload import (
    build_raw_payload_envelope,
    extract_payload_samples,
    limit_samples,
)
from polylogue.storage.backends.connection import default_db_path

# UUID pattern for detecting dynamic keys
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_HIGH_CARDINALITY_KEY_THRESHOLD = 128
_PATHLIKE_KEY_RATIO_THRESHOLD = 0.35
_STRUCTURE_EXEMPLARS_PER_FINGERPRINT = 8
_FINGERPRINT_MAX_DEPTH = 8
_FINGERPRINT_ARRAY_SAMPLE = 8


@dataclass
class ProviderConfig:
    """Configuration for a provider's schema generation."""

    name: str
    description: str
    db_provider_name: str | None = None  # Provider name in polylogue DB
    session_dir: Path | None = None  # For JSONL session-based providers
    max_sessions: int | None = None
    sample_granularity: str = "document"  # "document" | "record"
    record_type_key: str | None = None  # best-effort stratification key


# Provider configurations
PROVIDERS: dict[str, ProviderConfig] = {
    "chatgpt": ProviderConfig(
        name="chatgpt",
        description="ChatGPT message format",
        db_provider_name="chatgpt",
        sample_granularity="document",
    ),
    "claude-code": ProviderConfig(
        name="claude-code",
        description="Claude Code message format",
        db_provider_name="claude-code",
        sample_granularity="record",
        record_type_key="type",
    ),
    "claude-ai": ProviderConfig(
        name="claude-ai",
        description="Claude AI web message format",
        db_provider_name="claude",  # DB uses "claude"
        sample_granularity="document",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        description="Gemini AI Studio message format",
        db_provider_name="gemini",
        sample_granularity="document",
    ),
    "codex": ProviderConfig(
        name="codex",
        description="OpenAI Codex CLI session format",
        session_dir=Path.home() / ".codex/sessions",
        max_sessions=100,
        sample_granularity="record",
        record_type_key="type",
    ),
}


@dataclass
class GenerationResult:
    """Result of schema generation."""

    provider: str
    schema: dict[str, Any] | None
    sample_count: int
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.schema is not None and self.error is None


# =============================================================================
# Schema Utilities
# =============================================================================


def is_dynamic_key(key: str) -> bool:
    """Check if a key looks like a dynamic identifier (UUID, hash, etc)."""
    if UUID_PATTERN.match(key):
        return True
    if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
        return True
    return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))


def _looks_pathlike_key(key: str) -> bool:
    """Heuristic for key names that likely encode file/user-specific paths."""
    if "/" in key or "\\" in key:
        return True
    if key.count(".") >= 2:
        return True
    return ":" in key and len(key) > 2


def _should_collapse_high_cardinality_keys(keys: list[str]) -> bool:
    """Collapse key-heavy objects that are likely data maps, not fixed schemas."""
    if len(keys) >= _HIGH_CARDINALITY_KEY_THRESHOLD:
        return True
    if len(keys) < 24:
        return False
    pathlike = sum(1 for key in keys if _looks_pathlike_key(key))
    return (pathlike / len(keys)) >= _PATHLIKE_KEY_RATIO_THRESHOLD


def collapse_dynamic_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse dynamic key properties into additionalProperties.

    Handles objects like ChatGPT's `mapping` where keys are UUIDs.
    """
    if not isinstance(schema, dict):
        return schema

    if "properties" in schema:
        props = schema["properties"]
        static_props = {}
        dynamic_schemas = []
        key_names = list(props.keys())

        for key, value in props.items():
            collapsed_value = collapse_dynamic_keys(value)
            if is_dynamic_key(key):
                dynamic_schemas.append(collapsed_value)
            else:
                static_props[key] = collapsed_value

        if _should_collapse_high_cardinality_keys(key_names):
            dynamic_schemas.extend(static_props.values())
            static_props = {}
            schema["x-polylogue-high-cardinality-keys"] = True

        if dynamic_schemas:
            schema["properties"] = static_props
            merged = _merge_schemas(dynamic_schemas)
            schema["additionalProperties"] = merged
            # Signals to drift detection that unknown keys here are expected IDs.
            schema["x-polylogue-dynamic-keys"] = True
            if "required" in schema:
                schema["required"] = [r for r in schema["required"] if r in static_props]
        else:
            schema["properties"] = static_props

    if "items" in schema:
        schema["items"] = collapse_dynamic_keys(schema["items"])

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [collapse_dynamic_keys(s) for s in schema[keyword]]

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        schema["additionalProperties"] = collapse_dynamic_keys(schema["additionalProperties"])

    return schema


def _merge_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple schemas into one using genson."""
    if not GENSON_AVAILABLE:
        return schemas[0] if schemas else {}
    builder = SchemaBuilder()
    for schema in schemas:
        builder.add_schema(schema)
    return dict(builder.to_schema())


def _structure_fingerprint(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = _FINGERPRINT_MAX_DEPTH,
) -> Any:
    """Build a hashable structural fingerprint for schema-dedup heuristics.

    The fingerprint intentionally ignores concrete scalar values and keeps only
    structural/type shape so repeated record variants can be collapsed before
    feeding Genson.
    """
    if depth >= max_depth:
        return ("depth-limit", type(value).__name__)

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool",)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number",)
    if isinstance(value, str):
        return ("string",)

    if isinstance(value, list):
        item_shapes = {
            _structure_fingerprint(item, depth=depth + 1, max_depth=max_depth)
            for item in value[:_FINGERPRINT_ARRAY_SAMPLE]
        }
        return ("array", tuple(sorted(item_shapes, key=repr)))

    if isinstance(value, dict):
        props: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = value[key]
            normalized_key = "*" if is_dynamic_key(key) else key
            props.append(
                (
                    normalized_key,
                    _structure_fingerprint(child, depth=depth + 1, max_depth=max_depth),
                )
            )
        return ("object", tuple(props))

    return ("other", type(value).__name__)


# =============================================================================
# Field Statistics & Schema Annotations
# =============================================================================

# Format detection patterns — ordered by specificity (most specific first)
_FORMAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("uuid4", re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)),
    ("uuid", re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("hex-id", re.compile(r"^[0-9a-f]{24,}$", re.I)),
    ("iso8601", re.compile(
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")),
    ("unix-epoch-str", re.compile(r"^\d{10}(\.\d+)?$")),
    ("url", re.compile(r"^https?://")),
    ("mime-type", re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+\-]*$", re.I)),
    ("base64", re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
]

# Thresholds
_ENUM_MAX_CARDINALITY = 50  # max distinct values to count as enum-like
_ENUM_VALUE_CAP = 200  # max values to track per field (memory bound during collection)
_ENUM_OUTPUT_CAP = 20  # max values to emit in x-polylogue-values (schema size bound)
_ENUM_MIN_COUNT = 2  # suppress one-off values on large corpora (privacy + stability)
_ENUM_MIN_FREQ = 0.03  # suppress extremely rare enum values on large corpora
_REF_MATCH_THRESHOLD = 0.7  # fraction of values that must match keys


@dataclass
class FieldStats:
    """Statistics collected for a single JSON path across all samples."""

    path: str
    observed_values: Counter = field(default_factory=Counter)
    detected_formats: Counter = field(default_factory=Counter)
    num_min: float | None = None
    num_max: float | None = None
    total_samples: int = 0
    present_count: int = 0
    array_lengths: list[int] = field(default_factory=list)
    is_multiline: int = 0  # count of values containing newlines
    value_count: int = 0  # total non-null values seen
    # Maps each observed string value → set of conversation IDs that contained it.
    # Populated only when conversation_ids are supplied to _collect_field_stats.
    # Used to enforce the cross-conversation privacy threshold in _annotate_schema.
    value_conversation_ids: dict[str, set[str]] = field(default_factory=dict)

    @property
    def frequency(self) -> float:
        """Fraction of samples where this field was present."""
        return self.present_count / self.total_samples if self.total_samples else 0.0

    @property
    def dominant_format(self) -> str | None:
        """Most common detected format, if it covers ≥80% of values."""
        if not self.detected_formats or not self.value_count:
            return None
        fmt, count = self.detected_formats.most_common(1)[0]
        if count / self.value_count >= 0.8:
            return fmt
        return None

    @property
    def is_enum_like(self) -> bool:
        """Whether this field has low-cardinality string values."""
        if not self.observed_values:
            return False
        return len(self.observed_values) <= _ENUM_MAX_CARDINALITY


def _detect_string_format(value: str) -> str | None:
    """Detect the format of a string value."""
    if not value or len(value) > 500:
        return None
    for fmt_name, pattern in _FORMAT_PATTERNS:
        if pattern.match(value):
            return fmt_name
    return None


def _detect_numeric_format(value: float | int) -> str | None:
    """Detect whether a numeric value is a Unix epoch timestamp."""
    if isinstance(value, bool):
        return None
    try:
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return None
        # Unix epoch range: 2000-01-01 to 2040-01-01
        if 946684800.0 <= fval <= 2208988800.0:
            return "unix-epoch"
    except (TypeError, ValueError):
        pass
    return None


def _collect_field_stats(
    samples: list[dict[str, Any]],
    *,
    conversation_ids: list[str | None] | None = None,
    max_depth: int = 15,
) -> dict[str, FieldStats]:
    """Walk all samples and collect per-JSON-path statistics.

    Tracks: string value sets (for enum detection), format patterns,
    numeric ranges, field frequency, and array lengths.

    Args:
        samples: Raw data dicts to analyze
        conversation_ids: Optional parallel list of conversation IDs for each sample.
            When provided, each string value is annotated with the conversation(s)
            it appeared in, enabling the cross-conversation privacy threshold in
            _annotate_schema (min_conversation_count).  Length must equal len(samples).
        max_depth: Maximum nesting depth to traverse

    Returns:
        Mapping of JSON path → FieldStats
    """
    all_stats: dict[str, FieldStats] = {}
    # Track all dict key sets for reference detection
    dict_key_sets: dict[str, set[str]] = {}  # path → set of observed keys

    def _ensure_stats(path: str) -> FieldStats:
        if path not in all_stats:
            all_stats[path] = FieldStats(path=path)
        return all_stats[path]

    def _walk(value: Any, path: str, depth: int, sample_idx: int) -> None:
        if depth > max_depth:
            return

        stats = _ensure_stats(path)
        stats.total_samples = max(stats.total_samples, sample_idx + 1)

        if value is None:
            return

        stats.present_count += 1
        stats.value_count += 1

        if isinstance(value, dict):
            # Track keys for reference detection
            if path not in dict_key_sets:
                dict_key_sets[path] = set()
            dict_key_sets[path].update(value.keys())

            # Separate static vs dynamic keys
            static_keys = {}
            dynamic_values = []
            for k, v in value.items():
                if is_dynamic_key(k):
                    dynamic_values.append(v)
                else:
                    static_keys[k] = v

            # Walk static properties
            for k, v in static_keys.items():
                _walk(v, f"{path}.{k}", depth + 1, sample_idx)

            # Walk dynamic properties under wildcard path
            for v in dynamic_values:
                _walk(v, f"{path}.*", depth + 1, sample_idx)

        elif isinstance(value, list):
            stats.array_lengths.append(len(value))
            for _i, item in enumerate(value):
                # Use [*] for array items (not [0], [1] — we want aggregate stats)
                _walk(item, f"{path}[*]", depth + 1, sample_idx)

        elif isinstance(value, str):
            # Track observed values (capped for memory)
            if len(stats.observed_values) < _ENUM_VALUE_CAP:
                stats.observed_values[value] += 1
                # Track which conversation this value came from (for privacy threshold)
                if conversation_ids is not None:
                    conv_id = (
                        conversation_ids[sample_idx]
                        if sample_idx < len(conversation_ids)
                        else None
                    )
                    if conv_id is not None:
                        stats.value_conversation_ids.setdefault(value, set()).add(conv_id)

            # Detect format
            fmt = _detect_string_format(value)
            if fmt:
                stats.detected_formats[fmt] += 1

            # Track multiline content
            if "\n" in value:
                stats.is_multiline += 1

        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            fval = float(value)
            if not (math.isnan(fval) or math.isinf(fval)):
                if stats.num_min is None or fval < stats.num_min:
                    stats.num_min = fval
                if stats.num_max is None or fval > stats.num_max:
                    stats.num_max = fval

                # Detect numeric format (unix epoch)
                fmt = _detect_numeric_format(value)
                if fmt:
                    stats.detected_formats[fmt] += 1

    for idx, sample in enumerate(samples):
        _walk(sample, "$", 0, idx)

    # Fix total_samples for all stats (some paths only seen in subset)
    n = len(samples)
    for stats in all_stats.values():
        stats.total_samples = n

    # Reference detection: for each string field, check if its values
    # are mostly keys in some dict field
    for path, stats in all_stats.items():
        if stats.observed_values:
            observed = set(stats.observed_values.keys())
            if len(observed) > _ENUM_MAX_CARDINALITY:
                # High-cardinality string field — check for references
                for dict_path, keys in dict_key_sets.items():
                    if dict_path == path:
                        continue
                    if not keys:
                        continue
                    overlap = len(observed & keys)
                    if overlap / len(observed) >= _REF_MATCH_THRESHOLD:
                        stats._ref_target = dict_path  # type: ignore[attr-defined]

    return all_stats


_SAFE_ENUM_MAX_LEN = 50  # structural enums are short tokens, not content

_FILE_EXTENSIONS = frozenset({
    ".pdf", ".txt", ".json", ".jpg", ".jpeg", ".png", ".gif",
    ".md", ".html", ".csv", ".tsv", ".doc", ".docx",
    ".xls", ".xlsx", ".zip", ".gz", ".tar", ".py", ".js", ".ts",
})

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]|$)")
_HIGH_ENTROPY_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,}$")

_IDENTIFIER_FIELD_TOKENS = frozenset({
    "id", "ids", "uuid", "guid", "key", "keys", "token", "tokens",
    "hash", "checksum", "digest",
    "resourceid", "fileid", "messageid", "conversationid", "sessionid",
    "promptid", "parentid", "childid", "attachmentid",
    "requestid", "responseid", "traceid", "runid",
    "userid", "threadid", "clientid",
})

# Field names whose values are always user content, never structural enums.
# This complements the value-level filter to catch content that *looks* like
# technical identifiers (e.g. snake_case user titles, domain-like page titles).
_CONTENT_FIELD_NAMES = frozenset({
    "title", "text", "url", "description", "address", "phone",
    "location", "query", "prompt", "summary", "instructions",
    # Free-form message/IO content — never structural
    "body", "message", "input", "output",
    "breadcrumbs", "display_title", "page_title", "leaf_description",
    "clicked_from_title", "clicked_from_url", "content_url", "image_url",
    "website_url", "provider_url", "request_query", "featured_tag",
    "merchants", "price", "evidence_text", "attribution",
    "async_task_title", "serialization_title",
    "branching_from_conversation_title", "branching_from_conversation_owner",
    "country", "owner", "state", "subtitles",
})


def _is_content_field(path: str) -> bool:
    """Return True if a schema path points to a known content field."""
    # Extract terminal field name from dotted path
    terminal = path.rsplit(".", 1)[-1] if "." in path else path
    # Strip array markers like "[*]"
    terminal = terminal.split("[")[0]
    return terminal in _CONTENT_FIELD_NAMES


def _path_field_names(path: str) -> list[str]:
    """Extract concrete field-name segments from a schema path."""
    names: list[str] = []
    for segment in path.split("."):
        if not segment or segment in {"$", "*"}:
            continue
        name = segment.split("[", 1)[0]
        if not name or name == "*":
            continue
        names.append(name)
    return names


def _looks_identifier_field_name(name: str) -> bool:
    """Return True when a field name is likely an identifier slot."""
    if not name:
        return False

    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    if normalized in _IDENTIFIER_FIELD_TOKENS:
        return True

    lowered = name.lower()
    if lowered.endswith(("_id", "-id", "_ids", "-ids")):
        return True
    return bool(name.endswith(("Id", "ID", "Ids", "IDs")))


def _is_identifier_field(path: str) -> bool:
    """Return True if any field segment in path is identifier-like."""
    return any(_looks_identifier_field_name(name) for name in _path_field_names(path))


def _looks_high_entropy_token(value: str) -> bool:
    """Detect opaque identifier-like tokens from value shape alone."""
    if not _HIGH_ENTROPY_TOKEN_RE.match(value):
        return False
    has_alpha = any(ch.isalpha() for ch in value)
    has_digit = any(ch.isdigit() for ch in value)
    if not (has_alpha and has_digit):
        return False
    unique_ratio = len(set(value)) / len(value)
    return unique_ratio >= 0.45


def _is_safe_enum_value(value: str, *, path: str = "$") -> bool:
    """Return True if a string value is safe to include in schema annotations.

    Uses a conservative allowlist approach: only values that look like
    technical identifiers (roles, content types, status codes, MIME types)
    pass through. Anything that could be user content — URLs, filenames,
    natural language text, timestamps, locations — is rejected.

    The goal is to preserve structural enum metadata in committed schemas
    without leaking personal data from conversations.
    """
    if _is_identifier_field(path):
        return False
    if not value or len(value) > _SAFE_ENUM_MAX_LEN:
        return False
    if not value.isascii():
        return False
    if " " in value or "\n" in value:
        return False
    if "://" in value:
        return False
    if "@" in value:
        return False
    if value.startswith(("+", "/")):
        return False
    if _looks_high_entropy_token(value):
        return False
    if "/" in value:
        segments = [part for part in value.split("/") if part]
        if segments and _looks_high_entropy_token(segments[-1]):
            return False
    lower = value.lower()
    if any(lower.endswith(ext) for ext in _FILE_EXTENSIONS):
        return False
    if _TIMESTAMP_RE.match(value):
        return False
    # Likely personal-name token (e.g. "Alice", "JohnSmith") rather than structure.
    if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)*$", value):
        return False
    # Block domain-name-like values (contain dots with known public TLDs)
    if "." in value and re.search(r"\.(com|org|net|pl|io|de|uk|ru|fr|co)\b", lower):
        return False
    # Block internal/private network hostnames (.local, .lan, .corp, .internal, .home)
    if "." in value and re.search(r"\.(local|lan|corp|internal|home)\b", lower):
        return False
    return True


def _annotate_schema(
    schema: dict[str, Any],
    stats: dict[str, FieldStats],
    path: str = "$",
    *,
    min_conversation_count: int = 1,
) -> dict[str, Any]:
    """Apply x-polylogue-* annotations to a schema based on collected field stats.

    Walks the schema tree in parallel with the stats paths. Adds:
    - x-polylogue-values: observed enum values for low-cardinality string fields
    - x-polylogue-format: detected string/numeric format
    - x-polylogue-range: [min, max] for numeric fields
    - x-polylogue-frequency: field presence frequency (omitted if ~1.0)
    - x-polylogue-ref: JSON path this field references
    - x-polylogue-array-lengths: [min, max] for array fields
    - x-polylogue-multiline: true for fields with multiline content
    """
    if not isinstance(schema, dict):
        return schema

    field_stats = stats.get(path)

    # Annotate this node
    if field_stats:
        # Frequency annotation (omit if always present)
        # Skip for array item paths — denominator is item count, not sample count
        if "[*]" not in path:
            freq = field_stats.frequency
            if 0.0 < freq < 0.95:
                schema["x-polylogue-frequency"] = round(freq, 3)

        # Format detection (checked first — format suppresses enum for ID-like fields)
        fmt = field_stats.dominant_format
        if fmt:
            schema["x-polylogue-format"] = fmt

        # String enum values (skip if format indicates generated/unique values,
        # or if the field is known to contain user content)
        _id_formats = {"uuid4", "uuid", "hex-id", "base64"}
        if (field_stats.is_enum_like
                and field_stats.observed_values
                and fmt not in _id_formats
                and not _is_content_field(path)):
            # Top values by frequency — cap output, filter out content/PII.
            # On larger datasets, require repeat observations to avoid leaking
            # one-off values while retaining structural enums.
            total = max(field_stats.value_count, 1)
            min_count = _ENUM_MIN_COUNT if total >= 20 else 1
            min_freq = _ENUM_MIN_FREQ if total >= 50 else 0.0
            sorted_vals: list[Any] = []
            for value, count in field_stats.observed_values.most_common(_ENUM_VALUE_CAP):
                if isinstance(value, str):
                    if not _is_safe_enum_value(value, path=path):
                        continue
                    if count < min_count:
                        continue
                    if min_freq and (count / total) < min_freq:
                        continue
                    # Cross-conversation privacy threshold: suppress values seen in
                    # fewer than min_conversation_count distinct conversations.
                    # Only enforced when conversation tracking is available.
                    if (
                        min_conversation_count > 1
                        and field_stats.value_conversation_ids
                    ):
                        n_convs = len(field_stats.value_conversation_ids.get(value, set()))
                        if n_convs < min_conversation_count:
                            continue
                sorted_vals.append(value)
                if len(sorted_vals) >= _ENUM_OUTPUT_CAP:
                    break
            if sorted_vals:
                schema["x-polylogue-values"] = sorted_vals

        # Numeric range
        if field_stats.num_min is not None and field_stats.num_max is not None:
            schema["x-polylogue-range"] = [field_stats.num_min, field_stats.num_max]

        # Array length range
        if field_stats.array_lengths:
            schema["x-polylogue-array-lengths"] = [
                min(field_stats.array_lengths),
                max(field_stats.array_lengths),
            ]

        # Multiline content
        if field_stats.is_multiline and field_stats.value_count and field_stats.is_multiline / field_stats.value_count > 0.3:
                schema["x-polylogue-multiline"] = True

        # Reference detection
        ref_target = getattr(field_stats, "_ref_target", None)
        if ref_target:
            schema["x-polylogue-ref"] = ref_target

    # Recurse into properties
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            schema["properties"][prop_name] = _annotate_schema(
                prop_schema, stats, f"{path}.{prop_name}",
                min_conversation_count=min_conversation_count,
            )

    # Recurse into additionalProperties (dynamic keys)
    ap = schema.get("additionalProperties")
    if isinstance(ap, dict):
        schema["additionalProperties"] = _annotate_schema(
            ap, stats, f"{path}.*",
            min_conversation_count=min_conversation_count,
        )

    # Recurse into items
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _annotate_schema(
            schema["items"], stats, f"{path}[*]",
            min_conversation_count=min_conversation_count,
        )

    # Recurse into anyOf/oneOf/allOf
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [
                _annotate_schema(s, stats, path, min_conversation_count=min_conversation_count)
                for s in schema[keyword]
            ]

    return schema


# =============================================================================
# Sample Loaders
# =============================================================================


def _resolve_provider_config(provider_name: str) -> ProviderConfig:
    config = next((c for c in PROVIDERS.values() if c.db_provider_name == provider_name), None)
    if config is not None:
        return config
    return ProviderConfig(
        name=provider_name,
        description=f"{provider_name} export format",
        db_provider_name=provider_name,
        sample_granularity="document",
    )


def _iter_samples_from_db(
    provider_name: str,
    *,
    db_path: Path,
    config: ProviderConfig,
) -> Any:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT raw_content, source_path, provider_name
            FROM raw_conversations
            WHERE provider_name = ?
            ORDER BY acquired_at DESC
            """,
            (provider_name,),
        )
        while True:
            rows = cursor.fetchmany(250)
            if not rows:
                break
            for row in rows:
                try:
                    envelope = build_raw_payload_envelope(
                        row[0],
                        source_path=row[1],
                        fallback_provider=row[2],
                        jsonl_dict_only=True,
                    )
                except Exception:
                    continue
                yield from extract_payload_samples(
                    envelope.payload,
                    sample_granularity=config.sample_granularity,
                    max_samples=None,
                    record_type_key=config.record_type_key,
                )
    finally:
        conn.close()


def _iter_samples_from_sessions(
    session_dir: Path,
    *,
    max_sessions: int | None,
) -> Any:
    if not session_dir.exists():
        return

    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(jsonl_files) > max_sessions:
        step = len(jsonl_files) // max_sessions
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            yield parsed
        except OSError:
            continue


def load_samples_from_db(
    provider_name: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw samples from polylogue database."""
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return []

    config = _resolve_provider_config(provider_name)
    samples = list(_iter_samples_from_db(provider_name, db_path=db_path, config=config))
    if max_samples is None:
        return samples
    return limit_samples(
        samples,
        limit=max_samples,
        stratify=config.sample_granularity == "record",
        record_type_key=config.record_type_key,
    )


def load_samples_from_sessions(
    session_dir: Path,
    max_sessions: int | None = None,
    max_samples: int | None = None,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Load samples from JSONL session files."""
    samples = list(_iter_samples_from_sessions(session_dir, max_sessions=max_sessions))
    if max_samples is None:
        return samples
    return limit_samples(
        samples,
        limit=max_samples,
        stratify=True,
        record_type_key=record_type_key,
    )


def get_sample_count_from_db(
    provider_name: str,
    db_path: Path | None = None,
) -> int:
    """Get total message count for a provider in database."""
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("""
            SELECT COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ? AND m.provider_meta IS NOT NULL
        """, (provider_name,)).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


# =============================================================================
# Schema Generation
# =============================================================================


def _remove_nested_required(schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Remove 'required' arrays from nested objects.

    Genson marks fields as required if they appear in all samples, but this
    is too strict for real data where fields can be optional. We keep top-level
    required (depth=0) but remove from nested objects.

    Args:
        schema: JSON schema dict
        depth: Current nesting depth (0 = root)

    Returns:
        Modified schema with nested required arrays removed
    """
    if not isinstance(schema, dict):
        return schema

    # Remove 'required' from nested objects (depth > 0)
    if depth > 0 and "required" in schema:
        del schema["required"]

    # Recurse into properties
    if "properties" in schema:
        for key, prop in schema["properties"].items():
            schema["properties"][key] = _remove_nested_required(prop, depth + 1)

    # Recurse into additionalProperties (e.g. mapping's UUID-keyed nodes)
    ap = schema.get("additionalProperties")
    if isinstance(ap, dict):
        schema["additionalProperties"] = _remove_nested_required(ap, depth + 1)

    # Recurse into items (arrays)
    if "items" in schema:
        schema["items"] = _remove_nested_required(schema["items"], depth + 1)

    # Handle anyOf/oneOf/allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_remove_nested_required(s, depth + 1) for s in schema[key]]

    return schema


def generate_schema_from_samples(
    samples: list[dict[str, Any]],
    *,
    annotate: bool = True,
    max_stats_samples: int = 500,
    max_genson_samples: int | None = None,
) -> dict[str, Any]:
    """Generate JSON schema from samples using genson, with optional annotations.

    Schema inference uses ALL samples for structural completeness.
    Annotation stats collection subsamples to bound computation for large datasets
    (statistical properties like enum values and format detection stabilize well
    before 500 samples).

    Args:
        samples: List of data dicts to infer schema from
        annotate: Whether to add x-polylogue-* annotations from field stats
        max_stats_samples: Max samples for stats collection (0 = use all)
        max_genson_samples: Optional cap for structural inference input.
            Defaults to None (use complete dataset).

    Returns:
        JSON Schema dict with optional annotations
    """
    if not GENSON_AVAILABLE:
        raise ImportError("genson is required for schema generation. Install with: pip install genson")

    if not samples:
        return {"type": "object", "description": "No samples available"}

    # Optional cap for faster debug iterations. Default path uses full dataset.
    genson_samples = samples
    if max_genson_samples and len(samples) > max_genson_samples:
        import random
        rng = random.Random(0)  # Deterministic, different seed from stats sampling
        genson_samples = rng.sample(samples, max_genson_samples)

    builder = SchemaBuilder()
    for sample in genson_samples:
        builder.add_object(sample)

    schema = builder.to_schema()
    schema = collapse_dynamic_keys(schema)

    # Remove required arrays from nested objects - genson is too strict
    schema = _remove_nested_required(schema)

    # Collect field statistics and annotate schema
    if annotate:
        stats_samples = samples
        if max_stats_samples and len(samples) > max_stats_samples:
            import random
            rng = random.Random(42)  # Deterministic for reproducibility
            stats_samples = rng.sample(samples, max_stats_samples)

        field_stats = _collect_field_stats(stats_samples)
        schema = _annotate_schema(schema, field_stats)

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    return schema


def generate_provider_schema(
    provider: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> GenerationResult:
    """Generate schema for a provider.

    Args:
        provider: Provider name (chatgpt, claude-code, etc.)
        db_path: Path to polylogue database
        max_samples: Optional sample limit

    Returns:
        GenerationResult with schema or error
    """
    if db_path is None:
        db_path = default_db_path()
    if not GENSON_AVAILABLE:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error="genson not installed",
        )

    config = PROVIDERS.get(provider)
    if not config:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error=f"Unknown provider: {provider}. Known: {list(PROVIDERS.keys())}",
        )

    try:
        if max_samples is not None:
            # Explicit sampling mode (debug/fast path).
            if config.db_provider_name:
                samples = load_samples_from_db(
                    config.db_provider_name,
                    db_path=db_path,
                    max_samples=max_samples,
                )
            elif config.session_dir:
                samples = load_samples_from_sessions(
                    config.session_dir,
                    max_sessions=config.max_sessions,
                    max_samples=max_samples,
                    record_type_key=config.record_type_key,
                )
            else:
                samples = []

            if not samples:
                return GenerationResult(
                    provider=provider,
                    schema=None,
                    sample_count=0,
                    error="No samples found",
                )

            schema = generate_schema_from_samples(
                samples,
                max_genson_samples=max_samples,
            )
            sample_count = len(samples)
        else:
            # Full-dataset mode (default): stream samples into genson so we
            # don't hold the entire corpus in memory.
            import random

            builder = SchemaBuilder()
            reservoir: list[dict[str, Any]] = []
            reservoir_size = 500
            reservoir_rng = random.Random(42)
            sample_count = 0
            fingerprint_counts: dict[Any, int] = {}

            if config.db_provider_name:
                sample_iter = _iter_samples_from_db(
                    config.db_provider_name,
                    db_path=db_path,
                    config=config,
                )
            elif config.session_dir:
                sample_iter = _iter_samples_from_sessions(
                    config.session_dir,
                    max_sessions=config.max_sessions,
                )
            else:
                sample_iter = iter(())

            for sample in sample_iter:
                sample_count += 1
                fingerprint = _structure_fingerprint(sample)
                seen = fingerprint_counts.get(fingerprint, 0)
                if seen < _STRUCTURE_EXEMPLARS_PER_FINGERPRINT:
                    builder.add_object(sample)
                    fingerprint_counts[fingerprint] = seen + 1

                if len(reservoir) < reservoir_size:
                    reservoir.append(sample)
                else:
                    j = reservoir_rng.randint(0, sample_count - 1)
                    if j < reservoir_size:
                        reservoir[j] = sample

            if sample_count == 0:
                return GenerationResult(
                    provider=provider,
                    schema=None,
                    sample_count=0,
                    error="No samples found",
                )

            schema = builder.to_schema()
            schema = collapse_dynamic_keys(schema)
            schema = _remove_nested_required(schema)
            if config.sample_granularity == "record":
                schema.pop("required", None)
            if reservoir:
                field_stats = _collect_field_stats(reservoir)
                schema = _annotate_schema(schema, field_stats)
            schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

        schema["title"] = f"{provider} export format"
        schema["description"] = config.description

        # Version metadata for drift detection and schema registry
        schema["$id"] = f"polylogue://schemas/{provider}/v1"
        schema["x-polylogue-version"] = 1
        schema["x-polylogue-generated-at"] = datetime.now(tz=timezone.utc).isoformat()
        schema["x-polylogue-sample-count"] = sample_count
        schema["x-polylogue-generator"] = "polylogue.schemas.schema_inference"
        schema["x-polylogue-sample-granularity"] = config.sample_granularity

        return GenerationResult(
            provider=provider,
            schema=schema,
            sample_count=sample_count,
        )
    except Exception as e:
        return GenerationResult(
            provider=provider,
            schema=None,
            sample_count=0,
            error=str(e),
        )


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
) -> list[GenerationResult]:
    """Generate schemas for all (or specified) providers.

    Args:
        output_dir: Directory to write schema files
        db_path: Path to polylogue database
        providers: Optional list of providers (default: all)
        max_samples: Optional sample cap for fast/debug generation.
            None means full dataset.

    Returns:
        List of GenerationResults
    """
    if db_path is None:
        db_path = default_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []

    for provider in provider_list:
        result = generate_provider_schema(
            provider,
            db_path=db_path,
            max_samples=max_samples,
        )
        results.append(result)

        if result.success and result.schema:
            output_path = output_dir / f"{provider}.schema.json.gz"
            compressed = gzip.compress(
                json.dumps(result.schema, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            )
            output_path.write_bytes(compressed)
            # Remove legacy uncompressed file if present
            legacy_path = output_dir / f"{provider}.schema.json"
            if legacy_path.exists():
                legacy_path.unlink()

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================


def cli_main(args: list[str] | None = None) -> int:
    """CLI entry point for schema generation.

    Returns exit code (0 = success).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate JSON schemas from polylogue data"
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()) + ["all"],
        default="all",
        help="Provider to generate schema for",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("polylogue/schemas/providers"),
        help="Output directory for schemas",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to polylogue database (default: XDG data home)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for fast/debug generation (default: full dataset)",
    )

    parsed = parser.parse_args(args)

    providers = None if parsed.provider == "all" else [parsed.provider]
    results = generate_all_schemas(
        output_dir=parsed.output_dir,
        db_path=parsed.db_path,
        providers=providers,
        max_samples=parsed.max_samples,
    )

    # Report results
    success = []
    failed = []
    for r in results:
        if r.success:
            print(f"✓ {r.provider}: {r.sample_count:,} samples")
            success.append(r.provider)
        else:
            print(f"✗ {r.provider}: {r.error}")
            failed.append(r.provider)

    print(f"\nGenerated {len(success)} schemas")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
