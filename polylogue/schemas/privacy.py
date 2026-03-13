"""Privacy guard logic for schema inference.

Heuristics that determine whether a string value or field path is safe to
include in committed schema annotations.  All functions are intentionally
conservative: anything that could be user content, PII, or an identifier
is rejected.
"""

from __future__ import annotations

import re
from typing import Any


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


def _is_content_field(path: str) -> bool:
    """Return True if a schema path points to a known content field."""
    # Extract terminal field name from dotted path
    terminal = path.rsplit(".", 1)[-1] if "." in path else path
    # Strip array markers like "[*]"
    terminal = terminal.split("[")[0]
    return terminal in _CONTENT_FIELD_NAMES


def _is_safe_enum_value(value: Any, *, path: str = "$") -> bool:
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
    return "." not in value or not re.search(r"\.(local|lan|corp|internal|home)\b", lower)
