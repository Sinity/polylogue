"""Lightweight CLI read-view metadata.

This module is intentionally metadata-only.  It lets ``read --views`` and
Click option ownership checks run without importing executable read-view
handlers, the archive API bridge, or storage/query stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.archive.viewport import read_view_choices

ReadViewSessionPolicy = Literal["optional", "required", "query_or_session", "none"]
ReadViewOptionName = str


MESSAGE_READ_VIEW_OPTION_NAMES = frozenset({"limit", "offset"})
CONTEXT_READ_VIEW_OPTION_NAMES = frozenset({"related_limit"})
CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES = frozenset(
    {
        "max_sessions",
        "no_redact",
        "context_origin",
        "context_query",
        "project_path",
        "project_repo",
        "since",
        "until",
    }
)
NEIGHBOR_READ_VIEW_OPTION_NAMES = frozenset({"limit", "window_hours"})
CORRELATION_READ_VIEW_OPTION_NAMES = frozenset(
    {"confidence_threshold", "github_api", "otlp", "repo_path", "since_hours"}
)
CHRONICLE_READ_VIEW_OPTION_NAMES = frozenset({"limit"})


@dataclass(frozen=True, slots=True)
class ReadViewHandlerMetadata:
    """Executable handler metadata needed by static CLI surfaces."""

    view_id: str
    session_policy: ReadViewSessionPolicy
    accepted_options: frozenset[ReadViewOptionName] = frozenset()
    accepts_query_set: bool = False


READ_VIEW_HANDLER_METADATA: dict[str, ReadViewHandlerMetadata] = {
    "summary": ReadViewHandlerMetadata("summary", "optional", accepts_query_set=True),
    "transcript": ReadViewHandlerMetadata("transcript", "optional", accepts_query_set=True),
    "dialogue": ReadViewHandlerMetadata("dialogue", "required", accepts_query_set=True),
    "messages": ReadViewHandlerMetadata("messages", "required", MESSAGE_READ_VIEW_OPTION_NAMES),
    "raw": ReadViewHandlerMetadata("raw", "required", MESSAGE_READ_VIEW_OPTION_NAMES),
    "context": ReadViewHandlerMetadata("context", "required", CONTEXT_READ_VIEW_OPTION_NAMES),
    "context-image": ReadViewHandlerMetadata("context-image", "none", CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES),
    "neighbors": ReadViewHandlerMetadata("neighbors", "query_or_session", NEIGHBOR_READ_VIEW_OPTION_NAMES),
    "correlation": ReadViewHandlerMetadata("correlation", "required", CORRELATION_READ_VIEW_OPTION_NAMES),
    "temporal": ReadViewHandlerMetadata("temporal", "optional", accepts_query_set=True),
    "chronicle": ReadViewHandlerMetadata(
        "chronicle",
        "optional",
        CHRONICLE_READ_VIEW_OPTION_NAMES,
        accepts_query_set=True,
    ),
}


def read_view_option_names() -> frozenset[ReadViewOptionName]:
    """Return every view-specific option name owned by read-view handlers."""

    return frozenset(
        option_name for metadata in READ_VIEW_HANDLER_METADATA.values() for option_name in metadata.accepted_options
    )


def validate_read_view_metadata_registry() -> None:
    """Fail fast if profile metadata and handler metadata drift."""

    profile_ids = set(read_view_choices())
    metadata_ids = set(READ_VIEW_HANDLER_METADATA)
    missing = sorted(profile_ids - metadata_ids)
    extra = sorted(metadata_ids - profile_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing metadata: {', '.join(missing)}")
        if extra:
            details.append(f"metadata without profiles: {', '.join(extra)}")
        raise RuntimeError("read-view metadata registry drift: " + "; ".join(details))


validate_read_view_metadata_registry()


__all__ = [
    "CHRONICLE_READ_VIEW_OPTION_NAMES",
    "CONTEXT_IMAGE_READ_VIEW_OPTION_NAMES",
    "CONTEXT_READ_VIEW_OPTION_NAMES",
    "CORRELATION_READ_VIEW_OPTION_NAMES",
    "MESSAGE_READ_VIEW_OPTION_NAMES",
    "NEIGHBOR_READ_VIEW_OPTION_NAMES",
    "READ_VIEW_HANDLER_METADATA",
    "ReadViewHandlerMetadata",
    "ReadViewOptionName",
    "ReadViewSessionPolicy",
    "read_view_option_names",
    "validate_read_view_metadata_registry",
]
