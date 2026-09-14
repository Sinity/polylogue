"""The structured-log field allowlist — the PII boundary, as data.

This module is deliberately dependency-free: it must be importable from any
tier without creating a cycle, and its contents are the single place a
reviewer looks to answer "can a transcript reach the log?".

The boundary is an **allowlist**, not a denylist. A field name that is not
registered here cannot be emitted: :func:`polylogue.logging.emit` drops it and
reports the drop as its own event. Adding a field is therefore a deliberate,
reviewable act, and no amount of call-site carelessness can leak session
content through a name nobody approved.

Two consequences worth stating plainly:

* Content-bearing names (``text``, ``content``, ``body``, ``prompt``,
  ``search_text``, ``payload`` ...) are absent, and :data:`FORBIDDEN_FIELDS`
  additionally names them so the drop event can say *why* rather than merely
  "unknown field".
* Exactly one field, ``error_detail``, carries free text. It is truncated and
  it is the only field a redacting renderer needs to remove. See
  :data:`QUARANTINED_FIELDS`.
"""

from __future__ import annotations

from typing import Final, Literal

FieldKind = Literal[
    "identifier",  # opaque id or stable token: session_id, run_id, origin
    "count",  # non-negative integer measurement
    "duration",  # elapsed seconds/milliseconds
    "flag",  # boolean
    "token",  # closed vocabulary member: outcome, level, stage
    "path",  # filesystem path — local-only, see SENSITIVITY notes
    "text",  # quarantined free text, truncated on emit
]

#: Free-text fields. These are truncated to :data:`TEXT_FIELD_MAX_CHARS` and are
#: the only fields a redacting renderer must strip. Keep this set at one entry.
QUARANTINED_FIELDS: Final[frozenset[str]] = frozenset({"error_detail"})

#: Fields that name a local filesystem location. Not session content, but
#: operator-identifying, so exported logs may want them relativized.
LOCAL_ONLY_FIELDS: Final[frozenset[str]] = frozenset({"path", "root", "log_file", "db_path"})

TEXT_FIELD_MAX_CHARS: Final[int] = 300

#: Names that are rejected *by name* with an explicit reason, so a call site
#: that tries to log content gets a pointed drop event instead of a vague one.
#: Membership here is belt-and-braces: none of these are in FIELDS either.
FORBIDDEN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "text",
        "content",
        "body",
        "prompt",
        "completion",
        "message",
        "message_text",
        "search_text",
        "payload",
        "raw",
        "raw_bytes",
        "blob",
        "data",
        "value",
        "args",
        "kwargs",
        "response",
        "request_body",
        "token_value",
        "secret",
        "password",
        "api_key",
        "authorization",
        "email",
        "user",
        "username",
        "home",
    }
)


def _fields() -> dict[str, FieldKind]:
    """Build the registry. Grouped by concern for review, flattened for lookup."""
    registry: dict[str, FieldKind] = {}

    def add(kind: FieldKind, *names: str) -> None:
        for name in names:
            registry[name] = kind

    # -- correlation -----------------------------------------------------
    # These carry a unit of work across threads, awaits and the writer lease.
    add(
        "identifier",
        "trace_id",
        "span_id",
        "parent_span_id",
        "run_id",
        "pass_id",
        "operation_id",
        "request_id",
    )

    # -- subject identity ------------------------------------------------
    # Opaque archive identifiers. A session_id is origin:native_id — a handle,
    # never content.
    add(
        "identifier",
        "session_id",
        "message_id",
        "block_id",
        "raw_id",
        "artifact_id",
        "blob_hash",
        "content_hash",
        "source_id",
        "member_id",
        "cursor_id",
        "tool_id",
        "branch_point_message_id",
    )

    # -- classification tokens -------------------------------------------
    add(
        "token",
        "event",
        "level",
        "outcome",
        "origin",
        "provider",
        "source_kind",
        "stage",
        "domain",
        "actor",
        "component",
        "tier",
        "reason",
        "error_type",
        "identity_source",
        "inheritance_mode",
        "tool_outcome",
        "state",
        "phase",
        "backend",
        "logger",
        "thread",
        "method",
        "route",
        # meta: used by log.field_rejected to name the offending field
        "field",
        "source_event",
    )

    # -- measurements ----------------------------------------------------
    add(
        "count",
        "sessions",
        "messages",
        "blocks",
        "raws",
        "files",
        "considered",
        "ingested",
        "skipped",
        "failed",
        "deferred",
        "pending",
        "retried",
        "queued",
        "active",
        "bytes",
        "rows",
        "attempts",
        "errors",
        "warnings",
        "debt",
        "backlog",
        "position",
        "limit",
        "status_code",
        "exit_code",
        "pid",
    )

    add("duration", "duration_ms", "elapsed_ms", "timeout_ms", "age_ms")

    add("flag", "ok", "changed", "cached", "dry_run", "forced", "degraded", "converged", "held")

    add("path", *sorted(LOCAL_ONLY_FIELDS))

    add("text", *sorted(QUARANTINED_FIELDS))

    return registry


#: name -> kind. The complete set of emittable field names.
FIELDS: Final[dict[str, FieldKind]] = _fields()

#: Closed vocabulary for the ``outcome`` field. ``unmeasured`` exists so a
#: probe that never produced an answer can say so instead of defaulting to
#: success — the campaign's recurring defect.
OUTCOMES: Final[frozenset[str]] = frozenset({"ok", "empty", "degraded", "error", "refused", "unmeasured", "skipped"})


def field_kind(name: str) -> FieldKind | None:
    """Return the registered kind for ``name``, or ``None`` if unregistered."""
    return FIELDS.get(name)


def rejection_reason(name: str) -> str | None:
    """Explain why ``name`` may not be emitted, or ``None`` if it may."""
    if name in FIELDS:
        return None
    if name in FORBIDDEN_FIELDS:
        return "content_field"
    return "unregistered_field"
