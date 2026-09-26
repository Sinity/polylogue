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
    "timings",  # bounded phase-name -> finite milliseconds
    "epoch",  # finite Unix timestamp
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
        "source_name",
        "member_id",
        "cursor_id",
        "tool_id",
        "generation_id",
        "branch_point_message_id",
        "derivation_key",
        "candidate_ref",
        "assertion_id",
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
        "effect",
        "operation",
        "thread",
        "method",
        "route",
        # daemon periodic maintenance loop name ("wal checkpoint", "fts merge")
        "loop",
        # operation mode within a stage ("PASSIVE" for a WAL checkpoint)
        "mode",
        # sub-kind of a recorded event or projection ("profile", "cost")
        "kind",
        "check_name",
        "severity",
        "status",
        "action",
        "policy",
        "service",
        "family",
        "signal_name",
        "table_name",
        "schema_name",
        # meta: used by log.field_rejected to name the offending field
        "field",
        # comma-joined configuration key names a layer refused. Drawn from a
        # closed set of declared setting names (never a user value), so the
        # refusal event can say *which* keys it ignored instead of reporting
        # only that some were.
        "config_keys",
        "source_event",
        "evidence",
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
        "succeeded",
        "refused",
        "deferred",
        "pending",
        "retried",
        "queued",
        "active",
        "bytes",
        "size",
        "rows",
        "attempts",
        "errors",
        "warnings",
        "reclaimed",
        "debt",
        "backlog",
        "position",
        "limit",
        "status_code",
        "exit_code",
        "pid",
        # reconciliation dispositions: rows cleared, kept for inspection, or
        # left classified-but-undecided.
        "cleared",
        "retained",
        "unresolved",
        # WAL checkpoint: frames before/after, pages blocked by a reader, and
        # pages actually written back.
        "bytes_before",
        "bytes_after",
        "busy_pages",
        "checkpointed_pages",
        # fan-out sizes a daemon pass reports about itself.
        "tiers",
        "sources",
        "loops",
        "services",
        "orphaned",
        "alerts",
        "delivered",
        # bounded queue/spool depth
        "depth",
        "admitted",
        "accepted",
        "rejected",
        "escalated",
        "idempotent",
        "duplicates",
        "isolated",
        "attempted",
        "confirmed",
        "remaining",
        "subjects",
        "candidates",
        "scanned",
        "computed",
        "dropped",
        "removed",
        "repaired",
        "gaps",
        "runs",
        "calls",
        "cohorts",
        "transport_failures",
        "payload_failures",
        "stage_timings_omitted",
    )

    add(
        "duration",
        "duration_ms",
        "elapsed_ms",
        "timeout_ms",
        "age_ms",
        "wait_ms",
        "hold_ms",
        "budget_ms",
    )

    add("timings", "stage_timings_ms")
    add("epoch", "mtime")

    add(
        "flag",
        "ok",
        "changed",
        "cached",
        "dry_run",
        "forced",
        "degraded",
        "converged",
        "held",
        "enabled",
        "available",
        "more_pending",
    )

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
