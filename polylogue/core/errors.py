"""Polylogue error hierarchy.

All project exceptions inherit from PolylogueError, enabling:
- ``except PolylogueError`` at top-level boundaries (CLI, MCP server)
- Fine-grained catches deeper in the stack (``except DriveAuthError``)

Hierarchy (subclasses defined in their respective modules):
    PolylogueError                          # this module
    ├── ConfigError                         # config.py
    ├── DriveError                          # sources/drive_client.py
    │   ├── DriveAuthError
    │   └── DriveNotFoundError
    ├── DatabaseError                       # this module
    │   └── SqliteVecError                  # storage/search_providers/sqlite_vec.py
    └── UIError                             # ui/facade.py
"""

from __future__ import annotations

from http import HTTPStatus


class PolylogueError(Exception):
    """Base class for all Polylogue errors.

    Every derived error should set ``is_transient`` (whether retrying might
    succeed) and ``http_status_code`` (the best HTTP status when surfaced
    through the daemon API or MCP boundary).
    """

    is_transient: bool = False
    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR


class RawCASFrontierError(PolylogueError):
    """Retryable compare-and-swap conflict while advancing raw authority."""

    is_transient = True


class DatabaseError(PolylogueError):
    """Base class for database errors."""

    http_status_code: int = HTTPStatus.SERVICE_UNAVAILABLE


class ArchiveTierUnavailableError(DatabaseError):
    """A required archive tier cannot be read at its resolved path."""

    code = "archive_tier_unavailable"

    def __init__(self, *, tier: str, path: str, reason: str, guidance: str) -> None:
        self.tier = tier
        self.path = path
        self.reason = reason
        self.guidance = guidance
        super().__init__(f"{tier} tier unavailable at {path}: {reason}. {guidance}")


class SchemaRefusalError(DatabaseError):
    """Base class for schema refusals raised before a tier can be served.

    Version mismatches and derived-schema identity mismatches carry different
    remediation metadata, but callers that only need to reject an unreadable
    tier should be able to handle both through one stable ancestor.
    """


class SchemaVersionMismatchError(SchemaRefusalError):
    """Raised when the on-disk schema version cannot be served by this runtime.

    The runtime expects ``expected_version`` (the build-time ``SCHEMA_VERSION``
    constant). The database reports ``current_version``. There is no automatic
    in-place upgrade path that the runtime is willing to apply for this
    transition.

    Both versions are exposed as attributes so call sites can format a
    structured message (and the daemon health surface can render it) without
    re-parsing the human-readable string.
    """

    def __init__(
        self,
        message: str,
        *,
        current_version: int,
        expected_version: int,
        generation_id: str | None = None,
        lifecycle_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.current_version = current_version
        self.expected_version = expected_version
        self.generation_id = generation_id
        self.lifecycle_action = lifecycle_action


class SchemaSkewError(SchemaRefusalError):
    """A tier cannot be served by this runtime's schema contract."""

    code = "schema_skew"
    http_status_code = HTTPStatus.CONFLICT

    def __init__(
        self,
        tier: str,
        expected: object,
        found: object,
        remedy: str | None = None,
    ) -> None:
        self.tier = tier
        self.expected = expected
        self.found = found
        self.remedy = remedy or "daemon convergence"
        if remedy is None:
            message = (
                f"{tier} derived schema identity mismatch: expected {expected}, found {found!r}; "
                "stale derived tier; daemon convergence rebuilds it"
            )
        else:
            message = f"{tier} schema skew: expected {expected}, found {found}. {remedy}"
        super().__init__(message)


SchemaSkew = SchemaSkewError


class EmbeddingRetrievalNotReadyError(DatabaseError):
    """Raised when ``--similar``/``--semantic`` is asked for but vectors aren't ready.

    Carries an operator-actionable message naming the current readiness
    status and the next step (``polylogue ops embed status`` →
    ``polylogue ops embed backfill``/``enable``). Unlike a generic
    :class:`DatabaseError`, this class lets surfaces forward the message
    verbatim to the client because the contents are by construction free
    of secrets — the readiness status enum is a closed vocabulary
    (``ready``/``partial``/``pending``/``disabled``/``none``) and the
    follow-up command names are fixed strings, not user data.

    Used by the operations layer's
    ``_resolve_vector_provider_for_search`` so that CLI, MCP, and HTTP
    surfaces all surface the same actionable error instead of the CLI
    getting the message and MCP getting only the exception class name
    (#1503 AC4).
    """

    http_status_code: int = HTTPStatus.CONFLICT

    def __init__(self, message: str, *, readiness_status: str) -> None:
        super().__init__(message)
        self.readiness_status = readiness_status


class InsightMaintenanceRequiresDaemonError(PolylogueError):
    """Session-insight maintenance was asked for outside its sealed owner.

    A rebuild sweep is authorized page by page: the scope is frozen into a
    manifest, staged as immutable preview pages, sealed into accepted machine
    parts, and only then started one ordinal at a time
    (``OperationExecutor.begin_accepted_insight_part``).  That sequence needs
    durable audit authority, a pinned index generation, and the resident
    session-profile publication owner — none of which a library process holds.
    Letting the generic prepare/authorize/execute path run the rebuild instead
    would turn an unsealed staging page into execution authority, and would let
    ``session_ids=None`` become an accepted full sweep after the fact.

    The message names the sanctioned route verbatim so CLI, MCP, and HTTP
    surfaces can forward it: it contains only fixed strings, never user data.
    """

    code = "daemon_required"
    http_status_code: int = HTTPStatus.SERVICE_UNAVAILABLE

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "session-insight maintenance runs only through its sealed accepted-part owner; "
                "run `polylogued run` and submit the daemon operation 'maintenance.insights.rebuild' "
                "(a library or MCP process cannot hold that authority)"
            )
        )


class UnsupportedInsightFilterError(PolylogueError):
    """A read surface accepted an insight filter it cannot evaluate.

    Returning an empty list for a plumbed-but-unimplemented filter makes
    "this filter is not wired" indistinguishable from "nothing matches" — the
    caller reads an unmeasured state as a measured zero. Refusing names the
    filter and the route instead.
    """

    code = "unsupported_insight_filter"
    http_status_code: int = HTTPStatus.BAD_REQUEST

    def __init__(self, *, filter_name: str, route: str, detail: str | None = None) -> None:
        self.filter_name = filter_name
        self.route = route
        super().__init__(
            f"{route} cannot evaluate the '{filter_name}' filter"
            + (f": {detail}" if detail else "; it is accepted but not implemented")
        )


class PostFilterAfterLimitError(PolylogueError):
    """A post-filtered read scope is too large to evaluate before its page.

    A filter with no SQL reduction must be applied over the whole matched set
    before the page is cut, or the page becomes the denominator ("of the newest
    N, the matching ones"). Above the declared candidate cap the honest answer
    is a named refusal, not a quietly mis-scoped page.
    """

    code = "post_filter_scope_too_large"
    http_status_code: int = HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    def __init__(self, *, filter_name: str, route: str, candidate_count: int, cap: int) -> None:
        self.filter_name = filter_name
        self.route = route
        self.candidate_count = candidate_count
        self.cap = cap
        super().__init__(
            f"{route} must evaluate '{filter_name}' over {candidate_count} candidate sessions, "
            f"above the declared cap of {cap}; narrow the scope (origin:, since/until) and retry"
        )


__all__ = [
    "ArchiveTierUnavailableError",
    "InsightMaintenanceRequiresDaemonError",
    "DatabaseError",
    "EmbeddingRetrievalNotReadyError",
    "PolylogueError",
    "PostFilterAfterLimitError",
    "RawCASFrontierError",
    "SchemaRefusalError",
    "SchemaVersionMismatchError",
    "SchemaSkew",
    "SchemaSkewError",
    "UnsupportedInsightFilterError",
]
