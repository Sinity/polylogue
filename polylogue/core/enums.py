"""Closed vocabularies and SQL CHECK helpers for archive work."""

from __future__ import annotations

from enum import StrEnum

from polylogue.core.provider_identity import canonical_runtime_provider


class PolylogueStrEnum(StrEnum):
    """Base class for persisted string enums."""

    def __str__(self) -> str:
        return self.value


def enum_values(enum_type: type[PolylogueStrEnum]) -> tuple[str, ...]:
    """Return persisted values for a closed enum."""
    return tuple(item.value for item in enum_type)


def sql_string_literal(value: str) -> str:
    """Return a SQLite string literal for an enum value."""
    return "'" + value.replace("'", "''") + "'"


def sql_value_list(enum_type: type[PolylogueStrEnum]) -> str:
    """Return comma-separated SQLite literals for an enum CHECK."""
    return ", ".join(sql_string_literal(value) for value in enum_values(enum_type))


def sql_check_in(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return ``column IN (...)`` for a non-null enum column."""
    return f"{column} IN ({sql_value_list(enum_type)})"


def nullable_sql_check_in(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return a nullable enum CHECK expression."""
    return f"({sql_check_in(column, enum_type)} OR {column} IS NULL)"


class Origin(PolylogueStrEnum):
    """Archive source-origin tokens."""

    CLAUDE_CODE_SESSION = "claude-code-session"
    CODEX_SESSION = "codex-session"
    GEMINI_CLI_SESSION = "gemini-cli-session"
    HERMES_SESSION = "hermes-session"
    ANTIGRAVITY_SESSION = "antigravity-session"
    BEADS_ISSUE = "beads-issue"
    GROK_EXPORT = "grok-export"
    CHATGPT_EXPORT = "chatgpt-export"
    CLAUDE_AI_EXPORT = "claude-ai-export"
    # bd polylogue-tbun: Claude Design is a distinct product with its own wire
    # format (camelCase contentBlocks/turnChanges/authorAccountUuid, content
    # is a dict not a list) -- not claude.ai with a flag. See
    # sources/parsers/claude/ai_parser.py's design-chat parser.
    CLAUDE_DESIGN_SESSION = "claude-design-session"
    AISTUDIO_DRIVE = "aistudio-drive"
    UNKNOWN_EXPORT = "unknown-export"

    @classmethod
    def from_string(cls, value: str | Origin | None) -> Origin:
        """Normalize an origin token to the enum, defaulting to UNKNOWN_EXPORT."""
        if value is None:
            return cls.UNKNOWN_EXPORT
        try:
            return cls(str(value))
        except ValueError:
            return cls.UNKNOWN_EXPORT


class Provider(PolylogueStrEnum):
    """Legacy runtime provider tokens retained during the archive transition."""

    CHATGPT = "chatgpt"
    CLAUDE_AI = "claude-ai"
    CLAUDE_DESIGN = "claude-design"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    GEMINI = "gemini"
    GEMINI_CLI = "gemini-cli"
    HERMES = "hermes"
    ANTIGRAVITY = "antigravity"
    BEADS = "beads"
    GROK = "grok"
    DRIVE = "drive"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | Provider | None) -> Provider:
        """Normalize provider string to enum, defaulting to UNKNOWN."""
        normalized = canonical_runtime_provider(str(value) if value is not None else None)
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class SessionKind(PolylogueStrEnum):
    """Closed session lifecycle/type vocabulary."""

    STANDARD = "standard"
    TEMPORARY = "temporary"

    @classmethod
    def normalize(cls, value: object) -> SessionKind:
        """Normalize a session-kind token, defaulting to standard."""
        if isinstance(value, SessionKind):
            return value
        if value is None:
            return cls.STANDARD
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.STANDARD


#: Role synonym vocabulary — single source of truth for role normalization.
#: Maps each canonical role to its accepted synonyms (case-insensitive).
#: Used by both Role.normalize() and SQL role-filter expansion.
ROLE_SYNONYMS: dict[str, frozenset[str]] = {
    "user": frozenset({"user", "human"}),
    "assistant": frozenset({"assistant", "model", "ai"}),
    "system": frozenset({"system", "developer"}),
    "tool": frozenset({"tool", "function", "tool_use", "tool_result", "progress", "result"}),
    "unknown": frozenset({"unknown"}),
}


class Role(PolylogueStrEnum):
    """Canonical session roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, raw: str) -> Role:
        """Normalize a provider role string to a canonical role."""
        lowered = raw.strip().lower()
        if not lowered:
            raise ValueError("Role cannot be empty. Handle missing roles at parse time.")

        for role_name, synonyms in ROLE_SYNONYMS.items():
            if lowered in synonyms:
                return cls(role_name)
        return cls.UNKNOWN


class MessageType(PolylogueStrEnum):
    """Normalized message type for filtering and read surfaces."""

    MESSAGE = "message"
    SUMMARY = "summary"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    CONTEXT = "context"
    PROTOCOL = "protocol"

    @classmethod
    def normalize(cls, value: object) -> MessageType:
        """Coerce provider/parser message-type values to a canonical type."""
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        if not candidate:
            return cls.MESSAGE
        for item in cls:
            if item.value == candidate:
                return item
        return cls.MESSAGE

    @classmethod
    def validate_filter_token(cls, value: object) -> MessageType:
        """Validate one user-supplied message-type filter token."""
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        for item in cls:
            if item.value == candidate:
                return item
        valid = ", ".join(item.value for item in cls)
        msg = f"Unknown message type {str(value)!r}. Valid message types: {valid}"
        raise ValueError(msg)


class MaterialOrigin(PolylogueStrEnum):
    """Archive-visible authoredness/material-origin axis for messages.

    ``Role`` preserves provider/API envelope truth. Material origin answers
    what kind of material the row represents for accounting, projections, and
    user-facing prose filters.
    """

    HUMAN_AUTHORED = "human_authored"
    ASSISTANT_AUTHORED = "assistant_authored"
    OPERATOR_COMMAND = "operator_command"
    RUNTIME_PROTOCOL = "runtime_protocol"
    RUNTIME_CONTEXT = "runtime_context"
    TOOL_RESULT = "tool_result"
    GENERATED_CONTEXT_PACK = "generated_context_pack"
    GENERATED_ANALYSIS_PACK = "generated_analysis_pack"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, value: object) -> MaterialOrigin:
        if isinstance(value, MaterialOrigin):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        if not candidate:
            return cls.UNKNOWN
        for item in cls:
            if item.value == candidate:
                return item
        return cls.UNKNOWN

    @classmethod
    def validate_filter_token(cls, value: object) -> MaterialOrigin:
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        if not candidate:
            msg = "Material origin cannot be empty"
            raise ValueError(msg)
        for item in cls:
            if item.value == candidate:
                return item
        valid = ", ".join(item.value for item in cls)
        msg = f"Unknown material origin {str(value)!r}. Valid material origins: {valid}"
        raise ValueError(msg)


class BlockType(PolylogueStrEnum):
    """Canonical stored and parsed block kinds.

    Single block-kind vocabulary across the parse and storage layers (the
    `blocks.block_type` CHECK validates against it). `reasoning` is a
    storage-side superset value; parsers may emit it where a provider
    distinguishes reasoning from thinking.
    """

    TEXT = "text"
    THINKING = "thinking"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    CODE = "code"
    DOCUMENT = "document"

    @classmethod
    def from_string(cls, value: str | BlockType) -> BlockType:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class WebConstructType(PolylogueStrEnum):
    """Typed web-export constructs projected out of provider payloads."""

    CANVAS = "canvas"
    CONTENT_REFERENCE = "content_reference"
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT = "search_result"
    SELECTED_SOURCE = "selected_source"
    IMAGE_RESULT = "image_result"
    ASYNC_TASK = "async_task"
    AUDIO_ASSET = "audio_asset"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    TOKEN_BUDGET = "token_budget"
    VOICE_NOTE = "voice_note"


class SemanticBlockType(PolylogueStrEnum):
    """Canonical semantic classifications for stored content blocks."""

    OTHER = "other"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    SHELL = "shell"
    GIT = "git"
    SEARCH = "search"
    WEB = "web"
    AGENT = "agent"
    SUBAGENT = "subagent"
    THINKING = "thinking"

    @classmethod
    def from_string(cls, value: str | SemanticBlockType) -> SemanticBlockType:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class TitleSource(PolylogueStrEnum):
    """Classification for how a session's title was derived.

    polylogue-5dfu: this used to also carry ``USER`` and ``UNKNOWN``.
    ``USER`` had no producer anywhere (write- or read-time) -- deleted.
    ``UNKNOWN`` was a second, redundant spelling of "no title evidence" on an
    already-nullable column (every read site branching on ``title_source``
    treated ``NULL`` and ``'unknown'`` identically) -- deleted in favor of
    ``NULL`` alone. ``PATH`` looks unused the same way at a glance (nothing
    ever *stores* it in ``sessions.title_source``), but it has a genuine
    read-time producer: ``archive_tiers/archive.py``'s ``_summary_from_row``
    assigns it whenever a session has neither a real provider title nor a
    display name, synthesizing a structural label instead -- kept.
    """

    ORIGIN = "origin"
    PATH = "path"
    HEURISTIC = "heuristic"


class BranchType(PolylogueStrEnum):
    """Classification for how a session relates to its parent."""

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    FORK = "fork"
    SUBAGENT = "subagent"


class LinkType(PolylogueStrEnum):
    """Archive cross-session edge vocabulary.

    polylogue-5dfu: ``REPAIRED`` was deleted -- it duplicated
    ``TopologyEdgeStatus.REPAIRED`` (same string, different column/meaning:
    a link *type* vs. a link's exceptional *status*) and had no producer,
    fixture, or doc reference anywhere. ``FORK`` and ``RESUME`` look equally
    unused from a live-archive row count alone (both are 0 rows today) but
    each has a concrete, named producer: ``FORK`` is emitted by
    ``sources/parsers/hermes_state.py``'s ``_branch_type`` whenever a Hermes
    session's ``model_config._branched_from`` is set (real code, just never
    yet hit by an ingested Hermes session); ``RESUME`` is the "resume
    lineage edge" cross-repo fixture documented in
    ``docs/material-protocol-v1.md`` -- Sinex (``sinex-4j2.1.1``) is expected
    to emit it over the material-protocol-v1 wire once that side lands.
    """

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    SUBAGENT = "subagent"
    BRANCH = "branch"
    FORK = "fork"
    RESUME = "resume"


class TopologyEdgeStatus(PolylogueStrEnum):
    """Exceptional-marker vocabulary for a ``session_links`` row's ``status``.

    polylogue-5dfu: this used to declare 4 members
    (unresolved/resolved/repaired/quarantined) while the DDL ``CHECK`` and
    ``queries/session_links.py``'s ``_status_value`` projection both already
    narrowed it to 2 -- ``UNRESOLVED``/``RESOLVED`` were never actually
    stored, because resolvedness is already carried by
    ``resolved_dst_session_id IS NOT NULL`` and neither member was ever
    constructed anywhere outside a Pydantic field default. Narrowed to match
    what the column actually stores: an exceptional marker recording *why*
    an edge needed intervention, not the ordinary resolved/unresolved state.
    """

    REPAIRED = "repaired"
    QUARANTINED = "quarantined"


class StopReason(PolylogueStrEnum):
    """Provider-reported terminal state for one assistant turn.

    polylogue-cuxz.8: the wire carries this on 608,608 Claude assistant
    messages while three derived columns (delegation_facts.result_status,
    .parent_terminal_state, session_profiles.terminal_state) each guess at
    the same fact and are 85-99% 'unknown'. Persisting the provider's own
    value directly is what lets those consumers stop guessing. Vocabulary is
    Anthropic's own five-value ``stop_reason`` enumeration; a provider that
    reports something outside this set leaves the column NULL (unknown)
    rather than widening the CHECK to a guess.
    """

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    STOP_SEQUENCE = "stop_sequence"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"


class ToolResultUnknownReason(PolylogueStrEnum):
    """Why ``blocks.tool_result_is_error`` is NULL for a tool_result block.

    polylogue-cuxz.8: NULL alone conflates three distinct causes -- keeping
    them distinguishable is the point (72% of blocks.tool_result_is_error is
    NULL archive-wide, and "unknown" must not silently mean "known to be
    fine"). NULL on this column (rather than one of these three) means the
    outcome IS known (tool_result_is_error is set) -- this column only ever
    describes an unknown outcome's reason.
    """

    # The provider's own record carried no outcome signal at all (no
    # is_error/exit_code field present in the source structure).
    NOT_REPORTED = "not_reported"
    # The provider reported an outcome signal, but the parser has a positive
    # reason not to trust it for this record shape (e.g. a known-unreliable
    # sentinel value for this origin).
    DISTRUSTED = "distrusted"
    # This origin's parser does not yet read the field the provider carries.
    NOT_READ = "not_read"


class SessionRefKind(PolylogueStrEnum):
    """Closed vocabulary for ``session_refs.kind`` (tracker-agnostic).

    polylogue-cgfy: 20,702 Claude Code sessions carry a pr-link the parser
    never reads. ``session_refs`` generalizes beyond pull requests so a
    future issue-tracker reference lands in the same relation rather than a
    second single-purpose table.
    """

    PULL_REQUEST = "pull_request"
    ISSUE = "issue"


class PasteBoundary(PolylogueStrEnum):
    """Boundary quality for detected paste spans."""

    EXACT = "exact"
    PROJECTED = "projected"
    WHOLE_MESSAGE_FALLBACK = "whole_message_fallback"
    HASH_ONLY = "hash_only"


class ValidationStatus(PolylogueStrEnum):
    """Persisted raw-schema validation outcome."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def from_string(cls, value: str | ValidationStatus) -> ValidationStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class IngestOutcome(PolylogueStrEnum):
    """Closed, typed disposition for one ``ingest_attempts`` row (polylogue-cnu3).

    Before this enum, ``ingest_attempts.error_message`` was free-form text, so
    a basic OriginSpec question ("how often did strict validation reject real
    input?") required re-grepping logs and source. Each member is a distinct,
    structurally-detected outcome -- never derived by text-matching against
    ``error_message`` -- so it stays queryable without guessing.

    ``SUCCESS``: the attempt completed (including a no-op idempotent
    re-ingest of unchanged content).
    ``VALIDATION_REJECTED``: strict schema validation (or a provider parser's
    own structural validator, e.g. a ``pydantic.ValidationError``) rejected
    the input.
    ``UNSUPPORTED_SHAPE``: the artifact was recognized but is not admitted
    for session parsing by policy (e.g. a non-session artifact kind), or
    parsing produced no materializable sessions.
    ``CORRUPT_INPUT``: the raw bytes could not be decoded as the expected
    payload shape (empty blob, undecodable JSON/UTF-8).
    ``TRANSIENT_ERROR``: a retryable infrastructure failure (SQLite
    lock/busy contention) at the archive-write boundary.
    ``PARSER_DEFECT``: an unexpected exception from parsing/transform that
    is not one of the above structurally-detected classes -- a real bug
    bucket, not a guess.
    ``LEGACY_UNKNOWN``: a historical row written before this vocabulary
    existed, or a code path not yet classified. Never guess-assigned; rows
    default here and stay here until a real classification is added.

    Deliberately deferred to follow-up work (not yet members, so no attempt
    can look falsely covered by them): ``MATERIALIZATION_FAILED``/
    index-failure and ``CANCELED`` (AC2's remaining two classes) -- the
    daemon convergence layer that would emit them needs its own wiring pass.
    """

    SUCCESS = "success"
    VALIDATION_REJECTED = "validation_rejected"
    UNSUPPORTED_SHAPE = "unsupported_shape"
    CORRUPT_INPUT = "corrupt_input"
    TRANSIENT_ERROR = "transient_error"
    PARSER_DEFECT = "parser_defect"
    LEGACY_UNKNOWN = "legacy_unknown"

    @classmethod
    def from_string(cls, value: str | IngestOutcome) -> IngestOutcome:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


#: Retryability for each :class:`IngestOutcome`. ``True`` means the daemon's
#: retry policy may safely loop on it; ``False`` means looping would just
#: repeat the same rejection; ``None`` (only ``LEGACY_UNKNOWN``) means the
#: historical row carries no retryability evidence at all. A retryable
#: outcome must never be reported terminal, and a non-retryable defect must
#: never be silently retried forever (polylogue-cnu3 AC3).
INGEST_OUTCOME_RETRYABLE: dict[IngestOutcome, bool | None] = {
    IngestOutcome.SUCCESS: False,
    IngestOutcome.VALIDATION_REJECTED: False,
    IngestOutcome.UNSUPPORTED_SHAPE: False,
    IngestOutcome.CORRUPT_INPUT: False,
    IngestOutcome.TRANSIENT_ERROR: True,
    IngestOutcome.PARSER_DEFECT: False,
    IngestOutcome.LEGACY_UNKNOWN: None,
}


class ValidationMode(PolylogueStrEnum):
    """Configured raw-schema validation strictness."""

    OFF = "off"
    ADVISORY = "advisory"
    STRICT = "strict"

    @classmethod
    def from_string(cls, value: str | ValidationMode) -> ValidationMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class ArtifactSupportStatus(PolylogueStrEnum):
    """Durable support state for an observed raw artifact."""

    SUPPORTED_PARSEABLE = "supported_parseable"
    RECOGNIZED_UNPARSED = "recognized_unparsed"
    UNSUPPORTED_PARSEABLE = "unsupported_parseable"
    DECODE_FAILED = "decode_failed"
    PARTIAL_DECODE = "partial_decode"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | ArtifactSupportStatus) -> ArtifactSupportStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class PlanStage(PolylogueStrEnum):
    """Supported ingest/runtime planning stages."""

    ALL = "all"
    ACQUIRE = "acquire"
    CUSTOM = "custom"
    PARSE = "parse"
    MATERIALIZE = "materialize"
    RENDER = "render"
    SITE = "site"
    INDEX = "index"
    SCHEMA = "schema"
    REPROCESS = "reprocess"
    PUBLISH = "publish"

    @classmethod
    def from_string(cls, value: str | PlanStage) -> PlanStage:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class AssertionKind(PolylogueStrEnum):
    """Closed vocabulary for ``user.db`` assertion rows.

    The unified assertions table collapses the old user-tier overlay
    mini-systems. The SQLite column is stored as ``TEXT`` so the vocabulary can
    grow without forcing a user-tier schema bump; this enum is the typed
    runtime and surface boundary.
    """

    MARK = "mark"
    HIGHLIGHT = "highlight"
    ANNOTATION = "annotation"
    CORRECTION = "correction"
    SUPPRESSION = "suppression"
    TAG = "tag"
    METADATA = "metadata"
    SAVED_QUERY = "saved_query"
    RECALL_PACK = "recall_pack"
    WORKSPACE_NOTE = "workspace_note"
    NOTE = "note"
    DECISION = "decision"
    CAVEAT = "caveat"
    LESSON = "lesson"
    BLOCKER = "blocker"
    HANDOFF = "handoff"
    JUDGMENT = "judgment"
    RUN_STATE = "run_state"
    PROMPT_EVAL = "prompt_eval"
    ONTOLOGY_CANDIDATE = "ontology_candidate"
    """Agent-proposed archive-specific annotation schema plus nomination evidence.
    Informal tags/affinity can create this non-injected candidate, never a formal
    annotation fact or active schema on their own."""
    ONTOLOGY_GOVERNANCE = "ontology_governance"
    """Operator-authored receipt for accepting, renaming, splitting, or rejecting
    an ontology candidate and, when applicable, registering active schema rows."""
    TRANSFORM_CANDIDATE = "transform_candidate"
    PATHOLOGY = "pathology"
    FINDING = "finding"
    SECRET_CANDIDATE = "secret_candidate"
    """Candidate-only secret-detector finding (polylogue-27m). Never carries the
    matched literal -- ``value_json`` holds only a fingerprint hash, matched
    length, pattern id, and span coordinates. Always written with
    ``author_kind="detector"``, which the ``upsert_assertion`` chokepoint
    coerces to a non-injectable ``CANDIDATE`` status."""
    EXCISION_RECORD = "excision_record"
    """Durable, operator-authored receipt of a completed local excision: the
    removed content-hash markers, reason, actor, prior revision, and per-tier
    row counts. Written only after a standalone/off-mode apply commits."""
    EXCISION_REQUEST = "excision_request"
    """Durable mirror/primary-mode lifecycle-request/outbox row (polylogue-27m).
    ``value_json`` carries the request state machine (pending/acknowledged/
    confirmed/rejected), retry count, and target. Lives in ``user.db`` so the
    request survives an ``ops.db`` reset -- see ``polylogue/security/lifecycle.py``."""
    COMPARATIVE_JUDGMENT = "comparative_judgment"

    @classmethod
    def from_string(cls, value: str | AssertionKind) -> AssertionKind:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class ComparativeVerdict(PolylogueStrEnum):
    """Closed verdict vocabulary for comparative judgments (rxdo.9.11, mechanism K).

    ``PREFER_LEFT``/``PREFER_RIGHT`` apply only to pairwise (two-item)
    comparisons. ``TIE`` and ``INCOMPARABLE`` are semantically distinct: a tie
    means both items were judged equal on the dimension; incomparable means
    the dimension does not meaningfully apply to this pair. ``ABSTAIN`` and
    ``INSUFFICIENT_EVIDENCE`` must never be treated as weak preferences by
    downstream aggregation (:mod:`polylogue.insights.judgment.rankers`) --
    they contribute zero directed preference edges.
    """

    PREFER_LEFT = "prefer_left"
    PREFER_RIGHT = "prefer_right"
    TIE = "tie"
    INCOMPARABLE = "incomparable"
    ABSTAIN = "abstain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @classmethod
    def from_string(cls, value: str | ComparativeVerdict) -> ComparativeVerdict:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class AssertionStatus(PolylogueStrEnum):
    """Closed lifecycle state vocabulary for assertion rows."""

    ACTIVE = "active"
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    INACTIVE = "inactive"

    @classmethod
    def from_string(cls, value: str | AssertionStatus) -> AssertionStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class AssertionVisibility(PolylogueStrEnum):
    """Closed visibility vocabulary for assertion rows."""

    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"

    @classmethod
    def from_string(cls, value: str | AssertionVisibility) -> AssertionVisibility:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class RawAuthorityVerdict(PolylogueStrEnum):
    """Closed 5-value verdict vocabulary for raw-capture authority (polylogue-w6hql).

    Phase 2 of the raw-authority redesign: this is the single small vocabulary
    that downstream consumers (blob-GC invariant checks, operator surfaces)
    should read instead of reaching into the fragmented multi-table
    bookkeeping (``raw_authority_blockers``/``raw_authority_censuses``/
    ``raw_authority_census_plans``/``raw_authority_post_plans``/
    ``raw_authority_parser_census``/``raw_membership_census``). It is derived
    -- via :func:`polylogue.archive.raw_authority_verdict.derive_raw_authority_verdict`
    -- from the existing, already-proven per-raw evidence
    (:class:`polylogue.archive.revision_authority.HistoricalRevisionDecision`),
    not a new source of truth: it does not replace ``raw_sessions.revision_authority``
    (``asserted``/``byte_proven``/``quarantined``, the byte-provenance axis) or
    the fragmented tables' write paths, which remain in place this phase.

    - ``VERIFIED``: proven (byte_proven) head of a multi-revision cohort --
      nothing supersedes it and at least one other revision of the same
      logical source exists (otherwise it would be ``SOLE_COPY``).
    - ``SUPERSEDED``: proven, but a later revision (chain successor) or a
      byte-identical duplicate representative has taken its place as the
      cohort's current evidence.
    - ``SOLE_COPY``: the only captured revision of its logical source --
      trivially authoritative because there is nothing to compare against.
    - ``DIVERGED``: the cohort could not be reduced to a single proven chain
      (a byte-prefix fork, multiple incomparable roots, or any other
      condition that leaves this raw quarantined).
    - ``UNCHECKED``: no revision-authority classification has run over this
      raw yet (``revision_authority='asserted'``, the pre-governance default).
    """

    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    SOLE_COPY = "sole-copy"
    DIVERGED = "diverged"
    UNCHECKED = "unchecked"

    @classmethod
    def from_string(cls, value: str | RawAuthorityVerdict) -> RawAuthorityVerdict:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


__all__ = [
    "AssertionKind",
    "AssertionStatus",
    "AssertionVisibility",
    "ArtifactSupportStatus",
    "BlockType",
    "BranchType",
    "LinkType",
    "MaterialOrigin",
    "MessageType",
    "Origin",
    "PasteBoundary",
    "PlanStage",
    "PolylogueStrEnum",
    "Provider",
    "RawAuthorityVerdict",
    "Role",
    "SemanticBlockType",
    "SessionRefKind",
    "StopReason",
    "TitleSource",
    "ToolResultUnknownReason",
    "TopologyEdgeStatus",
    "ValidationMode",
    "ValidationStatus",
    "enum_values",
    "nullable_sql_check_in",
    "sql_check_in",
    "sql_string_literal",
    "sql_value_list",
]
