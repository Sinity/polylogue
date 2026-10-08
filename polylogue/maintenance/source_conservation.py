"""Source-to-archive conservation: every acquired source item types into one term.

Backs the ``source-conservation`` owner check of ``verify-archive``. The
forward universe is the durable acquisition ledger (``raw_sessions``,
``raw_hook_events``, ``history_sidecars``); the reverse universe is every
index row (sessions, messages, blocks, attachment refs). Each item lands in
exactly one term, in a fixed precedence, and each term cites the rule that
explains it. An item no rule explains is an *unexplained* term and turns the
check red; a typed exclusion never does.

A rule may cite another owner's durable ledger. ``raw_authority_blockers``
states, per unresolved blocker, that the authority frontier accepted one raw as
the revision head while the index materialized another raw of the same logical
source (``authority_blocked_head``); ``raw_session_memberships`` states which
logical sources a raw belongs to, and a raw whose memberships are all
quarantined with nothing in the cohort indexed is a whole logical session
missing from the index (``quarantined_cohort_unmaterialized``, blocking).

Phantom sessions (polylogue-b508) are a reverse-direction term: an index
session whose only source lineage is a declared non-session artifact
(sidecar, workflow journal, tool-result fragment, metadata fragment) or whose
identity carries an artifact-derived shape. They are reported as a
current-producer failure and never deleted here.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.archive.revision_authority import (
    WORK_EVENT_RAW_ID_PREFIX,
    is_work_event_raw_id,
    logical_head_cohort_sql,
    raw_receipt_order_sql,
)
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.raw_coordinates import read_captured_zip_coordinate_receipt
from polylogue.core.sqlite_introspection import table_exists
from polylogue.maintenance.source_manifest_continuity import SourceFrontier
from polylogue.sources.origin_specs import ORIGIN_SPECS, OriginArtifactRule
from polylogue.sources.value_bounds import VALUE_BOUND_REFUSED, VALUE_BOUND_REFUSED_HEAD
from polylogue.storage.sqlite.connection_profile import readonly_temp_staging

#: Identity prefixes that name provider fragments, never conversations:
#: ``toolu_`` is a tool_use block id (tool-result fragment) and ``wf_`` is a
#: workflow run id (workflow journal/snapshot).
FRAGMENT_IDENTITY_PREFIXES: tuple[str, ...] = ("toolu_", "wf_")

#: Identity suffixes left behind when a sidecar filename stem is mistaken for
#: a session id. Each entry names the artifact kind whose declared path
#: pattern produces it.
ARTIFACT_IDENTITY_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".meta", "agent_sidecar_meta"),
    (".metadata", "agent_sidecar_meta"),
)

_TERM_SOURCE_MISSING = "source_missing"
_TERM_SOURCE_LOST = "source_lost"
_TERM_MISSING_BLOB = "missing_blob"
_TERM_SOURCE_UNAVAILABLE = "source_unavailable"
_TERM_MATERIALIZED = "materialized"
_TERM_REVISION_SUPERSEDED = "revision_superseded"
_TERM_BYTE_DUPLICATE = "byte_duplicate_superseded"
_TERM_PARSE_FAILURE = "parse_failure"
_TERM_VALUE_BOUND_REFUSED = VALUE_BOUND_REFUSED
_TERM_VALIDATION_REJECTED = "validation_rejected"
_TERM_NON_SESSION_ARTIFACT = "non_session_artifact"
_TERM_DECODE_FAILED = "decode_failed"
_TERM_CENSUS_NON_SESSION = "census_non_session"
_TERM_UNCLASSIFIED_SHAPE = "unclassified_shape"
_TERM_PENDING = "pending"
_TERM_AUTHORITY_BLOCKED = "authority_blocked_head"
_TERM_QUARANTINED_COHORT = "quarantined_cohort_unmaterialized"
_TERM_UNEXPLAINED = "unexplained"

_TERM_HOOK_MATERIALIZED = "hook_session_materialized"
_TERM_HOOK_ACQUIRED = "hook_session_acquired"
_TERM_HOOK_NO_SESSION_ID = "hook_without_session_id"
_TERM_HOOK_NO_SOURCE = "hook_without_source_session"
_TERM_SIDECAR_RETAINED = "sidecar_retained"

_TERM_SESSION_WITHOUT_RAW = "session_without_raw"
_TERM_SESSION_ORPHAN = "session_orphan"
_TERM_PHANTOM_LINEAGE = "phantom_declared_non_session_lineage"
_TERM_PHANTOM_IDENTITY = "phantom_fragment_identity"
_TERM_MESSAGE_ORPHAN = "message_orphan"
_TERM_BLOCK_ORPHAN = "block_orphan"
_TERM_ATTACHMENT_REF_ORPHAN = "attachment_ref_orphan"
_TERM_ATTACHMENT_UNREFERENCED = "attachment_unreferenced"
_TERM_ATTACHMENT_UNOWNED = "attachment_unowned"
_TERM_ATTACHMENT_OWNER_MISSING = "attachment_owner_missing"
_TERM_FRONTIER_UNAVAILABLE = "frontier_unavailable"
_TERM_FRONTIER_UNACQUIRED = "frontier_unacquired"
_TERM_FRONTIER_DUPLICATE = "frontier_duplicate"
_TERM_FRONTIER_ORPHAN = "frontier_orphan"
_TERM_CONTENT_MISMATCH = "content_mismatch"
_TERM_ATTACHMENT_MISSING = "attachment_missing"

_RULES: dict[str, str] = {
    _TERM_SOURCE_MISSING: ("acquired source file no longer exists on disk; the archive retains its raw payload bytes"),
    _TERM_SOURCE_LOST: (
        "acquired source file no longer exists on disk and no raw payload blob is retained; the bytes are gone"
    ),
    _TERM_MISSING_BLOB: "acquired raw payload has no retained CAS body; an original source is not archive retention",
    _TERM_SOURCE_UNAVAILABLE: "source file or member inventory is unreadable; retention is unmeasured and retryable",
    _TERM_MATERIALIZED: "index session carries this raw_id, or the agent work event's session is indexed",
    _TERM_REVISION_SUPERSEDED: "another revision of the same logical source is materialized",
    _TERM_BYTE_DUPLICATE: "content-bound byte-duplicate supersession receipt names a materialized twin",
    _TERM_PARSE_FAILURE: "raw_sessions.parse_error records the typed parser refusal",
    _TERM_VALUE_BOUND_REFUSED: (
        "raw_sessions.parse_error records a decoded value longer than SQLite can store in one cell; "
        "the session is refused whole, never truncated (polylogue.sources.value_bounds)"
    ),
    _TERM_VALIDATION_REJECTED: "raw_sessions.validation_status = 'failed' records the schema refusal",
    _TERM_NON_SESSION_ARTIFACT: "raw_artifacts declares the item a non-session artifact kind",
    _TERM_DECODE_FAILED: "raw_artifacts.decode_error records the typed decode failure",
    _TERM_CENSUS_NON_SESSION: "raw_membership_census recorded a terminal non-session verdict",
    _TERM_UNCLASSIFIED_SHAPE: "artifact taxonomy holds no classification (unknown/unknown); a rule is missing",
    _TERM_PENDING: "acquired; convergence has not parsed it yet",
    _TERM_AUTHORITY_BLOCKED: (
        "an unresolved raw_authority_blockers row names this raw as the accepted revision head "
        "while the index materialized a different raw of the same logical source; the authority "
        "frontier owns the remedy and records it in that ledger"
    ),
    _TERM_QUARANTINED_COHORT: (
        "every raw_session_memberships row of this raw carries revision_authority 'quarantined' "
        "and no revision of its logical source materialized, so the whole logical session is "
        "missing from the index"
    ),
    _TERM_UNEXPLAINED: "parsed without refusal, yet no index session and no exclusion rule applies",
    _TERM_HOOK_MATERIALIZED: "hook event names a materialized session",
    _TERM_HOOK_ACQUIRED: "hook event names an acquired raw session (typed by that raw's term)",
    _TERM_HOOK_NO_SESSION_ID: "hook event carries no session identity",
    _TERM_HOOK_NO_SOURCE: "hook event names a session whose file was never acquired",
    _TERM_SIDECAR_RETAINED: "history sidecar is evidence for a session, never a session",
    _TERM_SESSION_WITHOUT_RAW: "index session has no raw_id",
    _TERM_SESSION_ORPHAN: "index session raw_id names no raw_sessions row",
    _TERM_PHANTOM_LINEAGE: "session lineage is a declared non-session artifact (sidecar/journal/fragment/metadata)",
    _TERM_PHANTOM_IDENTITY: "session identity carries a fragment or sidecar shape",
    _TERM_MESSAGE_ORPHAN: "message names no session",
    _TERM_BLOCK_ORPHAN: "block names no message",
    _TERM_ATTACHMENT_REF_ORPHAN: "attachment ref names no message",
    _TERM_ATTACHMENT_UNREFERENCED: (
        "attachment has no ref yet still carries a non-zero ref_count; its refs went away "
        "without the ref-count sweep, so the row is unreachable from every read path"
    ),
    _TERM_ATTACHMENT_UNOWNED: (
        "attachment has no ref and ref_count 0: written unreferenced because its owner was ambiguous "
        "or the provider never linked one; identity and bytes are retained as evidence"
    ),
    _TERM_ATTACHMENT_OWNER_MISSING: (
        "the session's writer recorded that an attachment's provider-named message is absent from the index "
        "(attachment_owner_gaps reason 'message_missing'): an owner was lost, not left unexplained"
    ),
    _TERM_FRONTIER_UNAVAILABLE: "configured source root could not be observed; the denominator is incomplete",
    _TERM_FRONTIER_UNACQUIRED: "configured source member has no acquired raw membership",
    _TERM_FRONTIER_DUPLICATE: "one configured source member has multiple acquired owners",
    _TERM_FRONTIER_ORPHAN: "acquired raw source identity is outside the configured frontier",
    _TERM_CONTENT_MISMATCH: "acquired logical membership and materialized semantic content disagree",
    _TERM_ATTACHMENT_MISSING: "attachment reference names no retained attachment row",
}

_BLOCKING: frozenset[str] = frozenset(
    {
        _TERM_SOURCE_LOST,
        _TERM_MISSING_BLOB,
        _TERM_SOURCE_UNAVAILABLE,
        _TERM_UNCLASSIFIED_SHAPE,
        _TERM_QUARANTINED_COHORT,
        _TERM_UNEXPLAINED,
        _TERM_SESSION_WITHOUT_RAW,
        _TERM_SESSION_ORPHAN,
        _TERM_PHANTOM_LINEAGE,
        _TERM_PHANTOM_IDENTITY,
        _TERM_MESSAGE_ORPHAN,
        _TERM_BLOCK_ORPHAN,
        _TERM_ATTACHMENT_REF_ORPHAN,
        _TERM_ATTACHMENT_UNREFERENCED,
        _TERM_ATTACHMENT_OWNER_MISSING,
        _TERM_FRONTIER_UNAVAILABLE,
        _TERM_FRONTIER_UNACQUIRED,
        _TERM_FRONTIER_DUPLICATE,
        _TERM_FRONTIER_ORPHAN,
        _TERM_CONTENT_MISMATCH,
        _TERM_ATTACHMENT_MISSING,
    }
)

_WARNING: frozenset[str] = frozenset(
    {_TERM_PENDING, _TERM_HOOK_NO_SOURCE, _TERM_AUTHORITY_BLOCKED, _TERM_VALUE_BOUND_REFUSED}
)


@dataclass(frozen=True, slots=True)
class ConservationTerm:
    """One typed outcome: how many items it explains and why."""

    name: str
    count: int
    rule: str
    blocking: bool
    sample: tuple[str, ...] = ()
    breakdown: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "count": self.count,
                "rule": self.rule,
                "blocking": self.blocking,
                "sample": list(self.sample),
                "breakdown": dict(sorted(self.breakdown.items())),
            }
        )


@dataclass(frozen=True, slots=True)
class SourceConservationReport:
    """Both directions of the source/archive equation, every term typed."""

    forward_total: int
    hook_total: int
    sidecar_total: int
    session_total: int
    terms: tuple[ConservationTerm, ...]
    frontier_sha256: str | None = None
    frontier_total: int = 0
    frontier_bytes: int = 0
    frontier_complete: bool | None = None
    frontier_root_states: dict[str, str] = field(default_factory=dict)

    @property
    def blocking_count(self) -> int:
        return sum(term.count for term in self.terms if term.blocking)

    @property
    def warning_count(self) -> int:
        return sum(term.count for term in self.terms if term.name in _WARNING)

    def term(self, name: str) -> ConservationTerm:
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(name)

    def summary(self) -> str:
        parts = [
            f"{self.forward_total:,} raw item(s), {self.hook_total:,} hook event(s), "
            f"{self.session_total:,} index session(s)"
        ]
        if self.frontier_sha256 is not None:
            parts.append(
                f"frontier={self.frontier_total:,} item(s), {self.frontier_bytes:,} byte(s) "
                f"digest={self.frontier_sha256}"
            )
        for term in self.terms:
            if term.count and term.name != _TERM_MATERIALIZED:
                marker = "!" if term.blocking else ""
                parts.append(f"{term.name}={term.count:,}{marker}")
        return "; ".join(parts)

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "forward_total": self.forward_total,
                "hook_total": self.hook_total,
                "sidecar_total": self.sidecar_total,
                "session_total": self.session_total,
                "frontier_sha256": self.frontier_sha256,
                "frontier_total": self.frontier_total,
                "frontier_bytes": self.frontier_bytes,
                "frontier_complete": self.frontier_complete,
                "frontier_root_states": dict(sorted(self.frontier_root_states.items())),
                "blocking_count": self.blocking_count,
                "warning_count": self.warning_count,
                "terms": {term.name: term.to_json() for term in self.terms},
            }
        )


def valid_byte_duplicate_supersession_expr(conn: sqlite3.Connection, *, raw_alias: str) -> str:
    """Return the receipt predicate shared by source/index coverage checks.

    A supersession receipt is authority only when it still names the same
    bytes and an index materialization of the recorded duplicate twin. Keep
    the predicate in one place so backlog freshness cannot classify a receipt
    differently from source-index coverage.
    """
    if not table_exists(conn, "raw_byte_duplicate_supersession_receipts"):
        return "0"
    return f"""
        EXISTS(
            SELECT 1
            FROM raw_byte_duplicate_supersession_receipts receipt
            JOIN raw_sessions twin ON twin.raw_id = receipt.duplicate_of_raw_id
            JOIN idx_tier.sessions twin_session
              ON twin_session.raw_id = twin.raw_id
             AND twin_session.session_id = receipt.duplicate_of_session_id
            WHERE receipt.raw_id = {raw_alias}.raw_id
              AND receipt.blob_hash = {raw_alias}.blob_hash
              AND receipt.blob_size = {raw_alias}.blob_size
              AND twin.blob_hash = {raw_alias}.blob_hash
              AND twin.blob_size = {raw_alias}.blob_size
              AND twin.origin IS {raw_alias}.origin
              AND twin.source_path IS {raw_alias}.source_path
              AND twin.source_index IS {raw_alias}.source_index
        )
    """


def raw_materialized_expr(*, raw_alias: str) -> str:
    """SQL truth of the index materializing this raw.

    A session names the raw it was written from. A retained agent work event
    is its own logical source and is materialized as an event row on the
    session it annotates, so its evidence is that session.
    """
    return (
        f"(EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = {raw_alias}.raw_id)"
        f" OR ({raw_alias}.raw_id GLOB '{WORK_EVENT_RAW_ID_PREFIX}*' AND EXISTS("
        f"SELECT 1 FROM idx_tier.sessions s WHERE s.origin = {raw_alias}.origin"
        f" AND s.native_id = {raw_alias}.native_id)))"
    )


def logical_head_cohort_expr(conn: sqlite3.Connection, *, raw_alias: str) -> str:
    """Return the durable identity used to group raw revisions into one head.

    A full-revision row retired into membership governance intentionally loses
    its raw-level ``logical_source_key``.  Its single retained membership key
    remains the authoritative identity, so use it before the legacy
    native-id/path fallback.  Shared raws can hold several membership keys;
    they have no one raw-level cohort and must keep that fallback instead of
    being arbitrarily assigned to one member.

    A cohort is a partition, so it cannot express the overlap between shared
    raws whose membership sets intersect without being equal. The
    ``shares_indexed_key`` column of :func:`raw_term_case` carries that
    relation alongside this partition.
    """
    return logical_head_cohort_sql(
        conn,
        raw_alias=raw_alias,
        has_memberships=table_exists(conn, "raw_session_memberships"),
    )


def _non_session_rules_by_origin() -> dict[str, tuple[OriginArtifactRule, ...]]:
    return {
        spec.origin.value: tuple(rule for rule in spec.artifact_rules if rule.parse_policy != "session")
        for spec in ORIGIN_SPECS
    }


def _declared_non_session_rule(
    rules_by_origin: dict[str, tuple[OriginArtifactRule, ...]], origin: str, source_path: str | None
) -> OriginArtifactRule | None:
    if not source_path:
        return None
    for rule in rules_by_origin.get(origin, ()):
        if rule.matches(source_path):
            return rule
    return None


def fragment_identity_shape(native_id: str) -> str | None:
    """Return the declared fragment/sidecar shape a session identity carries, if any."""
    for prefix in FRAGMENT_IDENTITY_PREFIXES:
        if native_id.startswith(prefix):
            return f"prefix:{prefix}"
    for suffix, kind in ARTIFACT_IDENTITY_SUFFIXES:
        if native_id.endswith(suffix):
            return f"suffix:{suffix}:{kind}"
    return None


def _inventory_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_ctime_ns, info.st_mtime_ns, info.st_size


def _member_inventory(
    container: Path,
    inventories: dict[Path, tuple[tuple[int, int, int, int, int], frozenset[str] | bool]],
) -> frozenset[str] | bool | None:
    """Measure a container through one descriptor; cache only unchanged evidence."""
    try:
        descriptor = os.open(container, os.O_RDONLY | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return False
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        try:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return False
            before = _inventory_identity(info)
            cached = inventories.get(container)
            if cached is not None and cached[0] == before:
                names = cached[1]
            else:
                try:
                    with zipfile.ZipFile(stream) as archive:
                        names = frozenset(info.filename for info in archive.infolist() if not info.is_dir())
                except zipfile.BadZipFile:
                    names = False
            if before != _inventory_identity(os.fstat(stream.fileno())):
                return None
            if before != _inventory_identity(container.stat()):
                return None
            inventories[container] = before, names
            return names
        except OSError:
            # An admitted container that disappears during the measurement
            # leaves a retryable observation, not a proof of permanent loss.
            return None


def _source_presence(
    archive_root: Path,
    source_path: str,
    inventories: dict[Path, tuple[tuple[int, int, int, int, int], frozenset[str] | bool]],
    *,
    captured_coordinate: str | None = None,
) -> bool | None:
    """Present, proven absent, or unavailable source evidence for this audit.

    A non-ZIP container proves no members remain. A permission or I/O fault
    cannot prove loss. Inventories are scoped to one audit and bound to the
    opened container's identity, never reused after a replacement or rewrite.
    """
    if captured_coordinate is not None:
        coordinate = read_captured_zip_coordinate_receipt(captured_coordinate)
        names = _member_inventory(Path(coordinate.canonical_container), inventories)
        return coordinate.member_name in names if isinstance(names, frozenset) else names
    direct = Path(source_path)
    if not direct.is_absolute():
        direct = archive_root / direct
    try:
        info = direct.stat()
    except (FileNotFoundError, NotADirectoryError):
        pass
    except OSError:
        return None
    else:
        return stat.S_ISREG(info.st_mode)
    return False


def typed_raw_cte(conn: sqlite3.Connection, *, name: str) -> str:
    """Return CTE text (no leading ``WITH``) binding ``name`` to typed raws.

    The named CTE carries ``raw_id`` and the ladder's verdict as ``term``, so
    another check can compose it beside its own CTEs and ask "does the one
    ladder explain this raw?" without restating the ladder.  ``rn = 1`` selects
    the logical head of each revision cohort.
    """
    heads_cte, term_case = raw_term_case(conn, cte_name=f"{name}__rows")
    return f"{heads_cte.strip().removeprefix('WITH ')}, {name} AS (SELECT *, {term_case} AS term FROM {name}__rows)"


def raw_term_case(conn: sqlite3.Connection, *, cte_name: str = "heads") -> tuple[str, str]:
    """Return the heads CTE and the CASE expression typing every raw row.

    The one ladder that types an acquired raw. ``source-index-coverage``
    consumes it through :func:`typed_raw_cte`, so a raw explained here can
    never be reported as untyped there; ``rn = 1`` selects the logical head of
    each revision cohort. ``conn`` is the source tier with the index tier
    attached as ``idx_tier``.
    """
    has_artifacts = table_exists(conn, "raw_artifacts")
    has_census = table_exists(conn, "raw_membership_census")
    census_expr = "(SELECT c.status FROM raw_membership_census c WHERE c.raw_id = r.raw_id)" if has_census else "NULL"
    kind_expr = (
        "(SELECT a.artifact_kind FROM raw_artifacts a WHERE a.raw_id = r.raw_id ORDER BY a.artifact_id LIMIT 1)"
        if has_artifacts
        else "NULL"
    )
    support_expr = (
        "(SELECT a.support_status FROM raw_artifacts a WHERE a.raw_id = r.raw_id ORDER BY a.artifact_id LIMIT 1)"
        if has_artifacts
        else "NULL"
    )
    parse_as_session_expr = (
        "(SELECT a.parse_as_session FROM raw_artifacts a WHERE a.raw_id = r.raw_id ORDER BY a.artifact_id LIMIT 1)"
        if has_artifacts
        else "NULL"
    )
    retained_expr = (
        "(r.blob_hash IS NOT NULL AND EXISTS(SELECT 1 FROM blob_refs b WHERE b.blob_hash = r.blob_hash))"
        if table_exists(conn, "blob_refs")
        else "(r.blob_hash IS NOT NULL)"
    )
    supersession_expr = valid_byte_duplicate_supersession_expr(conn, raw_alias="r")
    cohort_expr = logical_head_cohort_expr(conn, raw_alias="r")
    materialized_expr = raw_materialized_expr(raw_alias="r")
    # The authority frontier records, per unresolved blocker, which raw it
    # accepted as the head and which raw the index actually materialized. The
    # blocker's own reason is the rule, so cite it rather than restate it.
    # Extracted once per blocker into its own CTE: as a correlated subquery
    # this would re-parse every blocker's JSON for every raw row.
    has_blockers = table_exists(conn, "raw_authority_blockers")
    blocked_cte = (
        """
        blocked_heads AS (
            SELECT json_extract(b.expected_json, '$.index_preconditions.head_accepted_raw_id') AS head_raw_id,
                   MIN(b.reason) AS reason
            FROM raw_authority_blockers b
            WHERE b.resolved_at_ms IS NULL
            GROUP BY 1
        ),
        """
        if has_blockers
        else ""
    )
    blocked_join = "LEFT JOIN blocked_heads ON blocked_heads.head_raw_id = r.raw_id" if has_blockers else ""
    blocker_reason_expr = "blocked_heads.reason" if has_blockers else "NULL"
    # A membership row is the durable statement "this raw belongs to that
    # logical source". When every one of them is quarantined and no revision of
    # the cohort reached the index, the logical session itself is missing.
    has_memberships = table_exists(conn, "raw_session_memberships")
    quarantined_cohort_expr = (
        """
        (SELECT COUNT(*) > 0 AND COUNT(*) = SUM(m.revision_authority = 'quarantined')
         FROM raw_session_memberships m WHERE m.raw_id = r.raw_id)
        """
        if has_memberships
        else "0"
    )
    indexed_keys_cte = (
        """
        indexed_logical_keys AS (
            SELECT DISTINCT m.logical_source_key AS logical_source_key
            FROM raw_session_memberships m
            JOIN idx_tier.sessions s ON s.raw_id = m.raw_id
        ),
        """
        if has_memberships
        else ""
    )
    # Shares a logical source with a materialized raw. Membership sets overlap
    # without being equal, so this relation is not a partition and the cohort
    # window cannot carry it; it widens ``any_indexed``, never narrows it.
    shares_indexed_key_expr = (
        """
        EXISTS(
            SELECT 1 FROM raw_session_memberships m
            JOIN indexed_logical_keys k ON k.logical_source_key = m.logical_source_key
            WHERE m.raw_id = r.raw_id
        )
        """
        if has_memberships
        else "0"
    )
    heads_cte = f"""
        WITH {blocked_cte}{indexed_keys_cte}{cte_name} AS (
            SELECT
                r.raw_id,
                r.origin,
                r.source_path,
                r.blob_hash,
                r.revision_authority,
                r.parse_error,
                r.parsed_at_ms,
                r.validation_status,
                {census_expr} AS census_status,
                {kind_expr} AS artifact_kind,
                {support_expr} AS support_status,
                {parse_as_session_expr} AS parse_as_session,
                {supersession_expr} AS valid_supersession,
                {retained_expr} AS bytes_retained,
                {blocker_reason_expr} AS blocker_reason,
                {quarantined_cohort_expr} AS memberships_all_quarantined,
                {materialized_expr} AS self_indexed,
                ({shares_indexed_key_expr}) AS shares_indexed_key,
                MAX({materialized_expr})
                    OVER (PARTITION BY r.origin, {cohort_expr}) AS any_indexed,
                ROW_NUMBER() OVER (
                    PARTITION BY r.origin, {cohort_expr}
                    ORDER BY {raw_receipt_order_sql("r")} DESC, r.raw_id DESC
                ) AS rn
            FROM raw_sessions r
            {blocked_join}
        )
    """
    term_case = f"""
        CASE
            WHEN self_indexed = 1 THEN '{_TERM_MATERIALIZED}'
            WHEN any_indexed = 1 OR shares_indexed_key = 1 THEN '{_TERM_REVISION_SUPERSEDED}'
            WHEN valid_supersession = 1 THEN '{_TERM_BYTE_DUPLICATE}'
            WHEN instr(parse_error, '{VALUE_BOUND_REFUSED_HEAD}') > 0 THEN '{_TERM_VALUE_BOUND_REFUSED}'
            WHEN parse_error IS NOT NULL THEN '{_TERM_PARSE_FAILURE}'
            WHEN validation_status = 'failed' THEN '{_TERM_VALIDATION_REJECTED}'
            WHEN parse_as_session = 0 AND artifact_kind IS NOT NULL AND artifact_kind != 'unknown'
                THEN '{_TERM_NON_SESSION_ARTIFACT}'
            WHEN support_status = 'decode_failed' THEN '{_TERM_DECODE_FAILED}'
            WHEN census_status IN ('non_session', 'failed') THEN '{_TERM_CENSUS_NON_SESSION}'
            WHEN artifact_kind = 'unknown' THEN '{_TERM_UNCLASSIFIED_SHAPE}'
            WHEN parsed_at_ms IS NULL THEN '{_TERM_PENDING}'
            WHEN blocker_reason IS NOT NULL THEN '{_TERM_AUTHORITY_BLOCKED}'
            WHEN memberships_all_quarantined = 1 THEN '{_TERM_QUARANTINED_COHORT}'
            ELSE '{_TERM_UNEXPLAINED}'
        END
    """
    return heads_cte, term_case


def _sample(rows: Iterable[tuple[Any, ...]], limit: int) -> tuple[str, ...]:
    out: list[str] = []
    for row in rows:
        if len(out) >= limit:
            break
        out.append(str(row[0]))
    return tuple(out)


def audit_source_conservation(
    conn: sqlite3.Connection,
    *,
    archive_root: Path,
    sample_limit: int = 10,
    probe_filesystem: bool = True,
    frontier: SourceFrontier | None = None,
) -> SourceConservationReport:
    """Type every acquired source item and every index row; ``conn`` is the
    source tier with the index tier attached as ``idx_tier`` (read-only)."""
    if frontier is not None:
        frontier.verify_integrity()
    from polylogue.storage.blob_store import BlobStore

    blob_store = BlobStore(archive_root / "blob")
    heads_cte, term_case = raw_term_case(conn)
    forward_total = int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])

    typed_rows = conn.execute(
        f"{heads_cte} SELECT raw_id, origin, source_path, artifact_kind, bytes_retained, blocker_reason, "
        f"blob_hash, {term_case} AS term, "
        "(SELECT captured_coordinate FROM raw_container_coordinates AS coordinate "
        "WHERE coordinate.raw_id = heads.raw_id) AS captured_coordinate FROM heads"
    )

    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    breakdowns: dict[str, dict[str, int]] = {}
    inventories: dict[Path, tuple[tuple[int, int, int, int, int], frozenset[str] | bool]] = {}
    for (
        raw_id,
        origin,
        source_path,
        artifact_kind,
        bytes_retained,
        blocker_reason,
        blob_hash,
        term,
        coordinate,
    ) in typed_rows:
        # Archive retention is independent of whether reacquisition is possible.
        # Probe CAS metadata only; body fidelity belongs to retained-byte validation.
        retained = bool(bytes_retained)
        if blob_hash is not None:
            digest = bytes(blob_hash).hex() if isinstance(blob_hash, (bytes, memoryview)) else str(blob_hash)
            retained = retained and blob_store.exists(digest)
        present: bool | None = True
        # A work event's retained raw is its source; it has no acquired file.
        if probe_filesystem and not is_work_event_raw_id(str(raw_id)):
            present = _source_presence(archive_root, str(source_path), inventories, captured_coordinate=coordinate)
        if not retained:
            term = _TERM_SOURCE_LOST if present is False else _TERM_MISSING_BLOB
        elif present is None:
            term = _TERM_SOURCE_UNAVAILABLE
        elif not present:
            term = _TERM_SOURCE_MISSING
        counts[term] = counts.get(term, 0) + 1
        bucket = samples.setdefault(term, [])
        if len(bucket) < sample_limit:
            bucket.append(str(raw_id))
        if term == _TERM_NON_SESSION_ARTIFACT:
            key = f"{origin}:{artifact_kind}"
        elif term == _TERM_AUTHORITY_BLOCKED:
            key = f"{origin}:{blocker_reason}"
        else:
            key = str(origin)
        by = breakdowns.setdefault(term, {})
        by[key] = by.get(key, 0) + 1

    # Hook events.
    hook_total = 0
    hook_counts: dict[str, int] = {}
    hook_samples: dict[str, tuple[str, ...]] = {}
    if table_exists(conn, "raw_hook_events"):
        hook_case = f"""
            CASE
                WHEN h.session_native_id IS NULL THEN '{_TERM_HOOK_NO_SESSION_ID}'
                WHEN EXISTS(SELECT 1 FROM idx_tier.sessions s
                            WHERE s.session_id = h.origin || ':' || h.session_native_id)
                    THEN '{_TERM_HOOK_MATERIALIZED}'
                WHEN EXISTS(SELECT 1 FROM raw_sessions r
                            WHERE r.origin = h.origin AND r.native_id = h.session_native_id)
                    THEN '{_TERM_HOOK_ACQUIRED}'
                ELSE '{_TERM_HOOK_NO_SOURCE}'
            END
        """
        for term, count in conn.execute(
            f"SELECT {hook_case} AS term, COUNT(*) FROM raw_hook_events h GROUP BY term"
        ).fetchall():
            hook_counts[str(term)] = int(count)
            hook_total += int(count)
        hook_samples[_TERM_HOOK_NO_SOURCE] = _sample(
            conn.execute(
                f"SELECT h.hook_event_id FROM raw_hook_events h WHERE ({hook_case}) = ? LIMIT ?",
                (_TERM_HOOK_NO_SOURCE, sample_limit),
            ),
            sample_limit,
        )

    sidecar_total = 0
    if table_exists(conn, "history_sidecars"):
        sidecar_total = int(conn.execute("SELECT COUNT(*) FROM history_sidecars").fetchone()[0])

    # Reverse direction.
    session_total = int(conn.execute("SELECT COUNT(*) FROM idx_tier.sessions").fetchone()[0])
    without_raw_count = int(conn.execute("SELECT COUNT(*) FROM idx_tier.sessions WHERE raw_id IS NULL").fetchone()[0])
    without_raw = _sample(
        conn.execute(
            "SELECT session_id FROM idx_tier.sessions WHERE raw_id IS NULL ORDER BY session_id LIMIT ?",
            (sample_limit,),
        ),
        sample_limit,
    )
    orphan_query = """
        SELECT s.session_id FROM idx_tier.sessions s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id)
        ORDER BY s.session_id
    """
    session_orphan_count = int(
        conn.execute(
            """SELECT COUNT(*) FROM idx_tier.sessions s
               WHERE s.raw_id IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id)"""
        ).fetchone()[0]
    )
    session_orphans = _sample(conn.execute(f"{orphan_query} LIMIT ?", (sample_limit,)), sample_limit)

    has_artifacts = table_exists(conn, "raw_artifacts")
    parse_as_session_expr = (
        "(SELECT a.parse_as_session FROM raw_artifacts a WHERE a.raw_id = r.raw_id ORDER BY a.artifact_id LIMIT 1)"
        if has_artifacts
        else "NULL"
    )
    kind_expr = (
        "(SELECT a.artifact_kind FROM raw_artifacts a WHERE a.raw_id = r.raw_id ORDER BY a.artifact_id LIMIT 1)"
        if has_artifacts
        else "NULL"
    )
    rules_by_origin = _non_session_rules_by_origin()
    phantom_lineage_count = 0
    phantom_lineage_samples: list[str] = []
    phantom_lineage_breakdown: dict[str, int] = {}
    phantom_identity_count = 0
    phantom_identity_samples: list[str] = []
    phantom_identity_breakdown: dict[str, int] = {}
    for session_id, native_id, source_path, parse_as_session, artifact_kind, acquisition_origin in conn.execute(
        f"""
        SELECT s.session_id, s.native_id, r.source_path, {parse_as_session_expr}, {kind_expr}, r.origin
        FROM idx_tier.sessions s
        JOIN raw_sessions r ON r.raw_id = s.raw_id
        """
    ):
        lineage_class: str | None = None
        if (
            parse_as_session == 0
            and artifact_kind is not None
            and artifact_kind != "unknown"
            and artifact_kind != "terminal_superseded_deferred_cas_frontier"
        ):
            lineage_class = f"artifact:{artifact_kind}"
        else:
            if artifact_kind in {
                "terminal_superseded_deferred_cas_frontier",
                "raw_failure_carrier",
                "resolution_carrier",
            }:
                rule = None
            else:
                rule = _declared_non_session_rule(rules_by_origin, str(acquisition_origin), source_path)
            if rule is not None:
                lineage_class = f"rule:{rule.kind}"
        if lineage_class is not None:
            phantom_lineage_count += 1
            if len(phantom_lineage_samples) < sample_limit:
                phantom_lineage_samples.append(str(session_id))
            phantom_lineage_breakdown[lineage_class] = phantom_lineage_breakdown.get(lineage_class, 0) + 1
            continue
        shape = fragment_identity_shape(str(native_id))
        if shape is not None:
            phantom_identity_count += 1
            if len(phantom_identity_samples) < sample_limit:
                phantom_identity_samples.append(str(session_id))
            phantom_identity_breakdown[shape] = phantom_identity_breakdown.get(shape, 0) + 1

    message_orphans = conn.execute(
        """
        SELECT m.message_id FROM idx_tier.messages m
        WHERE NOT EXISTS (SELECT 1 FROM idx_tier.sessions s WHERE s.session_id = m.session_id)
        LIMIT ?
        """,
        (sample_limit,),
    ).fetchall()
    message_orphan_count = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM idx_tier.messages m
            WHERE NOT EXISTS (SELECT 1 FROM idx_tier.sessions s WHERE s.session_id = m.session_id)
            """
        ).fetchone()[0]
    )
    block_orphan_count = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM idx_tier.blocks b
            WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = b.message_id)
            """
        ).fetchone()[0]
    )
    block_orphans = conn.execute(
        """
        SELECT b.block_id FROM idx_tier.blocks b
        WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = b.message_id)
        LIMIT ?
        """,
        (sample_limit,),
    ).fetchall()
    attachment_ref_orphan_count = 0
    attachment_ref_orphans: list[tuple[Any, ...]] = []
    attachment_unreferenced_count = 0
    attachment_unreferenced: list[tuple[Any, ...]] = []
    attachment_unowned_count = 0
    attachment_unowned: list[tuple[Any, ...]] = []
    attachment_owner_missing_count = 0
    attachment_owner_missing: list[tuple[Any, ...]] = []
    attachment_missing_count = 0
    attachment_missing: list[tuple[Any, ...]] = []
    if table_exists(conn, "attachment_refs", schema="idx_tier"):
        attachment_ref_orphan_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM idx_tier.attachment_refs ar
                WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = ar.message_id AND m.session_id = ar.session_id)
                """
            ).fetchone()[0]
        )
        attachment_ref_orphans = conn.execute(
            """
            SELECT ar.ref_id FROM idx_tier.attachment_refs ar
            WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = ar.message_id AND m.session_id = ar.session_id)
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        # A ref-less attachment splits on ``ref_count``. The writer inserts an
        # owner-ambiguous row with ref_count 0 and keeps it out of the sweep,
        # so ref_count 0 is explained and non-blocking. A non-zero ref_count is
        # the witness that refs existed and went away without the sweep
        # running, leaving the row unreachable from every read path.
        unreferenced_predicate = """
            NOT EXISTS (SELECT 1 FROM idx_tier.attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
        """
        attachment_unreferenced_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM idx_tier.attachments a
                WHERE {unreferenced_predicate} AND a.ref_count != 0
                """
            ).fetchone()[0]
        )
        attachment_unreferenced = conn.execute(
            f"""
            SELECT a.attachment_id FROM idx_tier.attachments a
            WHERE {unreferenced_predicate} AND a.ref_count != 0
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        # ref_count 0 is explained by the writer's recorded owner gap, except
        # a named owner that no written message carries: that is a lost owner.
        owner_missing = """EXISTS (
            SELECT 1 FROM idx_tier.attachment_owner_gaps g
            WHERE g.attachment_id = a.attachment_id AND g.reason = 'message_missing'
        )"""
        attachment_unowned_count = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM idx_tier.attachments a
                WHERE {unreferenced_predicate} AND a.ref_count = 0 AND NOT {owner_missing}
                """
            ).fetchone()[0]
        )
        attachment_unowned = conn.execute(
            f"""
            SELECT a.attachment_id FROM idx_tier.attachments a
            WHERE {unreferenced_predicate} AND a.ref_count = 0 AND NOT {owner_missing}
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        # Counted from the gap ledger, not from attachment rows: a lost owner
        # whose bytes were never retained leaves no attachment row at all.
        attachment_owner_missing_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM idx_tier.attachment_owner_gaps WHERE reason = 'message_missing'"
            ).fetchone()[0]
        )
        attachment_owner_missing = conn.execute(
            """
            SELECT attachment_id FROM idx_tier.attachment_owner_gaps
            WHERE reason = 'message_missing'
            ORDER BY session_id, attachment_id
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        attachment_missing_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM idx_tier.attachment_refs ar
                WHERE NOT EXISTS (
                    SELECT 1 FROM idx_tier.attachments a WHERE a.attachment_id = ar.attachment_id
                )
                """
            ).fetchone()[0]
        )
        attachment_missing = conn.execute(
            """
            SELECT ar.attachment_id FROM idx_tier.attachment_refs ar
            WHERE NOT EXISTS (
                SELECT 1 FROM idx_tier.attachments a WHERE a.attachment_id = ar.attachment_id
            )
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()

    frontier_counts: dict[str, int] = {}
    frontier_samples: dict[str, list[str]] = {}
    if frontier is not None:
        # Spill the source denominator into the connection's file-backed TEMP
        # schema, then let SQLite join it to raw acquisition by path and digest.
        # Neither side is collected in Python; the raw DB remains opened read-only.
        with readonly_temp_staging(conn, temp_store="FILE"):
            frontier.copy_members_to(conn)
        owner_rows = conn.execute(
            """SELECT m.source_id, m.coordinate, COUNT(r.raw_id) AS owner_count
               FROM temp._polylogue_source_frontier_member m
               LEFT JOIN raw_sessions r
                 ON r.source_path = m.source_path
                AND lower(hex(r.blob_hash)) = lower(m.content_sha256)
               GROUP BY m.ordinal, m.source_id, m.coordinate
               ORDER BY m.ordinal"""
        )
        while page := owner_rows.fetchmany(512):
            for source_id, coordinate, owner_count in page:
                if owner_count == 0:
                    name = _TERM_FRONTIER_UNACQUIRED
                elif owner_count > 1:
                    name = _TERM_FRONTIER_DUPLICATE
                else:
                    continue
                frontier_counts[name] = frontier_counts.get(name, 0) + 1
                sample_bucket = frontier_samples.setdefault(name, [])
                if len(sample_bucket) < sample_limit:
                    sample_bucket.append(f"{source_id}:{coordinate}")
        for blocker in frontier.blockers:
            frontier_counts[_TERM_FRONTIER_UNAVAILABLE] = frontier_counts.get(_TERM_FRONTIER_UNAVAILABLE, 0) + 1
            sample_bucket = frontier_samples.setdefault(_TERM_FRONTIER_UNAVAILABLE, [])
            if len(sample_bucket) < sample_limit:
                sample_bucket.append(blocker)
        # Historical revisions at a configured path are already typed by the
        # forward classifier. Only raws outside every configured coordinate
        # are frontier orphans.
        frontier_orphan_count = int(
            conn.execute(
                """SELECT COUNT(*) FROM raw_sessions r
                   WHERE NOT EXISTS (
                       SELECT 1 FROM temp._polylogue_source_frontier_path p
                       WHERE p.source_path = r.source_path
                   )"""
            ).fetchone()[0]
        )
        orphan_rows = conn.execute(
            """SELECT r.raw_id FROM raw_sessions r
               WHERE NOT EXISTS (
                   SELECT 1 FROM temp._polylogue_source_frontier_path p
                   WHERE p.source_path = r.source_path
               ) ORDER BY r.raw_id LIMIT ?""",
            (sample_limit,),
        )
        frontier_counts[_TERM_FRONTIER_ORPHAN] = frontier_orphan_count
        frontier_samples[_TERM_FRONTIER_ORPHAN] = [str(row[0]) for row in orphan_rows]
        # Membership content is an independent semantic witness.  A row that
        # keeps its identity but changes its normalized content must not pass
        # merely because the raw was acquired and a session row exists.
        #
        # One raw can own several logical sessions (a grouped JSONL or ZIP
        # member emits one membership row per session it carries). Joining
        # membership to session on ``raw_id`` alone pairs every membership with
        # every sibling session, and the off-diagonal pairs disagree by
        # construction, so a correct multi-session acquisition reported
        # ``content_mismatch``. The witness is per MEMBERSHIP: its normalized
        # content must be the content of one of the sessions that raw
        # materialized. A permutation among the siblings of a single raw is not
        # distinguished -- that is not a source-conservation failure mode,
        # while the false positive was a blocking one.
        if table_exists(conn, "raw_session_memberships"):
            mismatch_predicate = """
                EXISTS (SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = m.raw_id)
                AND NOT EXISTS (
                    SELECT 1 FROM idx_tier.sessions s
                    WHERE s.raw_id = m.raw_id
                      AND s.content_hash IS NOT NULL
                      AND m.normalized_content_hash IS NOT NULL
                      AND s.content_hash = m.normalized_content_hash
                )
            """
            mismatches = conn.execute(
                f"""
                SELECT m.raw_id
                FROM raw_session_memberships m
                WHERE {mismatch_predicate}
                ORDER BY m.raw_id, m.logical_source_key
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()
            mismatch_count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM raw_session_memberships m
                    WHERE {mismatch_predicate}
                    """
                ).fetchone()[0]
            )
            if mismatch_count:
                frontier_counts[_TERM_CONTENT_MISMATCH] = mismatch_count
                frontier_samples[_TERM_CONTENT_MISMATCH] = [str(row[0]) for row in mismatches]

    def _term(
        name: str, count: int, sample: tuple[str, ...] = (), breakdown: dict[str, int] | None = None
    ) -> ConservationTerm:
        return ConservationTerm(
            name=name,
            count=count,
            rule=_RULES[name],
            blocking=name in _BLOCKING,
            sample=sample,
            breakdown=dict(breakdown or {}),
        )

    forward_order = (
        _TERM_SOURCE_MISSING,
        _TERM_SOURCE_LOST,
        _TERM_MISSING_BLOB,
        _TERM_SOURCE_UNAVAILABLE,
        _TERM_MATERIALIZED,
        _TERM_REVISION_SUPERSEDED,
        _TERM_BYTE_DUPLICATE,
        _TERM_VALUE_BOUND_REFUSED,
        _TERM_PARSE_FAILURE,
        _TERM_VALIDATION_REJECTED,
        _TERM_NON_SESSION_ARTIFACT,
        _TERM_DECODE_FAILED,
        _TERM_CENSUS_NON_SESSION,
        _TERM_UNCLASSIFIED_SHAPE,
        _TERM_PENDING,
        _TERM_AUTHORITY_BLOCKED,
        _TERM_QUARANTINED_COHORT,
        _TERM_UNEXPLAINED,
    )
    terms: list[ConservationTerm] = [
        _term(name, counts.get(name, 0), tuple(samples.get(name, ())), breakdowns.get(name)) for name in forward_order
    ]
    for name in (_TERM_HOOK_MATERIALIZED, _TERM_HOOK_ACQUIRED, _TERM_HOOK_NO_SESSION_ID, _TERM_HOOK_NO_SOURCE):
        terms.append(_term(name, hook_counts.get(name, 0), hook_samples.get(name, ())))
    terms.append(_term(_TERM_SIDECAR_RETAINED, sidecar_total))
    terms.extend(
        (
            _term(_TERM_SESSION_WITHOUT_RAW, without_raw_count, without_raw),
            _term(_TERM_SESSION_ORPHAN, session_orphan_count, session_orphans),
            _term(
                _TERM_PHANTOM_LINEAGE,
                phantom_lineage_count,
                tuple(phantom_lineage_samples),
                phantom_lineage_breakdown,
            ),
            _term(
                _TERM_PHANTOM_IDENTITY,
                phantom_identity_count,
                tuple(phantom_identity_samples),
                phantom_identity_breakdown,
            ),
            _term(_TERM_MESSAGE_ORPHAN, message_orphan_count, _sample(message_orphans, sample_limit)),
            _term(_TERM_BLOCK_ORPHAN, block_orphan_count, _sample(block_orphans, sample_limit)),
            _term(
                _TERM_ATTACHMENT_REF_ORPHAN, attachment_ref_orphan_count, _sample(attachment_ref_orphans, sample_limit)
            ),
            _term(
                _TERM_ATTACHMENT_UNREFERENCED,
                attachment_unreferenced_count,
                _sample(attachment_unreferenced, sample_limit),
            ),
            _term(
                _TERM_ATTACHMENT_UNOWNED,
                attachment_unowned_count,
                _sample(attachment_unowned, sample_limit),
            ),
            _term(
                _TERM_ATTACHMENT_OWNER_MISSING,
                attachment_owner_missing_count,
                _sample(attachment_owner_missing, sample_limit),
            ),
            _term(_TERM_ATTACHMENT_MISSING, attachment_missing_count, _sample(attachment_missing, sample_limit)),
        )
    )
    if frontier is not None:
        for name in (
            _TERM_FRONTIER_UNAVAILABLE,
            _TERM_FRONTIER_UNACQUIRED,
            _TERM_FRONTIER_DUPLICATE,
            _TERM_FRONTIER_ORPHAN,
            _TERM_CONTENT_MISMATCH,
        ):
            terms.append(_term(name, frontier_counts.get(name, 0), tuple(frontier_samples.get(name, ()))))
    return SourceConservationReport(
        forward_total=forward_total,
        hook_total=hook_total,
        sidecar_total=sidecar_total,
        session_total=session_total,
        terms=tuple(terms),
        frontier_sha256=frontier.frontier_sha256 if frontier is not None else None,
        frontier_total=frontier.item_count if frontier is not None else 0,
        frontier_bytes=frontier.byte_count if frontier is not None else 0,
        frontier_complete=frontier.complete if frontier is not None else None,
        frontier_root_states=(
            {source_id: state.value for source_id, state in frontier.root_states.items()}
            if frontier is not None
            else {}
        ),
    )


__all__ = [
    "ARTIFACT_IDENTITY_SUFFIXES",
    "FRAGMENT_IDENTITY_PREFIXES",
    "TYPED_ABSENCE_TERMS",
    "ConservationTerm",
    "SourceConservationReport",
    "audit_source_conservation",
    "fragment_identity_shape",
    "logical_head_cohort_expr",
    "raw_materialized_expr",
    "raw_term_case",
    "term_rule",
    "typed_raw_cte",
    "valid_byte_duplicate_supersession_expr",
]


#: Terms that name a durable state explaining why a head never materialized.
#: A head carrying one of these is typed, whatever else is true of it: a check
#: that reports it as having *no* typed state is reporting its own blind spot
#: (polylogue-5tkbt). ``unclassified_shape`` is deliberately absent -- its rule
#: says a classification is missing, which is untypedness, not an explanation.
TYPED_ABSENCE_TERMS: frozenset[str] = frozenset(
    {
        _TERM_VALIDATION_REJECTED,
        _TERM_NON_SESSION_ARTIFACT,
        _TERM_DECODE_FAILED,
        _TERM_AUTHORITY_BLOCKED,
        _TERM_QUARANTINED_COHORT,
        _TERM_VALUE_BOUND_REFUSED,
    }
)


def term_rule(name: str) -> str:
    """Return the declared rule that explains one term."""
    return _RULES[name]
