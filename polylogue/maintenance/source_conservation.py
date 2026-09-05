"""Source-to-archive conservation: every acquired source item types into one term.

Backs the ``source-conservation`` owner check of ``verify-archive``. The
forward universe is the durable acquisition ledger (``raw_sessions``,
``raw_hook_events``, ``history_sidecars``); the reverse universe is every
index row (sessions, messages, blocks, attachment refs). Each item lands in
exactly one term, in a fixed precedence, and each term cites the rule that
explains it. An item no rule explains is an *unexplained* term and turns the
check red; a typed exclusion never does.

Phantom sessions (polylogue-b508) are a reverse-direction term: an index
session whose only source lineage is a declared non-session artifact
(sidecar, workflow journal, tool-result fragment, metadata fragment) or whose
identity carries an artifact-derived shape. They are reported as a
current-producer failure and never deleted here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.archive.revision_authority import logical_head_cohort_sql
from polylogue.core.json import JSONDocument, json_document
from polylogue.sources.origin_specs import ORIGIN_SPECS, OriginArtifactRule
from polylogue.storage.introspection import table_exists

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
_TERM_MATERIALIZED = "materialized"
_TERM_REVISION_SUPERSEDED = "revision_superseded"
_TERM_BYTE_DUPLICATE = "byte_duplicate_superseded"
_TERM_PARSE_FAILURE = "parse_failure"
_TERM_VALIDATION_REJECTED = "validation_rejected"
_TERM_NON_SESSION_ARTIFACT = "non_session_artifact"
_TERM_DECODE_FAILED = "decode_failed"
_TERM_CENSUS_NON_SESSION = "census_non_session"
_TERM_UNCLASSIFIED_SHAPE = "unclassified_shape"
_TERM_PENDING = "pending"
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

_RULES: dict[str, str] = {
    _TERM_SOURCE_MISSING: ("acquired source file no longer exists on disk; the archive retains its raw payload bytes"),
    _TERM_SOURCE_LOST: (
        "acquired source file no longer exists on disk and no raw payload blob is retained; the bytes are gone"
    ),
    _TERM_MATERIALIZED: "index session carries this raw_id",
    _TERM_REVISION_SUPERSEDED: "another revision of the same logical source is materialized",
    _TERM_BYTE_DUPLICATE: "content-bound byte-duplicate supersession receipt names a materialized twin",
    _TERM_PARSE_FAILURE: "raw_sessions.parse_error records the typed parser refusal",
    _TERM_VALIDATION_REJECTED: "raw_sessions.validation_status = 'failed' records the schema refusal",
    _TERM_NON_SESSION_ARTIFACT: "raw_artifacts declares the item a non-session artifact kind",
    _TERM_DECODE_FAILED: "raw_artifacts.decode_error records the typed decode failure",
    _TERM_CENSUS_NON_SESSION: "raw_membership_census recorded a terminal non-session verdict",
    _TERM_UNCLASSIFIED_SHAPE: "artifact taxonomy holds no classification (unknown/unknown); a rule is missing",
    _TERM_PENDING: "acquired; convergence has not parsed it yet",
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
    _TERM_ATTACHMENT_UNREFERENCED: "attachment has no ref and therefore no source lineage",
}

_BLOCKING: frozenset[str] = frozenset(
    {
        _TERM_SOURCE_LOST,
        _TERM_UNCLASSIFIED_SHAPE,
        _TERM_UNEXPLAINED,
        _TERM_SESSION_WITHOUT_RAW,
        _TERM_SESSION_ORPHAN,
        _TERM_PHANTOM_LINEAGE,
        _TERM_PHANTOM_IDENTITY,
        _TERM_MESSAGE_ORPHAN,
        _TERM_BLOCK_ORPHAN,
        _TERM_ATTACHMENT_REF_ORPHAN,
        _TERM_ATTACHMENT_UNREFERENCED,
    }
)

_WARNING: frozenset[str] = frozenset({_TERM_PENDING, _TERM_HOOK_NO_SOURCE})


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


def logical_head_cohort_expr(conn: sqlite3.Connection, *, raw_alias: str) -> str:
    """Return the durable identity used to group raw revisions into one head.

    A full-revision row retired into membership governance intentionally loses
    its raw-level ``logical_source_key``.  Its single retained membership key
    remains the authoritative identity, so use it before the legacy
    native-id/path fallback.  Shared raws can hold several membership keys;
    they have no one raw-level cohort and must keep that fallback instead of
    being arbitrarily assigned to one member.
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


def _source_exists(archive_root: Path, source_path: str) -> bool:
    path = Path(source_path)
    if not path.is_absolute():
        path = archive_root / path
    return path.exists()


def _raw_term_case(conn: sqlite3.Connection) -> tuple[str, str]:
    """Return the ``heads`` CTE and the CASE expression typing every raw row."""
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
    heads_cte = f"""
        WITH heads AS (
            SELECT
                r.raw_id,
                r.origin,
                r.source_path,
                r.parse_error,
                r.parsed_at_ms,
                r.validation_status,
                {census_expr} AS census_status,
                {kind_expr} AS artifact_kind,
                {support_expr} AS support_status,
                {parse_as_session_expr} AS parse_as_session,
                {supersession_expr} AS valid_supersession,
                {retained_expr} AS bytes_retained,
                EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = r.raw_id) AS self_indexed,
                MAX(EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = r.raw_id))
                    OVER (PARTITION BY r.origin, {cohort_expr}) AS any_indexed
            FROM raw_sessions r
        )
    """
    term_case = f"""
        CASE
            WHEN self_indexed = 1 THEN '{_TERM_MATERIALIZED}'
            WHEN any_indexed = 1 THEN '{_TERM_REVISION_SUPERSEDED}'
            WHEN valid_supersession = 1 THEN '{_TERM_BYTE_DUPLICATE}'
            WHEN parse_error IS NOT NULL THEN '{_TERM_PARSE_FAILURE}'
            WHEN validation_status = 'failed' THEN '{_TERM_VALIDATION_REJECTED}'
            WHEN parse_as_session = 0 AND artifact_kind IS NOT NULL AND artifact_kind != 'unknown'
                THEN '{_TERM_NON_SESSION_ARTIFACT}'
            WHEN support_status = 'decode_failed' THEN '{_TERM_DECODE_FAILED}'
            WHEN census_status IN ('non_session', 'failed') THEN '{_TERM_CENSUS_NON_SESSION}'
            WHEN artifact_kind = 'unknown' THEN '{_TERM_UNCLASSIFIED_SHAPE}'
            WHEN parsed_at_ms IS NULL THEN '{_TERM_PENDING}'
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
) -> SourceConservationReport:
    """Type every acquired source item and every index row; ``conn`` is the
    source tier with the index tier attached as ``idx_tier`` (read-only)."""
    heads_cte, term_case = _raw_term_case(conn)
    forward_total = int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])

    typed_rows = conn.execute(
        f"{heads_cte} SELECT raw_id, origin, source_path, artifact_kind, bytes_retained, {term_case} AS term FROM heads"
    ).fetchall()

    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    breakdowns: dict[str, dict[str, int]] = {}
    missing_paths: dict[str, bool] = {}
    for raw_id, origin, source_path, artifact_kind, bytes_retained, term in typed_rows:
        if probe_filesystem:
            present = missing_paths.get(source_path)
            if present is None:
                present = _source_exists(archive_root, str(source_path))
                missing_paths[source_path] = present
            if not present:
                term = _TERM_SOURCE_MISSING if bytes_retained else _TERM_SOURCE_LOST
        counts[term] = counts.get(term, 0) + 1
        bucket = samples.setdefault(term, [])
        if len(bucket) < sample_limit:
            bucket.append(str(raw_id))
        key = str(origin) if term != _TERM_NON_SESSION_ARTIFACT else f"{origin}:{artifact_kind}"
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
    without_raw = conn.execute(
        "SELECT session_id FROM idx_tier.sessions WHERE raw_id IS NULL ORDER BY session_id"
    ).fetchall()
    orphans = conn.execute(
        """
        SELECT s.session_id FROM idx_tier.sessions s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id)
        ORDER BY s.session_id
        """
    ).fetchall()

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
    phantom_lineage: list[str] = []
    phantom_lineage_breakdown: dict[str, int] = {}
    phantom_identity: list[str] = []
    phantom_identity_breakdown: dict[str, int] = {}
    for session_id, native_id, origin, source_path, parse_as_session, artifact_kind in conn.execute(
        f"""
        SELECT s.session_id, s.native_id, s.origin, r.source_path, {parse_as_session_expr}, {kind_expr}
        FROM idx_tier.sessions s
        JOIN raw_sessions r ON r.raw_id = s.raw_id
        """
    ):
        lineage_class: str | None = None
        if parse_as_session == 0 and artifact_kind is not None and artifact_kind != "unknown":
            lineage_class = f"artifact:{artifact_kind}"
        else:
            rule = _declared_non_session_rule(rules_by_origin, str(origin), source_path)
            if rule is not None:
                lineage_class = f"rule:{rule.kind}"
        if lineage_class is not None:
            phantom_lineage.append(str(session_id))
            phantom_lineage_breakdown[lineage_class] = phantom_lineage_breakdown.get(lineage_class, 0) + 1
            continue
        shape = fragment_identity_shape(str(native_id))
        if shape is not None:
            phantom_identity.append(str(session_id))
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
    if table_exists(conn, "attachment_refs", schema="idx_tier"):
        attachment_ref_orphan_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM idx_tier.attachment_refs ar
                WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = ar.message_id)
                """
            ).fetchone()[0]
        )
        attachment_ref_orphans = conn.execute(
            """
            SELECT ar.ref_id FROM idx_tier.attachment_refs ar
            WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = ar.message_id)
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        attachment_unreferenced_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM idx_tier.attachments a
                WHERE NOT EXISTS (SELECT 1 FROM idx_tier.attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
                """
            ).fetchone()[0]
        )
        attachment_unreferenced = conn.execute(
            """
            SELECT a.attachment_id FROM idx_tier.attachments a
            WHERE NOT EXISTS (SELECT 1 FROM idx_tier.attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()

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
        _TERM_MATERIALIZED,
        _TERM_REVISION_SUPERSEDED,
        _TERM_BYTE_DUPLICATE,
        _TERM_PARSE_FAILURE,
        _TERM_VALIDATION_REJECTED,
        _TERM_NON_SESSION_ARTIFACT,
        _TERM_DECODE_FAILED,
        _TERM_CENSUS_NON_SESSION,
        _TERM_UNCLASSIFIED_SHAPE,
        _TERM_PENDING,
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
            _term(_TERM_SESSION_WITHOUT_RAW, len(without_raw), _sample(without_raw, sample_limit)),
            _term(_TERM_SESSION_ORPHAN, len(orphans), _sample(orphans, sample_limit)),
            _term(
                _TERM_PHANTOM_LINEAGE,
                len(phantom_lineage),
                tuple(phantom_lineage[:sample_limit]),
                phantom_lineage_breakdown,
            ),
            _term(
                _TERM_PHANTOM_IDENTITY,
                len(phantom_identity),
                tuple(phantom_identity[:sample_limit]),
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
        )
    )
    return SourceConservationReport(
        forward_total=forward_total,
        hook_total=hook_total,
        sidecar_total=sidecar_total,
        session_total=session_total,
        terms=tuple(terms),
    )


__all__ = [
    "ARTIFACT_IDENTITY_SUFFIXES",
    "FRAGMENT_IDENTITY_PREFIXES",
    "ConservationTerm",
    "SourceConservationReport",
    "audit_source_conservation",
    "fragment_identity_shape",
    "logical_head_cohort_expr",
    "valid_byte_duplicate_supersession_expr",
]
