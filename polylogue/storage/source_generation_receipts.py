"""Pinned, read-only evidence for one source-generation publication.

The caller owns the publication barrier and the two SQLite snapshots.  This
module deliberately does not open databases or turn the evidence into a
derivation claim: it only proves whether the source-43 manifest's exact raw
members have current parser and index witnesses.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum

from polylogue.archive.revision_authority import (
    BYTE_AUTHORITY_CENSUS_DETAIL,
    canonical_authority_logical_key,
    durable_authority_logical_keys,
    parser_census_is_complete,
)
from polylogue.archive.session_revision_membership import MembershipDecision
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.storage.raw_authority import (
    RAW_AUTHORITY_PARSER_FINGERPRINT,
    parser_census_logical_keys,
)
from polylogue.storage.sqlite.archive_tiers.revision_governance import _application_decision_for

_MAX_SOURCE_ITEMS = 10_000


class SourceGenerationBlocker(StrEnum):
    """A closed explanation for one unproven source-generation member."""

    GENERATION_MISSING = "generation_missing"
    ENUMERATION_INCOMPLETE = "enumeration_incomplete"
    RETIRED_MEMBER = "retired_member"
    PARSER_CENSUS_MISSING = "parser_census_missing"
    PARSER_CENSUS_MISMATCH = "parser_census_mismatch"
    APPLICATION_ABSENT = "application_absent"
    APPLICATION_STALE = "application_stale"
    APPLICATION_AMBIGUOUS = "application_ambiguous"
    HEAD_ABSENT = "head_absent"
    HEAD_STALE = "head_stale"
    HEAD_AMBIGUOUS = "head_ambiguous"
    SESSION_ABSENT = "session_absent"
    SESSION_STALE = "session_stale"
    SESSION_AMBIGUOUS = "session_ambiguous"


@dataclass(frozen=True, slots=True)
class SourceGenerationRetiredCoordinate:
    """A source-43 record whose foreign-key raw was retired.

    ``raw_id`` is necessarily unavailable after the durable source row is
    deleted, so the source-item and record coordinates are the only exact
    identity this read can honestly report.
    """

    source_item_id: str
    logical_coordinate: str
    record_coordinate: str


@dataclass(frozen=True, slots=True)
class SourceGenerationLogicalReceipt:
    """Index witnesses expected for one parsed logical member of one raw."""

    logical_source_key: str
    expected_session_id: str
    accepted_raw_id: str | None
    application_ids: tuple[str, ...]
    head_session_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    blockers: tuple[SourceGenerationBlocker, ...]

    @property
    def complete(self) -> bool:
        return not self.blockers


@dataclass(frozen=True, slots=True)
class SourceGenerationRawReceipt:
    """Parser census and exact current-index witnesses for one raw member."""

    raw_id: str
    parsed_at_ms: int | None
    parser_complete: bool
    parser_blockers: tuple[SourceGenerationBlocker, ...]
    logicals: tuple[SourceGenerationLogicalReceipt, ...]

    @property
    def complete(self) -> bool:
        return self.parser_complete and all(logical.complete for logical in self.logicals)


@dataclass(frozen=True, slots=True)
class SourceGenerationItemReceipt:
    """One source manifest input and every source-43 raw it enumerated."""

    source_item_id: str
    logical_coordinate: str
    enumeration_complete: bool
    blockers: tuple[SourceGenerationBlocker, ...]
    raws: tuple[SourceGenerationRawReceipt, ...]

    @property
    def complete(self) -> bool:
        return self.enumeration_complete and not self.blockers and all(raw.complete for raw in self.raws)


@dataclass(frozen=True, slots=True)
class IndexGenerationBinding:
    """Caller-observed active generation plus source snapshots stored in index."""

    active_generation: str
    source_snapshots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceGenerationReceipt:
    """Immutable, bounded source-to-current-index evidence projection."""

    source_generation_id: str
    active_generation: str
    generation_present: bool
    enumeration_complete: bool
    complete: bool
    blockers: tuple[SourceGenerationBlocker, ...]
    retired_raw_ids: tuple[str, ...]
    retired_coordinates: tuple[SourceGenerationRetiredCoordinate, ...]
    confirmed_raw_ids: tuple[str, ...]
    unresolved_raw_ids: tuple[str, ...]
    source_marker_missing_raw_ids: tuple[str, ...]
    index_generation_binding: IndexGenerationBinding
    items: tuple[SourceGenerationItemReceipt, ...]


def source_generation_receipt(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    active_generation: str,
) -> SourceGenerationReceipt:
    """Project source-43 membership through supplied source and index readers.

    Both connections must already be pinned by the caller.  Source tables and
    index tables are addressed as ``main`` intentionally: this is not a
    generic attachment/proxy layer and performs no connection setup.
    """
    if not source_generation_id:
        raise ValueError("source_generation_id must be non-empty")
    if not active_generation:
        raise ValueError("active_generation must be non-empty")

    binding = IndexGenerationBinding(active_generation=active_generation, source_snapshots=())
    generation = source_conn.execute(
        "SELECT item_count FROM main.source_generations WHERE source_generation_id = ?",
        (source_generation_id,),
    ).fetchone()
    if generation is None:
        return SourceGenerationReceipt(
            source_generation_id=source_generation_id,
            active_generation=active_generation,
            generation_present=False,
            enumeration_complete=False,
            complete=False,
            blockers=(SourceGenerationBlocker.GENERATION_MISSING,),
            retired_raw_ids=(),
            retired_coordinates=(),
            confirmed_raw_ids=(),
            unresolved_raw_ids=(),
            source_marker_missing_raw_ids=(),
            index_generation_binding=binding,
            items=(),
        )
    if int(generation[0]) > _MAX_SOURCE_ITEMS:
        raise ValueError(f"source generation receipt exceeds {_MAX_SOURCE_ITEMS} item limit")

    item_rows = source_conn.execute(
        """
        SELECT source_item_id, logical_coordinate, enumeration_fingerprint,
               enumerated_record_count, enumeration_digest, enumerated_at_ms
        FROM main.source_items
        WHERE source_generation_id = ?
        ORDER BY logical_coordinate, source_item_id
        """,
        (source_generation_id,),
    ).fetchall()
    if len(item_rows) > _MAX_SOURCE_ITEMS:
        raise ValueError(f"source generation receipt exceeds {_MAX_SOURCE_ITEMS} item limit")

    retired_coordinates: list[SourceGenerationRetiredCoordinate] = []
    items: list[SourceGenerationItemReceipt] = []
    for item in item_rows:
        item_id = str(item[0])
        coordinate = str(item[1])
        member_rows = source_conn.execute(
            """
            SELECT record_coordinate, raw_id, raw_blob_hash
            FROM main.source_item_raw_members
            WHERE source_generation_id = ? AND source_item_id = ?
            ORDER BY record_coordinate
            """,
            (source_generation_id, item_id),
        ).fetchall()
        retired = [row for row in member_rows if row[1] is None]
        retired_coordinates.extend(
            SourceGenerationRetiredCoordinate(item_id, coordinate, str(row[0])) for row in retired
        )
        enumeration_complete = _enumeration_complete(item, member_rows)
        item_blockers: list[SourceGenerationBlocker] = []
        if not enumeration_complete:
            item_blockers.append(SourceGenerationBlocker.ENUMERATION_INCOMPLETE)
        if retired:
            item_blockers.append(SourceGenerationBlocker.RETIRED_MEMBER)

        # Source-43 record members are the receipt denominator.  In
        # particular, do not turn a legacy item-level ``source_items.raw_id``
        # into an enumerated member: it has no record coordinate or exhausted
        # decoder witness.
        raw_ids = [str(row[1]) for row in member_rows if row[1] is not None]
        raws = tuple(_raw_receipt(source_conn, index_conn, raw_id) for raw_id in sorted(set(raw_ids)))
        items.append(
            SourceGenerationItemReceipt(
                source_item_id=item_id,
                logical_coordinate=coordinate,
                enumeration_complete=enumeration_complete,
                blockers=tuple(item_blockers),
                raws=raws,
            )
        )

    confirmed = sorted({raw.raw_id for item in items for raw in item.raws if raw.complete})
    unresolved = sorted({raw.raw_id for item in items for raw in item.raws if not raw.complete})
    raw_ids = sorted(set(confirmed + unresolved))
    source_snapshots: set[str] = set()
    for offset in range(0, len(raw_ids), 256):
        page = raw_ids[offset : offset + 256]
        placeholders = ",".join("?" for _ in page)
        source_snapshots.update(
            str(row[0])
            for row in index_conn.execute(
                f"SELECT source_snapshot FROM main.candidate_source_membership WHERE raw_id IN ({placeholders})",
                page,
            )
        )
    binding = IndexGenerationBinding(active_generation, tuple(sorted(source_snapshots)))
    marker_missing = sorted(
        {
            raw.raw_id
            for item in items
            for raw in item.raws
            if raw.complete and raw.logicals and raw.parsed_at_ms is None
        }
    )
    enumeration_complete = len(item_rows) == int(generation[0]) and all(item.enumeration_complete for item in items)
    complete = (
        len(item_rows) == int(generation[0])
        and enumeration_complete
        and not retired_coordinates
        and not unresolved
        and all(item.complete for item in items)
    )
    return SourceGenerationReceipt(
        source_generation_id=source_generation_id,
        active_generation=active_generation,
        generation_present=True,
        enumeration_complete=enumeration_complete,
        complete=complete,
        blockers=() if enumeration_complete else (SourceGenerationBlocker.ENUMERATION_INCOMPLETE,),
        # A retired member has a NULL FK.  Preserve the empty ID set rather
        # than inventing an ID from an old blob or item-level raw reference.
        retired_raw_ids=(),
        retired_coordinates=tuple(retired_coordinates),
        confirmed_raw_ids=tuple(confirmed),
        unresolved_raw_ids=tuple(unresolved),
        source_marker_missing_raw_ids=tuple(marker_missing),
        index_generation_binding=binding,
        items=tuple(items),
    )


def _enumeration_complete(item: tuple[object, ...], members: list[tuple[object, ...]]) -> bool:
    fingerprint, record_count, digest, enumerated_at_ms = item[2:]
    if fingerprint is None or record_count is None or digest is None or enumerated_at_ms is None:
        return False
    if int(record_count) != len(members):
        return False
    payload = [(str(row[0]), bytes(row[2]).hex()) for row in members]
    actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return str(digest) == actual


def _raw_receipt(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    raw_id: str,
) -> SourceGenerationRawReceipt:
    raw = source_conn.execute(
        """
        SELECT raw_id, source_index, parsed_at_ms, logical_source_key, revision_kind,
               source_revision, acquisition_generation
        FROM main.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if raw is None:
        return SourceGenerationRawReceipt(
            raw_id=raw_id,
            parsed_at_ms=None,
            parser_complete=False,
            parser_blockers=(SourceGenerationBlocker.PARSER_CENSUS_MISSING,),
            logicals=(),
        )
    memberships = source_conn.execute(
        """
        SELECT logical_source_key, provider_session_id, source_revision,
               normalized_content_hash, acquisition_generation, decision
        FROM main.raw_session_memberships
        WHERE raw_id = ?
        ORDER BY logical_source_key
        """,
        (raw_id,),
    ).fetchall()
    parser_complete, parser_blockers, logical_keys = _parser_census_state(source_conn, raw, memberships)
    memberships_by_key: dict[str, tuple[object, ...]] = {}
    membership_identity_matches = True
    for membership in memberships:
        try:
            key = canonical_authority_logical_key(str(membership[0]))
        except ValueError:
            membership_identity_matches = False
            continue
        if key.rpartition(":")[2] != str(membership[1]):
            membership_identity_matches = False
        memberships_by_key.setdefault(key, membership)
    if not membership_identity_matches:
        parser_complete = False
        parser_blockers = (SourceGenerationBlocker.PARSER_CENSUS_MISMATCH,)
    logicals = tuple(
        _logical_receipt(
            source_conn,
            index_conn,
            raw_id=raw_id,
            raw_source_revision=None if raw[5] is None else str(raw[5]),
            raw_acquisition_generation=None if raw[6] is None else int(raw[6]),
            logical_key=logical_key,
            membership=memberships_by_key.get(logical_key),
            parser_complete=parser_complete,
        )
        for logical_key in logical_keys
    )
    return SourceGenerationRawReceipt(
        raw_id=raw_id,
        parsed_at_ms=None if raw[2] is None else int(raw[2]),
        parser_complete=parser_complete,
        parser_blockers=parser_blockers,
        logicals=logicals,
    )


def _parser_census_state(
    source_conn: sqlite3.Connection,
    raw: tuple[object, ...],
    memberships: list[tuple[object, ...]],
) -> tuple[bool, tuple[SourceGenerationBlocker, ...], tuple[str, ...]]:
    """Apply archive-readiness' parser classifier plus exact membership census."""
    raw_id = str(raw[0])
    durable_keys = durable_authority_logical_keys(
        raw_logical_key=raw[3],
        revision_kind=raw[4],
        membership_logical_keys=(membership[0] for membership in memberships),
    )
    receipt = source_conn.execute(
        """
        SELECT parser_fingerprint, status, logical_keys_json
        FROM main.raw_authority_parser_census WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    if receipt is None:
        return False, (SourceGenerationBlocker.PARSER_CENSUS_MISSING,), durable_keys or ()
    typed_non_session = source_conn.execute(
        "SELECT EXISTS(SELECT 1 FROM main.raw_artifacts WHERE raw_id = ? AND parse_as_session = 0)",
        (raw_id,),
    ).fetchone()[0]
    membership_census = source_conn.execute(
        """
        SELECT parser_fingerprint, status, member_count, detail
        FROM main.raw_membership_census WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    parser_confirmed_non_session = (
        membership_census is not None
        and str(membership_census[0]) == RAW_AUTHORITY_PARSER_FINGERPRINT
        and str(membership_census[1]) == "non_session"
        and int(membership_census[2]) == 0
    )
    byte_governed_fragment = (
        int(raw[1]) < 0
        and membership_census is not None
        and str(membership_census[0]) == RAW_AUTHORITY_PARSER_FINGERPRINT
        and str(membership_census[1]) == "failed"
        and int(membership_census[2]) == 0
        and str(membership_census[3]) == BYTE_AUTHORITY_CENSUS_DETAIL
    )
    expected_membership_census = (
        parser_confirmed_non_session
        or byte_governed_fragment
        or (
            membership_census is not None
            and str(membership_census[0]) == RAW_AUTHORITY_PARSER_FINGERPRINT
            and str(membership_census[1]) == "complete"
            and int(membership_census[2]) == len(memberships)
        )
    )
    complete = (
        str(receipt[0]) == RAW_AUTHORITY_PARSER_FINGERPRINT
        and str(receipt[1]) == "complete"
        and expected_membership_census
        and parser_census_is_complete(
            recorded_keys=parser_census_logical_keys(receipt[2]),
            durable_keys=durable_keys,
            typed_non_session=bool(typed_non_session),
            parser_confirmed_non_session=parser_confirmed_non_session,
            byte_governed_fragment=byte_governed_fragment,
        )
    )
    return (
        (True, (), durable_keys or ())
        if complete
        else (False, (SourceGenerationBlocker.PARSER_CENSUS_MISMATCH,), durable_keys or ())
    )


def _logical_receipt(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    raw_id: str,
    raw_source_revision: str | None,
    raw_acquisition_generation: int | None,
    logical_key: str,
    membership: tuple[object, ...] | None,
    parser_complete: bool,
) -> SourceGenerationLogicalReceipt:
    origin, separator, native_id = logical_key.partition(":")
    if not separator or not native_id:
        return SourceGenerationLogicalReceipt(
            logical_source_key=logical_key,
            expected_session_id="",
            accepted_raw_id=None,
            application_ids=(),
            head_session_ids=(),
            session_ids=(),
            blockers=(SourceGenerationBlocker.PARSER_CENSUS_MISMATCH,),
        )
    expected_session_id = archive_session_id(origin, native_id)
    source_revision = raw_source_revision if membership is None else str(membership[2])
    acquisition_generation = raw_acquisition_generation if membership is None else int(membership[4])
    membership_application_decision = _membership_application_decision(
        None if membership is None or membership[5] is None else str(membership[5])
    )
    membership_content_hash = None if membership is None else bytes(membership[3])
    head = index_conn.execute(
        """
        SELECT session_id, accepted_raw_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, acquisition_generation
        FROM main.raw_revision_heads WHERE logical_source_key = ?
        """,
        (logical_key,),
    ).fetchall()
    head_session_ids = tuple(str(row[0]) for row in head if str(row[0]) == expected_session_id)
    blockers: list[SourceGenerationBlocker] = []
    accepted_raw_id: str | None = None
    current_head: tuple[object, ...] | None = None
    if not head:
        blockers.append(SourceGenerationBlocker.HEAD_ABSENT)
    elif len(head) != 1 or not head_session_ids:
        blockers.append(
            SourceGenerationBlocker.HEAD_STALE if len(head) == 1 else SourceGenerationBlocker.HEAD_AMBIGUOUS
        )
    else:
        current_head = tuple(head[0])
        accepted_raw_id = str(current_head[1])

    applications = index_conn.execute(
        """
        SELECT decision_id, source_revision, acquisition_generation, decision,
               accepted_raw_id, accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier
        FROM main.raw_revision_applications
        WHERE raw_id = ? AND logical_source_key = ? AND session_id = ?
        ORDER BY decision_id
        """,
        (raw_id, logical_key, expected_session_id),
    ).fetchall()
    application_ids = tuple(str(row[0]) for row in applications)
    valid_applications = (
        _current_or_prefix_applications(
            source_conn,
            raw_id=raw_id,
            source_revision=source_revision,
            acquisition_generation=acquisition_generation,
            membership_application_decision=membership_application_decision,
            membership_content_hash=membership_content_hash,
            head=current_head,
            applications=applications,
        )
        if current_head is not None
        else ()
    )
    if not application_ids:
        blockers.append(SourceGenerationBlocker.APPLICATION_ABSENT)
    elif not valid_applications:
        blockers.append(SourceGenerationBlocker.APPLICATION_STALE)
    elif len(valid_applications) != 1:
        blockers.append(SourceGenerationBlocker.APPLICATION_AMBIGUOUS)
    if not parser_complete:
        blockers.append(SourceGenerationBlocker.PARSER_CENSUS_MISMATCH)

    session_rows = ()
    if current_head is not None:
        session_rows = index_conn.execute(
            """
            SELECT session_id FROM main.sessions
            WHERE session_id = ? AND raw_id = ? AND content_hash = ?
            ORDER BY session_id
            """,
            (expected_session_id, current_head[1], current_head[3]),
        ).fetchall()
    session_ids = tuple(str(row[0]) for row in session_rows)
    any_session = index_conn.execute(
        "SELECT 1 FROM main.sessions WHERE session_id = ? LIMIT 1", (expected_session_id,)
    ).fetchone()
    if current_head is None or not session_ids:
        blockers.append(
            SourceGenerationBlocker.SESSION_STALE if any_session is not None else SourceGenerationBlocker.SESSION_ABSENT
        )
    elif len(session_ids) != 1:
        blockers.append(SourceGenerationBlocker.SESSION_AMBIGUOUS)
    return SourceGenerationLogicalReceipt(
        logical_source_key=logical_key,
        expected_session_id=expected_session_id,
        accepted_raw_id=accepted_raw_id,
        application_ids=application_ids,
        head_session_ids=head_session_ids,
        session_ids=session_ids,
        blockers=tuple(blockers),
    )


def _current_or_prefix_applications(
    source_conn: sqlite3.Connection,
    *,
    raw_id: str,
    source_revision: str | None,
    acquisition_generation: int | None,
    membership_application_decision: str | None,
    membership_content_hash: bytes | None,
    head: tuple[object, ...],
    applications: list[tuple[object, ...]],
) -> tuple[str, ...]:
    """Return exact application receipts that prove this raw's current effect.

    An immutable application can either name the current accepted head directly
    (the normal supersession/equivalent form), or record an accepted prefix in
    the source predecessor chain.  The latter remains a valid contribution to
    a later current head; demanding ``accepted_raw_id == raw_id`` would erase
    that durable provenance.
    """
    head_raw_id = str(head[1])
    head_identity = (*head[1:6], head[6])
    raw_is_prefix = _raw_is_predecessor(source_conn, raw_id=raw_id, accepted_raw_id=head_raw_id)
    valid: list[str] = []
    for application in applications:
        decision = str(application[3])
        application_identity = (*application[4:9], application[2])
        source_event_matches = (
            source_revision is not None
            and acquisition_generation is not None
            and str(application[1]) == source_revision
            and int(application[2]) == acquisition_generation
        )
        decision_matches = decision in {
            "selected_baseline",
            "applied_append",
            "reparse_reaffirmation",
            "superseded",
        } and (membership_application_decision is None or decision == membership_application_decision)
        names_current_head = decision_matches and source_event_matches and application_identity == head_identity
        accepted_self_prefix = (
            decision_matches
            and decision in {"selected_baseline", "applied_append", "reparse_reaffirmation"}
            and str(application[4]) == raw_id
            and str(application[5]) == source_revision
            and source_event_matches
            and raw_is_prefix
            and (
                bytes(application[6]) == membership_content_hash
                if membership_content_hash is not None
                else _byte_prefix_metadata_is_exact(
                    source_conn,
                    raw_id=raw_id,
                    accepted_raw_id=head_raw_id,
                    source_revision=source_revision,
                    acquisition_generation=acquisition_generation,
                    application=application,
                )
            )
        )
        if names_current_head or accepted_self_prefix:
            valid.append(str(application[0]))
    return tuple(valid)


def _raw_is_predecessor(source_conn: sqlite3.Connection, *, raw_id: str, accepted_raw_id: str) -> bool:
    """Read the exact source predecessor chain without reopening or recursion proxying."""
    row = source_conn.execute(
        """
        WITH RECURSIVE chain(raw_id, predecessor_raw_id, path) AS (
            SELECT raw_id, predecessor_raw_id, ',' || raw_id || ','
            FROM main.raw_sessions WHERE raw_id = ?
            UNION ALL
            SELECT candidate.raw_id, candidate.predecessor_raw_id,
                   chain.path || candidate.raw_id || ','
            FROM main.raw_sessions AS candidate
            JOIN chain ON candidate.raw_id = chain.predecessor_raw_id
            WHERE instr(chain.path, ',' || candidate.raw_id || ',') = 0
        )
        SELECT EXISTS(SELECT 1 FROM chain WHERE raw_id = ? AND raw_id != ?)
        """,
        (accepted_raw_id, raw_id, accepted_raw_id),
    ).fetchone()
    return bool(row[0])


def _membership_application_decision(membership_decision: str | None) -> str | None:
    """Use the sole source-membership to index-application translation."""
    if membership_decision is None:
        return None
    try:
        return _application_decision_for(MembershipDecision(membership_decision)).value
    except ValueError:
        return None


def _byte_prefix_metadata_is_exact(
    source_conn: sqlite3.Connection,
    *,
    raw_id: str,
    accepted_raw_id: str,
    source_revision: str | None,
    acquisition_generation: int | None,
    application: tuple[object, ...],
) -> bool:
    """Require a byte append's own frontier receipt and exact source chain."""
    if source_revision is None or acquisition_generation is None:
        return False
    candidate = source_conn.execute(
        """
        SELECT source_index, predecessor_raw_id, baseline_raw_id, append_end_offset,
               acquisition_generation
        FROM main.raw_sessions WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    current = source_conn.execute(
        """
        SELECT baseline_raw_id, acquisition_generation
        FROM main.raw_sessions WHERE raw_id = ?
        """,
        (accepted_raw_id,),
    ).fetchone()
    return (
        candidate is not None
        and current is not None
        and int(candidate[0]) < 0
        and candidate[1] is not None
        and candidate[2] is not None
        and candidate[3] is not None
        and int(candidate[4]) == acquisition_generation
        and current[0] == candidate[2]
        and int(current[1]) >= acquisition_generation
        and str(application[4]) == raw_id
        and str(application[5]) == source_revision
        and application[6] is not None
        and str(application[7]) == "byte"
        and int(application[8]) == int(candidate[3])
    )


__all__ = [
    "IndexGenerationBinding",
    "SourceGenerationBlocker",
    "SourceGenerationItemReceipt",
    "SourceGenerationLogicalReceipt",
    "SourceGenerationRawReceipt",
    "SourceGenerationReceipt",
    "SourceGenerationRetiredCoordinate",
    "source_generation_receipt",
]
