"""Pinned, read-only evidence for one source-generation publication.

The caller owns the publication barrier and the two SQLite snapshots.  This
module never reopens those archive snapshots or turns evidence into a
derivation claim. Private disk scratch measures exact membership and deduplicates
raws while the supplied snapshots remain the authority for every witness.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    canonical_authority_logical_key,
    durable_authority_logical_keys,
    parser_census_is_complete,
)
from polylogue.archive.session_revision_membership import MembershipDecision
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.storage.raw_authority import (
    parser_census_logical_keys,
    raw_authority_parser_fingerprint,
)
from polylogue.storage.sqlite.archive_tiers.revision_governance import _application_decision_for
from polylogue.storage.sqlite.archive_tiers.source_items import _measure_source_item_enumeration
from polylogue.storage.sqlite.connection_profile import scratch_connection_context

_SOURCE_PAGE_ITEMS = 256


class SourceGenerationBlocker(StrEnum):
    """A closed explanation for one unproven source-generation member."""

    GENERATION_MISSING = "generation_missing"
    ENUMERATION_INCOMPLETE = "enumeration_incomplete"
    MEMBER_REFUSED = "member_refused"
    MEMBER_UNSELECTED = "member_unselected"
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
    """One input's measured structural proof; raw witnesses are streamed separately."""

    source_item_id: str
    logical_coordinate: str
    enumeration_complete: bool
    blockers: tuple[SourceGenerationBlocker, ...]
    retired_count: int

    @property
    def source_complete(self) -> bool:
        return self.enumeration_complete and not self.blockers


@dataclass(frozen=True, slots=True)
class SourceGenerationReceiptPage:
    """At most 256 exact input headers from one pinned Source snapshot."""

    items: tuple[SourceGenerationItemReceipt, ...]
    next_cursor: tuple[str, str] | None


def source_generation_receipt_page(
    source_conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    after: tuple[str, str] | None = None,
    limit: int = _SOURCE_PAGE_ITEMS,
    check_stop: Callable[[], None] | None = None,
) -> SourceGenerationReceiptPage:
    """Project a keyset page of structural headers, never whole nested members."""
    if not 1 <= limit <= _SOURCE_PAGE_ITEMS:
        raise ValueError("source generation receipt page must be between 1 and 256 inputs")
    with closing(
        source_conn.execute(
            "SELECT source_item_id, logical_coordinate, enumeration_fingerprint, enumerated_record_count, "
            "enumeration_digest, enumerated_at_ms, enumerated_member_count, enumeration_member_digest "
            "FROM main.source_items WHERE source_generation_id=? AND (logical_coordinate, source_item_id) > (?, ?) "
            "ORDER BY logical_coordinate, source_item_id LIMIT ?",
            (source_generation_id, *(after or ("", "")), limit),
        )
    ) as cursor:
        rows = cursor.fetchall()
    items = []
    for row in rows:
        if check_stop is not None:
            check_stop()
        item_id = str(row[0])
        with closing(
            source_conn.execute(
                "SELECT COUNT(*) FROM main.source_item_raw_members "
                "WHERE source_generation_id=? AND source_item_id=? AND raw_id IS NULL",
                (source_generation_id, item_id),
            )
        ) as cursor:
            retired_count = int(cursor.fetchone()[0])
        with closing(
            source_conn.execute(
                "SELECT COALESCE(SUM(disposition='refused'),0), COALESCE(SUM(disposition='unselected'),0) "
                "FROM main.source_item_member_dispositions WHERE source_generation_id=? AND source_item_id=?",
                (source_generation_id, item_id),
            )
        ) as cursor:
            refused, unselected = cursor.fetchone()
        enumeration_complete = _enumeration_complete(source_conn, source_generation_id, row, check_stop=check_stop)
        blockers = []
        if not enumeration_complete:
            blockers.append(SourceGenerationBlocker.ENUMERATION_INCOMPLETE)
        if retired_count:
            blockers.append(SourceGenerationBlocker.RETIRED_MEMBER)
        if refused:
            blockers.append(SourceGenerationBlocker.MEMBER_REFUSED)
        if unselected:
            blockers.append(SourceGenerationBlocker.MEMBER_UNSELECTED)
        items.append(
            SourceGenerationItemReceipt(item_id, str(row[1]), enumeration_complete, tuple(blockers), retired_count)
        )
    return SourceGenerationReceiptPage(tuple(items), (str(rows[-1][1]), str(rows[-1][0])) if rows else None)


def iter_source_item_raw_receipts(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    check_stop: Callable[[], None] | None = None,
) -> Iterator[SourceGenerationRawReceipt]:
    """Stream unique raw witnesses through an owned disk key set on the same snapshots."""
    with scratch_connection_context(prefix="polylogue-source-raws-", filename="keys.db") as keys:
        keys.execute("PRAGMA journal_mode=DELETE")
        keys.execute("PRAGMA temp_store=FILE")
        keys.execute("BEGIN")
        keys.execute("CREATE TABLE raws(raw_id TEXT PRIMARY KEY) WITHOUT ROWID")
        with closing(
            source_conn.execute(
                "SELECT raw_id FROM main.source_item_raw_members WHERE source_generation_id=? AND source_item_id=?",
                (source_generation_id, source_item_id),
            )
        ) as cursor:
            for row in cursor:
                if check_stop is not None:
                    check_stop()
                if row[0] is not None:
                    keys.execute("INSERT INTO raws VALUES (?) ON CONFLICT DO NOTHING", (row[0],))
        with closing(keys.execute("SELECT raw_id FROM raws ORDER BY raw_id")) as cursor:
            for row in cursor:
                if check_stop is not None:
                    check_stop()
                yield _raw_receipt(source_conn, index_conn, str(row[0]))


def _enumeration_complete(
    source_conn: sqlite3.Connection,
    source_generation_id: str,
    item: tuple[object, ...],
    *,
    check_stop: Callable[[], None] | None = None,
) -> bool:
    fingerprint, record_count, digest, enumerated_at_ms, member_count, member_digest = item[2:]
    expected_count = _int_cell(record_count)
    expected_members = _int_cell(member_count)
    if (
        fingerprint is None
        or expected_count is None
        or _int_cell(enumerated_at_ms) is None
        or expected_members is None
        or digest is None
        or member_digest is None
    ):
        return False
    item_id = str(item[0])
    binding = (source_generation_id, item_id)
    with closing(
        source_conn.execute(
            "SELECT 1 FROM main.source_item_raw_members WHERE source_generation_id=? AND source_item_id=? "
            "AND record_coordinate LIKE '[\"zip-v2\"%' LIMIT 1",
            binding,
        )
    ) as cursor:
        container = cursor.fetchone() is not None
    with closing(
        source_conn.execute(
            "SELECT 1 FROM main.source_item_member_dispositions WHERE source_generation_id=? AND source_item_id=? LIMIT 1",
            binding,
        )
    ) as cursor:
        container |= cursor.fetchone() is not None
    callback_failure: BaseException | None = None

    def checkpoint() -> None:
        nonlocal callback_failure
        if check_stop is not None:
            try:
                check_stop()
            except BaseException as exc:
                callback_failure = exc
                raise

    try:
        with closing(
            source_conn.execute(
                "SELECT record_coordinate FROM main.source_item_raw_members "
                "WHERE source_generation_id=? AND source_item_id=? ORDER BY record_coordinate",
                binding,
            )
        ) as cursor:
            measured = _measure_source_item_enumeration(
                source_conn,
                source_generation_id=source_generation_id,
                source_item_id=item_id,
                record_coordinates=(str(row[0]) for row in cursor),
                member_ordinals=range(expected_members) if container else None,
                member_count=expected_members if container else None,
                check_stop=checkpoint,
            )
    except ValueError:
        if callback_failure is not None:
            raise callback_failure from None
        return False
    return measured[:4] == (expected_count, str(digest), expected_members, str(member_digest))


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
        if key.partition(":")[2] != str(membership[1]):
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
            raw_acquisition_generation=_int_cell(raw[6]),
            logical_key=logical_key,
            membership=memberships_by_key.get(logical_key),
            parser_complete=parser_complete,
        )
        for logical_key in logical_keys
    )
    return SourceGenerationRawReceipt(
        raw_id=raw_id,
        parsed_at_ms=_int_cell(raw[2]),
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
        SELECT parser_fingerprint, status, member_count, revision_authority
        FROM main.raw_membership_census WHERE raw_id = ?
        """,
        (raw_id,),
    ).fetchone()
    membership_count = _int_cell(membership_census[2]) if membership_census is not None else None
    source_index = _int_cell(raw[1])
    parser_confirmed_non_session = (
        membership_census is not None
        and str(membership_census[0]) == raw_authority_parser_fingerprint()
        and str(membership_census[1]) == "non_session"
        and membership_count == 0
    )
    byte_governed_fragment = (
        source_index is not None
        and source_index < 0
        and membership_census is not None
        and str(membership_census[0]) == raw_authority_parser_fingerprint()
        and str(membership_census[1]) == "failed"
        and membership_count == 0
        and str(membership_census[3]) == RawRevisionAuthority.BYTE_PROVEN.value
    )
    expected_membership_census = (
        parser_confirmed_non_session
        or byte_governed_fragment
        or (
            membership_census is not None
            and str(membership_census[0]) == raw_authority_parser_fingerprint()
            and str(membership_census[1]) == "complete"
            and membership_count == len(memberships)
        )
    )
    complete = (
        str(receipt[0]) == raw_authority_parser_fingerprint()
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
    acquisition_generation = raw_acquisition_generation if membership is None else _int_cell(membership[4])
    membership_application_decision = _membership_application_decision(
        None if membership is None or membership[5] is None else str(membership[5])
    )
    membership_content_hash = None if membership is None else _bytes_cell(membership[3])
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
        if (
            current_head is not None
            and _bytes_cell(current_head[3]) is not None
            and _int_cell(current_head[5]) is not None
            and _int_cell(current_head[6]) is not None
            and (membership is None or membership_content_hash is not None)
        )
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

    session_rows: list[tuple[object, ...]] = []
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
    head_content_hash = _bytes_cell(head[3])
    head_frontier = _int_cell(head[5])
    head_generation = _int_cell(head[6])
    raw_is_prefix = _raw_is_predecessor(source_conn, raw_id=raw_id, accepted_raw_id=head_raw_id)
    valid: list[str] = []
    for application in applications:
        decision = str(application[3])
        application_identity = (*application[4:9], application[2])
        application_generation = _int_cell(application[2])
        application_content_hash = _bytes_cell(application[6])
        source_event_matches = (
            source_revision is not None
            and acquisition_generation is not None
            and application_generation is not None
            and str(application[1]) == source_revision
            and application_generation == acquisition_generation
        )
        decision_matches = decision in {
            "selected_baseline",
            "applied_append",
            "reparse_reaffirmation",
            "superseded",
        } and (membership_application_decision is None or decision == membership_application_decision)
        application_frontier = _int_cell(application[8])
        names_current_head = (
            decision_matches
            and source_event_matches
            and head_content_hash is not None
            and head_frontier is not None
            and head_generation is not None
            and application_content_hash is not None
            and application_frontier is not None
            and application_identity == head_identity
        )
        accepted_self_prefix = (
            decision_matches
            and decision in {"selected_baseline", "applied_append", "reparse_reaffirmation"}
            and str(application[4]) == raw_id
            and str(application[5]) == source_revision
            and source_event_matches
            and raw_is_prefix
            and application_content_hash is not None
            and (
                application_content_hash == membership_content_hash
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
    candidate_source_index = _int_cell(candidate[0]) if candidate is not None else None
    candidate_append_end = _int_cell(candidate[3]) if candidate is not None else None
    candidate_generation = _int_cell(candidate[4]) if candidate is not None else None
    current_generation = _int_cell(current[1]) if current is not None else None
    application_end = _int_cell(application[8])
    application_content_hash = _bytes_cell(application[6])
    return (
        candidate is not None
        and current is not None
        and candidate_source_index is not None
        and candidate_source_index < 0
        and candidate[1] is not None
        and candidate[2] is not None
        and candidate_append_end is not None
        and candidate_generation == acquisition_generation
        and current[0] == candidate[2]
        and current_generation is not None
        and current_generation >= acquisition_generation
        and str(application[4]) == raw_id
        and str(application[5]) == source_revision
        and application_content_hash is not None
        and str(application[7]) == "byte"
        and application_end == candidate_append_end
    )


def _int_cell(value: object) -> int | None:
    """Accept only SQLite integer cells, excluding Python's bool subtype."""

    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bytes_cell(value: object) -> bytes | None:
    """Accept only SQLite BLOB cells; do not coerce text or numeric values."""

    return value if isinstance(value, bytes) else None


__all__ = [
    "SourceGenerationBlocker",
    "SourceGenerationItemReceipt",
    "source_generation_receipt_page",
    "iter_source_item_raw_receipts",
    "SourceGenerationLogicalReceipt",
    "SourceGenerationRawReceipt",
]
