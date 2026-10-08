"""Raw replay plans and the durable frontier blockers they authorize.

A replay plan is a pure content-addressed snapshot of one component, rebuilt
from current source and index evidence on demand -- it is never stored. The
source tier is the authority for the one durable relation here,
``raw_authority_blockers``: ``index.db`` may be rebuilt and ``ops.db`` may be
deleted, and neither event is allowed to erase an unresolved frontier
obligation or the operator resolution that closed one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.revision_authority import (
    InvalidParserCensusKeysError,
    canonical_authority_logical_key,
    raw_authority_parser_fingerprint,
)
from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.archive.session_revision_membership import MembershipDecision
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.storage.sqlite.archive_tiers.source_write import PENDING_RAW_LOGICAL_SOURCE_PREFIX
from polylogue.storage.sqlite.connection_profile import (
    attach_readonly_database,
    open_readonly_connection,
)

logger = get_logger(__name__)


def _readonly(path: Path) -> sqlite3.Connection:
    """Open a bounded, query-only maintenance reader."""
    return open_readonly_connection(path, timeout_class="background-read")


def iter_parser_census_logical_keys(logical_keys_json: object) -> Generator[str, None, None]:
    """Validate the existing ordered receipt while yielding one canonical key.

    Canonical duplicate detection belongs to the shared disk measurement.
    The durable JSON cell itself remains one SQLite value; this reader does
    not allocate its decoded list, normalized list and duplicate-key set.
    """
    import io

    import ijson

    from polylogue.archive.raw_payload.streams import raw_byte_stream

    with io.StringIO(str(logical_keys_json)) as text, raw_byte_stream(text) as stream:
        events = iter(ijson.basic_parse(stream))
        last: str | None = None
        try:
            if next(events, None) != ("start_array", None):
                raise InvalidParserCensusKeysError("parser identity receipt is not an array")
            for event, value in events:
                if event == "end_array":
                    if next(events, None) is not None:
                        raise InvalidParserCensusKeysError("parser identity receipt has trailing values")
                    return
                if event != "string" or not isinstance(value, str) or last is not None and value <= last:
                    raise InvalidParserCensusKeysError("parser identity receipt keys are not sorted unique strings")
                last = value
                try:
                    key = canonical_authority_logical_key(value)
                except ValueError as error:
                    raise InvalidParserCensusKeysError(
                        "parser identity receipt has an invalid authority key"
                    ) from error
                yield key
        except (ijson.JSONError, UnicodeError) as error:
            raise InvalidParserCensusKeysError("parser identity receipt JSON cannot establish authority") from error
        raise InvalidParserCensusKeysError("parser identity receipt is incomplete")


@dataclass(frozen=True, slots=True)
class RawReplayPlan:
    plan_id: str
    input_digest: str
    input_raw_ids: tuple[str, ...]
    logical_keys: tuple[str, ...]
    authority_witness: JSONDocument
    source_preconditions: JSONDocument
    index_preconditions: JSONDocument

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "plan_id": self.plan_id,
                "input_digest": self.input_digest,
                "input_raw_ids": list(self.input_raw_ids),
                "logical_keys": list(self.logical_keys),
                "authority_witness": self.authority_witness,
                "source_preconditions": self.source_preconditions,
                "index_preconditions": self.index_preconditions,
            }
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.hex()
    return value


def _rows(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params))
    names = tuple(column[0] for column in cursor.description or ())
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor]


def build_raw_replay_plan(conn: sqlite3.Connection, input_raw_ids: Sequence[str]) -> RawReplayPlan:
    """Snapshot one complete component from an attached source/index pair."""
    raw_ids = tuple(sorted(dict.fromkeys(input_raw_ids)))
    if not raw_ids:
        raise ValueError("raw replay plan requires at least one input raw id")
    marks = ",".join("?" for _ in raw_ids)
    source_rows = _rows(
        conn,
        _RAW_PLAN_SOURCE_SQL.format(marks=marks, index_prefix="index_tier."),
        raw_ids,
    )
    if tuple(str(row["raw_id"]) for row in source_rows) != raw_ids:
        raise RuntimeError("raw replay plan input disappeared during census")
    membership_rows = _rows(
        conn,
        _RAW_PLAN_MEMBERSHIP_SQL.format(marks=marks, index_prefix="index_tier."),
        raw_ids,
    )
    census_rows = _rows(
        conn,
        _RAW_PLAN_CENSUS_SQL.format(marks=marks, index_prefix="index_tier."),
        raw_ids,
    )
    parser_census_rows = _rows(
        conn,
        _RAW_PLAN_PARSER_CENSUS_SQL.format(marks=marks, index_prefix="index_tier."),
        raw_ids,
    )
    logical_keys = _raw_replay_logical_keys(source_rows, membership_rows)
    if logical_keys:
        key_marks = ",".join("?" for _ in logical_keys)
        head_rows = _rows(
            conn,
            _RAW_PLAN_HEAD_SQL.format(marks=key_marks, index_prefix="index_tier."),
            logical_keys,
        )
    else:
        head_rows = []
    session_rows = _rows(
        conn,
        _RAW_PLAN_SESSION_SQL.format(marks=marks, index_prefix="index_tier."),
        raw_ids,
    )
    return _raw_replay_plan_from_rows(
        raw_ids,
        logical_keys,
        source_rows=source_rows,
        membership_rows=membership_rows,
        census_rows=census_rows,
        parser_census_rows=parser_census_rows,
        head_rows=head_rows,
        session_rows=session_rows,
    )


def build_raw_replay_plans(
    archive_root: Path,
    components: Sequence[tuple[str, ...]],
    *,
    index_db_path: Path | None = None,
) -> tuple[RawReplayPlan, ...]:
    if not components:
        return ()
    if index_db_path is None:
        from polylogue.storage.archive_identity import resolve_active_index_path

        index_db_path = resolve_active_index_path(archive_root)
    with closing(_readonly(archive_root / "source.db")) as conn:
        attach_readonly_database(conn, index_db_path, alias="index_tier")
        return tuple(build_raw_replay_plan(conn, component) for component in components)


def unresolved_raw_authority_blockers(archive_root: Path) -> int:
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return 0
    with closing(_readonly(source_db)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_blockers'"
        ).fetchone()
        if exists is None:
            return 0
        return int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0]
        )


def validate_raw_replay_plan(
    archive_root: Path,
    plan: RawReplayPlan,
    *,
    index_db_path: Path | None = None,
) -> tuple[bool, JSONDocument]:
    try:
        observed = build_raw_replay_plans(
            archive_root,
            (plan.input_raw_ids,),
            index_db_path=index_db_path,
        )[0]
    except Exception as exc:
        logger.warning("raw replay plan validation could not rebuild %s", plan.plan_id, exc_info=True)
        return False, json_document({"error": f"{type(exc).__name__}: {exc}"})
    return observed == plan, observed.to_dict()


def raw_replay_application_receipt(
    archive_root: Path,
    plan: RawReplayPlan,
    *,
    index_db_path: Path | None = None,
) -> JSONDocument:
    if index_db_path is None:
        from polylogue.storage.archive_identity import resolve_active_index_path

        index_db_path = resolve_active_index_path(archive_root)
    with closing(_readonly(archive_root / "source.db")) as conn:
        attach_readonly_database(conn, index_db_path, alias="index_tier")
        return raw_replay_application_receipt_from_connection(conn, plan, index_db_path=index_db_path)


def raw_replay_application_receipt_from_connection(
    conn: sqlite3.Connection,
    plan: RawReplayPlan,
    *,
    index_db_path: Path,
) -> JSONDocument:
    """Read receipt authority from the caller's pinned source/index snapshot."""
    marks = ",".join("?" for _ in plan.input_raw_ids)
    source = _rows(
        conn,
        f"""
        SELECT raw_id, source_revision, predecessor_raw_id, baseline_raw_id,
               append_end_offset, parsed_at_ms, parse_error
        FROM raw_sessions WHERE raw_id IN ({marks}) ORDER BY raw_id
        """,
        plan.input_raw_ids,
    )
    memberships = _rows(
        conn,
        f"""
        SELECT raw_id, logical_source_key, source_revision, decision, decided_at_ms
        FROM raw_session_memberships
        WHERE raw_id IN ({marks}) ORDER BY raw_id, logical_source_key
        """,
        plan.input_raw_ids,
    )
    applications = _rows(
        conn,
        f"""
        SELECT decision_id, raw_id, session_id, logical_source_key, decision,
               source_revision, acquisition_generation, accepted_raw_id,
               accepted_source_revision, hex(accepted_content_hash) AS accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, baseline_raw_id,
               predecessor_raw_id, append_end_offset, decided_at_ms
        FROM index_tier.raw_revision_applications
        WHERE raw_id IN ({marks}) ORDER BY raw_id, decision_id
        """,
        plan.input_raw_ids,
    )
    if plan.logical_keys:
        key_marks = ",".join("?" for _ in plan.logical_keys)
        heads = _rows(
            conn,
            f"""
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision,
                   hex(accepted_content_hash) AS accepted_content_hash,
                   accepted_frontier_kind, accepted_frontier,
                   acquisition_generation, append_end_offset
            FROM index_tier.raw_revision_heads
            WHERE logical_source_key IN ({key_marks})
            ORDER BY logical_source_key
            """,
            plan.logical_keys,
        )
        sessions = _rows(
            conn,
            f"""
            SELECT s.session_id, s.raw_id, hex(s.content_hash) AS content_hash,
                   s.message_count
            FROM index_tier.sessions AS s
            JOIN index_tier.raw_revision_heads AS h ON h.session_id = s.session_id
            WHERE h.logical_source_key IN ({key_marks})
            ORDER BY s.session_id
            """,
            plan.logical_keys,
        )
    else:
        heads = []
        sessions = []
    return json_document(
        {
            "schema": "polylogue.raw-replay-application-receipt.v2",
            "index_db_path": str(index_db_path),
            "source_rows": source,
            "membership_rows": memberships,
            "application_rows": applications,
            "head_rows": heads,
            "session_rows": sessions,
        }
    )


def validate_raw_replay_application_receipt(
    plan: RawReplayPlan,
    receipt: Mapping[str, object],
) -> tuple[bool, tuple[str, ...]]:
    """Prove exact replay postconditions; parsed timestamps are never sufficient."""
    problems: list[str] = []
    if receipt.get("schema") != "polylogue.raw-replay-application-receipt.v2":
        problems.append("application receipt schema is not v2")

    def rows(name: str) -> list[Mapping[str, object]]:
        value = receipt.get(name)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            problems.append(f"{name} is not a row list")
            return []
        return value

    source_rows = rows("source_rows")
    membership_rows = rows("membership_rows")
    application_rows = rows("application_rows")
    head_rows = rows("head_rows")
    session_rows = rows("session_rows")
    source_ids = {str(row.get("raw_id")) for row in source_rows}
    if source_ids != set(plan.input_raw_ids):
        problems.append("source receipt raw ids do not match the immutable plan")
    if any(row.get("parsed_at_ms") is None or row.get("parse_error") is not None for row in source_rows):
        problems.append("source receipt contains an unparsed or parse-failed raw")
    expected_keys = set(plan.logical_keys)
    head_keys = {str(row.get("logical_source_key")) for row in head_rows}
    if not expected_keys:
        problems.append("executed replay plan has no logical authority keys")
    elif head_keys != expected_keys:
        problems.append("accepted head keys do not match the immutable plan")
    input_raw_ids = set(plan.input_raw_ids)
    source_by_raw_id = {str(row.get("raw_id")): row for row in source_rows}
    if len(source_by_raw_id) != len(source_rows):
        problems.append("source receipt contains duplicate raw ids")
    membership_revisions_by_raw_and_key: dict[tuple[str, str], set[str]] = {}
    for membership in membership_rows:
        source_revision = membership.get("source_revision")
        if source_revision is not None:
            membership_key = (str(membership.get("raw_id")), str(membership.get("logical_source_key")))
            membership_revisions_by_raw_and_key.setdefault(membership_key, set()).add(str(source_revision))
    witness = plan.authority_witness.get("memberships")
    expected_memberships = (
        {(str(row.get("raw_id")), str(row.get("logical_source_key"))) for row in witness if isinstance(row, dict)}
        if isinstance(witness, list)
        else set()
    )
    observed_memberships = {(str(row.get("raw_id")), str(row.get("logical_source_key"))) for row in membership_rows}
    if observed_memberships != expected_memberships:
        problems.append("membership receipt pairs do not match the immutable authority witness")
    terminal_membership_decisions = {
        MembershipDecision.APPLIED,
        MembershipDecision.SUPERSEDED_EQUIVALENT,
        MembershipDecision.SUPERSEDED_PREFIX,
        MembershipDecision.SUPERSEDED_BY_WINNER,
    }
    if any(row.get("decision") not in terminal_membership_decisions for row in membership_rows):
        problems.append("membership receipt contains a non-terminal decision")
    terminal_application_decisions = {
        ApplicationDecision.SELECTED_BASELINE,
        ApplicationDecision.APPLIED_APPEND,
        ApplicationDecision.SUPERSEDED,
        ApplicationDecision.REPARSE_REAFFIRMATION,
    }
    application_pairs = {(str(row.get("raw_id")), str(row.get("logical_source_key"))) for row in application_rows}
    if any(row.get("decision") not in terminal_application_decisions for row in application_rows):
        problems.append("application receipt contains a non-terminal decision")
    if any(raw_id not in input_raw_ids or key not in expected_keys for raw_id, key in application_pairs):
        problems.append("application receipt contains authority outside the immutable component")
    terminal_keys = {key for _, key in observed_memberships | application_pairs}
    if terminal_keys != expected_keys:
        problems.append("terminal receipt keys do not exactly match the immutable plan")
    head_session_ids = {str(row.get("session_id")) for row in head_rows}
    session_ids = {str(row.get("session_id")) for row in session_rows}
    if head_session_ids != session_ids:
        problems.append("accepted head sessions do not match materialized session rows")
    session_content = {str(row.get("session_id")): str(row.get("content_hash")) for row in session_rows}
    if any(
        str(row.get("accepted_content_hash")) != session_content.get(str(row.get("session_id"))) for row in head_rows
    ):
        problems.append("accepted head content hashes do not match materialized sessions")
    membership_decisions = {
        (str(row.get("raw_id")), str(row.get("logical_source_key"))): row.get("decision") for row in membership_rows
    }

    def membership_yield(application: Mapping[str, object]) -> bool:
        # A membership cohort that yields to a chain-governed head outside it
        # records SUPERSEDED with no accepted authority of its own; the head
        # it yielded to is certified by that head's own component.
        return (
            application.get("decision") == ApplicationDecision.SUPERSEDED.value
            and all(
                application.get(field) in (None, "")
                for field in (
                    "accepted_raw_id",
                    "accepted_source_revision",
                    "accepted_content_hash",
                    "accepted_frontier_kind",
                    "accepted_frontier",
                )
            )
            and membership_decisions.get((str(application.get("raw_id")), str(application.get("logical_source_key"))))
            == MembershipDecision.SUPERSEDED_EQUIVALENT
        )

    applications_by_key: dict[str, list[Mapping[str, object]]] = {}
    for application in application_rows:
        applications_by_key.setdefault(str(application.get("logical_source_key")), []).append(application)
    yielded_keys = {
        key for key, applications in applications_by_key.items() if all(membership_yield(row) for row in applications)
    }
    if any(
        str(row.get("accepted_raw_id")) not in input_raw_ids
        for row in head_rows
        if str(row.get("logical_source_key")) not in yielded_keys
    ):
        problems.append("accepted heads do not point into the immutable input component")
    heads_by_key = {str(row.get("logical_source_key")): row for row in head_rows}
    if len(heads_by_key) != len(head_rows):
        problems.append("accepted head receipt contains duplicate logical authority keys")
    sessions_by_id: dict[str, Mapping[str, object]] = {}
    for session in session_rows:
        session_id = str(session.get("session_id"))
        existing_session = sessions_by_id.setdefault(session_id, session)
        if existing_session != session:
            problems.append(f"materialized session receipt conflicts for {session_id}")
    applications_matching_current_head: set[str] = set()
    for application in application_rows:
        key = str(application.get("logical_source_key"))
        raw_id = str(application.get("raw_id"))
        source = source_by_raw_id.get(raw_id)
        if source is None:
            continue
        source_revisions = set(membership_revisions_by_raw_and_key.get((raw_id, key), set()))
        if not source_revisions:
            source_revision = source.get("source_revision")
            if source_revision is not None:
                source_revisions.add(str(source_revision))
        if not source_revisions:
            problems.append(f"application has no source revision evidence for {raw_id}/{key}")
        elif str(application.get("source_revision")) not in source_revisions:
            problems.append(f"application source revision does not match membership evidence for {raw_id}/{key}")
        yielded = membership_yield(application)
        if yielded:
            yield_head = heads_by_key.get(key)
            if yield_head is None or yield_head.get("session_id") != application.get("session_id"):
                problems.append(f"membership yield does not name the current head for {raw_id}/{key}")
        accepted_raw_id = str(application.get("accepted_raw_id"))
        accepted_source = source_by_raw_id.get(accepted_raw_id)
        if yielded:
            pass
        elif accepted_source is None:
            problems.append(f"application accepted raw is outside the immutable component for {raw_id}/{key}")
        else:
            accepted_revisions = set(membership_revisions_by_raw_and_key.get((accepted_raw_id, key), set()))
            accepted_source_revision = accepted_source.get("source_revision")
            if accepted_source_revision is not None:
                accepted_revisions.add(str(accepted_source_revision))
            if str(application.get("accepted_source_revision")) not in accepted_revisions:
                problems.append(f"application accepted source revision has no source evidence for {raw_id}/{key}")
        frontier_kind = application.get("accepted_frontier_kind")
        frontier = application.get("accepted_frontier")
        if not yielded and (frontier_kind not in {"byte", "semantic"} or type(frontier) is not int or frontier < 0):
            problems.append(f"application accepted frontier is malformed for {raw_id}/{key}")
        # A semantic membership application is evidenced by its membership
        # row, not by the raw's byte-chain columns: a multi-session raw keeps a
        # pending-raw byte envelope whose chain fields describe no member.
        membership_evidenced = (frontier_kind == "semantic" or yielded) and (
            raw_id,
            key,
        ) in membership_revisions_by_raw_and_key
        for field in () if membership_evidenced else ("baseline_raw_id", "predecessor_raw_id"):
            if application.get(field) != source.get(field):
                problems.append(f"application {field} does not match source evidence for {raw_id}")
        try:
            decision = ApplicationDecision(str(application.get("decision")))
            content_hash = application.get("accepted_content_hash")
            # SQLite renders ``hex(NULL)`` as an empty string, not NULL.
            content_hash_hex = None if content_hash in (None, "") else bytes.fromhex(str(content_hash)).hex()
            decision_payload = {
                "accepted_raw_id": application.get("accepted_raw_id"),
                "accepted_source_revision": application.get("accepted_source_revision"),
                "accepted_content_hash": content_hash_hex,
                "accepted_frontier_kind": application.get("accepted_frontier_kind"),
                "accepted_frontier": application.get("accepted_frontier"),
                "acquisition_generation": application.get("acquisition_generation"),
                "append_end_offset": application.get("append_end_offset"),
                "baseline_raw_id": application.get("baseline_raw_id"),
                "decision": decision.value,
                "logical_source_key": application.get("logical_source_key"),
                "predecessor_raw_id": application.get("predecessor_raw_id"),
                "raw_id": application.get("raw_id"),
                "session_id": application.get("session_id"),
                "source_revision": application.get("source_revision"),
            }
            expected_decision_id = hashlib.sha256(
                json.dumps(decision_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if str(application.get("decision_id")) != expected_decision_id:
                problems.append(f"application decision identity is not exact for {raw_id}")
        except (TypeError, ValueError):
            problems.append(f"application decision identity is malformed for {raw_id}")
        head = heads_by_key.get(key)
        if head is None:
            problems.append(f"application receipt has no accepted head for {key}")
            continue
        application_authority = (
            str(application.get("session_id")),
            str(application.get("accepted_raw_id")),
            str(application.get("accepted_content_hash")),
        )
        head_authority = (
            str(head.get("session_id")),
            str(head.get("accepted_raw_id")),
            str(head.get("accepted_content_hash")),
        )
        # A prior application of this same raw/content may survive a later
        # membership reclassification that advances its generation. It is
        # still an exact historical decision, but only a row matching the
        # entire current frontier can certify the accepted head.
        if application_authority == head_authority and all(
            application.get(field) == head.get(field)
            for field in (
                "accepted_source_revision",
                "accepted_frontier_kind",
                "accepted_frontier",
                "acquisition_generation",
                "append_end_offset",
            )
        ):
            applications_matching_current_head.add(key)
        materialized_session = sessions_by_id.get(str(head.get("session_id")))
        if materialized_session is None:
            continue
        if str(materialized_session.get("raw_id")) != str(head.get("accepted_raw_id")) or str(
            materialized_session.get("content_hash")
        ) != str(head.get("accepted_content_hash")):
            problems.append(f"materialized session authority does not match the head for {key}")
    for key in sorted(expected_keys - applications_matching_current_head - yielded_keys):
        problems.append(f"no application accepted authority matches the current head for {key}")
    return not problems, tuple(problems)


def _raw_replay_plan_from_expected_json(expected_json: str) -> RawReplayPlan:
    """Rebuild the blocked plan from the blocker's own durable snapshot.

    polylogue-5dzj9: ``raw_authority_blockers.expected_json`` is written by
    every blocker writer as ``RawReplayPlan.to_dict()``, so the blocker row is
    self-sufficient evidence. Readers used to reach the same plan through a
    ``JOIN raw_authority_plans``; that table is per-pass bookkeeping the
    2026-09-15 ruling retires, and a durable authorization must not depend on
    it. This is a straight decode of the snapshot -- no reconstruction from a
    rebuildable tier, and no re-derivation from current evidence.
    """
    payload = json.loads(expected_json)
    return RawReplayPlan(
        plan_id=str(payload["plan_id"]),
        input_digest=str(payload["input_digest"]),
        input_raw_ids=tuple(str(value) for value in payload["input_raw_ids"]),
        logical_keys=tuple(str(value) for value in payload["logical_keys"]),
        authority_witness=json_document(payload["authority_witness"]),
        source_preconditions=json_document(payload["source_preconditions"]),
        index_preconditions=json_document(payload["index_preconditions"]),
    )


_FRONTIER_WITNESS_SCHEMA = "polylogue.raw-authority-frontier-plan.v1"

#: Key under ``raw_authority_blockers.observed_json`` carrying the writer's own
#: declaration of why the blocker exists.  Every blocker writer records one of
#: :data:`BLOCKER_ORIGINS`; readers that must treat blocker classes differently
#: (today: :func:`auto_resolve_stale_plan_blockers`) select on this value
#: positively rather than inferring a class from the absence of some other
#: property.  polylogue-l8tdh: the auto-resolver previously selected "every
#: blocker whose plan is not a frontier plan", which was a correct description
#: of stale-plan blockers only until a second non-frontier blocker writer
#: (:func:`reject_invalid_raw_replay_application`) appeared -- at which point
#: the loop silently began clearing the fail-closed blocker whose own
#: remediation text says automatic convergence must not resume until it is
#: resolved.  A new blocker class must now name itself here, and is not
#: auto-cleared unless it is explicitly declared auto-clearable.
BLOCKER_ORIGIN_KEY = "blocker_origin"

#: An unmet frontier authority obligation published by the reconciler.
BLOCKER_ORIGIN_FRONTIER_OBLIGATION = "frontier_obligation"


def _blocker_kind(*, witness_schema: str) -> str:
    """Classify the stored witness for prepared blocker acknowledgement.

    A ``frontier_obligation`` carries the current frontier plan shape
    (``authority_witness.schema == polylogue.raw-authority-frontier-plan.v1``)
    and covers every obligation state in
    ``polylogue.storage.raw_reconciler._OBLIGATION_STATES``. Anything else is
    a ``stale_plan``: a durable row whose snapshot predates the current plan
    shape, which the resolver re-derives from live evidence instead of
    trusting. Every blocker resolves through the same declared mutation; no
    kind grants an extra effect.
    """
    if witness_schema != _FRONTIER_WITNESS_SCHEMA:
        return "stale_plan"
    return "frontier_obligation"


def describe_raw_authority_blocker(archive_root: Path, blocker_id: str) -> JSONDocument | None:
    """Read-only lookup of one unresolved blocker's resolution-eligibility shape.

    Returns ``None`` when the blocker does not exist or is already resolved.
    This discovery read describes the stored blocker without opening a write
    transaction. Acknowledgement separately prepares the original evidence
    through the resident owner before authorizing its mutation.
    """
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return None
    with closing(_readonly(source_db)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_blockers'"
        ).fetchone()
        if exists is None:
            return None
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT b.blocker_id,
                   json_extract(b.expected_json, '$.plan_id') AS plan_id,
                   b.observed_pass_id, b.reason, b.created_at_ms,
                   COALESCE(json_extract(b.expected_json, '$.authority_witness.schema'), '') AS witness_schema
            FROM raw_authority_blockers AS b
            WHERE b.blocker_id = ? AND b.resolved_at_ms IS NULL
            """,
            (blocker_id,),
        ).fetchone()
    if row is None:
        return None
    kind = _blocker_kind(witness_schema=str(row["witness_schema"]))
    return json_document(
        {
            "blocker_id": str(row["blocker_id"]),
            "plan_id": str(row["plan_id"]),
            "observed_pass_id": (None if row["observed_pass_id"] is None else str(row["observed_pass_id"])),
            "reason": str(row["reason"]),
            "created_at_ms": int(row["created_at_ms"]),
            "kind": kind,
        }
    )


def list_unresolved_raw_authority_blockers(archive_root: Path, *, limit: int = 100, offset: int = 0) -> JSONDocument:
    """Read-only, paginated inventory of unresolved raw-authority blockers.

    Reports each row's ``kind`` (see :func:`_blocker_kind`), which describes
    how prepared acknowledgement reads its stored snapshot, not
    a different effect. This is the operator discovery surface for an
    exact ``--blocker-id``; it reads the blocker rows themselves, so it does
    not depend on any per-pass inspection record.

    Bounded to ``limit`` rows (1-500) per call like every other raw-authority
    reader in this module; ``offset`` plus the returned ``total_count``/
    ``truncated``/``next_offset`` let a caller page through an archive with
    more than 500 unresolved blockers instead of only ever seeing the first
    page.
    """
    if not 1 <= limit <= 500:
        raise ValueError("raw authority blocker list limit must be between 1 and 500")
    if offset < 0:
        raise ValueError("raw authority blocker list offset must be non-negative")

    def _empty_page() -> JSONDocument:
        return json_document(
            {
                "blockers": [],
                "offset": offset,
                "limit": limit,
                "returned_count": 0,
                "total_count": 0,
                "truncated": False,
                "next_offset": None,
            }
        )

    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return _empty_page()
    with closing(_readonly(source_db)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_blockers'"
        ).fetchone()
        if exists is None:
            return _empty_page()
        conn.row_factory = sqlite3.Row
        total_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0]
        )
        rows = conn.execute(
            """
            SELECT b.blocker_id,
                   json_extract(b.expected_json, '$.plan_id') AS plan_id,
                   b.observed_pass_id, b.reason, b.expected_json, b.observed_json, b.created_at_ms,
                   COALESCE(json_extract(b.expected_json, '$.authority_witness.schema'), '') AS witness_schema
            FROM raw_authority_blockers AS b
            WHERE b.resolved_at_ms IS NULL
            ORDER BY b.created_at_ms, b.blocker_id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    blockers = [
        json_document(
            {
                "blocker_id": str(row["blocker_id"]),
                "plan_id": str(row["plan_id"]),
                "observed_pass_id": (None if row["observed_pass_id"] is None else str(row["observed_pass_id"])),
                "reason": str(row["reason"]),
                "expected": json.loads(str(row["expected_json"])),
                "observed": json.loads(str(row["observed_json"])),
                "created_at_ms": int(row["created_at_ms"]),
                "kind": _blocker_kind(witness_schema=str(row["witness_schema"])),
            }
        )
        for row in rows
    ]
    next_offset = offset + len(blockers)
    return json_document(
        {
            "blockers": blockers,
            "offset": offset,
            "limit": limit,
            "returned_count": len(blockers),
            "total_count": total_count,
            "truncated": next_offset < total_count,
            "next_offset": next_offset if next_offset < total_count else None,
        }
    )


__all__ = [
    "BLOCKER_ORIGIN_FRONTIER_OBLIGATION",
    "BLOCKER_ORIGIN_KEY",
    "raw_authority_parser_fingerprint",
    "RawReplayPlan",
    "build_raw_replay_plan",
    "build_raw_replay_plans",
    "describe_raw_authority_blocker",
    "list_unresolved_raw_authority_blockers",
    "iter_parser_census_logical_keys",
    "raw_replay_application_receipt",
    "raw_replay_application_receipt_from_connection",
    "unresolved_raw_authority_blockers",
    "validate_raw_replay_application_receipt",
    "validate_raw_replay_plan",
]


def validated_parser_census_logical_keys(values: Generator[object, None, None]) -> Generator[str, None, None]:
    """Validate ordered receipt keys and settle the actual decoded row stream."""
    last: str | None = None
    with closing(values):
        for value in values:
            check_compute_cancelled()
            if not isinstance(value, str) or last is not None and value <= last:
                raise InvalidParserCensusKeysError("parser identity receipt keys are not sorted unique strings")
            last = value
            try:
                key = canonical_authority_logical_key(value)
            except ValueError as error:
                raise InvalidParserCensusKeysError("parser identity receipt has an invalid authority key") from error
            yield key


_RAW_PLAN_CENSUS_SQL = """
        SELECT raw_id, parser_fingerprint, status, member_count, detail
        FROM raw_membership_census
        WHERE raw_id IN ({marks}) ORDER BY raw_id
        """


_RAW_PLAN_HEAD_SQL = """
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision, hex(accepted_content_hash) AS accepted_content_hash,
                   accepted_frontier_kind, accepted_frontier,
                   acquisition_generation, append_end_offset
            FROM {index_prefix}raw_revision_heads
            WHERE logical_source_key IN ({marks}) ORDER BY logical_source_key
            """


_RAW_PLAN_MEMBERSHIP_SQL = """
        SELECT raw_id, logical_source_key, provider_session_id, source_revision,
               hex(normalized_content_hash) AS normalized_content_hash,
               message_count, predecessor_raw_id, acquisition_generation,
               revision_authority, decision
        FROM raw_session_memberships
        WHERE raw_id IN ({marks})
        ORDER BY raw_id, logical_source_key
        """


_RAW_PLAN_PARSER_CENSUS_SQL = """
        SELECT raw_id, parser_fingerprint, status, logical_keys_json, detail
        FROM raw_authority_parser_census
        WHERE raw_id IN ({marks}) ORDER BY raw_id
        """


_RAW_PLAN_SESSION_SQL = """
        SELECT session_id, raw_id, hex(content_hash) AS content_hash, message_count
        FROM {index_prefix}sessions
        WHERE raw_id IN ({marks}) ORDER BY session_id
        """


_RAW_PLAN_SOURCE_SQL = """
        SELECT raw_id, origin, native_id, source_path, source_index,
               hex(blob_hash) AS blob_hash, blob_size, logical_source_key,
               revision_kind, source_revision, predecessor_source_revision,
               predecessor_raw_id, baseline_raw_id, append_start_offset,
               append_end_offset, acquisition_generation, revision_authority
        FROM raw_sessions WHERE raw_id IN ({marks}) ORDER BY raw_id
        """


def _raw_replay_logical_keys(
    source_rows: list[dict[str, object]], membership_rows: list[dict[str, object]]
) -> tuple[str, ...]:
    # A multi-session raw keeps its pending-raw byte envelope while each of its
    # sessions is governed by its own membership key. The envelope names no
    # session, so it has no accepted head to prove; only its members do.
    membership_raw_ids = {str(row["raw_id"]) for row in membership_rows}
    return tuple(
        sorted(
            {
                str(value)
                for row in (*source_rows, *membership_rows)
                if (value := row.get("logical_source_key")) is not None
                and not (
                    str(value).startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX)
                    and str(row["raw_id"]) in membership_raw_ids
                )
            }
        )
    )


def _raw_replay_plan_from_rows(
    raw_ids: tuple[str, ...],
    logical_keys: tuple[str, ...],
    *,
    source_rows: list[dict[str, object]],
    membership_rows: list[dict[str, object]],
    census_rows: list[dict[str, object]],
    parser_census_rows: list[dict[str, object]],
    head_rows: list[dict[str, object]],
    session_rows: list[dict[str, object]],
) -> RawReplayPlan:
    authority_witness = json_document(
        {
            "parser_census": parser_census_rows,
            "membership_census": census_rows,
            "memberships": membership_rows,
            "revision_heads": head_rows,
        }
    )
    source_preconditions = json_document(
        {"raw_sessions": source_rows, "raw_authority_parser_census": parser_census_rows}
    )
    index_preconditions = json_document({"sessions": session_rows, "revision_heads": head_rows})
    identity = {
        "schema": "polylogue.raw-replay-plan.v2",
        "input_raw_ids": list(raw_ids),
        "logical_keys": list(logical_keys),
        "authority_witness": authority_witness,
        "source_preconditions": source_preconditions,
        "index_preconditions": index_preconditions,
    }
    input_digest = _digest(identity)
    return RawReplayPlan(
        plan_id=f"raw-replay:{input_digest}",
        input_digest=input_digest,
        input_raw_ids=raw_ids,
        logical_keys=logical_keys,
        authority_witness=authority_witness,
        source_preconditions=source_preconditions,
        index_preconditions=index_preconditions,
    )
