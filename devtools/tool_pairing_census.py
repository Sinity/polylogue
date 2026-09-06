"""Account for every tool call the archive records without a paired result.

Runs over the canonical per-tool identity plan -- ``action_pairs`` for the call
side, the ``tool_result`` covering index for the answer side -- so the whole
archive is classified without the archive-wide anti-join that a naive reading
needs (measured: over 90s and cancelled on a 40 GB index; this reads both sides
from indexes instead).

Every no-result call and every unmatched physical result lands in exactly one
class. The classes say what the evidence says: a provider construct that emits
no answer, a source that no longer survives, an acquisition the source has
already outgrown, a call the transcript abandoned, or a defect. ``unknown`` is
a real class and its count is expected to be zero; a non-zero count is the
census reporting that the taxonomy is incomplete, not a rounding error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection
from typing import NamedTuple, cast

from devtools.tool_evidence_oracle import SUPPORTED_ORIGINS, declare_tool_evidence
from polylogue.config import get_config
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

#: How a call is identified on the wire.
IDENTITY_NATIVE = "provider_native_id"
IDENTITY_STRUCTURAL = "structural_id"
IDENTITY_ABSENT = "no_id"

#: Where the call sits in its session's stored transcript.
POSITION_TAIL = "tail"
POSITION_NEAR_TAIL = "near_tail"
POSITION_INTERIOR = "interior"

#: What survives of the source the call came from.
SOURCE_PRESENT = "present"
SOURCE_BYTES_ABSENT = "bytes_absent"
SOURCE_RECORD_ABSENT = "record_absent"
SOURCE_UNCHECKED = "unchecked"

#: Whether the acquisition covers the source as it now stands.
COMPLETION_SETTLED = "settled"
COMPLETION_SUPERSEDED = "superseded_acquisition"
COMPLETION_UNKNOWN = "unknown"

#: Terminal classification. Every cohort carries exactly one.
CLASS_UNSUPPORTED_CONSTRUCT = "unsupported_provider_construct"
CLASS_SIDECAR_OWNED = "sidecar_owned"
CLASS_SOURCE_ABSENT = "source_absent"
CLASS_SOURCE_TRUNCATED = "source_truncated"
CLASS_IN_FLIGHT = "in_flight_or_interrupted"
CLASS_SOURCE_OMISSION = "source_omission"
CLASS_PROVIDER_FANOUT = "provider_fanout"
CLASS_PARSER_DEFECT = "parser_or_pairing_defect"
CLASS_HISTORICAL_DAMAGE = "historical_derived_damage"
CLASS_UNKNOWN = "unknown"

#: Classes that name a defect somebody has to own.
DEFECT_CLASSES: frozenset[str] = frozenset({CLASS_PARSER_DEFECT, CLASS_HISTORICAL_DAMAGE})


@dataclass(frozen=True, slots=True)
class ConstructRule:
    """One provider construct, recognized from stored identity or tool name."""

    construct: str
    origins: frozenset[str]
    classification: str
    rationale: str
    tool_id_like: str | None = None
    tool_names: frozenset[str] = frozenset()


#: Recognized constructs, in match order. The first rule whose origin and shape
#: match names the construct; anything unmatched is a plain provider-native
#: call and is classified from position, source survival and acquisition state.
CONSTRUCT_RULES: tuple[ConstructRule, ...] = (
    ConstructRule(
        construct="codex_code_mode_child",
        origins=frozenset({"codex-session"}),
        classification=CLASS_UNSUPPORTED_CONSTRUCT,
        rationale=(
            "One Codex `exec` call carries a script containing several child calls and the wire "
            "returns one combined output for the script; children the combined output does not "
            "cover have no answer to pair. The parent call id and child index stay on the block's "
            "tool_input provenance."
        ),
        tool_id_like="%::polylogue-child::%",
    ),
    ConstructRule(
        construct="codex_web_search_call",
        origins=frozenset({"codex-session"}),
        classification=CLASS_UNSUPPORTED_CONSTRUCT,
        rationale=(
            "The rollout emits a `web_search_call` item and no matching output item; the retrieved "
            "sources arrive inside the following assistant message."
        ),
        tool_names=frozenset({"web_search_call"}),
    ),
    ConstructRule(
        construct="claude_web_structural_pair",
        origins=frozenset({"claude-ai-export", "claude-design-session"}),
        classification="",
        rationale=(
            "The Claude web transcript pairs a tool_use segment with the tool_result segment that "
            "follows it and puts no id on either; the parser assigns one structural id per pair."
        ),
        tool_id_like="structural:claude-web:%",
    ),
)

_PLAIN_CONSTRUCT = "provider_native_call"

#: Gap sessions retained per cohort as replay candidates. Replay samples a few
#: and some fail to load, so the list needs slack -- but not the whole archive.
_REPLAY_CANDIDATES_PER_COHORT = 64


class CallCohortKey(NamedTuple):
    """The axes a no-result call is counted along."""

    origin: str
    construct: str
    identity: str
    position: str
    source_presence: str
    completion: str
    classification: str


class ResultCohortKey(NamedTuple):
    """The axes an unmatched physical result is counted along."""

    origin: str
    construct: str
    identity: str
    source_presence: str
    classification: str


@dataclass(frozen=True, slots=True)
class CensusArgs:
    archive_root: Path | None
    index_db: Path | None
    source_db: Path | None
    near_tail_messages: int
    check_source: bool
    replay_sessions: int
    json: bool


@dataclass(frozen=True, slots=True)
class TimedQuery:
    name: str
    seconds: float
    plan: tuple[str, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools archive tool-pairing-census",
        description="Classify every tool call/result pairing gap in an archive against declared evidence.",
    )
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--index-db", type=Path, default=None, help="Read a specific index database.")
    parser.add_argument("--source-db", type=Path, default=None, help="Read a specific source database.")
    parser.add_argument(
        "--near-tail-messages",
        type=int,
        default=2,
        help="How many messages before a session's last one still count as its tail.",
    )
    parser.add_argument(
        "--no-source-check",
        dest="check_source",
        action="store_false",
        help="Skip source survival and acquisition state; every cohort reports source presence 'unchecked'.",
    )
    parser.add_argument(
        "--replay-sessions",
        type=int,
        default=3,
        help=(
            "Sessions per cohort to re-parse from surviving source bytes, deciding whether a gap "
            "is the source's or the producer's. Zero skips replay and leaves those cohorts "
            "classified from stored evidence alone."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    return parser


# --------------------------------------------------------------------------
# SQL helpers
# --------------------------------------------------------------------------

_ORIGIN_EXPR = "substr({column}, 1, instr({column}, ':') - 1)"


def _as_int(value: object) -> int:
    """Read one SQLite aggregate cell, treating a NULL count as zero."""
    return int(value) if isinstance(value, int) else 0


def _origin_of(session_id: str) -> str:
    head, _, _ = session_id.partition(":")
    return head


def _timed(
    conn: Connection, name: str, sql: str, params: Sequence[object] = ()
) -> tuple[list[tuple[object, ...]], TimedQuery]:
    plan = tuple(str(row[-1]) for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", tuple(params)).fetchall())
    started = time.monotonic()
    rows = conn.execute(sql, tuple(params)).fetchall()
    return rows, TimedQuery(name=name, seconds=round(time.monotonic() - started, 3), plan=plan)


def _table_columns(conn: Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _identity_of(tool_id: str | None) -> str:
    if not tool_id:
        return IDENTITY_ABSENT
    return IDENTITY_STRUCTURAL if tool_id.startswith("structural:") else IDENTITY_NATIVE


def _construct_of(origin: str, tool_id: str | None, tool_name: str | None) -> tuple[str, ConstructRule | None]:
    for rule in CONSTRUCT_RULES:
        if origin not in rule.origins:
            continue
        if rule.tool_id_like is not None:
            needle = rule.tool_id_like.strip("%")
            if tool_id and needle in tool_id:
                return rule.construct, rule
            continue
        if tool_name is not None and tool_name in rule.tool_names:
            return rule.construct, rule
    return _PLAIN_CONSTRUCT, None


# --------------------------------------------------------------------------
# Source survival and acquisition state
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceState:
    presence: str
    completion: str


def _source_states(
    source_db: Path | None,
    blob_root: Path | None,
    session_ids: Iterable[str],
    *,
    enabled: bool,
) -> dict[str, SourceState]:
    """Resolve source survival and acquisition freshness for cohort sessions."""
    sessions = sorted(set(session_ids))
    if not enabled or source_db is None or not source_db.exists():
        return {session_id: SourceState(SOURCE_UNCHECKED, COMPLETION_UNKNOWN) for session_id in sessions}
    states: dict[str, SourceState] = {}
    conn = open_readonly_connection(source_db, validate_schema=False)
    try:
        for session_id in sessions:
            origin, _, native_id = session_id.partition(":")
            row = conn.execute(
                """
                SELECT blob_hash, acquired_at_ms, file_mtime_ms
                FROM raw_sessions
                WHERE origin = ? AND native_id = ?
                ORDER BY acquired_at_ms DESC
                LIMIT 1
                """,
                (origin, native_id),
            ).fetchone()
            if row is None:
                states[session_id] = SourceState(SOURCE_RECORD_ABSENT, COMPLETION_UNKNOWN)
                continue
            blob_hash, acquired_at_ms, file_mtime_ms = row
            completion = COMPLETION_UNKNOWN
            if isinstance(acquired_at_ms, int) and isinstance(file_mtime_ms, int):
                completion = COMPLETION_SUPERSEDED if file_mtime_ms > acquired_at_ms else COMPLETION_SETTLED
            presence = SOURCE_PRESENT
            if blob_root is not None and isinstance(blob_hash, (bytes, bytearray)):
                digest = bytes(blob_hash).hex()
                if not (blob_root / digest[:2] / digest[2:]).exists():
                    presence = SOURCE_BYTES_ABSENT
            states[session_id] = SourceState(presence, completion)
    finally:
        conn.close()
    return states


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def _classify_call(
    *,
    rule: ConstructRule | None,
    position: str,
    source: SourceState,
) -> str:
    """Name the one class a no-result call belongs to."""
    if rule is not None and rule.classification:
        return rule.classification
    if source.presence == SOURCE_RECORD_ABSENT:
        return CLASS_SOURCE_ABSENT
    if source.presence == SOURCE_BYTES_ABSENT:
        return CLASS_SOURCE_TRUNCATED
    if position != POSITION_INTERIOR:
        return CLASS_IN_FLIGHT
    # The transcript keeps running past the call and never records an answer:
    # an abandoned invocation, which is what `no_result` means.
    return CLASS_SOURCE_OMISSION


def _classify_result(*, owner_present: bool, source: SourceState) -> str:
    if source.presence == SOURCE_RECORD_ABSENT:
        return CLASS_SOURCE_ABSENT
    if owner_present:
        return CLASS_PROVIDER_FANOUT
    return CLASS_SOURCE_OMISSION


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(args: CensusArgs) -> dict[str, object]:
    config = get_config()
    archive_root = (args.archive_root or config.archive_root).expanduser().resolve()
    index_db = args.index_db or archive_root / "index.db"
    source_db = args.source_db or archive_root / "source.db"
    blob_root = archive_root / "blob"

    queries: list[TimedQuery] = []
    # The census reads archives whose derived schema predates the current
    # code; refusing them would make the historical cohorts unmeasurable.
    conn = open_readonly_connection(index_db, validate_schema=False)
    try:
        action_columns = _table_columns(conn, "action_pairs")
        block_columns = _table_columns(conn, "blocks")

        # Only aggregates cross into Python for the paired majority: the whole
        # call side is millions of rows on a real archive and materializing it
        # is what the per-tool identity plan exists to avoid.
        origin_rows, timing = _timed(
            conn,
            "calls_by_origin",
            f"""
            SELECT {_ORIGIN_EXPR.format(column="session_id")} AS origin,
                   COUNT(*),
                   SUM(tool_result_block_id IS NOT NULL)
            FROM action_pairs
            GROUP BY origin
            """,
        )
        queries.append(timing)

        result_total_rows, timing = _timed(
            conn,
            "physical_results",
            """
            SELECT COUNT(*)
            FROM blocks INDEXED BY idx_blocks_tool_result_outcome
            WHERE block_type = 'tool_result'
            """,
        )
        queries.append(timing)

        # The no-result calls are the denominator this census classifies, and
        # each one's transcript position comes back with it -- reading every
        # message of every affected session would restore the unbounded scan.
        gap_rows, timing = _timed(
            conn,
            "no_result_calls",
            """
            SELECT ap.session_id, ap.tool_id, ap.tool_name, m.position,
                   (SELECT MAX(position) FROM messages tail WHERE tail.session_id = ap.session_id)
            FROM action_pairs ap
            LEFT JOIN messages m ON m.message_id = ap.message_id
            WHERE ap.tool_result_block_id IS NULL
            """,
        )
        queries.append(timing)

        # SQLite compares the two identity aggregates and returns only the
        # residue, so the anti-join never lands in Python.
        unmatched_rows, timing = _timed(
            conn,
            "unmatched_results",
            """
            SELECT r.session_id, r.tool_id, r.n - COALESCE(u.n, 0), COALESCE(u.n, 0) > 0
            FROM (
                SELECT session_id, tool_id, COUNT(*) AS n
                FROM blocks INDEXED BY idx_blocks_tool_result_outcome
                WHERE block_type = 'tool_result'
                GROUP BY session_id, tool_id
            ) r
            LEFT JOIN (
                SELECT session_id, tool_id, COUNT(*) AS n
                FROM action_pairs
                WHERE tool_id IS NOT NULL AND tool_id <> ''
                GROUP BY session_id, tool_id
            ) u ON u.session_id = r.session_id AND u.tool_id = r.tool_id
            WHERE r.tool_id IS NULL OR r.tool_id = '' OR r.n > COALESCE(u.n, 0)
            """,
        )
        queries.append(timing)
    finally:
        conn.close()

    archived_results = _as_int(result_total_rows[0][0]) if result_total_rows else 0

    origin_totals: dict[str, dict[str, int]] = {}
    for origin_value, call_count, paired_count in origin_rows:
        origin_totals[str(origin_value)] = {
            "calls": _as_int(call_count),
            "paired": _as_int(paired_count),
            "no_result": 0,
            "unmatched_results": 0,
        }

    unmatched_results: list[tuple[str, str | None, int, bool]] = [
        (str(session_id), str(tool_id) if tool_id else None, _as_int(surplus), bool(owner_present))
        for session_id, tool_id, surplus, owner_present in unmatched_rows
        if _as_int(surplus) > 0
    ]

    gap_sessions = {str(row[0]) for row in gap_rows}
    cohort_sessions = gap_sessions | {session_id for session_id, _, _, _ in unmatched_results}
    source_states = _source_states(
        source_db if source_db.exists() else None,
        blob_root if blob_root.exists() else None,
        cohort_sessions,
        enabled=args.check_source,
    )
    unchecked = SourceState(SOURCE_UNCHECKED, COMPLETION_UNKNOWN)

    call_cohorts: dict[CallCohortKey, int] = {}
    cohort_members: dict[tuple[str, str], list[str]] = {}
    for session_row, tool_id_value, tool_name_value, position_value, last_position in gap_rows:
        session_id = str(session_row)
        tool_id = str(tool_id_value) if tool_id_value else None
        tool_name = str(tool_name_value) if tool_name_value else None
        origin = _origin_of(session_id)
        totals = origin_totals.setdefault(origin, {"calls": 0, "paired": 0, "no_result": 0, "unmatched_results": 0})
        totals["no_result"] += 1
        construct, rule = _construct_of(origin, tool_id, tool_name)
        if rule is None or not rule.classification:
            members = cohort_members.setdefault((origin, construct), [])
            # Replay samples a handful per cohort; keeping every gap session
            # would grow this list with the archive for no extra evidence.
            if len(members) < _REPLAY_CANDIDATES_PER_COHORT and (not members or members[-1] != session_id):
                members.append(session_id)
        position = _position_of(
            position_value,
            last_position,
            near_tail=args.near_tail_messages,
        )
        source = source_states.get(session_id, unchecked)
        classification = _classify_call(rule=rule, position=position, source=source)
        call_key = CallCohortKey(
            origin=origin,
            construct=construct,
            identity=_identity_of(tool_id),
            position=position,
            source_presence=source.presence,
            completion=source.completion,
            classification=classification,
        )
        call_cohorts[call_key] = call_cohorts.get(call_key, 0) + 1

    result_cohorts: dict[ResultCohortKey, int] = {}
    for session_id, tool_id, count, owner_present in unmatched_results:
        origin = _origin_of(session_id)
        totals = origin_totals.setdefault(origin, {"calls": 0, "paired": 0, "no_result": 0, "unmatched_results": 0})
        totals["unmatched_results"] += count
        construct, _rule = _construct_of(origin, tool_id, None)
        source = source_states.get(session_id, unchecked)
        classification = _classify_result(owner_present=owner_present, source=source)
        result_key = ResultCohortKey(
            origin=origin,
            construct=construct,
            identity=_identity_of(tool_id),
            source_presence=source.presence,
            classification=classification,
        )
        result_cohorts[result_key] = result_cohorts.get(result_key, 0) + count

    replay: list[ReplayFinding] = []
    if args.replay_sessions > 0 and source_db.exists() and blob_root.exists():
        replay = _replay_cohorts(
            source_db=source_db,
            blob_root=blob_root,
            cohort_sessions=cohort_members,
            archived_answers=_archived_answers(index_db, cohort_members),
            sample=args.replay_sessions,
        )
    call_cohorts = _apply_replay(call_cohorts, replay)

    calls = [{**key._asdict(), "count": count} for key, count in sorted(call_cohorts.items())]
    results = [{**key._asdict(), "count": count} for key, count in sorted(result_cohorts.items())]

    by_class: dict[str, int] = {}
    for cohort in (*calls, *results):
        name = str(cohort["classification"])
        by_class[name] = by_class.get(name, 0) + cast("int", cohort["count"])

    unexplained_total = 0
    conservation: list[dict[str, object]] = []
    for origin, totals in sorted(origin_totals.items()):
        unexplained = totals["calls"] - totals["paired"] - totals["no_result"]
        unexplained_total += unexplained
        conservation.append(
            {
                "origin": origin,
                "calls": totals["calls"],
                "paired": totals["paired"],
                "no_result": totals["no_result"],
                "unmatched_results": totals["unmatched_results"],
                "unexplained": unexplained,
            }
        )

    return {
        "archive": {
            "root": str(archive_root),
            "index_db": str(index_db),
            "source_checked": args.check_source and source_db.exists(),
        },
        "derived_schema": {
            "action_pairs_has_tool_outcome": "tool_outcome" in action_columns,
            "blocks_has_tool_outcome": "tool_outcome" in block_columns,
        },
        "denominator": {
            "tool_calls": sum(totals["calls"] for totals in origin_totals.values()),
            "paired_calls": sum(totals["paired"] for totals in origin_totals.values()),
            "no_result_calls": sum(totals["no_result"] for totals in origin_totals.values()),
            "physical_results": archived_results,
            "unmatched_results": sum(totals["unmatched_results"] for totals in origin_totals.values()),
        },
        "constructs": [
            {
                "construct": rule.construct,
                "origins": sorted(rule.origins),
                "classification": rule.classification or "derived_from_evidence",
                "rationale": rule.rationale,
            }
            for rule in CONSTRUCT_RULES
        ],
        "calls": calls,
        "results": results,
        "classification_totals": by_class,
        "conservation": conservation,
        "replay": [finding.to_dict() for finding in replay],
        "verdict": {
            "unclassified": by_class.get(CLASS_UNKNOWN, 0),
            "defects": sum(by_class.get(name, 0) for name in DEFECT_CLASSES),
            "unexplained": unexplained_total,
        },
        "plan": [{"query": item.name, "seconds": item.seconds, "steps": list(item.plan)} for item in queries],
    }


# --------------------------------------------------------------------------
# Replay: what does the current production route make of the same bytes?
# --------------------------------------------------------------------------

#: What re-parsing a cohort's surviving source bytes shows.
REPLAY_SOURCE_STATES_NO_ANSWER = "source_declares_no_answer"
REPLAY_PRODUCER_LOSES_ANSWER = "current_parse_loses_a_declared_answer"
REPLAY_ARCHIVE_TRAILS_PARSER = "archive_trails_current_parse"
REPLAY_AGREES = "archive_matches_current_parse"
REPLAY_UNAVAILABLE = "source_bytes_unavailable"


@dataclass(frozen=True, slots=True)
class ReplayFinding:
    origin: str
    construct: str
    session_id: str
    state: str
    declared_answers: int
    parsed_answers: int
    archived_answers: int

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "construct": self.construct,
            "session_id": self.session_id,
            "state": self.state,
            "declared_answers": self.declared_answers,
            "parsed_answers": self.parsed_answers,
            "archived_answers": self.archived_answers,
        }


#: How a replay state resolves the cohort it was sampled from.
REPLAY_CLASSIFICATION = {
    REPLAY_SOURCE_STATES_NO_ANSWER: CLASS_SOURCE_OMISSION,
    REPLAY_PRODUCER_LOSES_ANSWER: CLASS_PARSER_DEFECT,
    REPLAY_ARCHIVE_TRAILS_PARSER: CLASS_HISTORICAL_DAMAGE,
    REPLAY_AGREES: CLASS_SOURCE_OMISSION,
}


def _load_source_payload(source_db: Path, blob_root: Path, session_id: str) -> tuple[object, str] | None:
    """Read one session's acquired bytes back out of the blob store."""
    origin, _, native_id = session_id.partition(":")
    conn = open_readonly_connection(source_db, validate_schema=False)
    try:
        row = conn.execute(
            """
            SELECT blob_hash, source_path
            FROM raw_sessions
            WHERE origin = ? AND native_id = ?
            ORDER BY acquired_at_ms DESC
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not isinstance(row[0], (bytes, bytearray)):
        return None
    digest = bytes(row[0]).hex()
    blob = blob_root / digest[:2] / digest[2:]
    if not blob.exists():
        return None
    raw = blob.read_bytes()
    try:
        return json.loads(raw), str(row[1] or "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        records: list[object] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return (records, str(row[1] or "")) if records else None


def _payload_for_session(payload: object, session_id: str) -> object:
    """Narrow a multi-conversation export down to the one session under replay."""
    _origin, _, native_id = session_id.partition(":")
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict) and native_id in {
                entry.get("uuid"),
                entry.get("id"),
                entry.get("conversation_id"),
            }:
                return entry
    return payload


def _parsed_answer_count(payload: object, session_id: str, source_path: str) -> int | None:
    """Count the answers the current production route pairs for one session."""
    from polylogue.core.enums import BlockType
    from polylogue.sources.dispatch import detect_provider, parse_payload

    scoped = _payload_for_session(payload, session_id)
    provider = detect_provider(scoped, source_path or None)
    if provider is None:
        return None
    try:
        sessions = parse_payload(provider, scoped, session_id.partition(":")[2], source_path=source_path or None)
    except Exception:  # a refusing parser is a replay outcome, not a census crash
        return None
    answered = 0
    for parsed in sessions:
        uses: dict[str, int] = {}
        results: dict[str, int] = {}
        for message in parsed.messages:
            for block in message.blocks:
                if block.type is BlockType.TOOL_USE and block.tool_id:
                    uses[block.tool_id] = uses.get(block.tool_id, 0) + 1
                elif block.type is BlockType.TOOL_RESULT and block.tool_id:
                    results[block.tool_id] = results.get(block.tool_id, 0) + 1
        answered += sum(min(count, results.get(tool_id, 0)) for tool_id, count in uses.items())
    return answered


def _archived_answers(index_db: Path, cohort_sessions: dict[tuple[str, str], list[str]]) -> dict[str, int]:
    """Count the answers the archive already holds, for replay candidates only.

    Replay compares three readings of one session; the archive's is the only
    one that comes from the index, and it is needed for the sampled sessions
    alone rather than for every session in the archive.
    """
    sessions = sorted({session_id for members in cohort_sessions.values() for session_id in members})
    if not sessions:
        return {}
    answers: dict[str, int] = {}
    conn = open_readonly_connection(index_db, validate_schema=False)
    try:
        for session_id in sessions:
            row = conn.execute(
                "SELECT COUNT(*) FROM action_pairs WHERE session_id = ? AND tool_result_block_id IS NOT NULL",
                (session_id,),
            ).fetchone()
            answers[session_id] = int(row[0]) if row else 0
    finally:
        conn.close()
    return answers


def _replay_cohorts(
    *,
    source_db: Path,
    blob_root: Path,
    cohort_sessions: dict[tuple[str, str], list[str]],
    archived_answers: dict[str, int],
    sample: int,
) -> list[ReplayFinding]:
    findings: list[ReplayFinding] = []
    for (origin, construct), sessions in sorted(cohort_sessions.items()):
        if origin not in SUPPORTED_ORIGINS:
            continue
        sampled = 0
        for session_id in sessions:
            if sampled >= sample:
                break
            loaded = _load_source_payload(source_db, blob_root, session_id)
            if loaded is None:
                continue
            payload, source_path = loaded
            scoped = _payload_for_session(payload, session_id)
            try:
                declared = declare_tool_evidence(scoped, origin=origin)
            except ValueError:
                continue
            declared_answers = sum(1 for call in declared.calls if call.outcome != "no_result")
            parsed_answers = _parsed_answer_count(payload, session_id, source_path)
            if parsed_answers is None:
                continue
            sampled += 1
            archived = archived_answers.get(session_id, 0)
            if parsed_answers < declared_answers:
                state = REPLAY_PRODUCER_LOSES_ANSWER
            elif archived < parsed_answers:
                state = REPLAY_ARCHIVE_TRAILS_PARSER
            elif declared_answers == parsed_answers == archived:
                state = REPLAY_AGREES
            else:
                state = REPLAY_SOURCE_STATES_NO_ANSWER
            findings.append(
                ReplayFinding(
                    origin=origin,
                    construct=construct,
                    session_id=session_id,
                    state=state,
                    declared_answers=declared_answers,
                    parsed_answers=parsed_answers,
                    archived_answers=archived,
                )
            )
        if sampled == 0 and sessions:
            findings.append(
                ReplayFinding(
                    origin=origin,
                    construct=construct,
                    session_id=sessions[0],
                    state=REPLAY_UNAVAILABLE,
                    declared_answers=0,
                    parsed_answers=0,
                    archived_answers=0,
                )
            )
    return findings


def _apply_replay(
    call_cohorts: dict[CallCohortKey, int],
    replay: Sequence[ReplayFinding],
) -> dict[CallCohortKey, int]:
    """Let replay overrule a stored-evidence guess for the cohorts it sampled.

    A cohort classified from stored evidence alone cannot tell an abandoned
    call from a producer that drops an answer the source states. Replay reads
    the same bytes through the current route and through the independent
    oracle, so where it ran, its verdict is the classification.
    """
    verdicts: dict[tuple[str, str], str] = {}
    for finding in replay:
        if finding.state == REPLAY_UNAVAILABLE:
            continue
        classification = REPLAY_CLASSIFICATION[finding.state]
        cohort = (finding.origin, finding.construct)
        # A defect found in any sampled session is the cohort's answer; a
        # single clean sample must not clear a reproduced loss.
        if verdicts.get(cohort) in DEFECT_CLASSES:
            continue
        verdicts[cohort] = classification
    if not verdicts:
        return call_cohorts
    updated: dict[CallCohortKey, int] = {}
    for key, count in call_cohorts.items():
        new_key = key
        # Source survival is decided by what is stored, not by a replay that
        # could only run where the bytes are still there.
        if key.source_presence in {SOURCE_PRESENT, SOURCE_UNCHECKED} and (key.origin, key.construct) in verdicts:
            new_key = key._replace(classification=verdicts[(key.origin, key.construct)])
        updated[new_key] = updated.get(new_key, 0) + count
    return updated


def _position_of(position: object, last_position: object, *, near_tail: int) -> str:
    if not isinstance(position, int) or not isinstance(last_position, int):
        return POSITION_INTERIOR
    if position >= last_position:
        return POSITION_TAIL
    if position >= last_position - near_tail:
        return POSITION_NEAR_TAIL
    return POSITION_INTERIOR


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    args = CensusArgs(
        archive_root=parsed.archive_root,
        index_db=parsed.index_db,
        source_db=parsed.source_db,
        near_tail_messages=parsed.near_tail_messages,
        check_source=parsed.check_source,
        replay_sessions=parsed.replay_sessions,
        json=parsed.json,
    )
    report = build_report(args)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    denominator = cast("dict[str, int]", report["denominator"])
    verdict = cast("dict[str, int]", report["verdict"])
    print(
        "tool-pairing-census: "
        f"calls={denominator['tool_calls']} "
        f"no_result={denominator['no_result_calls']} "
        f"unmatched_results={denominator['unmatched_results']} "
        f"unclassified={verdict['unclassified']} "
        f"defects={verdict['defects']} "
        f"unexplained={verdict['unexplained']}"
    )
    totals = cast("dict[str, int]", report["classification_totals"])
    for name, count in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {name}: {count}")
    for item in cast("list[dict[str, object]]", report["plan"]):
        print(f"  query {item['query']}: {item['seconds']}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
