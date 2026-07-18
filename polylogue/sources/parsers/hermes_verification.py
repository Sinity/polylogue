"""Hermes ``verification_evidence.db`` importer (fs1.3 / polylogue-wj25).

Hermes's coding-verification ledger (``agent/verification_evidence.py`` in
the bundled hermes-agent checkout) is a passive, self-recorded log of
commands the agent actually ran to verify its own work: lint/typecheck/
build/format/check/test runs plus ad-hoc verification scripts, each with the
literal command, a canonical command name, exit code, pass/fail status, and
a bounded output summary. A separate ``verification_state`` table tracks,
per ``(session_id, root)`` pair, the last recorded event and the file paths
changed since.

This is Hermes's own claim-vs-evidence ledger. Importing it lets Polylogue
correlate what a Hermes session claimed against what Hermes itself verified
-- the flagship differentiator for the Hermes bridge (see polylogue-fs1,
polylogue-wj25).

Schema (live-verified against ``~/.hermes/verification_evidence.db``,
``meta.schema_version = 1``, 2026-07-18; producer source
``agent/verification_evidence.py``):

- ``verification_events(id, created_at, session_id, cwd, root, command,
  canonical_command, kind, scope, status, exit_code, output_summary)``.
  ``exit_code`` is ``NOT NULL`` (always known, unlike Polylogue's
  ``tool_result_exit_code`` NULL-means-unknown convention elsewhere) --
  every recorded event has a definite outcome. ``status`` is exactly
  ``"passed"`` or ``"failed"`` (``exit_code == 0`` vs not); ``kind`` is one
  of ``lint``/``typecheck``/``build``/``format``/``check``/``test``/
  ``ad_hoc``; ``scope`` is ``targeted`` or ``full``.
- ``verification_state(session_id, root, last_event_id, last_edit_at,
  changed_paths_json)``, primary key ``(session_id, root)``.

Payload hygiene: unlike ATOF/ATIF (which never copy LLM prompt/tool-argument
content, only structural presence), ``command``/``canonical_command``/
``output_summary`` round-trip verbatim here by design -- they ARE the
evidence, not conversational content, and the producer already bounds
``output_summary`` to 2000 characters. This matches the existing precedent
that Polylogue retains real command text for tool_use/tool_result blocks
elsewhere in the archive.

Session correlation: grouped by the producer's own ``session_id`` field
(the same raw Hermes session id used by state.db/ATOF/ATIF), landing as a
distinct ``verification:<hermes_session_id>`` observer-evidence session --
the same deferred-physical-merge pattern ``hermes_spans.py`` already
established for ATOF/ATIF, not a new mechanism. A ``session_id`` of
``"default"`` is the producer's own fallback for "session id unknown"
(``session_id=str(session_id or "default")``) and is surfaced as an
explicit ambiguous-correlation caveat, never silently treated as a real id.

Retention: the producer prunes events older than 30 days and caps events at
100 per ``(session_id, root)`` / 10,000 total unreferenced -- this import
reflects only what Hermes currently retains, not a complete historical
ledger; the fidelity declaration says so.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal, TypeAlias

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from .hermes_state import HermesFidelityCapability, HermesImportFidelity

HERMES_VERIFICATION_DB_MARKER = "hermes_verification_evidence_db"
_DEFAULT_SESSION_ID = "default"
_REQUIRED_EVENT_COLUMNS = frozenset(
    {
        "id",
        "created_at",
        "session_id",
        "cwd",
        "root",
        "command",
        "canonical_command",
        "kind",
        "scope",
        "status",
        "exit_code",
        "output_summary",
    }
)
_REQUIRED_STATE_COLUMNS = frozenset({"session_id", "root", "last_event_id", "last_edit_at", "changed_paths_json"})

HermesVerificationEventType: TypeAlias = Literal["hermes_verification_event", "hermes_verification_state"]


def observer_session_provider_id(hermes_session_id: str) -> str:
    """Return the verification-evidence session identity for a Hermes session id.

    Distinct from ``hermes_spans.observer_session_provider_id`` (ATOF/ATIF
    use the ``observer:`` prefix): a physical merge across these
    independently-acquired evidence classes is deferred, same discipline as
    ATOF/ATIF's own deferred merge into the state-db conversational session.
    """
    return f"verification:{hermes_session_id}"


def marker_payload(path: Path, *, profile_root: Path | None = None) -> JSONDocument:
    """Return the JSON marker that routes a raw SQLite blob to this parser."""
    payload: JSONDocument = {
        "polylogue_artifact": HERMES_VERIFICATION_DB_MARKER,
        "verification_db_path": str(path),
    }
    if profile_root is not None:
        payload["profile_root"] = str(profile_root)
    return payload


def looks_like_verification_evidence_db_payload(payload: JSONDocument) -> bool:
    return payload.get("polylogue_artifact") == HERMES_VERIFICATION_DB_MARKER and isinstance(
        payload.get("verification_db_path"), str
    )


def looks_like_verification_evidence_db_path(path: Path) -> bool:
    """Return true when *path* is a readable Hermes verification_evidence.db."""
    try:
        with _connect_readonly(path) as conn:
            return _has_required_tables(conn)
    except sqlite3.Error:
        return False


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _has_required_tables(conn: sqlite3.Connection) -> bool:
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('meta', 'verification_events', 'verification_state')"
        ).fetchall()
    }
    if tables != {"meta", "verification_events", "verification_state"}:
        return False
    return _REQUIRED_EVENT_COLUMNS.issubset(_columns(conn, "verification_events")) and _REQUIRED_STATE_COLUMNS.issubset(
        _columns(conn, "verification_state")
    )


def parse_verification_evidence_db_payload(payload: JSONDocument, fallback_id: str) -> list[ParsedSession]:
    path_value = payload.get("verification_db_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Hermes verification_evidence.db marker is missing verification_db_path")
    return parse_verification_evidence_db(Path(path_value), fallback_id=fallback_id)


def parse_verification_evidence_db(path: Path, *, fallback_id: str | None = None) -> list[ParsedSession]:
    """Parse every verification event/state row from a Hermes verification_evidence.db file."""
    del fallback_id
    with _connect_readonly(path) as conn:
        if not _has_required_tables(conn):
            raise ValueError(f"{path} is not a Hermes verification_evidence.db file")
        schema_version = _schema_version(conn)
        event_rows = list(conn.execute("SELECT * FROM verification_events ORDER BY session_id, id").fetchall())
        state_rows = list(conn.execute("SELECT * FROM verification_state ORDER BY session_id, root").fetchall())

    grouped_events: dict[str, list[ParsedSessionEvent]] = {}
    for row in event_rows:
        session_id = str(row["session_id"])
        grouped_events.setdefault(session_id, []).append(_verification_event(row, schema_version=schema_version))

    grouped_state: dict[str, list[ParsedSessionEvent]] = {}
    for row in state_rows:
        session_id = str(row["session_id"])
        grouped_state.setdefault(session_id, []).append(_verification_state_event(row))

    session_ids = sorted(set(grouped_events) | set(grouped_state))
    return [
        _verification_session(session_id, grouped_events.get(session_id, []), grouped_state.get(session_id, []))
        for session_id in session_ids
    ]


def _schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        return None
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return None


def _verification_event(row: sqlite3.Row, *, schema_version: int | None) -> ParsedSessionEvent:
    status = str(row["status"])
    exit_code = row["exit_code"]
    payload: dict[str, object] = {
        "event_id": row["id"],
        "session_id": row["session_id"],
        "schema_version": schema_version,
        "cwd": row["cwd"],
        "root": row["root"],
        "command": row["command"],
        "canonical_command": row["canonical_command"],
        "kind": row["kind"],
        "scope": row["scope"],
        "status": status,
        "is_error": status != "passed",
        "exit_code": int(exit_code) if exit_code is not None else None,
        "output_summary": row["output_summary"],
        "ambiguous_correlation": str(row["session_id"]) == _DEFAULT_SESSION_ID,
    }
    return ParsedSessionEvent(
        event_type="hermes_verification_event",
        timestamp=_optional_text(row["created_at"]),
        payload=payload,
    )


def _verification_state_event(row: sqlite3.Row) -> ParsedSessionEvent:
    changed_paths = _changed_paths(row["changed_paths_json"])
    payload = {
        "session_id": row["session_id"],
        "root": row["root"],
        "last_event_id": row["last_event_id"],
        "last_edit_at": row["last_edit_at"],
        "changed_paths": changed_paths,
        "changed_path_count": len(changed_paths),
        "ambiguous_correlation": str(row["session_id"]) == _DEFAULT_SESSION_ID,
    }
    return ParsedSessionEvent(
        event_type="hermes_verification_state",
        timestamp=_optional_text(row["last_edit_at"]),
        payload=payload,
    )


def _changed_paths(raw: object) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _verification_session(
    session_id: str,
    events: list[ParsedSessionEvent],
    state_events: list[ParsedSessionEvent],
) -> ParsedSession:
    passed = sum(1 for event in events if event.payload.get("status") == "passed")
    failed = sum(1 for event in events if event.payload.get("status") == "failed")
    roots = sorted(
        {str(event.payload.get("root")) for event in state_events} | {str(e.payload.get("root")) for e in events}
    )
    summary_text = (
        f"Hermes verification ledger: {len(events)} event(s) ({passed} passed, {failed} failed); "
        f"{len(state_events)} state row(s) across {len(roots)} root(s)."
    )
    provider_session_id = observer_session_provider_id(session_id)
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=provider_session_id,
        title=f"Hermes verification ledger: {session_id}",
        messages=[
            ParsedMessage(
                provider_message_id=f"{provider_session_id}:verification-summary",
                role=Role.SYSTEM,
                text=summary_text,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=summary_text)],
                position=0,
                variant_index=0,
                is_active_path=True,
                material_origin=MaterialOrigin.RUNTIME_CONTEXT,
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="hermes_verification_correlation",
                payload={
                    "hermes_conversation_session_id_prefix": session_id,
                    "join_key": "sessions.native_id",
                    "note": (
                        "This session carries Hermes's own claim-vs-evidence verification ledger "
                        "only; correlate with the state-db-ingested conversational session and any "
                        "ATOF/ATIF observer session sharing this raw Hermes session id."
                    ),
                },
            ),
            *events,
            *state_events,
        ],
        ingest_flags=["hermes:verification-evidence"],
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def import_fidelity_declaration(sessions: list[ParsedSession]) -> HermesImportFidelity:
    """Declare the fidelity this verification-evidence importer can substantiate.

    Aggregates across every returned observer session (one per distinct
    Hermes session id in the ledger), matching ``hermes_state.py``'s plural
    ``sessions: list[ParsedSession]`` fidelity shape rather than picking one
    session as an arbitrary representative.
    """
    events = [
        event
        for session in sessions
        for event in session.session_events
        if event.event_type == "hermes_verification_event"
    ]
    state_events = [
        event
        for session in sessions
        for event in session.session_events
        if event.event_type == "hermes_verification_state"
    ]
    total = max(len(events) + len(state_events), 1)

    def exact_if_observed(count: int, detail: str) -> HermesFidelityCapability:
        return HermesFidelityCapability(
            status="exact" if count else "absent", observed=count, expected=total, counts={}, detail=detail
        )

    ambiguous = sum(1 for event in events + state_events if event.payload.get("ambiguous_correlation"))
    capabilities = {
        "command_evidence": exact_if_observed(
            len(events),
            "command/canonical_command/kind/scope round-trip verbatim from verification_events; "
            "this is Hermes's own recorded evidence, not conversational content, so payload "
            "hygiene does not bound it (unlike ATOF/ATIF).",
        ),
        "outcome_evidence": exact_if_observed(
            len(events),
            "status/exit_code are structurally authoritative -- exit_code is NOT NULL in the "
            "producer schema, unlike the NULL-means-unknown tool_result_exit_code convention "
            "elsewhere in this archive.",
        ),
        "output_evidence": exact_if_observed(
            sum(1 for event in events if event.payload.get("output_summary")),
            "output_summary round-trips verbatim, bounded to 2000 characters by the producer.",
        ),
        "changed_paths": exact_if_observed(
            len(state_events), "verification_state.changed_paths_json round-trips as a typed path list."
        ),
        "correlation": HermesFidelityCapability(
            status="degraded" if ambiguous else ("exact" if events or state_events else "absent"),
            observed=(len(events) + len(state_events)) - ambiguous,
            expected=total,
            counts={"ambiguous": ambiguous},
            detail="Correlation is exact via the producer's own session_id field, except rows "
            "where Hermes recorded session_id='default' (its own fallback for unknown session "
            "identity), which are marked ambiguous rather than silently trusted.",
        ),
        "retention_completeness": HermesFidelityCapability(
            status="degraded",
            observed=len(events),
            expected=total,
            counts={},
            detail="The producer prunes events older than 30 days and caps at 100 events per "
            "(session_id, root) / 10,000 unreferenced total -- this import reflects only what "
            "Hermes currently retains, not a complete historical verification ledger.",
        ),
    }
    caveats = tuple(f"{name}: {cap.detail}" for name, cap in capabilities.items() if cap.status != "exact")
    return HermesImportFidelity(
        producer="Hermes coding-verification ledger (verification_evidence.db, live schema v1)",
        schema_version=1,
        profile_namespace=None,
        acquisition_method="sqlite_backup",
        retained_blob_reproducibility=HermesFidelityCapability(
            status="exact",
            observed=1,
            expected=1,
            counts={},
            detail="The raw verification_evidence.db bytes are snapshotted before parsing, like state.db.",
        ),
        capabilities=capabilities,
        caveats=caveats,
    )


__all__ = [
    "HERMES_VERIFICATION_DB_MARKER",
    "HermesVerificationEventType",
    "import_fidelity_declaration",
    "looks_like_verification_evidence_db_path",
    "looks_like_verification_evidence_db_payload",
    "marker_payload",
    "observer_session_provider_id",
    "parse_verification_evidence_db",
    "parse_verification_evidence_db_payload",
]
