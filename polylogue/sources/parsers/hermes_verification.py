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

PROFILE-QUALIFIED IDENTITY (polylogue-y9zx, mirrors fs1.14 for
``hermes_spans.py``/PR #3224): the raw Hermes session id alone is not a safe
archive join key -- two separate Hermes installs (profiles) can legitimately
reuse the same raw session id, and this parser previously built an
unqualified ``verification:<raw_id>`` identity that silently collapsed two
installs' verification-ledger evidence onto one archive session whenever
that happened. ``observer_session_provider_id``/
``hermes_verification_session_id_for`` now thread an optional
``profile_key`` through ``hermes_identity.py`` (the same shared helper
``hermes_spans.py`` and ``hermes_state.py`` use) so the verification-ledger
session identity and the ``session_links`` parent edge asserted back to the
state-db conversational session use the exact same qualifier.
``parent_session_provider_id`` is only ever set when the profile root is
known -- an unqualified raw id is not asserted as a parent, the fail-closed
side of the same fix: no edge is safer than a wrong one. The
``verification:`` family prefix keeps this identity distinguishable from
ATIF/ATOF's ``observer:`` family even when both qualify the same raw id
against the same profile.
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
from .hermes_identity import profile_key as _profile_key
from .hermes_identity import qualified_session_id as _qualified_session_id
from .hermes_identity import split_qualified_session_id as _split_qualified_session_id
from .hermes_state import HermesFidelityCapability, HermesFidelityStatus, HermesImportFidelity

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


def observer_session_provider_id(hermes_session_id: str, profile_key: str | None = None) -> str:
    """Return the verification-evidence session identity for a Hermes session id.

    Distinct from ``hermes_spans.atif_session_provider_id`` /
    ``hermes_spans.atof_session_provider_id`` (ATIF/ATOF use their own
    artifact-qualified ``observer:atif:``/``observer:atof:`` prefixes, fs1.14):
    a physical merge across these independently-acquired evidence classes is
    deferred, same discipline as ATOF/ATIF's own deferred merge into the
    state-db conversational session. The ``verification:`` family prefix
    keeps this identity distinguishable from ATIF/ATOF's ``observer:``
    family even when both qualify the same raw session id against the same
    profile.

    ``profile_key`` -- when known -- is folded into the identity using the
    exact same qualifier scheme the state.db parser uses
    (``hermes_identity.qualified_session_id``), mirroring
    ``hermes_spans.atif_session_provider_id`` /
    ``hermes_spans.atof_session_provider_id`` (polylogue-y9zx / fs1.14).
    Two separate Hermes installs (profiles) can legitimately reuse the same
    raw session id, and an unqualified ``verification:<raw_id>`` identity
    silently collapsed their verification-ledger evidence onto one archive
    session. When the profile root is not known at parse time
    (``profile_key=None``), the identity stays unqualified rather than
    guessing -- the fail-closed side of that same fix.
    """
    if profile_key:
        return f"verification:{_qualified_session_id(hermes_session_id, profile_key)}"
    return f"verification:{hermes_session_id}"


def hermes_verification_session_id_for(conversational_session_id: str) -> str:
    """Map a state-db-ingested Hermes session id to its verification-ledger counterpart.

    Mirrors ``hermes_spans.hermes_atif_session_id_for`` /
    ``hermes_spans.hermes_atof_session_id_for``: preserves whatever
    ``@profile-<key>`` qualifier the conversational id carries (see
    ``observer_session_provider_id`` docstring for why this matters) so a
    reader holding the qualified conversational session id looks up the
    verification evidence for the *same install*, not merely the same raw
    session id. Falls back to an unqualified id only when the conversational
    id itself carries no profile qualifier.
    """
    raw_id, profile_key = _split_qualified_session_id(conversational_session_id)
    return observer_session_provider_id(raw_id, profile_key)


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


def parse_verification_evidence_db_payload(
    payload: JSONDocument,
    fallback_id: str,
    *,
    profile_root: Path | None = None,
) -> list[ParsedSession]:
    """Parse a ``verification_db_path`` marker payload.

    ``profile_root`` is the caller-supplied (dispatch-time) profile root --
    the same convention ``hermes_spans.parse_atif_document`` uses. When the
    caller does not know it, the marker payload's own ``profile_root`` field
    (set by :func:`marker_payload`, e.g. in tests/replay flows that build the
    payload directly) is used as a fallback so this stays usable without
    going through dispatch.
    """
    path_value = payload.get("verification_db_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Hermes verification_evidence.db marker is missing verification_db_path")
    if profile_root is None:
        profile_value = payload.get("profile_root")
        profile_root = Path(profile_value) if isinstance(profile_value, str) and profile_value else None
    return parse_verification_evidence_db(Path(path_value), fallback_id=fallback_id, profile_root=profile_root)


def parse_verification_evidence_db(
    path: Path,
    *,
    fallback_id: str | None = None,
    profile_root: Path | None = None,
) -> list[ParsedSession]:
    """Parse every verification event/state row from a Hermes verification_evidence.db file.

    ``profile_root`` -- the directory the raw verification_evidence.db was
    acquired from, mirroring ``hermes_state.parse_state_db``'s own
    convention -- is the producer-positive evidence used to profile-qualify
    both this file's own observer-evidence session identity and the
    ``session_links`` parent edge it asserts back to the matching state-db
    conversational session (polylogue-y9zx / fs1.14). When ``profile_root``
    is ``None`` (unknown at parse time), no parent edge is asserted: a wrong
    guess would silently correlate this evidence with the wrong install's
    conversational session, which is worse than leaving the correlation
    unresolved.
    """
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
    profile_key_value = _profile_key(profile_root) if profile_root is not None else None
    return [
        _verification_session(
            session_id,
            grouped_events.get(session_id, []),
            grouped_state.get(session_id, []),
            profile_key=profile_key_value,
        )
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
    *,
    profile_key: str | None,
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
    provider_session_id = observer_session_provider_id(session_id, profile_key)
    # The parent join key: asserting it makes the generic session_links write
    # path (archive_tiers/write.py::_write_session_link, unchanged by this
    # parser) resolve this verification-ledger session against the
    # profile-qualified state-db conversational session sharing the same raw
    # id -- resolvable once that session is ingested, visible as unresolved
    # debt otherwise. Only asserted when the profile is known (see
    # parse_verification_evidence_db docstring): an unqualified raw id is
    # not a safe cross-profile join key.
    parent_session_provider_id = _qualified_session_id(session_id, profile_key) if profile_key else None
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
                    "profile_qualified": profile_key is not None,
                    "asserted_parent_session_provider_id": parent_session_provider_id,
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
        parent_session_provider_id=parent_session_provider_id,
        ingest_flags=["hermes:verification-evidence"],
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _parent_session_link_capability(sessions: list[ParsedSession]) -> HermesFidelityCapability:
    """Declare fidelity for the ``session_links`` parent edge to the conversational session.

    Mirrors ``hermes_spans._parent_session_link_capability`` (polylogue-y9zx
    / fs1.14), aggregated across every returned session the same way the
    other capabilities in this function are: ``inferred`` (not ``exact``)
    because the archive has not yet confirmed the target conversational
    session was actually ingested -- that confirmation is the generic
    ``session_links`` resolution machinery's job, not this parser's.
    """
    total = max(len(sessions), 1)
    linked = sum(1 for session in sessions if session.parent_session_provider_id)
    if not sessions:
        return HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total,
            counts={},
            detail="No sessions were produced, so no session_links parent edge could be asserted.",
        )
    status: HermesFidelityStatus
    if linked == len(sessions):
        status = "inferred"
    elif linked:
        status = "degraded"
    else:
        status = "absent"
    return HermesFidelityCapability(
        status=status,
        observed=linked,
        expected=total,
        counts={},
        detail=(
            "A profile-qualified session_links parent edge was asserted to the state-db "
            "conversational session sharing this raw Hermes session id and profile root, for "
            "every session where the profile root was known at ingest time; resolution against "
            "an actually-ingested session happens generically at write time. Sessions parsed "
            "without a known profile root fail closed -- an unqualified raw session id is not a "
            "safe cross-profile join key."
        ),
    )


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
        "parent_session_link": _parent_session_link_capability(sessions),
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
    "hermes_verification_session_id_for",
    "import_fidelity_declaration",
    "looks_like_verification_evidence_db_path",
    "looks_like_verification_evidence_db_payload",
    "marker_payload",
    "observer_session_provider_id",
    "parse_verification_evidence_db",
    "parse_verification_evidence_db_payload",
]
