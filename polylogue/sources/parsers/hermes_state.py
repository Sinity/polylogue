"""Hermes ``state.db`` parser.

Hermes's durable session source is the SQLite database under
``~/.hermes/state.db``.  The older JSON document parser in
``local_agent.py`` remains useful for exported snapshots, but this parser owns
the authoritative live state shape.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, SourceFidelityStatus, TitleSource
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.provider_identity import profile_root_for_artifact
from polylogue.sources.detection_projection import DetectorProjection
from polylogue.sources.parsers.hermes_tool_outcome import JSON_ENVELOPE_PREFIX, tool_result_outcome
from polylogue.sources.sqlite_export import (
    LogicalExportError,
    logical_source_context,
    logical_source_shape,
    readable_table_info,
)

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from .hermes_finish_reason import end_turn_from_finish_reason as _end_turn_from_finish_reason
from .hermes_finish_reason import stop_reason_from_finish_reason as _stop_reason_from_finish_reason
from .hermes_identity import profile_key as _profile_key
from .hermes_identity import qualified_session_id as _qualified_session_id
from .local_agent import (
    _codex_output_text_blocks,
    _codex_reasoning_blocks,
    _content_blocks_from_content,
    _content_text,
    _tool_use_block,
)

HERMES_STATE_DB_MARKER = "hermes_state_db"
_COMPACTION_END_REASONS = frozenset({"compression", "compaction"})
_REQUIRED_SESSION_COLUMNS = frozenset(
    {
        "id",
        "started_at",
    }
)
_REQUIRED_MESSAGE_COLUMNS = frozenset(
    {
        "id",
        "session_id",
        "role",
        "content",
        "timestamp",
    }
)
_HERMES_SIGNATURE_SESSION_COLUMNS = frozenset({"source", "model_config", "parent_session_id"})
_HERMES_SIGNATURE_MESSAGE_COLUMNS = frozenset({"tool_calls", "observed", "active", "compacted"})

_SESSION_CAPABILITIES: dict[str, frozenset[str]] = {
    "model": frozenset({"model", "model_config", "system_prompt"}),
    "lineage": frozenset({"parent_session_id", "end_reason"}),
    "usage": frozenset(
        {
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "api_call_count",
        }
    ),
    "cost_provenance": frozenset(
        {
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
        }
    ),
    "repository": frozenset({"cwd", "git_branch", "git_repo_root"}),
    "lifecycle": frozenset({"ended_at", "end_reason", "rewind_count", "archived", "expiry_finalized"}),
    "handoff": frozenset({"handoff_state", "handoff_platform", "handoff_error"}),
    "source_identity": frozenset({"source", "user_id"}),
    # Schema v18+ (Hermes #9006 gateway metadata consolidation): multi-channel
    # chat-platform routing identity for sessions bridged through a gateway
    # (Slack/Discord/etc.), absent for plain-CLI sessions.
    "gateway_identity": frozenset({"session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin_json"}),
    # Schema v19: compression-retry cooldown/error state, distinct from the
    # end_reason="compression"/"compaction" continuation lineage already
    # captured under "lifecycle".
    "compression_recovery": frozenset({"compression_failure_cooldown_until", "compression_failure_error"}),
}
_MESSAGE_CAPABILITIES: dict[str, frozenset[str]] = {
    "tooling": frozenset({"tool_call_id", "tool_calls", "tool_name", "finish_reason"}),
    "reasoning": frozenset(
        {
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
        }
    ),
    "provider_identity": frozenset({"platform_message_id"}),
    "message_state": frozenset({"observed", "active", "compacted"}),
    "usage": frozenset({"token_count"}),
}

_COST_FIELDS = (
    "estimated_cost_usd",
    "actual_cost_usd",
    "cost_status",
    "cost_source",
    "pricing_version",
    "billing_provider",
    "billing_base_url",
    "billing_mode",
)
_SESSION_METADATA_FIELDS = (
    "source",
    "user_id",
    "handoff_state",
    "handoff_platform",
    "handoff_error",
    "archived",
    "expiry_finalized",
    "session_key",
    "chat_id",
    "chat_type",
    "thread_id",
    "display_name",
    "origin_json",
    "compression_failure_cooldown_until",
    "compression_failure_error",
)


@dataclass(frozen=True, slots=True)
class HermesFidelityCapability:
    """One source capability and the evidence that supports its status."""

    status: SourceFidelityStatus
    observed: int
    expected: int
    counts: dict[str, int]
    detail: str


@dataclass(frozen=True, slots=True)
class HermesImportFidelity:
    """Machine-readable fidelity declaration for one Hermes source artifact."""

    producer: str
    schema_version: int | None
    profile_namespace: str | None
    acquisition_method: str
    retained_blob_reproducibility: HermesFidelityCapability
    capabilities: dict[str, HermesFidelityCapability]
    caveats: tuple[str, ...]


def marker_payload(
    path: Path,
    *,
    profile_root: Path | None = None,
    immutable: bool = False,
) -> JSONDocument:
    """Return the JSON marker that routes a raw SQLite blob to this parser."""
    payload: JSONDocument = {
        "polylogue_artifact": HERMES_STATE_DB_MARKER,
        "state_db_path": str(path),
    }
    if profile_root is not None:
        payload["profile_root"] = str(profile_root)
    if immutable:
        payload["sqlite_immutable"] = True
    return payload


def looks_like_state_db_payload(payload: JSONDocument) -> bool:
    return payload.get("polylogue_artifact") == HERMES_STATE_DB_MARKER and isinstance(payload.get("state_db_path"), str)


def looks_like_state_db_path(path: Path, *, immutable: bool = False) -> bool:
    """Return true when *path* is a readable Hermes state database or its export.

    Answered from the table/column shape alone, so probing a retained export
    never costs a reconstruction of its rows.
    """
    try:
        return _shape_has_required_tables(logical_source_shape(path, immutable=immutable))
    except (sqlite3.Error, OSError, LogicalExportError, ValueError):
        return False


def require_declared_export(path: Path, source_path: str | None, *, field: str) -> None:
    """Refuse a marker whose named database is not its own declared export.

    A typed refusal, not a silent skip: an artifact that carries the marker and
    then declines to be what it claims is a parse failure the ingest boundary
    records, which is the ordinary outcome for malformed evidence.
    """
    from polylogue.sources.sqlite_snapshot import is_declared_logical_export

    if source_path is None or not is_declared_logical_export(path, source_path):
        raise ValueError(
            f"Hermes marker {field} does not name the declared logical export for its own source; refusing to open it"
        )


def parse_state_db_payload(
    payload: JSONDocument,
    fallback_id: str,
    *,
    source_path: str | None = None,
    profile_identity: str | None = None,
) -> list[ParsedSession]:
    """Parse a ``state_db_path`` marker payload from its own declared export.

    ``state_db_path`` names a local database to open, and the marker itself is
    an ordinary JSON object that the Hermes detector recognises by shape. Any
    imported document could therefore carry the marker and steer this read at a
    database the operator never meant to ingest -- a confused deputy that files
    another database's content into the archive under a false source identity.

    Minting a marker already requires the path to be the declared logical
    export for its source (``raw_payload.decode``); this is the same check on
    the consuming side, so the two routes agree about which bytes a marker may
    name.
    """
    path_value = payload.get("state_db_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Hermes state.db marker is missing state_db_path")
    require_declared_export(Path(path_value), source_path, field="state_db_path")
    profile_value = payload.get("profile_root")
    profile_root = Path(profile_value) if isinstance(profile_value, str) and profile_value else None
    return parse_state_db(
        Path(path_value),
        fallback_id=fallback_id,
        profile_root=profile_root,
        profile_identity=profile_identity,
        immutable=payload.get("sqlite_immutable") is True,
    )


def parse_state_db(
    path: Path,
    *,
    fallback_id: str | None = None,
    profile_root: Path | None = None,
    immutable: bool = False,
    profile_identity: str | None = None,
) -> list[ParsedSession]:
    """Parse every session revision from a Hermes ``state.db`` file."""
    del fallback_id
    with _readonly_context(path, immutable=immutable) as conn:
        return _parse_state_connection(conn, path, profile_root=profile_root, profile_identity=profile_identity)


def _parse_state_connection(
    conn: sqlite3.Connection, path: Path, *, profile_root: Path | None = None, profile_identity: str | None = None
) -> list[ParsedSession]:
    """Parse on the caller-owned read transaction without reopening its source."""
    return list(
        iter_state_db_sessions(
            conn,
            path,
            message_sink_factory=list,
            event_sink_factory=list,
            profile_root=profile_root,
            profile_identity=profile_identity,
        )
    )


def iter_state_db_sessions(
    conn: sqlite3.Connection,
    path: Path,
    *,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]],
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]],
    profile_root: Path | None = None,
    profile_identity: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> Iterator[ParsedSession]:
    """Yield settled Hermes sessions while the caller-owned source stays open.

    The first pass records only ancestry facts (session identity, own message
    count, and own leaf) in private SQLite scratch. The second pass parses one
    session into caller supplied sinks, applying its already-resolved offset
    and branch point before yielding it. No transcript or session cohort is
    retained by this iterator. The caller owns the source transaction and the
    sink lifetime; closing this generator also closes and removes its scratch.
    """
    conn.row_factory = sqlite3.Row
    if not _has_required_tables(conn):
        raise ValueError(f"{path} is not a Hermes state.db file")
    session_columns = _columns(conn, "sessions")
    message_columns = _columns(conn, "messages")
    schema_version = _schema_version(conn)
    resolved_profile_root = profile_root or profile_root_for_artifact(path)
    profile_key = profile_identity if profile_identity is not None else _profile_key(resolved_profile_root)
    query = "SELECT * FROM sessions ORDER BY COALESCE(started_at, 0), id"
    check = check_cancelled or (lambda: None)

    with tempfile.TemporaryDirectory(prefix="polylogue-hermes-stream-") as scratch_dir:
        grouping = sqlite3.connect(Path(scratch_dir) / "ancestry.sqlite3")
        try:
            grouping.execute("PRAGMA journal_mode = DELETE")
            grouping.execute("PRAGMA temp_store = FILE")
            state = _DiskCompressionState(grouping)
            grouping.execute(
                "CREATE TABLE hermes_parent_evidence (session_id TEXT COLLATE BINARY PRIMARY KEY, end_reason TEXT) WITHOUT ROWID"
            )
            with closing(conn.execute(query)) as cursor:
                for row in cursor:
                    check()
                    grouping.execute(
                        "INSERT INTO hermes_parent_evidence VALUES (?, ?) "
                        "ON CONFLICT(session_id) DO UPDATE SET end_reason = excluded.end_reason",
                        (str(row["id"]), _optional_text(_row_value(row, "end_reason"))),
                    )

            grouping.execute(
                "CREATE TABLE hermes_stream_selected (session_id TEXT COLLATE BINARY PRIMARY KEY, "
                "ordinal INTEGER NOT NULL, own_count INTEGER NOT NULL, own_leaf TEXT, kept INTEGER NOT NULL, "
                "original_parent TEXT, inherited INTEGER NOT NULL, branch_point TEXT, composed_count INTEGER NOT NULL, "
                "composed_leaf TEXT, final_parent TEXT) WITHOUT ROWID"
            )
            with closing(conn.execute(query)) as cursor:
                for ordinal, row in enumerate(cursor):
                    check()
                    raw_id = str(row["id"])
                    raw_parent = _row_value(row, "parent_session_id")
                    parent_raw_id = _optional_text(raw_parent)
                    parent = (
                        grouping.execute(
                            "SELECT end_reason FROM hermes_parent_evidence WHERE session_id = ?", (str(raw_parent),)
                        ).fetchone()
                        if raw_parent is not None
                        else None
                    )
                    branch_type = _branch_type(row, {"end_reason": parent[0]} if parent is not None else None)
                    session_id = _qualified_session_id(raw_id, profile_key)
                    parent_id = _qualified_session_id(parent_raw_id, profile_key) if parent_raw_id else None
                    prompt = _optional_text(_row_value(row, "system_prompt"))
                    own_count = int(bool(prompt))
                    own_leaf = f"{session_id}:system" if prompt else None
                    leaf_columns = ["id", "session_id"]
                    if "platform_message_id" in message_columns:
                        leaf_columns.append("platform_message_id")
                    with closing(
                        conn.execute(
                            f"SELECT {', '.join(leaf_columns)} FROM messages WHERE session_id = ? ORDER BY id",
                            (raw_id,),
                        )
                    ) as message_cursor:
                        for message_row in message_cursor:
                            check()
                            own_count += 1
                            own_leaf = _message_provider_id(message_row)
                    facts = {
                        "kept": bool(own_count),
                        "own_count": own_count,
                        "own_leaf": own_leaf,
                        "original_parent": parent_id,
                    }
                    state.add(
                        _CompressionIdentity(
                            ordinal,
                            session_id,
                            parent_id,
                            branch_type is BranchType.CONTINUATION,
                        ),
                        json.dumps(facts, ensure_ascii=True),
                    )

            for node, compression_parent in _compression_resolution_order(state.roots(), state):
                check()
                facts = json.loads(_compression_facts_at(grouping, node.ordinal))
                if compression_parent is None:
                    inherited = 0
                    branch_point = None
                    parent_leaf = None
                else:
                    parent_row = grouping.execute(
                        "SELECT own_count, composed_count, composed_leaf FROM hermes_stream_selected "
                        "WHERE session_id = ?",
                        (compression_parent,),
                    ).fetchone()
                    if parent_row is None:
                        inherited = 0
                        branch_point = None
                        parent_leaf = None
                    else:
                        inherited = int(parent_row[1])
                        branch_point = str(parent_row[2]) if parent_row[2] is not None else None
                        parent_leaf = branch_point
                own_count = int(facts["own_count"])
                composed_count = inherited + own_count
                composed_leaf = facts["own_leaf"] if own_count else parent_leaf
                grouping.execute(
                    "INSERT INTO hermes_stream_selected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET ordinal=excluded.ordinal, own_count=excluded.own_count, "
                    "own_leaf=excluded.own_leaf, kept=excluded.kept, original_parent=excluded.original_parent, "
                    "inherited=excluded.inherited, branch_point=excluded.branch_point, "
                    "composed_count=excluded.composed_count, composed_leaf=excluded.composed_leaf",
                    (
                        node.session_id,
                        node.ordinal,
                        own_count,
                        facts["own_leaf"],
                        int(bool(facts["kept"])),
                        facts["original_parent"],
                        inherited,
                        branch_point,
                        composed_count,
                        composed_leaf,
                        None,
                    ),
                )

            # Match _without_unreferenced_empty_sessions: follow the source
            # parent edge through empty sessions to the nearest kept ancestor.
            for node in state.roots():
                check()
                selected = grouping.execute(
                    "SELECT kept, original_parent FROM hermes_stream_selected WHERE session_id = ?",
                    (node.session_id,),
                ).fetchone()
                if selected is None:
                    continue
                final_parent = selected[1]
                seen: set[str] = set()
                while final_parent is not None:
                    if final_parent in seen:
                        final_parent = None
                        break
                    seen.add(final_parent)
                    ancestor = grouping.execute(
                        "SELECT kept, original_parent FROM hermes_stream_selected WHERE session_id = ?",
                        (final_parent,),
                    ).fetchone()
                    if ancestor is None or bool(ancestor[0]):
                        break
                    final_parent = str(ancestor[1]) if ancestor[1] is not None else None
                grouping.execute(
                    "UPDATE hermes_stream_selected SET final_parent = ? WHERE session_id = ?",
                    (final_parent, node.session_id),
                )

            with closing(conn.execute(query)) as cursor:
                for ordinal, original_row in enumerate(cursor):
                    check()
                    row = original_row
                    raw_id = str(row["id"])
                    session_id = _qualified_session_id(raw_id, profile_key)
                    settled = grouping.execute(
                        "SELECT ordinal, own_count, kept, inherited, branch_point, final_parent "
                        "FROM hermes_stream_selected WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if settled is None or not bool(settled[2]):
                        continue
                    selected_ordinal = int(settled[0])
                    if selected_ordinal != ordinal:
                        row = conn.execute(query + " LIMIT 1 OFFSET ?", (selected_ordinal,)).fetchone()
                        if row is None:
                            raise ValueError("Hermes selected duplicate session disappeared during parsing")
                        raw_id = str(row["id"])
                    parent_raw = _row_value(row, "parent_session_id")
                    parent = (
                        grouping.execute(
                            "SELECT end_reason FROM hermes_parent_evidence WHERE session_id = ?", (str(parent_raw),)
                        ).fetchone()
                        if parent_raw is not None
                        else None
                    )
                    message_sink = message_sink_factory()
                    event_sink = event_sink_factory()
                    session = _parse_session_row(
                        conn,
                        row,
                        parent_row={"end_reason": parent[0]} if parent is not None else None,
                        profile_root=resolved_profile_root,
                        profile_identity=profile_identity,
                        schema_version=schema_version,
                        session_columns=session_columns,
                        message_columns=message_columns,
                        message_sink=message_sink,
                        event_sink=event_sink,
                        position_offset=int(settled[3]),
                        check_cancelled=check_cancelled,
                    )
                    updates: dict[str, object] = {"parent_session_provider_id": settled[5]}
                    if settled[4] is not None:
                        updates["branch_point_provider_message_id"] = settled[4]
                    yield session.model_copy(update=updates)
        finally:
            grouping.close()


def _compression_facts_at(grouping: sqlite3.Connection, ordinal: int) -> str:
    row = grouping.execute("SELECT facts FROM hermes_compression_nodes WHERE ordinal = ?", (ordinal,)).fetchone()
    if row is None:
        raise ValueError("Hermes compression node disappeared during parsing")
    return str(row[0])


def _without_unreferenced_empty_sessions(sessions: list[ParsedSession]) -> list[ParsedSession]:
    """Drop content-less sessions, re-parenting their children onto a kept ancestor.

    A compression continuation that contributed no messages carries no row
    the archive admits (positive-evidence admission and the writer both skip
    empty sessions), yet a later child names it as
    ``parent_session_provider_id``. Pointing that child at the nearest
    ancestor that is kept keeps its lineage edge resolvable. The child's
    ``branch_point_provider_message_id`` already names the composed leaf,
    which for an empty parent is its own ancestor's last message.
    """
    by_id = {session.provider_session_id: session for session in sessions}
    kept_ids = {session.provider_session_id for session in sessions if session.messages or session.instructions_text}

    def kept_ancestor(parent_id: str | None) -> str | None:
        seen: set[str] = set()
        while parent_id is not None and parent_id in by_id and parent_id not in kept_ids and parent_id not in seen:
            seen.add(parent_id)
            parent_id = by_id[parent_id].parent_session_provider_id
        return parent_id if parent_id is None or parent_id in kept_ids or parent_id not in by_id else None

    kept: list[ParsedSession] = []
    for session in sessions:
        if session.provider_session_id not in kept_ids:
            continue
        parent_id = session.parent_session_provider_id
        if parent_id is not None and parent_id in by_id and parent_id not in kept_ids:
            session = session.model_copy(update={"parent_session_provider_id": kept_ancestor(parent_id)})
        kept.append(session)
    return kept


def import_fidelity_declaration(
    sessions: list[ParsedSession],
    *,
    acquisition_method: Literal["logical_export", "json_fallback"],
) -> HermesImportFidelity:
    """Declare the source fidelity that the Hermes parser can substantiate.

    The declaration is deliberately conservative: values derived by the
    normalizer are marked inferred, and evidence that no source artifact
    carries is absent rather than represented as an empty successful value.
    """

    if acquisition_method == "json_fallback":
        return _json_fallback_fidelity(sessions)
    return _logical_export_fidelity(sessions)


@dataclass(slots=True)
class _LogicalFidelityEvidence:
    """Count the same parser evidence without retaining transcript objects."""

    sessions: int = 0
    messages: int = 0
    schema_versions: set[int] = field(default_factory=set)
    profile_keys: set[str] = field(default_factory=set)
    source_capabilities: dict[str, int] = field(default_factory=dict)
    state_counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(("active", "observed", "rewound", "compacted"), 0)
    )
    state_capable: int = 0
    material_counts: dict[str, int] = field(default_factory=dict)
    cost_events: int = 0

    def message(self, message: ParsedMessage) -> None:
        self.messages += 1
        key = message.material_origin.value
        self.material_counts[key] = self.material_counts.get(key, 0) + 1

    def event(self, event: ParsedSessionEvent) -> None:
        payload = event.payload
        if event.event_type == "hermes_identity":
            version = payload.get("schema_version")
            if isinstance(version, int):
                self.schema_versions.add(version)
            profile = payload.get("profile_key")
            if isinstance(profile, str) and profile:
                self.profile_keys.add(profile)
            capabilities = payload.get("session_capabilities")
            if isinstance(capabilities, list):
                for name in ("lifecycle", "lineage", "repository", "gateway_identity", "compression_recovery"):
                    self.source_capabilities[name] = self.source_capabilities.get(name, 0) + int(name in capabilities)
        elif event.event_type == "hermes_message_state":
            state = payload.get("state")
            if isinstance(state, str) and state in self.state_counts:
                self.state_counts[state] += 1
            self.state_capable += bool(payload.get("capability_present"))
        elif event.event_type == "token_count" and any(payload.get(name) is not None for name in _COST_FIELDS):
            self.cost_events += 1

    def merge(self, other: _LogicalFidelityEvidence) -> None:
        self.sessions += other.sessions
        self.messages += other.messages
        self.schema_versions.update(other.schema_versions)
        self.profile_keys.update(other.profile_keys)
        self.state_capable += other.state_capable
        self.cost_events += other.cost_events
        for target, counts in (
            (self.source_capabilities, other.source_capabilities),
            (self.state_counts, other.state_counts),
            (self.material_counts, other.material_counts),
        ):
            for name, value in counts.items():
                target[name] = target.get(name, 0) + value

    def counts(self) -> dict[str, object]:
        return {
            "sessions": self.sessions,
            "messages": self.messages,
            "schema_versions": sorted(self.schema_versions),
            "profile_keys": sorted(self.profile_keys),
            "state_capable": self.state_capable,
            "cost_events": self.cost_events,
            "source_capabilities": self.source_capabilities,
            "state_counts": self.state_counts,
            "material_counts": self.material_counts,
        }

    @classmethod
    def from_counts(cls, values: Mapping[str, object]) -> _LogicalFidelityEvidence:
        return cls(
            sessions=cast(int, values["sessions"]),
            messages=cast(int, values["messages"]),
            schema_versions=set(cast(list[int], values["schema_versions"])),
            profile_keys=set(cast(list[str], values["profile_keys"])),
            state_capable=cast(int, values["state_capable"]),
            cost_events=cast(int, values["cost_events"]),
            source_capabilities=cast(dict[str, int], values["source_capabilities"]),
            state_counts=cast(dict[str, int], values["state_counts"]),
            material_counts=cast(dict[str, int], values["material_counts"]),
        )


def _logical_export_fidelity(sessions: list[ParsedSession]) -> HermesImportFidelity:
    evidence = _LogicalFidelityEvidence(sessions=len(sessions))
    for session in sessions:
        for message in session.messages:
            evidence.message(message)
        for event in session.session_events:
            evidence.event(event)
    return _logical_export_fidelity_from_evidence(evidence)


def _logical_export_fidelity_from_evidence(evidence: _LogicalFidelityEvidence) -> HermesImportFidelity:
    total_sessions = evidence.sessions
    schema_versions = evidence.schema_versions
    profile_keys = evidence.profile_keys

    def source_capability(name: str) -> HermesFidelityCapability:
        return _fidelity_capability(
            observed=evidence.source_capabilities.get(name, 0),
            expected=total_sessions,
            detail=f"{name.replace('_', ' ')} columns are present in the inspected Hermes schema.",
        )

    state_counts = evidence.state_counts
    state_capable = evidence.state_capable
    material_counts: dict[str, int] = {
        origin.value: evidence.material_counts[origin.value]
        for origin in MaterialOrigin
        if origin.value in evidence.material_counts
    }
    cost_status = _fidelity_capability(
        observed=evidence.cost_events,
        expected=total_sessions,
        detail="Cost rows retain actual/estimated values with source, status, pricing, and billing provenance when present.",
    )
    capabilities = {
        "profile_namespace": _fidelity_capability(
            observed=len(profile_keys),
            expected=1,
            detail="Profile roots are hashed into stable namespace qualifiers; raw profile paths are not exposed.",
        ),
        "message_state": _fidelity_capability(
            observed=state_capable,
            expected=evidence.messages,
            counts=state_counts,
            detail="Active, observed, rewound, and compacted states are retained per source message.",
        ),
        "material_origin": HermesFidelityCapability(
            status="inferred",
            observed=evidence.messages,
            expected=evidence.messages,
            counts=material_counts,
            detail="Material origin is normalized from role plus the source observed flag; Hermes has no explicit addressing field.",
        ),
        "cost_provenance": cost_status,
        "lifecycle": source_capability("lifecycle"),
        "relationship": source_capability("lineage"),
        "repository": source_capability("repository"),
        "gateway_identity": source_capability("gateway_identity"),
        "compression_recovery": source_capability("compression_recovery"),
        "runtime_spans": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="The SQLite snapshot contains no runtime-span stream.",
        ),
        "span_snapshot_merge": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="No runtime spans were supplied to enrich this snapshot revision.",
        ),
    }
    caveats = tuple(
        f"{name}: {capability.detail}" for name, capability in capabilities.items() if capability.status != "exact"
    )
    return HermesImportFidelity(
        producer="Hermes state.db",
        schema_version=next(iter(schema_versions)) if len(schema_versions) == 1 else None,
        profile_namespace=next(iter(profile_keys)) if len(profile_keys) == 1 else None,
        acquisition_method="logical_export",
        retained_blob_reproducibility=HermesFidelityCapability(
            status="exact",
            observed=total_sessions,
            expected=total_sessions,
            counts={},
            detail="The import path snapshots SQLite bytes before parsing; retained bytes reproduce normalized Hermes revisions.",
        ),
        capabilities=capabilities,
        caveats=caveats,
    )


def _json_fallback_fidelity(sessions: list[ParsedSession]) -> HermesImportFidelity:
    return json_fallback_fidelity_counts(
        sessions=len(sessions), messages=sum(len(session.messages) for session in sessions)
    )


def json_fallback_fidelity_counts(*, sessions: int, messages: int) -> HermesImportFidelity:
    """Declare the same fallback fidelity from complete streamed counts."""
    total_sessions = sessions
    capabilities = {
        "profile_namespace": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=1,
            counts={},
            detail="JSON fallback has no installation/profile namespace.",
        ),
        "message_state": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=messages,
            counts={},
            detail="JSON fallback does not retain Hermes message-state columns.",
        ),
        "material_origin": HermesFidelityCapability(
            status="inferred",
            observed=messages,
            expected=messages,
            counts={},
            detail="Material origin is inferred from normalized message roles in the fallback export.",
        ),
        "cost_provenance": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="JSON fallback has no structured cost provenance.",
        ),
        "lifecycle": HermesFidelityCapability(
            status="inferred",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="Fallback lifecycle interpretation is limited to document fields.",
        ),
        "relationship": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="JSON fallback has no state-db lineage evidence.",
        ),
        "repository": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="JSON fallback has no repository capability declaration.",
        ),
        "runtime_spans": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="JSON fallback has no runtime-span stream.",
        ),
        "span_snapshot_merge": HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="Fallback data cannot prove a span-plus-snapshot merge.",
        ),
    }
    return HermesImportFidelity(
        producer="Hermes JSON fallback",
        schema_version=None,
        profile_namespace=None,
        acquisition_method="json_fallback",
        retained_blob_reproducibility=HermesFidelityCapability(
            status="absent",
            observed=0,
            expected=total_sessions,
            counts={},
            detail="Fallback JSON does not prove retained SQLite snapshot reproducibility.",
        ),
        capabilities=capabilities,
        caveats=tuple(
            f"{name}: {capability.detail}" for name, capability in capabilities.items() if capability.status != "exact"
        ),
    )


def _fidelity_capability(
    *,
    observed: int,
    expected: int,
    detail: str,
    counts: dict[str, int] | None = None,
) -> HermesFidelityCapability:
    status: SourceFidelityStatus
    if observed == 0:
        status = "absent"
    elif observed < expected:
        status = "degraded"
    else:
        status = "exact"
    return HermesFidelityCapability(
        status=status,
        observed=observed,
        expected=expected,
        counts={} if counts is None else counts,
        detail=detail,
    )


#: The one access pattern this parser issues per session: ``messages`` filtered
#: by ``session_id`` and ordered by ``id``. A retained export reconstructs with
#: no index at all, so without this hint every session re-scans and re-sorts the
#: whole reconstructed table. Honoured only for the private reconstruction; a
#: live Hermes database is never indexed on the archive's behalf.
_MESSAGE_READ_INDEXES: tuple[tuple[str, tuple[str, ...]], ...] = (("messages", ("session_id", "id")),)


@contextmanager
def _readonly_context(path: Path, *, immutable: bool = False) -> Iterator[sqlite3.Connection]:
    """Keep the logical-source reader and reconstruction on its creator."""
    with logical_source_context(path, immutable=immutable, read_indexes=_MESSAGE_READ_INDEXES) as connection:
        connection.row_factory = sqlite3.Row
        yield connection


def _has_required_tables(conn: sqlite3.Connection) -> bool:
    return _shape_has_required_tables(_connection_shape(conn))


def _connection_shape(conn: sqlite3.Connection) -> Mapping[str, tuple[str, ...]]:
    tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    return {table: tuple(sorted(_columns(conn, table))) for table in tables}


def _shape_has_required_tables(shape: Mapping[str, tuple[str, ...]]) -> bool:
    if not {"schema_version", "sessions", "messages"} <= set(shape):
        return False
    session_columns = set(shape["sessions"])
    message_columns = set(shape["messages"])
    return (
        _REQUIRED_SESSION_COLUMNS.issubset(session_columns)
        and _REQUIRED_MESSAGE_COLUMNS.issubset(message_columns)
        and _HERMES_SIGNATURE_SESSION_COLUMNS.issubset(session_columns)
        and _HERMES_SIGNATURE_MESSAGE_COLUMNS.issubset(message_columns)
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in readable_table_info(conn, table)}


def _schema_version(conn: sqlite3.Connection) -> int | None:
    if "schema_version" not in {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }:
        return None
    row = conn.execute("SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1").fetchone()
    return _non_negative_int(row[0]) if row else None


def _capabilities(columns: set[str], capability_map: Mapping[str, frozenset[str]]) -> list[str]:
    return sorted(name for name, fields in capability_map.items() if fields.issubset(columns))


def _identity_event(
    *,
    raw_session_id: str,
    profile_key: str,
    schema_version: int | None,
    session_columns: set[str],
    message_columns: set[str],
) -> ParsedSessionEvent:
    return ParsedSessionEvent(
        event_type="hermes_identity",
        payload={
            "raw_session_id": raw_session_id,
            "profile_key": profile_key,
            "schema_version": schema_version,
            "session_capabilities": _capabilities(session_columns, _SESSION_CAPABILITIES),
            "message_capabilities": _capabilities(message_columns, _MESSAGE_CAPABILITIES),
        },
    )


def _session_metadata_events(
    row: sqlite3.Row,
    *,
    session_columns: set[str],
) -> list[ParsedSessionEvent]:
    payload = {field: _row_value(row, field) for field in _SESSION_METADATA_FIELDS if field in session_columns}
    if not payload:
        return []
    return [
        ParsedSessionEvent(
            event_type="hermes_session_metadata",
            timestamp=_epoch_iso(_row_value(row, "ended_at")) or _epoch_iso(_row_value(row, "started_at")),
            payload=payload,
        )
    ]


def _message_state_event(
    row: sqlite3.Row,
    message: ParsedMessage,
    *,
    message_columns: set[str],
) -> ParsedSessionEvent:
    active = _sqlite_bool(_row_value(row, "active"), default=True)
    observed = _sqlite_bool(_row_value(row, "observed"), default=False)
    compacted = _sqlite_bool(_row_value(row, "compacted"), default=False)
    if compacted:
        state = "compacted"
    elif not active:
        state = "rewound"
    elif observed:
        state = "observed"
    else:
        state = "active"
    return ParsedSessionEvent(
        event_type="hermes_message_state",
        timestamp=message.timestamp,
        source_message_provider_id=message.provider_message_id,
        payload={
            "state": state,
            "active": active,
            "observed": observed,
            "compacted": compacted,
            "capability_present": "message_state" in _capabilities(message_columns, _MESSAGE_CAPABILITIES),
        },
    )


def _material_origin(role: Role, *, observed: bool) -> MaterialOrigin:
    if observed:
        return MaterialOrigin.RUNTIME_CONTEXT
    if role is Role.USER:
        return MaterialOrigin.HUMAN_AUTHORED
    if role is Role.ASSISTANT:
        return MaterialOrigin.ASSISTANT_AUTHORED
    if role is Role.TOOL:
        return MaterialOrigin.TOOL_RESULT
    if role is Role.SYSTEM:
        return MaterialOrigin.RUNTIME_PROTOCOL
    return MaterialOrigin.UNKNOWN


def _row_value(row: sqlite3.Row | Mapping[str, object], key: str) -> object | None:
    try:
        return cast(object, row[key])
    except (IndexError, KeyError):
        return None


def _sqlite_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return default


@dataclass(slots=True)
class _StateSessionInspection:
    evidence: _LogicalFidelityEvidence = field(default_factory=_LogicalFidelityEvidence)
    blocks: int = 0
    actions: int = 0
    latest_timestamp: str | None = None
    active_leaf_id: str | None = None

    def message(self, message: ParsedMessage) -> None:
        self.evidence.message(message)
        self.blocks += len(message.blocks)
        self.actions += sum(block.type is BlockType.TOOL_USE for block in message.blocks)
        if message.timestamp:
            self.latest_timestamp = message.timestamp
        if message.is_active_path:
            self.active_leaf_id = message.provider_message_id


def _parse_session_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    parent_row: sqlite3.Row | Mapping[str, object] | None,
    profile_root: Path,
    profile_identity: str | None,
    schema_version: int | None,
    session_columns: set[str],
    message_columns: set[str],
    inspection: _StateSessionInspection | None = None,
    message_sink: MutableSequence[ParsedMessage] | None = None,
    event_sink: MutableSequence[ParsedSessionEvent] | None = None,
    position_offset: int = 0,
    check_cancelled: Callable[[], None] | None = None,
) -> ParsedSession:
    raw_session_id = str(row["id"])
    profile_key = profile_identity if profile_identity is not None else _profile_key(profile_root)
    session_id = _qualified_session_id(raw_session_id, profile_key)
    materialized_messages: list[ParsedMessage] = []
    messages: MutableSequence[ParsedMessage] = message_sink if message_sink is not None else materialized_messages
    state_events: list[ParsedSessionEvent] = []
    materialized_events: list[ParsedSessionEvent] = []
    sink_active_leaf_position: int | None = None

    def append_message(message: ParsedMessage) -> None:
        nonlocal sink_active_leaf_position
        if inspection is None:
            if message_sink is not None and message.is_active_leaf is not False:
                message = message.model_copy(update={"is_active_leaf": False})
            messages.append(message)
            if message_sink is not None and message.is_active_path:
                sink_active_leaf_position = len(messages) - 1
        else:
            inspection.message(message)

    def append_event(event: ParsedSessionEvent) -> None:
        if inspection is None:
            if event_sink is None:
                state_events.append(event)
            else:
                event_sink.append(event)
        else:
            inspection.evidence.event(event)

    system_prompt = _optional_text(_row_value(row, "system_prompt"))
    model_name = _optional_text(_row_value(row, "model"))
    if system_prompt:
        position = position_offset + (0 if inspection is None else inspection.evidence.messages)
        append_message(
            ParsedMessage(
                provider_message_id=f"{session_id}:system",
                role=Role.SYSTEM,
                text=system_prompt,
                timestamp=_epoch_iso(row["started_at"]),
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=system_prompt)],
                position=position,
                variant_index=0,
                is_active_path=True,
                model_name=model_name,
            )
        )
    for message_row in conn.execute(
        """
            SELECT *
            FROM messages
            WHERE session_id = ?
            ORDER BY id
            """,
        (raw_session_id,),
    ):
        if check_cancelled is not None:
            check_cancelled()
        position = position_offset + len(messages) if inspection is None else inspection.evidence.messages
        parsed = _parse_message_row(message_row, position=position, fallback_model=model_name)
        append_message(parsed)
        append_event(_message_state_event(message_row, parsed, message_columns=message_columns))
        for event in _reasoning_evidence_events(message_row, parsed):
            append_event(event)
    if inspection is None and message_sink is None:
        materialized_messages = _mark_active_leaf(materialized_messages)
        messages = materialized_messages
    elif inspection is None and sink_active_leaf_position is not None:
        leaf = messages[sink_active_leaf_position]
        if not leaf.is_active_leaf:
            messages[sink_active_leaf_position] = leaf.model_copy(update={"is_active_leaf": True})
    parent_raw_id = _optional_text(_row_value(row, "parent_session_id"))
    parent_id = _qualified_session_id(parent_raw_id, profile_key) if parent_raw_id else None
    prefix_events = [
        _identity_event(
            raw_session_id=raw_session_id,
            profile_key=profile_key,
            schema_version=schema_version,
            session_columns=session_columns,
            message_columns=message_columns,
        ),
        *_usage_and_lifecycle_events(row, messages, session_columns=session_columns),
        *_session_metadata_events(row, session_columns=session_columns),
    ]
    if message_sink is not None:
        active_leaf_id = (
            messages[sink_active_leaf_position].provider_message_id if sink_active_leaf_position is not None else None
        )
    else:
        active_leaf_id = next(
            (message.provider_message_id for message in reversed(messages) if message.is_active_path),
            None,
        )
    if inspection is not None:
        for event in prefix_events:
            inspection.evidence.event(event)
        for event in state_events:
            inspection.evidence.event(event)
        session_events: MutableSequence[ParsedSessionEvent] = materialized_events
        active_leaf_id = inspection.active_leaf_id
    elif event_sink is not None:
        # Message-state and reasoning events were appended as the message
        # cursor advanced. Prefix events precede them in the public parser's
        # established event order, so insert the small session-level prefix
        # in reverse at index zero. Disk-backed sinks implement this without
        # materializing the event cohort.
        for event in reversed(prefix_events):
            event_sink.insert(0, event)
        session_events = event_sink
    else:
        materialized_events = [*prefix_events, *state_events]
        session_events = materialized_events
    provider_title = _optional_text(_row_value(row, "title"))
    session = ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=session_id,
        title=provider_title or raw_session_id,
        title_source=TitleSource.ORIGIN if provider_title else None,
        created_at=_epoch_iso(row["started_at"]),
        updated_at=_epoch_iso(_row_value(row, "ended_at"))
        or (_latest_message_timestamp(messages) if inspection is None else inspection.latest_timestamp),
        messages=materialized_messages,
        active_leaf_message_provider_id=active_leaf_id,
        session_events=materialized_events,
        parent_session_provider_id=parent_id,
        branch_type=_branch_type(row, parent_row) if parent_id else None,
        instructions_text=system_prompt,
        reported_cost_usd=_reported_cost(row),
        models_used=[model_name] if model_name else [],
        working_directories=[cwd] if (cwd := _optional_text(_row_value(row, "cwd"))) else [],
        git_branch=_optional_text(_row_value(row, "git_branch")),
        git_repository_url=_optional_text(_row_value(row, "git_repo_root")),
        ingest_flags=["hermes:state-db", f"hermes:schema-v{schema_version or 'unknown'}"],
    )
    if message_sink is not None or event_sink is not None:
        return session.model_copy(update={"messages": messages, "session_events": session_events})
    return session


def _parse_message_row(
    row: sqlite3.Row,
    *,
    position: int,
    fallback_model: str | None,
) -> ParsedMessage:
    content = _decode_content(row["content"])
    text = _content_text(content)
    blocks = _content_blocks_from_content(content)
    output_text_blocks = _codex_output_text_blocks(_row_value(row, "codex_message_items"), covered_text=text)
    blocks.extend(output_text_blocks)
    if text is None and output_text_blocks:
        text = "\n".join(block.text for block in output_text_blocks if block.text)
    reasoning = _optional_text(_row_value(row, "reasoning_content")) or _optional_text(_row_value(row, "reasoning"))
    if reasoning:
        metadata = _reasoning_metadata(row)
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=reasoning, metadata=metadata or None))
    blocks.extend(_codex_reasoning_blocks(_row_value(row, "codex_reasoning_items"), covered_text=reasoning))
    for tool_index, tool_call in enumerate(_json_list(_row_value(row, "tool_calls")), start=1):
        tool_record = json_document(tool_call)
        if tool_record:
            blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{row['id']}-{tool_index}"))
    role = Role.normalize(_optional_text(row["role"]) or "unknown")
    tool_call_id = _optional_text(_row_value(row, "tool_call_id"))
    if role is Role.TOOL and text:
        is_error, exit_code, outcome_reason = tool_result_outcome(row["content"])
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_call_id,
                tool_name=_optional_text(_row_value(row, "tool_name")),
                text=text,
                is_error=is_error,
                exit_code=exit_code,
                outcome_unknown_reason=outcome_reason,
            )
        )
    token_count = _non_negative_int(_row_value(row, "token_count"))
    observed = _sqlite_bool(_row_value(row, "observed"), default=False)
    active = _sqlite_bool(_row_value(row, "active"), default=True)
    return ParsedMessage(
        provider_message_id=_message_provider_id(row),
        role=role,
        text=text,
        timestamp=_epoch_iso(row["timestamp"]),
        blocks=blocks,
        position=position,
        variant_index=0,
        is_active_path=active,
        material_origin=_material_origin(role, observed=observed),
        model_name=fallback_model,
        output_tokens=token_count if role is Role.ASSISTANT else (0 if token_count is not None else None),
        input_tokens=token_count if role is Role.USER else (0 if token_count is not None else None),
        end_turn=_end_turn_from_finish_reason(_row_value(row, "finish_reason")),
        stop_reason=_stop_reason_from_finish_reason(_row_value(row, "finish_reason")),
    )


def _message_provider_id(row: sqlite3.Row) -> str:
    platform_id = _optional_text(_row_value(row, "platform_message_id"))
    if platform_id:
        return platform_id
    return f"{row['session_id']}:message:{row['id']}"


def _usage_and_lifecycle_events(
    row: sqlite3.Row,
    messages: Sequence[ParsedMessage],
    *,
    session_columns: set[str],
) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    # The SQLite row distinguishes NULL from zero. If its producer used a
    # column DEFAULT 0 for an omitted insert, that earlier omission is no
    # longer recoverable here; the row's zero is the available evidence.
    total_usage = {
        event_key: value
        for event_key, wire_key in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cached_input_tokens", "cache_read_tokens"),
            ("cache_write_tokens", "cache_write_tokens"),
            ("reasoning_output_tokens", "reasoning_tokens"),
        )
        if (value := _non_negative_int(_row_value(row, wire_key))) is not None
    }
    has_cost_evidence = any(field in session_columns for field in _COST_FIELDS)
    if total_usage or has_cost_evidence:
        cost_payload = {
            field: (
                _optional_float(_row_value(row, field))
                if field in {"estimated_cost_usd", "actual_cost_usd"}
                else _row_value(row, field)
            )
            for field in _COST_FIELDS
            if field in session_columns
        }
        events.append(
            ParsedSessionEvent(
                event_type="token_count",
                timestamp=_epoch_iso(_row_value(row, "ended_at")) or _latest_message_timestamp(messages),
                payload={
                    "type": "token_count",
                    "model": _optional_text(_row_value(row, "model")),
                    "total_token_usage": total_usage,
                    "api_call_count": _non_negative_int(_row_value(row, "api_call_count")) or 0,
                    **cost_payload,
                },
            )
        )
    end_reason = _optional_text(_row_value(row, "end_reason"))
    if end_reason in _COMPACTION_END_REASONS:
        events.append(
            ParsedSessionEvent(
                event_type="compaction",
                timestamp=_epoch_iso(_row_value(row, "ended_at")),
                payload={"summary": f"Hermes session ended via {end_reason}", "end_reason": end_reason},
            )
        )
    if _non_negative_int(_row_value(row, "rewind_count")):
        events.append(
            ParsedSessionEvent(
                event_type="rewind",
                timestamp=_epoch_iso(_row_value(row, "ended_at")),
                payload={
                    "summary": "Hermes session was rewound",
                    "rewind_count": _non_negative_int(_row_value(row, "rewind_count")),
                },
            )
        )
    return events


def _branch_type(row: sqlite3.Row, parent_row: sqlite3.Row | Mapping[str, object] | None) -> BranchType | None:
    config = _json_mapping(_row_value(row, "model_config"))
    if config.get("_branched_from") is not None:
        return BranchType.FORK
    if config.get("_delegate_from") is not None or _optional_text(_row_value(row, "source")) == "tool":
        return BranchType.SUBAGENT
    if parent_row is not None and _optional_text(_row_value(parent_row, "end_reason")) in _COMPACTION_END_REASONS:
        return BranchType.CONTINUATION
    return None


@dataclass(frozen=True, slots=True)
class _CompressionIdentity:
    ordinal: int
    session_id: str
    parent_id: str | None
    continuation: bool


class _MemoryCompressionState:
    def __init__(self, coordinates: list[_CompressionIdentity]) -> None:
        self.nodes = {node.session_id: node for node in coordinates}
        self.states: dict[str, int] = {}
        self.stack: list[_CompressionIdentity] = []

    def lookup(self, key: str) -> _CompressionIdentity | None:
        return self.nodes.get(key)

    def status(self, key: str) -> int:
        return self.states.get(key, 0)

    def mark(self, node: _CompressionIdentity, status: int) -> None:
        self.states[node.session_id] = status

    def push(self, node: _CompressionIdentity) -> None:
        self.stack.append(node)

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> _CompressionIdentity:
        return self.stack[-1]

    def has_stack(self) -> bool:
        return bool(self.stack)


class _DiskCompressionState:
    """Private preview facts retain Python-string identity and parser order."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.depth = 0
        connection.execute(
            "CREATE TABLE hermes_compression_nodes (ordinal INTEGER PRIMARY KEY, "
            "session_id TEXT NOT NULL, parent_id TEXT, continuation INTEGER NOT NULL, facts TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE hermes_compression_identity (session_id TEXT COLLATE BINARY PRIMARY KEY, "
            "last_ordinal INTEGER NOT NULL, status INTEGER NOT NULL DEFAULT 0, chosen_ordinal INTEGER) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE hermes_compression_stack (depth INTEGER PRIMARY KEY, ordinal INTEGER NOT NULL)"
        )

    @staticmethod
    def coordinate(row: tuple[object, ...]) -> _CompressionIdentity:
        return _CompressionIdentity(
            int(cast(int, row[0])), str(row[1]), str(row[2]) if row[2] is not None else None, bool(row[3])
        )

    def add(self, node: _CompressionIdentity, facts: str) -> None:
        self.connection.execute(
            "INSERT INTO hermes_compression_nodes VALUES (?, ?, ?, ?, ?)",
            (node.ordinal, node.session_id, node.parent_id, int(node.continuation), facts),
        )
        self.connection.execute(
            "INSERT INTO hermes_compression_identity(session_id, last_ordinal) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET last_ordinal = excluded.last_ordinal",
            (node.session_id, node.ordinal),
        )

    def roots(self) -> Iterator[_CompressionIdentity]:
        with closing(
            self.connection.execute(
                "SELECT ordinal, session_id, parent_id, continuation FROM hermes_compression_nodes ORDER BY ordinal"
            )
        ) as cursor:
            for row in cursor:
                yield self.coordinate(row)

    def lookup(self, key: str) -> _CompressionIdentity | None:
        row = self.connection.execute(
            "SELECT n.ordinal, n.session_id, n.parent_id, n.continuation FROM hermes_compression_identity AS i "
            "JOIN hermes_compression_nodes AS n ON n.ordinal = i.last_ordinal WHERE i.session_id = ?",
            (key,),
        ).fetchone()
        return self.coordinate(row) if row is not None else None

    def status(self, key: str) -> int:
        row = self.connection.execute(
            "SELECT status FROM hermes_compression_identity WHERE session_id = ?", (key,)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def mark(self, node: _CompressionIdentity, status: int) -> None:
        self.connection.execute(
            "UPDATE hermes_compression_identity SET status = ?, chosen_ordinal = CASE WHEN ? = 2 THEN ? ELSE chosen_ordinal END "
            "WHERE session_id = ?",
            (status, status, node.ordinal, node.session_id),
        )

    def push(self, node: _CompressionIdentity) -> None:
        self.depth += 1
        self.connection.execute("INSERT INTO hermes_compression_stack VALUES (?, ?)", (self.depth, node.ordinal))

    def pop(self) -> None:
        self.connection.execute("DELETE FROM hermes_compression_stack WHERE depth = ?", (self.depth,))
        self.depth -= 1

    def top(self) -> _CompressionIdentity:
        row = self.connection.execute(
            "SELECT n.ordinal, n.session_id, n.parent_id, n.continuation FROM hermes_compression_stack AS s "
            "JOIN hermes_compression_nodes AS n USING(ordinal) WHERE s.depth = ?",
            (self.depth,),
        ).fetchone()
        assert row is not None
        return self.coordinate(row)

    def has_stack(self) -> bool:
        return self.depth > 0

    def selected_facts(self, key: str) -> str:
        row = self.connection.execute(
            "SELECT n.facts FROM hermes_compression_identity AS i JOIN hermes_compression_nodes AS n "
            "ON n.ordinal = i.chosen_ordinal WHERE i.session_id = ?",
            (key,),
        ).fetchone()
        assert row is not None
        return str(row[0])


def _compression_resolution_order(
    roots: Iterable[_CompressionIdentity], state: _MemoryCompressionState | _DiskCompressionState
) -> Iterator[tuple[_CompressionIdentity, str | None]]:
    """Own the parser's first-root/last-parent choice and cycle settlement."""
    for root in roots:
        state.push(root)
        while state.has_stack():
            node = state.top()
            if state.status(node.session_id) == 2:
                state.pop()
                continue
            parent = state.lookup(node.parent_id) if node.parent_id is not None and node.continuation else None
            if parent is None:
                state.mark(node, 2)
                yield node, None
                state.pop()
                continue
            if state.status(parent.session_id) == 2:
                state.mark(node, 2)
                yield node, parent.session_id
                state.pop()
                continue
            if state.status(node.session_id) == 1:
                state.mark(node, 2)
                yield node, None
                state.pop()
                continue
            state.mark(node, 1)
            state.push(parent)


def _inspect_state_connection(
    conn: sqlite3.Connection,
    grouping: sqlite3.Connection,
    path: Path,
    *,
    profile_root: Path | None = None,
    profile_identity: str | None = None,
) -> tuple[dict[str, object], HermesImportFidelity]:
    """Inspect exact parser evidence while retaining only aggregate facts."""
    conn.row_factory = sqlite3.Row
    if not _has_required_tables(conn):
        raise ValueError(f"{path} is not a Hermes state.db file")
    session_columns = _columns(conn, "sessions")
    message_columns = _columns(conn, "messages")
    schema_version = _schema_version(conn)
    resolved_profile_root = profile_root or profile_root_for_artifact(path)
    state = _DiskCompressionState(grouping)
    grouping.execute(
        "CREATE TABLE hermes_parent_evidence (session_id TEXT COLLATE BINARY PRIMARY KEY, end_reason TEXT) WITHOUT ROWID"
    )
    query = "SELECT * FROM sessions ORDER BY COALESCE(started_at, 0), id"
    # The list parser's rows_by_id uses Python str keys and keeps the last
    # row. SQLite collation/affinity must not choose a different parent.
    with closing(conn.execute(query)) as cursor:
        for row in cursor:
            grouping.execute(
                "INSERT INTO hermes_parent_evidence VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET end_reason = excluded.end_reason",
                (str(row["id"]), _optional_text(_row_value(row, "end_reason"))),
            )
    with closing(conn.execute(query)) as cursor:
        for ordinal, row in enumerate(cursor):
            parent_id = _row_value(row, "parent_session_id")
            parent = (
                grouping.execute(
                    "SELECT end_reason FROM hermes_parent_evidence WHERE session_id = ?", (str(parent_id),)
                ).fetchone()
                if parent_id is not None
                else None
            )
            inspection = _StateSessionInspection()
            header = _parse_session_row(
                conn,
                row,
                parent_row={"end_reason": parent[0]} if parent is not None else None,
                profile_root=resolved_profile_root,
                profile_identity=profile_identity,
                schema_version=schema_version,
                session_columns=session_columns,
                message_columns=message_columns,
                inspection=inspection,
            )
            inspection.evidence.sessions = 1
            facts = {
                "kept": bool(inspection.evidence.messages or header.instructions_text),
                "blocks": inspection.blocks,
                "actions": inspection.actions,
                "fidelity": inspection.evidence.counts(),
            }
            state.add(
                _CompressionIdentity(
                    ordinal,
                    header.provider_session_id,
                    header.parent_session_provider_id,
                    header.branch_type is BranchType.CONTINUATION,
                ),
                json.dumps(facts, ensure_ascii=True),
            )
    for _node, _parent in _compression_resolution_order(state.roots(), state):
        pass
    evidence = _LogicalFidelityEvidence()
    references: list[str] = []
    blocks = 0
    actions = 0
    for node in state.roots():
        facts = json.loads(state.selected_facts(node.session_id))
        if not facts["kept"]:
            continue
        evidence.merge(_LogicalFidelityEvidence.from_counts(facts["fidelity"]))
        blocks += facts["blocks"]
        actions += facts["actions"]
        references.append(f"session:{Provider.HERMES.value}:{node.session_id}")
    return {
        "sessions": evidence.sessions,
        "messages": evidence.messages,
        "blocks": blocks,
        "actions": actions,
        "raw_records": evidence.sessions,
        "session_refs": references,
    }, _logical_export_fidelity_from_evidence(evidence)


def _segment_compression_continuations(sessions: list[ParsedSession]) -> list[ParsedSession]:
    """Declare each continuation child's inherited prefix instead of replaying it.

    A Hermes compression child's own ``messages`` rows already ARE its
    divergent tail: the parent's transcript is not repeated in the source.
    Composing that prefix back on at parse time -- only for the archive writer
    to align it off again in ``_extract_prefix_tail`` -- made one parse retain
    ``M * N * (N + 1) / 2`` messages for a chain of ``N`` links holding ``M``
    each. That is the term that turned a ~2 MB ``state.db`` into 1.6M messages
    and 5.2 GB RSS (polylogue-g4g60), and the reason a long chain used to be
    refused outright rather than ingested.

    Each child now keeps its own rows and *declares* the divergence instead:
    ``branch_point_provider_message_id`` names the last message of the
    parent's composed transcript. The archive writer binds that to a stored
    parent message and records a ``prefix-sharing`` edge, so reads recompose
    the whole chain while storage holds each message exactly once (#2467) --
    which is what the writer was reducing the composed prefix to anyway.

    What the pass still carries forward is the composed LENGTH and the
    composed trailing id: two values per link, not a transcript. The length
    keeps each child's own messages at the positions they occupy in the
    composed transcript, so the stored rows are the same rows the
    compose-then-strip path wrote.

    A parent that contributes no messages of its own has no branch point in
    its own identity namespace, so no divergence is asserted for its child
    rather than asserting one that cannot bind.
    """
    resolved: dict[str, ParsedSession] = {}
    composed_lengths: dict[str, int] = {}
    #: Provider-native id of the last message of a session's COMPOSED
    #: transcript, which is its own last message unless it contributed none.
    composed_leaf_ids: dict[str, str | None] = {}

    def settle(session: ParsedSession, *, inherited: int, branch_point: str | None) -> None:
        session_id = session.provider_session_id
        messages = (
            session.messages
            if not inherited
            else [
                message.model_copy(update={"position": inherited + position})
                for position, message in enumerate(session.messages)
            ]
        )
        update: dict[str, object] = {}
        if messages is not session.messages:
            update["messages"] = messages
        if branch_point is not None:
            update["branch_point_provider_message_id"] = branch_point
        resolved[session_id] = session.model_copy(update=update) if update else session
        composed_lengths[session_id] = inherited + len(messages)
        composed_leaf_ids[session_id] = (
            messages[-1].provider_message_id
            if messages
            else composed_leaf_ids.get(str(session.parent_session_provider_id))
        )

    coordinates = [
        _CompressionIdentity(
            ordinal,
            session.provider_session_id,
            session.parent_session_provider_id,
            session.branch_type is BranchType.CONTINUATION,
        )
        for ordinal, session in enumerate(sessions)
    ]
    state = _MemoryCompressionState(coordinates)
    for coordinate, parent_id in _compression_resolution_order(coordinates, state):
        settle(
            sessions[coordinate.ordinal],
            inherited=composed_lengths[parent_id] if parent_id is not None else 0,
            branch_point=composed_leaf_ids[parent_id] if parent_id is not None else None,
        )

    return [resolved[session.provider_session_id] for session in sessions]


def _reported_cost(row: sqlite3.Row) -> float | None:
    actual = _optional_float(_row_value(row, "actual_cost_usd"))
    estimated = _optional_float(_row_value(row, "estimated_cost_usd"))
    return actual if actual is not None else estimated


def _reasoning_metadata(row: sqlite3.Row) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
        parsed = _json_value(_row_value(row, key))
        if parsed is not None:
            metadata[key] = parsed
    return metadata


# reasoning_details/codex_reasoning_items/codex_message_items are Hermes's
# captured Codex-native response items. `_reasoning_metadata` merges them into
# the THINKING block's `ParsedContentBlock.metadata` as an in-process carrier
# only: `blocks` has no metadata column and the write path
# (`storage/sqlite/archive_tiers/write.py:_block_language`) reads exactly one
# key back out of it -- `language` -- so metadata alone reaches nothing
# durable. `session_events`, keyed to the message via
# `source_message_provider_id`, is where the structured item belongs, the same
# precedent as `hermes_spans.py`'s `hermes_tool_availability_span` and
# `claude/common.py`'s `claude_ai_web_tool_evidence`.
#
# The prose inside `codex_message_items` is assistant output rather than
# evidence about it, so `_codex_output_text_blocks` also projects it into a
# TEXT block on the owning message; the event keeps what a block cannot hold
# (item ids, `phase`, `status`, `encrypted_content`).
def _reasoning_evidence_events(row: sqlite3.Row, message: ParsedMessage) -> list[ParsedSessionEvent]:
    evidence = _reasoning_metadata(row)
    if not evidence:
        return []
    return [
        ParsedSessionEvent(
            event_type="hermes_reasoning_evidence",
            timestamp=message.timestamp,
            source_message_provider_id=message.provider_message_id,
            payload=evidence,
        )
    ]


def _decode_content(value: object) -> object:
    if isinstance(value, str) and value.startswith(JSON_ENVELOPE_PREFIX):
        try:
            return json.loads(value[len(JSON_ENVELOPE_PREFIX) :])
        except json.JSONDecodeError:
            return value
    return value


def _json_mapping(value: object) -> dict[str, object]:
    parsed = _json_value(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_list(value: object) -> list[object]:
    parsed = _json_value(value)
    return parsed if isinstance(parsed, list) else []


def _json_value(value: object) -> object | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return cast(object, json.loads(value))
    except json.JSONDecodeError:
        return None


def _epoch_iso(value: object) -> str | None:
    seconds = _optional_float(value)
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def _latest_message_timestamp(messages: Sequence[ParsedMessage]) -> str | None:
    for message in reversed(messages):
        if message.timestamp:
            return message.timestamp
    return None


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    if not messages:
        return messages
    leaf_position = next(
        (position for position, message in reversed(list(enumerate(messages))) if message.is_active_path),
        None,
    )
    return [
        message.model_copy(update={"is_active_leaf": position == leaf_position})
        for position, message in enumerate(messages)
    ]


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() and value >= 0 else None
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            parsed = float(value)
        except OverflowError:
            return None
        return parsed if math.isfinite(parsed) and parsed >= 0 else None
    return None


__all__ = [
    "HERMES_STATE_DB_MARKER",
    "HermesFidelityCapability",
    "SourceFidelityStatus",
    "HermesImportFidelity",
    "import_fidelity_declaration",
    "looks_like_state_db_path",
    "looks_like_state_db_payload",
    "marker_payload",
    "parse_state_db",
    "parse_state_db_payload",
    "iter_state_db_sessions",
    "require_declared_export",
]


def detection_projection() -> DetectorProjection:
    """Keep the marker and declaration used by the state-export detector."""
    return DetectorProjection(
        fields={"polylogue_artifact": DetectorProjection(), "state_db_path": DetectorProjection()}
    )
