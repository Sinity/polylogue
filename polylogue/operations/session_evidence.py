"""Per-session evidence relations, read through the pinned operation reader.

These are the read models behind ``read --view file-edits``,
``--view agent-policies`` and ``--view web-content``: index-tier relations
that ride one exact session reference, have no query-grammar unit of their own
(design D3, ``cli/read_view_registry.py``), and are bounded by construction --
one session's file edits, its agent-policy facts, its web constructs.

Before this module each of those views opened the archive in this process
through the Python API facade, which is what ``IN_PROCESS_READ_VIEWS``
ratchets down (polylogue-r3cuz).  The rows here are the *same* rows the facade
returns, field for field and in the same order: the SQL and the record mappers
are shared with the async readers rather than restated, and the projections
below are the ones the CLI already rendered.  Moving a view must not change
its document.

Which relations are answered whole is decided by whether a *row* is bounded,
not by whether the relation is.  ``hooks`` is an aggregate summary and
``agent-policies`` is a handful of short policy facts, so both are answered
whole and the ``session.read`` result reports ``complete`` with no
continuation.

Everything else is *windowed*.  ``events`` and ``raw`` graduated first,
because both already accepted a row bound and one of them reported the
truncated count as its total.  ``file-edits`` and ``web-content`` followed for
a different reason: their rows carry unbounded payloads -- ``original_file``
is the pre-edit contents of whatever a tool call touched, ``text`` is a
fetched page body -- so a real session could exceed the 8 MiB operation-result
bound, be refused by ``_require_deliverable_window``, and have no retry
available, because a whole-evidence kind rejects window coordinates outright.
Being unreadable is not a bound.

Windowed readers answer ``(rows, total)`` where ``total`` is the **relation's
own** row count, never the returned count; ``operations/evidence_window.py``
decides the page and mints the continuation, and ``EvidenceWindowBody``
refuses a clipped page that claims to be whole.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from polylogue.operations.evidence_payloads import (
    DEFAULT_EVIDENCE_PAGE_BYTES,
    EvidencePayloadPage,
    read_evidence_payload_page,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

__all__ = [
    "SESSION_EVIDENCE_PAGE_READERS",
    "EvidencePageReader",
    "read_agent_policies_evidence",
    "read_file_edits_page",
    "read_raw_artifacts_page",
    "read_session_events_page",
    "read_session_evidence_window",
    "read_session_materials",
    "read_session_materials_page",
    "read_web_content_constructs_page",
    "session_materials_source_epoch",
]


def read_file_edits_page(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], int]:
    """Project one page of ``file_edits`` as ``read --view file-edits`` renders them.

    Paged rather than answered whole because a single row is not bounded: a
    file edit carries ``original_file``, the pre-edit contents of whatever the
    tool call touched, so one edit of a large file can exceed the 8 MiB
    operation-result bound on its own. Answered whole, such a session was
    materialized in full and then refused by ``_require_deliverable_window``
    with "retry with a smaller limit" -- advice a whole-evidence kind could not
    take, because it rejects window coordinates outright. There was no
    successful retry, so the session was simply unreadable through this view.

    The relation is read whole and sliced here, like ``session_events``: the
    order that must be preserved is the repository's own
    (``ORDER BY message_id, tool_use_block_id``), and composing the page from
    the full ordered list keeps "the same rows, in the same order" checkable.
    The returned total is the relation's own count, never the page's.
    """

    from polylogue.storage.sqlite.queries.file_edits import _SELECT_COLUMNS
    from polylogue.storage.sqlite.queries.mappers import _row_to_file_edit

    conn = archive._conn
    total = int(conn.execute("SELECT COUNT(*) FROM file_edits WHERE session_id = ?", (session_id,)).fetchone()[0])
    edits = (
        [
            _row_to_file_edit(row)
            for row in conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM file_edits WHERE session_id = ? "
                "ORDER BY message_id, tool_use_block_id LIMIT ? OFFSET ?",
                (session_id, max(limit, 0), max(offset, 0)),
            ).fetchall()
        ]
        if limit > 0
        else []
    )
    rows: list[dict[str, object]] = [
        {
            "tool_use_block_id": edit.tool_use_block_id,
            "message_id": str(edit.message_id),
            "file_path": edit.file_path,
            "structured_patch": edit.structured_patch,
            "original_file": edit.original_file,
            "old_string": edit.old_string,
            "new_string": edit.new_string,
            "replace_all": edit.replace_all,
            "user_modified": edit.user_modified,
            "observed_at_ms": edit.observed_at_ms,
        }
        for edit in edits
    ]
    return rows, total


def read_agent_policies_evidence(archive: ArchiveStore, session_id: str) -> dict[str, object]:
    """Project ``session_agent_policies`` rows as ``read --view agent-policies`` renders them."""

    from polylogue.storage.sqlite.queries.session_agent_policies import sync_session_agent_policies

    policies = sync_session_agent_policies(archive._conn, session_id)
    return {
        "session_id": session_id,
        "total": len(policies),
        "agent_policies": [
            {
                "policy_id": policy.policy_id,
                "position": policy.position,
                "approval_policy": policy.approval_policy,
                "sandbox_policy": policy.sandbox_policy,
                "network_policy": policy.network_policy,
                "observed_at_ms": policy.observed_at_ms,
                "source_message_id": policy.source_message_id,
            }
            for policy in policies
        ],
    }


def read_web_content_constructs_page(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], int]:
    """Project one page of ``web_content_constructs`` as ``read --view web-content`` renders them.

    Paged for the same reason as ``file_edits``: each construct carries
    ``text``, the fetched page or search result body, so a session with enough
    web evidence crossed the 8 MiB operation-result bound and became
    unreadable with no retry that could succeed.

    Sliced from the repository's own order
    (``ORDER BY message_id, block_id, position``); the reported total is the
    relation's own row count.
    """

    from polylogue.storage.sqlite.queries.mappers_archive import _row_to_web_content_construct
    from polylogue.storage.sqlite.queries.web_content_constructs import _SELECT_COLUMNS

    conn = archive._conn
    total = int(
        conn.execute("SELECT COUNT(*) FROM web_content_constructs WHERE session_id = ?", (session_id,)).fetchone()[0]
    )
    constructs = (
        [
            _row_to_web_content_construct(row)
            for row in conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM web_content_constructs WHERE session_id = ? "
                "ORDER BY message_id, block_id, position LIMIT ? OFFSET ?",
                (session_id, max(limit, 0), max(offset, 0)),
            ).fetchall()
        ]
        if limit > 0
        else []
    )
    rows: list[dict[str, object]] = [
        {
            "construct_id": construct.construct_id,
            "message_id": str(construct.message_id),
            "block_id": construct.block_id,
            "position": construct.position,
            "provider": construct.provider,
            "construct_type": construct.construct_type,
            "provider_key": construct.provider_key,
            "title": construct.title,
            "url": construct.url,
            "text": construct.text,
            "source_id": construct.source_id,
            "group_id": construct.group_id,
            "group_title": construct.group_title,
            "query": construct.query,
            "asset_pointer": construct.asset_pointer,
            "mime_type": construct.mime_type,
            "status": construct.status,
            "task_id": construct.task_id,
            "task_type": construct.task_type,
            "rank": construct.rank,
            "start_index": construct.start_index,
            "end_index": construct.end_index,
        }
        for construct in constructs
    ]
    return rows, total


def read_session_events_page(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], int]:
    """Project one page of ``session_events`` as ``read --view events`` renders them.

    The rows are composed through the *same* record hydrator the Python API
    facade reads them with (``storage/hydrators.session_event_from_record``)
    and projected with the same five fields, so the move cannot quietly change
    a timestamp's spelling or drop a payload
    (``tests/unit/operations/test_session_evidence_readers.py``).

    The relation is read whole and sliced here rather than windowed in SQL:
    a session's timeline events are bounded by the session, the order that
    must be preserved is the repository's own (``ORDER BY position``), and
    composing the page from the full ordered list is what makes "the same rows
    the facade returns, in the same order" checkable instead of asserted.
    The returned total is that full count -- which is exactly the number the
    old payload could not report, because it reported the clipped one.
    """

    from polylogue.storage.hydrators import session_event_from_record
    from polylogue.storage.sqlite.queries.session_events import sync_session_events_batch

    records = sync_session_events_batch(archive._conn, [session_id]).get(session_id, [])
    events = [session_event_from_record(record) for record in records]
    page = events[offset : offset + limit] if limit else []
    rows: list[dict[str, object]] = [
        {
            "event_id": str(event.id),
            "event_index": event.event_index,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat() if event.timestamp is not None else None,
            "payload": event.payload,
        }
        for event in page
    ]
    return rows, len(events)


def read_raw_artifacts_page(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], int]:
    """Project one page of source-tier acquisition rows as ``read --view raw`` renders them.

    This relation was already genuinely windowed -- ``raw_artifacts_for_session``
    applies ``LIMIT``/``OFFSET`` in SQL and counts the relation separately --
    so the move is a change of *who reports the bound*, not of how the page is
    read.  The four projected keys are the ones the view has always rendered;
    ``source_name`` is among them and the source row carries no such column,
    so it has always come back empty.  Preserving it is deliberate: dropping a
    key while moving the view is the silent JSON regression this whole route
    is guarded against, and correcting the raw view's document is a decision
    about that document, not about which executor answers it.
    """

    artifacts, total = archive.raw_artifacts_for_session(session_id, limit=limit, offset=offset)
    rows = [
        {
            "raw_id": artifact.get("raw_id", ""),
            "source_name": artifact.get("source_name", ""),
            "source_path": artifact.get("source_path", ""),
            "blob_size": artifact.get("blob_size", 0),
        }
        for artifact in artifacts
    ]
    return rows, total


#: The session-scoped material relation, reached through its indexed link.
#: The material route (``sources/codex_state_evidence.py``) records every
#: admission's ``referrer_ref`` together with a ``refers_to`` link to that same
#: ref, in one transaction, and ``material_evidence_links`` is indexed on
#: ``(evidence_ref, relation)``. Joining back on ``referrer_ref`` keeps the
#: relation exactly "materials whose referrer is this session" -- the rows
#: ``excise --session`` follows -- while every page, count and epoch costs the
#: session's own materials rather than a scan of the archive-wide table.
_SESSION_MATERIALS_FROM = (
    "FROM material_evidence_links AS l "
    "JOIN material_observations AS m ON m.material_id = l.material_id AND m.referrer_ref = l.evidence_ref "
    "WHERE l.evidence_ref IN (?, ?) AND l.relation = 'refers_to'"
)
_SESSION_MATERIALS_ORDER = " ORDER BY m.created_at_ms, m.material_id"


def _session_material_refs(session_id: str) -> tuple[str, str]:
    """Codex goals and memories name the session id itself; other admissions use ``session:``."""

    return (session_id, f"session:{session_id}")


def _material_content(media_type: str | None, charset: str | None, payload: bytes) -> tuple[object, str]:
    """Decode retained bytes for display, returning ``(content, content_form)``.

    A material is admitted whatever its bytes turn out to be:
    ``admit_material`` keeps an ``application/json`` payload that failed
    extraction as ``acquisition_state="malformed"``. Such a row is a valid
    observation, so a decoding failure never aborts the page; the row is
    returned with the text it has (``"text"``) or none (``"undecodable"``) and
    the caller reads its typed ``acquisition_state``.
    """

    media = (media_type or "").split(";", 1)[0].strip().lower()
    if media == "application/json":
        try:
            return json.loads(payload), "json"
        except ValueError:
            pass
    elif not media.startswith("text/"):
        return None, "bytes"
    try:
        return payload.decode(charset or "utf-8"), "text"
    except (LookupError, UnicodeDecodeError):
        return None, "undecodable"


def _session_material_rows(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int | None,
    offset: int,
) -> tuple[list[dict[str, object]], int]:
    """Read source-tier materials whose referrer is this session, with their content."""

    from polylogue.storage.materials import get_material, read_material

    conn = archive.source_connection
    refs = _session_material_refs(session_id)
    total = int(conn.execute("SELECT COUNT(*) " + _SESSION_MATERIALS_FROM, refs).fetchone()[0])
    bound = "" if limit is None else " LIMIT ? OFFSET ?"
    params: tuple[object, ...] = refs if limit is None else (*refs, max(limit, 0), max(offset, 0))
    ids = [
        str(row[0])
        for row in conn.execute(
            "SELECT m.material_id " + _SESSION_MATERIALS_FROM + _SESSION_MATERIALS_ORDER + bound,
            params,
        ).fetchall()
    ]
    rows: list[dict[str, object]] = []
    for material_id in ids:
        material = get_material(conn, material_id)
        if material is None:
            continue
        content: object = None
        content_form = "absent"
        if material.blob_hash is not None:
            content, content_form = _material_content(
                material.media_type, material.media_charset, read_material(conn, material_id)
            )
        rows.append(
            {
                "material_id": material.material_id,
                "source_uri": material.source_uri,
                "acquisition_state": material.acquisition_state,
                "media_type": material.media_type,
                "filename": material.filename,
                "byte_size": material.byte_size,
                "blob_hash": material.blob_hash,
                "privacy_classification": material.privacy_classification,
                "acquired_at_ms": material.acquired_at_ms,
                "created_at_ms": material.created_at_ms,
                "content": content,
                "content_form": content_form,
            }
        )
    return rows, total


def session_materials_source_epoch(archive: ArchiveStore, session_id: str) -> str:
    """Fingerprint the ordered membership of the session's material relation.

    The relation lives in ``source.db``, which the index/user archive frame
    does not cover, so a material admitted ahead of a continuation's offset
    would shift the next page without making the token stale. The window
    binds this fingerprint instead: it moves exactly when the ordered
    ``(created_at_ms, material_id)`` sequence the offsets index into moves.
    """

    digest = hashlib.sha256()
    for created_at_ms, material_id in archive.source_connection.execute(
        "SELECT m.created_at_ms, m.material_id " + _SESSION_MATERIALS_FROM + _SESSION_MATERIALS_ORDER,
        _session_material_refs(session_id),
    ):
        digest.update(f"{int(created_at_ms)}\0{material_id}\n".encode())
    return f"source:materials:{digest.hexdigest()}"


def read_session_materials_page(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], int]:
    """Project one page of the session's retained materials for ``read --view materials``.

    Codex goals and memories terminate here and nowhere else: they have no
    index-tier relation, so without this read the objective, status and memory
    text were retained but unreachable from every public surface. Windowed
    because a material's bytes are not bounded (a memory is split into text
    parts, a fetched document is one row). Ordered by admission,
    ``created_at_ms, material_id``; the returned total is the relation's own
    count.
    """

    if limit <= 0:
        return [], _session_material_rows(archive, session_id, limit=0, offset=0)[1]
    return _session_material_rows(archive, session_id, limit=limit, offset=offset)


def read_session_materials(archive: ArchiveStore, session_id: str) -> list[dict[str, object]]:
    """Every retained material for one session, in the windowed read's order."""

    return _session_material_rows(archive, session_id, limit=None, offset=0)[0]


#: One page of a per-session evidence relation: ``(archive, session_id, limit,
#: offset)`` in, the page's rows and the relation's own total out.
EvidencePageReader = Callable[["ArchiveStore", str, int, int], tuple[list[dict[str, object]], int]]

#: Per-session evidence relations that are *paged* rather than answered whole,
#: keyed by the ``session.read`` kind that names them.  Every surface that
#: pages one of them -- ``session.read`` on the daemon and CLI, MCP ``read`` and
#: ``get`` through the Python API -- reads it through this table.
SESSION_EVIDENCE_PAGE_READERS: dict[str, EvidencePageReader] = {
    "events": lambda archive, session_id, limit, offset: read_session_events_page(
        archive, session_id, limit=limit, offset=offset
    ),
    "raw": lambda archive, session_id, limit, offset: read_raw_artifacts_page(
        archive, session_id, limit=limit, offset=offset
    ),
    "file-edits": lambda archive, session_id, limit, offset: read_file_edits_page(
        archive, session_id, limit=limit, offset=offset
    ),
    "web-content": lambda archive, session_id, limit, offset: read_web_content_constructs_page(
        archive, session_id, limit=limit, offset=offset
    ),
    "materials": lambda archive, session_id, limit, offset: read_session_materials_page(
        archive, session_id, limit=limit, offset=offset
    ),
}

#: Windowed kinds whose rows live outside the archive frame's tiers. ``raw``
#: is absent because its relation is keyed by the session's own ``raw_id``,
#: the ``raw_sessions`` primary key, so it holds at most one row and has no
#: offset to shift.
_SOURCE_RELATION_EPOCHS: dict[str, Callable[[ArchiveStore, str], str]] = {
    "materials": session_materials_source_epoch,
}


def read_session_evidence_window(
    archive: ArchiveStore,
    kind: str,
    *,
    ref: str,
    limit: int,
    offset: int,
    continuation: str | None,
    max_bytes: int = DEFAULT_EVIDENCE_PAGE_BYTES,
) -> Mapping[str, object] | None:
    """Answer one page of a windowed per-session relation, or ``None`` for an unknown session.

    ``ref`` is the caller's session reference; it is the request identity the
    continuation is bound to, so a token resumes only the read that minted it.
    """

    from polylogue.operations.evidence_window import EVIDENCE_WINDOW_FAMILIES, read_evidence_window

    try:
        session_id = archive.resolve_session_id(ref.removeprefix("session:"))
    except KeyError:
        return None
    reader = SESSION_EVIDENCE_PAGE_READERS[kind]
    source_epoch = _SOURCE_RELATION_EPOCHS.get(kind)

    def read_payload(
        page_limit: int, page_offset: int, cursor: Mapping[str, object] | None, budget: int
    ) -> EvidencePayloadPage:
        return read_evidence_payload_page(
            archive._conn,
            kind=kind,
            session_id=session_id,
            limit=page_limit,
            offset=page_offset,
            cursor=cursor,
            budget=budget,
            read_rows=lambda count, start: reader(archive, session_id, count, start),
        )

    return read_evidence_window(
        archive,
        EVIDENCE_WINDOW_FAMILIES[kind],
        ref=ref,
        limit=limit,
        offset=offset,
        continuation=continuation,
        read=lambda page_limit, page_offset: reader(archive, session_id, page_limit, page_offset),
        source_epoch=None if source_epoch is None else (lambda: source_epoch(archive, session_id)),
        read_payload=read_payload if kind in {"file-edits", "web-content"} else None,
        max_bytes=max_bytes,
    )
