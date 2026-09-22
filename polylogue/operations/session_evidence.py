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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

__all__ = [
    "read_agent_policies_evidence",
    "read_file_edits_page",
    "read_raw_artifacts_page",
    "read_session_events_page",
    "read_web_content_constructs_page",
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

    from polylogue.storage.sqlite.queries.file_edits import sync_file_edits_for_session

    edits = sync_file_edits_for_session(archive._conn, session_id)
    page = edits[offset : offset + limit] if limit else []
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
        for edit in page
    ]
    return rows, len(edits)


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

    from polylogue.storage.sqlite.queries.web_content_constructs import sync_web_content_constructs_for_session

    constructs = sync_web_content_constructs_for_session(archive._conn, session_id)
    page = constructs[offset : offset + limit] if limit else []
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
        for construct in page
    ]
    return rows, len(constructs)


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
