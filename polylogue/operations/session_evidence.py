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

Each relation is answered whole.  That is a property of the relation, not a
convenience: a session's file edits are bounded by its tool calls, so there is
no window to decide, and the ``session.read`` result reports ``complete``
with no continuation.  A relation that later needs paging graduates to the
windowed contract rather than quietly truncating here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

__all__ = [
    "read_agent_policies_evidence",
    "read_file_edits_evidence",
    "read_web_content_constructs_evidence",
]


def read_file_edits_evidence(archive: ArchiveStore, session_id: str) -> dict[str, object]:
    """Project ``file_edits`` rows exactly as ``read --view file-edits`` renders them."""

    from polylogue.storage.sqlite.queries.file_edits import sync_file_edits_for_session

    edits = sync_file_edits_for_session(archive._conn, session_id)
    return {
        "session_id": session_id,
        "total": len(edits),
        "file_edits": [
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
        ],
    }


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


def read_web_content_constructs_evidence(archive: ArchiveStore, session_id: str) -> dict[str, object]:
    """Project ``web_content_constructs`` rows as ``read --view web-content`` renders them."""

    from polylogue.storage.sqlite.queries.web_content_constructs import sync_web_content_constructs_for_session

    constructs = sync_web_content_constructs_for_session(archive._conn, session_id)
    return {
        "session_id": session_id,
        "total": len(constructs),
        "web_content_constructs": [
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
        ],
    }
