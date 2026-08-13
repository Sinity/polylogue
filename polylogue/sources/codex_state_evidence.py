"""Durable session-linked evidence derived from retained Codex state."""

from __future__ import annotations

from json import dumps as json_dumps
from typing import Any

from polylogue.core.enums import Origin, Provider
from polylogue.sources.parsers import codex_state


def write_codex_thread_state_evidence(
    archive: Any,
    snapshot: codex_state.CodexStateSnapshot,
    *,
    source_path: str,
    acquired_at_ms: int,
) -> None:
    """Attach state-db thread metadata to existing Codex sessions.

    The state database is evidence about sessions, never a session itself.
    Both live ingest and retained-raw replay call this writer so a
    source-only acquisition is completed from the immutable snapshot when
    the derived tier returns.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

    for thread in snapshot.threads:
        payload: dict[str, object] = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "cwd": thread.cwd,
            "source": thread.source,
            "model": thread.model,
            "agent_nickname": thread.agent_nickname,
            "agent_role": thread.agent_role,
            "archived": thread.archived,
        }
        encoded = json_dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=encoded,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-title:{thread.thread_id}",
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="codex_thread_title",
                payload=payload,
                observed_at_ms=thread.updated_at_ms or acquired_at_ms,
                native_id=f"{thread.thread_id}:codex_thread_title",
                session_native_id=thread.thread_id,
            ),
        )
    for edge in snapshot.spawn_edges:
        payload = {
            "parent_thread_id": edge.parent_thread_id,
            "child_thread_id": edge.child_thread_id,
            "status": edge.status,
        }
        encoded = json_dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=encoded,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-spawn-edge:{edge.parent_thread_id}:{edge.child_thread_id}",
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="codex_thread_spawn_edge",
                payload=payload,
                observed_at_ms=acquired_at_ms,
                native_id=f"{edge.parent_thread_id}:{edge.child_thread_id}:codex_thread_spawn_edge",
                session_native_id=edge.parent_thread_id,
            ),
        )


__all__ = ["write_codex_thread_state_evidence"]
