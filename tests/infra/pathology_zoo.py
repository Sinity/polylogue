"""Curated production-ingest fixture archive for known pathology shapes.

The zoo is test memory, not a second writer. Every member starts as a wire
artifact and reaches SQLite only through :func:`parse_sources_archive`.
Production owns the manifest and read-only invariants. This module owns only
fixture generation and the SQL mutations that make those invariants red.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

from polylogue.config import Source
from polylogue.core.enums import Origin
from polylogue.maintenance.pathology_zoo import (
    PATHOLOGY_ZOO_MANIFEST as PRODUCTION_PATHOLOGY_ZOO_MANIFEST,
)
from polylogue.maintenance.pathology_zoo import (
    PathologyZooMember,
)
from polylogue.maintenance.pathology_zoo import (
    pathology_zoo_session_ids as production_pathology_zoo_session_ids,
)
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources.hooks import drain_hook_event_spool, enqueue_hook_event
from tests.infra.source_builders import SyntheticAntigravityLanguageServerClient

CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID = "9ed2056f-b415-4f51-b18e-5265f21a67bf"
CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN = Origin.CLAUDE_AI_EXPORT.value
CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY = f"claude-ai:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}"


@dataclass(frozen=True, slots=True)
class PathologyZoo:
    """A materialized archive plus the production-owned manifest."""

    archive_root: Path
    manifest: tuple[PathologyZooMember, ...]

    def members_for(self, pathology: str) -> tuple[PathologyZooMember, ...]:
        return tuple(member for member in self.manifest if member.pathology == pathology)

    @property
    def canary_session_ids(self) -> tuple[str, ...]:
        """Expose the production manifest's replayable session identities."""
        return production_pathology_zoo_session_ids()


@dataclass(frozen=True, slots=True)
class PathologyZooMutation:
    """Fixture-only SQL that deliberately falsifies one production invariant."""

    tier: str
    statement: str
    parameters: tuple[object, ...] = ()

    def apply(self, archive_root: Path) -> None:
        if self.tier == "index":
            from polylogue.storage.archive_identity import resolve_active_index_path

            path = resolve_active_index_path(archive_root)
        else:
            path = archive_root / f"{self.tier}.db"
        with sqlite3.connect(path) as connection:
            connection.execute(self.statement, self.parameters)
            connection.commit()


_PATHOLOGY_ZOO_MUTATIONS: dict[str, PathologyZooMutation] = {
    "whale-component": PathologyZooMutation(
        "index", "UPDATE sessions SET message_count = 47 WHERE session_id = ?", ("codex-session:zoo-whale",)
    ),
    "append-self-describing": PathologyZooMutation(
        "source", "DELETE FROM raw_sessions WHERE source_path LIKE '%zoo-02-00.jsonl'"
    ),
    "append-opaque": PathologyZooMutation(
        "source", "DELETE FROM raw_sessions WHERE source_path LIKE '%zoo-04-00.jsonl'"
    ),
    "fork-prefix-tail": PathologyZooMutation(
        "index",
        "UPDATE session_links SET resolved_dst_session_id = NULL WHERE src_session_id = ?",
        ("codex-session:zoo-lineage-child",),
    ),
    "lineage-cycle": PathologyZooMutation(
        "index",
        "UPDATE session_links SET status = NULL WHERE src_session_id = ?",
        ("codex-session:zoo-cycle-a",),
    ),
    "grouped-jsonl": PathologyZooMutation(
        "index",
        "DELETE FROM sessions WHERE session_id = ?",
        ("claude-code-session:zoo-group-b",),
    ),
    "quarantined-head": PathologyZooMutation(
        "index",
        "DELETE FROM session_links WHERE src_session_id = ?",
        ("codex-session:zoo-orphan-head",),
    ),
    "empty-session": PathologyZooMutation(
        "index",
        "UPDATE sessions SET message_count = 1 WHERE session_id = ?",
        ("claude-code-session:zoo-empty",),
    ),
    "hook-event": PathologyZooMutation(
        "source", "DELETE FROM raw_hook_events WHERE hook_event_id = 'hook:zoo-hook-event'"
    ),
    "claude-design": PathologyZooMutation(
        "index", "UPDATE sessions SET origin = 'codex-session' WHERE native_id = 'zoo-design-session'"
    ),
    "vintage-reorder": PathologyZooMutation(
        "source", "DELETE FROM raw_sessions WHERE source_path LIKE '%vintage-new.json'"
    ),
    "claude-vintage-live-proof": PathologyZooMutation(
        "source",
        """
        DELETE FROM raw_sessions
        WHERE raw_id = (
            SELECT r.raw_id
            FROM raw_sessions AS r
            JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
            WHERE r.origin = ?
              AND m.logical_source_key = ?
              AND r.source_path LIKE ?
            ORDER BY r.source_path DESC
            LIMIT 1
        )
        """,
        (
            Origin.CLAUDE_AI_EXPORT.value,
            f"claude-ai:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}",
            "%claude-live-proof-new.json",
        ),
    ),
    "lifecycle-anchor-drift": PathologyZooMutation(
        "index",
        "UPDATE session_events SET source_message_provider_id = 'lifecycle-a' WHERE session_id = ? AND event_type = 'generation_lifecycle'",
        ("chatgpt-export:zoo-lifecycle-anchor-drift",),
    ),
    "non-stream-safe": PathologyZooMutation(
        "index", "DELETE FROM sessions WHERE session_id = ?", ("antigravity-session:zoo-06-00:zoo-06-00.md",)
    ),
    "attachment-with-bytes": PathologyZooMutation(
        "index",
        "UPDATE attachments SET blob_hash = NULL, acquisition_status = 'unfetched' WHERE attachment_id IN (SELECT attachment_id FROM attachment_refs WHERE session_id = ?)",
        ("aistudio-drive:zoo-attachment-bytes",),
    ),
    "attachment-without-bytes": PathologyZooMutation(
        "index",
        "UPDATE attachments SET acquisition_status = 'acquired' WHERE attachment_id IN (SELECT attachment_id FROM attachment_refs WHERE session_id = ?)",
        ("claude-ai-export:zoo-attachment-metadata",),
    ),
    "events-sidecars": PathologyZooMutation(
        "index", "DELETE FROM session_events WHERE session_id = ?", ("chatgpt-export:zoo-lifecycle-anchor-drift",)
    ),
}


_CODEX = "codex-session"
_CLAUDE_AI = "claude-ai-export"
_CLAUDE_CODE = "claude-code-session"
_CHATGPT = "chatgpt-export"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: Iterable[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _code_record(
    session_id: str, message_id: str, role: str, text: str, *, parent: str | None = None
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": role,
        "sessionId": session_id,
        "uuid": message_id,
        "timestamp": "2026-08-04T00:00:00Z",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }
    if parent is not None:
        record["parentUuid"] = parent
    return record


def _codex_records(
    session_id: str, texts: tuple[str, ...], *, parent: str | None = None, subagent: bool = False
) -> tuple[dict[str, object], ...]:
    meta: dict[str, object] = {"id": session_id, "timestamp": "2026-08-04T00:00:00Z"}
    if parent is not None:
        meta["forked_from_id"] = parent
    if subagent:
        meta["source"] = {"subagent": {"thread_spawn": True}}
    records: list[dict[str, object]] = [{"type": "session_meta", "payload": meta}]
    for position, text in enumerate(texts):
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{session_id}-m{position}",
                    "role": "user" if position % 2 == 0 else "assistant",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    return tuple(records)


def _chatgpt_payload(conversation_id: str, nodes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": conversation_id,
        "conversation_id": conversation_id,
        "create_time": 1_700_000_000.0,
        "current_node": str(nodes[-1]["id"]),
        "mapping": {str(node["id"]): node for node in nodes},
    }


def _chatgpt_node(
    message_id: str, role: str, text: str, *, parent: str | None = None, lifecycle: bool = False
) -> dict[str, object]:
    message: dict[str, object] = {
        "id": message_id,
        "author": {"role": role},
        "content": {"content_type": "text", "parts": [text]},
        "create_time": 1_700_000_000.0,
    }
    if lifecycle:
        message["metadata"] = {"finished_duration_sec": 5}
    node: dict[str, object] = {"id": message_id, "message": message, "children": []}
    if parent is not None:
        node["parent"] = parent
    return node


CLAUDE_VINTAGE_LIVE_PROOF_MESSAGE_IDS = (
    "64878c9e-2642-437b-a384-0961184f84ea",
    "4c341ad3-dbd9-4224-9ab4-dcb4f833b3f9",
    "49cad03c-d06e-48a8-8f0a-81e40ef234cd",
)


def _claude_vintage_live_proof_payload(*, nested_target: bool) -> dict[str, object]:
    """Return the sanitized old/new wire pair from the parent measurement.

    The parent measurement identified a Claude.ai export-vintage difference:
    one version carries a message's text at the top level, while another
    carries the same text in one ``content`` text segment. The IDs and prose
    here are synthetic because the cited live bytes were not recoverable.
    """
    target: dict[str, object] = {
        "uuid": CLAUDE_VINTAGE_LIVE_PROOF_MESSAGE_IDS[2],
        "sender": "human",
    }
    if nested_target:
        target["content"] = [{"type": "text", "text": "sanitized measured target turn"}]
    else:
        target["text"] = "sanitized measured target turn"
    return {
        "uuid": CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID,
        "title": "sanitized measured Claude vintage cohort",
        "created_at": "2026-07-31T00:00:00Z",
        "updated_at": "2026-07-31T00:03:00Z",
        "chat_messages": [
            {
                "uuid": CLAUDE_VINTAGE_LIVE_PROOF_MESSAGE_IDS[0],
                "sender": "human",
                "text": "sanitized first turn",
                "created_at": "2026-07-31T00:00:00Z",
            },
            {
                "uuid": CLAUDE_VINTAGE_LIVE_PROOF_MESSAGE_IDS[1],
                "sender": "assistant",
                "text": "sanitized assistant turn",
                "created_at": "2026-07-31T00:01:00Z",
            },
            {
                **target,
                "created_at": "2026-07-31T00:02:00Z",
            },
        ],
    }


def write_claude_vintage_live_proof_pair(root: Path) -> tuple[Path, Path]:
    """Write the sanitized measured-shape pair into a pathology wire root."""
    manual = root / "manual"
    return (
        _write_json(
            manual / "claude-live-proof-old.json",
            _claude_vintage_live_proof_payload(nested_target=False),
        ),
        _write_json(
            manual / "claude-live-proof-new.json",
            _claude_vintage_live_proof_payload(nested_target=True),
        ),
    )


def _write_manual_members(root: Path) -> tuple[Path, ...]:
    manual = root / "manual"
    grouped_records = (
        _code_record("zoo-group-a", "group-a-1", "user", "grouped A"),
        _code_record("zoo-group-b", "group-b-1", "user", "grouped B"),
        _code_record("zoo-group-a", "group-a-2", "assistant", "grouped A reply", parent="group-a-1"),
        _code_record("zoo-group-b", "group-b-2", "assistant", "grouped B reply", parent="group-b-1"),
    )
    vintage_nodes = [
        _chatgpt_node("vintage-user", "user", "same conversation"),
        _chatgpt_node("vintage-assistant", "assistant", "same answer", parent="vintage-user"),
    ]
    lifecycle_nodes = [
        _chatgpt_node("lifecycle-user", "user", "do the work"),
        _chatgpt_node("lifecycle-a", "assistant", "draft", parent="lifecycle-user", lifecycle=True),
        _chatgpt_node("lifecycle-b", "assistant", "answer", parent="lifecycle-a", lifecycle=True),
    ]
    return (
        _write_jsonl(
            manual / "lineage-parent.jsonl", _codex_records("zoo-lineage-parent", ("shared prompt", "parent tail"))
        ),
        _write_jsonl(
            manual / "lineage-child.jsonl",
            _codex_records("zoo-lineage-child", ("shared prompt", "child tail"), parent="zoo-lineage-parent"),
        ),
        _write_jsonl(manual / "lineage-cycle-a.jsonl", _codex_records("zoo-cycle-a", ("cycle A",))),
        _write_jsonl(
            manual / "lineage-cycle-b.jsonl",
            _codex_records("zoo-cycle-b", ("cycle B",), parent="zoo-cycle-a", subagent=True),
        ),
        _write_jsonl(
            manual / "lineage-orphan.jsonl",
            _codex_records("zoo-orphan-head", ("unresolved parent",), parent="never-ingested"),
        ),
        _write_jsonl(manual / "grouped.jsonl", grouped_records),
        _write_jsonl(manual / "empty.jsonl", ({"type": "progress", "sessionId": "zoo-empty", "uuid": "empty-1"},)),
        _write_json(manual / "vintage-old.json", _chatgpt_payload("zoo-vintage-reorder", vintage_nodes)),
        _write_json(
            manual / "vintage-new.json", _chatgpt_payload("zoo-vintage-reorder", list(reversed(vintage_nodes)))
        ),
        _write_json(manual / "lifecycle-old.json", _chatgpt_payload("zoo-lifecycle-anchor-drift", lifecycle_nodes)),
        _write_json(
            manual / "lifecycle-new.json",
            _chatgpt_payload("zoo-lifecycle-anchor-drift", list(reversed(lifecycle_nodes))),
        ),
        _write_json(
            manual / "content-blocks-old.json",
            {
                "uuid": "zoo-content-blocks-vintage",
                "chat_messages": [{"uuid": "content-1", "sender": "human", "text": "same content"}],
            },
        ),
        _write_json(
            manual / "content-blocks-new.json",
            {
                "uuid": "zoo-content-blocks-vintage",
                "chat_messages": [
                    {"uuid": "content-1", "sender": "human", "content": [{"type": "text", "text": "same content"}]}
                ],
            },
        ),
        *write_claude_vintage_live_proof_pair(root),
        _write_json(
            manual / "attachment-metadata.json",
            {
                "uuid": "zoo-attachment-metadata",
                "chat_messages": [
                    {
                        "uuid": "attachment-1",
                        "sender": "human",
                        "text": "attachment metadata",
                        "attachments": [{"id": "zoo-no-bytes", "name": "metadata-only.txt", "mime_type": "text/plain"}],
                    }
                ],
            },
        ),
        _write_json(
            manual / "design.json",
            {
                "id": "zoo-design-session",
                "project": {"id": "zoo-project"},
                "messages": [
                    {
                        "uuid": "design-1",
                        "role": "assistant",
                        "content": {"contentBlocks": [{"type": "text", "text": "design response"}]},
                    }
                ],
            },
        ),
    )


def _write_generated_members(root: Path, *, indexes: tuple[int, ...] | None = None) -> tuple[Path, ...]:
    generated = root / "generated"
    specs = (
        CorpusSpec.for_provider(
            "codex",
            count=1,
            messages_min=48,
            messages_max=48,
            seed=31,
            style="tool-heavy",
            session_native_ids=("zoo-whale",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
        CorpusSpec.for_provider(
            "codex",
            count=1,
            messages_min=2,
            messages_max=2,
            seed=32,
            session_native_ids=("zoo-append-self",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
        CorpusSpec.for_provider(
            "codex",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=33,
            session_native_ids=("zoo-append-self",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
        CorpusSpec.for_provider(
            "codex",
            count=1,
            messages_min=2,
            messages_max=2,
            seed=34,
            session_native_ids=("zoo-append-opaque",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
        CorpusSpec.for_provider(
            "codex",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=35,
            session_native_ids=("zoo-append-opaque",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
        CorpusSpec.for_provider(
            "gemini",
            count=1,
            messages_min=2,
            messages_max=2,
            seed=36,
            style="demo-attachments",
            session_native_ids=("zoo-attachment-bytes",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
        CorpusSpec.for_provider(
            "antigravity",
            count=1,
            messages_min=2,
            messages_max=2,
            seed=37,
            session_native_ids=("zoo-06-00:zoo-06-00.md",),
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
    )
    paths: list[Path] = []
    for index in indexes or tuple(range(len(specs))):
        written = SyntheticCorpus.write_spec_artifacts(specs[index], generated, prefix=f"zoo-{index:02d}")
        paths.extend(written.files)
    return tuple(paths)


def _bind_durable_paths(
    manifest: tuple[PathologyZooMember, ...], wire_root: Path, hook_event_path: Path
) -> tuple[PathologyZooMember, ...]:
    return tuple(
        replace(
            member,
            durable_paths=(str(hook_event_path),)
            if member.member_id == "hook-event"
            else (str(wire_root / "generated" / "conversations" / "zoo-06-00:zoo-06-00.md.pb"),)
            if member.member_id == "non-stream-safe"
            else tuple(str(wire_root / raw_path) for raw_path in member.raw_paths),
        )
        for member in manifest
    )


def _validate_manifest(root: Path, manifest: tuple[PathologyZooMember, ...]) -> None:
    try:
        with sqlite3.connect(root / "index.db") as conn:
            session_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}
    except sqlite3.Error as exc:
        raise RuntimeError("pathology zoo has no queryable archive after production ingest") from exc
    missing = sorted({session_id for member in manifest for session_id in member.session_ids} - session_ids)
    if missing:
        raise RuntimeError(f"pathology zoo members missing after production ingest: {missing}")
    with sqlite3.connect(root / "source.db") as conn:
        for member in manifest:
            table = "raw_hook_events" if member.member_id == "hook-event" else "raw_sessions"
            missing_paths = [
                path
                for path in member.durable_paths
                if conn.execute(f"SELECT 1 FROM {table} WHERE source_path = ?", (path,)).fetchone() is None
            ]
            if missing_paths:
                raise RuntimeError(
                    f"pathology zoo durable {table} rows missing for {member.member_id}: {missing_paths}"
                )


def build_pathology_zoo(archive_root: Path) -> PathologyZoo:
    """Build the v0 zoo through the production source-to-archive harness."""
    wire_root = archive_root / "wire"
    _write_generated_members(wire_root, indexes=(0, 1, 3, 5, 6))
    _write_manual_members(wire_root)
    sources = [Source(name="pathology-zoo", path=wire_root), Source(name="antigravity", path=wire_root / "generated")]
    with patch(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        SyntheticAntigravityLanguageServerClient,
    ):
        asyncio.run(parse_sources_archive(archive_root, sources, parse_workers=1))
    _write_generated_members(wire_root, indexes=(2, 4))
    _write_jsonl(
        wire_root / "manual" / "lineage-cycle-z-a-update.jsonl",
        _codex_records("zoo-cycle-a", ("cycle A", "cycle A revised"), parent="zoo-cycle-b", subagent=True),
    )
    with patch(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        SyntheticAntigravityLanguageServerClient,
    ):
        asyncio.run(parse_sources_archive(archive_root, sources, parse_workers=1))
    hook_event_path = enqueue_hook_event(
        event_id="zoo-hook-event",
        provider="claude-code",
        event_type="PostToolUse",
        session_id="zoo-hook-parent",
        timestamp="2026-08-04T00:00:00Z",
        payload={"tool_name": "Bash", "tool_call_id": "zoo-hook-call"},
        root=archive_root / "hook-spool",
    )
    if drain_hook_event_spool(archive_root, root=archive_root / "hook-spool").acknowledged != 1:
        raise RuntimeError("pathology zoo hook event did not drain through the durable source route")
    manifest = _bind_durable_paths(PRODUCTION_PATHOLOGY_ZOO_MANIFEST, wire_root, hook_event_path)
    _validate_manifest(archive_root, manifest)
    return PathologyZoo(archive_root=archive_root, manifest=manifest)


def make_pathology_zoo_member_red(archive_root: Path, member_id: str) -> None:
    """Apply only the fixture mutation for one production manifest member."""
    try:
        mutation = _PATHOLOGY_ZOO_MUTATIONS[member_id]
    except KeyError as exc:
        raise ValueError(f"no fixture mutation for pathology-zoo member {member_id!r}") from exc
    mutation.apply(archive_root)


__all__ = ["PathologyZoo", "PathologyZooMutation", "build_pathology_zoo", "make_pathology_zoo_member_red"]
