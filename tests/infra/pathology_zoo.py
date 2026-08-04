"""Curated production-ingest fixture archive for known pathology shapes.

The zoo is test memory, not a second writer.  Every member starts as a wire
artifact and reaches SQLite only through :func:`parse_sources_archive`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources.hooks import drain_hook_event_spool, enqueue_hook_event


@dataclass(frozen=True, slots=True)
class PathologyZooMember:
    """One queryable pathology label and its motivating Beads evidence."""

    member_id: str
    pathology: str
    motivating_beads: tuple[str, ...]
    session_ids: tuple[str, ...]
    raw_paths: tuple[str, ...]
    archive_verification_checks: tuple[str, ...]
    durable_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PathologyZoo:
    """A materialized archive plus its manifest of intentional pathologies."""

    archive_root: Path
    manifest: tuple[PathologyZooMember, ...]

    def members_for(self, pathology: str) -> tuple[PathologyZooMember, ...]:
        return tuple(member for member in self.manifest if member.pathology == pathology)

    @property
    def canary_session_ids(self) -> tuple[str, ...]:
        """Every replayable zoo session supplied to the production canary selector."""
        return tuple(sorted({session_id for member in self.manifest for session_id in member.session_ids}))

    @property
    def archive_verification_checks(self) -> tuple[str, ...]:
        """Registry checks that consume the fixture's durable pathology evidence."""
        return tuple(sorted({check for member in self.manifest for check in member.archive_verification_checks}))


_CODEX = "codex-session"
_CLAUDE_AI = "claude-ai-export"
_CLAUDE_CODE = "claude-code-session"
_CHATGPT = "chatgpt-export"
_DESIGN = "claude-design-session"
_GEMINI = "aistudio-drive"


def pathology_zoo_manifest() -> tuple[PathologyZooMember, ...]:
    """Return the stable labels consumers should iterate rather than duplicate."""
    return (
        PathologyZooMember(
            "whale-component",
            "whale-component",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-whale",),
            ("generated/zoo-00-00.jsonl",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "append-self-describing",
            "append-revision-chain",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-append-self",),
            ("generated/zoo-01-00.jsonl", "generated/zoo-02-00.jsonl"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "append-opaque",
            "append-revision-chain",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-append-opaque",),
            ("generated/zoo-03-00.jsonl", "generated/zoo-04-00.jsonl"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "fork-prefix-tail",
            "fork-resume-prefix-tail-lineage",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-lineage-parent", f"{_CODEX}:zoo-lineage-child"),
            ("manual/lineage-parent.jsonl", "manual/lineage-child.jsonl"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "lineage-cycle",
            "lineage-cycle-candidate",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-cycle-a", f"{_CODEX}:zoo-cycle-b"),
            ("manual/lineage-cycle-a.jsonl", "manual/lineage-cycle-b.jsonl", "manual/lineage-cycle-z-a-update.jsonl"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "grouped-jsonl",
            "multi-session-raw",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-group-a", f"{_CLAUDE_CODE}:zoo-group-b"),
            ("manual/grouped.jsonl",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "quarantined-head",
            "quarantined-head-open-blocker",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-orphan-head",),
            ("manual/lineage-orphan.jsonl",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "empty-session",
            "genuinely-empty-session",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-empty",),
            ("manual/empty.jsonl",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "hook-event",
            "hook-event-raw",
            ("polylogue-yazae",),
            (),
            ("hook-spool/zoo-hook-event.json",),
            ("blob-refs-liveness",),
        ),
        PathologyZooMember(
            "claude-design",
            "claude-design-session-origin",
            ("polylogue-yazae",),
            (f"{_DESIGN}:zoo-design-session",),
            ("manual/design.json",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "vintage-reorder",
            "export-vintage-reorder",
            ("polylogue-yazae", "polylogue-slshy"),
            (f"{_CHATGPT}:zoo-vintage-reorder",),
            ("manual/vintage-old.json", "manual/vintage-new.json"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "content-blocks-vintage",
            "content-blocks-vintage",
            ("polylogue-yazae", "polylogue-0qfy"),
            (f"{_CLAUDE_AI}:zoo-content-blocks-vintage",),
            ("manual/content-blocks-old.json", "manual/content-blocks-new.json"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "lifecycle-anchor-drift",
            "lifecycle-anchor-drift",
            ("polylogue-yazae", "polylogue-uqwd"),
            (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
            ("manual/lifecycle-old.json", "manual/lifecycle-new.json"),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "non-stream-safe",
            "non-stream-safe-origin",
            ("polylogue-yazae",),
            ("antigravity-session:zoo-06-00:zoo-06-00.md",),
            ("generated",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "attachment-with-bytes",
            "attachment-with-acquired-bytes",
            ("polylogue-yazae",),
            (f"{_GEMINI}:zoo-attachment-bytes",),
            ("generated/zoo-05-00.json",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "attachment-without-bytes",
            "attachment-without-acquired-bytes",
            ("polylogue-yazae",),
            (f"{_CLAUDE_AI}:zoo-attachment-metadata",),
            ("manual/attachment-metadata.json",),
            ("corpus-absences",),
        ),
        PathologyZooMember(
            "events-sidecars",
            "session-events-and-sidecars",
            ("polylogue-yazae",),
            (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
            ("manual/lifecycle-old.json",),
            ("corpus-absences",),
        ),
    )


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
            origin="test.pathology-zoo",
            tags=("pathology-zoo",),
        ),
    )
    paths: list[Path] = []
    for index in indexes or tuple(range(len(specs))):
        spec = specs[index]
        written = SyntheticCorpus.write_spec_artifacts(spec, generated, prefix=f"zoo-{index:02d}")
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
    with sqlite3.connect(root / "index.db") as conn:
        cycle = conn.execute(
            "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
            (f"{_CODEX}:zoo-cycle-a",),
        ).fetchone()
        orphan = conn.execute(
            "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
            (f"{_CODEX}:zoo-orphan-head",),
        ).fetchone()
    if cycle != ("quarantined", None):
        raise RuntimeError(f"pathology zoo cycle link was not quarantined: {cycle}")
    if orphan != (None, None):
        raise RuntimeError(f"pathology zoo orphan link was not unresolved: {orphan}")


def build_pathology_zoo(archive_root: Path) -> PathologyZoo:
    """Build the v0 zoo through the production source-to-archive harness."""
    wire_root = archive_root / "wire"
    _write_generated_members(wire_root, indexes=(0, 1, 3, 5, 6))
    _write_manual_members(wire_root)
    sources = [Source(name="pathology-zoo", path=wire_root), Source(name="antigravity", path=wire_root / "generated")]
    asyncio.run(
        parse_sources_archive(
            archive_root,
            sources,
            parse_workers=1,
        )
    )
    _write_generated_members(wire_root, indexes=(2, 4))
    _write_jsonl(
        wire_root / "manual" / "lineage-cycle-z-a-update.jsonl",
        _codex_records("zoo-cycle-a", ("cycle A", "cycle A revised"), parent="zoo-cycle-b", subagent=True),
    )
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
    manifest = _bind_durable_paths(pathology_zoo_manifest(), wire_root, hook_event_path)
    _validate_manifest(archive_root, manifest)
    return PathologyZoo(archive_root=archive_root, manifest=manifest)


__all__ = [
    "PathologyZoo",
    "PathologyZooMember",
    "build_pathology_zoo",
    "pathology_zoo_manifest",
]
