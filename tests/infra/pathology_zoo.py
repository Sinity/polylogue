"""Curated production-ingest fixture archive for known pathology shapes.

The zoo is test memory, not a second writer.  Every member starts as a wire
artifact and reaches SQLite only through :func:`parse_sources_archive`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus


@dataclass(frozen=True, slots=True)
class PathologyZooMember:
    """One queryable pathology label and its motivating Beads evidence."""

    member_id: str
    pathology: str
    motivating_beads: tuple[str, ...]
    session_ids: tuple[str, ...]
    raw_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PathologyZooIntegrationGap:
    """A consumer named by the zoo design but not yet extensible in this tree."""

    consumer: str
    follow_up: str


@dataclass(frozen=True, slots=True)
class PathologyZoo:
    """A materialized archive plus its manifest of intentional pathologies."""

    archive_root: Path
    manifest: tuple[PathologyZooMember, ...]

    def members_for(self, pathology: str) -> tuple[PathologyZooMember, ...]:
        return tuple(member for member in self.manifest if member.pathology == pathology)


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
            ("generated/codex-00.jsonl",),
        ),
        PathologyZooMember(
            "append-self-describing",
            "append-revision-chain",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-append-self",),
            ("generated/codex-01.jsonl", "generated/codex-02.jsonl"),
        ),
        PathologyZooMember(
            "append-opaque",
            "append-revision-chain",
            ("polylogue-yazae",),
            (f"{_CODEX}:zoo-append-opaque",),
            ("generated/codex-03.jsonl", "generated/codex-04.jsonl"),
        ),
        PathologyZooMember(
            "fork-prefix-tail",
            "fork-resume-prefix-tail-lineage",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-lineage-parent", f"{_CLAUDE_CODE}:zoo-lineage-child"),
            ("manual/lineage.jsonl",),
        ),
        PathologyZooMember(
            "lineage-cycle",
            "lineage-cycle-candidate",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-cycle-a", f"{_CLAUDE_CODE}:zoo-cycle-b"),
            ("manual/lineage.jsonl",),
        ),
        PathologyZooMember(
            "grouped-jsonl",
            "multi-session-raw",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-group-a", f"{_CLAUDE_CODE}:zoo-group-b"),
            ("manual/grouped.jsonl",),
        ),
        PathologyZooMember(
            "quarantined-head",
            "quarantined-head-open-blocker",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-orphan-head",),
            ("manual/lineage.jsonl",),
        ),
        PathologyZooMember(
            "empty-session",
            "genuinely-empty-session",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-empty",),
            ("manual/empty.jsonl",),
        ),
        PathologyZooMember(
            "hook-event",
            "hook-event-raw",
            ("polylogue-yazae",),
            (f"{_CLAUDE_CODE}:zoo-hook-event",),
            ("manual/hook-event.jsonl",),
        ),
        PathologyZooMember(
            "claude-design",
            "claude-design-session-origin",
            ("polylogue-yazae",),
            (f"{_DESIGN}:zoo-design-session",),
            ("manual/design.json",),
        ),
        PathologyZooMember(
            "vintage-reorder",
            "export-vintage-reorder",
            ("polylogue-yazae", "polylogue-slshy"),
            (f"{_CHATGPT}:zoo-vintage-reorder",),
            ("manual/vintage-old.json", "manual/vintage-new.json"),
        ),
        PathologyZooMember(
            "content-blocks-vintage",
            "content-blocks-vintage",
            ("polylogue-yazae", "polylogue-0qfy"),
            (f"{_CLAUDE_AI}:zoo-content-blocks-vintage",),
            ("manual/content-blocks-old.json", "manual/content-blocks-new.json"),
        ),
        PathologyZooMember(
            "lifecycle-anchor-drift",
            "lifecycle-anchor-drift",
            ("polylogue-yazae", "polylogue-uqwd"),
            (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
            ("manual/lifecycle-old.json", "manual/lifecycle-new.json"),
        ),
        PathologyZooMember(
            "non-stream-safe",
            "non-stream-safe-origin",
            ("polylogue-yazae",),
            ("antigravity-session:zoo-06-00:zoo-06-00.md",),
            ("generated/brain/zoo-06-00/zoo-06-00.md.metadata.json",),
        ),
        PathologyZooMember(
            "attachment-with-bytes",
            "attachment-with-acquired-bytes",
            ("polylogue-yazae",),
            (f"{_GEMINI}:zoo-attachment-bytes",),
            ("generated/gemini-00.json",),
        ),
        PathologyZooMember(
            "attachment-without-bytes",
            "attachment-without-acquired-bytes",
            ("polylogue-yazae",),
            (f"{_CLAUDE_AI}:zoo-attachment-metadata",),
            ("manual/attachment-metadata.json",),
        ),
        PathologyZooMember(
            "events-sidecars",
            "session-events-and-sidecars",
            ("polylogue-yazae",),
            (f"{_CHATGPT}:zoo-lifecycle-anchor-drift",),
            ("manual/lifecycle-old.json",),
        ),
    )


def pathology_zoo_integration_gaps() -> tuple[PathologyZooIntegrationGap, ...]:
    """Keep absent consumer hooks visible until their owning lanes expose one."""
    return (
        PathologyZooIntegrationGap(
            "polylogue-t0m73",
            "No registry pytest binding exists in this checkout; iterate pathology_zoo_manifest() when the red-twin extension point lands.",
        ),
        PathologyZooIntegrationGap(
            "polylogue-0x7nh",
            "No differ canary-set extension point exists in this checkout; include pathology_zoo_manifest() members when the canary registry lands.",
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
    lineage_records = (
        _code_record("zoo-lineage-parent", "parent-1", "user", "shared prompt"),
        _code_record("zoo-lineage-parent", "parent-2", "assistant", "parent tail", parent="parent-1"),
        _code_record("zoo-lineage-child", "parent-1", "user", "shared prompt"),
        _code_record("zoo-lineage-child", "child-2", "assistant", "child tail", parent="parent-1"),
        _code_record("zoo-cycle-a", "cycle-a", "user", "cycle A", parent="cycle-b"),
        _code_record("zoo-cycle-b", "cycle-b", "assistant", "cycle B", parent="cycle-a"),
        _code_record("zoo-orphan-head", "orphan-1", "assistant", "unresolved parent", parent="missing-parent"),
    )
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
        _write_jsonl(manual / "lineage.jsonl", lineage_records),
        _write_jsonl(manual / "grouped.jsonl", grouped_records),
        _write_jsonl(manual / "empty.jsonl", ({"type": "progress", "sessionId": "zoo-empty", "uuid": "empty-1"},)),
        _write_jsonl(
            manual / "hook-event.jsonl", (_code_record("zoo-hook-event", "hook-1", "user", "hook-backed transcript"),)
        ),
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


def _write_generated_members(root: Path) -> tuple[Path, ...]:
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
    for index, spec in enumerate(specs):
        written = SyntheticCorpus.write_spec_artifacts(spec, generated, prefix=f"zoo-{index:02d}")
        paths.extend(written.files)
    return tuple(paths)


def _validate_manifest(root: Path, manifest: tuple[PathologyZooMember, ...]) -> None:
    try:
        with sqlite3.connect(root / "index.db") as conn:
            session_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}
    except sqlite3.Error as exc:
        raise RuntimeError("pathology zoo has no queryable archive after production ingest") from exc
    missing = sorted({session_id for member in manifest for session_id in member.session_ids} - session_ids)
    if missing:
        raise RuntimeError(f"pathology zoo members missing after production ingest: {missing}")


def build_pathology_zoo(archive_root: Path) -> PathologyZoo:
    """Build the v0 zoo through the production source-to-archive harness."""
    wire_root = archive_root / "wire"
    _write_generated_members(wire_root)
    _write_manual_members(wire_root)
    asyncio.run(
        parse_sources_archive(
            archive_root,
            [Source(name="pathology-zoo", path=wire_root), Source(name="antigravity", path=wire_root / "generated")],
            parse_workers=1,
        )
    )
    manifest = pathology_zoo_manifest()
    _validate_manifest(archive_root, manifest)
    return PathologyZoo(archive_root=archive_root, manifest=manifest)


__all__ = [
    "PathologyZoo",
    "PathologyZooIntegrationGap",
    "PathologyZooMember",
    "build_pathology_zoo",
    "pathology_zoo_integration_gaps",
    "pathology_zoo_manifest",
]
