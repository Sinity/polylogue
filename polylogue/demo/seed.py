"""Seed the deterministic demo archive without daemon scheduling."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID, build_demo_corpus_specs, seed_demo_user_overlays
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync

from .constructs import evaluate_demo_constructs
from .models import DemoSeedResult

DEMO_SOURCE_DIRNAME = "demo-fixture-world-source"


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    """Temporarily run relative-source ingestion from *path*."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def materialize_demo_source(root: Path, *, force: bool = False) -> Path:
    """Write deterministic demo source artifacts under ``root``."""

    source_root = root / DEMO_SOURCE_DIRNAME
    if force and source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    SyntheticCorpus.write_specs_artifacts(
        build_demo_corpus_specs(),
        source_root,
        prefix="demo",
        index_width=2,
    )
    _write_demo_temporary_sources(source_root)
    _write_demo_lineage_sources(source_root)
    return source_root


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_demo_temporary_sources(source_root: Path) -> None:
    """Write a Claude.ai temporary-chat fixture through the parser path."""

    _write_json(
        source_root / "claude-ai" / "temporary-demo.json",
        {
            "uuid": "demo-temporary-claude-ai",
            "name": "Temporary demo context check",
            "is_temporary": True,
            "created_at": "2026-07-04T09:50:00Z",
            "updated_at": "2026-07-04T09:50:03Z",
            "chat_messages": [
                {
                    "uuid": "temporary-u0",
                    "sender": "human",
                    "text": "Please answer without retaining this temporary context.",
                    "created_at": "2026-07-04T09:50:01Z",
                    "updated_at": "2026-07-04T09:50:01Z",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please answer without retaining this temporary context.",
                        }
                    ],
                },
                {
                    "uuid": "temporary-a1",
                    "sender": "assistant",
                    "text": "The archive keeps the evidence but marks this session temporary.",
                    "created_at": "2026-07-04T09:50:02Z",
                    "updated_at": "2026-07-04T09:50:03Z",
                    "content": [
                        {
                            "type": "text",
                            "text": "The archive keeps the evidence but marks this session temporary.",
                        },
                        {
                            "type": "token_budget",
                            "remaining": 2048,
                        },
                    ],
                },
            ],
        },
    )


def _codex_session_meta(
    session_id: str,
    *,
    timestamp: str,
    forked_from_id: str | None = None,
    subagent_role: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"id": session_id, "timestamp": timestamp}
    if forked_from_id is not None:
        payload["forked_from_id"] = forked_from_id
    if subagent_role is not None and forked_from_id is not None:
        payload["source"] = {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": forked_from_id,
                    "depth": 1,
                    "agent_role": subagent_role,
                }
            }
        }
    return {"type": "session_meta", "payload": payload}


def _codex_message(
    message_id: str,
    role: str,
    timestamp: str,
    content: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {
            "id": message_id,
            "type": "message",
            "role": role,
            "timestamp": timestamp,
            "content": content,
        },
    }


def _input_text(text: str) -> dict[str, object]:
    return {"type": "input_text", "text": text}


def _output_text(text: str) -> dict[str, object]:
    return {"type": "output_text", "text": text}


def _write_demo_lineage_sources(source_root: Path) -> None:
    """Write explicit Codex parent/fork/subagent fixtures for topology demos."""

    parent_id = "demo-lineage-parent"
    fork_id = "demo-lineage-fork"
    subagent_id = "demo-lineage-subagent"
    base_user = "Map the demo lineage base context."
    base_assistant = "I have the base context and can branch the analysis."

    _write_jsonl(
        source_root / "codex" / "lineage-parent.jsonl",
        (
            _codex_session_meta(parent_id, timestamp="2026-07-04T10:00:00Z"),
            _codex_message("parent-u0", "user", "2026-07-04T10:00:01Z", [_input_text(base_user)]),
            _codex_message("parent-a1", "assistant", "2026-07-04T10:00:02Z", [_output_text(base_assistant)]),
            _codex_message(
                "parent-a2",
                "assistant",
                "2026-07-04T10:00:03Z",
                [
                    _output_text("Delegating a topology check to a focused subagent."),
                    {
                        "type": "tool_use",
                        "id": "task-demo-lineage",
                        "name": "Task",
                        "input": {
                            "subagent_type": "Explore",
                            "prompt": "Inspect the demo lineage child and report caveats.",
                            "child_session_id": "codex-session:demo-lineage-subagent",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "task-demo-lineage",
                        "content": "Subagent completed. Session: codex-session:demo-lineage-subagent",
                    },
                ],
            ),
        ),
    )
    _write_jsonl(
        source_root / "codex" / "lineage-fork.jsonl",
        (
            _codex_session_meta(fork_id, timestamp="2026-07-04T10:01:00Z", forked_from_id=parent_id),
            _codex_message("fork-u0", "user", "2026-07-04T10:01:01Z", [_input_text(base_user)]),
            _codex_message("fork-a1", "assistant", "2026-07-04T10:01:02Z", [_output_text(base_assistant)]),
            _codex_message(
                "fork-u2",
                "user",
                "2026-07-04T10:01:03Z",
                [_input_text("Now take the forked branch and audit construct validity.")],
            ),
            _codex_message(
                "fork-a3",
                "assistant",
                "2026-07-04T10:01:04Z",
                [_output_text("The fork diverges into demo corpus construct checks.")],
            ),
        ),
    )
    _write_jsonl(
        source_root / "codex" / "lineage-subagent.jsonl",
        (
            _codex_session_meta(
                subagent_id,
                timestamp="2026-07-04T10:02:00Z",
                forked_from_id=parent_id,
                subagent_role="Explore",
            ),
            _codex_message(
                "subagent-a0",
                "assistant",
                "2026-07-04T10:02:01Z",
                [_output_text("Subagent report: lineage fixture has a parent, a fork, and a resolved child link.")],
            ),
        ),
    )


def _materialize_session_insights(archive_root: Path, session_ids: list[str]) -> None:
    """Build the session-profile insight read models for *session_ids*.

    ``parse_sources_archive`` writes the ``sessions``/``messages`` tree but does
    not materialize the derived insight tables (``session_profiles`` and
    siblings); the daemon convergence path normally does that in a separate
    stage. The no-daemon demo seed must run the same rebuild so that the
    postmortem / session-digest read surfaces resolve against a populated demo
    archive instead of an empty one. Passing ``session_ids`` makes
    ``rebuild_session_insights_sync`` commit internally and skip the full-table
    delete/rebuild path.
    """

    if not session_ids:
        return
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.row_factory = sqlite3.Row
        rebuild_session_insights_sync(conn, session_ids=session_ids)
    finally:
        conn.close()


# Deterministic per-assistant-message Opus usage injected into the demo
# claude-code session so cost surfaces (`analyze --postmortem` top session,
# cost rollups) render a believable bundle on the demo archive instead of $0.
# The demo corpus is fully synthetic; the synthetic generator does not emit
# token usage, and adding it there would ripple every synthetic test snapshot.
# Scoping the injection to the one demo session keeps it ripple-free.
_DEMO_USAGE_MODEL = "claude-opus-4-8"
_DEMO_USAGE = {
    "input_tokens": 8000,
    "output_tokens": 1500,
    "cache_read_tokens": 40000,
    "cache_write_tokens": 6000,
}


def _inject_demo_session_usage(archive_root: Path) -> None:
    """Set deterministic Opus token usage on the demo claude-code assistant turns.

    Cost is materialized into ``session_profiles.total_cost_usd`` from the
    per-message token columns, so this must run *before*
    ``_materialize_session_insights``. Demo-scoped by session id: no synthetic
    generator change, no test-snapshot ripple.
    """

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.execute(
            "UPDATE messages SET model_name = ?, input_tokens = ?, output_tokens = ?, "
            "cache_read_tokens = ?, cache_write_tokens = ? "
            "WHERE session_id = ? AND role = 'assistant'",
            (
                _DEMO_USAGE_MODEL,
                _DEMO_USAGE["input_tokens"],
                _DEMO_USAGE["output_tokens"],
                _DEMO_USAGE["cache_read_tokens"],
                _DEMO_USAGE["cache_write_tokens"],
                DEMO_CLAUDE_CODE_SESSION_ID,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# Canonical repo identity for the demo claude-code session so the postmortem
# `repos_touched` metric renders a believable project instead of an empty list.
# The synthetic demo corpus emits no cwd/repo attribution, so profile
# materialization derives no repo names; this is set *after* materialization
# (which would otherwise overwrite it with the empty attribution result) and is
# scoped to the one demo session.
_DEMO_REPO_NAMES = '["polylogue"]'


def _inject_demo_session_repos(archive_root: Path) -> None:
    """Set a canonical repo name on the demo claude-code session profile."""

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.execute(
            "UPDATE session_profiles SET repo_names_json = ? WHERE session_id = ?",
            (_DEMO_REPO_NAMES, DEMO_CLAUDE_CODE_SESSION_ID),
        )
        conn.commit()
    finally:
        conn.close()


def demo_source_specs(source_root: Path) -> list[Source]:
    """Return relative source specs for the materialized demo world."""

    return [
        Source(name="chatgpt", path=Path("chatgpt")),
        Source(name="claude-ai", path=Path("claude-ai")),
        Source(name="claude-code", path=Path("claude-code")),
        Source(name="codex", path=Path("codex")),
        Source(name="gemini", path=Path("gemini")),
    ]


async def seed_demo_archive(
    archive_root: Path,
    *,
    force: bool = False,
    with_overlays: bool = False,
) -> DemoSeedResult:
    """Materialize, ingest, and optionally overlay the deterministic demo archive."""

    source_root = materialize_demo_source(archive_root, force=force)
    with _pushd(source_root):
        result = await parse_sources_archive(archive_root, demo_source_specs(source_root))

    session_ids = sorted(result.processed_ids)
    _inject_demo_session_usage(archive_root)
    _materialize_session_insights(archive_root, session_ids)
    _inject_demo_session_repos(archive_root)

    overlay = seed_demo_user_overlays(archive_root) if with_overlays else None
    construct_coverage = evaluate_demo_constructs(archive_root)
    return DemoSeedResult(
        archive_root=archive_root,
        source_root=source_root,
        session_count=int(result.counts["sessions"]),
        message_count=int(result.counts["messages"]),
        session_ids=tuple(sorted(result.processed_ids)),
        overlays_seeded=overlay is not None,
        assertion_count=len(overlay.assertion_ids) if overlay else 0,
        construct_coverage=construct_coverage,
    )


__all__ = ["DEMO_SOURCE_DIRNAME", "demo_source_specs", "materialize_demo_source", "seed_demo_archive"]
