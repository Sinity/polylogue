"""Codex title evidence is acquisition-bound and survives retained replay."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider, TitleSource
from polylogue.sources.live import WatchSource
from tests.infra.live_batch import prepared_live_batch_processor
from tests.infra.retained_replay import publish_retained_payload


def _codex_stream(session_id: str, text: str) -> bytes:
    rows = [
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
    ]
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def _title(archive: Path) -> tuple[str | None, str | None]:
    with sqlite3.connect(archive / "index.db") as conn:
        row = conn.execute("SELECT title, title_source FROM sessions").fetchone()
    assert row is not None
    return row[0], row[1]


def test_retained_replay_does_not_discover_unacquired_history(tmp_path: Path) -> None:
    session_id = "aaaa1111-2222-3333-4444-555566667777"
    content = _codex_stream(session_id, "opening prompt from acquired bytes")
    rollout = tmp_path / ".codex" / "sessions" / "2026" / f"rollout-{session_id}.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_bytes(content)
    (rollout.parents[2] / "history.jsonl").write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "ambient title must be ignored"}) + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive"
    asyncio.run(
        publish_retained_payload(
            archive,
            provider=Provider.CODEX,
            payload=content,
            source_path=str(rollout),
            acquired_at_ms=1,
        )
    )

    assert _title(archive) == ("opening prompt from acquired bytes", TitleSource.HEURISTIC.value)


def test_live_acquisition_carries_codex_history_title_into_retained_replay(tmp_path: Path) -> None:
    async def scenario() -> None:
        session_id = "bbbb1111-2222-3333-4444-555566667777"
        content = _codex_stream(session_id, "opening prompt from acquired bytes")
        codex_root = tmp_path / ".codex"
        rollout = codex_root / "sessions" / "2026" / f"rollout-{session_id}.jsonl"
        rollout.parent.mkdir(parents=True)
        rollout.write_bytes(content)
        index = codex_root / "session_index.jsonl"
        index.write_text(json.dumps({"id": session_id}) + "\n", encoding="utf-8")
        history = codex_root / "history.jsonl"
        history.write_text(
            json.dumps({"session_id": session_id, "ts": 1, "text": "Acquired title"}) + "\n",
            encoding="utf-8",
        )
        import polylogue.sources.live.watcher as live_watcher

        archive = tmp_path / "archive"
        async with prepared_live_batch_processor(
            archive,
            (WatchSource(name="codex-state", root=codex_root),),
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        ) as processor:
            await processor.ingest_files([index, history], emit_event=False)
        await publish_retained_payload(
            archive,
            provider=Provider.CODEX,
            payload=content,
            source_path=str(rollout),
            acquired_at_ms=1,
        )
        assert _title(archive) == ("Acquired title", TitleSource.ORIGIN.value)

    asyncio.run(scenario())
