"""Live intake publishes the retained-replay interpretation of the same bytes.

Each test drives the production live batch owner against a synthetic Codex
session and inspects the stored archive rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.parse_prefetch import LiveParseStage
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, WatchSource
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _codex_lines(native_id: str, messages: tuple[tuple[str, str], ...], *, meta: bool = True) -> bytes:
    rows: list[dict[str, object]] = (
        [{"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}] if meta else []
    )
    for message_id, text in messages:
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": message_id,
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _ingest(archive_root: Path, source: Path, *, parse_stage: LiveParseStage | None = None) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    processor = LiveBatchProcessor(
        Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
        (WatchSource(name="codex", root=source.parent),),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=_PARSER_FINGERPRINT,
        parse_stage=parse_stage,
        read_snapshot=open_operation_read,
    )
    metrics = asyncio.run(processor.ingest_files([source], emit_event=False))
    assert metrics.failed_file_count == 0
    assert metrics.succeeded_file_count == 1


def _titles(archive_root: Path) -> list[tuple[str, str | None]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return [(str(row[0]), row[1]) for row in conn.execute("SELECT title, title_source FROM sessions")]


def test_live_append_keeps_the_chain_title_winner(tmp_path: Path) -> None:
    """A tail append must not replace the title a full replay would store.

    Anti-vacuity: dropping the aggregate title carry in the tail-merge write
    (``apply_raw_revision_replay``) stores the appended chunk's own first
    prompt, ``"later prompt"``; dropping live enrichment stores the native id.
    """
    source = tmp_path / "sessions" / "chain.jsonl"
    source.parent.mkdir()
    first = _codex_lines("chain-session", (("m0", "opening prompt"),))
    source.write_bytes(first)
    archive_root = tmp_path / "archive"
    _ingest(archive_root, source)
    assert _titles(archive_root) == [("opening prompt", "heuristic")]

    source.write_bytes(first + _codex_lines("chain-session", (("m1", "later prompt"),), meta=False))
    _ingest(archive_root, source)

    assert _titles(archive_root) == [("opening prompt", "heuristic")]
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_prewarm_seals_existing_retained_members_outside_the_writer(tmp_path: Path) -> None:
    """A pending live path carries sealed carriers for its prior raws.

    Anti-vacuity: without ``prepare_live_retained_raws`` the stage hands the
    writer no retained member, and the writer must reparse the prior raw
    under its lease. The carrier is also bound to the raw's descriptor, so a
    changed descriptor stops ``current`` from accepting it.
    """
    source = tmp_path / "sessions" / "revised.jsonl"
    source.parent.mkdir()
    source.write_bytes(_codex_lines("revised-session", (("m0", "first draft"),)))
    archive_root = tmp_path / "archive"
    _ingest(archive_root, source)
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        [prior_raw_id] = [str(row[0]) for row in archive.source_connection.execute("SELECT raw_id FROM raw_sessions")]

    source.write_bytes(_codex_lines("revised-session", (("m0", "second draft"),)))
    stage = LiveParseStage(max_workers=1, shard_directory=tmp_path / "parse-shards")
    try:
        stage.warm_paths(
            [(str(source), Provider.CODEX, True)], archive_root=archive_root, read_snapshot=open_operation_read
        )
        prepared = stage.pop_path(str(source), blob_hash=hashlib.sha256(source.read_bytes()).hexdigest())
        assert prepared is not None and prepared.error is None and not prepared.deferred
        [current] = list(prepared.iter_sessions())
        assert current.title == "second draft"
        retained = stage.take_retained_path(str(source))
        assert set(retained) == {prior_raw_id}
        member = retained[prior_raw_id]
        with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
            assert member.current(archive)
        [prior] = list(member.artifact.session_sequence())
        assert prior.title == "first draft"
        member.discard()
        prepared.discard()
    finally:
        stage.shutdown()


def test_live_claude_code_intake_uses_retained_index_titles_parsed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live transcripts take the curated retained index title, as replay does.

    Anti-vacuity: without worker enrichment both sessions keep the heuristic
    first-prompt title; without the content-addressed parse memo the shared
    index is reparsed for every transcript (``parses`` grows with files).
    """
    import polylogue.sources.parsers.claude.index as claude_index
    import polylogue.sources.retained_assembly as retained_assembly
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    project = tmp_path / "live" / ".claude" / "projects" / "-synthetic-project"
    project.mkdir(parents=True)
    transcripts: list[Path] = []
    entries: list[dict[str, object]] = []
    for index in range(2):
        session_id = f"aaaaaaaa-1111-2222-3333-44444444444{index}"
        transcript = project / f"{session_id}.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": f"u{index}",
                    "sessionId": session_id,
                    "timestamp": "2026-07-20T10:00:00.000Z",
                    "message": {"role": "user", "content": f"prompt {index}"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        transcripts.append(transcript)
        entries.append({"sessionId": session_id, "fullPath": str(transcript), "summary": f"Curated {index}"})
    index_path = project / "sessions-index.json"
    index_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    parses = 0
    original_parse = claude_index.parse_sessions_index_bytes

    def counting_parse(payload: bytes) -> object:
        nonlocal parses
        parses += 1
        return original_parse(payload)

    monkeypatch.setattr(claude_index, "parse_sessions_index_bytes", counting_parse)
    retained_assembly._parsed_retained_cache.clear()
    watch = WatchSource(
        name="claude-code",
        root=project.parent,
        suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
    )

    def ingest(paths: list[Path]) -> None:
        processor = LiveBatchProcessor(
            Polylogue(archive_root=archive_root, db_path=archive_root / "index.db"),
            (watch,),
            cursor=CursorStore(archive_root / "index.db"),
            parser_fingerprint=_PARSER_FINGERPRINT,
            read_snapshot=open_operation_read,
        )
        metrics = asyncio.run(processor.ingest_files(paths, emit_event=False))
        assert metrics.failed_file_count == 0

    ingest([index_path])
    ingest(transcripts)

    assert sorted(_titles(archive_root)) == [("Curated 0", "origin"), ("Curated 1", "origin")]
    assert parses == 1
