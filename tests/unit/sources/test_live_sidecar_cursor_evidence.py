"""A ``tool-results/`` sidecar cursor may only advance over retained bytes.

A Claude Code sidecar leaves no index-tier trace: its single record anywhere
in the archive is the source-tier row. A cursor commit is a claim that the
bytes were consumed and their evidence retained, so for a sidecar path that
claim is checked against ``source.db`` rather than against the raw id the
batch believes it wrote (polylogue-6xqhp: an ops.db cursor at end of file,
``record_count`` 0, with no ``raw_sessions``/``raw_artifacts``/
``history_sidecars`` row naming the path -- bytes consumed, nothing retained,
and no route left that can re-read them).

Anti-vacuity: delete the ``_is_tool_result_sidecar_path`` branch from
``LiveBatchProcessor._source_tier_evidence_retained`` and
``test_sidecar_cursor_refuses_to_advance_without_source_tier_evidence`` goes
red -- the cursor lands at end of file with ``record_count`` 0, reproducing
the incident row exactly.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore

_SESSION_ID = "de99ba60-ccc4-43a7-b882-1dd1f2672db7"


def _owner_transcript_lines(sidecar: Path) -> list[dict[str, object]]:
    """A minimal Claude Code transcript whose tool result overflowed to ``sidecar``."""
    pointer = f"<persisted-output>Output too large. Full output saved to: {sidecar}</persisted-output>"
    return [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": _SESSION_ID,
            "timestamp": "2026-07-20T10:00:00Z",
            "message": {"role": "user", "content": "run it"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": _SESSION_ID,
            "timestamp": "2026-07-20T10:00:01Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "sessionId": _SESSION_ID,
            "timestamp": "2026-07-20T10:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": pointer}],
            },
        },
    ]


def _build_session_tree(root: Path, *, sidecar_names: tuple[str, ...]) -> tuple[Path, list[Path], list[Path]]:
    """Lay out ``projects/<proj>/`` with an owner transcript, subagents and sidecars."""
    project = root / "-realm-project-x"
    sidecar_dir = project / _SESSION_ID / "tool-results"
    subagents_dir = project / _SESSION_ID / "subagents"
    sidecar_dir.mkdir(parents=True)
    subagents_dir.mkdir(parents=True)

    sidecars = [sidecar_dir / name for name in sidecar_names]
    for index, sidecar in enumerate(sidecars):
        sidecar.write_text(f"persisted tool output {index}\n" * 64, encoding="utf-8")

    owner = project / f"{_SESSION_ID}.jsonl"
    owner.write_text(
        "\n".join(json.dumps(line) for line in _owner_transcript_lines(sidecars[0])) + "\n",
        encoding="utf-8",
    )

    # Subagent transcripts carrying only file-history envelopes: real Claude
    # Code output that parses to no conversational evidence, which is the
    # owner-side failure the incident batch recorded.
    subagents = []
    for name in ("agent-first", "agent-second"):
        subagent = subagents_dir / f"{name}.jsonl"
        subagent.write_text(
            json.dumps(
                {
                    "type": "file-history-snapshot",
                    "uuid": f"{name}-1",
                    "sessionId": _SESSION_ID,
                    "timestamp": "2026-07-20T10:00:03Z",
                    "snapshot": {"trackedFileBackups": {}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        subagents.append(subagent)
    return owner, subagents, sidecars


def _make_processor(workspace_env: dict[str, Path], root: Path) -> tuple[Polylogue, CursorStore, LiveBatchProcessor]:
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "index.db")
    cursor = CursorStore(workspace_env["data_root"] / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    return archive, cursor, processor


def _query_source_paths(archive_root: Path, sql: str) -> set[str]:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute(sql)}
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_sidecar_cursor_advances_only_beside_its_own_source_row(
    workspace_env: dict[str, Path],
) -> None:
    """Owner transcripts failing in the same batch leave the sidecar's evidence intact."""
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    owner, subagents, sidecars = _build_session_tree(root, sidecar_names=("bsq814i68.txt",))
    sidecar = sidecars[0]
    archive, cursor, processor = _make_processor(workspace_env, root)
    try:
        await processor.ingest_files([sidecar, owner, *subagents], emit_event=False)

        # The owner-side transcripts are the batch's parse failures.
        failed_owner_paths = _query_source_paths(
            workspace_env["archive_root"],
            "SELECT source_path FROM raw_sessions WHERE parse_error IS NOT NULL",
        )
        assert failed_owner_paths == {str(subagent) for subagent in subagents}

        # The sidecar kept its own source-tier row, so its cursor may advance.
        assert str(sidecar) in _query_source_paths(
            workspace_env["archive_root"], "SELECT source_path FROM raw_sessions"
        )
        assert str(sidecar) in _query_source_paths(
            workspace_env["archive_root"], "SELECT source_path FROM raw_artifacts"
        )
        record = cursor.get_record(sidecar)
        assert record is not None
        assert record.byte_offset == sidecar.stat().st_size
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_sidecar_cursor_refuses_to_advance_without_source_tier_evidence(
    workspace_env: dict[str, Path],
) -> None:
    """A raw id the batch reports but source.db does not hold is not evidence."""
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    owner, subagents, sidecars = _build_session_tree(root, sidecar_names=("bsq814i68.txt", "b0br6ndl9.txt"))
    retained, unretained = sidecars
    archive, cursor, processor = _make_processor(workspace_env, root)
    try:
        # Ingest everything except ``unretained``: source.db ends up with a
        # row for every path in the session tree but that one.
        await processor.ingest_files([retained, owner, *subagents], emit_event=False)
        assert str(unretained) not in _query_source_paths(
            workspace_env["archive_root"], "SELECT source_path FROM raw_sessions"
        )

        # The commit route is handed the raw id a batch believed it wrote.
        bytes_read = processor._record_full_cursor(unretained, raw_fingerprint="raw-id-never-retained")

        assert bytes_read == 0
        assert cursor.get_record(unretained) is None
        assert processor._last_cursor_write_stale is True
        debt = [
            entry
            for entry in cursor.list_convergence_debt(limit=50)
            if entry.subject_id == str(unretained) and entry.stage == "raw_parse_recovery"
        ]
        assert debt, "a refused cursor advance must leave retryable convergence debt"

        # Same call, same carried raw id, for the sidecar source.db does hold:
        # the check is a confirmation, not a blanket refusal.
        retained_bytes_read = processor._record_full_cursor(retained, raw_fingerprint="raw-id-never-retained")
        assert retained_bytes_read > 0
        retained_record = cursor.get_record(retained)
        assert retained_record is not None
        assert retained_record.byte_offset == retained.stat().st_size
    finally:
        await archive.close()
