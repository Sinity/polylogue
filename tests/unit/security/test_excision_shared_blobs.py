"""Excising a session forgets that session, not the blobs it shares (polylogue-bgnxh).

Blobs are content-addressed: two Claude Code sessions whose tool output
overflowed into byte-identical ``tool-results/`` sidecars own one blob hash.
Every test here drives production acquisition (``LiveBatchProcessor``) for two
such sessions, A and B, where A also overflowed a second output nobody else
has, then excises A through ``apply_session_excision``.

Anti-vacuity: marking every hash A's rows named (the rule before this change)
marks the shared sidecar hash, so B's later sidecar with the same bytes is
refused (its tool result falls back to the preview) and the shared-hash
assertions fail. Marking nothing leaves A's own sidecar hash unmarked and A's
transcript re-admissible.

Synthetic fixtures only: invented tool output, invented paths.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.core.enums import BlockType
from polylogue.pipeline.ids import session_content_hash
from polylogue.security.excision import apply_session_excision
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

_SESSION_A = "5c3d1e40-0000-4000-8000-00000000a001"
_SESSION_B = "5c3d1e40-0000-4000-8000-00000000b002"
_SHARED_TEXT = "zz_shared_build_log_line\n" * 400
_A_ONLY_TEXT = "zz_a_only_secret_output\n" * 400


def _overflow_envelope(sidecar: Path) -> str:
    return (
        "<persisted-output>\n"
        f"Output too large (9.8KB). Full output saved to: {sidecar}\n\n"
        "Preview (first 2KB):\nshort preview text without the payload"
    )


def _exchange(
    prefix: str, session_id: str, tool_use_id: str, result_text: str, *, parent: str | None, minute: int
) -> list[dict[str, object]]:
    stamp = f"2026-07-20T10:{minute:02d}:0"
    return [
        {
            "type": "user",
            "uuid": f"{prefix}-u1",
            "parentUuid": parent,
            "sessionId": session_id,
            "timestamp": f"{stamp}0Z",
            "message": {"role": "user", "content": "run it"},
        },
        {
            "type": "assistant",
            "uuid": f"{prefix}-a1",
            "parentUuid": f"{prefix}-u1",
            "sessionId": session_id,
            "timestamp": f"{stamp}1Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": f"{prefix}-u2",
            "parentUuid": f"{prefix}-a1",
            "sessionId": session_id,
            "timestamp": f"{stamp}2Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result_text}],
            },
        },
    ]


def _session_tree(root: Path, session_id: str, outputs: list[tuple[str, str]]) -> dict[str, Path]:
    """One transcript whose tool results each overflowed into their own sidecar."""
    project = root / f"-realm-project-{session_id[-4:]}"
    tool_results = project / session_id / "tool-results"
    tool_results.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    parent: str | None = None
    for minute, (tool_use_id, text) in enumerate(outputs):
        sidecar = tool_results / f"{tool_use_id}.txt"
        sidecar.write_text(text, encoding="utf-8")
        paths[tool_use_id] = sidecar
        records.extend(
            _exchange(tool_use_id, session_id, tool_use_id, _overflow_envelope(sidecar), parent=parent, minute=minute)
        )
        parent = f"{tool_use_id}-u2"
    transcript = project / f"{session_id}.jsonl"
    transcript.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    paths["transcript"] = transcript
    return paths


async def _ingest(workspace_env: dict[str, Path], root: Path, files: list[Path], *, cursor_name: str) -> None:
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "index.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),),
        cursor=CursorStore(workspace_env["data_root"] / cursor_name),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        await processor.ingest_files(files, emit_event=False)
    finally:
        await archive.close()


def _sha(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _session_row(archive_root: Path, native_id: str) -> tuple[str, str, str] | None:
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT session_id, raw_id, lower(hex(content_hash)) FROM sessions WHERE native_id = ?", (native_id,)
        ).fetchone()
    return None if row is None else (str(row[0]), str(row[1]), str(row[2]))


def _raw_hash(archive_root: Path, source_path: Path) -> bytes | None:
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT blob_hash FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms DESC LIMIT 1",
            (str(source_path),),
        ).fetchone()
    return None if row is None else bytes(row[0])


def _excised(archive_root: Path, blob_hash: bytes) -> bool:
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        return is_blob_hash_excised(conn, blob_hash)


def _sidecar_payloads(archive_root: Path, session_id: str) -> list[dict[str, object]]:
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT payload_json FROM session_events WHERE session_id = ? "
            "AND event_type = 'claude_tool_result_sidecar' ORDER BY position",
            (session_id,),
        ).fetchall()
    return [json.loads(str(row[0])) for row in rows]


def _rederived_hash(archive_root: Path, raw_id: str) -> str:
    """The identity of the session the archive re-derives from its retained bytes."""
    with ArchiveStore(archive_root, initialize=False, read_only=False) as store:
        [session] = parse_retained_raw_sessions(store, raw_id)
    return str(session_content_hash(session))


@pytest.mark.asyncio
async def test_excising_a_keeps_the_sidecar_it_shares_with_b_and_forgets_its_own(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    root = workspace_env["data_root"] / "projects"
    tree_a = _session_tree(root, _SESSION_A, [("toolu_a_shared", _SHARED_TEXT), ("toolu_a_only", _A_ONLY_TEXT)])
    tree_b = _session_tree(root, _SESSION_B, [("toolu_b_shared", _SHARED_TEXT)])
    await _ingest(
        workspace_env,
        root,
        [
            tree_a["toolu_a_shared"],
            tree_a["toolu_a_only"],
            tree_b["toolu_b_shared"],
            tree_a["transcript"],
            tree_b["transcript"],
        ],
        cursor_name="cursor.db",
    )

    session_a = _session_row(archive_root, _SESSION_A)
    session_b = _session_row(archive_root, _SESSION_B)
    assert session_a is not None and session_b is not None
    a_payload_hash = _raw_hash(archive_root, tree_a["transcript"])
    assert a_payload_hash is not None
    # Both sessions' sidecars are retained under the one shared hash.
    assert _raw_hash(archive_root, tree_a["toolu_a_shared"]) == _sha(_SHARED_TEXT)
    assert _raw_hash(archive_root, tree_b["toolu_b_shared"]) == _sha(_SHARED_TEXT)
    [b_event] = _sidecar_payloads(archive_root, session_b[0])
    assert b_event["acquisition_status"] == "matched"
    # The stored identity is the one the archive re-derives from B's
    # retained bytes.
    assert session_b[2] == _rederived_hash(archive_root, session_b[1])

    receipt = apply_session_excision(archive_root, session_a[0], reason="synthetic secret", actor="user:local")

    assert receipt.found is True
    assert _session_row(archive_root, _SESSION_A) is None
    # A is forgotten: its transcript and the output only it had are marked.
    assert _excised(archive_root, a_payload_hash)
    assert _excised(archive_root, _sha(_A_ONLY_TEXT))
    assert _sha(_A_ONLY_TEXT).hex() in receipt.removed_blob_hashes
    # B keeps the blob it shares with A: unmarked, still retained, readable.
    assert not _excised(archive_root, _sha(_SHARED_TEXT))
    assert _sha(_SHARED_TEXT).hex() in receipt.shared_blob_hashes
    assert _sha(_SHARED_TEXT).hex() not in receipt.removed_blob_hashes
    assert _raw_hash(archive_root, tree_b["toolu_b_shared"]) == _sha(_SHARED_TEXT)
    assert BlobStore(archive_root / "blob").read_all(_sha(_SHARED_TEXT).hex()) == _SHARED_TEXT.encode("utf-8")
    assert _session_row(archive_root, _SESSION_B) == session_b

    # B grows by a second output with the same bytes as the one it shared
    # with A; A's unchanged export arrives again. A fresh cursor makes the
    # watcher read both files from the start.
    tree_b = _session_tree(root, _SESSION_B, [("toolu_b_shared", _SHARED_TEXT), ("toolu_b_again", _SHARED_TEXT)])
    await _ingest(
        workspace_env,
        root,
        [tree_b["toolu_b_again"], tree_a["transcript"], tree_b["transcript"]],
        cursor_name="cursor-reingest.db",
    )

    assert _session_row(archive_root, _SESSION_A) is None, "re-ingesting A's export resurrected it"
    grown_b = _session_row(archive_root, _SESSION_B)
    assert grown_b is not None
    events = _sidecar_payloads(archive_root, grown_b[0])
    assert [(event["tool_use_id"], event["acquisition_status"]) for event in events] == [
        ("toolu_b_shared", "matched"),
        ("toolu_b_again", "matched"),
    ]
    assert _raw_hash(archive_root, tree_b["toolu_b_again"]) == _sha(_SHARED_TEXT)
    assert grown_b[2] != session_b[2]
    assert grown_b[2] == _rederived_hash(archive_root, grown_b[1])
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        texts = [
            str(row[0])
            for row in conn.execute(
                "SELECT text FROM blocks WHERE session_id = ? AND block_type = ?",
                (grown_b[0], BlockType.TOOL_RESULT.value),
            ).fetchall()
        ]
    assert len(texts) == 2 and all("zz_shared_build_log_line" in text for text in texts)
