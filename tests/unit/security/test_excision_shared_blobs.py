"""Excising a session forgets that session, not the blobs it shares (polylogue-bgnxh).

Blobs are content-addressed: two Claude Code sessions whose tool output
overflowed into byte-identical ``tool-results/`` sidecars own one blob hash.
The sidecar tests drive production acquisition (``LiveBatchProcessor``) for
two such sessions, A and B, where A also overflowed a second output nobody
else has, then excise A through the audited Excision operation.

Excision reaches the sidecar raws a session owns (polylogue-8j9rh): the
forgets-its-own test removes and marks A's own sidecar while naming the one it
shares with B, and the shared-sidecar test guards B's side -- a reach that
marked every hash A's sidecars named would refuse a later session's sidecar
with the same bytes. The subagent and gemini-cli tests pin ownership inside a
shared scope directory: excising one transcript's session leaves the files
another transcript owns, and files nobody claimed, where they are.

``test_excising_a_keeps_the_attachment_it_shares_with_b`` covers the
attachment class on the ingest batch's writer (``_write_session``) and the
acquire-time raw writer (``write_source_raw_session``). B's reference to the
shared attachment lives only in ``index.attachments``, the shape the live
batch writes; ignoring the index there marks B's attachment and refuses B's
next revision.

Synthetic fixtures only: invented tool output, invented paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.live import WatchSource
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceBlobRef,
    ContentExcisedError,
    is_blob_hash_excised,
    write_source_raw_session,
)
from tests.infra.excision import (
    plan_session_excision_from_root,
)
from tests.infra.excision_execution import execute_excision
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session
from tests.infra.live_batch import prepared_live_batch_processor
from tests.infra.retained_jsonl import prepared_source_fixture

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


async def _ingest(
    workspace_env: dict[str, Path],
    root: Path,
    files: list[Path],
    *,
    fresh_cursor: bool = False,
    source: WatchSource | None = None,
) -> None:
    """Ingest through the real live batch processor and its canonical Raw owner.

    ``fresh_cursor`` forgets these files' watcher cursors first (the disposable
    ops tier), so an unchanged file is genuinely reacquired rather than skipped
    at end-of-file: re-ingest then depends on the excision marker, not on the
    cursor, to keep excised content out.
    """
    archive_root = workspace_env["archive_root"]
    if fresh_cursor:
        spellings = [str(path) for path in files] + [str(path.resolve()) for path in files]
        marks = ",".join("?" for _ in spellings)
        with closing(sqlite3.connect(archive_root / "ops.db")) as ops:
            ops.execute(
                f"DELETE FROM ingest_cursor WHERE source_path IN ({marks}) OR canonical_source_path IN ({marks})",
                (*spellings, *spellings),
            )
            ops.commit()
    async with prepared_live_batch_processor(
        archive_root,
        (source or WatchSource(name="claude-code", root=root, layout=export_drop_layout((".jsonl",))),),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    ) as processor:
        await processor.ingest_files(files, emit_event=False)


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
    with prepared_source_fixture(archive_root) as source_read:
        [session] = parse_retained_raw_sessions(source_read, raw_id)
    return str(session_content_hash(session))


async def _ingest_a_and_b(workspace_env: dict[str, Path]) -> tuple[Path, dict[str, Path], dict[str, Path]]:
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
    )
    return root, tree_a, tree_b


@pytest.mark.asyncio
async def test_excising_a_forgets_the_tool_output_only_it_had(workspace_env: dict[str, Path]) -> None:
    """A's sidecar raws go; the hash only A had is marked, the one B shares is named.

    Anti-vacuity: dropping the sidecar seed from
    ``_resolve_session_excision_target`` leaves both of A's sidecar raws
    retained and the A-only hash unmarked.
    """
    archive_root = workspace_env["archive_root"]
    _root, tree_a, tree_b = await _ingest_a_and_b(workspace_env)
    session_a = _session_row(archive_root, _SESSION_A)
    assert session_a is not None

    plan = plan_session_excision_from_root(archive_root, session_a[0])
    assert plan.source_sidecar_rows == 2
    receipt = await asyncio.to_thread(
        execute_excision, archive_root, session_a[0], reason="synthetic secret", actor="user:local"
    )

    assert receipt["counts"]["source_sidecar_rows"] == 2
    assert _excised(archive_root, _sha(_A_ONLY_TEXT))
    assert _sha(_A_ONLY_TEXT).hex() in receipt["removed_blob_hashes"]
    assert _raw_hash(archive_root, tree_a["toolu_a_only"]) is None
    assert _raw_hash(archive_root, tree_a["toolu_a_shared"]) is None
    assert _sha(_SHARED_TEXT).hex() in receipt["shared_blob_hashes"]
    assert not _excised(archive_root, _sha(_SHARED_TEXT))
    assert _raw_hash(archive_root, tree_b["toolu_b_shared"]) == _sha(_SHARED_TEXT)


_PARENT_TEXT = "zz_parent_overflowed_output\n" * 400
_SUBAGENT_TEXT = "zz_subagent_overflowed_output\n" * 400
_ORPHAN_TEXT = "zz_sidecar_no_transcript_claims\n" * 40


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_excising_a_parent_keeps_its_subagents_sidecar(workspace_env: dict[str, Path]) -> None:
    """A parent and its subagent share one ``tool-results/`` directory.

    Excising the parent removes the sidecar the parent's own tool result
    owns, and leaves the subagent's sidecar and a file no transcript claims.
    The parent's sidecar is named by its preview's pointer, not by its
    ``tool_use_id``, so only the parent's matched sidecar event names it.

    Anti-vacuity: treating every file of the scope directory as the excised
    session's (``_SidecarOwnership.owns`` returning true) removes the
    subagent's sidecar raw and marks its hash; ignoring the session's sidecar
    events leaves the parent's pointer-named sidecar retained.
    """
    archive_root = workspace_env["archive_root"]
    root = workspace_env["data_root"] / "projects"
    project = root / "-realm-project-sub"
    session_dir = project / _SESSION_A
    tool_results = session_dir / "tool-results"
    tool_results.mkdir(parents=True)
    parent_sidecar = tool_results / "b7x3kq.txt"
    parent_sidecar.write_text(_PARENT_TEXT, encoding="utf-8")
    subagent_sidecar = tool_results / "toolu_subagent.txt"
    subagent_sidecar.write_text(_SUBAGENT_TEXT, encoding="utf-8")
    orphan_sidecar = tool_results / "orphan999.txt"
    orphan_sidecar.write_text(_ORPHAN_TEXT, encoding="utf-8")
    parent = project / f"{_SESSION_A}.jsonl"
    _write_jsonl(
        parent, _exchange("p", _SESSION_A, "toolu_parent", _overflow_envelope(parent_sidecar), parent=None, minute=0)
    )
    subagent = session_dir / "subagents" / "agent-a.jsonl"
    _write_jsonl(
        subagent,
        _exchange(
            "s", f"{_SESSION_A}-sub", "toolu_subagent", _overflow_envelope(subagent_sidecar), parent=None, minute=1
        ),
    )
    await _ingest(
        workspace_env,
        root,
        [parent_sidecar, subagent_sidecar, orphan_sidecar, parent, subagent],
    )
    session = _session_row(archive_root, _SESSION_A)
    assert session is not None
    assert _raw_hash(archive_root, subagent_sidecar) == _sha(_SUBAGENT_TEXT)

    receipt = await asyncio.to_thread(
        execute_excision, archive_root, session[0], reason="synthetic secret", actor="user:local"
    )

    assert receipt["counts"]["source_sidecar_rows"] == 1
    assert _raw_hash(archive_root, parent_sidecar) is None
    assert _excised(archive_root, _sha(_PARENT_TEXT))
    assert _raw_hash(archive_root, subagent_sidecar) == _sha(_SUBAGENT_TEXT)
    assert not _excised(archive_root, _sha(_SUBAGENT_TEXT))
    assert _raw_hash(archive_root, orphan_sidecar) == _sha(_ORPHAN_TEXT)
    assert not _excised(archive_root, _sha(_ORPHAN_TEXT))


_GEMINI_WIRE_SESSION = "gem-excise-1"
_GEMINI_TOOL_ID = "run_shell_command_1773524726450_0"
_GEMINI_TEXT = "zz_gemini_overflowed_output\n" * 400


def _gemini_snapshot(sidecar: Path) -> dict[str, object]:
    mask = (
        "<tool_output_masked>\n"
        "Output too large. Showing first 8,000 and last 32,000 characters. "
        f"For full output see: {sidecar}\n"
        "Output: HEAD-EXCERPT\n...\nTAIL-EXCERPT\n</tool_output_masked>"
    )
    return {
        "sessionId": _GEMINI_WIRE_SESSION,
        "projectHash": "hash-1",
        "kind": "main",
        "startTime": "2026-03-14T21:41:00.000Z",
        "lastUpdated": "2026-03-14T21:45:00.000Z",
        "messages": [
            {"id": "u1", "type": "user", "timestamp": "2026-03-14T21:41:00.000Z", "content": "run it"},
            {
                "id": "a1",
                "type": "gemini",
                "timestamp": "2026-03-14T21:41:02.000Z",
                "content": "ran it",
                "toolCalls": [
                    {
                        "id": _GEMINI_TOOL_ID,
                        "name": "run_shell_command",
                        "displayName": "Shell",
                        "description": "run a command",
                        "args": {"command": "echo hi"},
                        "renderOutputAsMarkdown": True,
                        "status": "success",
                        "timestamp": "2026-03-14T21:41:00.000Z",
                        "result": [
                            {
                                "functionResponse": {
                                    "id": _GEMINI_TOOL_ID,
                                    "name": "run_shell_command",
                                    "response": {"output": mask},
                                }
                            }
                        ],
                        "resultDisplay": "short display",
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_excising_a_gemini_chat_forgets_its_tool_output_sidecar(workspace_env: dict[str, Path]) -> None:
    """A gemini-cli chat's ``tool-outputs/session-<id>/`` sidecar goes with it.

    The directory is named for the wire ``sessionId`` and shared by every chat
    of that process, so a file no chat claimed stays.

    The sidecar is named ``<tool id>_<slug>``, so the chat's sidecar event,
    not the exact-stem rule, is what names it as the chat's.

    Anti-vacuity: dropping the gemini branch of ``_session_sidecar_raw_ids``
    leaves the sidecar raw retained and its hash unmarked.
    """
    archive_root = workspace_env["archive_root"]
    root = workspace_env["data_root"] / "gemini"
    project = root / "project-hash"
    outputs = project / "tool-outputs" / f"session-{_GEMINI_WIRE_SESSION}"
    outputs.mkdir(parents=True)
    sidecar = outputs / f"{_GEMINI_TOOL_ID}_stdout.txt"
    sidecar.write_text(_GEMINI_TEXT, encoding="utf-8")
    unclaimed = outputs / "read_file_1773524799999_0.txt"
    unclaimed.write_text(_ORPHAN_TEXT, encoding="utf-8")
    snapshot = project / "chats" / f"session-{_GEMINI_WIRE_SESSION}.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(_gemini_snapshot(sidecar)), encoding="utf-8")

    await _ingest(
        workspace_env,
        root,
        [sidecar, unclaimed, snapshot],
        source=WatchSource(name="gemini-cli", root=root, layout=export_drop_layout((".json",))),
    )
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        [(session_id,)] = conn.execute(
            "SELECT session_id FROM sessions WHERE origin = ?", (Origin.GEMINI_CLI_SESSION.value,)
        ).fetchall()
    assert _raw_hash(archive_root, sidecar) == _sha(_GEMINI_TEXT)
    assert _raw_hash(archive_root, unclaimed) == _sha(_ORPHAN_TEXT)

    receipt = await asyncio.to_thread(
        execute_excision, archive_root, str(session_id), reason="synthetic secret", actor="user:local"
    )

    assert receipt["counts"]["source_sidecar_rows"] == 1
    assert _raw_hash(archive_root, sidecar) is None
    assert _excised(archive_root, _sha(_GEMINI_TEXT))
    assert _raw_hash(archive_root, unclaimed) == _sha(_ORPHAN_TEXT)
    assert not _excised(archive_root, _sha(_ORPHAN_TEXT))


@pytest.mark.asyncio
async def test_excising_a_keeps_the_sidecar_it_shares_with_b(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    root, tree_a, tree_b = await _ingest_a_and_b(workspace_env)

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
    assert session_b[2] == await asyncio.to_thread(_rederived_hash, archive_root, session_b[1])

    receipt = await asyncio.to_thread(
        execute_excision, archive_root, session_a[0], reason="synthetic secret", actor="user:local"
    )

    assert receipt["found"] is True
    assert _session_row(archive_root, _SESSION_A) is None
    # A is forgotten: its transcript is marked.
    assert _excised(archive_root, a_payload_hash)
    # B keeps the blob it shares with A: unmarked, still retained, readable.
    assert not _excised(archive_root, _sha(_SHARED_TEXT))
    assert _sha(_SHARED_TEXT).hex() not in receipt["removed_blob_hashes"]
    assert _raw_hash(archive_root, tree_b["toolu_b_shared"]) == _sha(_SHARED_TEXT)
    assert BlobStore(archive_root / "blob").read_all(_sha(_SHARED_TEXT).hex()) == _SHARED_TEXT.encode("utf-8")
    assert _session_row(archive_root, _SESSION_B) == session_b

    # A new session C whose tool output has the same bytes as the blob A and
    # B shared arrives, together with A's unchanged export. C is admitted
    # with its sidecar matched; A is not resurrected.
    session_c = "5c3d1e40-0000-4000-8000-00000000c003"
    tree_c = _session_tree(root, session_c, [("toolu_c_shared", _SHARED_TEXT)])
    await _ingest(
        workspace_env,
        root,
        [tree_c["toolu_c_shared"], tree_a["transcript"], tree_c["transcript"]],
        fresh_cursor=True,
    )

    assert _session_row(archive_root, _SESSION_A) is None, "re-ingesting A's export resurrected it"
    stored_c = _session_row(archive_root, session_c)
    assert stored_c is not None
    events = _sidecar_payloads(archive_root, stored_c[0])
    assert [(event.get("tool_use_id"), event["acquisition_status"]) for event in events] == [
        ("toolu_c_shared", "matched")
    ], events
    assert _raw_hash(archive_root, tree_c["toolu_c_shared"]) == _sha(_SHARED_TEXT)
    assert stored_c[2] == await asyncio.to_thread(_rederived_hash, archive_root, stored_c[1])
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        texts = [
            str(row[0])
            for row in conn.execute(
                "SELECT text FROM blocks WHERE session_id = ? AND block_type = ?",
                (stored_c[0], BlockType.TOOL_RESULT.value),
            ).fetchall()
        ]
    assert len(texts) == 1 and "zz_shared_build_log_line" in texts[0]


_SHARED_ATTACHMENT = b"shared synthetic attachment bytes\n" * 64
_A_ONLY_ATTACHMENT = b"attachment only session A carried\n" * 64


def _attachment_ref(content: bytes) -> ArchiveSourceBlobRef:
    return ArchiveSourceBlobRef(
        blob_hash=hashlib.sha256(content).digest(),
        ref_type="attachment",
        source_path="/synthetic/export.json",
        size_bytes=len(content),
        acquired_at_ms=1_000,
    )


def _acquire(archive_root: Path, native_id: str, payload: bytes, refs: tuple[bytes, ...]) -> str:
    blob_store = BlobStore(archive_root / "blob")
    for content in refs:
        blob_hash, _size = blob_store.write_from_bytes(content)
        assert blob_hash == hashlib.sha256(content).hexdigest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        return write_source_raw_session(
            conn,
            origin=Origin.CLAUDE_AI_EXPORT.value,
            source_path=f"/synthetic/{native_id}.json",
            canonical_source_path=f"/synthetic/{native_id}.json",
            source_index=0,
            payload=payload,
            acquired_at_ms=1_000,
            native_id=native_id,
            additional_blob_refs=tuple(_attachment_ref(content) for content in refs),
        )


def _attachment_session(native_id: str, attachments: dict[str, bytes]) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=native_id,
        title="Synthetic",
        created_at="2026-04-02T00:00:00Z",
        updated_at="2026-04-02T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.normalize("user"), text="see attached")],
        attachments=[
            ParsedAttachment(
                provider_attachment_id=f"{native_id}-{name}",
                message_provider_id="m1",
                name=name,
                mime_type="text/plain",
                size_bytes=len(content),
                precomputed_blob=(hashlib.sha256(content).hexdigest(), len(content)),
            )
            for name, content in attachments.items()
        ],
    )


def test_excising_a_keeps_the_attachment_it_shares_with_b(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload_a = b'{"uuid": "att-a", "synthetic": "session a"}'
    raw_a = _acquire(archive_root, "att-a", payload_a, (_SHARED_ATTACHMENT, _A_ONLY_ATTACHMENT))
    raw_b = _acquire(archive_root, "att-b", b'{"uuid": "att-b", "synthetic": "session b"}', ())
    with (
        fixture_index_connection(archive_root / "index.db") as conn,
        sqlite3.connect(archive_root / "source.db") as source,
    ):
        for payload in (
            ("att-a", raw_a, {"shared.txt": _SHARED_ATTACHMENT, "own.txt": _A_ONLY_ATTACHMENT}),
            ("att-b", raw_b, {"shared.txt": _SHARED_ATTACHMENT}),
        ):
            native_id, raw_id, attachments = payload
            session = _attachment_session(native_id, attachments)
            preacquired = {
                attachment.acquisition_key: (
                    bytes.fromhex(attachment.precomputed_blob[0]),
                    attachment.precomputed_blob[1],
                    "acquired",
                )
                for attachment in session.attachments
                if attachment.precomputed_blob is not None
            }
            session_id = write_fixture_index_session(
                conn,
                session,
                raw_id=raw_id,
                source_conn=source,
                preacquired_attachment_blobs=preacquired,
            )
            assert session_id == f"{Origin.CLAUDE_AI_EXPORT.value}:{native_id}"
            conn.commit()
        session_a = f"{Origin.CLAUDE_AI_EXPORT.value}:att-a"

    shared = hashlib.sha256(_SHARED_ATTACHMENT).digest()
    own = hashlib.sha256(_A_ONLY_ATTACHMENT).digest()
    receipt = execute_excision(archive_root, session_a, reason="synthetic secret", actor="user:local")

    assert receipt["found"] is True
    assert _excised(archive_root, hashlib.sha256(payload_a).digest())
    assert _excised(archive_root, own)
    assert not _excised(archive_root, shared)
    assert receipt["shared_blob_hashes"] == [shared.hex()]
    with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
        kept = conn.execute(
            "SELECT a.acquisition_status FROM attachments AS a JOIN attachment_refs AS r "
            "ON r.attachment_id = a.attachment_id WHERE r.session_id = ? AND a.blob_hash = ?",
            (f"{Origin.CLAUDE_AI_EXPORT.value}:att-b", shared),
        ).fetchall()
    assert [tuple(row) for row in kept] == [("acquired",)]
    assert BlobStore(archive_root / "blob").read_all(shared.hex()) == _SHARED_ATTACHMENT

    # B's next revision still carries the shared attachment and is admitted;
    # A's unchanged export and a new raw carrying A's own attachment are not.
    _acquire(archive_root, "att-b", b'{"uuid": "att-b", "synthetic": "session b, revised"}', (_SHARED_ATTACHMENT,))
    with pytest.raises(ContentExcisedError):
        _acquire(archive_root, "att-a", payload_a, ())
    with pytest.raises(ContentExcisedError):
        _acquire(archive_root, "att-c", b'{"uuid": "att-c"}', (_A_ONLY_ATTACHMENT,))
