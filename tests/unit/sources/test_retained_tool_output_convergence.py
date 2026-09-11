"""Overflowed tool outputs survive the loss of their original tree (polylogue-cq1ql).

A Claude Code or Gemini CLI tool result that overflows its inline envelope
keeps only a preview in the transcript; the rest is a sibling file. Until this
bead the join read that sibling from the original filesystem *while deriving*,
so the same retained transcript produced the full output or only its preview
depending on whether the tree still existed
(``docs/design/retained-inputs-and-supersession.md`` D3, S8, S9).

Every test here drives production acquisition (``LiveBatchProcessor``) into a
real archive, removes the entire source tree, and then derives through the
production replay entry point (``parse_retained_raw_sessions``) -- never a
join helper handed a directory.

Anti-vacuity for all of them: the sidecar bytes are what carries the payload,
so deleting the *retained* rows (not the original path) is what turns them
red. ``test_retained_resolution_is_what_carries_the_full_text`` pins that
directly by removing the retained sidecar row and asserting the preview comes
back.

Synthetic fixtures only: invented tool output, invented paths.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import unicodedata
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.core.enums import BlockType
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_SESSION_ID = "5c3d1e40-0000-4000-8000-00000000aaaa"
_PARENT_NEEDLE = "zz_parent_full_output_needle"
_SUBAGENT_NEEDLE = "zz_subagent_full_output_needle"
_GEMINI_NEEDLE = "zz_gemini_full_output_needle"


# ── Claude Code fixtures ───────────────────────────────────────────


def _overflow_envelope(sidecar: Path) -> str:
    return (
        "<persisted-output>\n"
        f"Output too large (5.0KB). Full output saved to: {sidecar}\n\n"
        "Preview (first 2KB):\nshort preview text without the payload"
    )


def _tool_exchange(prefix: str, session_id: str, tool_use_id: str, result_text: str) -> list[dict[str, object]]:
    return [
        {
            "type": "user",
            "uuid": f"{prefix}-u1",
            "sessionId": session_id,
            "timestamp": "2026-07-20T10:00:00Z",
            "message": {"role": "user", "content": "run it"},
        },
        {
            "type": "assistant",
            "uuid": f"{prefix}-a1",
            "parentUuid": f"{prefix}-u1",
            "sessionId": session_id,
            "timestamp": "2026-07-20T10:00:01Z",
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
            "timestamp": "2026-07-20T10:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result_text}],
            },
        },
    ]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _claude_tree(root: Path, *, orphan: bool = True) -> dict[str, Path]:
    """A parent transcript, one subagent, and the ``tool-results/`` both share."""
    project = root / "-realm-project-x"
    session_dir = project / _SESSION_ID
    tool_results = session_dir / "tool-results"
    tool_results.mkdir(parents=True)

    parent_sidecar = tool_results / "toolu_parent.txt"
    parent_sidecar.write_text(f"{_PARENT_NEEDLE}\n" * 400, encoding="utf-8")
    subagent_sidecar = tool_results / "toolu_subagent.txt"
    subagent_sidecar.write_text(f"{_SUBAGENT_NEEDLE}\n" * 400, encoding="utf-8")
    paths = {
        "project": project,
        "session_dir": session_dir,
        "tool_results": tool_results,
        "parent_sidecar": parent_sidecar,
        "subagent_sidecar": subagent_sidecar,
    }
    if orphan:
        orphan_sidecar = tool_results / "orphan999.txt"
        orphan_sidecar.write_text("owned by nobody anywhere in the session\n", encoding="utf-8")
        paths["orphan_sidecar"] = orphan_sidecar

    parent = project / f"{_SESSION_ID}.jsonl"
    _write_jsonl(parent, _tool_exchange("p", _SESSION_ID, "toolu_parent", _overflow_envelope(parent_sidecar)))
    subagent = session_dir / "subagents" / "agent-a.jsonl"
    _write_jsonl(
        subagent,
        _tool_exchange("s", f"{_SESSION_ID}-sub", "toolu_subagent", _overflow_envelope(subagent_sidecar)),
    )
    paths["parent"] = parent
    paths["subagent"] = subagent
    return paths


# ── Gemini CLI fixtures ────────────────────────────────────────────


_GEMINI_SESSION_ID = "gem-sess-1"
_GEMINI_TOOL_ID = "run_shell_command_1773524726450_0"


def _gemini_mask(path: Path) -> str:
    return (
        "<tool_output_masked>\n"
        "Output too large. Showing first 8,000 and last 32,000 characters. "
        f"For full output see: {path}\n"
        "Output: HEAD-EXCERPT\n...\nTAIL-EXCERPT\n</tool_output_masked>"
    )


def _gemini_tree(root: Path) -> dict[str, Path]:
    project = root / "project-hash"
    chats = project / "chats"
    chats.mkdir(parents=True)
    outputs = project / "tool-outputs" / f"session-{_GEMINI_SESSION_ID}"
    outputs.mkdir(parents=True)
    sidecar = outputs / f"{_GEMINI_TOOL_ID}.txt"
    sidecar.write_text(f"{_GEMINI_NEEDLE}\n" * 400, encoding="utf-8")

    snapshot = chats / f"session-{_GEMINI_SESSION_ID}.json"
    snapshot.write_text(
        json.dumps(
            {
                "sessionId": _GEMINI_SESSION_ID,
                "projectHash": "hash-1",
                "kind": "chat",
                "startTime": "2026-03-14T21:41:00.000Z",
                "lastUpdated": "2026-03-14T21:45:00.000Z",
                "messages": [
                    {
                        "id": "u1",
                        "type": "user",
                        "timestamp": "2026-03-14T21:41:00.000Z",
                        "content": "run it",
                    },
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
                                            "response": {"output": _gemini_mask(sidecar)},
                                        }
                                    }
                                ],
                                "resultDisplay": "short display",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {"project": project, "snapshot": snapshot, "sidecar": sidecar, "outputs": outputs}


# ── harness ────────────────────────────────────────────────────────


def _processor(
    workspace_env: dict[str, Path],
    sources: tuple[WatchSource, ...],
) -> tuple[Polylogue, CursorStore, LiveBatchProcessor]:
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "index.db")
    cursor = CursorStore(workspace_env["data_root"] / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        sources,
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    return archive, cursor, processor


def _raw_id_for(archive_root: Path, source_path: Path) -> str:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT raw_id FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at_ms DESC, raw_id DESC LIMIT 1",
            (str(source_path),),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"no retained raw row for {source_path}"
    return str(row[0])


def _tool_result_texts(session: ParsedSession) -> list[str]:
    return [
        block.text or ""
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT
    ]


def _sidecar_events(session: ParsedSession, event_type: str) -> list[dict[str, object]]:
    return [dict(event.payload) for event in session.session_events if event.event_type == event_type]


def _derive_after_tree_removal(archive_root: Path, raw_id: str, tree_root: Path) -> list[ParsedSession]:
    """Delete the whole source tree, then derive from retained bytes alone.

    ``parse_retained_raw_sessions`` opens the blob publisher for write (see
    ``revision_governance.raw_revision_descriptor``), so the store cannot be
    read-only even though this derivation only reads.
    """
    shutil.rmtree(tree_root)
    assert not tree_root.exists()
    with ArchiveStore(archive_root, initialize=False, read_only=False) as store:
        return parse_retained_raw_sessions(store, raw_id)


# ── tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claude_full_tool_text_survives_the_loss_of_its_source_tree(
    workspace_env: dict[str, Path],
) -> None:
    """AC1: acquisition and derivation agree with the original paths gone.

    Anti-vacuity: reverting the join to a ``tool-results/`` directory read
    leaves the derived block holding ``Preview (first 2KB)`` instead of the
    needle, because the directory no longer exists.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    tree = _claude_tree(root)
    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),)
    )
    try:
        await processor.ingest_files(
            [
                tree["parent_sidecar"],
                tree["subagent_sidecar"],
                tree["orphan_sidecar"],
                tree["parent"],
                tree["subagent"],
            ],
            emit_event=False,
        )
        raw_id = _raw_id_for(workspace_env["archive_root"], tree["parent"])
    finally:
        await archive.close()

    [derived] = _derive_after_tree_removal(workspace_env["archive_root"], raw_id, root)

    [text] = _tool_result_texts(derived)
    assert _PARENT_NEEDLE in text
    assert "Preview (first 2KB)" not in text

    events = _sidecar_events(derived, "claude_tool_result_sidecar")
    matched = {event["tool_use_id"] for event in events if event["acquisition_status"] == "matched"}
    assert matched == {"toolu_parent"}, "ownership must not drift onto the subagent's call"
    debt = {(event["filename"], event["reason"]) for event in events if event["acquisition_status"] == "debt"}
    assert debt == {("orphan999.txt", "no_owning_tool_result_block")}


@pytest.mark.asyncio
async def test_claude_subagent_scope_keeps_ownership_after_the_tree_is_gone(
    workspace_env: dict[str, Path],
) -> None:
    """AC2: a shared scope neither duplicates nor misattributes its sidecars.

    Anti-vacuity: dropping the retained sibling index makes the parent's pass
    report ``toolu_subagent.txt`` as debt -- the fanout bug the session-scoped
    join exists to prevent, reintroduced through retained resolution.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    tree = _claude_tree(root)
    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),)
    )
    try:
        await processor.ingest_files(
            [
                tree["parent_sidecar"],
                tree["subagent_sidecar"],
                tree["orphan_sidecar"],
                tree["parent"],
                tree["subagent"],
            ],
            emit_event=False,
        )
        parent_raw = _raw_id_for(workspace_env["archive_root"], tree["parent"])
        subagent_raw = _raw_id_for(workspace_env["archive_root"], tree["subagent"])
    finally:
        await archive.close()

    shutil.rmtree(root)
    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=False) as store:
        [parent] = parse_retained_raw_sessions(store, parent_raw)
        [subagent] = parse_retained_raw_sessions(store, subagent_raw)

    parent_events = _sidecar_events(parent, "claude_tool_result_sidecar")
    subagent_events = _sidecar_events(subagent, "claude_tool_result_sidecar")

    assert {event["filename"] for event in parent_events} == {"toolu_parent.txt", "orphan999.txt"}
    assert [event["filename"] for event in subagent_events] == ["toolu_subagent.txt"]
    assert all(event["acquisition_status"] == "matched" for event in subagent_events)

    [subagent_text] = _tool_result_texts(subagent)
    assert _SUBAGENT_NEEDLE in subagent_text
    assert _PARENT_NEEDLE not in subagent_text

    # A shared scope never adds a message to either transcript.
    from polylogue.sources.parsers.claude import parse_code

    parent_baseline = parse_code(_tool_exchange("p", _SESSION_ID, "toolu_parent", "preview only"), "baseline-parent")
    assert len(parent.messages) == len(parent_baseline.messages)
    assert len(subagent.messages) == len(parent_baseline.messages)


@pytest.mark.asyncio
async def test_missing_expected_sidecar_stays_explicit_and_a_late_one_reconverges(
    workspace_env: dict[str, Path],
) -> None:
    """AC2: absence is an outcome, and a later arrival converges without duplication.

    Anti-vacuity: dropping the ``expected_sidecar_not_retained`` branch leaves
    the first derivation silent about a pointer it could not resolve, and the
    second derivation's message count assertion is what catches a late arrival
    being materialized as new content instead of replacing a preview.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    tree = _claude_tree(root, orphan=False)
    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),)
    )
    try:
        # The parent's own sidecar is never acquired: its scope is observed
        # (the subagent's file is), but the file the parent's preview points
        # at is not retained.
        await processor.ingest_files(
            [tree["subagent_sidecar"], tree["parent"], tree["subagent"]],
            emit_event=False,
        )
        parent_raw = _raw_id_for(workspace_env["archive_root"], tree["parent"])
    finally:
        await archive.close()

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=False) as store:
        [before] = parse_retained_raw_sessions(store, parent_raw)

    absent = [
        event
        for event in _sidecar_events(before, "claude_tool_result_sidecar")
        if event.get("reason") == "expected_sidecar_not_retained"
    ]
    assert [event["filename"] for event in absent] == ["toolu_parent.txt"]
    [preview] = _tool_result_texts(before)
    assert _PARENT_NEEDLE not in preview

    # The sidecar arrives late and is acquired on its own.
    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),)
    )
    try:
        await processor.ingest_files([tree["parent_sidecar"]], emit_event=False)
    finally:
        await archive.close()

    [after] = _derive_after_tree_removal(workspace_env["archive_root"], parent_raw, root)

    [recovered] = _tool_result_texts(after)
    assert _PARENT_NEEDLE in recovered
    assert len(after.messages) == len(before.messages)
    assert not [
        event for event in _sidecar_events(after, "claude_tool_result_sidecar") if event["acquisition_status"] == "debt"
    ]


@pytest.mark.asyncio
async def test_gemini_full_tool_output_survives_the_loss_of_its_source_tree(
    workspace_env: dict[str, Path],
) -> None:
    """AC1 for Gemini CLI: the masked envelope resolves from retained bytes.

    Anti-vacuity: removing the gemini-cli ``tool_result_sidecar`` artifact
    rule means the walk never acquires ``tool-outputs/``, so nothing is
    retained and the derived block keeps ``<tool_output_masked>``.
    """
    root = workspace_env["data_root"] / "gemini"
    root.mkdir(parents=True)
    tree = _gemini_tree(root)
    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="gemini-cli", root=root, suffixes=(".json",)),)
    )
    try:
        await processor.ingest_files([tree["sidecar"], tree["snapshot"]], emit_event=False)
        raw_id = _raw_id_for(workspace_env["archive_root"], tree["snapshot"])
    finally:
        await archive.close()

    [derived] = _derive_after_tree_removal(workspace_env["archive_root"], raw_id, root)

    [text] = _tool_result_texts(derived)
    assert _GEMINI_NEEDLE in text
    assert "<tool_output_masked>" not in text

    [event] = _sidecar_events(derived, "gemini_cli_tool_output_sidecar")
    assert event["acquisition_status"] == "matched"
    assert event["content_replaced"] is True


@pytest.mark.asyncio
async def test_retained_resolution_is_what_carries_the_full_text(
    workspace_env: dict[str, Path],
) -> None:
    """AC4's removal check, executed: delete the retained row, lose the output.

    The original path is already gone in every test above, so this deletes the
    other half -- the retained sidecar evidence -- and asserts the derivation
    falls back to the preview. That is the mutation which must turn the
    convergence tests red.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    tree = _claude_tree(root, orphan=False)
    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),)
    )
    try:
        await processor.ingest_files(
            [tree["parent_sidecar"], tree["subagent_sidecar"], tree["parent"], tree["subagent"]],
            emit_event=False,
        )
        raw_id = _raw_id_for(workspace_env["archive_root"], tree["parent"])
    finally:
        await archive.close()

    shutil.rmtree(root)
    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
        conn.execute("DELETE FROM raw_sessions WHERE source_path = ?", (str(tree["parent_sidecar"]),))

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=False) as store:
        [derived] = parse_retained_raw_sessions(store, raw_id)

    [text] = _tool_result_texts(derived)
    assert _PARENT_NEEDLE not in text
    assert "Preview (first 2KB)" in text


@pytest.mark.asyncio
async def test_original_sidecar_bytes_stay_recoverable_beside_normalized_text(
    workspace_env: dict[str, Path],
) -> None:
    """AC3: the raw bytes and the searchable text are separate durable objects.

    The sidecar is written in NFD; ``blocks.search_text`` and the
    content-addressed copy the ingest batch publishes are NFC. Both references
    must resolve, and the raw one must return the original bytes verbatim --
    a normalized-only retention would make the source bytes unrecoverable.

    Anti-vacuity: publishing the sidecar only under its normalized hash, or
    rewriting the retained raw blob to the normalized form, makes the exact
    byte comparison fail.
    """
    root = workspace_env["data_root"] / "projects"
    root.mkdir(parents=True)
    tree = _claude_tree(root, orphan=False)
    decomposed = unicodedata.normalize("NFD", f"café {_PARENT_NEEDLE}\n") * 200
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    tree["parent_sidecar"].write_text(decomposed, encoding="utf-8")
    original_bytes = tree["parent_sidecar"].read_bytes()

    archive, _cursor, processor = _processor(
        workspace_env, (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),)
    )
    try:
        await processor.ingest_files(
            [tree["parent_sidecar"], tree["subagent_sidecar"], tree["parent"], tree["subagent"]],
            emit_event=False,
        )
        raw_id = _raw_id_for(workspace_env["archive_root"], tree["parent_sidecar"])
    finally:
        await archive.close()

    shutil.rmtree(root)
    blob_root = workspace_env["archive_root"] / "blob"
    conn = sqlite3.connect(f"file:{workspace_env['archive_root'] / 'source.db'}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    raw_hash = str(row[0]).lower()

    from polylogue.storage.blob_store import BlobStore

    store = BlobStore(blob_root)
    assert store.exists(raw_hash), "the retained sidecar blob must resolve"
    assert store.read_all(raw_hash) == original_bytes
    assert raw_hash == hashlib.sha256(original_bytes).hexdigest()
    assert raw_hash != hashlib.sha256(unicodedata.normalize("NFC", decomposed).encode("utf-8")).hexdigest()
