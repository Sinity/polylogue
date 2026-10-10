"""Production context references, shared windows, and exact additive budgeting."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from polylogue.archive.context_models import ContextSpec
from polylogue.archive.hydration import archive_block_to_domain
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.context.compiler import compile_prose_with_refs_context_segment
from polylogue.context.product_image import _select_context_messages, compile_context_image
from polylogue.core.identity_law import block_id
from polylogue.operations.ref_resolution import resolve_ref_against_archive
from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow
from polylogue.surfaces.compaction import estimate_token_words, estimate_tokens


def test_hydrated_action_marker_resolves_and_missing_identity_is_a_gap(tmp_path: Path) -> None:
    sid = "codex-session:context-law"
    mid = sid + ":n:m1"
    bid = block_id(mid, content_identity="a" * 64)
    row = ArchiveBlockRow(
        block_id=bid,
        message_id=mid,
        block_type="tool_use",
        text=None,
        content_identity="a" * 64,
        content_occurrence=0,
        tool_name="Read",
        tool_id="call-1",
    )
    block = archive_block_to_domain(row)
    message = Message(id=mid, role=Role.ASSISTANT, blocks=[block])
    segment, recapped = compile_prose_with_refs_context_segment(session_id=sid, title=None, messages=[message])
    ref = next(ref.format() for ref in segment.object_refs if ref.kind == "action")
    assert ref == "action:" + bid
    assert f"<ref:{ref}> Read" in (segment.markdown or "") and not recapped
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE sessions(session_id TEXT,origin TEXT,title TEXT)")
        conn.execute(
            "CREATE TABLE blocks(block_id TEXT,message_id TEXT,session_id TEXT,block_type TEXT,position INTEGER,text TEXT,tool_name TEXT,semantic_type TEXT,tool_command TEXT,tool_path TEXT)"
        )
        conn.execute("INSERT INTO sessions VALUES(?,?,?)", (sid, "codex-session", "Neutral"))
        conn.execute(
            "INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?,?,?)",
            (bid, mid, sid, "tool_use", 0, None, "Read", None, None, "plain.py"),
        )
        archive: Any = SimpleNamespace(_conn=conn, archive_root=tmp_path)
        resolved = resolve_ref_against_archive(archive, ref)
        assert resolved.resolved and resolved.payload is not None
        assert resolved.payload["block_id"] == bid
        assert not resolve_ref_against_archive(archive, "action:" + block_id(mid, content_identity="f" * 64)).resolved
    missing = message.copy_with_projected_content(
        text=None, blocks=[{"type": "tool_use", "name": "Read"}], attachments=[]
    )
    gap, _ = compile_prose_with_refs_context_segment(session_id=sid, title=None, messages=[missing])
    assert "action_reference_missing_stable_block_identity" in gap.caveats
    assert not any(ref.kind == "action" for ref in gap.object_refs)


@pytest.mark.parametrize("max_messages,max_chars", [(None, None), (1, None), (None, 10), (1, 10)])
async def test_profiles_share_anchor_membership_clipping_and_omissions(
    max_messages: int | None, max_chars: int | None
) -> None:
    sid = "codex-session:window-law"
    messages = [
        Message(
            id=f"{sid}:n:m{i}",
            role=Role.USER,
            text=f"UNIQUE-{i} filler " + "word " * 12,
            blocks=[{"type": "text", "text": f"UNIQUE-{i} filler " + "word " * 12}],
        )
        for i in range(4)
    ]

    class Source:
        async def _compile_context_seed_query(
            self, spec: ContextSpec
        ) -> tuple[list[str], dict[str, str], list[object]]:
            return [sid], {sid: messages[1].id}, []

        async def get_session(self, session_id: str) -> object:
            return SimpleNamespace(title="Neutral", messages=messages)

        async def get_session_summary(self, session_id: str) -> None:
            return None

    common = ContextSpec(
        seed_refs=(f"session:{sid}",),
        max_tokens=10000,
        include_assertions=False,
        max_messages_per_session=max_messages,
        max_chars_per_message=max_chars,
    )
    default = await compile_context_image(Source(), common)
    prose = await compile_context_image(Source(), common.model_copy(update={"segment_profile": "prose_with_refs"}))
    left, right = default.segments[0], prose.segments[0]
    assert left.caveats == right.caveats
    for index in range(4):
        expected = max_messages is None or index == 1
        assert (f"UNIQUE-{index}" in (left.markdown or "")) is expected
        assert (f"UNIQUE-{index}" in (right.markdown or "")) is expected
    if max_chars is not None:
        assert "chars omitted from this message" in (right.markdown or "")
        assert "filler" not in (right.markdown or "")
    assert default.omitted == prose.omitted


def _original_budget_render(texts: list[str], budget: int | None, keep: int) -> tuple[str, bool]:
    """Full-render specification oracle; intentionally not the incremental algorithm."""
    rows = [("user" if index == 0 else "assistant", text) for index, text in enumerate(texts)]

    def render() -> str:
        return (
            "\n".join(
                [
                    "# Messages: Neutral",
                    "",
                    "Expand action markers with resolve_ref before relying on them.",
                    "",
                    *(f"{role}: {text}" for role, text in rows),
                ]
            ).rstrip()
            + "\n"
        )

    recapped = False
    limit = None if budget is None else max(1, int(budget * 0.6))
    if limit is not None and estimate_tokens(render()) > limit:
        protected = {0, *range(max(0, len(rows) - keep), len(rows))}
        for index, (role, text) in enumerate(rows):
            if index in protected:
                continue
            rows[index] = (role, "[recap] " + " ".join(text.split()[:12]))
            recapped = True
            if estimate_tokens(render()) <= limit:
                break
        for index, (role, text) in enumerate(rows):
            if estimate_tokens(render()) <= limit:
                break
            if text:
                rows[index] = (role, "[omitted]")
                recapped = True
        while rows and estimate_tokens(render()) > limit:
            rows.pop(0)
            recapped = True
        if estimate_tokens(render()) > limit:
            return "", True
    return render(), recapped


@pytest.mark.parametrize("budget", [None, 1, 40, 100, 300, 600, 10000])
@pytest.mark.parametrize("keep", [0, 2, 20])
def test_incremental_budget_matches_full_render_word_estimator(budget: int | None, keep: int) -> None:
    texts = ["first\n\t user", "!" * 129 + "  ", " old prose " * 30, "\t\n", "", "中" * 91, "final " * 80]
    # Empty text contributes no row under the existing authored-prose contract.
    texts = [text for text in texts if text]
    messages = [
        Message(id=f"s:n:{index}", role=Role.USER if index == 0 else Role.ASSISTANT, text=text)
        for index, text in enumerate(texts)
    ]
    segment, recapped = compile_prose_with_refs_context_segment(
        session_id="s", title="Neutral", messages=messages, max_tokens=budget, keep_last_messages=keep
    )
    expected, expected_recapped = _original_budget_render(texts, budget, keep)
    assert segment.markdown == expected and recapped == expected_recapped
    assert segment.token_estimate == estimate_tokens(expected)


@pytest.mark.parametrize("count", [128, 256, 512, 1024])
def test_budget_estimator_scans_linear_total_characters(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    import polylogue.context.compiler as compiler

    original_words = estimate_token_words
    original_tokens = estimate_tokens
    scanned = 0

    def words(text: str) -> int:
        nonlocal scanned
        scanned += len(text)
        return original_words(text)

    def tokens(text: str) -> int:
        nonlocal scanned
        scanned += len(text)
        return original_tokens(text)

    monkeypatch.setattr(compiler, "estimate_token_words", words)
    monkeypatch.setattr(compiler, "estimate_tokens", tokens)
    messages = [
        Message(id=f"s:n:{index}", role=Role.USER if index == 0 else Role.ASSISTANT, text="word " * 40)
        for index in range(count)
    ]
    segment, recapped = compiler.compile_prose_with_refs_context_segment(
        session_id="s", title="Neutral", messages=messages, max_tokens=100
    )
    assert recapped and segment.token_estimate <= 60
    # Original row (211 chars), at most one 78-char recap, one 20-char omission,
    # and the fixed header/final render: fewer than 320 chars per input row.
    assert scanned <= count * 320 + 1000, scanned
    scanned = 0
    segment, recapped = compiler.compile_prose_with_refs_context_segment(
        session_id="s", title="Neutral", messages=messages
    )
    assert not recapped and scanned == len(segment.markdown or "")


def test_character_projection_preserves_action_identity_and_source_content() -> None:
    mid = "codex-session:clipped:n:m1"
    bid = block_id(mid, content_identity="a" * 64)
    message = Message(
        id=mid,
        role=Role.ASSISTANT,
        text="authored prose much longer than the limit",
        blocks=[
            {"type": "text", "text": "authored prose much longer than the limit"},
            {"id": bid, "type": "tool_use", "tool_name": "Read", "tool_input": {"file_path": "private-body"}},
        ],
    )
    selected, before, after, clipped = _select_context_messages(
        [message], anchor_message_id=None, max_messages=1, max_chars_per_message=8
    )
    segment, _ = compile_prose_with_refs_context_segment(
        session_id="codex-session:clipped",
        title=None,
        messages=selected,
        omitted_before=before,
        omitted_after=after,
        clipped_messages=clipped,
    )
    assert f"<ref:action:{bid}> Read" in (segment.markdown or "")
    assert "authored" in (segment.markdown or "") and "much longer" not in (segment.markdown or "")
    assert "private-body" not in (segment.markdown or "")
    assert selected[0].blocks[1]["id"] == bid
    assert message.blocks[0]["text"] == "authored prose much longer than the limit"
    assert clipped == 1
