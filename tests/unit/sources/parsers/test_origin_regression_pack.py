"""Table-driven source-parser regression pack covering every supported origin.

One parametrised test enumerates synthetic per-origin fixtures and asserts the
common parser contract.  Each fixture record declares exactly which assertions
are relevant for its origin; items marked ``N/A`` are skipped so the test table
does not overfit to parser-specific idiosyncrasies.

Origins covered
---------------
* chatgpt-export        — Provider.CHATGPT
* claude-ai-export      — Provider.CLAUDE_AI
* claude-code-session   — Provider.CLAUDE_CODE
* codex-session         — Provider.CODEX
* gemini-cli-session    — Provider.GEMINI_CLI
* aistudio-drive        — Provider.DRIVE / Provider.GEMINI (chunkedPrompt)
* antigravity-session   — Provider.ANTIGRAVITY  (markdown-export path)
* hermes-session        — Provider.HERMES

Origins skipped
---------------
* None — all listed origins have live parsers in the tree.

Note on provider_meta
---------------------
The AC for #1833 mentions "no provider_meta escape hatch in public parser
output."  Eliminating ``provider_meta`` from the ``ParsedSession`` /
``ParsedMessage`` models is a CODE change tracked in a dedicated follow-up
issue (see below).  This pack does NOT assert absence of ``provider_meta``
because doing so would fail on the current models and conflate test work with
model surgery.  A separate issue is filed per the orchestrator instructions.

Follow-up: eliminate provider_meta escape hatch from parser-output models
          — see #1858 (Ref #1790). Tracked as a new issue filed during this task.

Ref #1833.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.sources.parsers.antigravity import (
    BRAIN_METADATA_FRAGMENT_FLAG,
    AntigravitySessionSummary,
    parse_markdown_export,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.chatgpt import looks_like as chatgpt_looks_like
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.claude import looks_like_ai as claude_ai_looks_like
from polylogue.sources.parsers.claude import looks_like_code as claude_code_looks_like
from polylogue.sources.parsers.claude import parse_ai as claude_ai_parse
from polylogue.sources.parsers.claude import parse_code as claude_code_parse
from polylogue.sources.parsers.codex import looks_like as codex_looks_like
from polylogue.sources.parsers.codex import parse as codex_parse
from polylogue.sources.parsers.drive import looks_like as drive_looks_like
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.sources.parsers.local_agent import (
    looks_like_gemini_cli,
    looks_like_hermes,
    parse_gemini_cli,
    parse_hermes,
)
from polylogue.types import BlockType

# ---------------------------------------------------------------------------
# Sentinel for "not applicable for this origin"
# ---------------------------------------------------------------------------

_NA = object()  # sentinel — this assertion is skipped for the fixture


# ---------------------------------------------------------------------------
# Fixture contract dataclass
# ---------------------------------------------------------------------------


@dataclass
class OriginFixture:
    """Describes one parser fixture and its expected contract."""

    label: str
    """Human-readable label used as the pytest ID."""

    provider: Provider
    """Expected ``Provider`` on the parsed session."""

    session_id: str
    """Expected ``provider_session_id``."""

    min_messages: int
    """Lower bound on ``len(session.messages)``."""

    # Detection
    looks_like_fn: Any  # callable(payload) -> bool
    payload: Any  # the input payload passed to both looks_like and parse
    parse_fn: Any  # callable(payload, fallback_id) -> ParsedSession

    # Optional contract items — use _NA to skip
    expected_title: Any = _NA  # str | None | _NA
    has_tool_use: bool | object = _NA  # True/False or _NA
    has_thinking: bool | object = _NA  # True/False or _NA
    has_paste: bool | object = _NA  # True/False or _NA
    has_attachment: bool | object = _NA  # True/False or _NA
    has_working_dir: bool | object = _NA  # True/False or _NA
    has_git_branch: bool | object = _NA  # True/False or _NA
    has_git_repo: bool | object = _NA  # True/False or _NA
    ingest_flags_include: list[str] = field(default_factory=list)
    ingest_flags_exclude: list[str] = field(default_factory=list)

    # For session-ID stability test, we re-call parse_fn a second time and
    # compare; this is always exercised unless parse_fn requires a tmp_path.
    skip_stability: bool = False


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _chatgpt_payload() -> dict[str, Any]:
    """Minimal ChatGPT export with a paste marker, code-execution, and thinking blocks.

    ChatGPT uses ``content_type`` to distinguish block semantics:
    - ``"thoughts"``        → THINKING block
    - ``"code"``            → CODE block (code interpreter input)
    - ``"execution_output"``→ TOOL_RESULT block (code interpreter output)
    - default text parts    → TEXT blocks

    ChatGPT does NOT produce TOOL_USE blocks in the polylogue parser; the
    tool-execution surface is represented by CODE + TOOL_RESULT block pairs.
    """
    return {
        "id": "chatgpt-session-reg-1",
        "title": "ChatGPT regression fixture",
        "mapping": {
            "u1": {
                "id": "u1",
                "parent": None,
                "children": ["a1"],
                "message": {
                    "id": "u1",
                    "author": {"role": "user"},
                    "content": {"parts": ["[Pasted text #1]\nrun checks"]},
                    "create_time": 1_704_067_200.0,
                },
            },
            "a1": {
                "id": "a1",
                "parent": "u1",
                "children": ["code1"],
                "message": {
                    "id": "a1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "thoughts",
                        "text": "I should run the code interpreter.",
                    },
                    "create_time": 1_704_067_201.0,
                    "metadata": {"model_slug": "gpt-4o"},
                },
            },
            "code1": {
                "id": "code1",
                "parent": "a1",
                "children": ["t1"],
                "message": {
                    "id": "code1",
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "code",
                        "text": "import subprocess\nresult = subprocess.run(['pytest'])",
                    },
                    "create_time": 1_704_067_202.0,
                },
            },
            "t1": {
                "id": "t1",
                "parent": "code1",
                "children": [],
                "message": {
                    "id": "t1",
                    "author": {"role": "tool"},
                    "content": {
                        "content_type": "execution_output",
                        "text": "3 passed in 0.5s",
                    },
                    "create_time": 1_704_067_203.0,
                },
            },
        },
        "current_node": "t1",
    }


def _claude_ai_payload() -> dict[str, Any]:
    """Minimal Claude AI web-export with thinking and tool_use blocks."""
    return {
        "uuid": "claude-ai-session-reg-1",
        "name": "Claude AI regression fixture",
        "chat_messages": [
            {
                "uuid": "m1",
                "sender": "human",
                "text": "Run the tests.",
                "created_at": "2026-01-01T10:00:00.000Z",
                "attachments": [
                    {
                        "id": "att-1",
                        "file_name": "plan.md",
                        "file_type": "text/markdown",
                        "file_size": 128,
                    }
                ],
            },
            {
                "uuid": "m2",
                "sender": "assistant",
                "text": "Running now.",
                "created_at": "2026-01-01T10:00:01.000Z",
                "content": [
                    {"type": "thinking", "thinking": "I should run pytest first."},
                    {"type": "tool_use", "id": "tu-1", "name": "bash", "input": {"cmd": "pytest"}},
                ],
            },
            {
                "uuid": "m3",
                "sender": "assistant",
                "text": "Done.",
                "created_at": "2026-01-01T10:00:02.000Z",
            },
        ],
    }


def _claude_code_payload() -> list[Any]:
    """Minimal Claude Code JSONL with tool_use, thinking, paste, and cwd."""
    return [
        {
            "type": "user",
            "uuid": "cc-u1",
            "sessionId": "claude-code-session-reg-1",
            "cwd": "/workspace/polylogue",
            "timestamp": 1_704_067_200,
            "message": {
                "role": "user",
                "content": "[Pasted text #1]\nrun checks",
            },
        },
        {
            "type": "assistant",
            "uuid": "cc-a1",
            "sessionId": "claude-code-session-reg-1",
            "cwd": "/workspace/polylogue",
            "timestamp": 1_704_067_201,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I'll run pytest."},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "pytest"},
                    },
                ],
            },
        },
        {
            "type": "tool",
            "uuid": "cc-t1",
            "sessionId": "claude-code-session-reg-1",
            "timestamp": 1_704_067_202,
            "message": {
                "role": "tool",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "passed"}],
            },
        },
    ]


def _codex_payload() -> list[Any]:
    """Codex envelope format with git context."""
    return [
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-session-reg-1",
                "timestamp": "2026-01-01T10:00:00Z",
                "git": {
                    "branch": "feature/test",
                    "repository_url": "https://github.com/Sinity/polylogue",
                    "commit_hash": "abc1234",
                },
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run checks."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Running now."}],
            },
        },
    ]


def _gemini_cli_payload() -> dict[str, Any]:
    """Gemini CLI session with thinking and tool_use."""
    return {
        "sessionId": "gemini-cli-session-reg-1",
        "projectHash": "project-hash-x",
        "startTime": "2026-02-01T08:00:00.000Z",
        "lastUpdated": "2026-02-01T08:01:00.000Z",
        "kind": "chat",
        "summary": "Gemini CLI regression fixture",
        "messages": [
            {
                "id": "g1",
                "timestamp": "2026-02-01T08:00:01.000Z",
                "type": "user",
                "content": ["analyse the codebase"],
            },
            {
                "id": "g2",
                "timestamp": "2026-02-01T08:00:02.000Z",
                "type": "gemini",
                "content": "Understood.",
                "model": "gemini-2.0-flash",
                "durationMs": 500,
                "tokens": {"total": 20},
                "thoughts": [{"text": "Let me think first."}],
                "toolCalls": [
                    {
                        "id": "tc-1",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    }
                ],
            },
        ],
    }


def _aistudio_drive_payload() -> dict[str, Any]:
    """AI Studio / Drive chunkedPrompt format."""
    return {
        "id": "aistudio-drive-session-reg-1",
        "title": "AI Studio regression fixture",
        "createTime": "2026-03-01T09:00:00Z",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "chunk-1",
                    "role": "user",
                    "text": "Summarise the changes.",
                    "createTime": "2026-03-01T09:00:01Z",
                },
                {
                    "id": "chunk-2",
                    "role": "model",
                    "text": "The changes include…",
                    "createTime": "2026-03-01T09:00:02Z",
                },
            ]
        },
    }


def _antigravity_payload() -> tuple[str, AntigravitySessionSummary]:
    """Antigravity markdown-export fixture (language-server path)."""
    markdown = """\
# Chat Session

Note: _This is purely the output of the chat session._

### User Input

Implement the feature.

### Planner Response

I will start by reading the spec.

### User Input

Good, proceed.

### Planner Response

Done — all tests pass.
"""
    summary = AntigravitySessionSummary(
        cascade_id="antigravity-session-reg-1",
        title="Antigravity regression fixture",
        workspace_name="polylogue",
        last_modified_time="2026-04-01T12:00:00Z",
    )
    return markdown, summary


def _hermes_payload() -> dict[str, Any]:
    """Hermes (local-inference agent) session with tool_use and reasoning."""
    return {
        "session_id": "hermes-session-reg-1",
        "model": "hermes-3",
        "platform": "linux",
        "session_start": "2026-05-01T07:00:00.000000",
        "last_updated": "2026-05-01T07:01:00.000000",
        "system_prompt": "You are a helpful coding assistant.",
        "tools": [{"name": "shell"}],
        "message_count": 3,
        "messages": [
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "Running pytest now.",
                "model": "hermes-3-override",
                "durationMs": 800,
                "usage": {"input_tokens": 5, "output_tokens": 8},
                "reasoning_content": "I need to call pytest.",
                "tool_calls": [
                    {
                        "id": "call-h1",
                        "function": {
                            "name": "shell",
                            "arguments": '{"cmd":"pytest"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-h1",
                "content": "3 passed",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Build the fixture table
# ---------------------------------------------------------------------------
#
# antigravity uses a bespoke parse_fn because its signature is
# parse_markdown_export(markdown, summary) → ParsedSession, not
# parse_fn(payload, fallback_id).  We wrap it in a lambda here.

_AG_MARKDOWN, _AG_SUMMARY = _antigravity_payload()


def _ag_parse(payload: Any, fallback_id: str) -> ParsedSession:
    markdown, summary = payload
    return parse_markdown_export(markdown, summary)


def _ag_looks_like(payload: Any) -> bool:
    """Antigravity language-server export detected by the summary cascade_id field."""
    _, summary = payload
    return bool(summary.cascade_id)


ORIGIN_FIXTURES: list[OriginFixture] = [
    # ------------------------------------------------------------------
    # chatgpt-export
    # ------------------------------------------------------------------
    OriginFixture(
        label="chatgpt-export",
        provider=Provider.CHATGPT,
        session_id="chatgpt-session-reg-1",
        min_messages=4,
        looks_like_fn=chatgpt_looks_like,
        payload=_chatgpt_payload(),
        parse_fn=chatgpt_parse,
        expected_title="ChatGPT regression fixture",
        # ChatGPT parser does not produce TOOL_USE blocks; code execution
        # surfaces as CODE (input) + TOOL_RESULT (output) block pairs.
        has_tool_use=False,
        has_thinking=True,
        has_paste=True,
        has_attachment=False,
        has_working_dir=False,
        has_git_branch=False,
        has_git_repo=False,
    ),
    # ------------------------------------------------------------------
    # claude-ai-export
    # ------------------------------------------------------------------
    OriginFixture(
        label="claude-ai-export",
        provider=Provider.CLAUDE_AI,
        session_id="claude-ai-session-reg-1",
        min_messages=3,
        looks_like_fn=claude_ai_looks_like,
        payload=_claude_ai_payload(),
        parse_fn=claude_ai_parse,
        expected_title="Claude AI regression fixture",
        has_tool_use=True,
        has_thinking=True,
        has_paste=False,
        has_attachment=True,
        has_working_dir=False,
        has_git_branch=False,
        has_git_repo=False,
    ),
    # ------------------------------------------------------------------
    # claude-code-session
    # ------------------------------------------------------------------
    OriginFixture(
        label="claude-code-session",
        provider=Provider.CLAUDE_CODE,
        session_id="claude-code-session-reg-1",
        min_messages=3,
        looks_like_fn=claude_code_looks_like,
        payload=_claude_code_payload(),
        parse_fn=claude_code_parse,
        expected_title=_NA,  # title is derived from session_id for code sessions
        has_tool_use=True,
        has_thinking=True,
        has_paste=True,
        has_attachment=False,
        has_working_dir=True,
        has_git_branch=False,
        has_git_repo=False,
    ),
    # ------------------------------------------------------------------
    # codex-session
    # ------------------------------------------------------------------
    OriginFixture(
        label="codex-session",
        provider=Provider.CODEX,
        session_id="codex-session-reg-1",
        min_messages=2,
        looks_like_fn=codex_looks_like,
        payload=_codex_payload(),
        parse_fn=codex_parse,
        expected_title=_NA,  # codex title comes from session_id or git branch
        has_tool_use=False,
        has_thinking=False,
        has_paste=False,
        has_attachment=False,
        has_working_dir=False,
        has_git_branch=True,
        has_git_repo=True,
    ),
    # ------------------------------------------------------------------
    # gemini-cli-session
    # ------------------------------------------------------------------
    OriginFixture(
        label="gemini-cli-session",
        provider=Provider.GEMINI_CLI,
        session_id="gemini-cli-session-reg-1",
        min_messages=2,
        looks_like_fn=looks_like_gemini_cli,
        payload=_gemini_cli_payload(),
        parse_fn=parse_gemini_cli,
        expected_title="Gemini CLI regression fixture",
        has_tool_use=True,
        has_thinking=True,
        has_paste=False,
        has_attachment=False,
        has_working_dir=False,
        has_git_branch=False,
        has_git_repo=False,
    ),
    # ------------------------------------------------------------------
    # aistudio-drive  (chunkedPrompt / DRIVE provider)
    # ------------------------------------------------------------------
    OriginFixture(
        label="aistudio-drive",
        provider=Provider.DRIVE,
        session_id="aistudio-drive-session-reg-1",
        min_messages=2,
        looks_like_fn=drive_looks_like,
        payload=_aistudio_drive_payload(),
        parse_fn=lambda payload, fid: parse_chunked_prompt(Provider.DRIVE, payload, fid),
        expected_title="AI Studio regression fixture",
        has_tool_use=False,
        has_thinking=False,
        has_paste=False,
        has_attachment=False,
        has_working_dir=False,
        has_git_branch=False,
        has_git_repo=False,
    ),
    # ------------------------------------------------------------------
    # antigravity-session  (language-server markdown-export path)
    # ------------------------------------------------------------------
    OriginFixture(
        label="antigravity-session",
        provider=Provider.ANTIGRAVITY,
        session_id="antigravity-session-reg-1",
        min_messages=4,
        looks_like_fn=_ag_looks_like,
        payload=(_AG_MARKDOWN, _AG_SUMMARY),
        parse_fn=_ag_parse,
        expected_title="Antigravity regression fixture",
        has_tool_use=False,
        has_thinking=False,
        has_paste=False,
        has_attachment=False,
        has_working_dir=False,
        has_git_branch=False,
        has_git_repo=False,
        ingest_flags_exclude=[BRAIN_METADATA_FRAGMENT_FLAG],  # whole-transcript, not fragmented
    ),
    # ------------------------------------------------------------------
    # hermes-session
    # ------------------------------------------------------------------
    OriginFixture(
        label="hermes-session",
        provider=Provider.HERMES,
        session_id="hermes-session-reg-1",
        min_messages=3,
        looks_like_fn=looks_like_hermes,
        payload=_hermes_payload(),
        parse_fn=parse_hermes,
        expected_title=_NA,  # hermes titles session_id
        has_tool_use=True,
        has_thinking=True,
        has_paste=False,
        has_attachment=False,
        has_working_dir=False,
        has_git_branch=False,
        has_git_repo=False,
    ),
]


# ---------------------------------------------------------------------------
# Helpers for the assertions
# ---------------------------------------------------------------------------


def _all_blocks(session: ParsedSession) -> list[Any]:
    return [block for msg in session.messages for block in msg.blocks]


def _block_types(session: ParsedSession) -> set[BlockType]:
    return {block.type for block in _all_blocks(session)}


def _has_paste_spans(session: ParsedSession) -> bool:
    return any(msg.paste_spans for msg in session.messages)


def _has_paste_marker(session: ParsedSession) -> bool:
    """Detect paste either by explicit paste_spans or [Pasted text #N] in text."""
    if _has_paste_spans(session):
        return True
    return any(msg.text and "[Pasted text #" in msg.text for msg in session.messages)


def _has_attachments(session: ParsedSession) -> bool:
    return len(session.attachments) > 0


# ---------------------------------------------------------------------------
# Main parametrised test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ORIGIN_FIXTURES, ids=lambda f: f.label)
def test_origin_contract(fixture: OriginFixture) -> None:
    """Assert the per-origin parser contract for the given fixture."""

    # --- 1. Origin detection -------------------------------------------
    assert fixture.looks_like_fn(fixture.payload), (
        f"[{fixture.label}] looks_like returned False for the fixture payload"
    )

    # --- 2. Parse + session grouping -----------------------------------
    session = fixture.parse_fn(fixture.payload, fixture.session_id)
    assert isinstance(session, ParsedSession), f"[{fixture.label}] parse_fn must return a ParsedSession"

    # --- 3. Provider identity ------------------------------------------
    assert session.source_name is fixture.provider, (
        f"[{fixture.label}] expected source_name={fixture.provider!r}, got {session.source_name!r}"
    )

    # --- 4. Session ID -------------------------------------------------
    assert session.provider_session_id == fixture.session_id, (
        f"[{fixture.label}] provider_session_id mismatch: "
        f"expected {fixture.session_id!r}, got {session.provider_session_id!r}"
    )

    # --- 5. Session ID stability (re-parse produces the same ID) -------
    if not fixture.skip_stability:
        session2 = fixture.parse_fn(fixture.payload, fixture.session_id)
        assert session2.provider_session_id == session.provider_session_id, (
            f"[{fixture.label}] session_id not stable across re-parse"
        )

    # --- 6. Minimum message count (grouping check) ---------------------
    assert len(session.messages) >= fixture.min_messages, (
        f"[{fixture.label}] expected >= {fixture.min_messages} messages, got {len(session.messages)}"
    )

    # --- 7. Every message has a valid role -----------------------------
    valid_roles = {r.value for r in Role}
    for msg in session.messages:
        assert str(msg.role) in valid_roles, (
            f"[{fixture.label}] message {msg.provider_message_id!r} has invalid role {msg.role!r}"
        )

    # --- 8. active_leaf is set exactly once and is the last active msg -
    active_leaves = [msg for msg in session.messages if msg.is_active_leaf]
    if any(msg.is_active_leaf is not None for msg in session.messages):
        # Parser explicitly populated is_active_leaf
        assert len(active_leaves) == 1, (
            f"[{fixture.label}] expected exactly 1 active-leaf message, got {len(active_leaves)}"
        )
        assert session.active_leaf_message_provider_id == active_leaves[0].provider_message_id, (
            f"[{fixture.label}] active_leaf_message_provider_id does not match the is_active_leaf message"
        )

    # --- 9. Optional title assertion -----------------------------------
    if fixture.expected_title is not _NA:
        assert session.title == fixture.expected_title, (
            f"[{fixture.label}] title mismatch: expected {fixture.expected_title!r}, got {session.title!r}"
        )

    # --- 10. Block-type assertions -------------------------------------
    block_types = _block_types(session)

    if fixture.has_tool_use is not _NA:
        if fixture.has_tool_use:
            assert BlockType.TOOL_USE in block_types, (
                f"[{fixture.label}] expected TOOL_USE block but none found. Block types present: {block_types}"
            )
        else:
            assert BlockType.TOOL_USE not in block_types, (
                f"[{fixture.label}] expected no TOOL_USE blocks but found some"
            )

    if fixture.has_thinking is not _NA:
        if fixture.has_thinking:
            assert BlockType.THINKING in block_types, (
                f"[{fixture.label}] expected THINKING block but none found. Block types present: {block_types}"
            )
        else:
            assert BlockType.THINKING not in block_types, (
                f"[{fixture.label}] expected no THINKING blocks but found some"
            )

    # --- 11. Paste detection -------------------------------------------
    if fixture.has_paste is not _NA:
        detected = _has_paste_marker(session)
        if fixture.has_paste:
            assert detected, f"[{fixture.label}] expected paste evidence but none found"
        else:
            assert not detected, f"[{fixture.label}] expected no paste evidence but found some"

    # --- 12. Attachment detection --------------------------------------
    if fixture.has_attachment is not _NA:
        has_att = _has_attachments(session)
        if fixture.has_attachment:
            assert has_att, f"[{fixture.label}] expected attachments but none found"
        else:
            assert not has_att, f"[{fixture.label}] expected no attachments but found some"

    # --- 13. working_directories (cwd/repo/path extraction) -----------
    if fixture.has_working_dir is not _NA:
        has_wd = bool(session.working_directories)
        if fixture.has_working_dir:
            assert has_wd, f"[{fixture.label}] expected working_directories to be populated"
        else:
            assert not has_wd, f"[{fixture.label}] expected empty working_directories"

    # --- 14. git_branch extraction ------------------------------------
    if fixture.has_git_branch is not _NA:
        has_branch = session.git_branch is not None
        if fixture.has_git_branch:
            assert has_branch, f"[{fixture.label}] expected git_branch to be populated"
        else:
            assert not has_branch, f"[{fixture.label}] expected git_branch to be None"

    # --- 15. git_repository_url extraction ----------------------------
    if fixture.has_git_repo is not _NA:
        has_repo = session.git_repository_url is not None
        if fixture.has_git_repo:
            assert has_repo, f"[{fixture.label}] expected git_repository_url to be populated"
        else:
            assert not has_repo, f"[{fixture.label}] expected git_repository_url to be None"

    # --- 16. ingest_flags contract ------------------------------------
    for flag in fixture.ingest_flags_include:
        assert flag in session.ingest_flags, (
            f"[{fixture.label}] expected ingest_flags to include {flag!r}, got {session.ingest_flags!r}"
        )
    for flag in fixture.ingest_flags_exclude:
        assert flag not in session.ingest_flags, (
            f"[{fixture.label}] expected ingest_flags to NOT include {flag!r}, got {session.ingest_flags!r}"
        )


# ---------------------------------------------------------------------------
# Antigravity degraded-flag regression (#1764)
# ---------------------------------------------------------------------------
# The brain-metadata fragmentation test extends the test in
# tests/unit/sources/parsers/test_antigravity.py.  We include it here so the
# regression pack is self-contained for the origins table.  The authoritative
# tests live in test_antigravity.py; this is an additive reference.


def test_antigravity_brain_metadata_session_sets_degraded_flag(tmp_path: Path) -> None:
    """Brain-metadata path sets BRAIN_METADATA_FRAGMENT_FLAG; markdown-export path does not.

    Extends test_antigravity.py without duplicating its assertions.
    Validates that the degraded flag is absent from the whole-transcript
    (markdown-export) fixture used in the table above.
    """
    from polylogue.sources.parsers.antigravity import parse_brain_metadata

    session_dir = tmp_path / "brain" / "session-regpack"
    session_dir.mkdir(parents=True)
    artifact_path = session_dir / "analysis.md"
    artifact_path.write_text("# Analysis\n\nFindings here.\n", encoding="utf-8")
    metadata_path = session_dir / "analysis.md.metadata.json"
    metadata: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": "Analysis",
        "updatedAt": "2026-04-15T14:00:00Z",
    }

    brain_session = parse_brain_metadata(metadata, metadata_path, "fallback")
    assert BRAIN_METADATA_FRAGMENT_FLAG in brain_session.ingest_flags, (
        "Brain-metadata session must carry the degraded flag"
    )

    # Whole-transcript fixture from the table must NOT carry the flag.
    whole_session = _ag_parse((_AG_MARKDOWN, _AG_SUMMARY), _AG_SUMMARY.cascade_id)
    assert BRAIN_METADATA_FRAGMENT_FLAG not in whole_session.ingest_flags, (
        "Markdown-export (whole-transcript) session must NOT carry the degraded flag"
    )


# ---------------------------------------------------------------------------
# Session-count sanity: fixtures enumerate distinct sessions
# ---------------------------------------------------------------------------


def test_all_fixture_session_ids_are_unique() -> None:
    """Every fixture must declare a distinct provider_session_id."""
    ids = [f.session_id for f in ORIGIN_FIXTURES]
    assert len(ids) == len(set(ids)), "Duplicate session_id in ORIGIN_FIXTURES table"


def test_all_fixture_labels_are_unique() -> None:
    """Every fixture must have a distinct label (becomes the pytest ID)."""
    labels = [f.label for f in ORIGIN_FIXTURES]
    assert len(labels) == len(set(labels)), "Duplicate label in ORIGIN_FIXTURES table"
