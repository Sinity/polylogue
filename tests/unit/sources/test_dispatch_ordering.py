"""Adversarial provider-detection ordering catalog (#1215).

``polylogue/sources/dispatch.py`` runs structural sequence detectors before
loose code/dict probes.  The ordering is fragile — a payload that resembles
two providers is claimed by whichever parser's ``looks_like()`` runs first.

This catalog locks the intended priority by asserting the current dispatch
order for every known adversarial pair.  Each entry includes a short comment
explaining the policy rationale.
"""

from __future__ import annotations

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.dispatch import detect_provider


def _payload(obj: object) -> object:
    """Normalise input so callers can write plain dicts/lists."""
    return obj


# ── Adversarial catalog ────────────────────────────────────────────
#
# Each entry: (payload, expected_provider, rationale)
# The payload is crafted to match *both* providers in the adversarial pair.
# The expected provider is the one the current dispatch order selects.

ADVERSARIAL_CATALOG: list[tuple[object, Provider, str]] = []


# ── Beads interaction vs loose session-record probes ──────────────
#
# Beads interaction rows may carry generic identifiers that resemble runtime
# records.  Their complete structural signature is stronger and must win before
# the Pydantic/dict-key session probes below.
ADVERSARIAL_CATALOG.append(
    (
        {
            "id": "int-1",
            "kind": "field_change",
            "created_at": "2026-07-08T20:14:36Z",
            "issue_id": "polylogue-7fj",
            "extra": {},
            "sessionId": "not-a-codex-session",
            "parentUuid": "runtime-shaped-noise",
        },
        Provider.BEADS,
        "beads interaction vs loose runtime probes: complete Beads structural detector runs before Pydantic/dict-key probes",
    )
)


# ── claude-code record stream vs codex envelope ───────────────────
#
# Both Claude Code JSONL and Codex session files are JSON records with
# a "sessionId" / "parentUuid" field.  Claude Code checks run before
# Codex in _detect_provider_from_sequence, so a payload that looks like
# both should be classified as Claude Code.

ADVERSARIAL_CATALOG.append(
    (
        [
            {
                "sessionId": "test-session",
                "parentUuid": "parent-001",
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        ],
        Provider.CLAUDE_CODE,
        "claude-code-record-stream vs codex: claude_code.looks_like runs before codex.looks_like in sequence detection",
    )
)


# ── chatgpt mapping vs claude chat_messages ───────────────────────
#
# ChatGPT exports have a top-level "mapping" dict; Claude.ai exports have
# "chat_messages".  A payload that has both would be ambiguous.
# chatgpt.looks_like runs before claude.looks_like_ai in single-record
# detection.

ADVERSARIAL_CATALOG.append(
    (
        {
            "id": "adversarial-conv",
            "conversation_id": "adversarial-conv",
            "create_time": 1_700_000_000.0,
            "current_node": "msg-1",
            "mapping": {
                "msg-1": {
                    "id": "msg-1",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "msg-1",
                        "content": {"content_type": "text", "parts": ["hello"]},
                        "author": {"role": "user"},
                    },
                }
            },
            "chat_messages": [{"id": "msg-1", "text": "hello", "sender": "human"}],
        },
        Provider.CHATGPT,
        "chatgpt mapping vs claude chat_messages: chatgpt.looks_like runs before claude.looks_like_ai",
    )
)


# ── codex envelope vs claude-code record ─────────────────────────
#
# A single record that has both Codex-specific fields (agentType, cascadeId)
# and Claude Code fields (sessionId, parentUuid).  In single-record
# detection, codex.looks_like runs before claude.looks_like_code.

ADVERSARIAL_CATALOG.append(
    (
        {
            "agentType": "general-purpose",
            "cascadeId": "cascade-001",
            "sessionId": "session-001",
            "parentUuid": "parent-001",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        },
        Provider.CODEX,
        "codex envelope vs claude-code record: codex.looks_like runs before claude.looks_like_code in single-record",
    )
)


# ── gemini chunked-prompt vs antigravity markdown ─────────────────
#
# Gemini Takeout exports have "chunkedPrompt" with "chunks"; Antigravity
# Markdown exports have "# Session with" headings.  A payload that
# has neither is classified by the fallback path.  gemini checks
# (_looks_like_gemini_mapping) run last, after antigravity checks.

ADVERSARIAL_CATALOG.append(
    (
        {"chunks": [{"text": "hello", "role": "user"}], "chunkedPrompt": True},
        Provider.GEMINI,
        "chunked-prompt: _looks_like_gemini_mapping matches chunks+chunkedPrompt structure",
    )
)


# ── drive takeout vs claude-ai export ─────────────────────────────
#
# Google Drive takeout exports have OAuth metadata; Claude.ai exports
# have "chat_messages".  A bare record without either falls through.

ADVERSARIAL_CATALOG.append(
    (
        {"chat_messages": [{"id": "m1", "text": "hello", "sender": "human"}], "title": "Test Chat"},
        Provider.CLAUDE_AI,
        "claude-ai: chat_messages list with sender field triggers claude.looks_like_ai",
    )
)


# ── Test runner ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("payload", "expected_provider", "rationale"),
    [(p, e, r) for (p, e, r) in ADVERSARIAL_CATALOG],
    ids=[f"entry_{i}" for i in range(len(ADVERSARIAL_CATALOG))],
)
def test_dispatch_ordering(payload: object, expected_provider: Provider, rationale: str) -> None:
    """Each adversarial payload must resolve to the expected provider.

    The rationale explains the policy: which provider's ``looks_like()``
    runs first in the current ``detect_provider`` implementation.
    """
    result = detect_provider(payload)
    assert result is expected_provider, (
        f"Expected {expected_provider.value} but got {result.value if result else None}.  Rationale: {rationale}"
    )


# ── Negative tests: unambiguous payloads must still work ────────────


@pytest.mark.parametrize(
    ("payload", "expected_provider"),
    [
        (
            {
                "id": "unambiguous-conv",
                "conversation_id": "unambiguous-conv",
                "create_time": 1_700_000_000.0,
                "current_node": "msg-1",
                "mapping": {
                    "msg-1": {
                        "id": "msg-1",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "msg-1",
                            "content": {"content_type": "text", "parts": ["hi"]},
                            "author": {"role": "user"},
                        },
                    }
                },
            },
            Provider.CHATGPT,
        ),
        ({"chat_messages": [{"id": "1", "text": "hi", "sender": "human"}]}, Provider.CLAUDE_AI),
        (
            [{"sessionId": "s1", "parentUuid": "p1", "role": "user", "content": [{"type": "text", "text": "hi"}]}],
            Provider.CLAUDE_CODE,
        ),
        (
            [
                {"type": "session_meta", "payload": {"id": "c1", "timestamp": "2025-01-01T00:00:00Z"}},
                {
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"type": "text", "text": "hi"}]},
                },
            ],
            Provider.CODEX,
        ),
        (
            {
                "conversations": [
                    {
                        "conversation": {"title": "hi"},
                        "responses": [{"response": {"sender": "human", "message": "hi"}}],
                    }
                ]
            },
            Provider.GROK,
        ),
    ],
)
def test_unambiguous_payloads_still_work(payload: object, expected_provider: Provider) -> None:
    """Regression: unambiguous payloads must still resolve correctly."""
    assert detect_provider(payload) is expected_provider


def test_relationship_index_records_do_not_detect_as_claude_code() -> None:
    """Regression for polylogue-9ykn (gvgi): a third-party graph-edge index
    JSONL (real shape: a sinex analysis artifact recording conversation
    parent/child edges) sitting under a watched Claude Code directory has no
    session/message envelope at all -- just conversation/parent/child/type/
    timestamp keys, whose bare ``type`` value happens to be the generic role
    word "assistant"/"user". Before this fix, ``claude.looks_like_code``
    treated a bare ``type in {"user", "assistant"}`` match as sufficient
    Claude Code evidence on its own, misdetecting this shape and (via the
    dispatch-level auto-detection paths that call ``detect_provider`` when no
    provider is already known from the watched directory) letting it become a
    phantom claude-code-session with one empty message per JSONL line -- the
    single largest contributor to the archive's empty-message rows (96,748 of
    101,765, ~95%, per the live-archive measurement in polylogue-gvgi).
    """
    payload = [
        {
            "conversation": f"conv-{index}",
            "parent": f"parent-{index}",
            "child": f"child-{index}",
            "type": "assistant" if index % 2 else "user",
            "timestamp": "2026-05-01T00:00:00.000Z",
        }
        for index in range(8)
    ]
    assert detect_provider(payload) is None
