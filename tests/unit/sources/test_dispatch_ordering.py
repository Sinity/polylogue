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
            "mapping": {"msg-1": {"message": {"content": {"parts": ["hello"]}, "author": {"role": "user"}}}},
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
            {"mapping": {"msg-1": {"message": {"content": {"parts": ["hi"]}, "author": {"role": "user"}}}}},
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
    ],
)
def test_unambiguous_payloads_still_work(payload: object, expected_provider: Provider) -> None:
    """Regression: unambiguous payloads must still resolve correctly."""
    assert detect_provider(payload) is expected_provider
