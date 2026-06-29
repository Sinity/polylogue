"""Canonical web-URL projection tests."""

from __future__ import annotations

import pytest

from polylogue.core.enums import Origin
from polylogue.core.web_urls import canonical_session_url, native_id_from_session_id


def test_native_id_extraction() -> None:
    assert native_id_from_session_id("chatgpt-export:6a4167ef-148c") == "6a4167ef-148c"
    assert native_id_from_session_id("claude-ai-export:2c2eab57") == "2c2eab57"
    # No prefix / malformed → None.
    assert native_id_from_session_id("no-colon-here") is None
    assert native_id_from_session_id("origin:") is None


def test_chatgpt_url_bare() -> None:
    assert (
        canonical_session_url(Origin.CHATGPT_EXPORT, "6a4167ef-148c")
        == "https://chatgpt.com/c/6a4167ef-148c"
    )


def test_chatgpt_url_with_project() -> None:
    assert (
        canonical_session_url(Origin.CHATGPT_EXPORT, "6a4167ef", "g-p-6a40343a")
        == "https://chatgpt.com/g/g-p-6a40343a/c/6a4167ef"
    )


def test_chatgpt_url_ignores_non_project_ref() -> None:
    # A custom-GPT gizmo (g-...) is not a project (g-p-...); fall back to bare.
    assert (
        canonical_session_url(Origin.CHATGPT_EXPORT, "abc", "g-customgpt")
        == "https://chatgpt.com/c/abc"
    )


def test_claude_ai_url() -> None:
    assert (
        canonical_session_url(Origin.CLAUDE_AI_EXPORT, "2c2eab57")
        == "https://claude.ai/chat/2c2eab57"
    )


@pytest.mark.parametrize(
    "origin",
    [Origin.CLAUDE_CODE_SESSION, Origin.CODEX_SESSION, Origin.GROK_EXPORT, Origin.AISTUDIO_DRIVE],
)
def test_local_or_unconfirmed_origins_have_no_url(origin: Origin) -> None:
    assert canonical_session_url(origin, "anything") is None


def test_missing_native_id() -> None:
    assert canonical_session_url(Origin.CHATGPT_EXPORT, None) is None
    assert canonical_session_url(Origin.CHATGPT_EXPORT, "") is None


def test_accepts_string_origin() -> None:
    assert (
        canonical_session_url("chatgpt-export", "x") == "https://chatgpt.com/c/x"
    )
