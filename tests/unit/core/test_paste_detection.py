"""Tests for paste-detection heuristics."""

from __future__ import annotations

from polylogue.archive.message.paste_detection import (
    detect_paste,
    has_paste_heuristic,
    has_paste_indicator,
    has_paste_marker,
)


def test_detect_paste_empty_text() -> None:
    assert detect_paste(None) == 0
    assert detect_paste("") == 0
    assert detect_paste("   ") == 0


def test_marker_and_heuristic_are_distinct_signals() -> None:
    # A real paste marker is ground truth: marker True, heuristic False.
    marker_only = "look at [Pasted text #1] for the details"
    assert has_paste_marker(marker_only) is True
    assert has_paste_heuristic(marker_only) is False

    # A long typed prose message is only a proxy: heuristic True, marker False.
    heuristic_only = "x" * 4001
    assert has_paste_marker(heuristic_only) is False
    assert has_paste_heuristic(heuristic_only) is True

    # Ordinary short prose matches neither.
    plain = "Can you help me fix this bug?"
    assert has_paste_marker(plain) is False
    assert has_paste_heuristic(plain) is False

    # detect_paste is the union selection gate over both signals.
    assert detect_paste(marker_only) == 1
    assert detect_paste(heuristic_only) == 1
    assert detect_paste(plain) == 0


def test_predicates_handle_empty_text() -> None:
    for empty in (None, "", "   "):
        assert has_paste_marker(empty) is False
        assert has_paste_heuristic(empty) is False


def test_detect_paste_short_typed_text() -> None:
    assert detect_paste("Can you help me fix this bug?") == 0
    assert detect_paste("A short message about the project architecture.") == 0


def test_detect_paste_long_text() -> None:
    long_text = "x" * 4001
    assert detect_paste(long_text) == 1


def test_detect_paste_exactly_at_threshold() -> None:
    assert detect_paste("x" * 4000) == 0
    assert detect_paste("x" * 4001) == 1


def test_detect_paste_chatlog_forwarding() -> None:
    assert detect_paste("previous chatlog attached below\n\n...") == 1
    assert detect_paste("showing you the last chatlog\n\n...") == 1
    assert detect_paste("showing you last chatlog\n\n...") == 1
    assert detect_paste("here is the full chatlog\n\n...") == 1
    assert detect_paste("attached log below\n\n...") == 1
    assert detect_paste("below is the chatlog\n\n...") == 1


def test_detect_paste_code_fence_dominant() -> None:
    code_heavy = "Here is the code:\n```\n" + ("x" * 800) + "\n```\n"
    assert detect_paste(code_heavy) == 1


def test_detect_paste_code_fence_balanced() -> None:
    balanced = "Here is the code:\n```\n" + ("x" * 30) + "\n```\nNow let me explain what this does in detail.\n" * 20
    assert detect_paste(balanced) == 0


def test_detect_paste_short_code_fence() -> None:
    assert detect_paste("```\nshort\n```\nCan you explain this?") == 0


# ---------------------------------------------------------------------------
# Paste-marker detection (Claude Code agent runtime expansion convention).
# ---------------------------------------------------------------------------


def test_detect_paste_marker_pasted_text() -> None:
    assert detect_paste("Look at [Pasted text #1] for details") == 1
    assert detect_paste("Compare [Pasted text #1] and [Pasted text #12]") == 1


def test_detect_paste_marker_pasted_content_image() -> None:
    assert detect_paste("Check [Pasted content #3]") == 1
    assert detect_paste("See [Pasted image #2]") == 1


def test_detect_paste_marker_case_insensitive() -> None:
    assert detect_paste("review [PASTED TEXT #1]") == 1


def test_detect_paste_marker_word_only_no_match() -> None:
    # The literal word "Pasted" without the bracketed marker shape must not trip.
    assert detect_paste("I pasted the answer earlier.") == 0


def test_detect_paste_marker_with_line_count_annotation() -> None:
    """Production Claude Code emits `[Pasted text #1 +6 lines]` not the bare
    `[Pasted text #1]` form. Pinning the matcher honors the line-count suffix
    so the heuristic actually fires on real prompts (#1583)."""
    assert detect_paste("Look at [Pasted text #1 +6 lines] for details") == 1
    assert detect_paste("[Pasted text #2 +123 lines] suffix prose") == 1
    assert detect_paste("[Pasted content #4 +2 lines]") == 1
    assert detect_paste("[Pasted image #1 +1 line]") == 1


def test_detect_paste_marker_must_close_in_same_bracket_pair() -> None:
    """The annotation between ``#N`` and ``]`` must not span another bracket."""
    assert detect_paste("[Pasted text #1 [nested] still here]") == 0


# ---------------------------------------------------------------------------
# Base64 blob detection.
# ---------------------------------------------------------------------------


def test_detect_paste_base64_blob() -> None:
    # Realistic base64 payload (mixed case + digits + structural chars).
    blob = ("AbCd1234+/" * 60) + "==Ef5678Gh"
    assert len(blob) >= 512
    assert detect_paste(f"prefix {blob} suffix") == 1


def test_detect_paste_base64_blob_below_threshold() -> None:
    blob = "AbCd1234+/" * 20  # 200 chars, well below 512
    assert detect_paste(f"prefix {blob} suffix") == 0


def test_detect_paste_uniform_long_run_not_base64() -> None:
    # ``"x" * 4000`` is caught by the length heuristic, not the base64 one.
    # Ensure a long single-character run shorter than the length threshold
    # does not falsely trip base64 detection.
    assert detect_paste(f"prefix {'x' * 600} suffix") == 0


# ---------------------------------------------------------------------------
# Hook event payload wiring (PreToolUse / PostToolUse / UserPromptSubmit).
# ---------------------------------------------------------------------------


def test_has_paste_indicator_user_prompt_submit_with_marker() -> None:
    payload = {"prompt": "Look at [Pasted text #1] and tell me what's wrong"}
    assert has_paste_indicator(payload) is True


def test_has_paste_indicator_user_prompt_submit_no_paste() -> None:
    payload = {"prompt": "Fix the bug in auth.py"}
    assert has_paste_indicator(payload) is False


def test_has_paste_indicator_pre_tool_use_with_pasted_input() -> None:
    payload = {
        "tool_name": "Edit",
        "tool_input": {"content": "[Pasted text #2]"},
    }
    assert has_paste_indicator(payload) is True


def test_has_paste_indicator_post_tool_use_no_paste() -> None:
    payload = {
        "tool_name": "Read",
        "tool_output": "short file contents",
    }
    assert has_paste_indicator(payload) is False


def test_has_paste_indicator_post_tool_use_long_output() -> None:
    payload = {
        "tool_name": "Read",
        "tool_output": "x" * 5000,
    }
    assert has_paste_indicator(payload) is True


def test_has_paste_indicator_wrapped_hook_record() -> None:
    # Full hook envelope as emitted by polylogue-hook → daemon.
    record = {
        "event_type": "UserPromptSubmit",
        "session_id": "test-001",
        "timestamp": "2026-05-18T12:00:00Z",
        "provider": "claude-code",
        "payload": {"prompt": "Review [Pasted text #1]"},
    }
    assert has_paste_indicator(record) is True


def test_has_paste_indicator_wrapped_hook_record_no_paste() -> None:
    record = {
        "event_type": "PreToolUse",
        "session_id": "test-001",
        "timestamp": "2026-05-18T12:00:00Z",
        "provider": "claude-code",
        "payload": {"tool_name": "Bash", "tool_input": {"command": "ls"}},
    }
    assert has_paste_indicator(record) is False


def test_has_paste_indicator_none() -> None:
    assert has_paste_indicator(None) is False


def test_has_paste_indicator_empty_payload() -> None:
    assert has_paste_indicator({}) is False


def test_has_paste_indicator_codex_user_prompt_submit() -> None:
    record = {
        "event_type": "UserPromptSubmit",
        "session_id": "codex-001",
        "timestamp": "2026-05-18T12:00:00Z",
        "provider": "codex",
        "payload": {"prompt": "Look at [Pasted text #1]"},
    }
    assert has_paste_indicator(record) is True


def test_has_paste_indicator_base64_blob_in_tool_output() -> None:
    blob = ("AbCd1234+/" * 60) + "==Ef5678Gh"
    payload = {
        "tool_name": "Read",
        "tool_output": f"header\n{blob}\nfooter",
    }
    assert has_paste_indicator(payload) is True


def test_has_paste_indicator_message_consistency_with_detect_paste() -> None:
    """Hook indicator and direct detect_paste must agree on the same text."""
    marker_text = "[Pasted text #5]"
    assert detect_paste(marker_text) == 1
    assert has_paste_indicator({"prompt": marker_text}) is True
