"""Tests for paste-detection heuristics."""

from __future__ import annotations

from polylogue.archive.message.paste_detection import detect_paste


def test_detect_paste_empty_text() -> None:
    assert detect_paste(None) == 0
    assert detect_paste("") == 0
    assert detect_paste("   ") == 0


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
