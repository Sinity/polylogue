"""Hostile-but-decodable import content costs bounded time and loses one node, not the file.

Provider exports and downloaded ZIPs are untrusted input. Each case below is a
document a provider's own decoder accepts, and each one previously either
refused the entire import through an uncaught exception or made the parser's
cost quadratic in a length the document controls.

Anti-vacuity is named per test.
"""

from __future__ import annotations

import time

import pytest

from polylogue.sources.parsers import chatgpt
from polylogue.sources.parsers.antigravity import _ACTIVITY_MARKER_RE
from polylogue.sources.parsers.chatgpt import _generation_branch_key
from polylogue.sources.parsers.codex import _codex_tool_output_text


def test_deeply_nested_codex_tool_output_is_kept_verbatim_not_raised() -> None:
    """A nesting depth above the interpreter's limit is a non-JSON verdict.

    Anti-vacuity: remove the ``_CODEX_SANITIZER_MAX_DEPTH`` guard and this
    raises ``RecursionError`` out of the parser, losing the whole rollout --
    ``json.loads`` decodes far deeper than the recursive Python walk follows.
    """
    nested = "[" * 40_000 + "]" * 40_000

    assert _codex_tool_output_text(nested) == nested


@pytest.mark.uses_real_clock("measures parser wall-clock cost against a stated budget")
def test_antigravity_accepted_command_marker_does_not_rescan_the_whole_body() -> None:
    """An unterminated marker on every line must not restart a full-suffix scan.

    Anti-vacuity: restore the ``(?P<v>.*?)`` lazy wildcard and this budget is
    exceeded -- measured ~6.4s for this exact input against ~0.003s with the
    character class.
    """
    body = "\n".join(f"*User accepted the command `open{index}" for index in range(8_000))

    started = time.monotonic()
    assert _ACTIVITY_MARKER_RE.findall(body) == []
    assert time.monotonic() - started < 1.0


def test_antigravity_accepted_command_still_spans_newlines() -> None:
    """The multi-line heredoc argument the marker exists for is preserved."""
    matched = [
        match.group("v_accepted_command")
        for match in _ACTIVITY_MARKER_RE.finditer("*User accepted the command `line1\nline2`*")
    ]

    assert matched == ["line1\nline2"]


@pytest.mark.uses_real_clock("measures parser wall-clock cost against a stated budget")
def test_chatgpt_generation_branch_key_walk_is_memoized() -> None:
    """A long assistant chain is walked once, not once per node.

    Anti-vacuity: drop the ``memo`` argument at the call site and this budget
    is exceeded for the chain length below.
    """
    mapping: dict[str, object] = {"u": {"id": "u", "parent": None, "message": {"author": {"role": "user"}}}}
    previous = "u"
    for index in range(6_000):
        node_id = f"a{index}"
        mapping[node_id] = {
            "id": node_id,
            "parent": previous,
            "message": {
                "id": node_id,
                "author": {"role": "assistant"},
                "metadata": {"finished_duration_sec": 1.0},
            },
        }
        previous = node_id

    started = time.monotonic()
    chatgpt._extract_generation_timings(mapping)
    assert time.monotonic() - started < 5.0


def test_chatgpt_generation_branch_key_memo_agrees_with_the_unmemoized_walk() -> None:
    """The cache changes no verdict, including for a parent cycle."""
    cyclic: dict[str, object] = {
        "a": {"parent": "b", "message": {"author": {"role": "assistant"}}},
        "b": {"parent": "a", "message": {"author": {"role": "assistant"}}},
    }

    assert _generation_branch_key(cyclic, "a") == _generation_branch_key(cyclic, "a", {})
    assert _generation_branch_key(cyclic, "b") == _generation_branch_key(cyclic, "b", {})


def test_malformed_chatgpt_parts_loses_one_node_not_the_bundle() -> None:
    """``content.parts`` is export content and need not be a list.

    Anti-vacuity: restore ``parts = content.get("parts") or []`` and this
    raises ``TypeError: 'int' object is not iterable``, losing every message in
    the conversation rather than the one malformed node.
    """
    document = {
        "title": "t",
        "create_time": 1.0,
        "update_time": 2.0,
        "mapping": {
            "a": {
                "id": "a",
                "parent": None,
                "children": ["b"],
                "message": {
                    "id": "a",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hi"]},
                    "create_time": 1.0,
                },
            },
            "b": {
                "id": "b",
                "parent": "a",
                "children": [],
                "message": {
                    "id": "b",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": 1},
                    "create_time": 2.0,
                },
            },
        },
    }

    session = chatgpt.parse(document, fallback_id="conv")

    assert [message.text for message in session.messages] == ["hi"]
