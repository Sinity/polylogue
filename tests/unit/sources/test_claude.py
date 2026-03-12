"""Pinned Claude Code semantic regressions.

Cross-provider semantic ownership lives in ``test_unified_semantic_laws.py``.
This file keeps Claude-specific helper and parser regressions that still add
unique value.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.pipeline.semantic import (
    detect_context_compaction,
    extract_file_changes,
    extract_thinking_traces,
    extract_tool_invocations,
    parse_git_operation,
)
from polylogue.sources.parsers.claude import parse_code


@pytest.mark.parametrize(
    ("blocks", "expected_texts"),
    [
        ([{"type": "thinking", "thinking": "I need to analyze this carefully."}], ["I need to analyze this carefully."]),
        ([{"type": "thinking", "thinking": "First thought"}, {"type": "text", "text": "Response"}, {"type": "thinking", "thinking": "Second thought"}], ["First thought", "Second thought"]),
        ([{"type": "thinking", "thinking": ""}, {"type": "thinking", "thinking": None}], []),
        ([{"type": "text", "text": "Hello"}], []),
    ],
    ids=["single", "multiple", "empty", "none"],
)
def test_extract_thinking_traces_contract(blocks: list[dict[str, object]], expected_texts: list[str]) -> None:
    traces = extract_thinking_traces(blocks)
    assert [trace["text"] for trace in traces] == expected_texts


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        ([{"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/test.py"}}], {"file": True, "search": False, "git": False, "subagent": False}),
        ([{"type": "tool_use", "name": "Glob", "id": "tool-1", "input": {}}], {"file": False, "search": True, "git": False, "subagent": False}),
        ([{"type": "tool_use", "name": "Bash", "id": "tool-1", "input": {"command": "git status"}}], {"file": False, "search": False, "git": True, "subagent": False}),
        ([{"type": "tool_use", "name": "Task", "id": "tool-1", "input": {"subagent_type": "Explore"}}], {"file": False, "search": False, "git": False, "subagent": True}),
        ([{"type": "tool_use", "name": "Bash", "id": "tool-1", "input": {"command": "ls -la"}}], {"file": False, "search": False, "git": False, "subagent": False}),
    ],
    ids=["read", "search", "git", "subagent", "plain-bash"],
)
def test_extract_tool_invocations_contract(blocks: list[dict[str, object]], expected: dict[str, bool]) -> None:
    invocation = extract_tool_invocations(blocks)[0]
    assert invocation.get("is_file_operation", False) is expected["file"]
    assert invocation.get("is_search_operation", False) is expected["search"]
    assert invocation.get("is_git_operation", False) is expected["git"]
    assert invocation.get("is_subagent", False) is expected["subagent"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"tool_name": "Bash", "input": {"command": "git status"}}, {"command": "status"}),
        ({"tool_name": "Bash", "input": {"command": 'git commit -m "Fix bug"'}}, {"command": "commit", "message": "Fix bug"}),
        ({"tool_name": "Bash", "input": {"command": "git checkout feature-branch"}}, {"command": "checkout", "branch": "feature-branch"}),
        ({"tool_name": "Bash", "input": {"command": "git push origin main"}}, {"command": "push", "remote": "origin", "branch": "main"}),
        ({"tool_name": "Bash", "input": {"command": "git add file1.py file2.py"}}, {"command": "add", "files": ["file1.py", "file2.py"]}),
    ],
    ids=["status", "commit", "checkout", "push", "add"],
)
def test_parse_git_operation_contract(payload: dict[str, object], expected: dict[str, object]) -> None:
    result = parse_git_operation(payload)
    assert result is not None
    for key, value in expected.items():
        if key == "files":
            assert set(result[key]) == set(value)
        else:
            assert result[key] == value


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([{"tool_name": "Read", "input": {"file_path": "/test.py"}}], {"operation": "read", "path": "/test.py"}),
        ([{"tool_name": "Write", "input": {"file_path": "/new.py", "content": "print('hello')"}}], {"operation": "write", "path": "/new.py"}),
        ([{"tool_name": "Edit", "input": {"file_path": "/test.py", "old_string": "old code", "new_string": "new code"}}], {"operation": "edit", "path": "/test.py"}),
    ],
    ids=["read", "write", "edit"],
)
def test_extract_file_changes_contract(payload: list[dict[str, object]], expected: dict[str, str]) -> None:
    changes = extract_file_changes(payload)
    assert len(changes) == 1
    assert changes[0]["operation"] == expected["operation"]
    assert changes[0]["path"] == expected["path"]


def test_extract_file_changes_truncates_long_content() -> None:
    long_content = "x" * 1000
    changes = extract_file_changes([
        {"tool_name": "Write", "input": {"file_path": "/test.py", "content": long_content}},
    ])
    assert len(changes[0]["new_content"]) <= 500


@pytest.mark.parametrize(
    ("item", "should_detect"),
    [
        ({"type": "summary", "message": {"content": "Summary of the conversation so far..."}, "timestamp": 1704067200}, True),
        ({"type": "summary", "message": {"content": [{"type": "text", "text": "Conversation summary here"}]}}, True),
        ({"type": "user", "message": {"content": "Hello"}}, False),
    ],
    ids=["summary-text", "summary-blocks", "non-summary"],
)
def test_context_compaction_detection_contract(item: dict[str, object], should_detect: bool) -> None:
    result = detect_context_compaction(item)
    if should_detect:
        assert result is not None
        assert "summary" in result["summary"].lower()
    else:
        assert result is None


def test_parse_code_semantic_projection_contract() -> None:
    payload = [
        {
            "type": "assistant",
            "uuid": "msg-1",
            "timestamp": 1704067200000,
            "costUSD": 0.01,
            "durationMs": 1000,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Here is my answer"},
                    {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/test.py"}},
                    {"type": "tool_use", "name": "Task", "id": "tool-2", "input": {"subagent_type": "Explore", "prompt": "Find config files"}},
                ],
            },
        },
        {
            "type": "summary",
            "message": {"content": "Summary of conversation"},
            "timestamp": 1704067201000,
        },
        {
            "type": "assistant",
            "uuid": "msg-2",
            "timestamp": 1704067202000,
            "costUSD": 0.02,
            "durationMs": 2000,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "id": "tool-3", "input": {"command": "git commit -m 'Fix bug'"}},
                ],
            },
        },
    ]

    result = parse_code(payload, "test-session")

    assert len(result.messages) == 2
    assert result.messages[0].provider_meta is None
    first_types = [block.type for block in result.messages[0].content_blocks]
    assert "thinking" in first_types and "tool_use" in first_types
    bash_blocks = [block for block in result.messages[1].content_blocks if block.tool_name == "Bash"]
    assert len(bash_blocks) == 1
    assert bash_blocks[0].tool_input is not None
    assert bash_blocks[0].tool_input.get("command") == "git commit -m 'Fix bug'"
    assert result.provider_meta is not None
    assert len(result.provider_meta["context_compactions"]) == 1
    assert result.provider_meta["total_cost_usd"] == pytest.approx(0.03)
    assert result.provider_meta["total_duration_ms"] == 3000


@given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_thinking_extraction_never_crashes(texts: list[str]) -> None:
    blocks = [{"type": "thinking", "thinking": text} for text in texts]
    assert isinstance(extract_thinking_traces(blocks), list)


@given(
    st.lists(
        st.sampled_from(["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"]),
        min_size=0,
        max_size=10,
    )
)
@settings(max_examples=50)
def test_tool_extraction_never_crashes(tool_names: list[str]) -> None:
    blocks = [
        {"type": "tool_use", "name": name, "id": f"t{i}", "input": {}}
        for i, name in enumerate(tool_names)
    ]
    assert len(extract_tool_invocations(blocks)) == len(tool_names)
