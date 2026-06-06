"""Pin parser emission of typed session-context fields on ParsedSession.

These tests anchor the universal-semantics graduation work for #864: parsers
must populate the typed `working_directories`, `git_branch`, and
`git_repository_url` fields on `ParsedSession`, not only the legacy
`provider_meta` dict entries. Downstream consumers (storage writes,
attribution, insight rebuild) can then move to the typed surface without
parser changes.

The legacy `provider_meta` keys (`working_directories`, `gitBranch`, `git`)
are also asserted so that the transitional dual-write contract is explicit.
"""

from __future__ import annotations

from polylogue.sources.parsers.claude import parse_code
from polylogue.sources.parsers.claude.index import (
    SessionIndexEntry,
    enrich_session_from_index,
)
from polylogue.sources.parsers.codex import parse as parse_codex


def test_parse_code_promotes_working_directories_to_typed_field() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-A",
            "timestamp": 1704067200,
            "cwd": "/home/user/projectA",
            "message": {"role": "user", "content": "hello"},
        },
        {
            "type": "assistant",
            "uuid": "a-1",
            "sessionId": "sess-A",
            "timestamp": 1704067201,
            "cwd": "/home/user/projectB",
            "message": {"role": "assistant", "content": "hi", "model": "claude-3"},
        },
    ]

    result = parse_code(items, "fallback")

    assert result.working_directories == ["/home/user/projectA", "/home/user/projectB"]
    # Transitional: provider_meta still carries the same list until storage reads
    # from the typed field directly (next step in #864 graduation).
    assert result.provider_meta is not None
    assert result.provider_meta["working_directories"] == [
        "/home/user/projectA",
        "/home/user/projectB",
    ]


def test_parse_code_typed_working_directories_empty_when_no_cwd() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-B",
            "timestamp": 1704067200,
            "message": {"role": "user", "content": "hello"},
        },
    ]

    result = parse_code(items, "fallback")

    assert result.working_directories == []


def test_parse_codex_promotes_session_context_to_typed_fields() -> None:
    items: list[object] = [
        {
            "type": "session_meta",
            "payload": {
                "id": "sess-codex-1",
                "timestamp": "2024-01-01T00:00:00Z",
                "git": {
                    "branch": "main",
                    "repository_url": "https://github.com/example/repo",
                },
            },
        },
        {"type": "turn_context", "payload": {"cwd": "/srv/codex/workspace"}},
        {
            "type": "response_item",
            "id": "m-1",
            "timestamp": "2024-01-01T00:00:01Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        },
    ]

    result = parse_codex(items, "fallback")

    assert result.working_directories == ["/srv/codex/workspace"]
    assert result.git_branch == "main"
    assert result.git_repository_url == "https://github.com/example/repo"
    # Transitional: legacy provider_meta keys still populated.
    assert result.provider_meta is not None
    assert result.provider_meta["working_directories"] == ["/srv/codex/workspace"]
    assert result.provider_meta["git"] == {
        "branch": "main",
        "repository_url": "https://github.com/example/repo",
    }


def test_parse_codex_typed_git_fields_none_when_absent() -> None:
    items: list[object] = [
        {
            "type": "session_meta",
            "payload": {
                "id": "sess-codex-bare",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        },
        {
            "type": "response_item",
            "id": "m-1",
            "timestamp": "2024-01-01T00:00:01Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        },
    ]

    result = parse_codex(items, "fallback")

    assert result.git_branch is None
    assert result.git_repository_url is None
    assert result.working_directories == []


def test_session_index_enrichment_promotes_git_branch_when_typed_field_empty() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-claude-idx",
            "timestamp": 1704067200,
            "message": {"role": "user", "content": "hello"},
        },
    ]
    parsed = parse_code(items, "fallback")
    assert parsed.git_branch is None

    entry = SessionIndexEntry(
        session_id="sess-claude-idx",
        full_path="/tmp/sess-claude-idx.jsonl",
        first_prompt="first user prompt",
        summary="Index summary",
        message_count=1,
        created="2024-01-01T00:00:00Z",
        modified="2024-01-01T01:00:00Z",
        git_branch="feature/work",
        project_path="/home/user/project",
        is_sidechain=False,
    )

    enriched = enrich_session_from_index(parsed, entry)

    assert enriched.git_branch == "feature/work"
    # Transitional: legacy provider_meta key kept for current storage readers.
    assert enriched.provider_meta is not None
    assert enriched.provider_meta["gitBranch"] == "feature/work"


def test_session_index_does_not_overwrite_existing_typed_git_branch() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-claude-idx2",
            "timestamp": 1704067200,
            "message": {"role": "user", "content": "hello"},
        },
    ]
    parsed = parse_code(items, "fallback").model_copy(update={"git_branch": "main"})

    entry = SessionIndexEntry(
        session_id="sess-claude-idx2",
        full_path="/tmp/sess-claude-idx2.jsonl",
        first_prompt="prompt",
        summary="Summary",
        message_count=1,
        created=None,
        modified=None,
        git_branch="feature/other",
        project_path=None,
        is_sidechain=False,
    )

    enriched = enrich_session_from_index(parsed, entry)

    # Typed field is preserved; provider_meta still records index-derived branch
    # for backwards compatibility with current consumers.
    assert enriched.git_branch == "main"
    assert enriched.provider_meta is not None
    assert enriched.provider_meta["gitBranch"] == "feature/other"
