"""Advanced ConversationFilter contracts.

This owner file covers content-aware filters, sorting projections, summary-path
edge cases, branch relations, delete cascades, and semantic capability filters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import ConversationSummary
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from tests.infra.storage_records import ConversationBuilder


@pytest.fixture
def filter_db_empty(tmp_path):
    db_path = tmp_path / "filter_empty.db"
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    return db_path


@pytest.fixture
def filter_db_advanced(tmp_path):
    db_path = tmp_path / "filter_advanced.db"

    (
        ConversationBuilder(db_path, "conv-thinking")
        .provider("claude")
        .title("Complex Problem Analysis")
        .add_message("m1", role="user", text="Solve this complex math problem")
        .add_message(
            "m2",
            role="assistant",
            text="The answer is 42.",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "Let me break this down step by step..."}]},
        )
        .add_message("m3", role="user", text="Can you explain further?")
        .metadata({"tags": ["math", "complex"], "summary": "Math problem solving"})
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-tools")
        .provider("claude")
        .title("API Integration Help")
        .add_message("m4", role="user", text="How do I call an API?")
        .add_message(
            "m5",
            role="assistant",
            text="I'll help you with that.",
            provider_meta={"content_blocks": [{"type": "tool_use", "tool_name": "bash", "input": {}}]},
        )
        .add_message("m6", role="user", text="Show me an example")
        .add_message("m7", role="assistant", text="Here is a complete working example with error handling.")
        .metadata({"tags": ["api", "integration"]})
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-attachments")
        .provider("chatgpt")
        .title("Document Analysis")
        .add_message("m8", role="user", text="Please analyze this document")
        .add_message("m9", role="assistant", text="I see the file contains important data.")
        .add_attachment("att1", message_id="m8", mime_type="application/pdf", size_bytes=5000)
        .metadata({"tags": ["documents"]})
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-summary-only")
        .provider("claude")
        .title("Brief Chat")
        .add_message("m10", role="user", text="Hello there")
        .add_message("m11", role="assistant", text="Hi how are you")
        .metadata({"summary": "Brief greeting exchange", "tags": ["greeting"]})
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-multi-attach")
        .provider("chatgpt")
        .title("Multiple File Analysis")
        .add_message("m12", role="user", text="Analyze these files please")
        .add_message("m13", role="assistant", text="I can see both files clearly.")
        .add_message("m14", role="user", text="What are the main differences?")
        .add_attachment("att2", message_id="m12", mime_type="image/png", size_bytes=2000)
        .add_attachment("att3", message_id="m12", mime_type="application/pdf", size_bytes=3000)
        .metadata({"tags": ["analysis"]})
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-long-messages")
        .provider("claude")
        .title("Deep Discussion")
        .add_message(
            "m15",
            role="user",
            text="Tell me everything you know about quantum computing including the fundamentals principles and applications",
        )
        .add_message(
            "m16",
            role="assistant",
            text="Quantum computing is a revolutionary field that leverages quantum mechanical phenomena like superposition and entanglement to perform computations exponentially faster than classical computers in certain domains such as cryptography and optimization.",
        )
        .metadata({"tags": ["quantum"]})
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-plain")
        .provider("codex")
        .title("Simple")
        .add_message("m17", role="user", text="What is two plus two")
        .metadata({"tags": ["simple"]})
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path


@pytest.fixture
def filter_repo_advanced(filter_db_advanced):
    return ConversationRepository(backend=SQLiteBackend(db_path=filter_db_advanced))


@pytest.fixture
def filter_repo_empty(filter_db_empty):
    return ConversationRepository(backend=SQLiteBackend(db_path=filter_db_empty))


ADVANCED_SELECTION_CASES = [
    ("has-thinking", lambda f: f.has("thinking"), {"conv-thinking"}),
    ("has-tools", lambda f: f.has("tools"), {"conv-tools"}),
    ("has-attachments", lambda f: f.has("attachments"), {"conv-attachments", "conv-multi-attach"}),
    ("has-summary", lambda f: f.has("summary"), {"conv-summary-only", "conv-thinking"}),
    ("provider-and-thinking", lambda f: f.provider("claude").has("thinking"), {"conv-thinking"}),
    ("exclude-provider-with-attachments", lambda f: f.exclude_provider("claude").has("attachments"), {"conv-attachments", "conv-multi-attach"}),
    ("contradictory-thinking-provider", lambda f: f.has("thinking").exclude_provider("claude"), set()),
]


SUMMARY_CASES = [
    ("provider", lambda f: f.provider("claude"), ["conv-long-messages", "conv-summary-only", "conv-tools", "conv-thinking"]),
    ("tag", lambda f: f.tag("analysis"), ["conv-multi-attach"]),
    ("exclude-tag", lambda f: f.exclude_tag("simple"), ["conv-long-messages", "conv-multi-attach", "conv-summary-only", "conv-attachments", "conv-tools", "conv-thinking"]),
    ("title", lambda f: f.title("Complex"), ["conv-thinking"]),
    ("has-summary", lambda f: f.has("summary"), ["conv-summary-only", "conv-thinking"]),
]


INVALID_SUMMARY_FILTERS = [
    (lambda f: f.has("thinking"), "Cannot use list_summaries"),
    (lambda f: f.exclude_text("error"), "Cannot use list_summaries"),
    (lambda f: f.where(lambda c: True), "Cannot use list_summaries"),
    (lambda f: f.sort("words"), "Cannot use list_summaries"),
]


SORT_FIELDS = {
    "tokens": lambda conv: sum(len(message.text or "") for message in conv.messages) // 4,
    "words": lambda conv: sum(message.word_count for message in conv.messages),
    "longest": lambda conv: max((message.word_count for message in conv.messages), default=0),
    "messages": lambda conv: len(conv.messages),
}


@pytest.fixture
def filter_db_with_branches(tmp_path):
    db_path = tmp_path / "filter_branches.db"
    (
        ConversationBuilder(db_path, "root-conv")
        .provider("claude")
        .title("Root Conversation")
        .add_message("m1", role="user", text="Initial question")
        .add_message("m2", role="assistant", text="Initial answer")
        .save()
    )
    (
        ConversationBuilder(db_path, "continuation-conv")
        .provider("claude")
        .title("Continuation")
        .parent_conversation("root-conv")
        .branch_type("continuation")
        .add_message("m3", role="user", text="Follow-up question")
        .add_message("m4", role="assistant", text="Follow-up answer")
        .save()
    )
    (
        ConversationBuilder(db_path, "sidechain-conv")
        .provider("claude")
        .title("Sidechain")
        .parent_conversation("root-conv")
        .branch_type("sidechain")
        .add_message("m5", role="user", text="Different direction")
        .add_message("m6", role="assistant", text="Sidechain answer")
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    return db_path


@pytest.fixture
def filter_repo_branches(filter_db_with_branches):
    return ConversationRepository(backend=SQLiteBackend(db_path=filter_db_with_branches))


BRANCH_CASES = [
    ("root", lambda f: f.is_root(), {"root-conv"}),
    ("continuation", lambda f: f.is_continuation(), {"continuation-conv"}),
    ("sidechain", lambda f: f.is_sidechain(), {"sidechain-conv"}),
    ("parent", lambda f: f.parent("root-conv"), {"continuation-conv", "sidechain-conv"}),
    ("not-root", lambda f: f.is_root(False), {"continuation-conv", "sidechain-conv"}),
    ("not-continuation", lambda f: f.is_continuation(False), {"root-conv", "sidechain-conv"}),
    ("not-sidechain", lambda f: f.is_sidechain(False), {"continuation-conv", "root-conv"}),
    ("has-branches", lambda f: f.has_branches(), set()),
]


@pytest.fixture
def populated_db(tmp_path):
    db_path = tmp_path / "cascade.db"
    (
        ConversationBuilder(db_path, "cascade-conv")
        .provider("claude")
        .title("Cascade Test")
        .add_message("m1", role="user", text="Hello world")
        .add_message("m2", role="assistant", text="Hi there")
        .add_attachment("att1", message_id="m1", mime_type="image/png", size_bytes=1024)
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    return db_path, ConversationRepository(backend=SQLiteBackend(db_path=db_path))


@pytest.fixture
def filter_db_semantic(tmp_path):
    db_path = tmp_path / "filter_semantic.db"
    (
        ConversationBuilder(db_path, "conv-file-ops")
        .provider("claude-code")
        .title("File editing session")
        .add_message("m1", role="user", text="Edit the config file")
        .add_message(
            "m2",
            role="assistant",
            text="Reading and updating config.",
            provider_meta={
                "content_blocks": [
                    {"type": "tool_use", "tool_name": "Read", "semantic_type": "file_read"},
                    {"type": "tool_use", "tool_name": "Edit", "semantic_type": "file_edit"},
                ]
            },
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-git-ops")
        .provider("claude-code")
        .title("Git commit session")
        .add_message("m3", role="user", text="Commit these changes")
        .add_message(
            "m4",
            role="assistant",
            text="Running git commit.",
            provider_meta={"content_blocks": [{"type": "tool_use", "tool_name": "Bash", "semantic_type": "git"}]},
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-subagent")
        .provider("claude-code")
        .title("Delegating to subagent")
        .add_message("m5", role="user", text="Explore the codebase")
        .add_message(
            "m6",
            role="assistant",
            text="Spawning exploration agent.",
            provider_meta={"content_blocks": [{"type": "tool_use", "tool_name": "Task", "semantic_type": "subagent"}]},
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-mixed")
        .provider("claude-code")
        .title("Complex coding task")
        .add_message("m7", role="user", text="Write and commit a new module")
        .add_message(
            "m8",
            role="assistant",
            text="Writing and committing.",
            provider_meta={
                "content_blocks": [
                    {"type": "tool_use", "tool_name": "Write", "semantic_type": "file_write"},
                    {"type": "tool_use", "tool_name": "Bash", "semantic_type": "git"},
                ]
            },
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-shell-only")
        .provider("claude-code")
        .title("Shell command")
        .add_message("m9", role="user", text="Run tests")
        .add_message(
            "m10",
            role="assistant",
            text="Running pytest.",
            provider_meta={"content_blocks": [{"type": "tool_use", "tool_name": "Bash", "semantic_type": "shell"}]},
        )
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    return db_path


@pytest.fixture
def filter_repo_semantic(filter_db_semantic):
    return ConversationRepository(backend=SQLiteBackend(db_path=filter_db_semantic))


SEMANTIC_CASES = [
    ("file-ops", lambda f: f.has_file_operations(), {"conv-file-ops", "conv-mixed"}),
    ("git-ops", lambda f: f.has_git_operations(), {"conv-git-ops", "conv-mixed"}),
    ("subagent", lambda f: f.has_subagent_spawns(), {"conv-subagent"}),
    ("git-and-file", lambda f: f.has_git_operations().has_file_operations(), {"conv-mixed"}),
    ("subagent-and-git", lambda f: f.has_subagent_spawns().has_git_operations(), set()),
    ("provider-and-git", lambda f: f.provider("claude-code").has_git_operations(), {"conv-git-ops", "conv-mixed"}),
]


class TestAdvancedConversationFilterContracts:
    @pytest.mark.parametrize(("case_name", "build_filter", "expected_ids"), ADVANCED_SELECTION_CASES)
    @pytest.mark.asyncio
    async def test_advanced_selection_matrix(self, filter_repo_advanced, case_name, build_filter, expected_ids):
        result = await build_filter(ConversationFilter(filter_repo_advanced)).list()
        assert {conversation.id for conversation in result} == expected_ids, case_name

    @pytest.mark.asyncio
    async def test_unknown_has_type_is_a_noop(self, filter_repo_advanced):
        result = await ConversationFilter(filter_repo_advanced).has("nonexistent_type").list()
        baseline = await ConversationFilter(filter_repo_advanced).list()
        assert [conversation.id for conversation in result] == [conversation.id for conversation in baseline]

    @pytest.mark.parametrize(("sort_field", "reverse"), [(field, reverse) for field in SORT_FIELDS for reverse in (False, True)])
    @pytest.mark.asyncio
    async def test_sort_projection_contract_matrix(self, filter_repo_advanced, sort_field, reverse):
        filter_obj = ConversationFilter(filter_repo_advanced).sort(sort_field)
        if reverse:
            filter_obj = filter_obj.reverse()

        result = await filter_obj.list()
        metrics = [SORT_FIELDS[sort_field](conversation) for conversation in result]
        comparator = all(metrics[index] <= metrics[index + 1] for index in range(len(metrics) - 1)) if reverse else all(
            metrics[index] >= metrics[index + 1] for index in range(len(metrics) - 1)
        )
        assert comparator

    @pytest.mark.parametrize(
        ("sample_size", "build_filter"),
        [
            (0, lambda f: f),
            (1, lambda f: f),
            (2, lambda f: f.provider("claude")),
            (9999, lambda f: f),
        ],
    )
    @pytest.mark.asyncio
    async def test_sample_contract_matrix(self, filter_repo_advanced, sample_size, build_filter):
        population = await build_filter(ConversationFilter(filter_repo_advanced)).list()
        sampled = await build_filter(ConversationFilter(filter_repo_advanced)).sample(sample_size).list()
        assert len(sampled) <= min(sample_size, len(population))
        assert {conversation.id for conversation in sampled}.issubset({conversation.id for conversation in population})

    @pytest.mark.parametrize(
        "build_filter",
        [
            lambda f: f.limit(0).sample(5),
            lambda f: f.sample(5).limit(0),
            lambda f: f.sort("messages").limit(0),
            lambda f: f.provider("claude").tag("quantum").sort("date").sample(10).limit(0),
        ],
    )
    @pytest.mark.asyncio
    async def test_limit_zero_short_circuits_pipeline(self, filter_repo_advanced, build_filter):
        assert await build_filter(ConversationFilter(filter_repo_advanced)).list() == []

    @pytest.mark.parametrize(("case_name", "build_filter", "expected_ids"), SUMMARY_CASES)
    @pytest.mark.asyncio
    async def test_list_summaries_contract_matrix(self, filter_repo_advanced, case_name, build_filter, expected_ids):
        summaries = await build_filter(ConversationFilter(filter_repo_advanced)).list_summaries()
        assert [summary.id for summary in summaries] == expected_ids, case_name
        assert all(isinstance(summary, ConversationSummary) for summary in summaries)

    @pytest.mark.asyncio
    async def test_list_summaries_preserves_sample_and_limit_bounds(self, filter_repo_advanced):
        sampled = await ConversationFilter(filter_repo_advanced).sample(2).list_summaries()
        limited = await ConversationFilter(filter_repo_advanced).limit(2).list_summaries()
        assert len(sampled) <= 2
        assert len(limited) <= 2

    @pytest.mark.parametrize(("build_filter", "error_match"), INVALID_SUMMARY_FILTERS)
    @pytest.mark.asyncio
    async def test_list_summaries_rejects_content_dependent_filters(self, filter_repo_advanced, build_filter, error_match):
        with pytest.raises(ValueError, match=error_match):
            await build_filter(ConversationFilter(filter_repo_advanced)).list_summaries()

    @pytest.mark.asyncio
    async def test_list_summaries_rejection_message_is_exact(self, filter_repo_advanced):
        with pytest.raises(ValueError) as exc_info:
            await ConversationFilter(filter_repo_advanced).has("thinking").list_summaries()

        assert str(exc_info.value) == (
            "Cannot use list_summaries() with content-dependent filters "
            "(regex, has:thinking, has:tools, etc.). Use list() instead."
        )

    def test_apply_summary_filters_contract(self, filter_repo_advanced):
        filter_obj = ConversationFilter(filter_repo_advanced).provider("claude")
        summaries = [
            ConversationSummary(id="claude-hit", provider="claude"),
            ConversationSummary(id="chatgpt-hit", provider="chatgpt"),
        ]
        assert [summary.id for summary in filter_obj._apply_summary_filters(summaries)] == ["claude-hit"]

    @pytest.mark.asyncio
    async def test_list_summaries_applies_provider_filter_after_summary_search(self, filter_repo_advanced):
        filter_obj = ConversationFilter(filter_repo_advanced).contains("needle").provider("claude")
        filter_obj._fetch_summary_candidates = AsyncMock(return_value=[ConversationSummary(id="claude-hit", provider="claude"), ConversationSummary(id="chatgpt-hit", provider="chatgpt")])  # type: ignore[method-assign]
        result = await filter_obj.list_summaries()
        assert [summary.id for summary in result] == ["claude-hit"]

    @pytest.mark.parametrize(
        ("method_name", "expected"),
        [
            ("list", []),
            ("list_summaries", []),
            ("first", None),
            ("count", 0),
            ("delete", 0),
            ("pick", None),
        ],
    )
    @pytest.mark.asyncio
    async def test_empty_repository_terminal_contract(self, filter_repo_empty, method_name, expected):
        result = await getattr(ConversationFilter(filter_repo_empty), method_name)()
        assert result == expected

    @pytest.mark.asyncio
    async def test_empty_repository_with_filters_stays_empty(self, filter_repo_empty):
        result = await ConversationFilter(filter_repo_empty).provider("claude").tag("python").has("thinking").list()
        assert result == []


class TestBranchingAndMutationContracts:
    @pytest.mark.parametrize(("case_name", "build_filter", "expected_ids"), BRANCH_CASES)
    @pytest.mark.asyncio
    async def test_branching_contract_matrix(self, filter_repo_branches, case_name, build_filter, expected_ids):
        result = await build_filter(ConversationFilter(filter_repo_branches)).list()
        assert {conversation.id for conversation in result} == expected_ids, case_name

    @pytest.mark.asyncio
    async def test_delete_cascade_contract(self, populated_db):
        db_path, repo = populated_db
        deleted = await ConversationFilter(repo).id("cascade-conv").delete()
        assert deleted == 1

        with open_connection(db_path) as conn:
            counts = {
                "messages": conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = 'cascade-conv'").fetchone()[0],
                "refs": conn.execute("SELECT COUNT(*) FROM attachment_refs WHERE conversation_id = 'cascade-conv'").fetchone()[0],
                "fts": conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = 'cascade-conv'").fetchone()[0],
                "attachments": conn.execute("SELECT COUNT(*) FROM attachments WHERE attachment_id = 'att1'").fetchone()[0],
            }
        assert counts == {"messages": 0, "refs": 0, "fts": 0, "attachments": 0}

    @pytest.mark.parametrize(("case_name", "build_filter", "expected_ids"), SEMANTIC_CASES)
    @pytest.mark.asyncio
    async def test_semantic_filter_contract_matrix(self, filter_repo_semantic, case_name, build_filter, expected_ids):
        result = await build_filter(ConversationFilter(filter_repo_semantic)).list()
        assert {conversation.id for conversation in result} == expected_ids, case_name
