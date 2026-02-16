"""Tests for session continuation and message branching support.

These tests verify the unified branching model that works across all providers:
- Session continuations (Codex child sessions)
- Sidechains (Claude Code)
- Message branches (ChatGPT edits)

Also includes full-pipeline integration tests that exercise the REAL pipeline:
raw JSON/JSONL → parser → prepare_records → database.

This ensures branching metadata extraction and resolution actually works end-to-end.

Note on filename conventions:
- Files named `claude_code_*.json` or `claude-code_*.json` → detected as claude-code (grouped)
- Files named `codex_*.json` → detected as codex (grouped)
- Without the provider name in filename, detection depends on content structure
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.prepare import prepare_records
from polylogue.sources import iter_source_conversations
import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.repository import ConversationRepository
from tests.infra.helpers import ConversationBuilder, db_setup


def _make_test_repository(db_path: Path) -> ConversationRepository:
    """Create a repository for testing with a specific db_path."""
    backend = SQLiteBackend(db_path=db_path)
    return ConversationRepository(backend=backend)


class TestClaudeCodeSidechain:
    """Test Claude Code sidechain detection."""

    @pytest.mark.asyncio
    async def test_sidechain_detected_from_branch_type(self, workspace_env: Path) -> None:
        """Conversations with branch_type='sidechain' should have is_sidechain=True."""
        db_path = db_setup(workspace_env)

        # Create a sidechain conversation (as would be detected during import)
        conv = await (
            ConversationBuilder(db_path, "sidechain-test")
            .provider("claude-code")
            .title("Sidechain Test")
            .branch_type("sidechain")  # Set by parser when isSidechain detected
            .add_message("m1", role="user", text="Hello")
            .add_message("m2", role="assistant", text="Hi", provider_meta={"isSidechain": True})
            .async_build()
        )

        assert conv.branch_type == "sidechain"
        assert conv.is_sidechain is True
        assert conv.is_continuation is False
        assert conv.is_root is True  # No parent

    @pytest.mark.asyncio
    async def test_non_sidechain_has_no_branch_type(self, workspace_env: Path) -> None:
        """Normal conversations should have no branch_type."""
        db_path = db_setup(workspace_env)

        conv = await (
            ConversationBuilder(db_path, "normal-test")
            .provider("claude-code")
            .title("Normal Test")
            .add_message("m1", role="user", text="Hello")
            .add_message("m2", role="assistant", text="Hi")
            .async_build()
        )

        assert conv.branch_type is None
        assert conv.is_sidechain is False
        assert conv.is_root is True


class TestChatGPTBranching:
    """Test ChatGPT message branching (edit branches)."""

    @pytest.mark.asyncio
    async def test_branch_index_extracted(self, workspace_env: Path) -> None:
        """Messages should have correct branch_index based on position in parent's children."""
        db_path = db_setup(workspace_env)

        # Create conversation with branching messages
        conv = await (
            ConversationBuilder(db_path, "branch-test")
            .provider("chatgpt")
            .title("Branch Test")
            .add_message("m1", role="user", text="Question", branch_index=0)
            .add_message("m2", role="assistant", text="Answer 1", parent_message_id="m1", branch_index=0)
            .add_message("m3", role="assistant", text="Answer 2 (edited)", parent_message_id="m1", branch_index=1)
            .async_build()
        )

        messages = conv.messages
        assert len(messages) == 3

        # Check branch indexes
        m1 = next(m for m in messages if "Question" in (m.text or ""))
        m2 = next(m for m in messages if "Answer 1" in (m.text or ""))
        m3 = next(m for m in messages if "Answer 2" in (m.text or ""))

        assert m1.branch_index == 0
        assert m2.branch_index == 0
        assert m2.is_branch is False  # mainline
        assert m3.branch_index == 1
        assert m3.is_branch is True  # branch

    @pytest.mark.asyncio
    async def test_mainline_messages_filters_branches(self, workspace_env: Path) -> None:
        """mainline_messages() should return only branch_index=0 messages."""
        db_path = db_setup(workspace_env)

        conv = await (
            ConversationBuilder(db_path, "mainline-test")
            .provider("chatgpt")
            .title("Mainline Test")
            .add_message("m1", role="user", text="Q1", branch_index=0)
            .add_message("m2", role="assistant", text="A1", branch_index=0)
            .add_message("m3", role="assistant", text="A1-alt", branch_index=1)
            .add_message("m4", role="user", text="Q2", branch_index=0)
            .async_build()
        )

        mainline = conv.mainline_messages()
        assert len(mainline) == 3  # m1, m2, m4
        assert all(m.branch_index == 0 for m in mainline)


class TestCodexContinuation:
    """Test Codex session continuation detection."""

    @pytest.mark.asyncio
    async def test_continuation_with_parent_session(self, workspace_env: Path) -> None:
        """Child sessions should have parent_id and branch_type='continuation'."""
        db_path = db_setup(workspace_env)

        # Create parent session
        parent = await (
            ConversationBuilder(db_path, "parent-session")
            .provider("codex")
            .title("Parent Session")
            .add_message("m1", role="user", text="Start")
            .async_build()
        )

        # Create child session with parent reference
        child = await (
            ConversationBuilder(db_path, "child-session")
            .provider("codex")
            .title("Child Session")
            .parent_conversation(str(parent.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue")
            .async_build()
        )

        assert child.parent_id == parent.id
        assert child.branch_type == "continuation"
        assert child.is_continuation is True
        assert child.is_root is False


class TestTreeTraversal:
    """Test session tree traversal methods."""

    async def test_get_parent(self, workspace_env: Path) -> None:
        """get_parent should return the parent conversation."""
        db_path = db_setup(workspace_env)

        parent = await (
            ConversationBuilder(db_path, "parent")
            .provider("codex")
            .add_message("m1", role="user", text="Start")
            .async_build()
        )

        child = await (
            ConversationBuilder(db_path, "child")
            .provider("codex")
            .parent_conversation(str(parent.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue")
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)
        found_parent = await _repo.get_parent(str(child.id))

        assert found_parent is not None
        assert found_parent.id == parent.id

    async def test_get_children(self, workspace_env: Path) -> None:
        """get_children should return all direct children."""
        db_path = db_setup(workspace_env)

        parent = await (
            ConversationBuilder(db_path, "parent")
            .provider("codex")
            .add_message("m1", role="user", text="Start")
            .async_build()
        )

        child1 = await (
            ConversationBuilder(db_path, "child1")
            .provider("codex")
            .parent_conversation(str(parent.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue 1")
            .async_build()
        )

        child2 = await (
            ConversationBuilder(db_path, "child2")
            .provider("codex")
            .parent_conversation(str(parent.id))
            .branch_type("sidechain")
            .add_message("m3", role="user", text="Branch")
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)
        children = await _repo.get_children(str(parent.id))

        assert len(children) == 2
        child_ids = {c.id for c in children}
        assert child1.id in child_ids
        assert child2.id in child_ids

    async def test_get_root(self, workspace_env: Path) -> None:
        """get_root should walk up to find the root conversation."""
        db_path = db_setup(workspace_env)

        root = await (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .async_build()
        )

        middle = await (
            ConversationBuilder(db_path, "middle")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Middle")
            .async_build()
        )

        leaf = await (
            ConversationBuilder(db_path, "leaf")
            .provider("codex")
            .parent_conversation(str(middle.id))
            .branch_type("continuation")
            .add_message("m3", role="user", text="Leaf")
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)

        # From leaf, should find root
        found_root = await _repo.get_root(str(leaf.id))
        assert found_root is not None
        assert found_root.id == root.id

        # From root, should return itself
        found_root2 = await _repo.get_root(str(root.id))
        assert found_root2 is not None
        assert found_root2.id == root.id

    async def test_get_session_tree(self, workspace_env: Path) -> None:
        """get_session_tree should return all conversations in the tree."""
        db_path = db_setup(workspace_env)

        root = await (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .async_build()
        )

        child1 = await (
            ConversationBuilder(db_path, "child1")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Child 1")
            .async_build()
        )

        child2 = await (
            ConversationBuilder(db_path, "child2")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("sidechain")
            .add_message("m3", role="user", text="Child 2")
            .async_build()
        )

        grandchild = await (
            ConversationBuilder(db_path, "grandchild")
            .provider("codex")
            .parent_conversation(str(child1.id))
            .branch_type("continuation")
            .add_message("m4", role="user", text="Grandchild")
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)

        # Get tree from any node
        tree = await _repo.get_session_tree(str(grandchild.id))

        assert len(tree) == 4
        tree_ids = {c.id for c in tree}
        assert root.id in tree_ids
        assert child1.id in tree_ids
        assert child2.id in tree_ids
        assert grandchild.id in tree_ids


class TestBranchingFilters:
    """Test branching filter methods."""

    @pytest.mark.asyncio
    async def test_filter_is_continuation(self, workspace_env: Path) -> None:
        """is_continuation filter should find continuation conversations."""
        db_path = db_setup(workspace_env)

        root = await (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .async_build()
        )

        cont = await (
            ConversationBuilder(db_path, "continuation")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue")
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)

        continuations = await _repo.filter().is_continuation().list()
        assert len(continuations) == 1
        assert continuations[0].id == cont.id

    @pytest.mark.asyncio
    async def test_filter_is_sidechain(self, workspace_env: Path) -> None:
        """is_sidechain filter should find sidechain conversations."""
        db_path = db_setup(workspace_env)

        # Normal conversation
        await (
            ConversationBuilder(db_path, "normal")
            .provider("claude-code")
            .add_message("m1", role="user", text="Normal")
            .async_build()
        )

        # Sidechain conversation
        sidechain = await (
            ConversationBuilder(db_path, "sidechain")
            .provider("claude-code")
            .branch_type("sidechain")
            .add_message("m2", role="user", text="Side", provider_meta={"isSidechain": True})
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)

        sidechains = await _repo.filter().is_sidechain().list()
        assert len(sidechains) == 1
        assert sidechains[0].id == sidechain.id

    @pytest.mark.asyncio
    async def test_filter_is_root(self, workspace_env: Path) -> None:
        """is_root filter should find root conversations only."""
        db_path = db_setup(workspace_env)

        root = await (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .async_build()
        )

        await (
            ConversationBuilder(db_path, "child")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Child")
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)

        roots = await _repo.filter().is_root().list()
        assert len(roots) == 1
        assert roots[0].id == root.id

    @pytest.mark.asyncio
    async def test_filter_has_branches(self, workspace_env: Path) -> None:
        """has_branches filter should find conversations with branching messages."""
        db_path = db_setup(workspace_env)

        # Conversation without branches
        await (
            ConversationBuilder(db_path, "linear")
            .provider("chatgpt")
            .add_message("m1", role="user", text="Q", branch_index=0)
            .add_message("m2", role="assistant", text="A", branch_index=0)
            .async_build()
        )

        # Conversation with branches
        branching = await (
            ConversationBuilder(db_path, "branching")
            .provider("chatgpt")
            .add_message("m3", role="user", text="Q", branch_index=0)
            .add_message("m4", role="assistant", text="A1", branch_index=0)
            .add_message("m5", role="assistant", text="A2", branch_index=1)
            .async_build()
        )

        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)

        with_branches = await _repo.filter().has_branches().list()
        assert len(with_branches) == 1
        assert with_branches[0].id == branching.id


class TestCodexContinuationPipeline:
    """Test Codex continuation (child session) through the full pipeline."""

    def test_codex_with_two_session_metas_extracts_parent(self, tmp_path: Path):
        """Codex parser: two session_metas → parent_conversation_provider_id set."""
        # Child session includes its own session_meta first, then parent's
        payload = [
            {"type": "session_meta", "payload": {"id": "child-session-uuid", "timestamp": "2025-01-02T10:00:00Z"}},
            {"type": "session_meta", "payload": {"id": "parent-session-uuid", "timestamp": "2025-01-01T10:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg-1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Continue from parent"}],
                },
            },
        ]

        source_file = tmp_path / "codex_child_session.json"
        source_file.write_text(json.dumps(payload), encoding="utf-8")

        source = Source(name="codex", path=source_file)
        conversations = list(iter_source_conversations(source))

        assert len(conversations) == 1
        convo = conversations[0]

        # Verify parser extracted branching fields
        assert convo.provider_conversation_id == "child-session-uuid"
        assert convo.parent_conversation_provider_id == "parent-session-uuid"
        assert convo.branch_type == "continuation"

    async def test_codex_continuation_persisted_to_database(self, workspace_env: Path, tmp_path: Path):
        """Full pipeline: Codex child session → correct parent_conversation_id in DB."""
        db_path = db_setup(workspace_env)
        repository = _make_test_repository(db_path)

        # First, create and ingest the parent session
        parent_payload = [
            {"type": "session_meta", "payload": {"id": "parent-uuid", "timestamp": "2025-01-01T10:00:00Z"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "p-msg-1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Parent question"}],
                },
            },
        ]

        # Use codex_ prefix for proper provider detection
        parent_file = tmp_path / "codex_parent.json"
        parent_file.write_text(json.dumps(parent_payload), encoding="utf-8")

        source = Source(name="codex", path=parent_file)
        parent_convos = list(iter_source_conversations(source))
        assert len(parent_convos) == 1

        parent_cid, _, _ = await prepare_records(
            parent_convos[0],
            source_name="codex",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Now create and ingest the child session (continuation)
        child_payload = [
            {"type": "session_meta", "payload": {"id": "child-uuid", "timestamp": "2025-01-02T10:00:00Z"}},
            {
                "type": "session_meta",
                "payload": {"id": "parent-uuid", "timestamp": "2025-01-01T10:00:00Z"},
            },  # Parent context
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "c-msg-1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Continue from parent"}],
                },
            },
        ]

        child_file = tmp_path / "codex_child.json"
        child_file.write_text(json.dumps(child_payload), encoding="utf-8")

        source = Source(name="codex", path=child_file)
        child_convos = list(iter_source_conversations(source))
        assert len(child_convos) == 1

        # Verify parser extracted parent reference
        child_parsed = child_convos[0]
        assert child_parsed.parent_conversation_provider_id == "parent-uuid"
        assert child_parsed.branch_type == "continuation"

        # Ingest through full pipeline
        child_cid, _, _ = await prepare_records(
            child_parsed,
            source_name="codex",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Verify via domain model
        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)
        child_conv = await _repo.get(child_cid)

        assert child_conv is not None
        assert child_conv.parent_id == parent_cid  # Resolved to internal ID
        assert child_conv.branch_type == "continuation"
        assert child_conv.is_continuation is True
        assert child_conv.is_root is False


class TestClaudeCodeSidechainPipeline:
    """Test Claude Code sidechain detection through the full pipeline."""

    def test_claude_code_sidechain_flag_extracts_branch_type(self, tmp_path: Path):
        """Claude Code parser: isSidechain in message → branch_type='sidechain'."""
        payload = [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-1",
                "message": {"content": "User message"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-1",
                "isSidechain": True,  # This marks it as a sidechain
                "message": {"content": "Assistant in sidechain"},
            },
        ]

        # IMPORTANT: Use claude_code_ prefix for proper provider detection
        # Without this, "claude" in filename → wrong provider, list gets unpacked
        source_file = tmp_path / "claude_code_sidechain.json"
        source_file.write_text(json.dumps(payload), encoding="utf-8")

        source = Source(name="claude-code", path=source_file)
        conversations = list(iter_source_conversations(source))

        assert len(conversations) == 1
        convo = conversations[0]

        # Verify parser detected sidechain
        assert convo.branch_type == "sidechain"

    async def test_claude_code_sidechain_persisted_to_database(self, workspace_env: Path, tmp_path: Path):
        """Full pipeline: Claude Code sidechain → branch_type stored in DB."""
        db_path = db_setup(workspace_env)
        repository = _make_test_repository(db_path)

        payload = [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sidechain-session",
                "message": {"content": "User in sidechain"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sidechain-session",
                "isSidechain": True,
                "message": {"content": "Assistant in sidechain"},
            },
        ]

        # Use claude_code_ prefix for proper provider detection
        source_file = tmp_path / "claude_code_sidechain.json"
        source_file.write_text(json.dumps(payload), encoding="utf-8")

        source = Source(name="claude-code", path=source_file)
        conversations = list(iter_source_conversations(source))
        assert len(conversations) == 1

        parsed = conversations[0]
        assert parsed.branch_type == "sidechain"

        # Ingest through full pipeline
        cid, _, _ = await prepare_records(
            parsed,
            source_name="claude-code",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Verify via domain model
        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)
        conv = await _repo.get(cid)

        assert conv is not None
        assert conv.branch_type == "sidechain"
        assert conv.is_sidechain is True


class TestChatGPTBranchingPipeline:
    """Test ChatGPT message branching through the full pipeline."""

    def test_chatgpt_branch_index_extracted(self, tmp_path: Path):
        """ChatGPT parser: multiple children → correct branch_index calculated."""
        # Create a mapping where one parent has multiple children (branches)
        payload = {
            "title": "Branched Conversation",
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "children": ["user-q"],
                },
                "user-q": {
                    "id": "user-q",
                    "parent": "root",
                    "children": ["asst-a1", "asst-a2"],  # Two branches!
                    "message": {
                        "id": "user-q",
                        "author": {"role": "user"},
                        "content": {"parts": ["What is 2+2?"]},
                        "create_time": 1700000000,
                    },
                },
                "asst-a1": {
                    "id": "asst-a1",
                    "parent": "user-q",
                    "children": [],
                    "message": {
                        "id": "asst-a1",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["4"]},
                        "create_time": 1700000001,
                    },
                },
                "asst-a2": {
                    "id": "asst-a2",
                    "parent": "user-q",
                    "children": [],
                    "message": {
                        "id": "asst-a2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["The answer is four."]},
                        "create_time": 1700000002,
                    },
                },
            },
        }

        source_file = tmp_path / "chatgpt_branched.json"
        source_file.write_text(json.dumps([payload]), encoding="utf-8")

        source = Source(name="chatgpt", path=source_file)
        conversations = list(iter_source_conversations(source))

        assert len(conversations) == 1
        convo = conversations[0]

        # Find the branching messages
        messages_by_id = {m.provider_message_id: m for m in convo.messages}

        # User question has parent_message_provider_id pointing to root
        user_msg = messages_by_id.get("user-q")
        assert user_msg is not None
        assert user_msg.parent_message_provider_id == "root"
        assert user_msg.branch_index == 0  # Only child of root

        # First assistant answer: branch_index=0 (first in parent's children)
        asst1 = messages_by_id.get("asst-a1")
        assert asst1 is not None
        assert asst1.parent_message_provider_id == "user-q"
        assert asst1.branch_index == 0

        # Second assistant answer: branch_index=1 (second in parent's children)
        asst2 = messages_by_id.get("asst-a2")
        assert asst2 is not None
        assert asst2.parent_message_provider_id == "user-q"
        assert asst2.branch_index == 1

    async def test_chatgpt_branching_persisted_to_database(self, workspace_env: Path, tmp_path: Path):
        """Full pipeline: ChatGPT branches → parent_message_id and branch_index stored."""
        db_path = db_setup(workspace_env)
        repository = _make_test_repository(db_path)

        payload = {
            "title": "Branched Test",
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["q1"]},
                "q1": {
                    "id": "q1",
                    "parent": "root",
                    "children": ["a1", "a2"],
                    "message": {
                        "id": "q1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Question"]},
                        "create_time": 1700000000,
                    },
                },
                "a1": {
                    "id": "a1",
                    "parent": "q1",
                    "children": [],
                    "message": {
                        "id": "a1",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Answer 1"]},
                        "create_time": 1700000001,
                    },
                },
                "a2": {
                    "id": "a2",
                    "parent": "q1",
                    "children": [],
                    "message": {
                        "id": "a2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Answer 2 (regenerated)"]},
                        "create_time": 1700000002,
                    },
                },
            },
        }

        source_file = tmp_path / "chatgpt_branched.json"
        source_file.write_text(json.dumps([payload]), encoding="utf-8")

        source = Source(name="chatgpt", path=source_file)
        conversations = list(iter_source_conversations(source))
        assert len(conversations) == 1

        parsed = conversations[0]

        # Ingest through full pipeline
        cid, _, _ = await prepare_records(
            parsed,
            source_name="chatgpt",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Verify via domain model
        _backend = SQLiteBackend(db_path=db_path)
        _repo = ConversationRepository(backend=_backend)
        conv = await _repo.get(cid)

        assert conv is not None
        messages = conv.messages

        # Find messages by text
        a1 = next((m for m in messages if "Answer 1" in (m.text or "")), None)
        a2 = next((m for m in messages if "Answer 2" in (m.text or "")), None)

        assert a1 is not None
        assert a2 is not None

        # Verify branch indexes persisted correctly
        assert a1.branch_index == 0  # First child
        assert a2.branch_index == 1  # Second child (branch)

        # Verify is_branch property
        assert a1.is_branch is False  # mainline
        assert a2.is_branch is True  # branch

        # Verify mainline_messages filters correctly
        mainline = conv.mainline_messages()
        mainline_texts = [m.text for m in mainline]
        assert "Answer 1" in str(mainline_texts)
        assert "Answer 2" not in str(mainline_texts)


class TestPrepareRecordsParentResolution:
    """Test that prepare_records correctly resolves provider IDs to internal IDs."""

    async def test_parent_conversation_id_resolved_to_internal_format(self, workspace_env: Path, tmp_path: Path):
        """parent_conversation_provider_id gets hashed to polylogue internal ID format."""
        db_path = db_setup(workspace_env)
        repository = _make_test_repository(db_path)

        from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage

        # First create the parent conversation (FK constraint requires parent to exist)
        parent_parsed = ParsedConversation(
            provider_name="codex",
            provider_conversation_id="parent-id",
            title="Parent Session",
            messages=[
                ParsedMessage(provider_message_id="pm1", role="user", text="Parent message"),
            ],
        )

        parent_cid, _, _ = await prepare_records(
            parent_parsed,
            source_name="codex",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Now create a child ParsedConversation with parent reference
        child_parsed = ParsedConversation(
            provider_name="codex",
            provider_conversation_id="child-id",
            title="Child Session",
            messages=[
                ParsedMessage(provider_message_id="m1", role="user", text="Hello"),
            ],
            parent_conversation_provider_id="parent-id",
            branch_type="continuation",
        )

        child_cid, _, _ = await prepare_records(
            child_parsed,
            source_name="codex",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Query database directly to check parent_conversation_id format
        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_conversation_id, branch_type FROM conversations WHERE conversation_id = ?",
                (child_cid,),
            ).fetchone()

        # Parent ID should be in internal format: "provider:provider_id_hash"
        assert row["parent_conversation_id"] is not None
        assert row["parent_conversation_id"].startswith("codex:")  # Internal format
        assert row["parent_conversation_id"] == parent_cid  # Should match actual parent
        assert row["branch_type"] == "continuation"

    async def test_parent_message_id_resolved_within_conversation(self, workspace_env: Path, tmp_path: Path):
        """parent_message_provider_id gets resolved to internal message ID."""
        db_path = db_setup(workspace_env)
        repository = _make_test_repository(db_path)

        from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage

        # Create messages with parent references (like ChatGPT branches)
        parsed = ParsedConversation(
            provider_name="chatgpt",
            provider_conversation_id="conv-1",
            title="Test",
            messages=[
                ParsedMessage(provider_message_id="q1", role="user", text="Question"),
                ParsedMessage(
                    provider_message_id="a1",
                    role="assistant",
                    text="Answer 1",
                    parent_message_provider_id="q1",
                    branch_index=0,
                ),
                ParsedMessage(
                    provider_message_id="a2",
                    role="assistant",
                    text="Answer 2",
                    parent_message_provider_id="q1",
                    branch_index=1,
                ),
            ],
        )

        cid, _, _ = await prepare_records(
            parsed,
            source_name="chatgpt",
            archive_root=tmp_path,
            backend=repository.backend,
            repository=repository,
        )

        # Query database directly
        with open_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT provider_message_id, parent_message_id, branch_index
                FROM messages
                WHERE conversation_id = ?
                ORDER BY provider_message_id
                """,
                (cid,),
            ).fetchall()

        msg_by_provider_id = {r["provider_message_id"]: r for r in rows}

        # a1 and a2 should have parent_message_id pointing to q1's internal ID
        q1_row = msg_by_provider_id["q1"]
        a1_row = msg_by_provider_id["a1"]
        a2_row = msg_by_provider_id["a2"]

        # q1 has no parent
        assert q1_row["parent_message_id"] is None

        # a1 and a2 have parent pointing to q1's internal message ID
        # The internal ID format is derived from conversation_id + provider_message_id
        assert a1_row["parent_message_id"] is not None
        assert a2_row["parent_message_id"] is not None
        assert a1_row["parent_message_id"] == a2_row["parent_message_id"]  # Same parent

        # Branch indexes preserved
        assert a1_row["branch_index"] == 0
        assert a2_row["branch_index"] == 1
