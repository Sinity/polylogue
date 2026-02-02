"""Tests for session continuation and message branching support.

These tests verify the unified branching model that works across all providers:
- Session continuations (Codex child sessions)
- Sidechains (Claude Code)
- Message branches (ChatGPT edits)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from tests.helpers import ConversationBuilder, db_setup


class TestClaudeCodeSidechain:
    """Test Claude Code sidechain detection."""

    def test_sidechain_detected_from_branch_type(self, workspace_env: Path) -> None:
        """Conversations with branch_type='sidechain' should have is_sidechain=True."""
        db_path = db_setup(workspace_env)

        # Create a sidechain conversation (as would be detected during import)
        conv = (
            ConversationBuilder(db_path, "sidechain-test")
            .provider("claude-code")
            .title("Sidechain Test")
            .branch_type("sidechain")  # Set by importer when isSidechain detected
            .add_message("m1", role="user", text="Hello")
            .add_message("m2", role="assistant", text="Hi", provider_meta={"isSidechain": True})
            .build()
        )

        assert conv.branch_type == "sidechain"
        assert conv.is_sidechain is True
        assert conv.is_continuation is False
        assert conv.is_root is True  # No parent

    def test_non_sidechain_has_no_branch_type(self, workspace_env: Path) -> None:
        """Normal conversations should have no branch_type."""
        db_path = db_setup(workspace_env)

        conv = (
            ConversationBuilder(db_path, "normal-test")
            .provider("claude-code")
            .title("Normal Test")
            .add_message("m1", role="user", text="Hello")
            .add_message("m2", role="assistant", text="Hi")
            .build()
        )

        assert conv.branch_type is None
        assert conv.is_sidechain is False
        assert conv.is_root is True


class TestChatGPTBranching:
    """Test ChatGPT message branching (edit branches)."""

    def test_branch_index_extracted(self, workspace_env: Path) -> None:
        """Messages should have correct branch_index based on position in parent's children."""
        db_path = db_setup(workspace_env)

        # Create conversation with branching messages
        conv = (
            ConversationBuilder(db_path, "branch-test")
            .provider("chatgpt")
            .title("Branch Test")
            .add_message("m1", role="user", text="Question", branch_index=0)
            .add_message("m2", role="assistant", text="Answer 1", parent_message_id="m1", branch_index=0)
            .add_message("m3", role="assistant", text="Answer 2 (edited)", parent_message_id="m1", branch_index=1)
            .build()
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

    def test_mainline_messages_filters_branches(self, workspace_env: Path) -> None:
        """mainline_messages() should return only branch_index=0 messages."""
        db_path = db_setup(workspace_env)

        conv = (
            ConversationBuilder(db_path, "mainline-test")
            .provider("chatgpt")
            .title("Mainline Test")
            .add_message("m1", role="user", text="Q1", branch_index=0)
            .add_message("m2", role="assistant", text="A1", branch_index=0)
            .add_message("m3", role="assistant", text="A1-alt", branch_index=1)
            .add_message("m4", role="user", text="Q2", branch_index=0)
            .build()
        )

        mainline = conv.mainline_messages()
        assert len(mainline) == 3  # m1, m2, m4
        assert all(m.branch_index == 0 for m in mainline)


class TestCodexContinuation:
    """Test Codex session continuation detection."""

    def test_continuation_with_parent_session(self, workspace_env: Path) -> None:
        """Child sessions should have parent_id and branch_type='continuation'."""
        db_path = db_setup(workspace_env)

        # Create parent session
        parent = (
            ConversationBuilder(db_path, "parent-session")
            .provider("codex")
            .title("Parent Session")
            .add_message("m1", role="user", text="Start")
            .build()
        )

        # Create child session with parent reference
        child = (
            ConversationBuilder(db_path, "child-session")
            .provider("codex")
            .title("Child Session")
            .parent_conversation(str(parent.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue")
            .build()
        )

        assert child.parent_id == parent.id
        assert child.branch_type == "continuation"
        assert child.is_continuation is True
        assert child.is_root is False


class TestTreeTraversal:
    """Test session tree traversal methods."""

    def test_get_parent(self, workspace_env: Path) -> None:
        """get_parent should return the parent conversation."""
        db_path = db_setup(workspace_env)

        parent = (
            ConversationBuilder(db_path, "parent")
            .provider("codex")
            .add_message("m1", role="user", text="Start")
            .build()
        )

        child = (
            ConversationBuilder(db_path, "child")
            .provider("codex")
            .parent_conversation(str(parent.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue")
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)
        found_parent = p.repository.get_parent(str(child.id))

        assert found_parent is not None
        assert found_parent.id == parent.id

    def test_get_children(self, workspace_env: Path) -> None:
        """get_children should return all direct children."""
        db_path = db_setup(workspace_env)

        parent = (
            ConversationBuilder(db_path, "parent")
            .provider("codex")
            .add_message("m1", role="user", text="Start")
            .build()
        )

        child1 = (
            ConversationBuilder(db_path, "child1")
            .provider("codex")
            .parent_conversation(str(parent.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue 1")
            .build()
        )

        child2 = (
            ConversationBuilder(db_path, "child2")
            .provider("codex")
            .parent_conversation(str(parent.id))
            .branch_type("sidechain")
            .add_message("m3", role="user", text="Branch")
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)
        children = p.repository.get_children(str(parent.id))

        assert len(children) == 2
        child_ids = {c.id for c in children}
        assert child1.id in child_ids
        assert child2.id in child_ids

    def test_get_root(self, workspace_env: Path) -> None:
        """get_root should walk up to find the root conversation."""
        db_path = db_setup(workspace_env)

        root = (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .build()
        )

        middle = (
            ConversationBuilder(db_path, "middle")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Middle")
            .build()
        )

        leaf = (
            ConversationBuilder(db_path, "leaf")
            .provider("codex")
            .parent_conversation(str(middle.id))
            .branch_type("continuation")
            .add_message("m3", role="user", text="Leaf")
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)

        # From leaf, should find root
        found_root = p.repository.get_root(str(leaf.id))
        assert found_root is not None
        assert found_root.id == root.id

        # From root, should return itself
        found_root2 = p.repository.get_root(str(root.id))
        assert found_root2 is not None
        assert found_root2.id == root.id

    def test_get_session_tree(self, workspace_env: Path) -> None:
        """get_session_tree should return all conversations in the tree."""
        db_path = db_setup(workspace_env)

        root = (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .build()
        )

        child1 = (
            ConversationBuilder(db_path, "child1")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Child 1")
            .build()
        )

        child2 = (
            ConversationBuilder(db_path, "child2")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("sidechain")
            .add_message("m3", role="user", text="Child 2")
            .build()
        )

        grandchild = (
            ConversationBuilder(db_path, "grandchild")
            .provider("codex")
            .parent_conversation(str(child1.id))
            .branch_type("continuation")
            .add_message("m4", role="user", text="Grandchild")
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)

        # Get tree from any node
        tree = p.repository.get_session_tree(str(grandchild.id))

        assert len(tree) == 4
        tree_ids = {c.id for c in tree}
        assert root.id in tree_ids
        assert child1.id in tree_ids
        assert child2.id in tree_ids
        assert grandchild.id in tree_ids


class TestBranchingFilters:
    """Test branching filter methods."""

    def test_filter_is_continuation(self, workspace_env: Path) -> None:
        """is_continuation filter should find continuation conversations."""
        db_path = db_setup(workspace_env)

        root = (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .build()
        )

        cont = (
            ConversationBuilder(db_path, "continuation")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Continue")
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)

        continuations = p.filter().is_continuation().list()
        assert len(continuations) == 1
        assert continuations[0].id == cont.id

    def test_filter_is_sidechain(self, workspace_env: Path) -> None:
        """is_sidechain filter should find sidechain conversations."""
        db_path = db_setup(workspace_env)

        # Normal conversation
        (
            ConversationBuilder(db_path, "normal")
            .provider("claude-code")
            .add_message("m1", role="user", text="Normal")
            .build()
        )

        # Sidechain conversation
        sidechain = (
            ConversationBuilder(db_path, "sidechain")
            .provider("claude-code")
            .branch_type("sidechain")
            .add_message("m2", role="user", text="Side", provider_meta={"isSidechain": True})
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)

        sidechains = p.filter().is_sidechain().list()
        assert len(sidechains) == 1
        assert sidechains[0].id == sidechain.id

    def test_filter_is_root(self, workspace_env: Path) -> None:
        """is_root filter should find root conversations only."""
        db_path = db_setup(workspace_env)

        root = (
            ConversationBuilder(db_path, "root")
            .provider("codex")
            .add_message("m1", role="user", text="Root")
            .build()
        )

        (
            ConversationBuilder(db_path, "child")
            .provider("codex")
            .parent_conversation(str(root.id))
            .branch_type("continuation")
            .add_message("m2", role="user", text="Child")
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)

        roots = p.filter().is_root().list()
        assert len(roots) == 1
        assert roots[0].id == root.id

    def test_filter_has_branches(self, workspace_env: Path) -> None:
        """has_branches filter should find conversations with branching messages."""
        db_path = db_setup(workspace_env)

        # Conversation without branches
        (
            ConversationBuilder(db_path, "linear")
            .provider("chatgpt")
            .add_message("m1", role="user", text="Q", branch_index=0)
            .add_message("m2", role="assistant", text="A", branch_index=0)
            .build()
        )

        # Conversation with branches
        branching = (
            ConversationBuilder(db_path, "branching")
            .provider("chatgpt")
            .add_message("m3", role="user", text="Q", branch_index=0)
            .add_message("m4", role="assistant", text="A1", branch_index=0)
            .add_message("m5", role="assistant", text="A2", branch_index=1)
            .build()
        )

        from polylogue import Polylogue

        p = Polylogue(db_path=db_path)

        with_branches = p.filter().has_branches().list()
        assert len(with_branches) == 1
        assert with_branches[0].id == branching.id
