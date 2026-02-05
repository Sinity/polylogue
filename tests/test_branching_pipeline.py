"""Full-pipeline integration tests for branching features.

Unlike test_branching.py which uses ConversationBuilder (direct DB writes),
these tests exercise the REAL pipeline: raw JSON/JSONL → importer → prepare_ingest → database.

This ensures branching metadata extraction and resolution actually works end-to-end.

Note on filename conventions:
- Files named `claude_code_*.json` or `claude-code_*.json` → detected as claude-code (grouped)
- Files named `codex_*.json` → detected as codex (grouped)
- Without the provider name in filename, detection depends on content structure
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue import Polylogue
from polylogue.config import Source
from polylogue.pipeline.ingest import prepare_ingest
from polylogue.sources import iter_source_conversations
from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
from polylogue.storage.repository import ConversationRepository
from tests.helpers import db_setup


def _make_test_repository(db_path: Path) -> ConversationRepository:
    """Create a repository for testing with a specific db_path."""
    backend = SQLiteBackend(db_path=db_path)
    return ConversationRepository(backend=backend)


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

        # Verify importer extracted branching fields
        assert convo.provider_conversation_id == "child-session-uuid"
        assert convo.parent_conversation_provider_id == "parent-session-uuid"
        assert convo.branch_type == "continuation"

    def test_codex_continuation_persisted_to_database(self, workspace_env: Path, tmp_path: Path):
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

        with open_connection(db_path) as conn:
            parent_cid, _, _ = prepare_ingest(
                parent_convos[0],
                source_name="codex",
                archive_root=tmp_path,
                conn=conn,
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

        # Verify importer extracted parent reference
        child_parsed = child_convos[0]
        assert child_parsed.parent_conversation_provider_id == "parent-uuid"
        assert child_parsed.branch_type == "continuation"

        # Ingest through full pipeline
        with open_connection(db_path) as conn:
            child_cid, _, _ = prepare_ingest(
                child_parsed,
                source_name="codex",
                archive_root=tmp_path,
                conn=conn,
                repository=repository,
            )

        # Verify via domain model
        p = Polylogue(db_path=db_path)
        child_conv = p.repository.get(child_cid)

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

        # Verify importer detected sidechain
        assert convo.branch_type == "sidechain"

    def test_claude_code_sidechain_persisted_to_database(self, workspace_env: Path, tmp_path: Path):
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
        with open_connection(db_path) as conn:
            cid, _, _ = prepare_ingest(
                parsed,
                source_name="claude-code",
                archive_root=tmp_path,
                conn=conn,
                repository=repository,
            )

        # Verify via domain model
        p = Polylogue(db_path=db_path)
        conv = p.repository.get(cid)

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

    def test_chatgpt_branching_persisted_to_database(self, workspace_env: Path, tmp_path: Path):
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
        with open_connection(db_path) as conn:
            cid, _, _ = prepare_ingest(
                parsed,
                source_name="chatgpt",
                archive_root=tmp_path,
                conn=conn,
                repository=repository,
            )

        # Verify via domain model
        p = Polylogue(db_path=db_path)
        conv = p.repository.get(cid)

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


class TestPrepareIngestParentResolution:
    """Test that prepare_ingest correctly resolves provider IDs to internal IDs."""

    def test_parent_conversation_id_resolved_to_internal_format(self, workspace_env: Path, tmp_path: Path):
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

        with open_connection(db_path) as conn:
            parent_cid, _, _ = prepare_ingest(
                parent_parsed,
                source_name="codex",
                archive_root=tmp_path,
                conn=conn,
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

        with open_connection(db_path) as conn:
            child_cid, _, _ = prepare_ingest(
                child_parsed,
                source_name="codex",
                archive_root=tmp_path,
                conn=conn,
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

    def test_parent_message_id_resolved_within_conversation(self, workspace_env: Path, tmp_path: Path):
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

        with open_connection(db_path) as conn:
            cid, _, _ = prepare_ingest(
                parsed,
                source_name="chatgpt",
                archive_root=tmp_path,
                conn=conn,
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
