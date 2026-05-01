"""Focused branching contracts across domain views, repository traversal, and pipeline persistence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.config import Source
from polylogue.lib.models import Conversation
from polylogue.lib.roles import Role
from polylogue.pipeline.prepare import prepare_records
from polylogue.pipeline.prepare_models import PersistedConversationResult
from polylogue.sources import iter_source_conversations
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository
from polylogue.types import Provider
from tests.infra.storage_records import ConversationBuilder, db_setup

WorkspaceEnv = dict[str, Path]
PayloadFactory = Callable[[], object]
ParsedAssertion = Callable[[ParsedConversation], bool]


def _make_repository(db_path: Path) -> ConversationRepository:
    return ConversationRepository(backend=SQLiteBackend(db_path=db_path))


def _prepare_fields(result: PersistedConversationResult) -> tuple[str, dict[str, int], bool]:
    return result.conversation_id, result.counts, result.content_changed


def _write_payload(tmp_path: Path, filename: str, payload: object) -> Path:
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parse_single(source_name: str, source_path: Path) -> ParsedConversation:
    conversations = list(iter_source_conversations(Source(name=source_name, path=source_path)))
    assert len(conversations) == 1
    return conversations[0]


def _require_conversation(value: Conversation | None) -> Conversation:
    if value is None:
        raise AssertionError("expected seeded conversation")
    return value


def _codex_continuation_payload(*, child_id: str, parent_id: str) -> list[dict[str, object]]:
    return [
        {"type": "session_meta", "payload": {"id": child_id, "timestamp": "2025-01-02T10:00:00Z"}},
        {"type": "session_meta", "payload": {"id": parent_id, "timestamp": "2025-01-01T10:00:00Z"}},
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


def _claude_sidechain_payload(*, session_id: str = "sidechain-session") -> list[dict[str, object]]:
    return [
        {"type": "user", "uuid": "u1", "sessionId": session_id, "message": {"content": "User message"}},
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": session_id,
            "isSidechain": True,
            "message": {"content": "Assistant in sidechain"},
        },
    ]


def _chatgpt_branch_payload(*, title: str = "Branched Conversation") -> dict[str, object]:
    return {
        "title": title,
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
                    "content": {"parts": ["Answer 2 (branch)"]},
                    "create_time": 1700000002,
                },
            },
        },
    }


async def _seed_branch_archive(db_path: Path) -> dict[str, str]:
    root = _require_conversation(
        await ConversationBuilder(db_path, "root")
        .provider("codex")
        .title("Root")
        .add_message("root-user", role="user", text="Root")
        .build(),
    )
    continuation = _require_conversation(
        await ConversationBuilder(db_path, "continuation")
        .provider("codex")
        .title("Continuation")
        .parent_conversation(str(root.id))
        .branch_type("continuation")
        .add_message("cont-user", role="user", text="Continue")
        .build(),
    )
    sidechain = _require_conversation(
        await ConversationBuilder(db_path, "sidechain")
        .provider("claude-code")
        .title("Sidechain")
        .parent_conversation(str(root.id))
        .branch_type("sidechain")
        .add_message("side-user", role="user", text="User")
        .add_message("side-assistant", role="assistant", text="Side answer", provider_meta={"isSidechain": True})
        .build(),
    )
    grandchild = _require_conversation(
        await ConversationBuilder(db_path, "grandchild")
        .provider("codex")
        .title("Grandchild")
        .parent_conversation(str(continuation.id))
        .branch_type("continuation")
        .add_message("grand-user", role="user", text="Grandchild")
        .build(),
    )
    branched = _require_conversation(
        await ConversationBuilder(db_path, "branching")
        .provider("chatgpt")
        .title("Branched")
        .add_message("q1", role="user", text="Question", provider_message_id="q1", branch_index=0)
        .add_message(
            "a1",
            role="assistant",
            text="Answer 1",
            provider_message_id="a1",
            parent_message_id="q1",
            parent_message_provider_id="q1",
            branch_index=0,
        )
        .add_message(
            "a2",
            role="assistant",
            text="Answer 2",
            provider_message_id="a2",
            parent_message_id="q1",
            parent_message_provider_id="q1",
            branch_index=1,
        )
        .build(),
    )
    return {
        "root": str(root.id),
        "continuation": str(continuation.id),
        "sidechain": str(sidechain.id),
        "grandchild": str(grandchild.id),
        "branching": str(branched.id),
    }


class TestBranchDomainViews:
    @pytest.mark.asyncio
    async def test_branch_flags_and_message_views_contract(self, workspace_env: WorkspaceEnv) -> None:
        db_path = db_setup(workspace_env)
        ids = await _seed_branch_archive(db_path)

        async with _make_repository(db_path) as repo:
            continuation = await repo.get(ids["continuation"])
            sidechain = await repo.get(ids["sidechain"])
            branching = await repo.get(ids["branching"])

        assert continuation is not None and continuation.is_continuation is True
        assert continuation.is_sidechain is False
        assert continuation.is_root is False

        assert sidechain is not None and sidechain.is_sidechain is True
        assert sidechain.is_continuation is False
        assert sidechain.is_root is False

        assert branching is not None
        assert [message.id for message in branching.mainline_messages()] == ["q1", "a1"]
        branches = list(branching.iter_branches())
        assert len(branches) == 1
        assert branches[0][0] == "q1"
        assert [message.branch_index for message in branches[0][1]] == [0, 1]
        assert [message.is_branch for message in branches[0][1]] == [False, True]


class TestBranchRepositoryTraversal:
    @pytest.mark.asyncio
    async def test_tree_and_filter_contract(self, workspace_env: WorkspaceEnv) -> None:
        db_path = db_setup(workspace_env)
        ids = await _seed_branch_archive(db_path)

        async with _make_repository(db_path) as repo:
            parent = await repo.get_parent(ids["continuation"])
            children = await repo.get_children(ids["root"])
            root = await repo.get_root(ids["grandchild"])
            tree = await repo.get_session_tree(ids["grandchild"])
            continuations = await repo.filter().is_continuation().list()
            sidechains = await repo.filter().is_sidechain().list()
            roots = await repo.filter().is_root().list()
            with_branches = await repo.filter().has_branches().list()

        assert parent is not None and str(parent.id) == ids["root"]
        assert {str(conv.id) for conv in children} == {ids["continuation"], ids["sidechain"]}
        assert root is not None and str(root.id) == ids["root"]
        assert {str(conv.id) for conv in tree} == {
            ids["root"],
            ids["continuation"],
            ids["sidechain"],
            ids["grandchild"],
        }
        assert {str(conv.id) for conv in continuations} == {ids["continuation"], ids["grandchild"]}
        assert {str(conv.id) for conv in sidechains} == {ids["sidechain"]}
        assert ids["root"] in {str(conv.id) for conv in roots}
        assert [str(conv.id) for conv in with_branches] == [ids["branching"]]


PARSER_CASES: tuple[tuple[str, str, PayloadFactory, ParsedAssertion], ...] = (
    (
        "codex",
        "codex_child.json",
        lambda: _codex_continuation_payload(child_id="child-session-uuid", parent_id="parent-session-uuid"),
        lambda convo: (
            convo.provider_conversation_id == "child-session-uuid"
            and convo.parent_conversation_provider_id == "parent-session-uuid"
            and convo.branch_type == "continuation"
        ),
    ),
    (
        "claude-code",
        "claude_code_sidechain.json",
        _claude_sidechain_payload,
        lambda convo: convo.branch_type == "sidechain",
    ),
    (
        "chatgpt",
        "chatgpt_branched.json",
        lambda: [_chatgpt_branch_payload()],
        lambda convo: (
            {message.provider_message_id: message.branch_index for message in convo.messages}
            == {"q1": 0, "a1": 0, "a2": 1}
        ),
    ),
)


class TestBranchParserContracts:
    @pytest.mark.parametrize(
        "source_name,filename,payload_factory,assertion", PARSER_CASES, ids=[case[0] for case in PARSER_CASES]
    )
    def test_source_parser_extracts_branch_metadata(
        self,
        tmp_path: Path,
        source_name: str,
        filename: str,
        payload_factory: PayloadFactory,
        assertion: ParsedAssertion,
    ) -> None:
        source_path = _write_payload(tmp_path, filename, payload_factory())
        parsed = _parse_single(source_name, source_path)
        assert assertion(parsed)


class TestBranchPipelinePersistence:
    @pytest.mark.asyncio
    async def test_codex_continuation_pipeline_contract(self, workspace_env: WorkspaceEnv, tmp_path: Path) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_path = _write_payload(
                tmp_path,
                "codex_parent.json",
                [
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
                ],
            )
            parent_cid, _, _ = _prepare_fields(
                await prepare_records(
                    _parse_single("codex", parent_path),
                    source_name="codex",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )

            child_path = _write_payload(
                tmp_path,
                "codex_child.json",
                _codex_continuation_payload(child_id="child-uuid", parent_id="parent-uuid"),
            )
            child_cid, _, _ = _prepare_fields(
                await prepare_records(
                    _parse_single("codex", child_path),
                    source_name="codex",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )
            child = await repo.get(child_cid)

        assert child is not None
        assert str(child.parent_id) == parent_cid
        assert child.branch_type == "continuation"
        assert child.is_continuation is True

    @pytest.mark.asyncio
    async def test_claude_sidechain_pipeline_contract(self, workspace_env: WorkspaceEnv, tmp_path: Path) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            source_path = _write_payload(tmp_path, "claude_code_sidechain.json", _claude_sidechain_payload())
            conversation_id, _, _ = _prepare_fields(
                await prepare_records(
                    _parse_single("claude-code", source_path),
                    source_name="claude-code",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )
            conversation = await repo.get(conversation_id)

        assert conversation is not None
        assert conversation.branch_type == "sidechain"
        assert conversation.is_sidechain is True

    @pytest.mark.asyncio
    async def test_chatgpt_branch_pipeline_and_parent_resolution_contract(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parsed = _parse_single(
                "chatgpt", _write_payload(tmp_path, "chatgpt_branched.json", [_chatgpt_branch_payload()])
            )
            conversation_id, _, _ = _prepare_fields(
                await prepare_records(
                    parsed,
                    source_name="chatgpt",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )
            conversation = await repo.get(conversation_id)

        assert conversation is not None
        answers = {message.text: message for message in conversation.messages}
        assert answers["Answer 1"].branch_index == 0
        assert answers["Answer 2 (branch)"].branch_index == 1
        assert answers["Answer 1"].parent_id == answers["Answer 2 (branch)"].parent_id
        assert [message.text for message in conversation.mainline_messages()] == ["Question", "Answer 1"]

    @pytest.mark.asyncio
    async def test_prepare_records_resolves_parent_conversation_and_message_ids(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_id, _, _ = _prepare_fields(
                await prepare_records(
                    ParsedConversation(
                        provider_name=Provider.CODEX,
                        provider_conversation_id="parent-id",
                        title="Parent Session",
                        messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="Parent message")],
                    ),
                    source_name="codex",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )
            child_id, _, _ = _prepare_fields(
                await prepare_records(
                    ParsedConversation(
                        provider_name=Provider.CODEX,
                        provider_conversation_id="child-id",
                        title="Child Session",
                        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="Hello")],
                        parent_conversation_provider_id="parent-id",
                        branch_type=BranchType.CONTINUATION,
                    ),
                    source_name="codex",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )
            branch_id, _, _ = _prepare_fields(
                await prepare_records(
                    ParsedConversation(
                        provider_name=Provider.CHATGPT,
                        provider_conversation_id="conv-1",
                        title="Branched Session",
                        messages=[
                            ParsedMessage(provider_message_id="q1", role=Role.USER, text="Question"),
                            ParsedMessage(
                                provider_message_id="a1",
                                role=Role.ASSISTANT,
                                text="Answer 1",
                                parent_message_provider_id="q1",
                                branch_index=0,
                            ),
                            ParsedMessage(
                                provider_message_id="a2",
                                role=Role.ASSISTANT,
                                text="Answer 2",
                                parent_message_provider_id="q1",
                                branch_index=1,
                            ),
                        ],
                    ),
                    source_name="chatgpt",
                    archive_root=tmp_path,
                    backend=repo.backend,
                    repository=repo,
                )
            )

        with open_connection(db_path) as conn:
            child_row = conn.execute(
                "SELECT parent_conversation_id, branch_type FROM conversations WHERE conversation_id = ?",
                (child_id,),
            ).fetchone()
            message_rows = conn.execute(
                "SELECT provider_message_id, parent_message_id, branch_index FROM messages WHERE conversation_id = ? ORDER BY provider_message_id",
                (branch_id,),
            ).fetchall()

        assert child_row["parent_conversation_id"] == parent_id
        assert child_row["branch_type"] == "continuation"
        rows_by_provider_id = {row["provider_message_id"]: row for row in message_rows}
        assert rows_by_provider_id["q1"]["parent_message_id"] is None
        assert rows_by_provider_id["a1"]["parent_message_id"] == rows_by_provider_id["a2"]["parent_message_id"]
        assert rows_by_provider_id["a1"]["branch_index"] == 0
        assert rows_by_provider_id["a2"]["branch_index"] == 1
