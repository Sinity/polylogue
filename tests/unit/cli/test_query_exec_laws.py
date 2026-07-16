"""Archive query executor contracts and CLI search behavior.

After the archive-route cleanup (#1743) the root query dispatch routes
unconditionally to ``execute_archive_query``. These tests exercise that
archive executor directly (``async_execute_query`` over a seeded/mocked archive
``index.db``) and the user-facing CLI search surface via ``CliRunner``. The
legacy ``_execute_query_plan`` / ``_handle_*`` route-handler algebra and its
mock-based contract tests were retired with the executor itself.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Session
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.stats import ArchiveStats
from polylogue.cli.query import async_execute_query, project_query_results
from polylogue.cli.query_contracts import (
    QueryAction,
    QueryDeliveryTarget,
    QueryExecutionPlan,
    QueryMutationSpec,
    QueryOutputSpec,
)
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.services import build_runtime_services
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope
from polylogue.surfaces.payloads import decode_search_cursor
from tests.infra.builders import make_conv, make_msg

pytestmark = pytest.mark.query_routing
SearchWorkspace = dict[str, Path]


@pytest.fixture
def search_workspace(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """CLI workspace seeded with searchable sessions in the archive store.

    The query path the root CLI reads resolves to
    ``archive_root/index.db``, so the builders target that store directly.
    """
    from datetime import datetime, timedelta, timezone

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
    from tests.infra.storage_records import SessionBuilder

    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    index_db = cli_workspace["archive_root"] / "index.db"
    now = datetime.now(timezone.utc)

    (
        SessionBuilder(index_db, "conv1")
        .provider("chatgpt")
        .title("Python Error Handling")
        .created_at((now - timedelta(days=1)).isoformat())
        .updated_at((now - timedelta(days=1)).isoformat())
        .add_message("m1", role="user", text="How to handle exceptions in Python?")
        .add_message("m2", role="assistant", text="Use try-except blocks for Python exception handling.")
        .save()
    )
    (
        SessionBuilder(index_db, "conv2")
        .provider("claude-ai")
        .title("JavaScript Async Patterns")
        .created_at((now - timedelta(days=10)).isoformat())
        .updated_at((now - timedelta(days=10)).isoformat())
        .add_message("m3", role="user", text="Explain async/await in JavaScript")
        .add_message("m4", role="assistant", text="Async/await is JavaScript syntax for promises.")
        .save()
    )
    (
        SessionBuilder(index_db, "conv3")
        .provider("claude-code")
        .title("Rust Ownership")
        .created_at((now - timedelta(hours=6)).isoformat())
        .updated_at((now - timedelta(hours=6)).isoformat())
        .add_message("m5", role="user", text="What is ownership in Rust?")
        .add_message("m6", role="assistant", text="Rust ownership ensures memory safety without garbage collection.")
        .save()
    )
    (
        SessionBuilder(index_db, "run-hit")
        .provider("codex")
        .git_repository_url("polylogue")
        .git_branch("feature/query-runs")
        .working_directories(["/realm/project/polylogue"])
        .title("Run query rendering")
        .created_at((now - timedelta(minutes=30)).isoformat())
        .updated_at((now - timedelta(minutes=30)).isoformat())
        .add_message("m7", role="user", text="Wire run query unit rendering.")
        .add_message(
            "m8",
            role="assistant",
            text="Delegating the run query rendering check.",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "tool-run",
                    "name": "Task",
                    "tool_input": {
                        "subagent_type": "Explore",
                        "taskId": "task-run",
                        "child_session_id": "codex-session:child-run",
                        "prompt": "Map run query unit CLI rendering.",
                    },
                },
                {
                    "type": "tool_result",
                    "tool_id": "tool-run",
                    "text": "Subagent done: run query unit rendered.\n2 passed in 0.12s",
                },
            ],
        )
        .save()
    )
    (
        SessionBuilder(index_db, "run-hit-subagent")
        .provider("codex")
        .parent_session("ext-run-hit")
        .branch_type("subagent")
        .git_repository_url("polylogue")
        .git_branch("feature/query-runs")
        .working_directories(["/realm/project/polylogue"])
        .title("Run query rendering subagent")
        .created_at((now - timedelta(minutes=29)).isoformat())
        .updated_at((now - timedelta(minutes=29)).isoformat())
        .add_message("m9", role="assistant", text="Run query unit rendered.")
        .save()
    )

    with sqlite3.connect(cli_workspace["archive_root"] / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        initialize_archive_tier(conn, ArchiveTier.USER)
        upsert_assertion(
            conn,
            assertion_id="assertion-cli-decision",
            target_ref="session:chatgpt-export:ext-conv1",
            kind=AssertionKind.DECISION,
            key="query-unit/parity",
            body_text="Python terminal assertion query rows must render in the CLI.",
            author_ref="user:operator",
            author_kind="user",
            evidence_refs=("message:chatgpt-export:ext-conv1:m2",),
            status="active",
            visibility="private",
            now_ms=1_700_000_010_000,
        )

    # Materialize derived insight read-models (session_profiles, work events)
    # so the SQL-backed query units have rows to return, mirroring a real
    # post-ingest archive. Run/observed-event/context-snapshot query units are
    # source-derived (polylogue-dab) and need no separate materialization.
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync

    with sqlite3.connect(index_db) as insight_conn:
        insight_conn.row_factory = sqlite3.Row
        rebuild_session_insights_sync(insight_conn)
        insight_conn.commit()

    return cli_workspace


def _make_env(*, repo: MagicMock | None = None, config: MagicMock | None = None) -> AppEnv:
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.confirm = MagicMock(return_value=True)
    if repo is not None:
        if not isinstance(repo.get_render_projection, AsyncMock):
            repo.get_render_projection = AsyncMock(return_value=None)
        if not isinstance(repo.get_session_stats, AsyncMock):
            repo.get_session_stats = AsyncMock(return_value={})
        if not isinstance(repo.get_message_counts_batch, AsyncMock):
            repo.get_message_counts_batch = AsyncMock(return_value={})
        if not isinstance(repo.aggregate_message_stats, AsyncMock):
            repo.aggregate_message_stats = AsyncMock(return_value={})
        if not isinstance(repo.get_sessions_batch, AsyncMock):
            repo.get_sessions_batch = AsyncMock(return_value=[])
        if not isinstance(repo.get_messages_batch, AsyncMock):
            repo.get_messages_batch = AsyncMock(return_value={})
        if not isinstance(repo.get_attachments_batch, AsyncMock):
            repo.get_attachments_batch = AsyncMock(return_value={})
        if not isinstance(repo.list_summaries_by_query, AsyncMock):
            repo.list_summaries_by_query = AsyncMock(return_value=[])
    return AppEnv(ui=ui, services=build_runtime_services(config=config, repository=repo))


def _as_mock(value: object) -> MagicMock:
    if not isinstance(value, MagicMock):
        raise TypeError(f"expected MagicMock, got {type(value).__name__}")
    return value


def _delivery_targets(*destinations: str) -> tuple[QueryDeliveryTarget, ...]:
    return tuple(QueryDeliveryTarget.parse(destination) for destination in destinations)


def _output_spec(
    output_format: str = "markdown",
    *,
    destinations: tuple[str, ...] = ("stdout",),
    fields: str | None = None,
    list_mode: bool = False,
    print_url: bool = False,
) -> QueryOutputSpec:
    return QueryOutputSpec(
        output_format=output_format,
        destinations=_delivery_targets(*destinations),
        fields=fields,
        list_mode=list_mode,
        print_url=print_url,
    )


def _mutation_spec(
    *,
    set_meta: tuple[tuple[str, str], ...] = (),
    add_tags: tuple[str, ...] = (),
    delete_matched: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> QueryMutationSpec:
    return QueryMutationSpec(
        set_meta=set_meta,
        add_tags=add_tags,
        delete_matched=delete_matched,
        dry_run=dry_run,
        force=force,
    )


def _sample_session() -> Session:
    return make_conv(
        id="conv-transform",
        provider=Provider.CLAUDE_AI,
        title="Transform Contract",
        messages=[
            make_msg(
                id="m-user",
                role=Role.USER,
                text="hello",
                material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
            ),
            make_msg(
                id="m-thinking",
                role=Role.ASSISTANT,
                text="chain",
                material_origin=MaterialOrigin.RUNTIME_CONTEXT,
                blocks=[{"type": "thinking", "text": "chain"}],
            ),
            make_msg(id="m-tool", role=Role.TOOL, text="tool output"),
            make_msg(
                id="m-assistant",
                role=Role.ASSISTANT,
                text="answer",
                material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# project_query_results
# ---------------------------------------------------------------------------


def test_project_query_results_contract() -> None:
    plan = QueryExecutionPlan(
        selection=SessionQuerySpec(),
        action=QueryAction.SHOW,
        output=_output_spec(),
        mutation=_mutation_spec(),
    )
    session = _sample_session()

    projected = project_query_results([session], plan)

    assert projected == [session]
    assert [message.id for message in session.messages] == [
        "m-user",
        "m-thinking",
        "m-tool",
        "m-assistant",
    ]


class TestBuildQueryExecutionPlan:
    """``build_query_execution_plan`` is still wired through ``cli/select.py``."""

    def test_delete_without_filters_raises(self) -> None:
        from polylogue.cli.query_contracts import QueryPlanError, build_query_execution_plan

        with pytest.raises(QueryPlanError, match="delete requires at least one filter"):
            build_query_execution_plan({"delete_matched": True, "query": ()})

    @pytest.mark.parametrize(
        ("params", "expected_action"),
        [
            ({"count_only": True, "query": ()}, QueryAction.COUNT),
            ({"stream": True, "query": ("abc",)}, QueryAction.STREAM),
            ({"stats_only": True, "query": ()}, QueryAction.STATS),
            ({"stats_by": "origin", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "action", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "tool", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "repo", "query": ()}, QueryAction.STATS_BY),
            ({"stats_by": "work-kind", "query": ()}, QueryAction.STATS_BY),
            ({"add_tag": ["x"], "query": ()}, QueryAction.MODIFY),
            ({"delete_matched": True, "tag": ("review",), "query": ()}, QueryAction.DELETE),
            ({"open_result": True, "query": ("abc",)}, QueryAction.OPEN),
            ({"query": ("abc",)}, QueryAction.SHOW),
        ],
    )
    def test_action_selection(self, params: dict[str, object], expected_action: QueryAction) -> None:
        from polylogue.cli.query_contracts import build_query_execution_plan

        plan = build_query_execution_plan(params)
        assert plan.action == expected_action

    def test_stream_format_converts_json_to_json_lines(self) -> None:
        from polylogue.cli.query_contracts import build_query_execution_plan

        plan = build_query_execution_plan({"stream": True, "output_format": "json", "query": ("abc",)})
        assert plan.output.stream_format() == "json-lines"

    def test_summary_list_preference_requires_plain_listing_shape(self) -> None:
        from polylogue.cli.query_contracts import build_query_execution_plan

        plan = build_query_execution_plan({"list_mode": True, "query": ("abc",)})
        assert plan.prefers_summary_list() is True

    def test_mutation_fields_are_normalized(self) -> None:
        from polylogue.cli.query_contracts import build_query_execution_plan

        plan = build_query_execution_plan(
            {
                "set_meta": [("priority", 3)],
                "add_tag": ["todo", "review"],
                "force": True,
                "dry_run": True,
                "provider": "claude-ai",
                "query": (),
            }
        )
        assert plan.mutation.set_meta == (("priority", "3"),)
        assert plan.mutation.add_tags == ("todo", "review")
        assert plan.mutation.force is True
        assert plan.mutation.dry_run is True


# ---------------------------------------------------------------------------
# Archive query executor contracts
# ---------------------------------------------------------------------------


def test_async_execute_query_archive_lists_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(
            self,
            *,
            limit: int,
            offset: int,
            sort: str | None,
            reverse: bool,
            origin: str | None,
            origins: tuple[str, ...],
            excluded_origins: tuple[str, ...],
            tags: tuple[str, ...],
            excluded_tags: tuple[str, ...],
            repo_names: tuple[str, ...],
            project_refs: tuple[str, ...],
            has_types: tuple[str, ...],
            has_tool_use: bool,
            has_thinking: bool,
            has_paste: bool,
            tool_terms: tuple[str, ...],
            excluded_tool_terms: tuple[str, ...],
            action_terms: tuple[str, ...],
            excluded_action_terms: tuple[str, ...],
            action_sequence: tuple[str, ...],
            action_text_terms: tuple[str, ...],
            referenced_paths: tuple[str, ...],
            cwd_prefix: str | None,
            typed_only: bool,
            message_type: str | None,
            title: str | None,
            min_messages: int | None,
            max_messages: int | None,
            min_words: int | None,
            max_words: int | None,
            since_ms: int | None,
            until_ms: int | None,
            since_session_id: str | None,
            sample: bool,
        ) -> list[ArchiveSessionSummary]:
            assert limit == 3
            assert offset == 0
            assert sample is False
            assert sort is None
            assert reverse is False
            assert origin is None
            assert origins == ()
            assert excluded_origins == ()
            assert tags == ()
            assert excluded_tags == ()
            assert repo_names == ()
            assert has_types == ()
            assert has_tool_use is False
            assert has_thinking is False
            assert has_paste is False
            assert tool_terms == ()
            assert excluded_tool_terms == ()
            assert action_terms == ()
            assert excluded_action_terms == ()
            assert action_sequence == ()
            assert action_text_terms == ()
            assert referenced_paths == ()
            assert cwd_prefix is None
            assert typed_only is False
            assert message_type is None
            assert title is None
            assert min_messages is None
            assert max_messages is None
            assert min_words is None
            assert max_words is None
            assert since_ms is None
            assert until_ms is None
            assert since_session_id is None
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Copied",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=("archive",),
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(async_execute_query(env, {"archive": True, "limit": 2, "output_format": "json"}))

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["items"][0]["id"] == "codex-session:native-1"
    assert payload["items"][0]["origin"] == "codex-session"


def test_async_execute_query_uses_archive_when_index_db_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)
    calls: list[object] = []

    def fake_execute_archive_query(env_arg: AppEnv, request: object) -> None:
        assert env_arg is env
        calls.append(request)

    monkeypatch.setattr("polylogue.cli.archive_query.execute_archive_query", fake_execute_archive_query)

    asyncio.run(async_execute_query(env, {"limit": 2}))

    assert len(calls) == 1


def test_async_execute_query_archive_projects_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["limit"] == 2
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Projected",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=("archive",),
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {"archive": True, "limit": 1, "fields": "id,title", "output_format": "json"},
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["items"] == [{"id": "codex-session:native-1", "title": "Projected"}]


def test_async_execute_query_archive_routes_pure_structured_terms_to_list(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_summaries(self, *_args: object, **_kwargs: object) -> list[ArchiveSessionSearchHit]:
            raise AssertionError("pure structured positional queries must not use FTS search")

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["limit"] == 2
            assert kwargs["offset"] == 0
            assert kwargs["origins"] == ("codex-session",)
            assert kwargs["repo_names"] == ("polylogue",)
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Structured",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=(),
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("repo:polylogue", "origin:codex-session"),
                "limit": 1,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["items"][0]["id"] == "codex-session:native-1"


def test_async_execute_query_archive_exact_id_clause_reads_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            assert token == "chatgpt-export:6a49af33-d698-83ed-a11f-8750f19027e1"
            return token

        def search_summaries(self, *_args: object, **_kwargs: object) -> list[ArchiveSessionSearchHit]:
            raise AssertionError("exact id: refs must not fall through to FTS")

        def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
            assert session_id == "chatgpt-export:6a49af33-d698-83ed-a11f-8750f19027e1"
            return ArchiveSessionEnvelope(
                session_id=session_id,
                native_id="6a49af33-d698-83ed-a11f-8750f19027e1",
                origin="chatgpt-export",
                title="Captured ChatGPT session",
                active_leaf_message_id=None,
                messages=(),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("id:chatgpt-export:6a49af33-d698-83ed-a11f-8750f19027e1",),
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "session"
    assert payload["session_id"] == "chatgpt-export:6a49af33-d698-83ed-a11f-8750f19027e1"
    assert payload["messages"] == []


def test_async_execute_query_archive_bare_native_ref_resolves_before_fts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)
    native_id = "6a49af33-d698-83ed-a11f-8750f19027e1"

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            assert token == native_id
            return f"chatgpt-export:{native_id}"

        def search_summaries(self, *_args: object, **_kwargs: object) -> list[ArchiveSessionSearchHit]:
            raise AssertionError("bare native refs must resolve before lexical search")

        def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
            assert session_id == f"chatgpt-export:{native_id}"
            return ArchiveSessionEnvelope(
                session_id=session_id,
                native_id=native_id,
                origin="chatgpt-export",
                title="Captured ChatGPT session",
                active_leaf_message_id=None,
                messages=(),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": (native_id,),
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "session"
    assert payload["session_id"] == f"chatgpt-export:{native_id}"
    assert payload["native_id"] == native_id


def test_async_execute_query_archive_unresolved_bare_ref_falls_back_to_fts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            assert token == "missing-native-12345"
            raise KeyError(token)

        def search_summaries(self, query: str, **kwargs: object) -> list[ArchiveSessionSearchHit]:
            assert query == "missing-native-12345"
            assert kwargs["session_id"] is None
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "query": ("missing-native-12345",),
                    "output_format": "json",
                },
            )
        )
    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "search"
    assert payload["items"] == []


def test_async_execute_query_archive_uses_daemon_for_supported_session_pages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    config = MagicMock()
    config.archive_root = archive_root
    config.db_path = archive_root / "index.db"
    config.daemon_url = "http://127.0.0.1:9876"
    config.api_auth_token = None
    env = _make_env(repo=MagicMock(), config=config)
    seen: dict[str, object] = {}

    def fake_fetch(config_arg: object, query_params: dict[str, object]) -> dict[str, object]:
        assert config_arg is config
        seen.update(query_params)
        return {
            "items": [
                {
                    "id": "codex-session:native-1",
                    "origin": "codex-session",
                    "title": "Daemon page",
                    "message_count": 3,
                }
            ],
            "total": 1,
            "limit": 2,
            "offset": 0,
        }

    monkeypatch.setattr("polylogue.cli.archive_query._fetch_daemon_sessions_payload", fake_fetch)
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        MagicMock(side_effect=AssertionError("supported daemon page must not open SQLite")),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("repo:polylogue", "origin:codex-session"),
                "limit": 2,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert seen == {"limit": 2, "offset": 0, "query": "repo:polylogue origin:codex-session"}
    assert payload["source"] == "daemon"
    assert payload["mode"] == "list"
    assert payload["items"][0]["id"] == "codex-session:native-1"


def test_async_execute_query_archive_preserves_daemon_degraded_search_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    config = MagicMock()
    config.archive_root = archive_root
    config.db_path = archive_root / "index.db"
    config.daemon_url = "http://127.0.0.1:9876"
    config.api_auth_token = None
    env = _make_env(repo=MagicMock(), config=config)

    def fake_fetch(config_arg: object, query_params: dict[str, object]) -> dict[str, object]:
        assert config_arg is config
        assert query_params["query"] == "search needle"
        return {
            "query": "search needle",
            "retrieval_lane": "dialogue",
            "hits": [],
            "total": None,
            "route_state": {
                "state": "degraded",
                "route": "/api/sessions",
                "reason": "Search index unavailable: message FTS table is missing or degraded.",
                "component": "message_fts",
                "stale_available": False,
            },
            "diagnostics": {
                "message": "Search index unavailable: message FTS table is missing or degraded.",
                "reasons": [{"code": "search_index_degraded", "severity": "warning"}],
            },
        }

    monkeypatch.setattr("polylogue.cli.archive_query._fetch_daemon_sessions_payload", fake_fetch)
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        MagicMock(side_effect=AssertionError("degraded daemon search must not open SQLite as no-results fallback")),
    )

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "query": ("search", "needle"),
                    "limit": 2,
                    "output_format": "json",
                },
            )
        )

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "daemon"
    assert payload["mode"] == "search"
    assert payload["total"] is None
    assert payload["items"] == []
    assert payload["route_state"]["state"] == "degraded"
    assert payload["route_state"]["component"] == "message_fts"
    assert payload["diagnostics"]["reasons"][0]["code"] == "search_index_degraded"


def test_async_execute_query_archive_preserves_daemon_degraded_search_plain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    config = MagicMock()
    config.archive_root = archive_root
    config.db_path = archive_root / "index.db"
    config.daemon_url = "http://127.0.0.1:9876"
    config.api_auth_token = None
    env = _make_env(repo=MagicMock(), config=config)

    monkeypatch.setattr(
        "polylogue.cli.archive_query._fetch_daemon_sessions_payload",
        lambda *_args: {
            "query": "search needle",
            "retrieval_lane": "dialogue",
            "hits": [],
            "total": None,
            "route_state": {
                "state": "degraded",
                "route": "/api/sessions",
                "reason": "Search index unavailable: message FTS table is missing or degraded.",
                "component": "message_fts",
                "stale_available": False,
            },
        },
    )
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        MagicMock(side_effect=AssertionError("degraded daemon search must not open SQLite as no-results fallback")),
    )

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "query": ("search", "needle"),
                    "limit": 2,
                    "output_format": "markdown",
                },
            )
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Search index unavailable" in captured.err
    assert "No sessions matched" not in captured.out


def test_async_execute_query_archive_falls_back_when_daemon_unavailable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    config.db_path = archive_root / "index.db"
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["repo_names"] == ("polylogue",)
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Fallback",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=(),
                )
            ]

    monkeypatch.setattr("polylogue.cli.archive_query._fetch_daemon_sessions_payload", lambda *_args: None)
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("repo:polylogue",),
                "limit": 1,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert "source" not in payload
    assert payload["items"][0]["title"] == "Fallback"


def test_async_execute_query_archive_keeps_stats_local(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    config.db_path = archive_root / "index.db"
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_session_ids(self, query: str, **kwargs: object) -> tuple[str, ...]:
            assert query == "needle"
            return ("codex-session:native-1",)

        def stats_by(self, group_by: str, **kwargs: object) -> dict[str, int]:
            assert group_by == "origin"
            assert kwargs["session_ids"] == ("codex-session:native-1",)
            return {"codex-session": 1}

    monkeypatch.setattr(
        "polylogue.cli.archive_query._fetch_daemon_sessions_payload",
        MagicMock(side_effect=AssertionError("stats paths must remain local")),
    )
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("needle",),
                "stats_by": "origin",
                "limit": 10,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "stats_by"
    assert payload["items"] == [{"count": 1, "group": "codex-session"}]


def test_daemon_session_fetch_has_wall_clock_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.cli import archive_query

    def slow_fetch(*_args: object, **_kwargs: object) -> dict[str, object]:
        time.sleep(0.2)
        return {"items": [], "total": 0}

    monkeypatch.setattr(archive_query, "_DAEMON_FAST_PATH_TIMEOUT_S", 0.01)
    monkeypatch.setattr(archive_query, "_fetch_daemon_sessions_payload_once", slow_fetch)

    started = time.monotonic()
    payload = archive_query._fetch_daemon_sessions_payload_with_deadline(
        "http://127.0.0.1:9876",
        None,
        {"limit": 1},
    )
    elapsed = time.monotonic() - started

    assert payload is None
    assert elapsed < 0.1


def test_async_execute_query_archive_sorts_lists(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["sort"] == "messages"
            assert kwargs["reverse"] is True
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {"archive": True, "sort": "messages", "reverse": True, "output_format": "json"},
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["items"] == []


def test_async_execute_query_archive_delivers_to_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    output_path = tmp_path / "out" / "sessions.json"
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Delivered",
                    created_at=None,
                    updated_at=None,
                    message_count=1,
                    word_count=2,
                    tags=(),
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "output": str(output_path),
                "output_format": "json",
            },
        )
    )

    assert json.loads(output_path.read_text())["items"][0]["title"] == "Delivered"
    assert capsys.readouterr().out == ""
    _as_mock(env.ui.console).print.assert_called_once_with(f"Wrote to {output_path}")


def test_async_execute_query_archive_samples_copied_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["limit"] == 3
            assert kwargs["offset"] == 0
            assert kwargs["sample"] is True
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(async_execute_query(env, {"archive": True, "sample": 3, "output_format": "json"}))

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["limit"] == 3


def test_async_execute_query_archive_outputs_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stats(self, **kwargs: object) -> ArchiveStats:
            assert kwargs["origin"] == "codex-session"
            assert kwargs["tags"] == ("archive",)
            assert kwargs["session_ids"] == ()
            return ArchiveStats(
                total_sessions=1,
                total_messages=3,
                total_attachments=0,
                origins={"codex-session": 1},
                db_size_bytes=4096,
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "stats_only": True,
                "origin": "codex-session",
                "tag": "archive",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "stats"
    assert payload["total_sessions"] == 1
    assert payload["origins"] == {"codex-session": 1}


def test_async_execute_query_archive_count_uses_query_match_scope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def count_search_sessions(self, query: str, **kwargs: object) -> int:
            assert query == "needle"
            assert kwargs["origin"] == "codex-session"
            assert kwargs["tags"] == ("archive",)
            return 2

        def count_sessions(self, **_kwargs: object) -> int:
            raise AssertionError("queried analyze --count must not count the unsearched archive")

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("needle",),
                "count_only": True,
                "origin": "codex-session",
                "tag": "archive",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"mode": "count", "origin": "codex-session", "count": 2}


def test_async_execute_query_archive_count_applies_boolean_predicate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for polylogue-70qb: `then analyze --count` on a boolean
    (`sessions where <predicate>`) query must apply the compiled predicate,
    not fall back to the unfiltered archive count."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def count_search_sessions(self, query: str, **kwargs: object) -> int:
            raise AssertionError("boolean-only query must not take the FTS-search count path")

        def count_sessions(self, **kwargs: object) -> int:
            assert kwargs.get("boolean_predicate") is not None
            return 5

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("sessions where origin:codex-session",),
                "count_only": True,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 5


def test_async_execute_query_archive_search_stats_are_not_page_capped_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_session_ids(self, query: str, **kwargs: object) -> tuple[str, ...]:
            assert query == "needle"
            assert kwargs["limit"] is None
            return ("codex-session:native-1", "codex-session:native-2")

        def stats(self, **kwargs: object) -> ArchiveStats:
            assert kwargs["session_ids"] == ("codex-session:native-1", "codex-session:native-2")
            return ArchiveStats(total_sessions=2, total_messages=5)

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("needle",),
                "stats_only": True,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "stats"
    assert payload["total_sessions"] == 2
    assert payload["total_messages"] == 5


def test_async_execute_query_archive_outputs_grouped_search_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_session_ids(self, query: str, **kwargs: object) -> tuple[str, ...]:
            assert query == "needle"
            assert kwargs["limit"] == 10
            return ("codex-session:native-1",)

        def search_summaries(self, *_args: object, **_kwargs: object) -> list[ArchiveSessionSearchHit]:
            raise AssertionError("stats grouping must not consume paged search hits")

        def stats_by(self, group_by: str, **kwargs: object) -> dict[str, int]:
            assert group_by == "tool"
            assert kwargs["session_ids"] == ("codex-session:native-1",)
            return {"read": 1}

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("needle",),
                "stats_by": "tool",
                "limit": 10,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "stats_by"
    assert payload["group_by"] == "tool"
    assert payload["items"] == [{"count": 1, "group": "read"}]


def test_async_execute_query_archive_search_maps_provider_to_origin(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_summaries(
            self,
            query: str,
            *,
            limit: int,
            offset: int,
            sort: str | None,
            reverse: bool,
            session_id: str | None,
            origin: str | None,
            origins: tuple[str, ...],
            excluded_origins: tuple[str, ...],
            tags: tuple[str, ...],
            excluded_tags: tuple[str, ...],
            repo_names: tuple[str, ...],
            project_refs: tuple[str, ...],
            has_types: tuple[str, ...],
            has_tool_use: bool,
            has_thinking: bool,
            has_paste: bool,
            tool_terms: tuple[str, ...],
            excluded_tool_terms: tuple[str, ...],
            action_terms: tuple[str, ...],
            excluded_action_terms: tuple[str, ...],
            action_sequence: tuple[str, ...],
            action_text_terms: tuple[str, ...],
            referenced_paths: tuple[str, ...],
            cwd_prefix: str | None,
            typed_only: bool,
            message_type: str | None,
            title: str | None,
            min_messages: int | None,
            max_messages: int | None,
            min_words: int | None,
            max_words: int | None,
            since_ms: int | None,
            until_ms: int | None,
            since_session_id: str | None,
        ) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            assert limit == 6
            assert offset == 0
            assert sort is None
            assert reverse is False
            assert session_id is None
            assert origin == "codex-session"
            assert origins == ("codex-session",)
            assert excluded_origins == ("chatgpt-export",)
            assert tags == ("review", "archive")
            assert excluded_tags == ("archived",)
            assert repo_names == ("polylogue",)
            assert has_types == ("tool_use", "thinking")
            assert has_tool_use is True
            assert has_thinking is True
            assert has_paste is False
            assert tool_terms == ("read",)
            assert excluded_tool_terms == ("write",)
            assert action_terms == ("file_read",)
            assert excluded_action_terms == ("file_write",)
            assert action_sequence == ("file_read", "shell")
            assert action_text_terms == ("README.md",)
            assert referenced_paths == ("/workspace/polylogue/README.md", "pyproject.toml")
            assert cwd_prefix == "/realm/project/polylogue"
            assert typed_only is True
            assert message_type == "tool_use"
            assert title == "Copied"
            assert min_messages == 1
            assert max_messages is None
            assert min_words == 3
            assert max_words is None
            assert since_ms == 1767312000000
            assert until_ms is None
            assert since_session_id is None
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    title="Copied",
                    snippet="[needle]",
                )
            ]

        def read_summary(self, session_id: str) -> ArchiveSessionSummary:
            assert session_id == "codex-session:native-1"
            return ArchiveSessionSummary(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Copied",
                created_at=None,
                updated_at=None,
                message_count=1,
                word_count=1,
                tags=(),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("needle",),
                "origin": "codex-session",
                "exclude_origin": "chatgpt-export",
                "tag": "review,archive",
                "exclude_tag": "archived",
                "repo": "polylogue",
                "has_type": "tool_use,thinking",
                "filter_has_tool_use": True,
                "filter_has_thinking": True,
                "tool": "Read",
                "exclude_tool": "Write",
                "action": "file_read",
                "exclude_action": "file_write",
                "action_sequence": "file_read,shell",
                "action_text": "README.md",
                "referenced_path": ("/workspace/polylogue/README.md", "pyproject.toml"),
                "cwd_prefix": "/realm/project/polylogue",
                "typed_only": True,
                "message_type": "tool-use",
                "title": "Copied",
                "min_messages": 1,
                "min_words": 3,
                "since": "2026-01-02T00:00:00Z",
                "limit": 5,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "search"
    assert payload["origin"] == "codex-session"
    assert payload["items"][0]["session"]["id"] == "codex-session:native-1"
    assert payload["items"][0]["session"]["origin"] == "codex-session"
    assert payload["items"][0]["match"]["message_id"] == "codex-session:native-1:m1"


def test_async_execute_query_archive_filters_multiple_providers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["origin"] is None
            assert kwargs["origins"] == ("codex-session", "chatgpt-export")
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "origin": "codex-session,chatgpt-export",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["items"] == []


def test_async_execute_query_archive_searches_within_session_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            assert token == "native-1"
            return "codex-session:native-1"

        def search_summaries(self, query: str, **kwargs: object) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            assert kwargs["session_id"] == "codex-session:native-1"
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    title="Copied",
                    snippet="[needle]",
                )
            ]

        def read_summary(self, session_id: str) -> ArchiveSessionSummary:
            assert session_id == "codex-session:native-1"
            return ArchiveSessionSummary(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Copied",
                created_at=None,
                updated_at=None,
                message_count=1,
                word_count=1,
                tags=(),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "native-1",
                "query": ("needle",),
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "search"
    assert payload["items"][0]["session"]["id"] == "codex-session:native-1"


def test_async_execute_query_archive_filters_since_session_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["since_session_id"] == "codex-session:anchor"
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "since_session_id": "codex-session:anchor",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "list"
    assert payload["items"] == []


def test_async_execute_query_archive_paginates_lists_with_cursor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)
    calls: list[tuple[int, int]] = []

    def summary(native_id: str) -> ArchiveSessionSummary:
        return ArchiveSessionSummary(
            session_id=f"codex-session:{native_id}",
            native_id=native_id,
            origin="codex-session",
            title=native_id,
            created_at=None,
            updated_at=None,
            message_count=1,
            word_count=1,
            tags=(),
        )

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            limit_value = kwargs["limit"]
            offset_value = kwargs["offset"]
            assert isinstance(limit_value, int)
            assert isinstance(offset_value, int)
            limit = limit_value
            offset = offset_value
            calls.append((limit, offset))
            rows = [summary("one"), summary("two"), summary("three"), summary("four")]
            return rows[offset : offset + limit]

    fake = FakeArchiveStore()
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: fake),
    )

    asyncio.run(async_execute_query(env, {"archive": True, "limit": 2, "output_format": "json"}))
    first_page = json.loads(capsys.readouterr().out)
    cursor = first_page["next_cursor"]

    assert [item["id"] for item in first_page["items"]] == ["codex-session:one", "codex-session:two"]
    assert decode_search_cursor(cursor).r == 2
    assert first_page["next_offset"] == 2

    asyncio.run(
        async_execute_query(
            env,
            {"archive": True, "limit": 2, "cursor": cursor, "output_format": "json"},
        )
    )
    second_page = json.loads(capsys.readouterr().out)

    assert calls == [(3, 0), (3, 2)]
    assert [item["id"] for item in second_page["items"]] == ["codex-session:three", "codex-session:four"]
    assert second_page["next_cursor"] is None


def test_async_execute_query_archive_open_prints_session_url(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            assert token == "codex-session:native-1"
            return token

        def read_session(self, session_id: str) -> None:
            raise AssertionError(f"open should not hydrate session {session_id}")

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "open_result": True,
                "print_url": True,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"url": "http://127.0.0.1:8766/?session=codex-session%3Anative-1"}


def test_async_execute_query_archive_open_uses_first_list_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def list_summaries(self, **kwargs: object) -> list[ArchiveSessionSummary]:
            assert kwargs["limit"] == 2
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:first",
                    native_id="first",
                    origin="codex-session",
                    title="First",
                    created_at=None,
                    updated_at=None,
                    message_count=1,
                    word_count=1,
                    tags=(),
                )
            ]

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    with patch("polylogue.cli.archive_query.webbrowser.open") as mock_open:
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "open_result": True,
                    "limit": 1,
                    "output_format": "json",
                },
            )
        )

    assert capsys.readouterr().out == ""
    mock_open.assert_called_once_with("http://127.0.0.1:8766/?session=codex-session%3Afirst")
    _as_mock(env.ui.console).print.assert_called_once_with(
        "Opened: http://127.0.0.1:8766/?session=codex-session%3Afirst"
    )


def test_async_execute_query_archive_streams_session_messages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            return token

        def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
            return ArchiveSessionEnvelope(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Streamed",
                active_leaf_message_id="codex-session:native-1:m2",
                messages=(
                    ArchiveMessageRow(
                        message_id="codex-session:native-1:m1",
                        native_id="m1",
                        role="user",
                        position=0,
                        variant_index=0,
                        is_active_path=True,
                        is_active_leaf=False,
                        blocks=(
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m1:0",
                                message_id="codex-session:native-1:m1",
                                block_type="text",
                                text="stream user",
                            ),
                        ),
                    ),
                    ArchiveMessageRow(
                        message_id="codex-session:native-1:m2",
                        native_id="m2",
                        role="assistant",
                        position=1,
                        variant_index=0,
                        is_active_path=True,
                        is_active_leaf=True,
                        blocks=(
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m2:0",
                                message_id="codex-session:native-1:m2",
                                block_type="text",
                                text="stream assistant",
                            ),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "stream": True,
                "limit": 1,
                "output_format": "json",
            },
        )
    )

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["role"] for line in lines] == ["user"]
    assert lines[0]["blocks"][0]["text"] == "stream user"


def test_async_execute_query_archive_accepts_lexical_retrieval_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_summaries(self, query: str, **kwargs: object) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    # Empty search is the no-results contract: status 2 with the (empty) search
    # envelope still emitted on stdout for machine consumers.
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "query": ("needle",),
                    "lexical": True,
                    "retrieval_lane": "dialogue",
                    "output_format": "json",
                },
            )
        )
    assert exc_info.value.code == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "search"
    assert payload["items"] == []


def test_async_execute_query_archive_sorts_search_terms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def search_summaries(self, query: str, **kwargs: object) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            assert kwargs["sort"] == "messages"
            assert kwargs["reverse"] is True
            return []

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    # Empty search is the no-results contract: status 2 with the (empty) search
    # envelope still emitted on stdout for machine consumers.
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "query": ("needle",),
                    "sort": "messages",
                    "reverse": True,
                    "output_format": "json",
                },
            )
        )
    assert exc_info.value.code == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "search"
    assert payload["items"] == []


def test_async_execute_query_archive_uses_vector_provider_for_semantic_search(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeVectorProvider:
        def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
            assert text == "meaningful prompt"
            assert limit == 6
            return [("codex-session:native-1:m1", 0.2), ("codex-session:native-2:m1", 0.3)]

    class FakeArchiveStore:
        index_db_path = archive_root / "index.db"

        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def semantic_summaries(
            self,
            scored_message_ids: list[tuple[str, float]],
            **kwargs: object,
        ) -> list[ArchiveSessionSearchHit]:
            assert scored_message_ids == [("codex-session:native-1:m1", 0.2), ("codex-session:native-2:m1", 0.3)]
            assert kwargs["limit"] == 6
            assert kwargs["offset"] == 0
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    title="Semantic",
                    snippet="semantic hit",
                ),
                ArchiveSessionSearchHit(
                    rank=2,
                    session_id="codex-session:native-2",
                    block_id="codex-session:native-2:m1:0",
                    message_id="codex-session:native-2:m1",
                    origin="codex-session",
                    title="Semantic 2",
                    snippet="semantic hit 2",
                ),
            ]

        def read_summary(self, session_id: str) -> ArchiveSessionSummary:
            native_id = session_id.removeprefix("codex-session:")
            return ArchiveSessionSummary(
                session_id=session_id,
                native_id=native_id,
                origin="codex-session",
                title=f"Semantic {native_id[-1]}",
                created_at=None,
                updated_at=None,
                message_count=1,
                word_count=1,
                tags=(),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )
    monkeypatch.setattr(
        "polylogue.cli.archive_query.create_vector_provider", lambda *args, **kwargs: FakeVectorProvider()
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "similar_text": "meaningful prompt",
                "limit": 1,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval_lane"] == "semantic"
    assert payload["items"][0]["session"]["id"] == "codex-session:native-1"
    assert decode_search_cursor(payload["next_cursor"]).lane == "semantic"


def test_async_execute_query_archive_accepts_explicit_semantic_lane(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeVectorProvider:
        def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
            assert text == "meaningful prompt"
            return [("codex-session:native-1:m1", 0.2)]

    class FakeArchiveStore:
        index_db_path = archive_root / "index.db"

        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def semantic_summaries(
            self,
            scored_message_ids: list[tuple[str, float]],
            **kwargs: object,
        ) -> list[ArchiveSessionSearchHit]:
            assert scored_message_ids == [("codex-session:native-1:m1", 0.2)]
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    title="Semantic",
                    snippet="semantic hit",
                )
            ]

        def read_summary(self, session_id: str) -> ArchiveSessionSummary:
            assert session_id == "codex-session:native-1"
            return ArchiveSessionSummary(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Semantic",
                created_at=None,
                updated_at=None,
                message_count=1,
                word_count=1,
                tags=(),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )
    monkeypatch.setattr(
        "polylogue.cli.archive_query.create_vector_provider", lambda *args, **kwargs: FakeVectorProvider()
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("meaningful prompt",),
                "retrieval_lane": "semantic",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval_lane"] == "semantic"
    assert payload["items"][0]["session"]["id"] == "codex-session:native-1"


def test_archive_tiers_semantic_query_uses_active_root_embeddings_db(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_root = tmp_path / "archive"
    active_root = tmp_path / "active"
    configured_root.mkdir()
    active_root.mkdir()
    (active_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = configured_root
    # db_path named index.db resolves the active archive root to its parent,
    # overriding the configured (empty) archive_root — the supported override.
    config.db_path = active_root / "index.db"
    env = _make_env(repo=MagicMock(), config=config)
    observed_vector_db_paths: list[Path] = []

    class FakeVectorProvider:
        def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
            assert text == "meaningful prompt"
            return [("codex-session:native-1:m1", 0.2)]

    class FakeArchiveStore:
        index_db_path = active_root / "index.db"

        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def semantic_summaries(
            self,
            scored_message_ids: list[tuple[str, float]],
            **kwargs: object,
        ) -> list[ArchiveSessionSearchHit]:
            assert scored_message_ids == [("codex-session:native-1:m1", 0.2)]
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    title="Semantic",
                    snippet="semantic hit",
                )
            ]

        def read_summary(self, session_id: str) -> ArchiveSessionSummary:
            assert session_id == "codex-session:native-1"
            return ArchiveSessionSummary(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Semantic",
                created_at=None,
                updated_at=None,
                message_count=1,
                word_count=1,
                tags=(),
            )

    def fake_open_existing(cls: type[object], root: Path) -> FakeArchiveStore:
        assert root == active_root
        return FakeArchiveStore()

    def fake_create_vector_provider(config_arg: object, *, db_path: Path) -> FakeVectorProvider:
        assert config_arg is config
        observed_vector_db_paths.append(db_path)
        return FakeVectorProvider()

    monkeypatch.setattr("polylogue.cli.archive_query.ArchiveStore.open_existing", classmethod(fake_open_existing))
    monkeypatch.setattr("polylogue.cli.archive_query.create_vector_provider", fake_create_vector_provider)

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "query": ("meaningful prompt",),
                "retrieval_lane": "semantic",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval_lane"] == "semantic"
    assert observed_vector_db_paths == [active_root / "embeddings.db"]


def test_async_execute_query_archive_adds_tags_to_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            return token

        def add_user_tags(self, session_ids: tuple[str, ...], tags: tuple[str, ...]) -> int:
            assert session_ids == ("codex-session:native-1",)
            assert tags == ("review", "ready")
            return 2

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "add_tag": ("review", "ready"),
                "output_format": "json",
            },
        )
    )

    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "operation": "add_tag",
        "affected_count": 2,
    }


def test_async_execute_query_archive_deletes_session_by_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            return token

        def delete_sessions(self, session_ids: tuple[str, ...]) -> int:
            assert session_ids == ("codex-session:native-1",)
            return 1

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "delete_matched": True,
                "force": True,
                "output_format": "json",
            },
        )
    )

    assert json.loads(capsys.readouterr().out) == {
        "status": "deleted",
        "operation": "delete",
        "session_count": 1,
        "affected_count": 1,
    }


def test_async_execute_query_archive_sets_session_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            return token

        def set_user_metadata(self, session_ids: tuple[str, ...], pairs: tuple[tuple[str, str], ...]) -> int:
            assert session_ids == ("codex-session:native-1",)
            assert pairs == (("priority", "high"),)
            return 1

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "set_meta": (("priority", "high"),),
                "output_format": "json",
            },
        )
    )

    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "operation": "set_meta",
        "affected_count": 1,
    }


def test_async_execute_query_archive_delete_dry_run_does_not_delete(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            return token

        def delete_sessions(self, session_ids: tuple[str, ...]) -> int:
            raise AssertionError("dry-run must not delete")

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "delete_matched": True,
                "dry_run": True,
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "preview"
    assert payload["session_count"] == 1
    assert payload["affected_count"] == 0
    assert payload["session_ids"] == ["codex-session:native-1"]


def test_async_execute_query_archive_rejects_combined_delete_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)
    monkeypatch.setattr("polylogue.cli.archive_query.ArchiveStore.open_existing", MagicMock())

    with pytest.raises(click.UsageError, match="cannot combine delete with --set"):
        asyncio.run(
            async_execute_query(
                env,
                {"archive": True, "delete_matched": True, "set_meta": (("priority", "1"),)},
            )
        )


def test_async_execute_query_archive_reads_session_by_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            assert token == "codex-session:native-1"
            return token

        def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
            assert session_id == "codex-session:native-1"
            return ArchiveSessionEnvelope(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Copied",
                active_leaf_message_id="codex-session:native-1:m1",
                messages=(
                    ArchiveMessageRow(
                        message_id="codex-session:native-1:m1",
                        native_id="m1",
                        role="user",
                        position=0,
                        variant_index=0,
                        is_active_path=True,
                        is_active_leaf=True,
                        blocks=(
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m1:0",
                                message_id="codex-session:native-1:m1",
                                block_type="text",
                                text="hello from v1",
                            ),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {"archive": True, "conv_id": "codex-session:native-1", "output_format": "json"},
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "session"
    assert payload["session_id"] == "codex-session:native-1"
    assert payload["source"] == "codex-session"
    assert payload["origin"] == "codex-session"
    assert payload["messages"][0]["blocks"][0]["text"] == "hello from v1"


def test_async_execute_query_archive_reads_session_messages_without_projection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def resolve_session_id(self, token: str) -> str:
            return token

        def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
            return ArchiveSessionEnvelope(
                session_id=session_id,
                native_id="native-1",
                origin="codex-session",
                title="Projected session",
                active_leaf_message_id="codex-session:native-1:m2",
                messages=(
                    ArchiveMessageRow(
                        message_id="codex-session:native-1:m1",
                        native_id="m1",
                        role="user",
                        position=0,
                        variant_index=0,
                        is_active_path=True,
                        is_active_leaf=False,
                        blocks=(
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m1:0",
                                message_id="codex-session:native-1:m1",
                                block_type="text",
                                text="keep user prose",
                            ),
                        ),
                    ),
                    ArchiveMessageRow(
                        message_id="codex-session:native-1:m2",
                        native_id="m2",
                        role="assistant",
                        position=1,
                        variant_index=0,
                        is_active_path=True,
                        is_active_leaf=True,
                        blocks=(
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m2:0",
                                message_id="codex-session:native-1:m2",
                                block_type="thinking",
                                text="drop reasoning",
                            ),
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m2:1",
                                message_id="codex-session:native-1:m2",
                                block_type="tool_use",
                                text=None,
                                tool_name="Read",
                                tool_id="tool-1",
                                semantic_type="file_read",
                            ),
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m2:2",
                                message_id="codex-session:native-1:m2",
                                block_type="tool_result",
                                text="drop file contents",
                                tool_id="tool-1",
                            ),
                            ArchiveBlockRow(
                                block_id="codex-session:native-1:m2:3",
                                message_id="codex-session:native-1:m2",
                                block_type="text",
                                text="keep assistant prose",
                            ),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        classmethod(lambda cls, root: FakeArchiveStore()),
    )

    asyncio.run(
        async_execute_query(
            env,
            {
                "archive": True,
                "conv_id": "codex-session:native-1",
                "output_format": "json",
            },
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert [message["role"] for message in payload["messages"]] == ["user", "assistant"]
    assert payload["messages"][0]["blocks"] == [
        {
            "block_id": "codex-session:native-1:m1:0",
            "block_type": "text",
            "message_id": "codex-session:native-1:m1",
            "semantic_type": None,
            "text": "keep user prose",
            "tool_id": None,
            "tool_name": None,
        }
    ]
    assert [block["block_type"] for block in payload["messages"][1]["blocks"]] == [
        "thinking",
        "tool_use",
        "tool_result",
        "text",
    ]


def test_async_execute_query_archive_rejects_unsupported_historical_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    env = _make_env(repo=MagicMock(), config=config)
    monkeypatch.setattr("polylogue.cli.archive_query.ArchiveStore.open_existing", MagicMock())

    with pytest.raises(click.UsageError, match="[Hh]ybrid retrieval requires lexical query"):
        asyncio.run(async_execute_query(env, {"archive": True, "retrieval_lane": "hybrid"}))


# ---------------------------------------------------------------------------
# CLI search surface (native, driven through CliRunner)
# ---------------------------------------------------------------------------


SEARCH_FILTER_CASES = [
    ("origin", ["Python", "--origin", "chatgpt-export"], 0, None),
    ("since_valid", ["Python", "--since", "__DYNAMIC_DATE__"], 0, None),
    # Archive input validation raises click.UsageError → status 2 (Click's
    # usage-error convention), consistent across all archive filter validation.
    ("since_invalid", ["Python", "--since", "not-a-date"], 1, "date"),
    ("limit_read", ["JavaScript", "--limit", "1", "then", "read"], 0, None),
]

SEARCH_FORMAT_CASES = [
    ("json_read", ["Python", "then", "read", "-f", "json"], "json_read"),
    ("json_single", ["JavaScript", "-f", "json", "--limit", "1"], "json_single"),
    (
        "json_unit_messages",
        ["-f", "json", "messages", "where", "role:assistant", "AND", "text:Python"],
        "json_unit_messages",
    ),
    ("read_mode", ["async", "then", "read"], "plain_list"),
    ("markdown", ["Rust", "-f", "markdown", "--limit", "1"], "markdown"),
]


class TestSearchQueryContracts:
    """Matrix coverage for search filters and output formats."""

    def test_debug_timing_keeps_query_output_on_stdout(
        self, search_workspace: SearchWorkspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opt-in timing reports real CLI phases without corrupting JSON output.

        This invokes the production Click root, query compiler, archive open,
        list execution, and renderer.  Removing any checkpoint around those
        production dependencies makes the corresponding stderr phase assertion
        fail while the JSON assertion protects the stdout/stderr contract.
        """
        from polylogue.cli import cli

        del search_workspace
        monkeypatch.setenv("POLYLOGUE_DEBUG_TIMING", "1")

        result = CliRunner().invoke(cli, ["--plain", "find", "origin:chatgpt-export", "-f", "json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["mode"] == "list"
        assert "polylogue timing" in result.stderr
        for phase in ("cli-callback", "archive-query-import", "config", "compile", "db-open", "execute", "render"):
            assert phase in result.stderr

    def test_structured_only_cli_query_skips_absent_message_fts(self, search_workspace: SearchWorkspace) -> None:
        """A field-only CLI query must keep working when lexical search is unavailable.

        This exercises the production ``ArchiveStore.list_summaries`` route
        against the same derived index the CLI opens.  Replacing the
        structured-only discriminator in ``_execute_archive_query_stdout``
        with an unconditional ``_query_hits`` call would instead invoke
        ``ArchiveStore.search_summaries`` and fail its FTS-readiness check.
        """
        from polylogue.cli import cli

        index_db = search_workspace["archive_root"] / "index.db"
        with sqlite3.connect(index_db) as conn:
            for trigger in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE messages_fts")

        result = CliRunner().invoke(
            cli,
            ["--plain", "find", "origin:chatgpt-export", "-f", "json"],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["mode"] == "list"
        assert [item["id"] for item in payload["items"]] == ["chatgpt-export:ext-conv1"]

    @pytest.mark.parametrize(
        "case_id,args,expected_exit,error_hint",
        SEARCH_FILTER_CASES,
    )
    def test_filter_contract(
        self,
        search_workspace: SearchWorkspace,
        case_id: str,
        args: list[str],
        expected_exit: int,
        error_hint: str | None,
    ) -> None:
        """Filter flags produce expected status codes and validation behavior."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        resolved_args = list(args)
        if "__DYNAMIC_DATE__" in resolved_args:
            idx = resolved_args.index("__DYNAMIC_DATE__")
            from datetime import datetime, timedelta

            resolved_args[idx] = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        result = runner.invoke(cli, ["--plain", "find", *resolved_args])
        assert result.exit_code == expected_exit, case_id
        if error_hint:
            assert error_hint in result.output.lower(), case_id

    @pytest.mark.parametrize(
        "case_id,args,expectation",
        SEARCH_FORMAT_CASES,
    )
    def test_output_contract(
        self, search_workspace: SearchWorkspace, case_id: str, args: list[str], expectation: str
    ) -> None:
        """Output format combinations produce parseable and mode-consistent output."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "find", *args])
        assert result.exit_code == 0, case_id

        if expectation == "json_list":
            # #1618: envelope shape, not bare array.
            data = json.loads(result.output)
            assert isinstance(data, dict), case_id
            assert isinstance(data.get("items"), list), case_id
            assert data["items"] and "id" in data["items"][0], case_id
        elif expectation == "json_read":
            data = json.loads(result.output)
            assert isinstance(data, dict), case_id
            assert isinstance(data.get("items"), list), case_id
            assert data["items"], case_id
            first = data["items"][0]
            assert "match" in first, case_id
            assert first["session"]["id"] == "chatgpt-export:ext-conv1", case_id
        elif expectation == "json_single":
            data = json.loads(result.output)
            assert isinstance(data, (list, dict)), case_id
        elif expectation == "json_unit_messages":
            data = json.loads(result.output)
            assert data["mode"] == "query-unit", case_id
            assert data["unit"] == "message", case_id
            assert [item["message_id"] for item in data["items"]] == ["chatgpt-export:ext-conv1:m2"], case_id
            assert data["items"][0]["role"] == "assistant", case_id
        elif expectation == "plain_list":
            assert result.output.strip(), case_id
        elif expectation == "markdown":
            assert "#" in result.output or "Rust" in result.output, case_id

    def test_unit_source_applies_session_filters(self, search_workspace: SearchWorkspace) -> None:
        """Terminal row queries apply session filters instead of broadening rows."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "find",
                "messages",
                "where",
                "text:Python",
                "--origin",
                "claude-code-session",
                "-f",
                "json",
            ],
        )

        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "message"
        assert payload["items"] == []

    def test_assertion_unit_source_returns_json_envelope(self, search_workspace: SearchWorkspace) -> None:
        """Assertion terminal rows execute through the CLI JSON envelope."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "find",
                "-f",
                "json",
                "assertions",
                "where",
                "session.origin:chatgpt-export",
                "AND",
                "kind:decision",
                "AND",
                "status:active",
                "AND",
                "text:Python",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "assertion"
        assert [item["assertion_id"] for item in payload["items"]] == ["assertion-cli-decision"]

    def test_assertion_unit_source_renders_plain_rows(self, search_workspace: SearchWorkspace) -> None:
        """Assertion terminal rows render in text mode as well as machine envelopes."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "find",
                "assertions",
                "where",
                "kind:decision",
                "AND",
                "text:terminal",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "assertion-cli-decision [decision/active] Python terminal assertion" in result.output

    def test_action_unit_aggregate_routes_to_query_units(self, search_workspace: SearchWorkspace) -> None:
        """Action aggregate pipelines execute as terminal units, not session selectors."""
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = search_workspace["archive_root"] / "index.db"
        (
            SessionBuilder(index_db, "action-followup-cli")
            .provider("codex")
            .title("action followup cli")
            .add_message(
                "m-tool",
                role="assistant",
                text="Ran a failing command.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "bash-1",
                        "name": "Bash",
                        "tool_input": {"command": "pytest cli"},
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "bash-1",
                        "text": "failed",
                        "tool_result_is_error": 1,
                        "tool_result_exit_code": 1,
                    },
                ],
            )
            .add_message("m-followup", role="assistant", text="I will inspect the generated fixture rows.")
            .save()
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "--format",
                "json",
                "actions",
                "where",
                "is_error:true",
                "|",
                "group",
                "by",
                "followup_class",
                "|",
                "count",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["mode"] == "query-unit-aggregate"
        assert payload["unit"] == "action"
        assert {"group_key": "silent_proceed", "count": 1} in [
            {"group_key": item["group_key"], "count": item["count"]} for item in payload["items"]
        ]

    def test_delegation_unit_source_renders_bounded_rows(self, search_workspace: SearchWorkspace) -> None:
        from polylogue.cli import cli
        from tests.infra.storage_records import SessionBuilder

        index_db = search_workspace["archive_root"] / "index.db"
        instruction = "review CLI delegation evidence " + "carefully " * 40
        parent_builder = (
            SessionBuilder(index_db, "delegation-cli-parent")
            .provider("codex")
            .title("delegation parent session")
            .add_message(
                "dispatch",
                role="assistant",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_id": "task-cli",
                        "tool_name": "Task",
                        "semantic_type": "subagent",
                        "tool_input": {"prompt": instruction, "model": "mini"},
                    },
                    {"type": "tool_result", "tool_id": "task-cli", "text": "child report"},
                ],
            )
        )
        parent_id = parent_builder.native_session_id()
        parent_builder.save()
        child_builder = (
            SessionBuilder(index_db, "delegation-cli-child")
            .provider("codex")
            .title("delegation child session")
            .add_message("child-result", role="assistant", text="child report")
        )
        child_id = child_builder.native_session_id()
        child_builder.save()
        with sqlite3.connect(index_db) as conn:
            conn.execute(
                "UPDATE blocks SET semantic_type = 'subagent' WHERE tool_id = 'task-cli' AND block_type = 'tool_use'"
            )
            conn.execute(
                """
                INSERT INTO session_links (
                    src_session_id, dst_origin, dst_native_id, link_type,
                    resolved_dst_session_id, inheritance, method, observed_at_ms
                ) VALUES (?, 'codex-session', ?, 'subagent', ?, 'spawned-fresh', 'test', 1)
                """,
                (child_id, "ext-delegation-cli-parent", parent_id),
            )
            delegation_blocks = conn.execute(
                "SELECT block_type, tool_name, tool_id, semantic_type FROM blocks WHERE session_id = ?",
                (parent_id,),
            ).fetchall()
            assert delegation_blocks, "delegation fixture wrote no blocks"
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM actions WHERE session_id = ? AND semantic_type = 'subagent'",
                    (parent_id,),
                ).fetchone()[0]
                == 1
            ), [tuple(row) for row in delegation_blocks]
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM delegations WHERE parent_session_id = ?",
                    (parent_id,),
                ).fetchone()[0]
                == 1
            )

        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "--format",
                "json",
                "delegations",
                "where",
                "mapping_state:resolved",
                "AND",
                "instruction:evidence",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["unit"] == "delegation"
        [item] = payload["items"]
        assert item["parent_session_id"] == parent_id
        assert item["child_session_id"] == child_id
        assert item["instruction_preview"] == instruction[:240]
        assert item["instruction_truncated"] is True
        assert "instruction_payload" not in item

        read_result = CliRunner().invoke(
            cli,
            ["--plain", "read", item["delegation_ref"], "--format", "json"],
        )
        assert read_result.exit_code == 0, read_result.output
        card = json.loads(read_result.output)
        assert card["payload_kind"] == "delegation-card"
        assert card["summary"] == instruction[:240]
        assert card["payload"]["instruction"] == instruction
        assert card["payload"]["attempt"]["parent_session_id"] == parent_id
        assert card["payload"]["attempt"]["child_session_id"] == child_id

    def test_run_unit_source_renders_plain_rows(self, search_workspace: SearchWorkspace) -> None:
        """Run terminal rows render through the CLI instead of failing after query execution."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "find",
                "runs",
                "where",
                "session.repo:polylogue",
                "AND",
                "role:subagent",
                "AND",
                "agent:subagent",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "run:codex-session:ext-run-hit-subagent [subagent/completed]" in result.output
        assert "agent:codex/subagent" in result.output

    def test_context_snapshot_unit_source_routes_to_query_units(self, search_workspace: SearchWorkspace) -> None:
        """Context snapshot expressions stay on terminal query-unit routing, not stats."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "find",
                "-f",
                "json",
                "context-snapshots",
                "where",
                "boundary:session_start",
                "AND",
                "text:ext-conv1",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "context-snapshot"
        assert [item["session_id"] for item in payload["items"]] == ["chatgpt-export:ext-conv1"]

    def test_context_snapshot_unit_source_error_uses_descriptor_label(self, search_workspace: SearchWorkspace) -> None:
        """Root-query incompatibility errors use descriptor-owned source labels."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--plain",
                "--stream",
                "find",
                "context-snapshots",
                "where",
                "boundary:session_start",
            ],
        )

        assert result.exit_code == 2
        assert "context-snapshots where queries return context-snapshot rows" in result.output


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_no_results(self, search_workspace: SearchWorkspace) -> None:
        """Handle query with no matching results."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        # Query mode with non-matching term
        result = runner.invoke(cli, ["--plain", "find", "nonexistent_term_xyz"])
        # exit_code 2 = no results (valid outcome)
        assert result.exit_code == 2
        assert "no session" in result.output.lower() or "matched" in result.output.lower()

    def test_no_args_runs_archive_query_without_requiring_terms(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No args routes through the archive executor (no query term required).

        After the archive-route cleanup (#1743), a bare ``polylogue`` runs
        the query. On an
        empty archive that matches nothing, so the documented no-results
        contract (exit code 2) applies rather than a crash or usage error.
        """
        from polylogue.cli import cli

        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain"])
        assert result.exit_code in (0, 2)

    def test_search_case_insensitive(self, search_workspace: SearchWorkspace) -> None:
        """Search is case-insensitive."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        # Query mode with read action to ensure consistent output.
        result_lower = runner.invoke(cli, ["--plain", "find", "python", "then", "read", "-f", "json"])
        result_upper = runner.invoke(cli, ["--plain", "find", "PYTHON", "then", "read", "-f", "json"])

        # Both should have same exit code
        assert result_lower.exit_code == result_upper.exit_code

        if result_lower.exit_code == 0:
            # Both should find results (FTS5 is case-insensitive by default).
            # #1618: envelope shape carries items list under "items".
            data_lower = json.loads(result_lower.output)
            data_upper = json.loads(result_upper.output)
            items_lower = data_lower["items"] if isinstance(data_lower, dict) else data_lower
            items_upper = data_upper["items"] if isinstance(data_upper, dict) else data_upper
            assert len(items_lower) > 0
            assert len(items_upper) > 0

    def test_search_multiple_terms(self, search_workspace: SearchWorkspace) -> None:
        """Search with multiple query terms."""
        from polylogue.cli import cli

        del search_workspace

        runner = CliRunner()
        # Query mode: multiple positional args = multiple query terms
        result = runner.invoke(cli, ["--plain", "find", "Python", "exception", "then", "read", "-f", "json"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            # #1618: envelope shape, not bare array.
            data = json.loads(result.output)
            assert isinstance(data, dict)
            assert isinstance(data.get("items"), list)


class TestSearchIndexRebuild:
    """Tests for automatic index rebuild on missing index."""

    def test_search_handles_missing_index(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Search handles missing index gracefully."""
        from polylogue.cli import cli
        from tests.infra.storage_records import DbFactory

        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create session without building index
        db_path = cli_workspace["db_path"]
        factory = DbFactory(db_path)
        factory.create_session(
            id="c1",
            provider="test",
            title="Test",
            messages=[{"id": "m1", "role": "user", "text": "searchable content"}],
        )

        runner = CliRunner()
        # Query mode
        result = runner.invoke(cli, ["--plain", "find", "searchable"])
        # Should either succeed (rebuild worked) or report no results.
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            assert "searchable" in result.output.lower() or "c1" in result.output
        else:
            assert "no session" in result.output.lower() or "matched" in result.output.lower()


def test_daemon_unit_fast_path_defers_session_only_modes_to_local_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-only flags on unit queries must keep failing, daemon or not.

    Anti-vacuity: without the fast-path guard, a reachable daemon serves
    /api/query-units rows for ``--count`` unit queries, silently ignoring the
    flag instead of raising the local UsageError.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").touch()
    config = MagicMock()
    config.archive_root = archive_root
    config.db_path = archive_root / "index.db"
    config.daemon_url = "http://127.0.0.1:9876"
    config.api_auth_token = None
    env = _make_env(repo=MagicMock(), config=config)

    monkeypatch.setattr(
        "polylogue.cli.archive_query._fetch_daemon_payload",
        MagicMock(side_effect=AssertionError("session-only unit query must not take the daemon fast path")),
    )
    store = MagicMock()
    store.__enter__ = MagicMock(return_value=MagicMock())
    store.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(
        "polylogue.cli.archive_query.ArchiveStore.open_existing",
        MagicMock(return_value=store),
    )

    with pytest.raises(click.UsageError, match="do not combine"):
        asyncio.run(
            async_execute_query(
                env,
                {
                    "archive": True,
                    "query": ("messages where role:assistant AND text:timeout",),
                    "count_only": True,
                    "output_format": "json",
                },
            )
        )
