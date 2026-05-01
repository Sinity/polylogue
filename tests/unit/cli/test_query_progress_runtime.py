from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.cli.query_contracts import QueryDeliveryTarget, QueryOutputSpec
from polylogue.cli.query_progress import (
    QuerySlowNotice,
    _action_event_state,
    _action_fallback_detail,
    _async_method,
    _format_elapsed,
    build_query_slow_notice,
    observe_slow_query,
    should_emit_slow_query_notes,
    slow_query_notice_threshold,
)
from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.storage.action_events.artifacts import ActionEventArtifactState


def _output_spec(output_format: str) -> QueryOutputSpec:
    return QueryOutputSpec(
        output_format=output_format,
        destinations=(QueryDeliveryTarget.parse("stdout"),),
        fields=None,
        dialogue_only=False,
        message_roles=(),
        transform=None,
        list_mode=False,
        print_path=False,
    )


def test_slow_query_notice_threshold_and_output_mode_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS", raising=False)
    assert slow_query_notice_threshold() == 2.0

    monkeypatch.setenv("POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS", "1.5")
    assert slow_query_notice_threshold() == 1.5

    monkeypatch.setenv("POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS", "-4")
    assert slow_query_notice_threshold() == 0.0

    monkeypatch.setenv("POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS", "invalid")
    assert slow_query_notice_threshold() == 2.0

    assert should_emit_slow_query_notes(_output_spec("markdown")) is True
    assert should_emit_slow_query_notes(_output_spec("json")) is False


def test_format_elapsed_and_async_method_cover_all_branches() -> None:
    assert _format_elapsed(1.25) == "1.2s"
    assert _format_elapsed(12.4) == "12s"
    assert _format_elapsed(125.0) == "2m05s"

    async def async_method() -> str:
        return "ok"

    holder = SimpleNamespace(run=async_method, value="not-callable")
    assert _async_method(holder, "run") is async_method
    assert _async_method(holder, "value") is None
    assert _async_method(holder, "missing") is None


@pytest.mark.asyncio
async def test_action_event_state_handles_missing_methods_and_failures() -> None:
    assert await _action_event_state(object()) is None

    repo = SimpleNamespace(get_action_event_artifact_state=AsyncMock(side_effect=RuntimeError("boom")))
    assert await _action_event_state(repo) is None

    state = ActionEventArtifactState(
        source_conversations=2,
        materialized_conversations=1,
        materialized_rows=2,
        fts_rows=1,
        stale_rows=1,
    )
    ready_repo = SimpleNamespace(get_action_event_artifact_state=AsyncMock(return_value=state))
    assert await _action_event_state(ready_repo) == state
    assert (
        _action_fallback_detail(
            ActionEventArtifactState(
                source_conversations=1,
                materialized_conversations=1,
                materialized_rows=1,
                fts_rows=1,
            )
        )
        is None
    )


@pytest.mark.asyncio
async def test_build_query_slow_notice_handles_plan_failures_and_pending_action_models() -> None:
    selection = ConversationQuerySpec(action_terms=("shell",))

    with patch.object(ConversationQuerySpec, "to_plan", side_effect=RuntimeError("no plan")):
        notice = await build_query_slow_notice(object(), selection, route="search")
    assert notice == QuerySlowNotice(route="search", retrieval_lane="unknown", fallback_detail=None)

    plan = ConversationQuerySpec(action_terms=("shell",)).to_plan()
    pending_state = ActionEventArtifactState(
        source_conversations=3,
        materialized_conversations=1,
        materialized_rows=2,
        fts_rows=1,
        stale_rows=1,
    )
    repo = SimpleNamespace(get_action_event_artifact_state=AsyncMock(return_value=pending_state))
    with (
        patch.object(ConversationQuerySpec, "to_plan", return_value=plan),
        patch("polylogue.cli.query_progress.uses_action_read_model", return_value=True),
    ):
        notice = await build_query_slow_notice(repo, selection, route="stats")

    assert notice.route == "stats"
    assert notice.retrieval_lane == plan.retrieval_lane
    assert notice.fallback_detail == pending_state.repair_detail()


@pytest.mark.asyncio
async def test_observe_slow_query_handles_disabled_immediate_and_delayed_paths() -> None:
    assert (
        await observe_slow_query(
            asyncio.sleep(0, result="done"),
            enabled=False,
            notice_factory=AsyncMock(),
        )
        == "done"
    )

    immediate_notice_factory = AsyncMock()
    assert (
        await observe_slow_query(
            asyncio.sleep(0, result="fast"),
            enabled=True,
            threshold_seconds=0.05,
            notice_factory=immediate_notice_factory,
        )
        == "fast"
    )
    immediate_notice_factory.assert_not_awaited()

    emitted: list[str] = []

    async def _slow_operation() -> str:
        await asyncio.sleep(0.01)
        return "slow"

    delayed_notice_factory = AsyncMock(return_value=QuerySlowNotice(route="show", retrieval_lane="hybrid"))
    assert (
        await observe_slow_query(
            _slow_operation(),
            enabled=True,
            threshold_seconds=0.0,
            notice_factory=delayed_notice_factory,
            emit=emitted.append,
        )
        == "slow"
    )

    delayed_notice_factory.assert_awaited_once()
    assert emitted == ["Query still running after 0.0s (route: show; retrieval: hybrid)."]


@pytest.mark.asyncio
async def test_observe_slow_query_treats_notice_factory_failures_as_non_fatal() -> None:
    async def _operation() -> str:
        await asyncio.sleep(0.01)
        return "done"

    async def _notice() -> QuerySlowNotice:
        raise RuntimeError("notice failed")

    emitted: list[str] = []
    assert (
        await observe_slow_query(
            _operation(),
            enabled=True,
            threshold_seconds=0.0,
            notice_factory=_notice,
            emit=emitted.append,
        )
        == "done"
    )
    assert emitted == []


@pytest.mark.asyncio
async def test_observe_slow_query_does_not_wait_for_notice_after_operation_finishes() -> None:
    notice_started = asyncio.Event()
    notice_cancelled = asyncio.Event()

    async def _operation() -> str:
        await asyncio.sleep(0.01)
        return "done"

    async def _notice() -> QuerySlowNotice:
        notice_started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            notice_cancelled.set()
            raise
        return QuerySlowNotice(route="show", retrieval_lane="hybrid")

    result = await observe_slow_query(
        _operation(),
        enabled=True,
        threshold_seconds=0.0,
        notice_factory=_notice,
        emit=lambda _message: None,
    )

    assert result == "done"
    assert notice_started.is_set()
    assert notice_cancelled.is_set()


@pytest.mark.asyncio
async def test_observe_slow_query_cancels_operation_when_caller_cancels() -> None:
    operation_cancelled = asyncio.Event()
    notice_started = asyncio.Event()
    notice_cancelled = asyncio.Event()

    async def _operation() -> str:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            operation_cancelled.set()
            raise
        return "done"

    async def _notice() -> QuerySlowNotice:
        notice_started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            notice_cancelled.set()
            raise
        return QuerySlowNotice(route="show", retrieval_lane="hybrid")

    task = asyncio.create_task(
        observe_slow_query(
            _operation(),
            enabled=True,
            threshold_seconds=0.0,
            notice_factory=_notice,
            emit=lambda _message: None,
        )
    )
    while not notice_started.is_set():
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert operation_cancelled.is_set()
    assert notice_cancelled.is_set()
