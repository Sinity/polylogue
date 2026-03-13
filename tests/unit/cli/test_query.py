"""Focused entrypoint tests for CLI query execution.

Formatting, output, routing, and mutation contracts live in:
- test_query_fmt.py
- test_query_exec.py
- test_query_exec_laws.py
- test_query_plan.py

This file keeps only the direct entrypoint and error-handling seams that
are not worth expressing through the broader contract suites.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import ConfigError


def _make_env() -> MagicMock:
    env = MagicMock()
    env.repository = MagicMock()
    return env


def test_execute_query_runs_async_core() -> None:
    from polylogue.cli.query import execute_query

    env = _make_env()
    params = {"query": ("hello",)}

    with (
        patch("polylogue.cli.query.async_execute_query", new_callable=AsyncMock) as mock_async_execute,
        patch("asyncio.run") as mock_asyncio_run,
    ):
        execute_query(env, params)

    mock_async_execute.assert_called_once_with(env, params)
    mock_asyncio_run.assert_called_once()
    dispatched = mock_asyncio_run.call_args.args[0]
    assert asyncio.iscoroutine(dispatched)
    dispatched.close()


@pytest.mark.parametrize("exc_type", [ValueError, ImportError])
def test_create_query_vector_provider_swallows_expected_setup_errors(exc_type: type[Exception]) -> None:
    from polylogue.cli.query import _create_query_vector_provider

    with patch(
        "polylogue.storage.search_providers.create_vector_provider",
        side_effect=exc_type("vector unavailable"),
    ):
        assert _create_query_vector_provider(MagicMock()) is None


def test_create_query_vector_provider_logs_unexpected_failure() -> None:
    from polylogue.cli.query import _create_query_vector_provider

    with (
        patch(
            "polylogue.storage.search_providers.create_vector_provider",
            side_effect=RuntimeError("boom"),
        ),
        patch("polylogue.cli.query.logger.warning") as mock_warning,
    ):
        assert _create_query_vector_provider(MagicMock()) is None

    mock_warning.assert_called_once()
    assert mock_warning.call_args.args[0] == "Vector search setup failed: %s"


def test_async_execute_query_fails_on_config_error() -> None:
    from polylogue.cli.query import async_execute_query

    env = _make_env()

    with (
        patch("polylogue.cli.helpers.load_effective_config", side_effect=ConfigError("bad config")),
        patch("polylogue.cli.helpers.fail", side_effect=SystemExit("query: bad config")) as mock_fail,
        pytest.raises(SystemExit, match="query: bad config"),
    ):
        asyncio.run(async_execute_query(env, {}))

    mock_fail.assert_called_once_with("query", "bad config")


def test_async_execute_query_reports_query_plan_error() -> None:
    from polylogue.cli.query import async_execute_query
    from polylogue.cli.query_plan import QueryPlanError

    env = _make_env()

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.cli.query._create_query_vector_provider", return_value=None),
        patch(
            "polylogue.cli.query.build_query_execution_plan",
            side_effect=QueryPlanError("bad query plan"),
        ),
        patch("click.echo") as mock_echo,
        pytest.raises(SystemExit) as exc_info,
    ):
        asyncio.run(async_execute_query(env, {}))

    assert exc_info.value.code == 1
    mock_echo.assert_called_once_with("Error: bad query plan", err=True)


def test_async_execute_query_passes_vector_provider_into_filter_build() -> None:
    from polylogue.cli.query import async_execute_query
    from polylogue.cli.query_plan import QueryAction, QueryExecutionPlan, QueryMutationSpec, QueryOutputSpec
    from polylogue.lib.query_spec import ConversationQuerySpec

    env = _make_env()
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = False
    filter_chain.list = AsyncMock(return_value=[])
    selection = MagicMock(spec=ConversationQuerySpec)
    selection.build_filter.return_value = filter_chain
    vector_provider = object()
    plan = QueryExecutionPlan(
        selection=selection,
        action=QueryAction.SHOW,
        output=QueryOutputSpec("markdown", ("stdout",), None, False, None, False),
        mutation=QueryMutationSpec((), (), False, False, False),
    )

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.cli.query._create_query_vector_provider", return_value=vector_provider),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query_output.output_results") as mock_output_results,
    ):
        asyncio.run(async_execute_query(env, {}))

    selection.build_filter.assert_called_once_with(env.repository, vector_provider=vector_provider)
    mock_output_results.assert_called_once_with(env, [], {})
