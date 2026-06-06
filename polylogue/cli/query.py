"""Query execution entrypoint for the query-first CLI (only)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.query_contracts import QueryExecutionPlan, QueryParams
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.archive.models import Session
    from polylogue.config import Config
    from polylogue.protocols import VectorProvider


def handle_query_mode(
    ctx: click.Context,
    *,
    show_stats: object | None = None,
) -> None:
    """Handle query mode: run the archive query executor."""
    env: AppEnv = ctx.obj
    request = RootModeRequest.from_context(ctx)
    execute_query_request(env, request)


# Re-exported from the lightweight query_group module so that
# click_app can import the base class without pulling in the full
# archive/storage/operations import chain.
from polylogue.cli.query_group import QueryFirstGroupBase  # noqa: E402

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def project_query_results(results: list[Session], plan: QueryExecutionPlan) -> list[Session]:
    """Apply post-selection transforms consistently before final output."""
    from polylogue.cli import query_actions as _query_actions

    projected = results
    if plan.output.transform is not None:
        projected = _query_actions.apply_transform(projected, plan.output.transform)
    message_roles = plan.output.effective_message_roles()
    if message_roles:
        projected = [session.with_roles(message_roles) for session in projected]
    if plan.output.filters_content():
        projected = [session.with_content_projection(plan.output.content_projection) for session in projected]
    return projected


def _create_query_vector_provider(config: Config, *, db_path: Path | None = None) -> VectorProvider | None:
    """Best-effort vector provider setup for query execution."""
    from polylogue.storage.search_providers import create_vector_provider

    try:
        return create_vector_provider(config, db_path=db_path or config.db_path)
    except (ValueError, ImportError):
        return None
    except Exception as exc:
        logger.warning("Vector search setup failed: %s", exc, exc_info=True)
        return None


def execute_query_request(env: AppEnv, request: RootModeRequest) -> None:
    """Execute a typed root-mode request (path)."""
    run_coroutine_sync(async_execute_query_request(env, request))


def execute_query(env: AppEnv, params: QueryParams) -> None:
    """Execute a query-mode command from raw params."""
    execute_query_request(env, RootModeRequest.from_params(params))


async def async_execute_query(env: AppEnv, params: QueryParams) -> None:
    """Async compatibility wrapper for raw param execution."""
    await async_execute_query_request(env, RootModeRequest.from_params(params))


async def async_execute_query_request(env: AppEnv, request: RootModeRequest) -> None:
    """Async core of query execution."""
    from polylogue.cli.archive_query import execute_archive_query

    execute_archive_query(env, request)


__all__ = [
    "QueryFirstGroupBase",
    "async_execute_query",
    "async_execute_query_request",
    "execute_query",
    "execute_query_request",
    "handle_query_mode",
    "project_query_results",
]
