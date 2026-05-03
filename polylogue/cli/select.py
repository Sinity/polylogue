"""Query-backed CLI selectors."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, cast

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.conversation.models import Conversation, ConversationSummary
from polylogue.archive.query.spec import QuerySpecError
from polylogue.cli.query_contracts import (
    QueryPlanError,
    build_query_execution_plan,
    result_date,
    result_id,
    result_provider,
    result_title,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.core.json import JSONDocument, dumps
from polylogue.pipeline.tail_overlay import TailOverlayUnavailableError, tail_overlay_services

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.protocols import ConversationQueryRuntimeStore
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class _SelectQueryStore(Protocol):
    @property
    def backend(self) -> SQLiteBackend: ...

    async def list_summaries_by_query(self, query: object) -> list[ConversationSummary]: ...


SelectPrintField = Literal["id", "title", "provider", "json"]


@dataclass(frozen=True, slots=True)
class SelectConversationRow:
    """Single conversation selector row."""

    conversation_id: str
    provider: str
    title: str
    date: str | None

    @property
    def label(self) -> str:
        date = self.date or "unknown"
        return f"{self.provider} | {date} | {self.title} | {self.conversation_id}"

    def to_json(self) -> JSONDocument:
        return {
            "id": self.conversation_id,
            "provider": self.provider,
            "title": self.title,
            "date": self.date,
        }


def select_row_from_result(result: Conversation | ConversationSummary) -> SelectConversationRow:
    date = result_date(result)
    return SelectConversationRow(
        conversation_id=result_id(result),
        provider=result_provider(result),
        title=result_title(result),
        date=date.strftime("%Y-%m-%d") if isinstance(date, datetime) else None,
    )


def render_select_row(row: SelectConversationRow, print_field: SelectPrintField) -> str:
    if print_field == "id":
        return row.conversation_id
    if print_field == "title":
        return row.title
    if print_field == "provider":
        return row.provider
    return dumps(row.to_json())


def _fzf_input(rows: list[SelectConversationRow]) -> str:
    return "\n".join(f"{row.conversation_id}\t{row.label}" for row in rows)


def _parse_fzf_output(value: str) -> str | None:
    selected = value.strip()
    if not selected:
        return None
    return selected.split("\t", 1)[0]


def _choose_with_fzf(rows: list[SelectConversationRow]) -> SelectConversationRow | None:
    if shutil.which("fzf") is None:
        return None
    try:
        completed = subprocess.run(
            ["fzf", "--delimiter", "\t", "--with-nth", "2..", "--height", "40%", "--reverse"],
            input=_fzf_input(rows),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    selected_id = _parse_fzf_output(completed.stdout)
    if selected_id is None:
        return None
    return next((row for row in rows if row.conversation_id == selected_id), None)


def choose_select_row(env: AppEnv, rows: list[SelectConversationRow]) -> SelectConversationRow | None:
    """Choose one row, using fzf/prompt only when the terminal can support it."""
    if not rows:
        return None
    if env.ui.plain or not sys.stdin.isatty() or not sys.stdout.isatty():
        return rows[0]

    fzf_row = _choose_with_fzf(rows)
    if fzf_row is not None:
        return fzf_row

    choices = [row.label for row in rows]
    selected = env.ui.choose("Select conversation", choices)
    if selected is None:
        return None
    return next((row for row in rows if row.label == selected), None)


async def _select_conversation_rows_from_store(
    config: Config,
    repo: _SelectQueryStore,
    request: RootModeRequest,
    *,
    limit: int,
) -> list[SelectConversationRow]:
    from polylogue.cli.query import _create_query_vector_provider

    request = request.with_param_updates(limit=limit, list_mode=True)
    plan = build_query_execution_plan(request.query_params())
    vector_provider = _create_query_vector_provider(config, db_path=repo.backend.db_path)
    filter_chain = plan.selection.build_filter(
        cast("ConversationQueryRuntimeStore", repo),
        vector_provider=vector_provider,
    )

    if filter_chain.can_use_summaries():
        results: list[Conversation | ConversationSummary] = list(await filter_chain.list_summaries())
    else:
        results = list(await filter_chain.list())
    return [select_row_from_result(result) for result in results]


async def select_conversation_rows(env: AppEnv, request: RootModeRequest, *, limit: int) -> list[SelectConversationRow]:
    """Return selector rows from the same query/filter path as query verbs."""
    if request.params.get("tail"):
        async with tail_overlay_services(env.services) as overlay_services:
            return await _select_conversation_rows_from_store(
                overlay_services.get_config(),
                cast("_SelectQueryStore", overlay_services.get_repository()),
                request,
                limit=limit,
            )
    return await _select_conversation_rows_from_store(
        env.config,
        cast("_SelectQueryStore", env.repository),
        request,
        limit=limit,
    )


def _raise_select_query_error(exc: QuerySpecError | QueryPlanError | TailOverlayUnavailableError) -> None:
    if isinstance(exc, QuerySpecError):
        if exc.field in {"since", "until"}:
            click.echo(f"Error: Cannot parse date: '{exc.value}'", err=True)
            click.echo(
                "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
                err=True,
            )
        else:
            click.echo(f"Error: invalid {exc.field}: '{exc.value}'", err=True)
    else:
        click.echo(f"Error: {exc}", err=True)
    raise SystemExit(1) from exc


async def async_run_select(
    env: AppEnv,
    request: RootModeRequest,
    *,
    limit: int,
    print_field: SelectPrintField,
) -> None:
    try:
        rows = await select_conversation_rows(env, request, limit=limit)
    except (QuerySpecError, QueryPlanError, TailOverlayUnavailableError) as exc:
        _raise_select_query_error(exc)
    selected = choose_select_row(env, rows)
    if selected is None:
        if rows:
            click.echo("Selection cancelled.", err=True)
            raise SystemExit(1)
        click.echo("No conversations matched.", err=True)
        raise SystemExit(2)
    click.echo(render_select_row(selected, print_field))


def run_select(
    env: AppEnv,
    request: RootModeRequest,
    *,
    limit: int,
    print_field: SelectPrintField,
) -> None:
    run_coroutine_sync(async_run_select(env, request, limit=limit, print_field=print_field))


__all__ = [
    "SelectConversationRow",
    "SelectPrintField",
    "async_run_select",
    "choose_select_row",
    "render_select_row",
    "run_select",
    "select_conversation_rows",
    "select_row_from_result",
]
