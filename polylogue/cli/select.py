"""Query-backed CLI selectors."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Literal, NoReturn

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import QuerySpecError
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.cli.query_contracts import (
    result_date,
    result_id,
    result_origin,
    result_title,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.core.json import JSONDocument, dumps

if TYPE_CHECKING:
    from polylogue.config import Config


SelectPrintField = Literal["id", "title", "origin", "json"]


@dataclass(frozen=True, slots=True)
class SelectSessionRow:
    """Single session selector row."""

    session_id: str
    origin: str
    title: str
    date: str | None

    @property
    def label(self) -> str:
        date = self.date or "unknown"
        return f"{self.origin} | {date} | {self.title} | {self.session_id}"

    @property
    def preview(self) -> str:
        date = self.date or "unknown"
        return f"Origin: {self.origin}  Date: {date}  Title: {self.title}  ID: {self.session_id}"

    def to_json(self) -> JSONDocument:
        return {
            "id": self.session_id,
            "origin": self.origin,
            "title": self.title,
            "date": self.date,
        }


def select_row_from_result(result: Session | SessionSummary) -> SelectSessionRow:
    date = result_date(result)
    return SelectSessionRow(
        session_id=result_id(result),
        origin=result_origin(result),
        title=result_title(result),
        date=date.strftime("%Y-%m-%d") if isinstance(date, datetime) else None,
    )


def render_select_row(row: SelectSessionRow, print_field: SelectPrintField) -> str:
    if print_field == "id":
        return row.session_id
    if print_field == "title":
        return row.title
    if print_field == "origin":
        return row.origin
    return dumps(row.to_json())


def render_select_rows(rows: list[SelectSessionRow], print_field: SelectPrintField) -> str:
    """Render one or more selector rows for noninteractive output."""
    return "\n".join(render_select_row(row, print_field) for row in rows)


def _fzf_input(rows: list[SelectSessionRow]) -> str:
    return "\n".join(f"{row.session_id}\t{row.label}\t{row.preview}" for row in rows)


def _parse_fzf_output(value: str) -> str | None:
    selected = value.strip()
    if not selected:
        return None
    return selected.split("\t", 1)[0]


def _choose_with_fzf(rows: list[SelectSessionRow]) -> SelectSessionRow | None:
    if shutil.which("fzf") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "fzf",
                "--delimiter",
                "\t",
                "--with-nth",
                "2",
                "--height",
                "40%",
                "--reverse",
                "--preview",
                "echo {} | cut -f3-",
                "--preview-window",
                "down:4:wrap",
            ],
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
    return next((row for row in rows if row.session_id == selected_id), None)


def choose_select_row(env: AppEnv, rows: list[SelectSessionRow]) -> SelectSessionRow | None:
    """Choose one row, using fzf/prompt only when the terminal can support it."""
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    if env.ui.plain or not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    fzf_row = _choose_with_fzf(rows)
    if fzf_row is not None:
        return fzf_row

    choices = [row.label for row in rows]
    selected = env.ui.choose("Select session", choices)
    if selected is None:
        return None
    return next((row for row in rows if row.label == selected), None)


async def _select_session_rows_from_store(
    config: Config,
    request: RootModeRequest,
    *,
    limit: int,
) -> list[SelectSessionRow]:
    from polylogue.cli.query import _create_query_vector_provider
    from polylogue.paths import archive_file_set_root_for_paths

    spec = replace(request.query_spec(), limit=limit)
    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
    vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
    filter_chain = spec.build_filter(config, vector_provider=vector_provider)

    if filter_chain.can_use_summaries():
        results: list[Session | SessionSummary] = list(await filter_chain.list_summaries())
    else:
        results = list(await filter_chain.list())
    return [select_row_from_result(result) for result in results]


async def select_session_rows(env: AppEnv, request: RootModeRequest, *, limit: int) -> list[SelectSessionRow]:
    """Return selector rows from the same query/filter path as query verbs."""
    return await _select_session_rows_from_store(
        env.config,
        request,
        limit=limit,
    )


def _raise_select_query_error(exc: QuerySpecError) -> NoReturn:
    if isinstance(exc, QuerySpecError):
        if exc.field in {"since", "until"}:
            click.echo(f"Error: Cannot parse date: '{exc.value}'", err=True)
            click.echo(
                "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
                err=True,
            )
        else:
            click.echo(f"Error: invalid {exc.field}: '{exc.value}'", err=True)
    raise SystemExit(1) from exc


async def async_run_select(
    env: AppEnv,
    request: RootModeRequest,
    *,
    limit: int,
    print_field: SelectPrintField,
) -> None:
    try:
        rows = await select_session_rows(env, request, limit=limit)
    except QuerySpecError as exc:
        _raise_select_query_error(exc)
    selected = choose_select_row(env, rows)
    if selected is None:
        if rows:
            click.echo(render_select_rows(rows, print_field))
            return
        click.echo("No sessions matched.", err=True)
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
    "SelectSessionRow",
    "SelectPrintField",
    "async_run_select",
    "choose_select_row",
    "render_select_row",
    "render_select_rows",
    "run_select",
    "select_session_rows",
    "select_row_from_result",
]
