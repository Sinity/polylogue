"""Command-shape usage insight and shell-command normalization.

Design decision (``polylogue-uty2s``): this family is a *simple library
function*, not a second query language.  The concrete user question is
"which executable/subcommand shapes were actually run in this archive
window?" and the real consumer is
``ArchiveReadInsights.list_command_shape_usage``.  ``normalize_command_shapes``
keeps the evidence query small by handling shell syntax (pipelines,
``env``/``sh -c`` wrappers, and path-like arguments) before aggregation;
those transformations cannot be represented by a SQL/DSL ``GROUP BY`` over
``actions.tool_command``.  The aggregation remains read-through and has no
materializer, worker, or freshness lifecycle of its own.
"""

from __future__ import annotations

import math
import os
import shlex
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial

from polylogue.analysis.archive import PaginatedInsightQuery
from polylogue.analysis.archive_models import ArchiveInsightModel, ArchiveInsightProvenance
from polylogue.storage.sqlite.connection_profile import scratch_connection_context

COMMAND_SHAPES_INSIGHT_VERSION = 1


class CommandShapeUsage(ArchiveInsightModel):
    """Observed executions of one normalized command shape."""

    origin: str
    repository: str | None = None
    command_shape: str
    execution_count: int
    session_count: int
    last_used_at: str | None = None
    window_since: str | None = None
    window_until: str | None = None
    provenance: ArchiveInsightProvenance


class CommandShapeUsageQuery(PaginatedInsightQuery):
    origin: str | None = None
    session_id: str | None = None
    repository: str | None = None
    since: str | None = None
    until: str | None = None


def normalize_command_shapes(command: str | None) -> tuple[str, ...]:
    """Return executable/subcommand shapes, expanding shell pipelines.

    The parser is intentionally tool-agnostic: options and their values are
    omitted, while leading environment assignments and shell ``-c`` wrappers
    are transparent. A path-like positional token is also omitted so command
    arguments cannot turn into a new shape.
    """
    if not command or not command.strip():
        return ()
    try:
        tokens = list(shlex.shlex(command, posix=True, punctuation_chars="|;&"))
    except ValueError:
        return ()
    return _normalize_tokens(tokens)


def _normalize_tokens(tokens: list[str]) -> tuple[str, ...]:
    stages: list[list[str]] = [[]]
    for token in tokens:
        if token in {"|", ";", "&"}:
            stages.append([])
        else:
            stages[-1].append(token)
    shapes: list[str] = []
    for stage in stages:
        if not stage:
            continue
        # shlex emits && as two punctuation tokens.
        stage = [token for token in stage if token not in {"&&"}]
        if not stage:
            continue
        executable = os.path.basename(stage[0])
        if executable == "env":
            index = 1
            while index < len(stage) and ("=" in stage[index] or stage[index].startswith("-")):
                index += 1
            if index == len(stage):
                continue
            stage = stage[index:]
            executable = os.path.basename(stage[0])
        while stage and _assignment(stage[0]):
            stage.pop(0)
        if not stage:
            continue
        executable = os.path.basename(stage[0])
        if executable in {"sh", "bash", "zsh", "dash", "ksh", "fish"}:
            try:
                command_index = next(
                    i
                    for i, token in enumerate(stage[1:], 1)
                    if token == "-c" or (token.endswith("c") and token.startswith("-"))
                )
            except StopIteration:
                continue
            if command_index + 1 < len(stage):
                nested = shlex.shlex(stage[command_index + 1], posix=True, punctuation_chars="|;&")
                shapes.extend(_normalize_tokens(list(nested)))
            continue
        words = [executable]
        for token in stage[1:]:
            if token.startswith("-"):
                break
            if _path_like(token):
                continue
            words.append(token)
        shapes.append(" ".join(words))
    return tuple(shapes)


def _assignment(token: str) -> bool:
    name, separator, _value = token.partition("=")
    return bool(separator) and bool(name) and name.replace("_", "a").isalnum() and not name[0].isdigit()


def _path_like(token: str) -> bool:
    return token in {".", ".."} or token.startswith(("/", "./", "../", "~/")) or "/" in token


@contextmanager
def _command_shape_scratch() -> Iterator[tuple[sqlite3.Connection, sqlite3.Cursor]]:
    """Own one disposable fold and settle every resource before returning.

    The scratch connection and its directory belong to the canonical custody
    owner; this fold settles its own cursor and progress handler first.
    """
    with scratch_connection_context(prefix="polylogue-command-shapes-", filename="usage.db") as scratch:
        cursor: sqlite3.Cursor | None = None
        primary: BaseException | None = None
        try:
            cursor = scratch.cursor()
            yield scratch, cursor
        except BaseException as exc:
            primary = exc
            raise
        finally:
            faults: list[BaseException] = []
            if primary is not None:
                faults.append(primary)
            cleanup: list[Callable[[], object]] = [partial(scratch.set_progress_handler, None, 0)]
            if cursor is not None:
                cleanup.append(cursor.close)
            for finish in cleanup:
                try:
                    finish()
                except BaseException as exc:
                    if all(exc is not fault for fault in faults):
                        faults.append(exc)
            if faults and (primary is None or len(faults) > 1):
                if len(faults) == 1:
                    raise faults[0]
                raise BaseExceptionGroup("command-shape fold and cleanup failed", faults) from None


def build_command_shape_usage(
    rows: Iterable[Mapping[str, object]],
    query: CommandShapeUsageQuery,
    *,
    materialized_at: str,
    checkpoint: Callable[[], None] = lambda: None,
) -> list[CommandShapeUsage]:
    """Stream normalized executions to disk, then page their exact aggregate."""
    checkpoint()
    with _command_shape_scratch() as (scratch, cursor):
        cursor.execute("PRAGMA temp_store = FILE")
        cursor.execute("PRAGMA cache_size = -2048")
        cursor.execute(
            "CREATE TABLE usage (origin TEXT NOT NULL, repository TEXT NOT NULL, shape TEXT NOT NULL, "
            "session_id TEXT NOT NULL, executions INTEGER NOT NULL, last_ms REAL, "
            "PRIMARY KEY(origin, repository, shape, session_id)) WITHOUT ROWID, STRICT"
        )
        cancelled: BaseException | None = None

        def progress() -> int:
            nonlocal cancelled
            try:
                checkpoint()
            except BaseException as exc:
                cancelled = exc
                return 1
            return 0

        scratch.set_progress_handler(progress, 1000)

        def normalized_rows() -> Iterator[tuple[str, str, str, str, float | None]]:
            for row in rows:
                checkpoint()
                timestamp = row.get("occurred_at_ms")
                last_ms = (
                    float(timestamp)
                    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
                    else None
                )
                if last_ms is not None and not math.isfinite(last_ms):
                    raise ValueError("command-shape timestamp must be finite")
                for shape in normalize_command_shapes(_text(row.get("tool_command"))):
                    checkpoint()
                    yield (
                        str(row["origin"]),
                        _optional_text(row.get("repository")) or "",
                        shape,
                        str(row["session_id"]),
                        last_ms,
                    )

        try:
            # One disposable transaction; repeated stages retain multiplicity,
            # while each session contributes once to a shape's session count.
            cursor.execute("BEGIN")
            cursor.executemany(
                "INSERT INTO usage VALUES (?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(origin, repository, shape, session_id) DO UPDATE SET "
                "executions = executions + 1, "
                "last_ms = CASE WHEN excluded.last_ms IS NULL THEN last_ms "
                "WHEN last_ms IS NULL THEN excluded.last_ms ELSE MAX(last_ms, excluded.last_ms) END",
                normalized_rows(),
            )
            cursor.execute("SELECT COUNT(*) FROM (SELECT 1 FROM usage GROUP BY origin, repository, shape)")
            total = int(cursor.fetchone()[0])
            stop = query.offset + query.limit if query.limit is not None else None
            start, end, _step = slice(query.offset, stop).indices(total)
            cursor.execute(
                "SELECT origin, repository, shape, SUM(executions), COUNT(*), MAX(last_ms) FROM usage "
                "GROUP BY origin, repository, shape "
                "ORDER BY SUM(executions) DESC, shape, origin, repository LIMIT ? OFFSET ?",
                (max(0, end - start), start),
            )
            result: list[CommandShapeUsage] = []
            for origin, repository, shape, executions, sessions, last_ms in cursor:
                checkpoint()
                result.append(
                    CommandShapeUsage(
                        origin=origin,
                        repository=repository or None,
                        command_shape=shape,
                        execution_count=executions,
                        session_count=sessions,
                        last_used_at=_iso_ms(last_ms),
                        window_since=query.since,
                        window_until=query.until,
                        provenance=ArchiveInsightProvenance(
                            materializer_version=COMMAND_SHAPES_INSIGHT_VERSION,
                            materialized_at=materialized_at,
                            source_updated_at=_iso_ms(last_ms),
                            source_sort_key=float(last_ms) / 1000 if last_ms is not None else None,
                        ),
                    )
                )
            return result
        except sqlite3.Error as exc:
            if cancelled is not None and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                raise cancelled from None
            raise


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_text(value: object) -> str | None:
    value = _text(value)
    return value or None


def _iso_ms(value: object) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    from datetime import UTC, datetime

    return datetime.fromtimestamp(float(value) / 1000, UTC).isoformat()


__all__ = ["CommandShapeUsage", "CommandShapeUsageQuery", "build_command_shape_usage", "normalize_command_shapes"]
