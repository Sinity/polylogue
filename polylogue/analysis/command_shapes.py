"""Command-shape usage insight and shell-command normalization."""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence

from typing_extensions import TypedDict

from polylogue.analysis.archive import PaginatedInsightQuery
from polylogue.analysis.archive_models import ArchiveInsightModel, ArchiveInsightProvenance

COMMAND_SHAPES_INSIGHT_VERSION = 1


class CommandShapeUsage(ArchiveInsightModel):
    """Observed executions of one normalized command shape."""

    origin: str
    repository: str | None = None
    command_shape: str
    execution_count: int
    session_count: int
    last_used_at: str | None = None
    last_used_sort_key: float | None = None
    window_since: str | None = None
    window_until: str | None = None
    provenance: ArchiveInsightProvenance


class CommandShapeUsageQuery(PaginatedInsightQuery):
    origin: str | None = None
    session_id: str | None = None
    repository: str | None = None
    since: str | None = None
    until: str | None = None


class _Aggregate(TypedDict):
    count: int
    last_ms: float | None
    sessions: set[str]


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


def build_command_shape_usage(
    rows: Sequence[dict[str, object]],
    query: CommandShapeUsageQuery,
    *,
    materialized_at: str,
) -> list[CommandShapeUsage]:
    """Aggregate normalized action rows into stable public insight rows."""
    grouped: dict[tuple[str, str | None, str], _Aggregate] = {}
    for row in rows:
        for shape in normalize_command_shapes(_text(row.get("tool_command"))):
            key = (str(row["origin"]), _optional_text(row.get("repository")), shape)
            item = grouped.setdefault(key, {"count": 0, "last_ms": None, "sessions": set()})
            item["count"] = int(item["count"]) + 1
            item["sessions"].add(str(row["session_id"]))
            timestamp = row.get("occurred_at_ms")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                item["last_ms"] = max(item["last_ms"] or float(timestamp), float(timestamp))
    result: list[CommandShapeUsage] = []
    for (origin, repository, shape), item in grouped.items():
        last_ms = item["last_ms"]
        result.append(
            CommandShapeUsage(
                origin=origin,
                repository=repository,
                command_shape=shape,
                execution_count=int(item["count"]),
                session_count=len(item["sessions"]),
                last_used_at=_iso_ms(last_ms),
                last_used_sort_key=float(last_ms) / 1000 if last_ms is not None else None,
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
    result.sort(key=lambda item: (-item.execution_count, item.command_shape, item.origin, item.repository or ""))
    start = query.offset
    return result[start : start + query.limit] if query.limit is not None else result[start:]


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
