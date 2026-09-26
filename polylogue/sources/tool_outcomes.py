"""Resolve canonical tool outcomes from parser evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from polylogue.core.enums import BlockType, Origin, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.origin_specs import tool_outcome_unknown_reasons_for_origin
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSessionEvent
from polylogue.sources.prepared_message_sink import SqliteMessageSink


class _OutcomeIndex:
    """Per-call, indexed evidence with fixed-size Python state."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.executescript(
            """
            CREATE TABLE sidecar (
                tool_id TEXT NOT NULL,
                owner_present INTEGER NOT NULL,
                owner TEXT NOT NULL,
                outcome TEXT NOT NULL,
                exit_code INTEGER,
                PRIMARY KEY (tool_id, owner_present, owner)
            ) WITHOUT ROWID;
            CREATE TABLE sidecar_any (tool_id TEXT PRIMARY KEY, outcome TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE result_state (
                tool_id TEXT PRIMARY KEY,
                result_count INTEGER NOT NULL,
                use_count INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE results (
                tool_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                PRIMARY KEY (tool_id, ordinal)
            ) WITHOUT ROWID;
            """
        )

    @staticmethod
    def _owner_key(owner: str | None) -> tuple[int, str]:
        return (0, "") if owner is None else (1, owner)

    def add_sidecar(
        self, tool_id: str, owner: str | None, outcome: ToolOutcome, exit_code: int | None, *, origin: Origin
    ) -> None:
        prior = self.conn.execute("SELECT outcome FROM sidecar_any WHERE tool_id = ?", (tool_id,)).fetchone()
        if prior is not None and prior[0] != outcome.value:
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"conflicting execution evidence for tool_id={tool_id!r}"
            )
        owner_present, owner_value = self._owner_key(owner)
        prior = self.conn.execute(
            "SELECT outcome FROM sidecar WHERE tool_id = ? AND owner_present = ? AND owner = ?",
            (tool_id, owner_present, owner_value),
        ).fetchone()
        if prior is not None and prior[0] != outcome.value:
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"conflicting execution evidence for tool_id={tool_id!r}"
            )
        self.conn.execute("INSERT OR REPLACE INTO sidecar_any VALUES (?, ?)", (tool_id, outcome.value))
        self.conn.execute(
            """INSERT INTO sidecar VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (tool_id, owner_present, owner) DO UPDATE SET
                outcome = excluded.outcome,
                exit_code = COALESCE(excluded.exit_code, sidecar.exit_code)""",
            (tool_id, owner_present, owner_value, outcome.value, exit_code),
        )

    def _sidecar_value(self, tool_id: str | None, owner: str | None, column: str) -> str | int | None:
        if tool_id is None:
            return None
        owner_present, owner_value = self._owner_key(owner)
        row = self.conn.execute(
            f"SELECT {column} FROM sidecar WHERE tool_id = ? AND owner_present = ? AND owner = ?",
            (tool_id, owner_present, owner_value),
        ).fetchone()
        if row is not None and row[0] is not None:
            return cast(str | int | None, row[0])
        row = self.conn.execute(
            f"SELECT {column} FROM sidecar WHERE tool_id = ? AND owner_present = 0 AND owner = ''",
            (tool_id,),
        ).fetchone()
        return cast(str | int | None, row[0]) if row is not None else None

    def sidecar_outcome(self, tool_id: str | None, owner: str | None) -> ToolOutcome | None:
        value = self._sidecar_value(tool_id, owner, "outcome")
        return ToolOutcome(cast(str, value)) if value is not None else None

    def sidecar_exit(self, tool_id: str | None, owner: str | None) -> int | None:
        value = self._sidecar_value(tool_id, owner, "exit_code")
        return int(value) if value is not None else None

    def any_sidecar_outcome(self, tool_id: str) -> ToolOutcome | None:
        row = self.conn.execute("SELECT outcome FROM sidecar_any WHERE tool_id = ?", (tool_id,)).fetchone()
        return ToolOutcome(row[0]) if row is not None else None

    def add_result(self, tool_id: str, outcome: ToolOutcome) -> None:
        row = self.conn.execute("SELECT result_count FROM result_state WHERE tool_id = ?", (tool_id,)).fetchone()
        ordinal = row[0] if row is not None else 0
        if row is None:
            self.conn.execute("INSERT INTO result_state VALUES (?, 1, 0)", (tool_id,))
        else:
            self.conn.execute("UPDATE result_state SET result_count = result_count + 1 WHERE tool_id = ?", (tool_id,))
        self.conn.execute("INSERT INTO results VALUES (?, ?, ?)", (tool_id, ordinal, outcome.value))

    def next_result(self, tool_id: str) -> ToolOutcome | None:
        row = self.conn.execute(
            "SELECT result_count, use_count FROM result_state WHERE tool_id = ?", (tool_id,)
        ).fetchone()
        if row is None or row[1] >= row[0]:
            return None
        outcome = self.conn.execute(
            "SELECT outcome FROM results WHERE tool_id = ? AND ordinal = ?", (tool_id, row[1])
        ).fetchone()[0]
        self.conn.execute("UPDATE result_state SET use_count = use_count + 1 WHERE tool_id = ?", (tool_id,))
        return ToolOutcome(outcome)


@contextmanager
def _outcome_index(messages: Sequence[ParsedMessage]) -> Iterator[_OutcomeIndex]:
    if isinstance(messages, SqliteMessageSink):
        with (
            TemporaryDirectory(prefix="tool-outcomes-", dir=Path(messages.path).parent) as scratch,
            closing(sqlite3.connect(Path(scratch) / "evidence.sqlite3")) as conn,
        ):
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA cache_size = -1024")
            yield _OutcomeIndex(conn)
    else:
        with closing(sqlite3.connect(":memory:")) as conn:
            yield _OutcomeIndex(conn)


def derive_tool_outcomes(
    messages: list[ParsedMessage] | SqliteMessageSink,
    events: Sequence[ParsedSessionEvent],
    *,
    origin: Origin,
) -> list[ParsedMessage] | SqliteMessageSink:
    """Resolve tool outcomes from each origin's structured parser evidence.

    ``is_error``, exit codes and ``outcome_unknown_reason`` are all
    parser-normalized evidence; this function adds no meaning of its own and
    never invents a reason an origin's parser did not derive. Claude Code
    additionally carries outcome fields in its record-level execution event.
    A result without any such evidence is a parser defect and refuses the
    write. A tool-use without a paired result is a recorded interruption and
    receives the distinct, known ``no_result`` outcome.
    """
    with _outcome_index(messages) as index:
        _index_sidecars(index, events, origin=origin)
        _index_results(index, messages, origin=origin)
        if isinstance(messages, SqliteMessageSink):
            with messages.atomic_edit():
                for ordinal in range(len(messages)):
                    messages[ordinal] = _normalize_message(index, messages[ordinal], origin=origin)
            return messages
        return [_normalize_message(index, message, origin=origin) for message in messages]


def _index_sidecars(index: _OutcomeIndex, events: Sequence[ParsedSessionEvent], *, origin: Origin) -> None:
    # The record id scopes result evidence. Its tool-id-wide view applies only
    # to an unmatched use; an unowned sidecar applies to every matching result.
    for event in events:
        if origin is not Origin.CLAUDE_CODE_SESSION or event.event_type != "claude_tool_execution_result":
            continue
        tool_id = event.payload.get("tool_use_id")
        if not isinstance(tool_id, str):
            continue
        payload = event.payload
        raw_error = payload.get("is_error", payload.get("isError"))
        raw_exit = payload.get("exit_code", payload.get("exitCode"))
        event_candidates: list[ToolOutcome] = []
        if isinstance(raw_error, bool):
            event_candidates.append(ToolOutcome.ERROR if raw_error else ToolOutcome.OK)
        exit_code = raw_exit if isinstance(raw_exit, int) and not isinstance(raw_exit, bool) else None
        if exit_code is not None:
            event_candidates.append(ToolOutcome.ERROR if exit_code else ToolOutcome.OK)
        if not event_candidates:
            continue
        if len(set(event_candidates)) > 1:
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"conflicting execution evidence for tool_id={tool_id!r}"
            )
        index.add_sidecar(tool_id, event.source_message_provider_id, event_candidates[0], exit_code, origin=origin)


def _index_results(index: _OutcomeIndex, messages: Sequence[ParsedMessage], *, origin: Origin) -> None:
    for message in messages:
        for block in message.blocks:
            if block.type is not BlockType.TOOL_RESULT or not block.tool_id:
                continue
            result_candidates = _result_candidates(
                block, index, owner=message.provider_message_id, unknown_reason=block.outcome_unknown_reason
            )
            distinct = set(result_candidates)
            if ToolOutcome.NO_RESULT in distinct or len(distinct) > 1:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"conflicting result evidence for tool_id={block.tool_id!r}"
                )
            outcome = next(iter(distinct), None)
            if outcome is None:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"unsupported tool_result block shape tool_id={block.tool_id!r}"
                )
            index.add_result(block.tool_id, outcome)


def _normalize_message(index: _OutcomeIndex, message: ParsedMessage, *, origin: Origin) -> ParsedMessage:
    blocks: list[ParsedContentBlock] = []
    for block in message.blocks:
        if block.type is BlockType.TOOL_RESULT:
            unknown_reason = (
                None
                if _sidecar_resolves_not_reported(block, index, owner=message.provider_message_id)
                else block.outcome_unknown_reason
            )
            _require_declared_reason(unknown_reason, origin=origin, tool_id=block.tool_id)
            resolved_candidates = _result_candidates(
                block, index, owner=message.provider_message_id, unknown_reason=unknown_reason
            )
            if ToolOutcome.NO_RESULT in resolved_candidates:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"tool_result cannot be no_result for tool_id={block.tool_id!r}"
                )
            distinct = set(resolved_candidates)
            if len(distinct) > 1:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"conflicting result evidence for tool_id={block.tool_id!r}"
                )
            outcome = next(iter(distinct), None)
            if outcome is None:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"unsupported tool_result block shape tool_id={block.tool_id!r}"
                )
            if outcome is ToolOutcome.UNKNOWN and unknown_reason is None:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"unknown outcome without a structural reason for tool_id={block.tool_id!r}"
                )
            if outcome is not ToolOutcome.UNKNOWN and unknown_reason is not None:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"known outcome carries unknown reason for tool_id={block.tool_id!r}"
                )
            exit_code = block.exit_code
            if exit_code is None:
                exit_code = index.sidecar_exit(block.tool_id, message.provider_message_id)
            is_error = None if outcome is ToolOutcome.UNKNOWN else outcome is ToolOutcome.ERROR
            blocks.append(
                block.model_copy(
                    update={
                        "tool_outcome": outcome,
                        "is_error": is_error,
                        "exit_code": exit_code,
                        "outcome_unknown_reason": unknown_reason if outcome is ToolOutcome.UNKNOWN else None,
                    }
                )
            )
        elif block.type is BlockType.TOOL_USE:
            outcome = index.next_result(block.tool_id) if block.tool_id else None
            if outcome is None and block.tool_id:
                outcome = index.any_sidecar_outcome(block.tool_id)
            blocks.append(block.model_copy(update={"tool_outcome": outcome or ToolOutcome.NO_RESULT}))
        else:
            blocks.append(block)
    return message.model_copy(update={"blocks": blocks})


def _result_candidates(
    block: ParsedContentBlock,
    index: _OutcomeIndex,
    *,
    owner: str | None,
    unknown_reason: str | None,
) -> list[ToolOutcome]:
    candidates: list[ToolOutcome] = []
    if block.tool_outcome is not None:
        candidates.append(block.tool_outcome)
    if isinstance(block.is_error, bool):
        candidates.append(ToolOutcome.ERROR if block.is_error else ToolOutcome.OK)
    if isinstance(block.exit_code, int) and not isinstance(block.exit_code, bool):
        candidates.append(ToolOutcome.ERROR if block.exit_code else ToolOutcome.OK)
    sidecar_outcome = index.sidecar_outcome(block.tool_id, owner)
    if sidecar_outcome is not None:
        candidates.append(sidecar_outcome)
    sidecar_exit = index.sidecar_exit(block.tool_id, owner)
    if sidecar_exit is not None and not isinstance(block.exit_code, int):
        candidates.append(ToolOutcome.ERROR if sidecar_exit else ToolOutcome.OK)
    if unknown_reason is not None and not any(
        candidate in (ToolOutcome.OK, ToolOutcome.ERROR) for candidate in candidates
    ):
        candidates.append(ToolOutcome.UNKNOWN)
    return candidates


def _sidecar_resolves_not_reported(block: ParsedContentBlock, index: _OutcomeIndex, *, owner: str | None) -> bool:
    """Return whether matching Claude evidence supersedes an absent inline verdict.

    ``content[].tool_result`` is lowered before Claude Code's record-level
    ``toolUseResult`` sidecar is appended as a session event. That ordering
    legitimately leaves a parser-derived ``NOT_REPORTED`` reason on a block
    whose same-call sidecar later proves a known outcome. Only that narrow
    shape is stale: explicit block verdicts and stronger unknown reasons stay
    authoritative and are handled by the existing conflict checks.
    """
    return (
        block.tool_id is not None
        and index.sidecar_outcome(block.tool_id, owner) is not None
        and block.tool_outcome is None
        and block.is_error is None
        and block.exit_code is None
        and block.outcome_unknown_reason == ToolResultUnknownReason.NOT_REPORTED.value
    )


def _require_declared_reason(reason: str | None, *, origin: Origin, tool_id: str | None) -> None:
    """Refuse a reason the origin's parsers are not declared to derive.

    A reason with no owner is one nothing in this origin's record structures
    can account for -- the shape this contract exists to keep out of the
    archive. The refusal is per session and clears once the origin declares
    the construct its parser now reads.
    """
    if reason is None:
        return
    declared = tool_outcome_unknown_reasons_for_origin(origin)
    if ToolResultUnknownReason(reason) not in declared:
        raise ValueError(
            f"tool outcome derivation refused for origin {origin.value!r}: "
            f"undeclared unknown reason {reason!r} for tool_id={tool_id!r}"
        )
