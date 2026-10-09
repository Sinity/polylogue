"""Resolve canonical tool outcomes from parser evidence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from polylogue.core.enums import BlockType, Origin, ToolOutcome, ToolResultUnknownReason
from polylogue.core.tool_association import tool_association_ctes_sql
from polylogue.pipeline.ids import block_content_identity
from polylogue.sources.origin_specs import tool_outcome_unknown_reasons_for_origin
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSessionEvent
from polylogue.sources.prepared_message_sink import SqliteMessageSink


def _exact_key(value: str | None) -> bytes | None:
    """Bind a provider identifier losslessly, including lone surrogates.

    Provider IDs are exact facts; a lone surrogate cannot be encoded as SQLite
    TEXT, so every identifier in this private index is a ``surrogatepass``
    UTF-8 BLOB. Equality and grouping see the same exact value on both sides.
    """
    return None if value is None else value.encode("utf-8", "surrogatepass")


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
            CREATE INDEX sidecar_tool_outcome ON sidecar (tool_id, outcome);
            CREATE TABLE original_messages (
                ordinal INTEGER PRIMARY KEY,native_id TEXT,normalized_id TEXT,parent_native_id TEXT,
                parent_position INTEGER,declared_position INTEGER,role TEXT,position INTEGER,variant INTEGER
            );
            CREATE INDEX original_messages_native ON original_messages(native_id);
            CREATE INDEX original_messages_normalized ON original_messages(normalized_id);
            CREATE INDEX original_messages_position ON original_messages(declared_position);
            CREATE TABLE tool_messages (ordinal INTEGER PRIMARY KEY);
            CREATE TABLE association_blocks (
                session_key INTEGER NOT NULL,block_key TEXT PRIMARY KEY,message_key INTEGER NOT NULL,
                tool_id TEXT,is_use INTEGER,outcome TEXT,unknown_reason TEXT,block_position INTEGER
            );
            """
        )

    @staticmethod
    def _owner_key(owner: str | None) -> tuple[int, bytes]:
        return (0, b"") if owner is None else (1, owner.encode("utf-8", "surrogatepass"))

    def add_sidecar(
        self, tool_id: str, owner: str | None, outcome: ToolOutcome, exit_code: int | None, *, origin: Origin
    ) -> None:
        owner_present, owner_value = self._owner_key(owner)
        tool_key = _exact_key(tool_id)
        conflicts_with_ownerless = False
        if owner is None:
            # An ownerless record applies to every matching result. Check the
            # endpoints of the owned outcomes so distinct owned records remain
            # independent while an ownerless contradiction still refuses.
            for direction in ("ASC", "DESC"):
                with closing(
                    self.conn.execute(
                        f"""SELECT outcome FROM sidecar
                        WHERE tool_id = ? AND owner_present = 1
                        ORDER BY outcome {direction} LIMIT 1""",
                        (tool_key,),
                    )
                ) as rows:
                    prior_owned = rows.fetchone()
                if prior_owned is not None and prior_owned[0] != outcome.value:
                    conflicts_with_ownerless = True
                    break
        else:
            prior_ownerless = self.conn.execute(
                "SELECT outcome FROM sidecar WHERE tool_id = ? AND owner_present = 0 AND owner = x''",
                (tool_key,),
            ).fetchone()
            conflicts_with_ownerless = prior_ownerless is not None and prior_ownerless[0] != outcome.value
        prior = self.conn.execute(
            "SELECT outcome FROM sidecar WHERE tool_id = ? AND owner_present = ? AND owner = ?",
            (tool_key, owner_present, owner_value),
        ).fetchone()
        if conflicts_with_ownerless or (prior is not None and prior[0] != outcome.value):
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"conflicting execution evidence for tool_id={tool_id!r}"
            )
        self.conn.execute(
            """INSERT INTO sidecar VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (tool_id, owner_present, owner) DO UPDATE SET
                outcome = excluded.outcome,
                exit_code = COALESCE(excluded.exit_code, sidecar.exit_code)""",
            (tool_key, owner_present, owner_value, outcome.value, exit_code),
        )

    def _sidecar_value(self, tool_id: str | None, owner: str | None, column: str) -> str | int | None:
        if tool_id is None:
            return None
        owner_present, owner_value = self._owner_key(owner)
        tool_key = _exact_key(tool_id)
        row = self.conn.execute(
            f"SELECT {column} FROM sidecar WHERE tool_id = ? AND owner_present = ? AND owner = ?",
            (tool_key, owner_present, owner_value),
        ).fetchone()
        if row is not None and row[0] is not None:
            return cast(str | int | None, row[0])
        row = self.conn.execute(
            f"SELECT {column} FROM sidecar WHERE tool_id = ? AND owner_present = 0 AND owner = x''",
            (tool_key,),
        ).fetchone()
        return cast(str | int | None, row[0]) if row is not None else None

    def sidecar_outcome(self, tool_id: str | None, owner: str | None) -> ToolOutcome | None:
        value = self._sidecar_value(tool_id, owner, "outcome")
        return ToolOutcome(cast(str, value)) if value is not None else None

    def sidecar_exit(self, tool_id: str | None, owner: str | None) -> int | None:
        value = self._sidecar_value(tool_id, owner, "exit_code")
        return int(value) if value is not None else None

    def unmatched_sidecar_outcome(self, tool_id: str, *, origin: Origin) -> ToolOutcome | None:
        """Resolve an otherwise unmatched tool use only from one shared verdict.

        Owner-qualified sidecars normally resolve their own result blocks. If
        no result was paired, their tool-ID-wide projection is usable only
        when every record reports the same outcome; choosing one owner's
        verdict would silently attribute evidence across records.
        """
        with closing(
            self.conn.execute(
                "SELECT outcome FROM sidecar WHERE tool_id = ? ORDER BY outcome ASC LIMIT 1", (_exact_key(tool_id),)
            )
        ) as rows:
            first = rows.fetchone()
        if first is None:
            return None
        with closing(
            self.conn.execute(
                "SELECT outcome FROM sidecar WHERE tool_id = ? ORDER BY outcome DESC LIMIT 1", (_exact_key(tool_id),)
            )
        ) as rows:
            last = rows.fetchone()
        if last is not None and last[0] != first[0]:
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"ambiguous execution evidence for unmatched tool use tool_id={tool_id!r}"
            )
        return ToolOutcome(first[0])

    def use_outcome(self, message_ordinal: int, block_ordinal: int) -> ToolOutcome | None:
        row = self.conn.execute(
            "SELECT verdict FROM resolved_uses WHERE use_key=?", (f"{message_ordinal}:{block_ordinal}",)
        ).fetchone()
        return ToolOutcome(row[0]) if row is not None else None


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
    into: SqliteMessageSink | None = None,
) -> list[ParsedMessage] | SqliteMessageSink:
    """Resolve tool outcomes from each origin's structured parser evidence.

    ``is_error``, exit codes and ``outcome_unknown_reason`` are all
    parser-normalized evidence; this function adds no meaning of its own and
    never invents a reason an origin's parser did not derive. Claude Code
    additionally carries outcome fields in its record-level execution event.
    A result without any such evidence is a parser defect and refuses the
    write. A tool-use without a paired result is a recorded interruption and
    receives the distinct, known ``no_result`` outcome.

    With ``into``, a sink's normalized messages are appended to that empty
    sink in one pass instead of being rewritten in place.
    """
    with _outcome_index(messages) as index:
        _index_sidecars(index, events, origin=origin)
        _index_results(index, messages, origin=origin)
        if isinstance(messages, SqliteMessageSink) and into is not None:
            if len(into):
                raise ValueError("tool outcome normalization appends into an empty sink")
            for ordinal, message in enumerate(messages):
                into.append(_normalize_message(index, message, ordinal=ordinal, origin=origin))
            return into
        if isinstance(messages, SqliteMessageSink):
            # Only a message with a tool block is normalized into anything
            # other than itself; every other row already holds its value.
            with messages.atomic_edit():
                for (ordinal,) in index.conn.execute("SELECT ordinal FROM tool_messages ORDER BY ordinal"):
                    messages[ordinal] = _normalize_message(index, messages[ordinal], ordinal=ordinal, origin=origin)
            return messages
        return [
            _normalize_message(index, message, ordinal=ordinal, origin=origin)
            for ordinal, message in enumerate(messages)
        ]


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
    for ordinal, message in enumerate(messages):
        native_id = message.provider_message_id
        normalized_id = native_id.strip() if native_id else None
        index.conn.execute(
            "INSERT INTO original_messages VALUES(?,?,?,?,?,?,?,?,?)",
            (
                ordinal,
                _exact_key(native_id),
                _exact_key(normalized_id or None),
                _exact_key(message.parent_message_provider_id),
                message.parent_message_position,
                message.position,
                message.role.value,
                message.position if message.position is not None else ordinal,
                message.variant_index or 0,
            ),
        )
        if any(block.type in (BlockType.TOOL_USE, BlockType.TOOL_RESULT) for block in message.blocks):
            index.conn.execute("INSERT INTO tool_messages VALUES(?)", (ordinal,))
        for block_ordinal, block in enumerate(message.blocks):
            if block.type not in (BlockType.TOOL_USE, BlockType.TOOL_RESULT) or not block.tool_id:
                continue
            if block.type is BlockType.TOOL_USE:
                index.conn.execute(
                    "INSERT INTO association_blocks VALUES(?,?,?,?,?,?,?,?)",
                    (0, f"{ordinal}:{block_ordinal}", ordinal, _exact_key(block.tool_id), 1, None, None, block_ordinal),
                )
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
            unknown_reason = block.outcome_unknown_reason if outcome is ToolOutcome.UNKNOWN else None
            index.conn.execute(
                "INSERT INTO association_blocks VALUES(?,?,?,?,?,?,?,?)",
                (
                    0,
                    f"{ordinal}:{block_ordinal}",
                    ordinal,
                    _exact_key(block.tool_id),
                    0,
                    outcome.value,
                    unknown_reason,
                    block_ordinal,
                ),
            )
    # Use the same native uniqueness and last declared-position parent law
    # as the original writer. IDs stay provider facts; ordinals only locate
    # these exact occurrences in the caller-owned preparation table.
    index.conn.execute("""CREATE VIEW association_messages AS
        SELECT 0 AS session_key,m.ordinal AS message_key,
          COALESCE((SELECT p.ordinal FROM original_messages p
                    WHERE p.native_id=m.parent_native_id
                    AND (SELECT COUNT(*) FROM original_messages n
                         WHERE n.normalized_id=p.normalized_id)=1),
                   (SELECT MAX(p.ordinal) FROM original_messages p
                    WHERE p.declared_position=m.parent_position)) AS parent_key,
          m.role,m.position AS message_position,m.variant AS variant_index
        FROM original_messages m""")
    index.conn.execute(
        "CREATE TABLE resolved_uses AS WITH RECURSIVE "
        + tool_association_ctes_sql()
        + " SELECT use_key,verdict FROM tool_associations"
    )
    index.conn.execute("CREATE UNIQUE INDEX resolved_use_key ON resolved_uses(use_key)")


def _normalize_message(index: _OutcomeIndex, message: ParsedMessage, *, ordinal: int, origin: Origin) -> ParsedMessage:
    blocks: list[ParsedContentBlock] = []
    for block_ordinal, block in enumerate(message.blocks):
        # Association enriches a read model. Bind the original Source semantics
        # first so later results cannot rename an existing tool-use block.
        if block.source_content_identity is None:
            block = block.model_copy(update={"source_content_identity": block_content_identity(block)})
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
            outcome = index.use_outcome(ordinal, block_ordinal) if block.tool_id else None
            if outcome in (None, ToolOutcome.NO_RESULT) and block.tool_id:
                outcome = index.unmatched_sidecar_outcome(block.tool_id, origin=origin)
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
