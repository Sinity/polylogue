"""Codex JSONL session parser."""

from __future__ import annotations

import codecs
import hashlib
import itertools
import json
import math
import pickle
import re
import shlex
import sqlite3
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import IO, cast

from pydantic import ValidationError

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.provider.semantics import extract_codex_text
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.timestamps import parse_timestamp_pair
from polylogue.logging import DEBUG, WARNING, emit, get_logger
from polylogue.sources.detection_projection import DetectorProjection
from polylogue.sources.pickle_spool import PickleSpool
from polylogue.sources.providers.codex import CodexRecord
from polylogue.sources.tool_result_reasons import unknown_reason

from .base import (
    AdmissionLedger,
    AdmissionUnit,
    ParseAccounting,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    content_blocks_from_segments,
    mark_last_occurrence_as_active_leaf,
    parser_admission,
)
from .base_support import codex_unknown_wire_type

logger = get_logger(__name__)
_TimestampPair = tuple[datetime, str]
#: Tool names whose payload is a patch-format string rather than JSON.
#: Mirrors the ``apply_patch`` child-type aliases registered below, so the
#: standalone ``function_call`` path and the batched code-mode child path
#: recognise exactly the same set.
_PATCH_TOOL_NAMES = frozenset({"apply_patch", "patch"})

_EXECUTION_TOOL_NAMES = frozenset(
    {
        "bash",
        "exec",
        "exec_command",
        "functions.exec",
        "functions.exec_command",
        "local_shell_call",
        "run",
        "shell",
        "shell_command",
        "terminal",
    }
)
_CODE_MODE_EXEC_TOOL_NAMES = frozenset({"exec", "functions.exec"})
_CODE_MODE_CHILD_PROVENANCE_KEY = "_polylogue"
_CODE_MODE_CHILD_ID_MARKER = "::polylogue-child::"
_CODE_MODE_CHILD_COLLECTION_KEYS = (
    "calls",
    "tool_calls",
    "children",
    "operations",
    "actions",
    "invocations",
)
_CODE_MODE_RESULT_COLLECTION_KEYS = (
    "results",
    "tool_results",
    "children",
    "outputs",
    "responses",
)
# ``event_msg`` -> ``payload.item`` is the producer's own structural record of
# one operation the code-mode ``exec`` program performed: the argv it ran, the
# cwd, the full stdout, and -- the part no other record in this wire generation
# carries -- the exit code. The transport ``custom_tool_call_output`` states
# only what the model was shown. Each item type maps to the child registry
# types it can be the execution of; see ``_codex_lookahead`` for how one
# is attributed to a child.
_CODE_MODE_ITEM_CHILD_TYPES: dict[str, frozenset[str]] = {
    "CommandExecution": frozenset({"exec_command", "write_stdin", "wait"}),
    "FileChange": frozenset({"apply_patch"}),
}
# Preference order for an item's output text. ``aggregated_output`` is stdout
# and stderr interleaved in emission order; the others are the same bytes split
# or re-rendered, so exactly one is stored.
_CODE_MODE_ITEM_TEXT_KEYS = ("aggregated_output", "stdout", "formatted_output", "stderr")
_STRUCTURAL_PATH_KEYS = frozenset({"path", "file_path", "paths", "file_paths", "image_path"})
_STRUCTURAL_BYTE_KEYS = frozenset({"bytes", "byte_count", "bytes_written", "size_bytes", "written_bytes"})
# The parser walks a rollout twice (lookahead, then materialization). A stream
# whose decoded records fit this many retained bytes is replayed from a list;
# a larger one is spooled to the parse's scratch index. The budget chooses a
# storage tier only: both tiers replay the same records in the same order.
_CODEX_REPLAY_MEMORY_BUDGET_BYTES = 8 * 1024 * 1024
_LIST_REFERENCE_BYTES = 8


def _retained_record_bytes(value: object) -> int:
    """Estimate the object graph an in-memory replay keeps alive for ``value``.

    Codex stream records are JSON-shaped mappings and sequences, so the size
    is the sum of every reachable object's ``sys.getsizeof``. One giant nested
    payload therefore cannot evade the replay budget. The walk is iterative so
    nesting depth cannot raise ``RecursionError``; a container reached twice
    is charged once, which also stops at a reference cycle. A scalar shared
    between places is charged at each, which only spools sooner.
    """
    getsizeof = sys.getsizeof
    size = 0
    seen: set[int] = set()
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            if id(item) in seen:
                continue
            seen.add(id(item))
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            pending.extend(item)
        size += getsizeof(item)
    return size


def _drain(items: list[object]) -> Iterator[object]:
    """Yield ``items`` in order, releasing each from the list as it is yielded."""
    items.reverse()
    while items:
        yield items.pop()


class _CodexLookaheadIndex:
    """Disk-backed facts shared by the two passes over one rollout."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._spool: PickleSpool[object] | None = None
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS codex_message_echoes (
                record_index INTEGER PRIMARY KEY, signature BLOB NOT NULL,
                native_id TEXT, instant TEXT, turn_id TEXT, consumed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS codex_message_echo_signature
                ON codex_message_echoes(signature, consumed);
            CREATE TABLE IF NOT EXISTS codex_calls (
                record_index INTEGER PRIMARY KEY, tool_id BLOB, occurrence INTEGER,
                envelope BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS codex_outputs (
                record_index INTEGER PRIMARY KEY, tool_id BLOB NOT NULL,
                occurrence INTEGER NOT NULL, output BLOB NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS codex_outputs_pair
                ON codex_outputs(tool_id, occurrence);
            CREATE TABLE IF NOT EXISTS codex_occurrences (
                kind TEXT NOT NULL, tool_id BLOB NOT NULL, count INTEGER NOT NULL,
                PRIMARY KEY (kind, tool_id)
            );
            CREATE TABLE IF NOT EXISTS codex_items (
                item_order INTEGER PRIMARY KEY, item BLOB NOT NULL,
                open_call_index INTEGER, last_call_index INTEGER
            );
            CREATE TABLE IF NOT EXISTS codex_slots (
                slot INTEGER PRIMARY KEY, call_index INTEGER NOT NULL,
                child_index INTEGER NOT NULL, registry_type TEXT NOT NULL,
                command BLOB, claimed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS codex_slots_command
                ON codex_slots(command, claimed, slot);
            CREATE INDEX IF NOT EXISTS codex_slots_call
                ON codex_slots(call_index, claimed, slot);
            CREATE TABLE IF NOT EXISTS codex_matched (
                call_index INTEGER NOT NULL, child_index INTEGER NOT NULL,
                item BLOB NOT NULL, PRIMARY KEY (call_index, child_index)
            );
            CREATE TABLE IF NOT EXISTS codex_appended (
                call_index INTEGER NOT NULL, item_order INTEGER NOT NULL,
                item BLOB NOT NULL, PRIMARY KEY (call_index, item_order)
            );
            CREATE TABLE IF NOT EXISTS codex_resolved (
                record_index INTEGER PRIMARY KEY, envelope BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS codex_instruction_revisions (
                kind TEXT NOT NULL, key BLOB NOT NULL, value BLOB NOT NULL,
                PRIMARY KEY (kind, key)
            );
            CREATE TABLE IF NOT EXISTS codex_workdirs (value BLOB PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS codex_task_texts (
                key BLOB PRIMARY KEY, text BLOB NOT NULL,
                retained INTEGER NOT NULL DEFAULT 0, stored INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS codex_task_keys (
                value BLOB PRIMARY KEY, candidate_key BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS codex_task_events (
                event_index INTEGER PRIMARY KEY, key BLOB NOT NULL,
                text_chars INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS codex_replacement_texts (
                key BLOB PRIMARY KEY, text BLOB NOT NULL, text_chars INTEGER NOT NULL,
                occurrences INTEGER NOT NULL DEFAULT 1,
                retained INTEGER NOT NULL DEFAULT 0, stored INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS codex_replacement_keys (
                value BLOB PRIMARY KEY, candidate_key BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS codex_replacement_contexts (
                ordinal INTEGER PRIMARY KEY, insert_at INTEGER NOT NULL, key BLOB NOT NULL,
                timestamp TEXT, source_index INTEGER NOT NULL,
                entry_type BLOB, role BLOB, phase BLOB
            );
            """
        )

    def retain_records(self, records: Iterable[object], budget_bytes: int) -> list[object] | None:
        """Keep ``records`` in memory while they fit ``budget_bytes``, else spool them all.

        Returns the list when the whole stream fits. Otherwise every record,
        the ones already retained first, goes to the record spool in stream
        order and the result is ``None``: replay then reads the spool.
        """
        retained: list[object] = []
        retained_bytes = sys.getsizeof(retained)
        iterator = iter(records)
        for record in iterator:
            retained.append(record)
            retained_bytes += _LIST_REFERENCE_BYTES + _retained_record_bytes(record)
            if retained_bytes > budget_bytes:
                self.spool_records(itertools.chain(_drain(retained), iterator))
                return None
        return retained

    def spool_records(self, records: Iterable[object]) -> None:
        spool = self._spool = PickleSpool[object]()
        for record in records:
            spool.append(record)

    def replay_records(self) -> Iterator[object]:
        if self._spool is not None:
            yield from self._spool

    def close(self) -> None:
        if self._spool is not None:
            self._spool.close()
            self._spool = None

    def __iter__(self) -> Iterator[object]:
        return self.replay_records()

    def occurrence(self, kind: str, tool_id: str) -> int:
        row = self.connection.execute(
            """INSERT INTO codex_occurrences(kind, tool_id, count) VALUES (?, ?, 1)
               ON CONFLICT(kind, tool_id) DO UPDATE SET count = count + 1
               RETURNING count""",
            (kind, _sql_key(tool_id)),
        ).fetchone()
        assert row is not None
        return int(row[0]) - 1

    def add_message_echo(
        self, record_index: int, signature: tuple[str, str], evidence: tuple[str | None, str | None, str | None]
    ) -> None:
        self.connection.execute(
            "INSERT INTO codex_message_echoes(record_index, signature, native_id, instant, turn_id) VALUES (?, ?, ?, ?, ?)",
            (record_index, _signature_key(signature), *evidence),
        )

    def consume_message_echo(
        self, signature: tuple[str, str], evidence: tuple[str | None, str | None, str | None]
    ) -> bool:
        # Text equality alone cannot establish that two records are mirrors.
        # Consume one occurrence with positive correlation and no contradictory
        # time/turn evidence, so a later identical prompt remains a real turn.
        native_id, instant, turn_id = evidence
        row = self.connection.execute(
            """SELECT record_index FROM codex_message_echoes
               WHERE signature = ? AND consumed = 0
                 AND (native_id = ? OR instant = ? OR turn_id = ?)
                 AND (instant IS NULL OR ? IS NULL OR instant = ?)
                 AND (turn_id IS NULL OR ? IS NULL OR turn_id = ?)
               ORDER BY record_index LIMIT 1""",
            (_signature_key(signature), native_id, instant, turn_id, instant, instant, turn_id, turn_id),
        ).fetchone()
        if row is None:
            return False
        self.connection.execute("UPDATE codex_message_echoes SET consumed = 1 WHERE record_index = ?", (row[0],))
        return True

    def get(self, record_index: int) -> _CodexExecEnvelope | None:
        row = self.connection.execute(
            "SELECT envelope FROM codex_resolved WHERE record_index = ?", (record_index,)
        ).fetchone()
        return pickle.loads(row[0]) if row is not None else None

    def add_workdir(self, value: str) -> None:
        self.connection.execute("INSERT OR IGNORE INTO codex_workdirs VALUES (?)", (_sql_key(value),))

    def working_directories(self) -> list[str]:
        return sorted(pickle.loads(value) for (value,) in self.connection.execute("SELECT value FROM codex_workdirs"))


@dataclass(frozen=True, slots=True)
class _CodexExecChildType:
    kind: str
    aliases: frozenset[str]


_CODE_MODE_CHILD_REGISTRY = (
    _CodexExecChildType(
        kind="exec_command",
        aliases=frozenset(
            {
                "bash",
                "exec_command",
                "local_shell_call",
                "shell",
                "shell_command",
                "terminal",
                "unified_exec",
            }
        ),
    ),
    _CodexExecChildType(kind="apply_patch", aliases=frozenset({"apply_patch", "patch"})),
    _CodexExecChildType(kind="write_stdin", aliases=frozenset({"send_input", "write_stdin"})),
    _CodexExecChildType(kind="update_plan", aliases=frozenset({"plan", "update_plan"})),
    _CodexExecChildType(kind="wait", aliases=frozenset({"wait", "wait_for_cell", "wait_for_process"})),
    _CodexExecChildType(
        kind="web",
        aliases=frozenset(
            {
                "open_url",
                "search",
                "tool_search",
                "web",
                "web_open",
                "web_search",
            }
        ),
    ),
    _CodexExecChildType(
        kind="image",
        aliases=frozenset(
            {
                "generated_image",
                "image",
                "image_generation",
                "image_query",
                "view_image",
            }
        ),
    ),
    _CodexExecChildType(
        kind="mcp",
        aliases=frozenset(
            {
                "list_mcp_resource_templates",
                "list_mcp_resources",
                "read_mcp_resource",
            }
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class _CodexExecChildCall:
    tool_path: tuple[str, ...]
    tool_name: str
    registry_type: str
    argument: object
    raw_argument: str | None
    parse_state: str
    source_start: int | None = None
    source_end: int | None = None
    # The ``item_completed`` id when this child is the producer's own record of
    # an execution rather than a call read out of the program source. A program
    # that loops emits one source call and many executions, so the two lists
    # differ in length and only the item list is ground truth.
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class _CodexExecChildResult:
    text: str | None
    is_error: bool | None
    exit_code: int | None
    unknown_reason: str | None
    paths: tuple[str, ...]
    byte_count: int | None
    item_id: str | None = None


# Every key the code-mode item readers below consult by name:
# ``_code_mode_item_child`` (type, command, cwd, parsed_cmd, changes, id),
# ``_code_mode_item_commands`` (command, parsed_cmd),
# ``_code_mode_item_outcome`` (exit_code, status) and the matcher (type).
_CODE_MODE_ITEM_MATCH_KEYS = ("type", "id", "command", "parsed_cmd", "cwd", "changes", "exit_code", "status")


def _sql_key(value: object) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _signature_key(signature: tuple[str, str]) -> bytes:
    """Fixed-size scratch key of a message signature.

    A signature carries the message's whole normalized text; keying the
    B-tree by the pickled text put multi-kilobyte keys on overflow pages and
    made every membership probe read them back. The SHA-256 of the role and
    the exact code points is the same equality at 32 bytes.
    """
    role, text = signature
    return _text_digest(f"{role}\0{text}")


_DIGEST_WINDOW_CHARS = 1 << 20


#: Characters an unsettled tail may hold in memory before its marks spill.
_NFC_UNSETTLED_LIMIT_CHARS = 4 * _DIGEST_WINDOW_CHARS


def _is_nfd_starter(char: str) -> bool:
    """Whether ``char`` decomposes to a starter first (canonical combining class 0)."""
    return unicodedata.combining(char) == 0 and unicodedata.combining(unicodedata.normalize("NFD", char)[0]) == 0


class _SpilledMarkRun:
    """A starter and its run of combining marks, too long to hold, normalized on disk.

    NFC of such a run is its canonical decomposition, stably sorted by
    combining class, then composed with the starter. A stable sort by class is
    a bucketing, so the marks go to one scratch file per class in arrival
    order; composition only ever consumes a prefix of each bucket, so it reads
    one character at a time until the first that stays.
    """

    def __init__(self, unsettled: str) -> None:
        decomposed = unicodedata.normalize("NFD", unsettled)
        self._starter: str | None = None
        if decomposed and unicodedata.combining(decomposed[0]) == 0:
            self._starter, decomposed = decomposed[0], decomposed[1:]
        self._buckets: dict[int, IO[bytes]] = {}
        self.add(decomposed)

    def add(self, marks: str) -> None:
        grouped: dict[int, list[str]] = {}
        for char in unicodedata.normalize("NFD", marks):
            grouped.setdefault(unicodedata.combining(char), []).append(char)
        for combining_class, chars in grouped.items():
            bucket = self._buckets.get(combining_class)
            if bucket is None:
                bucket = self._buckets[combining_class] = tempfile.TemporaryFile()  # noqa: SIM115 -- closed in finish
            bucket.write("".join(chars).encode("utf-8", "surrogatepass"))

    def finish(self, update: Callable[[bytes], object]) -> str:
        """Hash the run's NFC form; return the starter instead when nothing follows it.

        A composed starter with no mark left after it can still compose with
        the next character, so it goes back to the caller's unsettled text.
        """
        try:
            starter = self._starter
            remainders: list[tuple[IO[bytes], int]] = []
            for combining_class in sorted(self._buckets):
                bucket = self._buckets[combining_class]
                bucket.seek(0)
                consumed = 0
                if starter is not None:
                    reader = codecs.getreader("utf-8")(bucket, "surrogatepass")
                    while char := reader.read(1):
                        composed = unicodedata.normalize("NFC", starter + char)
                        if len(composed) != 1:
                            break
                        starter = composed
                        consumed += len(char.encode("utf-8", "surrogatepass"))
                bucket.seek(0, 2)
                if bucket.tell() > consumed:
                    remainders.append((bucket, consumed))
            if not remainders:
                return starter or ""
            if starter is not None:
                update(starter.encode("utf-8", "surrogatepass"))
            for bucket, offset in remainders:
                bucket.seek(offset)
                while chunk := bucket.read(4 * _DIGEST_WINDOW_CHARS):
                    update(chunk)
            return ""
        finally:
            for bucket in self._buckets.values():
                bucket.close()
            self._buckets.clear()


def _nfc_text_digest(text: str) -> bytes:
    """``_text_digest`` of the NFC form of ``text``, without building that form.

    Normalization streams: output before the last starter of what is produced
    is final (later input can only compose with that starter), so each window
    is hashed as it settles and only its unsettled tail is held. A run of
    combining marks too long to hold is normalized on disk
    (:class:`_SpilledMarkRun`), so every value keeps its NFC key.
    """
    if text.isascii():
        return _text_digest(text)
    digest = hashlib.sha256()
    pending = ""
    run: _SpilledMarkRun | None = None
    for start in range(0, len(text), _DIGEST_WINDOW_CHARS):
        window = text[start : start + _DIGEST_WINDOW_CHARS]
        if run is not None:
            boundary = next((index for index, char in enumerate(window) if _is_nfd_starter(char)), len(window))
            run.add(window[:boundary])
            if boundary == len(window):
                continue
            pending, run = run.finish(digest.update), None
            window = window[boundary:]
        normalized = unicodedata.normalize("NFC", pending + window)
        index = len(normalized) - 1
        while index > 0 and unicodedata.combining(normalized[index]) != 0:
            index -= 1
        digest.update(normalized[:index].encode("utf-8", "surrogatepass"))
        pending = normalized[index:]
        if len(pending) > _NFC_UNSETTLED_LIMIT_CHARS:
            run, pending = _SpilledMarkRun(pending), ""
    if run is not None:
        pending = run.finish(digest.update)
    digest.update(unicodedata.normalize("NFC", pending).encode("utf-8", "surrogatepass"))
    return digest.digest()


def _text_digest(text: str) -> bytes:
    """Fixed-size scratch key of a candidate text, hashed in bounded windows.

    A candidate can be arbitrarily large, so it is keyed by the SHA-256 of its
    exact code points (UTF-8, surrogates passed through) and its text is held
    once in a scratch column. The windows keep the transient encoding to one
    window, never a second full-size copy.
    """
    digest = hashlib.sha256()
    for start in range(0, len(text), _DIGEST_WINDOW_CHARS):
        digest.update(text[start : start + _DIGEST_WINDOW_CHARS].encode("utf-8", "surrogatepass"))
    return digest.digest()


def _reduced_code_mode_item(item: dict[str, object]) -> dict[str, object]:
    """Reduce one ``item_completed`` payload to the evidence the parser reads.

    polylogue-ro922: a session file is untrusted input and the lookahead used
    to retain the whole ``payload.item`` mapping for every code-mode item for
    the length of the parse, so a 52 MB file of small items cost half a
    gigabyte resident. The readers consult a fixed key set plus one selected
    output text, and derive paths/byte counts structurally; the structural
    derivations are evaluated here, over the complete item, and stored under
    their own canonical keys first, so a later ``_structural_paths`` /
    ``_structural_byte_count`` over the reduction returns exactly what it
    would have returned over the original.
    """
    reduced: dict[str, object] = {}
    paths = _structural_paths(item)
    if paths:
        reduced["paths"] = list(paths)
    byte_count = _structural_byte_count(item)
    if byte_count is not None:
        reduced["byte_count"] = byte_count
    for key in _CODE_MODE_ITEM_MATCH_KEYS:
        if key in item:
            reduced[key] = item[key]
    for key in _CODE_MODE_ITEM_TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value:
            reduced[key] = value
            break
    return reduced


@dataclass(frozen=True, slots=True)
class _CodexExecEnvelope:
    transport_tool_name: str
    transport_tool_id: str | None
    transport_provider_message_id: str
    children: tuple[_CodexExecChildCall, ...]
    # Positionally aligned with ``children``; ``None`` where no result evidence
    # was recovered for that child.
    results: tuple[_CodexExecChildResult | None, ...] = ()


#: A run of characters that close nothing and start no escape, per quote.
_JS_PLAIN_RUNS = {
    '"': re.compile(r'[^"\\]+'),
    "'": re.compile(r"[^'\\]+"),
    "`": re.compile(r"[^`\\$]+"),
}
_JS_SURROGATE = re.compile("[\ud800-\udfff]")


class _JsLiteralError(ValueError):
    pass


def _iso_or_none(value: str | int | float | None) -> str | None:
    pair = parse_timestamp_pair(value)
    return pair[1] if pair is not None else None


def _newer_timestamp(
    current: _TimestampPair | None,
    value: str | None,
) -> _TimestampPair | None:
    if not isinstance(value, str) or not value:
        return current
    return _newer_timestamp_pair(current, parse_timestamp_pair(value))


def _newer_timestamp_pair(
    current: _TimestampPair | None,
    candidate: _TimestampPair | None,
) -> _TimestampPair | None:
    if candidate is None:
        return current
    if current is None or candidate[0] > current[0]:
        return candidate
    return current


def finalize_codex_session(
    session: ParsedSession,
    *,
    messages: Sequence[ParsedMessage],
    session_events: Sequence[ParsedSessionEvent],
    updated_at: str | None,
    unit_accounting: ParseAccounting,
    mark_active_leaf: bool,
    active_leaf_message_provider_id: str | None = None,
) -> ParsedSession:
    """Apply the canonical final session fields to an ordinary or prefix parse.

    Checkpointed prefixes use a read-only message view whose final row already
    carries its own active-leaf bit. Ordinary parsing asks this helper to mark
    the final occurrence in its sink, preserving the existing behavior.
    """
    if mark_active_leaf:
        if isinstance(messages, list):
            messages[:] = mark_last_occurrence_as_active_leaf(cast(list[ParsedMessage], messages))
        elif messages and isinstance(messages, MutableSequence):
            messages[-1] = messages[-1].model_copy(update={"is_active_leaf": True})
    if active_leaf_message_provider_id is None and messages:
        active_leaf_message_provider_id = messages[-1].provider_message_id if messages[-1].provider_message_id else None
    return session.model_copy(
        update={
            "messages": messages,
            "session_events": session_events,
            "updated_at": updated_at,
            "active_leaf_message_provider_id": active_leaf_message_provider_id,
            "unit_accounting": unit_accounting,
        }
    )


def _has_continuation_evidence(
    *,
    first_timestamp: _TimestampPair | None,
    second_timestamp: _TimestampPair | None,
    first_cwd: str | None,
    second_cwd: str | None,
    first_repo_url: str | None,
    second_repo_url: str | None,
) -> bool:
    """Structural test for the legacy (no `forked_from_id`) continuation fallback.

    A resumed Codex session physically replays the parent conversation's own
    original `session_meta` record as the file's second distinct `session_meta`
    (verified against real multi-meta rollout files: the replayed header's
    timestamp always *precedes* the new session's own start time, and reports
    the same `cwd`/git remote, because the resumed conversation continues in
    the same working tree). A bare count of session_meta records proves
    neither fact -- two structurally unrelated session_meta records
    concatenated into one payload would satisfy the count without satisfying
    this check, so the count alone is not sufficient evidence of a parent
    relationship.
    """
    if first_timestamp is None or second_timestamp is None:
        return False
    if second_timestamp[0] > first_timestamp[0]:
        # The candidate parent's own header postdates the child's -- not a
        # replayed prefix.
        return False
    cwd_match = bool(first_cwd) and bool(second_cwd) and first_cwd == second_cwd
    repo_match = bool(first_repo_url) and bool(second_repo_url) and first_repo_url == second_repo_url
    return cwd_match or repo_match


def _redacted_validation_errors(exc: ValidationError) -> str:
    """Structural summary of a validation failure, carrying no payload content.

    ``str(exc)`` and ``exc.errors()`` both reproduce the rejected input by
    default. This keeps only each error's ``loc`` (the field path) and
    ``type`` (the rule that fired), which is what a parse-skip diagnosis
    needs, and drops ``msg``/``input``/``ctx`` -- ``msg`` can quote the input
    and ``ctx`` can carry it verbatim.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('type', 'unknown')}"
        for error in exc.errors(include_url=False, include_context=False, include_input=False)
    )


def _validate_record(item: object, *, index: int, context: str = "record") -> CodexRecord | None:
    if not isinstance(item, dict):
        return None
    try:
        return CodexRecord.model_validate(item)
    except ValidationError as exc:
        # Never interpolate the ValidationError itself: Pydantic v2's __str__
        # embeds ``input_value``, i.e. raw captured payload content, into the
        # log line (2026-07-31 leak audit L13, polylogue-tztk). Only the
        # structural coordinates -- which field failed and how -- are needed
        # to diagnose a parse skip, and those carry no content.
        emit(
            "parser.codex.record_skipped",
            level=DEBUG,
            outcome="degraded",
            reason="codex_record_validation_failed",
            context=context,
            index=index,
            errors=_redacted_validation_errors(exc),
        )
        return None


def _dict_record(item: object) -> dict[str, object] | None:
    return item if isinstance(item, dict) else None


def _is_plausibly_codex_record(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("record_type") == "state":
        return True

    record_type = item.get("type")
    payload = item.get("payload")
    if record_type in {"session_meta", "response_item", "event_msg", "compacted", "turn_context"}:
        return isinstance(payload, dict)
    if isinstance(payload, dict):
        return True

    role = item.get("role")
    content = item.get("content")
    if record_type == "message" or isinstance(role, str):
        return "content" not in item or isinstance(content, list)

    return bool(item.get("id") and item.get("timestamp") and "message" not in item)


def _payload_record(record: dict[str, object]) -> dict[str, object] | None:
    return _dict_record(record.get("payload"))


def _record_type(record: dict[str, object]) -> str | None:
    value = record.get("type")
    return value if isinstance(value, str) else None


def _record_id(record: dict[str, object]) -> str | None:
    value = record.get("id")
    return value if isinstance(value, str) else None


def _record_timestamp(record: dict[str, object]) -> str | int | float | None:
    value = record.get("timestamp")
    return value if isinstance(value, str | int | float) else None


def _message_timestamp(record: dict[str, object], message_record: dict[str, object]) -> str | int | float | None:
    return _record_timestamp(message_record) or _record_timestamp(record)


def _record_instructions(record: dict[str, object]) -> str | None:
    value = record.get("instructions")
    return value if isinstance(value, str) else None


def _string_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_field(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        if value := _string_value(record.get(key)):
            return value
    return None


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str) and value.strip():
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return 0
    return 0


def _optional_int_field(record: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        if key in record:
            value = record[key]
            if value is None or isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value if value >= 0 else None
            if isinstance(value, float):
                return int(value) if math.isfinite(value) and value.is_integer() and value >= 0 else None
            if isinstance(value, str):
                try:
                    parsed = int(value.strip())
                except ValueError:
                    return None
                return parsed if parsed >= 0 else None
            return None
    return None


def _codex_token_usage_payload(record: dict[str, object] | None) -> dict[str, int]:
    if not record:
        return {}
    usage: dict[str, int] = {}
    field_aliases = {
        "input_tokens": ("input_tokens", "inputTokenCount"),
        "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens", "cached_tokens"),
        "cache_write_tokens": ("cache_write_tokens", "cache_creation_input_tokens", "cache_write_input_tokens"),
        "uncached_input_tokens": ("uncached_input_tokens", "uncachedInputTokens"),
        "output_tokens": ("output_tokens", "outputTokenCount"),
        "reasoning_output_tokens": ("reasoning_output_tokens", "reasoning_tokens"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
    }
    for public_key, aliases in field_aliases.items():
        value = _optional_int_field(record, *aliases)
        if value is not None:
            usage[public_key] = value
    return usage


def _turn_context_payload(payload: dict[str, object]) -> dict[str, object]:
    nested = payload.get("turn_context")
    if isinstance(nested, dict):
        merged = {str(key): value for key, value in nested.items()}
        merged.update({str(key): value for key, value in payload.items() if key != "turn_context"})
        return merged
    return payload


def _token_usage(record: dict[str, object]) -> dict[str, int | None]:
    """Extract per-message usage as disjoint additive pricing lanes.

    Codex input includes cache reads, while message pricing bills fresh input
    and cache reads separately. Normalize at the source; Claude already reports
    these lanes disjointly. The event rollup applies the same rule in
    ``provider_usage_disjoint_lanes``.
    """
    usage = _dict_record(record.get("usage")) or _dict_record(record.get("tokens")) or record
    input_value = _optional_int_field(usage, "input_tokens", "inputTokenCount")
    explicit_uncached_input = _optional_int_field(usage, "uncached_input_tokens", "uncachedInputTokens")
    cache_read_tokens = _optional_int_field(
        usage, "cache_read_tokens", "cache_read_input_tokens", "cached_input_tokens", "cached_tokens"
    )
    output_tokens = _optional_int_field(usage, "output_tokens", "outputTokenCount")
    cache_write_tokens = _optional_int_field(
        usage, "cache_write_tokens", "cache_creation_input_tokens", "cache_write_input_tokens"
    )
    return {
        "input_tokens": (
            explicit_uncached_input
            if explicit_uncached_input is not None
            else (max(input_value - (cache_read_tokens or 0), 0) if input_value is not None else None)
        ),
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def _session_meta_record(record: dict[str, object]) -> dict[str, object] | None:
    if _record_type(record) == "session_meta":
        return _payload_record(record)
    if _record_id(record) and _record_timestamp(record) and not _record_type(record):
        return record
    return None


def _is_envelope(record: dict[str, object]) -> bool:
    return isinstance(record.get("payload"), dict)


def _is_state(record: dict[str, object]) -> bool:
    return record.get("record_type") == "state"


def _is_direct_message(record: dict[str, object]) -> bool:
    return _record_type(record) == "message" or isinstance(record.get("role"), str)


def _is_message(record: dict[str, object]) -> bool:
    if _is_envelope(record):
        return _record_type(record) == "response_item"
    return _is_direct_message(record)


#: ``event_msg`` payload types that ARE conversational content. ``parse``
#: materializes them through ``_codex_event_message``; naming them here is what
#: lets stream classification agree with the parser (polylogue-1wom1) instead
#: of refusing a rollout whose only messages arrive in this shape.
_EVENT_MESSAGE_PAYLOAD_TYPES: frozenset[str] = frozenset({"user_message", "agent_message"})

#: Top-level records of the 2025 direct-message stream generation that carry,
#: unwrapped, exactly the payload a later ``response_item`` envelope wraps.
#: They are materialized through the same response-item route, so a tool call
#: and its output pair, and a reasoning summary lowers, identically in both
#: generations.
_CODEX_LEGACY_RESPONSE_RECORD_TYPES: frozenset[str] = frozenset({"function_call", "function_call_output", "reasoning"})

#: ``retained_context`` payload types. ``verified_answer`` is the user's
#: accepted answer set for a ``request_user_input`` call: question/answer
#: pairs keyed by that call's ``call_id``. It is materialized as a
#: ``verified_answer`` session event anchored to the call.
_CODEX_RETAINED_CONTEXT_PAYLOAD_TYPES: frozenset[str] = frozenset({"verified_answer"})

#: ``realtime_item`` payload types. Both are lifecycle markers of a realtime
#: session: they carry no conversation content, only the realtime session id,
#: the marker id and (on close) an outcome token. They are materialized as
#: session events under their own wire names, so the timeline keeps when a
#: realtime session ran and how it ended.
_CODEX_REALTIME_PAYLOAD_TYPES: frozenset[str] = frozenset({"realtime_session_started", "realtime_session_closed"})


def _legacy_response_record(record: dict[str, object]) -> dict[str, object] | None:
    """Return a 2025 top-level response record as its own response-item payload."""
    if _record_type(record) in _CODEX_LEGACY_RESPONSE_RECORD_TYPES and "payload" not in record:
        return record
    return None


def is_legacy_response_record(item: object) -> bool:
    """Whether ``item`` is an unwrapped 2025 response record the parser materializes.

    Such a record carries no generic envelope marker, so artifact candidacy
    counts it as Codex record evidence through this predicate. It reads only
    ``type`` and the presence of ``payload``, both of which candidacy
    projection retains.
    """
    record = _dict_record(item)
    return record is not None and _legacy_response_record(record) is not None


def _message_record(record: dict[str, object]) -> dict[str, object] | None:
    if _is_state(record):
        return None
    if _record_type(record) == "response_item":
        inner = _payload_record(record)
        return inner if inner is not None and _is_message(inner) else None
    if _record_type(record) == "event_msg":
        inner = _payload_record(record)
        if inner is not None and _record_type(inner) in _EVENT_MESSAGE_PAYLOAD_TYPES:
            return inner
        return None
    return record if _is_message(record) else None


def _git_context(record: dict[str, object]) -> dict[str, object] | None:
    git = _dict_record(record.get("git"))
    if git is None:
        return None
    payload = {str(key): value for key, value in git.items() if value is not None}
    return payload or None


def _record_payload(record: dict[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in record.items() if value is not None}


def _codex_turn_evidence(payload: dict[str, object]) -> dict[str, object]:
    """Lift Codex's three observed turn-correlation carriers.

    ``metadata.turn_id`` is the oldest and most consistently populated
    carrier, so it remains the authoritative scalar for compatibility.  The
    passthrough and direct fields are retained as evidence too; when they
    disagree, ``turn_id_conflict`` makes the ambiguity explicit instead of
    silently selecting whichever field happened to be visited last.
    """
    candidates: dict[str, str] = {}
    metadata = _dict_record(payload.get("metadata"))
    metadata_turn_id = _string_value(metadata.get("turn_id")) if metadata else None
    if metadata_turn_id:
        candidates["metadata.turn_id"] = metadata_turn_id
    passthrough = _dict_record(payload.get("internal_chat_message_metadata_passthrough"))
    passthrough_turn_id = _string_value(passthrough.get("turn_id")) if passthrough else None
    if passthrough_turn_id:
        candidates["internal_chat_message_metadata_passthrough.turn_id"] = passthrough_turn_id
    direct_turn_id = _string_value(payload.get("turn_id"))
    if direct_turn_id:
        candidates["turn_id"] = direct_turn_id
    if not candidates:
        return {}
    # The established metadata.turn_id behavior wins; passthrough is the
    # authority when metadata is absent, with a direct field as final fallback.
    authority = next(
        (
            name
            for name in ("metadata.turn_id", "internal_chat_message_metadata_passthrough.turn_id", "turn_id")
            if name in candidates
        ),
        next(iter(candidates)),
    )
    evidence: dict[str, object] = {"turn_id": candidates[authority], "turn_id_source": authority}
    distinct_values = sorted(set(candidates.values()))
    if len(distinct_values) > 1:
        evidence["turn_id_conflict"] = {
            "authority": authority,
            "values": dict(candidates),
        }
    return evidence


def _codex_source_references(value: object, *, field: str) -> list[dict[str, object]]:
    """Retain bounded source references without claiming byte acquisition.

    Codex ``local_images`` are paths/references into the producer's machine;
    they are not attachment bytes.  Keeping the reference in the event
    payload preserves identity and provenance while the explicit policy field
    prevents readers from treating it as an acquired/public asset.
    """
    if not isinstance(value, list):
        return []
    references: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, str) and item:
            references.append(
                {
                    "reference": _sanitize_codex_data_url(item),
                    "source": f"codex.user_message.{field}",
                    "acquired_bytes": False,
                    "path_disclosure": "provider_reference",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        reference: dict[str, object] = {
            "source": f"codex.user_message.{field}",
            "acquired_bytes": False,
            "path_disclosure": "provider_reference",
        }
        # Preserve identity/display metadata, never inline byte payloads.
        for key in ("path", "url", "uri", "name", "id", "mime_type", "media_type", "type", "text"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate:
                reference[key] = _sanitize_codex_data_url(candidate)
        if reference.keys() > {"source", "acquired_bytes", "path_disclosure"}:
            references.append(reference)
    return references


def _codex_text_element_references(value: object) -> list[dict[str, object]]:
    """Keep placeholder/range coordinates while excluding hidden content."""
    if not isinstance(value, list):
        return []
    references: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        reference: dict[str, object] = {
            "source": "codex.user_message.text_elements",
            "content_policy": "range_only",
        }
        for key in (
            "type",
            "id",
            "start",
            "end",
            "start_index",
            "end_index",
            "start_offset",
            "end_offset",
            "range",
            "placeholder",
            "kind",
        ):
            candidate = item.get(key)
            if isinstance(candidate, (str, int, float)) and not isinstance(candidate, bool):
                reference[key] = candidate
            elif isinstance(candidate, list) and key == "range":
                reference[key] = [entry for entry in candidate if isinstance(entry, (str, int, float))]
            elif isinstance(candidate, dict) and key == "range":
                # Current Codex builds use both ``[start, end]`` and a named
                # range object. Preserve the latter's coordinates without
                # copying provider-private text or arbitrary metadata.
                range_reference = {
                    range_key: range_value
                    for range_key, range_value in candidate.items()
                    if isinstance(range_value, (str, int, float))
                    and not isinstance(range_value, bool)
                    and range_key in {"start", "end", "start_index", "end_index", "start_offset", "end_offset"}
                }
                if range_reference:
                    reference[key] = range_reference
        references.append(reference)
    return references


def _codex_error_evidence(payload: dict[str, object]) -> dict[str, object]:
    """Lift user-facing error text and the producer's typed error code.

    Codex emits these fields both directly on an ``error`` item and nested
    under ``task_complete.error``.  The latter is especially important: a
    usage-limit completion otherwise looks identical to a normal completion.
    Keep the normalized names stable regardless of which wire placement was
    used, and do not infer a code from the human-readable message.
    """
    error = _dict_record(payload.get("error"))
    message = _string_value(payload.get("message"))
    if message is None and error is not None:
        message = _string_value(error.get("message"))
    error_info: object = payload.get("codex_error_info")
    if error_info is None and error is not None:
        error_info = error.get("codex_error_info")
    evidence: dict[str, object] = {}
    if message:
        evidence["message"] = message
    if isinstance(error_info, (str, dict)):
        evidence["codex_error_info"] = error_info
    return evidence


def _codex_settings_record(payload: dict[str, object]) -> dict[str, object]:
    """Resolve the top-level or nested settings shape used by Codex builds."""
    settings = _dict_record(payload.get("settings")) or _dict_record(payload.get("thread_settings"))
    if settings is None:
        return payload
    merged = {str(key): value for key, value in settings.items()}
    merged.update({str(key): value for key, value in payload.items() if key not in {"settings", "thread_settings"}})
    return merged


def _codex_collaboration_mode(value: object) -> dict[str, object] | None:
    """Retain named collaboration settings without copying opaque payloads."""
    if isinstance(value, str) and value:
        return {"mode": value}
    mode = _dict_record(value)
    if mode is None:
        return None
    retained: dict[str, object] = {}
    scalar_keys = (
        "kind",
        "mode",
        "name",
        "developer_instructions",
        "agent_role",
        "agent_nickname",
    )
    for key in scalar_keys:
        field_value = mode.get(key)
        if isinstance(field_value, (str, int, float, bool)):
            retained[key] = field_value
    nested_settings = _dict_record(mode.get("settings")) or _dict_record(mode.get("config"))
    if nested_settings is not None:
        for key in ("developer_instructions", "mode", "kind", "name", "agent_role", "agent_nickname"):
            field_value = nested_settings.get(key)
            if isinstance(field_value, (str, int, float, bool)) and key not in retained:
                retained[key] = field_value
    return retained or None


_CODEX_EVENT_ITEM_DUPLICATE_TYPES = frozenset({"CommandExecution", "FileChange"})


def _codex_semantic_response_fields(payload: dict[str, object]) -> dict[str, object]:
    """Extract the reviewed Codex event fields beyond generic identity keys."""
    compact: dict[str, object] = {}
    phase = _string_value(payload.get("phase"))
    if phase:
        compact["phase"] = phase
    compact.update(_codex_turn_evidence(payload))

    event_type = _string_value(payload.get("type"))
    if event_type == "thread_goal_updated":
        goal = _dict_record(payload.get("goal"))
        if goal:
            compact_goal: dict[str, object] = {}
            for key in ("objective", "status", "tokensUsed", "timeUsedSeconds"):
                if key in goal and isinstance(goal[key], (str, int, float, bool)):
                    compact_goal[key] = goal[key]
            if compact_goal:
                compact["goal"] = compact_goal
    elif event_type == "sub_agent_activity":
        for key in ("agent_thread_id", "agent_path", "kind"):
            value = _string_value(payload.get(key))
            if value:
                compact[key] = value
    elif event_type == "task_started":
        for key in ("model_context_window", "collaboration_mode_kind"):
            field_value = payload.get(key)
            if isinstance(field_value, (str, int, float, bool)):
                compact[key] = field_value
    elif event_type in {"task_complete", "turn_aborted"}:
        if event_type == "task_complete":
            message_value = payload.get("last_agent_message")
            if isinstance(message_value, str) and message_value:
                compact["last_agent_message_chars"] = len(message_value)
        reason = _string_value(payload.get("reason"))
        if reason:
            compact["reason"] = reason
        compact.update(_codex_error_evidence(payload))
    elif event_type == "thread_settings_applied":
        settings = _codex_settings_record(payload)
        for key in ("model", "model_name", "reasoning_effort", "effort", "personality"):
            value = _string_value(settings.get(key))
            if value:
                compact[key] = value
        collaboration_mode = _codex_collaboration_mode(settings.get("collaboration_mode"))
        if collaboration_mode:
            compact["collaboration_mode"] = collaboration_mode
    elif event_type in {
        "collab_agent_spawn_end",
        "collab_waiting_end",
        "collab_close_end",
        "collab_agent_interaction_end",
    }:
        for key in (
            "new_thread_id",
            "new_agent_nickname",
            "new_agent_role",
            "prompt",
            "receiver_thread_id",
            "receiver_agent_nickname",
            "receiver_agent_role",
            "status",
        ):
            value = _string_value(payload.get(key))
            if value:
                compact[key] = value
    elif event_type == "item_completed":
        item = _dict_record(payload.get("item"))
        if item:
            retained_item: dict[str, object] = {}
            item_type = _string_value(item.get("type"))
            # CommandExecution and FileChange are lowered to typed child
            # tool-result blocks. Their output/patch text must not be copied
            # into the generic event as a second public content route.
            item_keys: tuple[str, ...] = ("type", "id", "name", "path")
            if item_type not in _CODEX_EVENT_ITEM_DUPLICATE_TYPES:
                item_keys = (*item_keys, "text")
            for key in item_keys:
                item_value = item.get(key)
                if isinstance(item_value, (str, int, float, bool)):
                    retained_item[key] = item_value
            if retained_item:
                compact["item"] = retained_item
    elif event_type == "entered_review_mode":
        target = _dict_record(payload.get("target"))
        if target:
            compact["target"] = {
                key: target_value
                for key in ("instructions", "user_facing_hint")
                if isinstance((target_value := target.get(key)), str)
            }
    elif event_type == "exited_review_mode":
        review_output = _dict_record(payload.get("review_output"))
        if review_output:
            compact["review_output"] = {
                key: review_value
                for key in ("findings", "overall_correctness", "overall_explanation", "overall_confidence_score")
                if isinstance((review_value := review_output.get(key)), (str, int, float, bool, list))
            }
    elif event_type == "view_image_tool_call":
        path = _string_value(payload.get("path"))
        if path:
            compact["path"] = path
            compact["path_disclosure"] = "provider_reference"
            compact["acquired_bytes"] = False
    elif event_type == "web_search_end":
        query = _string_value(payload.get("query"))
        if query:
            compact["query"] = query
        action = _dict_record(payload.get("action"))
        queries = action.get("queries") if action else None
        if isinstance(queries, list):
            compact["action"] = {"queries": [query for query in queries if isinstance(query, str)]}
    elif event_type == "thread_rolled_back":
        num_turns = _optional_int_field(payload, "num_turns")
        if num_turns is not None:
            compact["num_turns"] = num_turns
    elif event_type == "error":
        compact.update(_codex_error_evidence(payload))
    elif event_type == "inter_agent_communication_metadata":
        # This is a top-level Codex record in newer exports. Retain the
        # observed orchestration fields explicitly; unknown nested metadata is
        # not promoted as a generic wire dump.
        for key in (
            "trigger_turn",
            "turn_id",
            "thread_id",
            "agent_thread_id",
            "agent_path",
            "agent_role",
            "agent_nickname",
            "sender_thread_id",
            "receiver_thread_id",
            "status",
            "kind",
            "message",
            "prompt",
        ):
            field_value = payload.get(key)
            if isinstance(field_value, (str, int, float, bool)):
                compact[key] = field_value
    elif event_type == "token_usage_record":
        # Newer exports wrap counters in ``usage``; older top-level records
        # place the same named counters directly beside ``type``. Both are
        # bounded numeric evidence, while opaque siblings remain excluded.
        usage = _dict_record(payload.get("usage")) or payload
        usage_payload = _codex_token_usage_payload(usage)
        if usage_payload:
            compact["usage"] = usage_payload
    if event_type == "user_message":
        local_images = _codex_source_references(payload.get("local_images"), field="local_images")
        if local_images:
            compact["local_images"] = local_images
        text_elements = _codex_text_element_references(payload.get("text_elements"))
        if text_elements:
            compact["text_elements"] = text_elements
    # Preserve observed lifecycle timing as scalar evidence without retaining a
    # provider envelope dump or guessing its units.
    for key in (
        "started_at",
        "start_time",
        "ended_at",
        "end_time",
        "completed_at",
        "completion_time",
        "duration_ms",
        "elapsed_ms",
        "elapsed_seconds",
    ):
        timing_value = payload.get(key)
        if isinstance(timing_value, (str, int, float)) and not isinstance(timing_value, bool):
            compact[key] = timing_value
    return compact


def _compact_response_payload(
    payload: dict[str, object],
    *,
    index: int,
    current_model_name: str | None = None,
    current_model_effort: str | None = None,
) -> dict[str, object]:
    compact: dict[str, object] = {"source_index": index}
    for key in ("type", "id", "call_id", "name", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            compact[key] = value
    timestamp = _record_timestamp(payload)
    if timestamp is not None:
        compact["timestamp"] = timestamp
    output = payload.get("output")
    if isinstance(output, str):
        compact["output_chars"] = len(output)
    elif output is not None:
        compact["has_output"] = True
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        compact["argument_chars"] = len(arguments)
    elif arguments is not None:
        compact["has_arguments"] = True
    cwd = _extract_cwd(payload)
    if cwd:
        compact["cwd"] = cwd
    compact.update(_codex_semantic_response_fields(payload))
    if compact.get("type") == "token_count":
        if current_model_name and not _string_field(compact, "model", "model_name"):
            compact["model"] = current_model_name
        if current_model_effort:
            compact["model_effort"] = current_model_effort
        info = _dict_record(payload.get("info")) or {}
        last_usage = _codex_token_usage_payload(_dict_record(info.get("last_token_usage")))
        total_usage = _codex_token_usage_payload(_dict_record(info.get("total_token_usage")))
        if not last_usage:
            last_usage = _codex_token_usage_payload(_dict_record(payload.get("last_token_usage")) or payload)
        if not total_usage:
            total_usage = _codex_token_usage_payload(_dict_record(payload.get("total_token_usage")))
        if last_usage:
            compact["last_token_usage"] = last_usage
        if total_usage:
            compact["total_token_usage"] = total_usage
        # Rate-limit windows are quota telemetry Codex reports alongside each
        # token_count tick -- small, bounded, and otherwise invisible.
        rate_limits = _dict_record(payload.get("rate_limits"))
        if rate_limits:
            compact_rate_limits: dict[str, object] = {}
            for lane in ("primary", "secondary"):
                window = _dict_record(rate_limits.get(lane))
                if not window:
                    continue
                lane_payload: dict[str, object] = {}
                used_percent = window.get("used_percent")
                if isinstance(used_percent, int | float) and not isinstance(used_percent, bool):
                    lane_payload["used_percent"] = used_percent
                for int_key in ("window_minutes", "resets_in_seconds"):
                    int_value = _optional_int_field(window, int_key)
                    if int_value is not None:
                        lane_payload[int_key] = int_value
                if lane_payload:
                    compact_rate_limits[lane] = lane_payload
            if compact_rate_limits:
                compact["rate_limits"] = compact_rate_limits
    elif compact.get("type") == "collab_agent_spawn_end":
        for key in ("new_thread_id", "new_agent_nickname", "new_agent_role"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                compact[key] = value
    elif compact.get("type") == "ghost_snapshot":
        # Codex's shadow-git snapshot (undo/diff tracking): the commit id,
        # its parent, and pre-existing untracked paths at snapshot time.
        ghost_commit = _dict_record(payload.get("ghost_commit"))
        if ghost_commit:
            compact["ghost_commit"] = dict(ghost_commit)
    elif compact.get("type") in {"exec_command_begin", "exec_command_end"}:
        # `aggregated_output`/`formatted_output`/`stdout`/`stderr` duplicate
        # text already captured verbatim as the paired function_call_output's
        # tool_result text (same call_id, same command output, just wrapped
        # with different transport metadata) -- deliberately not re-stored
        # here. `process_id` and `parsed_cmd` (Codex's own command
        # classification: search/read/etc with extracted query/path) are new.
        process_id = payload.get("process_id")
        if isinstance(process_id, str) and process_id:
            compact["process_id"] = process_id
        parsed_cmd = payload.get("parsed_cmd")
        if isinstance(parsed_cmd, list) and parsed_cmd:
            compact["parsed_cmd"] = parsed_cmd
    elif compact.get("type") in {"patch_apply_begin", "patch_apply_end"}:
        # `success`/`changes` were completely unread (polylogue-cgfy triage,
        # codex lane): the tool_use block on the paired function_call already
        # stores the *requested* patch text verbatim
        # (`_codex_tool_input`/`_PATCH_TOOL_NAMES`), and `stdout` duplicates a
        # human-readable summary already captured as the function_call_output
        # tool_result text -- same dedup rule as exec_command above. `changes`
        # is materially different from both: it is codex's own post-apply
        # per-file classification (add/update/delete + `unified_diff` +
        # `move_path` for renames), the structural equivalent of Claude
        # Code's `structuredPatch` (polylogue-cgfy). Nothing upstream
        # decomposes it, so it is retained verbatim, keyed by path, alongside
        # the boolean apply outcome.
        success = payload.get("success")
        if isinstance(success, bool):
            compact["success"] = success
        changes = _dict_record(payload.get("changes"))
        if changes:
            compact_changes: dict[str, object] = {}
            for changed_path, change in changes.items():
                change_record = _dict_record(change)
                if change_record is None:
                    continue
                entry: dict[str, object] = {}
                change_type = change_record.get("type")
                if isinstance(change_type, str) and change_type:
                    entry["type"] = change_type
                unified_diff = change_record.get("unified_diff")
                if isinstance(unified_diff, str) and unified_diff:
                    entry["unified_diff"] = unified_diff
                move_path = change_record.get("move_path")
                if isinstance(move_path, str) and move_path:
                    entry["move_path"] = move_path
                if entry:
                    compact_changes[str(changed_path)] = entry
            if compact_changes:
                compact["changes"] = compact_changes
    return compact


# response_item/event_msg inner ``type`` values that reach the generic
# session_event dispatch below (the `else` branch: not a message, not
# `compacted`/`turn_context`/`world_state`, no dedicated handler elsewhere in
# this file) with an explicit classification, rather than being stored
# verbatim under Codex's own wire name with no code aware of what that name
# means. Producer/consumer audit, polylogue-fuky (2026-08-02): every type
# below was previously an unaudited passthrough -- zero literal reference
# anywhere in the repo -- discovered via a live-archive `session_events`
# scan finding large row counts (318K-3K) under Codex wire names no code
# branches on. Each entry here has been read against raw wire samples (not
# inferred from row shape); this table does not change the *stored*
# `event_type` string for any of them -- every one still passes through
# under its own wire name (see ``_codex_response_item_event_type`` below).
# What changes is that a type NOT in this set (a future, never-examined
# Codex wire addition) no longer silently joins the same vocabulary
# unclassified -- it is routed to ``_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE``
# instead (fail loud/greppable, matching the
# ``_ATTACHMENT_UNCLASSIFIED_EVENT_TYPE`` precedent in
# ``sources/parsers/claude/code_parser.py``).
#
# TRANSIENT -- confirmed zero information content beyond the bare
# ``{"type": ...}`` marker already captured generically, read directly off
# raw wire samples:
#   context_compacted -- literal ``{"type": "context_compacted"}`` in every
#     sample read (live archive + source JSONL). Distinct from the
#     ``compacted`` record type handled explicitly above (~line 2148),
#     which DOES carry ``replacement_history`` -- this is a bare completion
#     marker for a separate/newer compaction notification path with
#     nothing else on the wire to capture.
#   agent_reasoning -- confirmed DUPLICATE, not merely transient: live-wire
#     comparison across three Codex sessions found `agent_reasoning.text`
#     values are the same live-streamed reasoning-summary bullets already
#     carried in full by the `reasoning` record's `summary[].text` (one
#     session: 156/156 identical; a second: 262/262 identical set; a third:
#     1,846 reasoning bullets vs. 1,859 agent_reasoning ticks, >99% overlap,
#     the residual being minor text-normalization differences on the same
#     underlying bullets, not new content). `reasoning` records are already
#     materialized as a THINKING-block ``ParsedMessage`` via
#     ``_codex_reasoning_message`` (index v50) -- `agent_reasoning` is a
#     live per-tick echo of that same content, matching the
#     "streaming ticks superseded by the final record" pattern documented
#     for Claude Code's `progress` subtypes in `claude/code_parser.py`. The
#     session_event this file emits for it is filtered back out at write
#     time (`_SESSION_EVENTS_REDUNDANT_TYPES` in
#     `storage/sqlite/archive_tiers/write.py`) rather than at parse time, to
#     keep this file's classification table describing what the WIRE means
#     (still a real, known type) separately from the WRITER's
#     zero-evidence-loss dedup decision. ``reasoning`` itself is NOT
#     reclassified here -- it already has a confirmed message consumer and
#     is out of scope for this audit.
#
# EVIDENCE, formerly under-captured by ``_compact_response_payload`` above,
# has a reviewed bounded extraction table in
# ``_codex_semantic_response_fields``. The generic lift still carries only
# identity/provenance fields; each semantic family below names its explicit
# destination and any duplicate-content exclusion. A future wire type remains
# fail-loud via ``_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE`` until it receives
# the same review.
#
# Rank this list against the SOURCE ROOT, never against live-archive row
# counts: the archive is a wire generation behind what Codex writes today, and
# a count taken from it understates a current record kind by orders of
# magnitude. ``item_completed`` was ranked negligible at 199 archive rows while
# the source root held roughly 315,000 (polylogue-vtyud).
#   thread_goal_updated (45,814 live rows) -- `goal.objective` (free text
#     session objective), `goal.status`/`tokensUsed`/`timeUsedSeconds`.
#   sub_agent_activity (41,769) -- `agent_thread_id`, `agent_path`, `kind`
#     (e.g. "interacted") -- subagent delegation evidence.
#   task_started (20,432) -- `turn_id`, `model_context_window`,
#     `collaboration_mode_kind`.
#   task_complete (17,055) -- `turn_id`, `last_agent_message`.
#   turn_aborted (3,394) -- `turn_id`, `reason` (e.g. "interrupted").
#   thread_settings_applied (3,548) -- full per-turn settings snapshot
#     (model, reasoning_effort, personality, collaboration_mode including
#     `developer_instructions` text) -- partially overlaps the
#     `turn_context` capture above (~line 2232) but is a distinct wire
#     record, not yet cross-checked for full redundancy.
#   collab_agent_spawn_end / collab_waiting_end / collab_close_end /
#     collab_agent_interaction_end (~1,000 combined) -- subagent-delegation
#     evidence (new_thread_id, new_agent_nickname, new_agent_role, prompt,
#     receiver_thread_id, receiver_agent_nickname/role, status text) beyond
#     the call_id/status the generic compactor already lifts.
#   item_completed -- `item.text` on a `Plan` item (full plan content) and the
#     `Extension`/`CollabAgentToolCall` item shapes. The `CommandExecution` and
#     `FileChange` shapes ARE read: they become code-mode child tool_results
#     with the producer's own exit code (see ``_codex_lookahead``).
#   entered_review_mode / exited_review_mode (13 each) --
#     `target.instructions`/`user_facing_hint` and
#     `review_output.findings`/`overall_correctness`/`overall_explanation`/
#     `overall_confidence_score`.
#   view_image_tool_call (62) -- `path` (the referenced image file).
#   web_search_end (1,357) -- `query`, `action.queries` -- a distinct
#     completion-marker wire shape from the already-handled
#     `web_search_call`/`web_search_output` pair (different call_id
#     namespace, `ws_...`); real search-query evidence, currently dropped.
#   thread_rolled_back (92) -- `num_turns` (rollback extent).
#   error (42) -- `message` (user-facing text, e.g. a usage-limit message)
#     and `codex_error_info` (a real error code, e.g.
#     "usage_limit_exceeded") -- operationally significant and currently
#     dropped entirely.
# Pre-existing types already dispatched/consumed elsewhere in this file
# (``_compact_response_payload``'s own elif chain above, ``_codex_tool_message``,
# ``_codex_event_message``, ``_codex_reasoning_message``,
# ``_codex_mcp_tool_call_messages``) -- unaffected by this audit, listed here
# only so the classifier below has a complete allowlist and none of them are
# ever misrouted into the unclassified bucket:
_CODEX_PRIOR_AUDITED_RESPONSE_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "token_count",
        "message_usage",
        "ghost_snapshot",
        "exec_command_begin",
        "exec_command_end",
        "patch_apply_begin",
        "patch_apply_end",
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search_call",
        "tool_search_output",
        "web_search_call",
        "web_search_output",
        "local_shell_call",
        "user_message",
        "agent_message",
        "mcp_tool_call_end",
    }
)

_CODEX_KNOWN_RESPONSE_ITEM_TYPES: frozenset[str] = _CODEX_PRIOR_AUDITED_RESPONSE_ITEM_TYPES | frozenset(
    {
        "context_compacted",
        "agent_reasoning",
        "thread_goal_updated",
        "sub_agent_activity",
        "task_started",
        "task_complete",
        "turn_aborted",
        "thread_settings_applied",
        "collab_agent_spawn_end",
        "collab_waiting_end",
        "collab_close_end",
        "collab_agent_interaction_end",
        "item_completed",
        "entered_review_mode",
        "exited_review_mode",
        "view_image_tool_call",
        "web_search_end",
        "thread_rolled_back",
        "error",
    }
)

# Fallback for a response_item/event_msg inner type not in the table above --
# e.g. a new Codex CLI version introducing a wire shape this repo has never
# read. FAIL LOUD: still persisted (never silently merged into the audited
# vocabulary above, which is exactly the "unaudited passthrough" defect this
# classification replaces), tagged with an event_type that is greppable/
# triageable on its own. The original wire type string is not lost -- it
# stays in the event payload's own ``type`` field (``_compact_response_payload``
# always lifts it when present).
_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE = "codex_unclassified_response_item"


# A session-level instruction text that changes mid-session. `user_instructions`
# and `developer_instructions` are re-declared on every ``turn_context``, so the
# first value fills the session's own slot (``instructions_text`` /
# ``codex_agent_identity``) and only a value distinct from every one seen before
# lands here. The payload key is ``instructions`` rather than ``text``, which the
# writer would also copy into the event's ``summary`` column.
_CODEX_INSTRUCTIONS_CHANGED_EVENT_TYPE = "codex_instructions_changed"


def _codex_instructions_changed_event(
    *,
    kind: str,
    instructions: str,
    revision: int,
    timestamp: str | None,
    source_index: int,
    effective_from_message_position: int,
) -> ParsedSessionEvent:
    """One newly observed distinct value of a session-level instruction text.

    ``effective_from_message_position`` is the next message's position, so the
    writer resolves ``boundary_message_id`` to the first message the new
    instructions applied to -- and applies the lineage position offset a raw
    position carried in the payload would not get. It stays NULL when no
    message follows the change.
    """
    return ParsedSessionEvent(
        event_type=_CODEX_INSTRUCTIONS_CHANGED_EVENT_TYPE,
        timestamp=timestamp,
        payload={
            "source_index": source_index,
            "instructions_kind": kind,
            "instructions": instructions,
            "revision": revision,
        },
        boundary_message_position=effective_from_message_position,
    )


# Text a compaction's ``replacement_history`` or a ``task_complete`` record
# carries that the parsed session holds nowhere else. Codex re-embeds the
# pre-compaction records in ``replacement_history``; measured over the 131
# rollout files carrying it in a 596-file sample (195,851 text values), 97.8%
# are already stored from the same file's live stream and 0.98% from an
# ancestor session whose prefix this file replays and which is ingested
# separately. The remaining 1.2% -- 2,351 occurrences collapsing to 397
# distinct values -- exists only here: turn-construction context Codex writes
# down nowhere else (``<environment_context>``, ``<skills_instructions>``,
# injected AGENTS.md text) and real user turns. Only those land as their own
# event. The payload key is ``content`` because the writer copies ``text`` and
# ``summary`` into the event's ``summary`` column.
_CODEX_REPLACEMENT_CONTEXT_EVENT_TYPE = "codex_replacement_context"


class _CodexInstructionRevisions:
    """Distinct instruction revisions, in first-seen order.

    ``turn_context`` re-declares the session prompt on every turn, so a
    rollout with T turns over R distinct revisions asks "have I seen this
    one?" T times.  A list scan makes that O(T*R) string comparisons over
    values that are routinely tens of kilobytes (AGENTS.md-style prompts);
    membership uses a set for direct object parsing and a scratch SQLite
    index for streamed parsing. The revision number is its first-seen rank.
    """

    __slots__ = ("_order", "_seen", "_index", "_kind")

    def __init__(self, index: _CodexLookaheadIndex | None = None, kind: str = "") -> None:
        self._order: list[str] = []
        self._seen: set[str] = set()
        self._index = index
        self._kind = kind

    def __contains__(self, value: object) -> bool:
        if self._index is not None:
            if not isinstance(value, str):
                return False
            return (
                self._index.connection.execute(
                    "SELECT 1 FROM codex_instruction_revisions WHERE kind = ? AND key = ?",
                    (self._kind, _text_digest(value)),
                ).fetchone()
                is not None
            )
        return value in self._seen

    def __len__(self) -> int:
        if self._index is not None:
            row = self._index.connection.execute(
                "SELECT count(*) FROM codex_instruction_revisions WHERE kind = ?", (self._kind,)
            ).fetchone()
            assert row is not None
            return int(row[0])
        return len(self._order)

    def add(self, value: str) -> None:
        if self._index is not None:
            self._index.connection.execute(
                "INSERT OR IGNORE INTO codex_instruction_revisions VALUES (?, ?, ?)",
                (self._kind, _text_digest(value), _sql_key(value)),
            )
            return
        self._order.append(value)
        self._seen.add(value)

    def values(self) -> tuple[str, ...]:
        if self._index is not None:
            return tuple(
                pickle.loads(value)
                for (value,) in self._index.connection.execute(
                    "SELECT value FROM codex_instruction_revisions WHERE kind = ? ORDER BY rowid", (self._kind,)
                )
            )
        return tuple(self._order)


class _CodexTextConservation:
    """Decides which candidate texts a parsed session does not already hold.

    Candidates are registered while parsing and resolved in one pass at the
    end, against every text the parser emits -- message text, block text, tool
    inputs, session-event payloads, the session's instructions. Candidates live
    in the parse's scratch index, so resolution costs one walk over the
    retained content and no memory proportional to the candidates.

    Resolution must run after the last event is appended: ``replacement_history``
    re-embeds records that may be parsed either before or after the compaction
    that carries them.
    """

    def __init__(self, index: _CodexLookaheadIndex) -> None:
        self._unresolved = 0
        self._index = index
        self._task_unresolved = 0
        self._contexts = 0

    def _lookup(self, keys_table: str, text: str, *, normalize: bool) -> bytes | None:
        """The candidate key registered for ``text`` (or its NFC form).

        The key is the SHA-256 of the exact code points, so a match is the
        text itself; the stored text is never read back to confirm it.
        """
        connection = self._index.connection
        query = f"SELECT candidate_key FROM {keys_table} WHERE value = ?"
        row = connection.execute(query, (_text_digest(text),)).fetchone()
        # The NFC copy is built only when the exact probe misses.
        if row is None and normalize and not text.isascii():
            row = connection.execute(query, (_nfc_text_digest(text),)).fetchone()
        return bytes(row[0]) if row is not None else None

    def _candidate(self, text: str, *, normalize: bool = True) -> bytes | None:
        return self._lookup("codex_replacement_keys", text, normalize=normalize)

    def _task_lookup(self, text: str, *, normalize: bool) -> bytes | None:
        return self._lookup("codex_task_keys", text, normalize=normalize)

    def add_task_completion(self, text: str, event_index: int) -> None:
        key = self._task_lookup(text, normalize=False)
        connection = self._index.connection
        if key is None:
            key = _text_digest(text)
            connection.execute("INSERT INTO codex_task_texts(key, text) VALUES (?, ?)", (key, _sql_key(text)))
            connection.execute("INSERT INTO codex_task_keys VALUES (?, ?)", (key, key))
            normalized_key = _nfc_text_digest(text)
            if normalized_key != key:
                connection.execute("INSERT OR IGNORE INTO codex_task_keys VALUES (?, ?)", (normalized_key, key))
            self._task_unresolved += 1
        connection.execute("INSERT INTO codex_task_events VALUES (?, ?, ?)", (event_index, key, len(text)))

    def finish_task_completions(self, events: MutableSequence[ParsedSessionEvent]) -> None:
        connection = self._index.connection
        for event_index, key, text_chars in connection.execute(
            "SELECT event_index, key, text_chars FROM codex_task_events ORDER BY event_index"
        ):
            row = connection.execute(
                "SELECT text, retained, stored FROM codex_task_texts WHERE key = ?", (key,)
            ).fetchone()
            assert row is not None
            text = pickle.loads(row[0])
            retained, stored = row[1], row[2]
            candidate_key = self._candidate(text, normalize=False)
            candidate = (
                connection.execute(
                    "SELECT retained, stored FROM codex_replacement_texts WHERE key = ?", (candidate_key,)
                ).fetchone()
                if candidate_key is not None
                else None
            )
            event = events[event_index]
            event.payload["last_agent_message_chars"] = text_chars
            if retained or stored or (candidate is not None and (candidate[0] or candidate[1])):
                event.payload["last_agent_message_retained"] = True
            else:
                event.payload["last_agent_message"] = text
                connection.execute("UPDATE codex_task_texts SET stored = 1 WHERE key = ?", (key,))
                if candidate_key is not None:
                    connection.execute("UPDATE codex_replacement_texts SET stored = 1 WHERE key = ?", (candidate_key,))
            events[event_index] = event

    def add(self, text: str) -> tuple[bytes | None, bool]:
        """Register ``text``; returns its candidate key and whether it is new.

        A value already registered only gains an occurrence. A value the
        session keeps as a task-completion text has no candidate of its own.
        """
        connection = self._index.connection
        existing = self._candidate(text, normalize=False)
        if existing is not None:
            connection.execute(
                "UPDATE codex_replacement_texts SET occurrences = occurrences + 1 WHERE key = ?", (existing,)
            )
            return existing, False
        if self._task_lookup(text, normalize=False) is not None:
            return None, False
        # The text is held once, in ``codex_replacement_texts.text``; every
        # other scratch column carries its fixed-size digest.
        key = _text_digest(text)
        connection.execute(
            "INSERT INTO codex_replacement_texts(key, text, text_chars) VALUES (?, ?, ?)",
            (key, _sql_key(text), len(text)),
        )
        connection.execute("INSERT INTO codex_replacement_keys VALUES (?, ?)", (key, key))
        # A value normalized differently on the two sides would otherwise read
        # as absent and be stored a second time. Only the candidates are
        # normalized; retained text is looked up as written, then normalized
        # only when it is not pure ASCII.
        normalized_key = _nfc_text_digest(text)
        if normalized_key != key:
            connection.execute("INSERT OR IGNORE INTO codex_replacement_keys VALUES (?, ?)", (normalized_key, key))
        self._unresolved += 1
        return key, True

    def add_context(
        self,
        key: bytes,
        *,
        insert_at: int,
        timestamp: str | None,
        source_index: int,
        entry_type: str | None,
        role: str | None,
        phase: str | None,
    ) -> None:
        """Record where a new candidate is stored if the session keeps it nowhere else."""
        self._index.connection.execute(
            "INSERT INTO codex_replacement_contexts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            # Annotations may hold a lone surrogate the scratch TEXT binding
            # cannot encode; they are held pickled and decoded on read.
            (
                self._contexts,
                insert_at,
                key,
                timestamp,
                source_index,
                _sql_key(entry_type),
                _sql_key(role),
                _sql_key(phase),
            ),
        )
        self._contexts += 1

    def finish_replacement_contexts(self, events: MutableSequence[ParsedSessionEvent]) -> None:
        """Splice each unretained candidate's event in after its compaction, once."""
        if not self._contexts:
            return
        connection = self._index.connection
        stored = """
            FROM codex_replacement_contexts AS context
            JOIN codex_replacement_texts AS text ON text.key = context.key
            WHERE text.retained = 0 AND text.stored = 0
        """
        # Iterated, never fetched whole: a rollout with one distinct value per
        # compaction yields one row per compaction here.
        for insert_at, count in connection.execute(
            f"SELECT context.insert_at, COUNT(*) {stored} GROUP BY context.insert_at"
        ):
            compaction = events[insert_at - 1]
            previous = compaction.payload.get("replacement_history_context_count")
            compaction.payload["replacement_history_context_count"] = (
                previous if isinstance(previous, int) else 0
            ) + count
            events[insert_at - 1] = compaction
        rows = connection.execute(
            "SELECT context.insert_at, context.timestamp, context.source_index, context.entry_type, "
            f"context.role, context.phase, text.text, text.occurrences {stored} "
            "ORDER BY context.insert_at, context.ordinal"
        )

        def insertions() -> Iterator[tuple[int, ParsedSessionEvent]]:
            for insert_at, timestamp, source_index, entry_blob, role_blob, phase_blob, text, occurrences in rows:
                entry_type, role, phase = (pickle.loads(blob) for blob in (entry_blob, role_blob, phase_blob))
                content = pickle.loads(text)
                # Release the pickled scratch value before the event is
                # encoded: only the decoded text and its JSON row coexist.
                del text
                payload: dict[str, object] = {
                    "source_index": source_index,
                    "context_kind": "replacement_history",
                    "content": content,
                    "content_chars": len(content),
                    "occurrences": occurrences,
                }
                if entry_type:
                    payload["entry_type"] = entry_type
                if role:
                    payload["role"] = role
                if phase:
                    payload["phase"] = phase
                yield (
                    insert_at,
                    ParsedSessionEvent(
                        event_type=_CODEX_REPLACEMENT_CONTEXT_EVENT_TYPE, timestamp=timestamp, payload=payload
                    ),
                )

        # One renumbering pass: inserting each context separately would shift
        # every later event once per context.
        insert_sorted = getattr(events, "insert_sorted", None)
        if insert_sorted is not None:
            insert_sorted(insertions())
            return
        pending = list(insertions())
        merged: list[ParsedSessionEvent] = []
        cursor = 0
        for index, event in enumerate(events):
            while cursor < len(pending) and pending[cursor][0] <= index:
                merged.append(pending[cursor][1])
                cursor += 1
            merged.append(event)
        merged.extend(event for _index, event in pending[cursor:])
        events[:] = merged

    def _mark(self, text: str) -> None:
        if not text or not (self._unresolved or self._task_unresolved):
            return
        if self._unresolved:
            candidate_key = self._candidate(text)
            if candidate_key is not None:
                updated = self._index.connection.execute(
                    "UPDATE codex_replacement_texts SET retained = 1 WHERE key = ? AND retained = 0", (candidate_key,)
                )
                self._unresolved -= updated.rowcount
        if self._task_unresolved:
            task_key = self._task_lookup(text, normalize=True)
            if task_key is not None:
                updated = self._index.connection.execute(
                    "UPDATE codex_task_texts SET retained = 1 WHERE key = ? AND retained = 0", (task_key,)
                )
                self._task_unresolved -= updated.rowcount

    def _mark_nested(self, value: object) -> None:
        if isinstance(value, str):
            self._mark(value)
        elif isinstance(value, dict):
            for item in value.values():
                self._mark_nested(item)
        elif isinstance(value, list | tuple):
            for item in value:
                self._mark_nested(item)

    def resolve(
        self,
        *,
        messages: Iterable[ParsedMessage],
        events: Iterable[ParsedSessionEvent],
        instructions_text: str | None,
    ) -> None:
        if not (self._unresolved or self._task_unresolved):
            return
        if instructions_text:
            self._mark(instructions_text)
        for message in messages:
            if message.text:
                self._mark(message.text)
            for block in message.blocks:
                if block.text:
                    self._mark(block.text)
                if block.tool_input:
                    self._mark_nested(dict(block.tool_input))
            if not (self._unresolved or self._task_unresolved):
                return
        for event in events:
            self._mark_nested(event.payload)
            if not (self._unresolved or self._task_unresolved):
                return


def _codex_response_item_event_type(inner_type: str | None, record_type: str | None) -> str:
    """Classify a response_item/event_msg inner ``type`` for ``session_events``.

    Every type in ``_CODEX_KNOWN_RESPONSE_ITEM_TYPES`` has been read against
    raw wire samples and passes through under its own Codex wire name
    unchanged. A type this repo has never examined is routed to
    ``_CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE`` instead of silently adopting
    its own wire name, so an unaudited type can never again commingle with
    the audited vocabulary above without a human noticing the greppable
    marker.
    """
    resolved = inner_type or record_type
    if resolved is None:
        return "response_item"
    if resolved in _CODEX_KNOWN_RESPONSE_ITEM_TYPES:
        return resolved
    return _CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE


def _extract_cwd(payload: dict[str, object] | None) -> str | None:
    if not payload:
        return None
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return cwd.strip()
    turn_context = payload.get("turn_context")
    if isinstance(turn_context, dict):
        nested = turn_context.get("cwd")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _is_js_identifier_start(char: str) -> bool:
    return char == "_" or char == "$" or char.isalpha()


def _is_js_identifier_part(char: str) -> bool:
    return _is_js_identifier_start(char) or char.isdigit()


class _JsLiteralParser:
    """Conservative parser for the JSON-like argument literals used by Code Mode.

    This deliberately accepts only literals. Expressions, interpolation, spreads,
    and references stay as raw evidence instead of being evaluated or guessed.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> object:
        self._skip_space_and_comments()
        value = self._parse_value()
        self._skip_space_and_comments()
        if self.position != len(self.text):
            raise _JsLiteralError("trailing JavaScript expression")
        return value

    def _peek(self) -> str | None:
        if self.position >= len(self.text):
            return None
        return self.text[self.position]

    def _skip_space_and_comments(self) -> None:
        while self.position < len(self.text):
            char = self.text[self.position]
            if char.isspace():
                self.position += 1
                continue
            if self.text.startswith("//", self.position):
                newline = self.text.find("\n", self.position + 2)
                self.position = len(self.text) if newline == -1 else newline + 1
                continue
            if self.text.startswith("/*", self.position):
                end = self.text.find("*/", self.position + 2)
                if end == -1:
                    raise _JsLiteralError("unterminated JavaScript comment")
                self.position = end + 2
                continue
            return

    def _parse_value(self) -> object:
        char = self._peek()
        if char is None:
            raise _JsLiteralError("missing JavaScript literal")
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char in {'"', "'", "`"}:
            return self._parse_string()
        if char == "-" or char.isdigit():
            return self._parse_number()
        if _is_js_identifier_start(char):
            identifier = self._parse_identifier()
            if identifier == "true":
                return True
            if identifier == "false":
                return False
            if identifier in {"null", "undefined"}:
                return None
            raise _JsLiteralError(f"non-literal JavaScript identifier: {identifier}")
        raise _JsLiteralError(f"unsupported JavaScript literal token: {char}")

    def _parse_object(self) -> dict[str, object]:
        self.position += 1
        result: dict[str, object] = {}
        self._skip_space_and_comments()
        if self._peek() == "}":
            self.position += 1
            return result
        while True:
            self._skip_space_and_comments()
            key_char = self._peek()
            if key_char in {'"', "'", "`"}:
                key_value = self._parse_string()
                if not isinstance(key_value, str):
                    raise _JsLiteralError("object key is not text")
                key = key_value
            elif key_char is not None and _is_js_identifier_start(key_char):
                key = self._parse_identifier()
            else:
                raise _JsLiteralError("unsupported JavaScript object key")
            self._skip_space_and_comments()
            if self._peek() != ":":
                raise _JsLiteralError("JavaScript object shorthand is not a literal")
            self.position += 1
            self._skip_space_and_comments()
            result[key] = self._parse_value()
            self._skip_space_and_comments()
            delimiter = self._peek()
            if delimiter == "}":
                self.position += 1
                return result
            if delimiter != ",":
                raise _JsLiteralError("missing JavaScript object delimiter")
            self.position += 1
            self._skip_space_and_comments()
            if self._peek() == "}":
                self.position += 1
                return result

    def _parse_array(self) -> list[object]:
        self.position += 1
        result: list[object] = []
        self._skip_space_and_comments()
        if self._peek() == "]":
            self.position += 1
            return result
        while True:
            self._skip_space_and_comments()
            result.append(self._parse_value())
            self._skip_space_and_comments()
            delimiter = self._peek()
            if delimiter == "]":
                self.position += 1
                return result
            if delimiter != ",":
                raise _JsLiteralError("missing JavaScript array delimiter")
            self.position += 1
            self._skip_space_and_comments()
            if self._peek() == "]":
                self.position += 1
                return result

    def _parse_identifier(self) -> str:
        start = self.position
        if self.position >= len(self.text) or not _is_js_identifier_start(self.text[self.position]):
            raise _JsLiteralError("missing JavaScript identifier")
        self.position += 1
        while self.position < len(self.text) and _is_js_identifier_part(self.text[self.position]):
            self.position += 1
        return self.text[start : self.position]

    def _parse_string(self) -> str:
        """One JavaScript string literal, consumed in runs, not character by character.

        JavaScript strings are UTF-16: an escaped high/low pair such as
        \\uD83D\\uDE00 is one character. A pair is combined as its low half is
        consumed, so the value equals what any JSON reader of the stored,
        escaped payload decodes without a second pass over the whole literal;
        a lone surrogate stays as it is.
        """
        quote = self.text[self.position]
        self.position += 1
        plain = _JS_PLAIN_RUNS[quote]
        parts: list[str] = []

        def push(piece: str) -> None:
            if _JS_SURROGATE.search(piece) is None:
                parts.append(piece)
                return
            # Surrogates stand alone in ``parts``, so the one before a low
            # half is exactly ``parts[-1]``.
            last = 0
            for match in _JS_SURROGATE.finditer(piece):
                if match.start() > last:
                    parts.append(piece[last : match.start()])
                unit = ord(match.group())
                if 0xDC00 <= unit <= 0xDFFF and parts and len(parts[-1]) == 1 and 0xD800 <= ord(parts[-1]) <= 0xDBFF:
                    parts[-1] = chr(0x10000 + ((ord(parts[-1]) - 0xD800) << 10) + (unit - 0xDC00))
                else:
                    parts.append(match.group())
                last = match.end()
            if last < len(piece):
                parts.append(piece[last:])

        while self.position < len(self.text):
            run = plain.match(self.text, self.position)
            if run is not None:
                push(run.group())
                self.position = run.end()
                continue
            char = self.text[self.position]
            self.position += 1
            if char == quote:
                return "".join(parts)
            if char == "$":
                # Only reached inside a template literal.
                if self._peek() == "{":
                    raise _JsLiteralError("template interpolation is not a literal")
                push(char)
                continue
            if self.position >= len(self.text):
                raise _JsLiteralError("unterminated JavaScript string escape")
            escaped = self.text[self.position]
            self.position += 1
            escapes = {
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "v": "\v",
                "0": "\0",
                "\\": "\\",
                "'": "'",
                '"': '"',
                "`": "`",
            }
            if escaped in escapes:
                push(escapes[escaped])
                continue
            if escaped in {"\n", "\r"}:
                if escaped == "\r" and self._peek() == "\n":
                    self.position += 1
                continue
            if escaped == "x":
                push(self._parse_hex_escape(2))
                continue
            if escaped == "u":
                if self._peek() == "{":
                    self.position += 1
                    end = self.text.find("}", self.position)
                    if end == -1:
                        raise _JsLiteralError("unterminated JavaScript Unicode escape")
                    token = self.text[self.position : end]
                    self.position = end + 1
                    try:
                        push(chr(int(token, 16)))
                    except (ValueError, OverflowError) as exc:
                        raise _JsLiteralError("invalid JavaScript Unicode escape") from exc
                else:
                    push(self._parse_hex_escape(4))
                continue
            # JavaScript treats an otherwise-unknown escaped character as the
            # character itself. Preserving it is safer than rejecting evidence.
            push(escaped)
        raise _JsLiteralError("unterminated JavaScript string")

    def _parse_hex_escape(self, width: int) -> str:
        token = self.text[self.position : self.position + width]
        if len(token) != width or any(char not in "0123456789abcdefABCDEF" for char in token):
            raise _JsLiteralError("invalid JavaScript hexadecimal escape")
        self.position += width
        return chr(int(token, 16))

    def _parse_number(self) -> int | float:
        match = re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", self.text[self.position :])
        if match is None:
            raise _JsLiteralError("invalid JavaScript number")
        token = match.group(0)
        self.position += len(token)
        return float(token) if any(char in token for char in ".eE") else int(token)


def _parse_js_literal(text: str) -> tuple[object, bool]:
    candidate = text.strip()
    if not candidate:
        return None, True
    try:
        return json.loads(candidate), True
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return _JsLiteralParser(candidate).parse(), True
    except _JsLiteralError:
        return candidate, False


@dataclass(slots=True)
class _JsScanRefusals:
    """Counted refusals from one JavaScript source scan.

    A ``/`` whose regex-literal-vs-division reading is not decidable by the
    previous-significant-token rule is refused and counted here rather than
    guessed. A refused slash is never consumed, so the scan continues past it
    instead of swallowing the remainder of the program.
    """

    unresolvable_slash: int = 0
    unterminated_regex: int = 0

    @property
    def total(self) -> int:
        return self.unresolvable_slash + self.unterminated_regex


#: Tokens after which a ``/`` begins a regular-expression literal rather than a
#: division. This is the standard practical disambiguation (the same rule real
#: JavaScript lexers use before parsing): after an operator, an opening
#: bracket, a separator or a statement keyword the next ``/`` cannot be a
#: division because there is no left operand.
_JS_REGEX_PRECEDING_PUNCTUATION = frozenset("(,=:[!&|?{};+-*%<>^~")
_JS_REGEX_PRECEDING_KEYWORDS = frozenset(
    {
        "return",
        "typeof",
        "instanceof",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "throw",
        "case",
        "do",
        "else",
        "yield",
        "await",
    }
)


def _previous_js_significant_token(source: str, position: int) -> str | None:
    """The last significant token before ``position``, or ``None`` at the start.

    Returns the literal sentinel ``"*/"`` when the preceding token is a block
    comment terminator: the token before that comment is not recoverable by a
    backward scan, so the caller refuses instead of guessing.
    """
    index = position - 1
    while index >= 0 and source[index].isspace():
        index -= 1
    if index < 0:
        return None
    if index >= 1 and source[index - 1 : index + 1] == "*/":
        return "*/"
    char = source[index]
    if _is_js_identifier_part(char):
        end = index + 1
        while index >= 0 and _is_js_identifier_part(source[index]):
            index -= 1
        return source[index + 1 : end]
    if char in "+-" and index >= 1 and source[index - 1] == char:
        # ``x++`` / ``y--`` are postfix updates, so the value is on the left.
        return char * 2
    return char


def _skip_js_regex_literal(source: str, position: int) -> int | None:
    """Skip a regex literal starting at ``source[position] == '/'``.

    Handles escapes and character classes, inside which ``/`` is an ordinary
    character. Returns ``None`` for an unterminated literal (a newline or the
    end of input before the closing delimiter).
    """
    index = position + 1
    in_class = False
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == "\n":
            return None
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        elif char == "/" and not in_class:
            index += 1
            while index < len(source) and _is_js_identifier_part(source[index]):
                index += 1
            return index
        index += 1
    return None


#: Where a JS scan must stop and do per-position work (polylogue-s8x8s).
#:
#: The scanners below used to advance one character at a time, asking
#: ``_skip_js_string_or_comment`` at every position in the source. Everything
#: between two interesting characters -- whitespace, operators, digits,
#: ordinary punctuation -- was a Python loop iteration that could only ever
#: decline. Jumping to the next stop with ``re.Pattern.search`` does that
#: skipping in C.
#:
#: Each pattern must be a SUPERSET of its scanner's interesting set:
#: over-matching costs one extra no-op iteration, under-matching silently
#: skips a token. ``_skip_js_string_or_comment`` returns a position only for
#: ``/``, ``"``, ``'`` and a backtick, so every pattern carries those four.
_JS_CALL_SCAN_STOP = re.compile(r"""[/"'`()]""")
_JS_ARGUMENT_SCAN_STOP = re.compile(r"""[/"'`()\[\]{},]""")
#: ``[^\W\d]`` is a superset of ``str.isalpha()``: an alphabetic character is
#: alphanumeric, so it is in ``\w``, and none is in ``\d``. It also covers the
#: ``_`` of ``_is_js_identifier_start``; ``$`` is the remaining case.
_JS_IDENTIFIER_SCAN_STOP = re.compile(r"""[/"'`$]|[^\W\d]""")


def _next_js_scan_stop(pattern: re.Pattern[str], source: str, position: int) -> int:
    """First index at or after ``position`` where ``pattern``'s scanner must look."""
    match = pattern.search(source, position)
    return len(source) if match is None else match.start()


def _skip_js_string_or_comment(
    source: str,
    position: int,
    *,
    refusals: _JsScanRefusals | None = None,
) -> int | None:
    if source.startswith("//", position):
        newline = source.find("\n", position + 2)
        return len(source) if newline == -1 else newline + 1
    if source.startswith("/*", position):
        end = source.find("*/", position + 2)
        return len(source) if end == -1 else end + 2
    if position < len(source) and source[position] == "/":
        previous = _previous_js_significant_token(source, position)
        if previous == "*/":
            if refusals is not None:
                refusals.unresolvable_slash += 1
            return None
        starts_regex = (
            previous is None
            or (len(previous) == 1 and previous in _JS_REGEX_PRECEDING_PUNCTUATION)
            or previous in _JS_REGEX_PRECEDING_KEYWORDS
        )
        if not starts_regex:
            return None
        regex_end = _skip_js_regex_literal(source, position)
        if regex_end is None:
            if refusals is not None:
                refusals.unterminated_regex += 1
            return None
        return regex_end
    if position >= len(source) or source[position] not in {'"', "'", "`"}:
        return None
    quote = source[position]
    position += 1
    # ``str.find`` runs the scan in C. The Python loop it replaces cost one
    # iteration per character of the literal; this one costs an iteration per
    # *escape*, and a literal with no escapes -- the overwhelmingly common
    # case -- resolves in two finds. Same acceptance: an unterminated literal
    # still consumes the rest of the input, and a trailing backslash still
    # swallows the byte after it and then the end of input.
    length = len(source)
    while position < length:
        closing = source.find(quote, position)
        escape = source.find("\\", position)
        if closing == -1:
            return length
        if escape == -1 or escape > closing:
            return closing + 1
        position = escape + 2
    return length


def _skip_js_space_and_comments(source: str, position: int) -> int:
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        skipped = _skip_js_string_or_comment(source, position)
        if skipped is not None and source.startswith(("//", "/*"), position):
            position = skipped
            continue
        return position
    return position


def _parse_js_member_chain(source: str, position: int) -> tuple[tuple[str, ...], int] | None:
    if position >= len(source) or not _is_js_identifier_start(source[position]):
        return None
    start = position
    position += 1
    while position < len(source) and _is_js_identifier_part(source[position]):
        position += 1
    parts = [source[start:position]]
    while True:
        position = _skip_js_space_and_comments(source, position)
        if position < len(source) and source[position] == ".":
            position = _skip_js_space_and_comments(source, position + 1)
            if position >= len(source) or not _is_js_identifier_start(source[position]):
                return tuple(parts), position
            start = position
            position += 1
            while position < len(source) and _is_js_identifier_part(source[position]):
                position += 1
            parts.append(source[start:position])
            continue
        if position < len(source) and source[position] == "[":
            member_start = _skip_js_space_and_comments(source, position + 1)
            if member_start >= len(source) or source[member_start] not in {'"', "'", "`"}:
                return tuple(parts), position
            parser = _JsLiteralParser(source[member_start:])
            try:
                member = parser._parse_string()
            except _JsLiteralError:
                return tuple(parts), position
            member_end = member_start + parser.position
            member_end = _skip_js_space_and_comments(source, member_end)
            if member_end >= len(source) or source[member_end] != "]":
                return tuple(parts), position
            parts.append(member)
            position = member_end + 1
            continue
        return tuple(parts), position


def _balanced_js_call_argument(
    source: str,
    open_position: int,
    *,
    refusals: _JsScanRefusals | None = None,
) -> tuple[str, int, bool]:
    depth = 1
    position = open_position + 1
    argument_start = position
    while position < len(source):
        skipped = _skip_js_string_or_comment(source, position, refusals=refusals)
        if skipped is not None:
            position = skipped
            continue
        char = source[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[argument_start:position], position + 1, True
        position = _next_js_scan_stop(_JS_CALL_SCAN_STOP, source, position + 1)
    return source[argument_start:], len(source), False


def _first_js_argument(arguments: str, *, refusals: _JsScanRefusals | None = None) -> str:
    depths = {"(": 0, "[": 0, "{": 0}
    closers = {")": "(", "]": "[", "}": "{"}
    position = 0
    while position < len(arguments):
        skipped = _skip_js_string_or_comment(arguments, position, refusals=refusals)
        if skipped is not None:
            position = skipped
            continue
        char = arguments[position]
        if char in depths:
            depths[char] += 1
        elif char in closers:
            opener = closers[char]
            depths[opener] = max(depths[opener] - 1, 0)
        elif char == "," and all(depth == 0 for depth in depths.values()):
            return arguments[:position]
        position = _next_js_scan_stop(_JS_ARGUMENT_SCAN_STOP, arguments, position + 1)
    return arguments


def _classify_code_mode_child(tool_path: tuple[str, ...]) -> str:
    normalized = tuple(part.strip().lower().replace("-", "_") for part in tool_path if part.strip())
    child_parts = normalized[1:] if normalized and normalized[0] in {"tools", "functions"} else normalized
    if not child_parts:
        return "unknown"

    # Namespaced tools are MCP delegations unless the namespace itself is a
    # first-class Codex family. This check intentionally precedes leaf aliases:
    # tools.mcp.repo.search is MCP, not a generic web search.
    if child_parts[0] == "web":
        return "web"
    if child_parts[0] == "image":
        return "image"
    if any("mcp" in part for part in child_parts) or len(child_parts) > 1:
        return "mcp"

    leaf = child_parts[-1]
    for spec in _CODE_MODE_CHILD_REGISTRY:
        if leaf in spec.aliases:
            return spec.kind
    return "unknown"


def _code_mode_tool_name(tool_path: tuple[str, ...], registry_type: str) -> str:
    child_parts = tool_path[1:] if tool_path and tool_path[0].lower() in {"tools", "functions"} else tool_path
    raw_name = ".".join(child_parts) if child_parts else "unknown"
    # First-class registry entries use stable names so downstream semantic
    # normalization does not depend on provider spelling. MCP and unknown
    # calls keep their exact names; their registry type remains in provenance.
    return registry_type if registry_type not in {"mcp", "unknown"} else raw_name


def _scan_code_mode_child_calls(
    source: str,
    *,
    refusals: _JsScanRefusals | None = None,
) -> tuple[_CodexExecChildCall, ...]:
    calls: list[_CodexExecChildCall] = []
    position = 0
    while position < len(source):
        skipped = _skip_js_string_or_comment(source, position, refusals=refusals)
        if skipped is not None:
            position = skipped
            continue
        if not _is_js_identifier_start(source[position]):
            position = _next_js_scan_stop(_JS_IDENTIFIER_SCAN_STOP, source, position + 1)
            continue
        parsed_chain = _parse_js_member_chain(source, position)
        if parsed_chain is None:
            position += 1
            continue
        tool_path, after_chain = parsed_chain
        after_chain = _skip_js_space_and_comments(source, after_chain)
        if not tool_path or tool_path[0].lower() not in {"tools", "functions"}:
            position = max(after_chain, position + 1)
            continue
        if len(tool_path) < 2 or after_chain >= len(source) or source[after_chain] != "(":
            position = max(after_chain, position + 1)
            continue
        raw_arguments, call_end, balanced = _balanced_js_call_argument(source, after_chain, refusals=refusals)
        first_argument = _first_js_argument(raw_arguments, refusals=refusals)
        argument, parsed = _parse_js_literal(first_argument)
        registry_type = _classify_code_mode_child(tool_path)
        calls.append(
            _CodexExecChildCall(
                tool_path=tool_path,
                tool_name=_code_mode_tool_name(tool_path, registry_type),
                registry_type=registry_type,
                argument=argument,
                raw_argument=first_argument.strip() or None,
                parse_state="parsed" if parsed and balanced else "malformed",
                source_start=position,
                source_end=call_end,
            )
        )
        position = max(call_end, position + 1)
    return tuple(calls)


def _mapping_string(record: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _structured_code_mode_child_calls(value: object) -> tuple[_CodexExecChildCall, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()
    if not isinstance(value, dict):
        return ()
    child_items: list[object] | None = None
    for key in _CODE_MODE_CHILD_COLLECTION_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, list):
            child_items = candidate
            break
    if child_items is None:
        return ()
    calls: list[_CodexExecChildCall] = []
    for item in child_items:
        if not isinstance(item, dict):
            calls.append(
                _CodexExecChildCall(
                    tool_path=("tools", "unknown"),
                    tool_name="unknown",
                    registry_type="unknown",
                    argument=item,
                    raw_argument=_codex_tool_output_text(item),
                    parse_state="malformed",
                )
            )
            continue
        raw_name = _mapping_string(item, "name", "tool_name", "tool", "operation", "kind", "type")
        if raw_name:
            name_parts = tuple(part for part in raw_name.replace("::", ".").split(".") if part)
            tool_path = (
                name_parts if name_parts and name_parts[0].lower() in {"tools", "functions"} else ("tools", *name_parts)
            )
        else:
            tool_path = ("tools", "unknown")
        argument: object = {}
        for key in ("arguments", "input", "action", "params", "parameters"):
            if key in item:
                argument = item[key]
                break
        if isinstance(argument, str):
            parsed_argument, parsed = _parse_js_literal(argument)
        else:
            parsed_argument, parsed = argument, True
        registry_type = _classify_code_mode_child(tool_path)
        calls.append(
            _CodexExecChildCall(
                tool_path=tool_path,
                tool_name=_code_mode_tool_name(tool_path, registry_type),
                registry_type=registry_type,
                argument=parsed_argument,
                raw_argument=argument if isinstance(argument, str) else _codex_tool_output_text(argument),
                parse_state="parsed" if raw_name and parsed else "malformed",
            )
        )
    return tuple(calls)


def _code_mode_source(value: object) -> str | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, dict):
            value = parsed
        else:
            return value
    if not isinstance(value, dict):
        return None
    for key in ("source", "code", "script", "javascript", "js", "command", "arguments", "input"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _code_mode_children(value: object) -> tuple[_CodexExecChildCall, ...]:
    structured = _structured_code_mode_child_calls(value)
    if structured:
        return structured
    source = _code_mode_source(value)
    return _scan_code_mode_child_calls(source) if source else ()


def _dedupe_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen or normalized == "/dev/null":
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _patch_touched_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    marker_prefixes = (
        "*** Add File:",
        "*** Update File:",
        "*** Delete File:",
        "*** Move to:",
        "*** Move File:",
    )
    for line in patch.splitlines():
        stripped = line.strip()
        for prefix in marker_prefixes:
            if stripped.startswith(prefix):
                paths.append(stripped.removeprefix(prefix).strip())
                break
        else:
            if stripped.startswith(("+++ ", "--- ")):
                candidate = stripped[4:].split("\t", 1)[0].strip()
                if candidate.startswith(("a/", "b/")):
                    candidate = candidate[2:]
                paths.append(candidate)
            elif stripped.startswith("diff --git "):
                pieces = stripped.removeprefix("diff --git ").split()
                if len(pieces) >= 2:
                    candidate = pieces[-1]
                    paths.append(candidate[2:] if candidate.startswith("b/") else candidate)
    return _dedupe_strings(paths)


def _structural_paths(value: object) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(item: object, *, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key).lower()
                if key in _STRUCTURAL_PATH_KEYS:
                    if isinstance(child, str):
                        paths.append(child)
                    elif isinstance(child, list):
                        paths.extend(value for value in child if isinstance(value, str))
                if isinstance(child, dict | list):
                    visit(child, depth=depth + 1)
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, dict | list):
                    visit(child, depth=depth + 1)

    visit(value, depth=0)
    return _dedupe_strings(paths)


def _structural_byte_count(value: object) -> int | None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            if str(raw_key).lower() in _STRUCTURAL_BYTE_KEYS and isinstance(child, int) and not isinstance(child, bool):
                return child if child >= 0 else None
        for child in value.values():
            if isinstance(child, dict | list):
                nested = _structural_byte_count(child)
                if nested is not None:
                    return nested
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict | list):
                nested = _structural_byte_count(child)
                if nested is not None:
                    return nested
    return None


#: Newer unified-exec builds put one ``Command: <rendered command>`` line in
#: front of the envelope. It is stripped only when the very next line starts
#: the envelope, so a command rendered over several lines leaves the result
#: unknown rather than letting a line inside the command pose as the envelope.
_CODEX_COMMAND_LINE_PREFIX_RE = re.compile(r"\ACommand: [^\n]*\n(?=(?:Chunk ID: [0-9a-f]+\n)?Wall time: )")
_CODEX_CHUNK_ID_PREFIX_RE = re.compile(r"\AChunk ID: [0-9a-f]+\n")
_CODEX_EXEC_ENVELOPE_OUTCOME_RE = re.compile(
    r"\AWall time: [0-9.]+ seconds\n"
    r"(?:Process exited with code (?P<exit_code>-?\d+)"
    r"|Process completed with exit code (?P<exit_code2>-?\d+)"
    r"|Process running with session ID \d+)"
)


def _codex_exec_envelope_outcome(output: object) -> tuple[bool | None, int | None]:
    """Read the exit outcome Codex's own unified-exec tool ("exec_command"/
    "write_stdin"/code-mode exec children) always stamps on its result text.

    Unlike a shell transcript, this preamble is generated by the Codex CLI
    itself, never the model: ``[Chunk ID: <hex>\\n]Wall time: <float>
    seconds\\n`` followed by exactly ``Process exited with code <N>`` (or the
    older ``Process completed with exit code <N>``) once the process has
    exited, or ``Process running with session ID <N>`` while a
    long-lived/chunked session is still attached. Matching is anchored at the
    very start of the field, so it can never fire on an unrelated occurrence
    of similar wording deep inside captured subprocess output (e.g. a CI log
    that itself prints "Process completed with exit code 1") -- that text
    would only ever appear after this preamble, never as a substitute for it.
    A still-running session has no outcome yet and stays honestly unknown.
    """
    if not isinstance(output, str):
        return None, None
    text = _CODEX_COMMAND_LINE_PREFIX_RE.sub("", output, count=1)
    text = _CODEX_CHUNK_ID_PREFIX_RE.sub("", text, count=1)
    match = _CODEX_EXEC_ENVELOPE_OUTCOME_RE.match(text)
    if match is None:
        return None, None
    exit_code_str = match.group("exit_code") or match.group("exit_code2")
    if exit_code_str is None:
        return None, None
    exit_code = int(exit_code_str)
    return exit_code != 0, exit_code


_CODEX_FREEFORM_OUTCOME_RE = re.compile(
    r"\AExit code: (?P<exit_code>-?\d+)\nWall time: [0-9.]+ seconds\n(?:Output:\n|Total output lines: \d+\n)"
)


def _codex_freeform_outcome(output: object) -> tuple[bool | None, int | None]:
    """Read the exit code Codex stamps on a freeform tool result.

    ``apply_patch`` and ``shell_command`` results begin with a header the CLI
    writes, never the model: ``Exit code: <N>\\nWall time: <float>
    seconds\\n`` followed by ``Output:`` or ``Total output lines: <N>``. The
    match is anchored at the start of the field and requires the whole header,
    so the same words inside captured output never count.
    """
    if not isinstance(output, str):
        return None, None
    match = _CODEX_FREEFORM_OUTCOME_RE.match(output)
    if match is None:
        return None, None
    exit_code = int(match.group("exit_code"))
    return exit_code != 0, exit_code


def _codex_tool_result_outcome(raw: object) -> tuple[bool | None, int | None, str | None]:
    """Resolve (is_error, exit_code, unknown reason) for a Codex tool-result payload.

    Tries the JSON-structural outcome first (``exit_code``/``is_error``
    fields nested in a decoded JSON object). When that resolves nothing, a
    string payload falls back to the CLI-written headers that state an exit
    code: the unified-exec envelope, then the freeform ``Exit code:`` header.

    A payload that announces itself as a JSON structure and does not decode is
    a declared outcome carrier the source did not retain intact; a decoded
    structure whose ``exit_code``/``is_error`` is present but off-type is a
    verdict this mapping does not read.
    """
    decoded = _decoded_json_value(raw) if isinstance(raw, str) else raw
    is_error, exit_code = _structural_outcome(decoded)
    if is_error is None and exit_code is None and isinstance(raw, str):
        is_error, exit_code = _codex_exec_envelope_outcome(raw)
        if is_error is None and exit_code is None:
            is_error, exit_code = _codex_freeform_outcome(raw)
    return (
        is_error,
        exit_code,
        unknown_reason(
            is_error=is_error,
            exit_code=exit_code,
            outcome_field_present=_carries_unread_outcome_field(decoded),
            source_intact=not _declares_undecoded_structure(raw, decoded),
        ),
    )


def _declares_undecoded_structure(raw: object, decoded: object) -> bool:
    """True when a payload announced a JSON structure that did not decode."""
    if decoded is not None or not isinstance(raw, str):
        return False
    return raw.lstrip()[:1] in {"{", "["}


def _carries_unread_outcome_field(value: object) -> bool:
    """True when an outcome key is present with a value ``_structural_outcome`` cannot read."""
    for wrapper in _outcome_wrappers(value):
        raw_exit = wrapper.get("exit_code")
        if raw_exit is not None and not (isinstance(raw_exit, int) and not isinstance(raw_exit, bool)):
            return True
        raw_error = wrapper.get("is_error")
        if raw_error is not None and not isinstance(raw_error, bool):
            return True
    return False


def _outcome_wrappers(value: object) -> list[dict[str, object]]:
    """Return the mappings a Codex outcome field can live in, outermost first."""
    wrappers: list[dict[str, object]] = []
    if isinstance(value, dict):
        wrappers.append(value)
        for key in ("metadata", "result", "output"):
            nested = value.get(key)
            if isinstance(nested, dict):
                wrappers.append(nested)
    return wrappers


def _structural_outcome(value: object) -> tuple[bool | None, int | None]:
    wrappers = _outcome_wrappers(value)
    exit_code: int | None = None
    is_error: bool | None = None
    for wrapper in wrappers:
        raw_exit = wrapper.get("exit_code")
        if isinstance(raw_exit, int) and not isinstance(raw_exit, bool):
            exit_code = raw_exit
            break
    for wrapper in wrappers:
        raw_error = wrapper.get("is_error")
        if isinstance(raw_error, bool):
            is_error = raw_error
            break
    if exit_code is not None:
        derived = exit_code != 0
        if is_error is None:
            is_error = derived
    if is_error is None:
        # A structural "timed_out": true (e.g. the `wait`/`write_stdin`
        # child tools' timeout envelope, `{"message": "Wait timed out.",
        # "timed_out": true}`) is itself the provider's own outcome signal --
        # the operation did not complete successfully -- even when no
        # exit_code/is_error field is present.
        for wrapper in wrappers:
            raw_timed_out = wrapper.get("timed_out")
            if isinstance(raw_timed_out, bool):
                if raw_timed_out:
                    is_error = True
                break
    return is_error, exit_code


def _decoded_json_value(value: object) -> object | None:
    if not isinstance(value, str):
        return value
    try:
        decoded: object = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded


def _decoded_structural_text_item(item: object) -> tuple[object, ...]:
    if not isinstance(item, dict):
        return ()
    item_type = item.get("type")
    if item_type not in {"input_text", "output_text"}:
        return ()
    text = item.get("text")
    if not isinstance(text, str):
        return ()
    decoded = _decoded_json_value(text)
    if isinstance(decoded, dict):
        for key in _CODE_MODE_RESULT_COLLECTION_KEYS:
            candidate = decoded.get(key)
            if isinstance(candidate, list) and all(isinstance(value, dict | list) for value in candidate):
                return tuple(candidate)
        return (decoded,)
    if isinstance(decoded, list) and all(isinstance(value, dict | list) for value in decoded):
        # A JSON array emitted as one content item represents ordered child
        # results only when every element is itself a structured value.
        return tuple(decoded)
    return ()


def _code_mode_result_items(output: object, *, child_count: int) -> tuple[object, ...]:
    if child_count <= 0:
        return ()
    parsed = _decoded_json_value(output)
    if isinstance(parsed, dict):
        for key in _CODE_MODE_RESULT_COLLECTION_KEYS:
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                return tuple(candidate[:child_count])
        if parsed.get("type") in {"input_text", "output_text", "input_image", "image"}:
            return _decoded_structural_text_item(parsed)[:child_count]
        # A non-content-item mapping is one exact child result when the
        # envelope has one child. It may carry no promoted outcome fields; in
        # that case the paired result is retained with outcome=unknown.
        return (parsed,) if child_count == 1 else ()
    if parsed is None and child_count == 1 and isinstance(output, str):
        # The whole output failed to decode as JSON -- when there is exactly
        # one child, the raw text itself (possibly Codex's own unified-exec
        # envelope, see _codex_exec_envelope_outcome) is that child's result.
        return (output,)
    if isinstance(parsed, list):
        if all(
            isinstance(item, dict) and item.get("type") in {"input_text", "output_text", "input_image", "image"}
            for item in parsed
        ):
            emitted: list[object] = []
            for item in parsed:
                emitted.extend(_decoded_structural_text_item(item))
                if len(emitted) >= child_count:
                    break
            return tuple(emitted[:child_count])
        return tuple(parsed[:child_count])
    return ()


def _code_mode_item_text(item: dict[str, object]) -> str | None:
    for key in _CODE_MODE_ITEM_TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _code_mode_item_outcome(item: dict[str, object]) -> tuple[bool | None, int | None]:
    """Read the operation outcome the producer states on an ``item_completed``.

    ``exit_code`` is the process status itself. ``status`` is the producer's own
    label over it (``completed`` exactly when the exit code is zero), and is the
    only signal a ``FileChange`` carries.
    """
    exit_code = item.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code != 0, exit_code
    status = item.get("status")
    if isinstance(status, str) and status:
        if status == "completed":
            return False, None
        if status == "failed":
            return True, None
    return None, None


def _code_mode_item_paths(item: dict[str, object]) -> tuple[str, ...]:
    changes = _dict_record(item.get("changes"))
    if changes:
        return _dedupe_strings([str(path) for path in changes])
    return _structural_paths(item)


def _code_mode_item_result(
    item: dict[str, object],
    *,
    existing: _CodexExecChildResult | None,
) -> _CodexExecChildResult:
    """Build a child result from the producer's own execution record.

    The item's text wins over the transport rendering: the transport carries
    what the model was shown, which is truncated, headed, or absent, while the
    item carries the operation's complete output. Nothing is lost by preferring
    it -- the transport rendering stays stored verbatim as the outer
    ``custom_tool_call_output`` tool_result text.
    """
    is_error, exit_code = _code_mode_item_outcome(item)
    text = _code_mode_item_text(item)
    paths = list(_code_mode_item_paths(item))
    byte_count: int | None = None
    if existing is not None:
        if text is None:
            text = existing.text
        paths.extend(existing.paths)
        byte_count = existing.byte_count
        if is_error is None and exit_code is None:
            is_error, exit_code = existing.is_error, existing.exit_code
    item_id = item.get("id")
    return _CodexExecChildResult(
        text=text,
        is_error=is_error,
        exit_code=exit_code,
        unknown_reason=unknown_reason(
            is_error=is_error,
            exit_code=exit_code,
            outcome_field_present="status" in item or "exit_code" in item,
        ),
        paths=_dedupe_strings(paths),
        byte_count=byte_count,
        item_id=str(item_id) if isinstance(item_id, str) and item_id else None,
    )


def _code_mode_item_child(item: dict[str, object]) -> _CodexExecChildCall:
    """Build a child call for an execution the program source did not name.

    A program that runs commands in a loop emits one ``tools.exec_command``
    call site and one ``item_completed`` per iteration, so the scan undercounts.
    The item states the argv, cwd and Codex's own command classification
    directly, which is stronger evidence than the call site it came from.
    """
    item_type = str(item.get("type") or "")
    registry_type = "apply_patch" if item_type == "FileChange" else "exec_command"
    argument: dict[str, object] = {}
    command = _normalized_command(item.get("command"))
    if command is not None:
        argument["command"] = command
    cwd = item.get("cwd")
    if isinstance(cwd, str) and cwd:
        argument["cwd"] = cwd
    parsed_cmd = item.get("parsed_cmd")
    if isinstance(parsed_cmd, list) and parsed_cmd:
        argument["parsed_cmd"] = parsed_cmd
    changes = _dict_record(item.get("changes"))
    if changes:
        argument["changes"] = changes
    item_id = item.get("id")
    return _CodexExecChildCall(
        tool_path=("tools", registry_type),
        tool_name=registry_type,
        registry_type=registry_type,
        argument=argument,
        raw_argument=None,
        parse_state="parsed",
        item_id=str(item_id) if isinstance(item_id, str) and item_id else None,
    )


def _code_mode_item_commands(item: dict[str, object]) -> tuple[str, ...]:
    """Every exact spelling of the command this item records.

    A code-mode program states a command as one shell string; the item records
    the argv the shell was launched with plus Codex's own re-extraction of it.
    Matching is exact string equality against one of these spellings, never a
    similarity test.
    """
    candidates: list[str] = []
    command = item.get("command")
    if isinstance(command, list) and command and all(isinstance(part, str) for part in command):
        candidates.append(shlex.join(command))
        candidates.append(command[-1])
    elif isinstance(command, str):
        candidates.append(command)
    parsed_cmd = item.get("parsed_cmd")
    if isinstance(parsed_cmd, list):
        for entry in parsed_cmd:
            entry_record = _dict_record(entry)
            if entry_record is None:
                continue
            entry_command = entry_record.get("cmd")
            if isinstance(entry_command, str):
                candidates.append(entry_command)
    return _dedupe_strings(candidate.strip() for candidate in candidates if candidate.strip())


def _code_mode_child_command(child: _CodexExecChildCall) -> str | None:
    if isinstance(child.argument, dict):
        for key in ("cmd", "command"):
            command = _normalized_command(child.argument.get(key))
            if command is not None:
                return command
    return _normalized_command(child.argument)


def _apply_code_mode_item_evidence(
    envelope: _CodexExecEnvelope,
    matched: Mapping[int, dict[str, object]],
    appended: Sequence[dict[str, object]],
) -> _CodexExecEnvelope:
    if not matched and not appended:
        return envelope
    children = list(envelope.children)
    results: list[_CodexExecChildResult | None] = [
        envelope.results[index] if index < len(envelope.results) else None for index in range(len(children))
    ]
    for child_index, item in matched.items():
        results[child_index] = _code_mode_item_result(item, existing=results[child_index])
    for item in appended:
        children.append(_code_mode_item_child(item))
        results.append(_code_mode_item_result(item, existing=None))
    return replace(envelope, children=tuple(children), results=tuple(results))


def _code_mode_child_results(output: object, *, child_count: int) -> tuple[_CodexExecChildResult, ...]:
    results: list[_CodexExecChildResult] = []
    for item in _code_mode_result_items(output, child_count=child_count):
        is_error, exit_code, reason = _codex_tool_result_outcome(item)
        results.append(
            _CodexExecChildResult(
                text=_codex_tool_output_text(item),
                is_error=is_error,
                exit_code=exit_code,
                unknown_reason=reason,
                paths=_structural_paths(item),
                byte_count=_structural_byte_count(item),
            )
        )
    return tuple(results)


def _response_inner_record(item: object) -> dict[str, object] | None:
    record = _dict_record(item)
    if record is None:
        return None
    if (legacy := _legacy_response_record(record)) is not None:
        return legacy
    if _record_type(record) not in {"response_item", "event_msg"}:
        return None
    inner = _payload_record(record)
    return inner if inner is not None and not _is_message(inner) else None


class _CodexLookaheadObserver:
    """The per-record half of the lookahead: facts a later record may need.

    Records are observed once, in stream order, while the stream is being
    retained for replay -- so a streamed rollout is read once for retention
    and lookahead together, then replayed once for materialization.
    ``finish`` resolves the collected facts after the last record.
    """

    __slots__ = ("_index", "_open_call_index", "_last_call_index", "_finished")

    def __init__(self, index_store: _CodexLookaheadIndex) -> None:
        self._index = index_store
        self._open_call_index: int | None = None
        self._last_call_index: int | None = None
        self._finished = False

    def observe(self, record_index: int, item: object) -> None:
        index_store = self._index
        connection = index_store.connection
        record = _dict_record(item)
        if record is not None:
            message_record = _message_record(record)
            if message_record is not None:
                raw_role = _effective_role(message_record)
                if raw_role and raw_role != "unknown":
                    text = extract_codex_text(_effective_content(message_record))
                    index_store.add_message_echo(
                        record_index,
                        (Role.normalize(raw_role).value, text),
                        _codex_message_echo_evidence(message_record, _message_timestamp(record, message_record)),
                    )
        inner = _response_inner_record(item)
        if inner is None:
            return
        payload = _record_payload(inner)
        record_type = _record_type(inner)
        if record_type == "item_completed":
            executed = _dict_record(payload.get("item"))
            if executed is not None and str(executed.get("type") or "") in _CODE_MODE_ITEM_CHILD_TYPES:
                connection.execute(
                    "INSERT INTO codex_items VALUES (?, ?, ?, ?)",
                    (
                        record_index,
                        pickle.dumps(_reduced_code_mode_item(executed), protocol=pickle.HIGHEST_PROTOCOL),
                        self._open_call_index,
                        self._last_call_index,
                    ),
                )
            return
        if record_type in {
            "function_call",
            "custom_tool_call",
            "tool_search_call",
            "web_search_call",
            "local_shell_call",
        }:
            tool_name = payload.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                tool_name = payload.get("execution")
            if not isinstance(tool_name, str) or tool_name.lower() not in _CODE_MODE_EXEC_TOOL_NAMES:
                return
            raw_arguments = payload.get("arguments")
            if raw_arguments is None:
                raw_arguments = payload.get("input")
            if raw_arguments is None:
                raw_arguments = payload.get("action")
            children = _code_mode_children(raw_arguments)
            if not children:
                return
            raw_tool_id = payload.get("call_id") or payload.get("id")
            tool_id = str(raw_tool_id) if raw_tool_id else None
            envelope = _CodexExecEnvelope(
                transport_tool_name=tool_name,
                transport_tool_id=tool_id,
                transport_provider_message_id=_codex_tool_record_message_id(payload, side="call"),
                children=children,
            )
            occurrence = index_store.occurrence("call", tool_id) if tool_id else None
            connection.execute(
                "INSERT INTO codex_calls VALUES (?, ?, ?, ?)",
                (
                    record_index,
                    _sql_key(tool_id) if tool_id is not None else None,
                    occurrence,
                    pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL),
                ),
            )
            self._open_call_index = record_index
            self._last_call_index = record_index
        elif record_type in {
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
            "web_search_output",
        }:
            self._open_call_index = None
            raw_tool_id = payload.get("call_id") or payload.get("id")
            if raw_tool_id:
                tool_id = str(raw_tool_id)
                connection.execute(
                    "INSERT INTO codex_outputs VALUES (?, ?, ?, ?)",
                    (
                        record_index,
                        _sql_key(tool_id),
                        index_store.occurrence("output", tool_id),
                        pickle.dumps(inner, protocol=pickle.HIGHEST_PROTOCOL),
                    ),
                )

    def observed(self, records: Iterable[object]) -> Iterator[object]:
        """Yield ``records`` unchanged, observing each one as it passes."""
        for record_index, item in enumerate(records, start=1):
            self.observe(record_index, item)
            yield item

    def finish(self) -> _CodexLookaheadIndex:
        if self._finished:
            raise RuntimeError("codex lookahead finished twice")
        self._finished = True
        return _resolve_codex_lookahead(self._index)


def _codex_lookahead(
    records: Iterable[object],
    index_store: _CodexLookaheadIndex,
) -> _CodexLookaheadIndex:
    """Resolve code-mode calls and duplicate signatures with disk-backed indexes."""
    observer = _CodexLookaheadObserver(index_store)
    for record_index, item in enumerate(records, start=1):
        observer.observe(record_index, item)
    return observer.finish()


def _resolve_codex_lookahead(index_store: _CodexLookaheadIndex) -> _CodexLookaheadIndex:
    connection = index_store.connection
    slot = 0
    for call_index, envelope_blob in connection.execute(
        "SELECT record_index, envelope FROM codex_calls ORDER BY record_index"
    ):
        envelope = pickle.loads(envelope_blob)
        for child_index, child in enumerate(envelope.children):
            connection.execute(
                "INSERT INTO codex_slots(slot, call_index, child_index, registry_type, command) VALUES (?, ?, ?, ?, ?)",
                (
                    slot,
                    call_index,
                    child_index,
                    child.registry_type,
                    _text_digest(command) if (command := _code_mode_child_command(child)) is not None else None,
                ),
            )
            slot += 1

    for item_order, item_blob, open_index, last_index in connection.execute(
        "SELECT item_order, item, open_call_index, last_call_index FROM codex_items ORDER BY item_order"
    ):
        executed = pickle.loads(item_blob)
        compatible = _CODE_MODE_ITEM_CHILD_TYPES[str(executed.get("type") or "")]
        chosen: tuple[int, int, int] | None = None
        for command in _code_mode_item_commands(executed):
            candidate = connection.execute(
                "SELECT slot, call_index, child_index, registry_type FROM codex_slots "
                "WHERE command = ? AND claimed = 0 ORDER BY slot LIMIT 1",
                (_text_digest(command),),
            ).fetchone()
            if candidate is not None and candidate[3] in compatible and (chosen is None or candidate[0] < chosen[0]):
                chosen = (candidate[0], candidate[1], candidate[2])
        if chosen is None and open_index is not None:
            candidate = connection.execute(
                "SELECT slot, call_index, child_index, registry_type FROM codex_slots "
                "WHERE call_index = ? AND claimed = 0 ORDER BY slot",
                (open_index,),
            )
            for candidate_slot, call_index, child_index, registry_type in candidate:
                if registry_type in compatible:
                    chosen = (candidate_slot, call_index, child_index)
                    break
        if chosen is None:
            host = open_index if open_index is not None else last_index
            if host is not None:
                connection.execute("INSERT INTO codex_appended VALUES (?, ?, ?)", (host, item_order, item_blob))
            continue
        connection.execute("UPDATE codex_slots SET claimed = 1 WHERE slot = ?", (chosen[0],))
        connection.execute("INSERT INTO codex_matched VALUES (?, ?, ?)", (chosen[1], chosen[2], item_blob))

    for call_index, tool_id, occurrence, envelope_blob in connection.execute(
        "SELECT record_index, tool_id, occurrence, envelope FROM codex_calls ORDER BY record_index"
    ):
        envelope = pickle.loads(envelope_blob)
        output_index = None
        if tool_id is not None:
            output_row = connection.execute(
                "SELECT record_index, output FROM codex_outputs WHERE tool_id = ? AND occurrence = ?",
                (tool_id, occurrence),
            ).fetchone()
            if output_row is not None:
                output_index, output_blob = output_row
                output_record = pickle.loads(output_blob)
                output = output_record.get("output")
                if output is None:
                    output = output_record.get("tools")
                if output is None:
                    output = output_record.get("result")
                envelope = replace(
                    envelope,
                    results=_code_mode_child_results(output, child_count=len(envelope.children)),
                )
        matched = {
            child_index: pickle.loads(item_blob)
            for child_index, item_blob in connection.execute(
                "SELECT child_index, item FROM codex_matched WHERE call_index = ? ORDER BY child_index",
                (call_index,),
            )
        }
        appended = [
            pickle.loads(item_blob)
            for (item_blob,) in connection.execute(
                "SELECT item FROM codex_appended WHERE call_index = ? ORDER BY item_order",
                (call_index,),
            )
        ]
        resolved = _apply_code_mode_item_evidence(envelope, matched, appended)
        resolved_blob = pickle.dumps(resolved, protocol=pickle.HIGHEST_PROTOCOL)
        connection.execute("INSERT INTO codex_resolved VALUES (?, ?)", (call_index, resolved_blob))
        if output_index is not None:
            connection.execute("INSERT INTO codex_resolved VALUES (?, ?)", (output_index, resolved_blob))
    return index_store


def _normalized_command(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return shlex.join(value)
    return None


def _child_tool_input(
    child: _CodexExecChildCall,
    *,
    child_index: int,
    envelope: _CodexExecEnvelope,
    result: _CodexExecChildResult | None,
) -> dict[str, object]:
    if isinstance(child.argument, dict):
        tool_input: dict[str, object] = {str(key): value for key, value in child.argument.items()}
    elif child.argument is None:
        tool_input = {}
    else:
        tool_input = {"input": child.argument}

    command: str | None = None
    if child.parse_state == "parsed" and child.registry_type == "exec_command":
        command = _normalized_command(tool_input.get("command")) or _normalized_command(tool_input.get("cmd"))
        if command is None:
            command = _normalized_command(child.argument)
    elif child.parse_state == "parsed" and child.registry_type == "apply_patch":
        patch = (
            _normalized_command(tool_input.get("patch"))
            or _normalized_command(tool_input.get("input"))
            or _normalized_command(tool_input.get("command"))
            or _normalized_command(child.argument)
        )
        if patch is not None:
            tool_input.setdefault("patch", patch)
            command = patch
            patch_paths = _patch_touched_paths(patch)
            if patch_paths:
                tool_input["paths"] = list(patch_paths)
                tool_input["path"] = patch_paths[0]
    if command is not None:
        tool_input["command"] = command

    paths = list(_structural_paths(tool_input))
    byte_count = _structural_byte_count(tool_input)
    if result is not None:
        paths.extend(result.paths)
        if byte_count is None:
            byte_count = result.byte_count
    normalized_paths = _dedupe_strings(paths)
    if normalized_paths:
        tool_input["paths"] = list(normalized_paths)
        tool_input.setdefault("path", normalized_paths[0])
    if byte_count is not None:
        tool_input["byte_count"] = byte_count
    if (child.parse_state != "parsed" or child.registry_type == "unknown") and child.raw_argument is not None:
        tool_input["raw_arguments"] = child.raw_argument

    provenance: dict[str, object] = {
        "kind": "codex.functions_exec_child",
        "registry_type": child.registry_type,
        "parse_state": child.parse_state,
        "raw_tool_path": ".".join(child.tool_path),
        "transport_child_index": child_index,
        "transport": {
            "provider_message_id": envelope.transport_provider_message_id,
            "tool_id": envelope.transport_tool_id,
            "tool_name": envelope.transport_tool_name,
            "block_position": 0,
        },
    }
    if child.source_start is not None and child.source_end is not None:
        provenance["source_span"] = [child.source_start, child.source_end]
    if child.item_id is not None:
        provenance["item_completed_id"] = child.item_id
    # Provenance discriminator. A child read out of the program source is a
    # *call site*, not an execution: ``function unused(){ tools.apply_patch(
    # "*** Update File: sensitive.py") }`` scans exactly like a call that ran.
    # Only the producer's item list is ground truth for execution, so the
    # route the child came in on and whatever execution evidence it carries
    # are both recorded. These land in ``blocks.tool_input`` (persisted JSON),
    # not ``ParsedContentBlock.metadata``, which the writer drops.
    if child.source_start is not None:
        provenance["call_site_origin"] = "program_source"
    elif child.item_id is not None:
        provenance["call_site_origin"] = "producer_item"
    else:
        provenance["call_site_origin"] = "structured_child_list"
    if child.item_id is not None or (result is not None and result.item_id is not None):
        provenance["execution_evidence"] = "item_completed"
    elif result is not None:
        provenance["execution_evidence"] = "transport_result"
    else:
        provenance["execution_evidence"] = "none"
    provenance["executed"] = provenance["execution_evidence"] != "none"
    if result is not None and (result.paths or result.byte_count is not None):
        result_fields: dict[str, object] = {}
        if result.paths:
            result_fields["paths"] = list(result.paths)
        if result.byte_count is not None:
            result_fields["byte_count"] = result.byte_count
        provenance["structural_result_fields"] = result_fields
    tool_input[_CODE_MODE_CHILD_PROVENANCE_KEY] = provenance
    return tool_input


def _child_tool_id(envelope: _CodexExecEnvelope, child_index: int) -> str | None:
    if not envelope.transport_tool_id:
        return None
    return f"{envelope.transport_tool_id}{_CODE_MODE_CHILD_ID_MARKER}{child_index}"


def _code_mode_child_use_blocks(envelope: _CodexExecEnvelope) -> list[ParsedContentBlock]:
    blocks: list[ParsedContentBlock] = []
    for child_index, child in enumerate(envelope.children):
        result = envelope.results[child_index] if child_index < len(envelope.results) else None
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name=child.tool_name,
                tool_id=_child_tool_id(envelope, child_index),
                tool_input=_child_tool_input(
                    child,
                    child_index=child_index,
                    envelope=envelope,
                    result=result,
                ),
            )
        )
    return blocks


#: ``world_state.state`` keys deliberately NOT stored (polylogue-w54q3). Each
#: one is the prompt scaffolding Codex re-sends verbatim on every snapshot --
#: instruction/context-file bodies, skill catalogs and usage hints -- not
#: evidence about the session: ``host_skills`` alone measures ~4.6 MB across
#: 443 observed records and ``agents_md`` ~9.0 MB, both near-identical text
#: repeated per snapshot. Every OTHER ``state`` key is stored, so a new
#: upstream key is carried rather than silently dropped, and
#: ``test_world_state_state_keys_are_stored_or_declared_exempt`` fails loudly
#: if a key is neither.
_WORLD_STATE_INSTRUCTION_TEXT_KEYS: frozenset[str] = frozenset(
    {
        "agents_md",
        "apps_instructions",
        "context_window_guidance",
        "environments_instructions",
        "host_skills",
        "managed_developer_instructions",
        "multi_agent_usage_hint",
        "orchestrator_skills",
        "plugins_instructions",
        "skills",
    }
)


# polylogue-9x22: ``ParsedContentBlock.metadata`` is never persisted -- the
# ``blocks`` table has no metadata column and the only key the write path
# reads back out of it is ``language`` (``storage/sqlite/archive_tiers/
# write.py:_block_language``). ``_code_mode_child_result_blocks`` below still
# attaches ``codex_functions_exec_*``/``paths``/``byte_count`` to
# ``metadata`` as an in-process carrier; ``_code_mode_child_result_evidence_
# events`` projects that same dict into ``session_events`` (same precedent
# as ``claude/common.py``'s ``claude_ai_web_tool_evidence``), keyed to the
# tool-result message that owns the child blocks.
def _code_mode_child_result_evidence_events(
    envelope: _CodexExecEnvelope,
    *,
    source_message_provider_id: str,
    timestamp: str | None,
) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    for child_index in range(len(envelope.results)):
        result = envelope.results[child_index]
        if result is None:
            continue
        metadata = _code_mode_child_result_metadata(envelope, child_index, result)
        events.append(
            ParsedSessionEvent(
                event_type="codex_functions_exec_child_result_evidence",
                timestamp=timestamp,
                source_message_provider_id=source_message_provider_id,
                payload=metadata,
            )
        )
    return events


def _code_mode_child_result_metadata(
    envelope: _CodexExecEnvelope,
    child_index: int,
    result: _CodexExecChildResult,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "codex_functions_exec_child_index": child_index,
        "codex_functions_exec_registry_type": envelope.children[child_index].registry_type,
    }
    if result.paths:
        metadata["paths"] = list(result.paths)
    if result.byte_count is not None:
        metadata["byte_count"] = result.byte_count
    if result.item_id is not None:
        metadata["codex_item_completed_id"] = result.item_id
    return metadata


def _code_mode_child_result_blocks(envelope: _CodexExecEnvelope) -> list[ParsedContentBlock]:
    blocks: list[ParsedContentBlock] = []
    for child_index, result in enumerate(envelope.results):
        if result is None:
            continue
        metadata = _code_mode_child_result_metadata(envelope, child_index, result)
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=_child_tool_id(envelope, child_index),
                text=result.text,
                metadata=metadata,
                is_error=result.is_error,
                exit_code=result.exit_code,
                outcome_unknown_reason=result.unknown_reason,
            )
        )
    return blocks


def _tool_input_from_arguments(value: object, *, tool_name: str) -> dict[str, object]:
    if isinstance(value, dict):
        tool_input = dict(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            tool_input = {"arguments": value}
        else:
            tool_input = dict(parsed) if isinstance(parsed, dict) else {"arguments": value}
    else:
        return {}

    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        return tool_input

    cmd = tool_input.get("cmd")
    if isinstance(cmd, str) and cmd.strip():
        return {**tool_input, "command": cmd}

    arguments = tool_input.get("arguments")
    if tool_name.lower() in _EXECUTION_TOOL_NAMES and isinstance(arguments, str) and arguments.strip():
        return {**tool_input, "command": arguments}

    # apply_patch carries its whole payload as a PATCH-FORMAT STRING under
    # ``arguments`` -- not JSON -- so the operated-on path lives in
    # ``*** Update File: <path>`` / ``*** Add File:`` / ``*** Delete File:``
    # header lines, where no ``json_extract`` can reach it. blocks.tool_path
    # and blocks.search_text are both generated from
    # ``$.file_path``/``$.path`` (archive_tiers/index.py:307,310), so every
    # Codex file edit was invisible to structured path queries AND to FTS:
    # tool_path coverage measured 0.07% for codex-session against 44% for
    # claude-code-session, and apply_patch is 95% of Codex tool calls
    # (18,984 of a 20,000 sample). It also left Codex sessions unable to earn
    # a path-bearing structural label (polylogue-a9hx).
    #
    # The batched code-mode child path already extracts these via
    # ``_patch_touched_paths``; this is the standalone ``function_call``
    # branch, which did not. Same helper, so both paths agree.
    if tool_name.lower() in _PATCH_TOOL_NAMES and isinstance(arguments, str) and arguments.strip():
        patch_paths = _patch_touched_paths(arguments)
        if patch_paths:
            # ``path`` is what the generated column reads; ``paths`` keeps the
            # full set, since 11% of patches touch more than one file
            # (554 of 5,000 sampled) and a single column cannot hold them.
            return {**tool_input, "patch": arguments, "path": patch_paths[0], "paths": list(patch_paths)}
    return tool_input


def _codex_material_origin(role: Role, message_type: MessageType, text: str | None) -> MaterialOrigin:
    material_origin = classify_material_origin(role=role, message_type=message_type, text=text)
    if material_origin is MaterialOrigin.UNKNOWN and role is Role.USER and message_type is MessageType.MESSAGE:
        return MaterialOrigin.HUMAN_AUTHORED
    return material_origin


def _codex_tool_record_message_id(payload: Mapping[str, object], *, side: str) -> str:
    """Return the native message identity of one Codex tool call or output record.

    A tool record's own ``id`` is its native message id when the producer
    wrote one. Rollouts normally carry only ``call_id``, and that value names
    the *call*, which both the request record and its output record repeat:
    using it bare would give two distinct messages one provider id. The
    declared record side (``call`` or ``output``) qualifies it, so each record
    keeps a distinct identity derived from source fields, never from position.
    The MCP pair uses the same scheme (``::mcp-call`` / ``::mcp-output``).
    """
    native_id = payload.get("id")
    if native_id:
        return str(native_id)
    call_id = payload.get("call_id")
    if call_id:
        return f"{call_id}::{side}"
    # polylogue-slshy: no positional fallback; an empty id lets the
    # content-anchor identity run instead.
    return ""


def _codex_tool_message(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    timestamp_fallback: str | int | float | None = None,
    exec_envelope: _CodexExecEnvelope | None = None,
) -> ParsedMessage | None:
    payload = _record_payload(record)
    record_type = _record_type(record)
    timestamp = _iso_or_none(_record_timestamp(record) or timestamp_fallback)
    if record_type in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "local_shell_call"}:
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = payload.get("execution")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = record_type
        tool_id = payload.get("call_id") or payload.get("id")
        raw_arguments = payload.get("arguments")
        if raw_arguments is None:
            raw_arguments = payload.get("input")
        if raw_arguments is None:
            raw_arguments = payload.get("action")
        blocks = [
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name=tool_name,
                tool_id=str(tool_id) if tool_id else None,
                tool_input=_tool_input_from_arguments(raw_arguments, tool_name=tool_name),
            )
        ]
        if exec_envelope is not None:
            blocks.extend(_code_mode_child_use_blocks(exec_envelope))
        return ParsedMessage(
            provider_message_id=_codex_tool_record_message_id(payload, side="call"),
            role=Role.ASSISTANT,
            text=tool_name,
            timestamp=timestamp,
            position=position,
            variant_index=0,
            is_active_path=True,
            blocks=blocks,
        )
    if record_type in {"function_call_output", "custom_tool_call_output", "tool_search_output", "web_search_output"}:
        tool_id = payload.get("call_id") or payload.get("id")
        output = payload.get("output")
        if output is None:
            output = payload.get("tools")
        if output is None:
            output = payload.get("result")
        output_text = _codex_tool_output_text(output)
        if not tool_id and not output_text:
            return None
        # Only exact structured fields (or Codex's own generated exec-tool
        # envelope, see _codex_exec_envelope_outcome) affect the outcome.
        # Arbitrary prose containing exit-code-like wording remains evidence
        # text with an unknown outcome.
        is_error, exit_code, reason = _codex_tool_result_outcome(output)
        blocks = [
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=str(tool_id) if tool_id else None,
                text=output_text,
                is_error=is_error,
                exit_code=exit_code,
                outcome_unknown_reason=reason,
            )
        ]
        if exec_envelope is not None:
            blocks.extend(_code_mode_child_result_blocks(exec_envelope))
        return ParsedMessage(
            provider_message_id=_codex_tool_record_message_id(payload, side="output"),
            role=Role.TOOL,
            text=output_text,
            timestamp=timestamp,
            position=position,
            variant_index=0,
            is_active_path=True,
            blocks=blocks,
        )
    return None


def _codex_reasoning_joined_text(value: object) -> str | None:
    """Join recoverable text out of a Codex ``reasoning`` record's `summary`/`content`.

    Both fields share the same OpenAI Responses-API shape: either a bare
    string, or a list of ``{"type": "summary_text"|"reasoning_text", "text": ...}``
    (or equivalent) dicts. Anything else (encrypted ciphertext, missing
    fields) yields no text -- that is a genuine absence, handled by the
    caller, not an extraction bug here.
    """
    if isinstance(value, str):
        return value or None
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for item in value:
        text: object = item.get("text") if isinstance(item, dict) else item
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def _codex_reasoning_message(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    timestamp_fallback: str | int | float | None = None,
) -> ParsedMessage | None:
    """Materialize a Codex ``reasoning`` response_item as a THINKING-block message.

    polylogue-vf9x: previously this record type was read only by
    ``_compact_response_payload``'s generic session_event compactor, which
    has no `reasoning`-specific branch -- neither `summary` nor `content` was
    read at all (not merely char-counted: the emitted session_event carries
    only ``{source_index, type}``), so every one of the measured 1,182,071
    Codex reasoning records in this operator's raw corpus contributed zero
    words to the archive, unreachable from FTS/search even in principle.

    `summary` (OpenAI's human-readable condensation) is present with
    recoverable text on ~24% of records measured; `content` (the full trace)
    is essentially always null on the wire -- Codex encrypts it into
    `encrypted_content` instead, which this archive cannot decrypt and does
    not attempt to store. Even when neither carries text, the message is
    still recorded (block text=None) so the FACT that the model reasoned
    here survives -- the same rationale as Claude Code's empty-body thinking
    blocks (base_support.py).

    Routed into `messages`/`blocks` (not left as a session_event only) so
    reasoning joins the normal content tree: FTS coverage, `thinking_count`,
    and the `material_origin`/`BlockType.THINKING` vocabulary every other
    origin's reasoning content already uses.
    """
    if _record_type(record) != "reasoning":
        return None
    payload = _record_payload(record)
    summary_text = _codex_reasoning_joined_text(payload.get("summary"))
    content_text = _codex_reasoning_joined_text(payload.get("content"))
    blocks: list[ParsedContentBlock] = []
    if summary_text:
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=summary_text))
    if content_text and content_text != summary_text:
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=content_text))
    if not blocks:
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=None))
    combined_text = "\n\n".join(t for t in (summary_text, content_text) if t) or None
    timestamp = _iso_or_none(_record_timestamp(record) or timestamp_fallback)
    return ParsedMessage(
        provider_message_id=_record_id(payload) or "",
        role=Role.ASSISTANT,
        text=combined_text,
        timestamp=timestamp,
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=blocks,
        message_type=MessageType.THINKING,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
    )


def _mcp_invocation_tool_name(invocation: dict[str, object]) -> str:
    server = _string_value(invocation.get("server"))
    tool = _string_value(invocation.get("tool"))
    if server and tool:
        return f"mcp__{server}__{tool}"
    return tool or server or "mcp_tool_call"


def _mcp_result_outcome(result: object) -> tuple[bool | None, str | None, str | None]:
    """Extract (is_error, text, unknown reason) from an ``mcp_tool_call_end`` ``result``.

    Codex wraps MCP results as a Rust-style ``{"Ok": ...}`` / ``{"Err": "..."}``
    tagged union. An Ok payload is a CallToolResult whose optional camelCase
    ``isError`` reports application failure independently of transport success.
    The required content vector and typed verdict distinguish this object from
    malformed payloads, whose complete text remains outcome-unknown. A mapping
    carrying neither tag is that union with a
    variant this mapping does not read; anything that is not a mapping is not
    the union at all and reports no outcome.
    """
    if not isinstance(result, dict):
        return None, _codex_tool_output_text(result), unknown_reason(is_error=None)
    if "Err" in result:
        return True, _codex_tool_output_text(result.get("Err")), None
    if "Ok" in result:
        payload = result.get("Ok")
        text = _codex_tool_output_text(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
            return None, text, unknown_reason(is_error=None, outcome_field_present=True)
        is_error = payload.get("isError")
        if is_error is None:
            return False, text, None
        if isinstance(is_error, bool):
            return is_error, text, None
        return None, text, unknown_reason(is_error=None, outcome_field_present=True)
    return None, _codex_tool_output_text(result), unknown_reason(is_error=None, outcome_field_present=True)


def _codex_mcp_tool_call_messages(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    timestamp_fallback: str | int | float | None = None,
) -> tuple[ParsedMessage, ParsedMessage] | None:
    """Parse a Codex ``mcp_tool_call_end`` record into a tool_use/tool_result pair.

    Unlike ``function_call``/``function_call_output``, Codex emits MCP tool
    invocations as a single self-contained record carrying both the request
    (``invocation.server``/``invocation.tool``/``invocation.arguments``) and
    the response (``result``) -- there is no paired ``mcp_tool_call_begin``.
    Previously this whole record fell through to the generic event-summary
    path, which drops the invocation and result entirely; this was the
    largest single unread surface in the corpus (arbitrary downstream MCP
    server responses, e.g. github/serena/sinex tool calls made from Codex).
    """
    payload = _record_payload(record)
    if payload.get("type") != "mcp_tool_call_end":
        return None
    invocation = _dict_record(payload.get("invocation"))
    if invocation is None:
        return None
    tool_name = _mcp_invocation_tool_name(invocation)
    call_id = payload.get("call_id")
    tool_id = str(call_id) if isinstance(call_id, str) and call_id else f"mcp-call-{index}"
    arguments = invocation.get("arguments")
    if isinstance(arguments, dict):
        tool_input: dict[str, object] = dict(arguments)
    elif arguments is not None:
        tool_input = {"arguments": arguments}
    else:
        tool_input = {}
    timestamp = _iso_or_none(_record_timestamp(record) or timestamp_fallback)
    use_message = ParsedMessage(
        provider_message_id=f"{tool_id}::mcp-call",
        role=Role.ASSISTANT,
        text=tool_name,
        timestamp=timestamp,
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name=tool_name,
                tool_id=tool_id,
                tool_input=tool_input,
            )
        ],
    )
    is_error, result_text, mcp_unknown_reason = _mcp_result_outcome(payload.get("result"))
    result_message = ParsedMessage(
        provider_message_id=f"{tool_id}::mcp-output",
        role=Role.TOOL,
        text=result_text,
        timestamp=timestamp,
        position=position + 1,
        variant_index=0,
        is_active_path=True,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_id,
                text=result_text,
                is_error=is_error,
                outcome_unknown_reason=mcp_unknown_reason,
            )
        ],
    )
    return use_message, result_message


def _codex_tool_output_text(output: object) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        sanitized = _sanitize_codex_large_inline_payloads(output)
        if sanitized != output:
            return str(sanitized)
        try:
            parsed = json.loads(output)
        except (ValueError, TypeError, RecursionError):
            # Nesting deeper than the interpreter's recursion limit is not a
            # decodable JSON envelope for this parser's purposes. It is the
            # same non-JSON verdict as a syntax error: the string is kept
            # verbatim, and the durable raw still holds the original bytes.
            return output
        sanitized_parsed = _sanitize_codex_large_inline_payloads(parsed)
        if sanitized_parsed != parsed:
            return json.dumps(sanitized_parsed, sort_keys=True)
        return output
    sanitized = _sanitize_codex_large_inline_payloads(output)
    return json.dumps(sanitized, sort_keys=True)


#: Container nesting this sanitizer will descend through. Tool output is
#: attacker-controlled: ``json.loads`` decodes far deeper than this recursive
#: Python walk can follow, so without a declared depth an alternating
#: ``[{[{...`` output raised ``RecursionError`` out of the parser and refused
#: the entire rollout. No real Codex tool output nests near this.
_CODEX_SANITIZER_MAX_DEPTH = 200


def _sanitize_codex_large_inline_payloads(value: object, depth: int = 0) -> object:
    if isinstance(value, str):
        return _sanitize_codex_data_url(value)
    if depth >= _CODEX_SANITIZER_MAX_DEPTH:
        # Below the declared depth the subtree is returned unchanged: it is
        # kept in full, not truncated. Only the data-URL rewrite is skipped,
        # and that is reported rather than inferred from the output.
        emit(
            "sources.codex.tool_output_sanitizer_depth_exceeded",
            level=WARNING,
            max_depth=_CODEX_SANITIZER_MAX_DEPTH,
        )
        return value
    if isinstance(value, list):
        return [_sanitize_codex_large_inline_payloads(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_codex_large_inline_payloads(item, depth + 1) for key, item in value.items()}
    return value


def _sanitize_codex_data_url(value: str) -> str:
    if not value.startswith("data:image/") or ";base64," not in value:
        return value
    header, encoded = value.split(",", 1)
    mime = header.removeprefix("data:").split(";", 1)[0] or "image/unknown"
    digest_builder = hashlib.sha256()
    for offset in range(0, len(encoded), 1024 * 1024):
        digest_builder.update(encoded[offset : offset + 1024 * 1024].encode("ascii", errors="ignore"))
    digest = digest_builder.hexdigest()
    padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
    approx_bytes = max(0, (len(encoded) * 3) // 4 - padding)
    return f"<inline image omitted; mime={mime}; approx_bytes={approx_bytes}; sha256_base64={digest}>"


def _codex_inline_image_blocks(content: object) -> tuple[ParsedContentBlock, ...]:
    """Return typed, bounded evidence for inline images without authored prose inflation."""
    if not isinstance(content, list):
        return ()
    blocks: list[ParsedContentBlock] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in {"input_image", "image"}:
            continue
        image_url = item.get("image_url")
        if not isinstance(image_url, str):
            continue
        summary = _sanitize_codex_data_url(image_url)
        if summary != image_url:
            header = image_url.split(",", 1)[0]
            mime = header.removeprefix("data:").split(";", 1)[0] or "image/unknown"
            blocks.append(ParsedContentBlock(type=BlockType.IMAGE, text=summary, media_type=mime))
    return tuple(blocks)


def _codex_message_echo_evidence(
    record: dict[str, object], timestamp: str | int | float | None
) -> tuple[str | None, str | None, str | None]:
    native_id = _string_field(record, "client_id", "id")
    pair = parse_timestamp_pair(timestamp)
    instant = pair[0].astimezone(timezone.utc).isoformat() if pair is not None else None
    turn_evidence = _codex_turn_evidence(record)
    turn_id = None if "turn_id_conflict" in turn_evidence else _string_value(turn_evidence.get("turn_id"))
    return native_id, instant, turn_id


def _codex_event_message(
    record: dict[str, object],
    *,
    index: int,
    position: int,
    echo_index: _CodexLookaheadIndex,
    timestamp_fallback: str | int | float | None = None,
) -> ParsedMessage | None:
    record_type = _record_type(record)
    if record_type not in {"user_message", "agent_message"}:
        return None
    text = record.get("message")
    if not isinstance(text, str) or not text.strip():
        return None
    role = Role.USER if record_type == "user_message" else Role.ASSISTANT
    if echo_index.consume_message_echo(
        (role.value, text),
        _codex_message_echo_evidence(record, _record_timestamp(record) or timestamp_fallback),
    ):
        return None
    message_type = classify_text_message_type(text) or MessageType.MESSAGE
    return ParsedMessage(
        # polylogue-slshy: no positional fallback (see above).
        provider_message_id=str(record.get("client_id") or record.get("id") or ""),
        role=role,
        text=text,
        timestamp=_iso_or_none(_record_timestamp(record) or timestamp_fallback),
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
        message_type=message_type,
        material_origin=_codex_material_origin(role, message_type, text),
    )


def _effective_role(record: dict[str, object]) -> str:
    payload = _payload_record(record)
    if payload is not None:
        value = payload.get("role")
        return value if isinstance(value, str) else "unknown"
    value = record.get("role")
    return value if isinstance(value, str) else "unknown"


def _effective_content(record: dict[str, object]) -> list[object]:
    payload = _payload_record(record)
    value = payload.get("content") if payload is not None else record.get("content")
    return value if isinstance(value, list) else []


def _message_type_from_codex_message(record: dict[str, object], text: str | None) -> MessageType:
    artifact_type = classify_text_message_type(text)
    return artifact_type or MessageType.MESSAGE


def looks_like(payload: Sequence[object]) -> bool:
    """Detect Codex JSONL format using typed validation.

    Newest format (envelope with typed payloads):
        {"type":"session_meta","payload":{"id":"...","timestamp":"...","git":{...}}}
        {"type":"response_item","payload":{"type":"message","role":"user","content":[...]}}

    Intermediate format (JSONL with session metadata + messages):
        {"id":"...","timestamp":"...","git":{...}}
        {"record_type":"state"}
        {"type":"message","role":"user","content":[...]}
    """
    if not isinstance(payload, list):
        return False

    for idx, item in enumerate(payload, start=1):
        if not _is_plausibly_codex_record(item):
            continue
        record = _validate_record(item, index=idx)
        if record is None:
            continue
        if record.format_type in ("envelope", "direct", "state"):
            return True
        if record.id and record.timestamp:
            return True

    return False


def _is_bare_token_usage_record(item: object) -> bool:
    """An older top-level ``token_usage_record``: named counters beside ``type``.

    It has no ``payload``, ``id`` or ``timestamp``, so the generic plausibility
    test refuses it, yet the parser extracts exactly these counters. Only a
    record that states at least one known counter qualifies; the type name
    alone is not evidence.
    """
    record = _dict_record(item)
    return (
        record is not None
        and _record_type(record) == "token_usage_record"
        and "payload" not in record
        and bool(_codex_token_usage_payload(_dict_record(record.get("usage")) or record))
    )


def is_supported_session_stream(payload: Sequence[object]) -> bool:
    """Return whether every record forms a materializable Codex session stream.

    This is stricter than :func:`looks_like`, which only needs one record to
    identify the parser. Artifact classification uses this full-stream
    contract for parser admission.
    """
    has_session_header = False
    has_message = False
    has_envelope_record = False
    has_direct_record = False

    for index, item in enumerate(payload, start=1):
        record = _dict_record(item)
        legacy_response = record is not None and _legacy_response_record(record) is not None
        bare_usage = _is_bare_token_usage_record(item)
        if not legacy_response and not bare_usage and not _is_plausibly_codex_record(item):
            return False
        if record is None or _validate_record(record, index=index, context="session stream") is None:
            return False
        record_type = _record_type(record)
        if bare_usage:
            # Counters beside ``type`` carry no generation of their own, so
            # the record joins either stream shape without marking it.
            continue
        if legacy_response:
            has_direct_record = True
            continue
        if record_type in _CODEX_SUPPORTED_OUTER_RECORD_TYPES:
            # ``session_meta`` is the shared header for both the envelope
            # stream and the legacy direct-message stream. It must not make a
            # valid header-plus-direct stream look like mixed generations.
            has_envelope_record = has_envelope_record or record_type != "session_meta"
            if not _is_envelope(record):
                return False
            session_meta = _session_meta_record(record)
            if session_meta is not None:
                has_session_header = has_session_header or _record_id(session_meta) is not None
            if _message_record(record) is not None:
                has_message = True
            continue
        if _is_state(record):
            continue
        if _is_direct_message(record):
            has_direct_record = True
            has_message = True
            continue
        session_meta = _session_meta_record(record)
        if session_meta is not None:
            has_session_header = has_session_header or _record_id(session_meta) is not None
            continue
        return False

    if not has_message:
        return False
    if has_direct_record and has_envelope_record:
        # The parser supports these wire formats independently, but their
        # records cannot be combined into one trustworthy session stream.
        return False
    # Legacy direct-message streams and headerless envelope append deltas use
    # the acquisition fallback id. A bare header remains ineligible because
    # ``has_message`` is false above.
    return has_session_header or has_direct_record or has_envelope_record


_CODEX_SUPPORTED_OUTER_RECORD_TYPES = frozenset(
    {
        "session_meta",
        "response_item",
        "event_msg",
        "compacted",
        "turn_context",
        "world_state",
        "inter_agent_communication_metadata",
        "token_usage_record",
        "retained_context",
        "realtime_item",
    }
)


def is_supported_outer_record(item: object) -> bool:
    """Whether the parser materializes this outer record rather than dropping it.

    Artifact classification applies the same predicate to every record of a
    stream, so a rollout holding a record the parser cannot materialize is
    refused as an unsupported shape instead of parsing with that record
    silently missing. It reads only ``record_type``, ``type``, ``role``,
    ``id`` and ``timestamp``, all of which candidacy projection retains.
    """
    record = _dict_record(item)
    if record is None:
        return False
    return (
        _is_state(record)
        or _record_type(record) in _CODEX_SUPPORTED_OUTER_RECORD_TYPES
        or _legacy_response_record(record) is not None
        or _is_direct_message(record)
        or (_record_id(record) is not None and _record_timestamp(record) is not None and not _record_type(record))
    )


def _codex_declared_payload_type(record: dict[str, object]) -> tuple[str, bool] | None:
    """For an envelope with a declared payload vocabulary, its payload type and membership."""
    record_type = _record_type(record)
    if record_type == "retained_context":
        vocabulary = _CODEX_RETAINED_CONTEXT_PAYLOAD_TYPES
    elif record_type == "realtime_item":
        vocabulary = _CODEX_REALTIME_PAYLOAD_TYPES
    else:
        return None
    payload = _payload_record(record)
    payload_type = _record_type(payload) if payload is not None else None
    return (payload_type or "missing", payload_type in vocabulary)


def _codex_retained_or_realtime_event(record: dict[str, object], *, index: int) -> ParsedSessionEvent | None:
    """Lower a ``retained_context`` or ``realtime_item`` record with a declared payload type."""
    declared = _codex_declared_payload_type(record)
    payload = _payload_record(record)
    if declared is None or not declared[1] or payload is None:
        return None
    event_type = declared[0]
    event_payload: dict[str, object] = {"source_index": index}
    if event_type == "verified_answer":
        for key in ("call_id", "turn_id"):
            if value := _string_value(payload.get(key)):
                event_payload[key] = value
        acceptance_order = _optional_int_field(payload, "acceptance_order")
        if acceptance_order is not None:
            event_payload["acceptance_order"] = acceptance_order
        questions = payload.get("questions")
        if isinstance(questions, list):
            # Each entry is retained whole: the question and the user's
            # answer are the content this record exists to carry.
            event_payload["questions"] = [
                {str(key): value for key, value in entry.items()} if isinstance(entry, dict) else entry
                for entry in questions
            ]
        anchor = _string_value(payload.get("call_id"))
    else:
        for key in ("id", "realtime_session_id", "outcome"):
            if value := _string_value(payload.get(key)):
                event_payload[key] = value
        anchor = None
    return ParsedSessionEvent(
        event_type=event_type,
        timestamp=_iso_or_none(_record_timestamp(record) or _record_timestamp(payload)),
        payload=event_payload,
        source_message_provider_id=anchor,
    )


def _account_codex_outer_record(
    ledger: AdmissionLedger,
    *,
    index: int,
    record: dict[str, object] | None,
) -> None:
    """Settle one source record as the materializing pass consumes it."""
    ledger.expect(AdmissionUnit.OUTER_RECORD, 1)
    record_type = _record_type(record) if record is not None else None
    payload_type = _codex_declared_payload_type(record) if record is not None else None
    if payload_type is not None and not payload_type[1]:
        # The outer type is declared but this payload type is not, so the
        # record reaches the session only as a typed unknown-record event.
        ledger.unknown(AdmissionUnit.OUTER_RECORD, index, f"{record_type}:{payload_type[0]}")
    elif is_supported_outer_record(record):
        ledger.materialized(AdmissionUnit.OUTER_RECORD, index, record_type or "direct")
    else:
        ledger.unknown(AdmissionUnit.OUTER_RECORD, index, record_type or "unsupported")


def _parse_records(
    records: Iterable[object],
    fallback_id: str,
    *,
    message_sink: MutableSequence[ParsedMessage] | None = None,
    event_sink: MutableSequence[ParsedSessionEvent] | None = None,
    _index: _CodexLookaheadIndex | None = None,
    _lookahead: _CodexLookaheadIndex | None = None,
) -> ParsedSession:
    """Parse Codex JSONL session file using typed CodexRecord model.

    Supports two format generations via CodexRecord.format_type:
    - "envelope": {"type":"session_meta"|"response_item", "payload":{...}}
    - "direct": {"type":"message", "role":"...", "content":[...]}
    - "state": {"record_type":"state"} (skip markers)

    The CodexRecord model handles format normalization via properties:
    - effective_role: Normalized role from any format
    - text_content: Extracted text from any format
    - format_type: Detected format generation
    """
    if _index is None:
        with closing(sqlite3.connect("")) as connection:
            connection.execute("PRAGMA cache_size = -8192")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            with closing(_CodexLookaheadIndex(connection)) as index_store:
                if isinstance(records, Sequence):
                    return _parse_records(
                        records,
                        fallback_id,
                        message_sink=message_sink,
                        event_sink=event_sink,
                        _index=index_store,
                    )
                # One read of the stream both retains it for replay and feeds
                # the lookahead, so the materializing pass is the only replay.
                observer = _CodexLookaheadObserver(index_store)
                in_memory = index_store.retain_records(observer.observed(records), _CODEX_REPLAY_MEMORY_BUDGET_BYTES)
                return _parse_records(
                    in_memory if in_memory is not None else index_store,
                    fallback_id,
                    message_sink=message_sink,
                    event_sink=event_sink,
                    _index=index_store,
                    _lookahead=observer.finish(),
                )

    code_mode_envelopes = _lookahead if _lookahead is not None else _codex_lookahead(records, _index)
    messages: MutableSequence[ParsedMessage] = message_sink if message_sink is not None else []
    session_events: MutableSequence[ParsedSessionEvent] = event_sink if event_sink is not None else []

    def append_message(message: ParsedMessage) -> None:
        messages.append(message.model_copy(update={"is_active_leaf": False}) if message_sink is not None else message)

    session_id = fallback_id
    session_timestamp: str | None = None
    session_timestamp_pair: _TimestampPair | None = None
    latest_message_timestamp: _TimestampPair | None = None
    session_metas_seen: list[str] = []  # Collect all session_meta IDs for parent tracking
    # Explicit lineage markers from the child's own (first) session_meta. Codex
    # records `forked_from_id` for forks/resumes and a `source.subagent.thread_spawn`
    # block for spawned subagents; both inherit the parent's context as a copied
    # prefix in this rollout. See docs/design/session-lineage-model.md.
    forked_from_id: str | None = None
    is_subagent_spawn = False
    # A spawned subagent usually records its parent only under
    # `source.subagent.thread_spawn.parent_thread_id`; `forked_from_id` is
    # present on a minority of spawns. A top-level `parent_thread_id` is the
    # third, rarest carrier. All three name the same parent thread.
    spawn_parent_thread_id: str | None = None
    meta_parent_thread_id: str | None = None
    # Structural evidence for the legacy (no forked_from_id) continuation
    # fallback below: the child's own cwd/git, and the same facts read off
    # the second distinct session_meta encountered (a resumed session
    # physically replays the parent's original session_meta as the next
    # record). Captured independently of `session_git`, which prefers the
    # *first* meta's git and must not be overwritten by the second's.
    first_meta_cwd: str | None = None
    first_meta_repo_url: str | None = None
    second_meta_timestamp_pair: _TimestampPair | None = None
    second_meta_cwd: str | None = None
    second_meta_repo_url: str | None = None
    session_git: dict[str, object] | None = None  # Git context from session metadata
    session_instructions: str | None = None  # System instructions from session metadata
    current_model_name: str | None = None
    current_model_effort: str | None = None
    message_position = 0
    previous_boundary_end = -1
    # Subagent/session identity facts that recur on every session_meta or
    # turn_context record for a given session (same value repeated per turn).
    # Captured once at first occurrence -- like session_instructions/session_git
    # above -- and emitted as a single one-time session_event after the loop,
    # rather than duplicated onto every turn_context event.
    session_agent_role: str | None = None
    session_agent_nickname: str | None = None
    session_model_provider: str | None = None
    session_developer_instructions: str | None = None
    # Every distinct instruction text seen on a ``turn_context`` after the one
    # that filled the session's own slot. Conserving these is what keeps a
    # session whose system prompt was edited mid-run from storing only the
    # prompt it started with.
    changed_user_instructions = _CodexInstructionRevisions(_index, "user")
    changed_developer_instructions = _CodexInstructionRevisions(_index, "developer")
    # Text values Codex re-embeds rather than emits on the live stream. Each is
    # registered here as it is met and resolved once, after the last event, so
    # only a value the session retains nowhere else is stored again.
    conservation = _CodexTextConservation(_index)
    admission = AdmissionLedger()

    for idx, item in enumerate(records, start=1):
        record = _dict_record(item)
        _account_codex_outer_record(admission, index=idx - 1, record=record)
        if record is None:
            continue

        # Handle compaction events (before message check so they don't fall through)
        if _record_type(record) == "compacted":
            timestamp = _iso_or_none(_record_timestamp(record))
            payload = _payload_record(record) or {}
            history = payload.get("replacement_history")
            history_list = history if isinstance(history, list) else []
            event_payload: dict[str, object] = {
                "source_index": idx,
                "summary": str(payload.get("message", "") or ""),
                "replacement_history_count": len(history_list),
            }
            # replacement_history re-embeds the pre-compaction records
            # (message/reasoning/ghost_snapshot); storing them again in full
            # would duplicate content the session already holds. What it adds
            # beyond the count is per-entry annotation Codex doesn't emit on
            # the live stream: an internal generation `phase` tag on the entry,
            # a `ghost_commit` on some entries, and inline images on content
            # items. Those are captured as bounded aggregates. Each text value
            # becomes a candidate resolved once at the end of the parse: it is
            # dropped when the session retains that value anywhere else, and
            # kept as its own event otherwise -- see
            # `_CODEX_REPLACEMENT_CONTEXT_EVENT_TYPE`.
            phase_counts: dict[str, int] = {}
            ghost_commit_count = 0
            image_count = 0
            history_text_count = 0
            boundary_start = previous_boundary_end + 1
            boundary_end = message_position - 1
            summary_text = str(event_payload["summary"])
            summary_position = message_position if summary_text else None
            compaction_event = ParsedSessionEvent(
                event_type="compaction",
                timestamp=timestamp,
                payload=event_payload,
                boundary_start_position=boundary_start,
                boundary_end_position=boundary_end,
                boundary_message_position=summary_position,
            )
            # The compaction is appended below; context events must
            # splice immediately after it, matching the historical ordering.
            insert_at = len(session_events) + 1
            for entry in history_list:
                if not isinstance(entry, dict):
                    continue
                if isinstance(entry.get("ghost_commit"), dict):
                    ghost_commit_count += 1
                phase = entry.get("phase")
                entry_phase = phase if isinstance(phase, str) and phase else None
                if entry_phase:
                    phase_counts[entry_phase] = phase_counts.get(entry_phase, 0) + 1
                entry_type = _string_value(entry.get("type"))
                entry_role = _string_value(entry.get("role"))
                entry_content = entry.get("content")
                if isinstance(entry_content, list):
                    for content_item in entry_content:
                        if not isinstance(content_item, dict):
                            continue
                        if isinstance(content_item.get("image_url"), str | dict):
                            image_count += 1
                        content_text = content_item.get("text")
                        if not isinstance(content_text, str) or not content_text:
                            continue
                        history_text_count += 1
                        candidate_key, is_new = conservation.add(content_text)
                        if is_new and candidate_key is not None:
                            conservation.add_context(
                                candidate_key,
                                insert_at=insert_at,
                                timestamp=timestamp,
                                source_index=idx,
                                entry_type=entry_type,
                                role=entry_role,
                                phase=entry_phase,
                            )
            if history_text_count:
                compaction_event.payload["replacement_history_text_count"] = history_text_count
            if phase_counts:
                compaction_event.payload["replacement_history_phase_counts"] = dict(sorted(phase_counts.items()))
            if ghost_commit_count:
                compaction_event.payload["replacement_history_ghost_commit_count"] = ghost_commit_count
            if image_count:
                compaction_event.payload["replacement_history_image_count"] = image_count
            session_events.append(compaction_event)
            # Context events are spliced in directly after their own compaction
            # event, so a reader meets the text where the compaction dropped it.
            # Materialize the compaction summary as a real message at the
            # boundary, mirroring Claude Code, so both providers present a uniform
            # summary message that replaces the prior context (#2467). The
            # pre-compaction messages stay stored once; the boundary marks where
            # context discontinues.
            if summary_text:
                append_message(
                    ParsedMessage(
                        provider_message_id="",
                        role=Role.SYSTEM,
                        text=summary_text,
                        timestamp=timestamp,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=summary_text)],
                        message_type=MessageType.SUMMARY,
                        position=message_position,
                        variant_index=0,
                        is_active_path=True,
                    )
                )
                message_position += 1
            previous_boundary_end = boundary_end
            continue

        # Handle turn-context events
        if _record_type(record) == "turn_context":
            timestamp = _iso_or_none(_record_timestamp(record))
            tc_payload: dict[str, object] = {}
            turn_payload = _payload_record(record)
            if turn_payload:
                tc_payload["source_index"] = idx
                normalized_turn_context = _turn_context_payload(turn_payload)
                cwd = _extract_cwd(turn_payload)
                if cwd:
                    tc_payload["cwd"] = cwd
                    _index.add_workdir(cwd)
                if model_name := _string_field(normalized_turn_context, "model", "model_name"):
                    current_model_name = model_name
                    tc_payload["model"] = model_name
                if model_effort := _string_field(normalized_turn_context, "effort", "model_effort"):
                    current_model_effort = model_effort
                    tc_payload["effort"] = model_effort
                if turn_id := _string_field(normalized_turn_context, "turn_id"):
                    tc_payload["turn_id"] = turn_id
                # `personality`/`summary`/`collaboration_mode` were unread
                # (polylogue-cgfy triage, codex lane): agent-persona,
                # reasoning-summary-verbosity, and collaboration-mode knobs
                # reported on every turn_context alongside model/effort, but
                # never carried through. `collaboration_mode.settings`
                # duplicates model/effort/developer_instructions already
                # captured from the top-level turn_context -- only the mode
                # name itself is new.
                if personality := _string_field(normalized_turn_context, "personality"):
                    tc_payload["personality"] = personality
                if reasoning_summary := _string_field(normalized_turn_context, "summary"):
                    tc_payload["reasoning_summary"] = reasoning_summary
                collaboration_mode_raw = _dict_record(normalized_turn_context.get("collaboration_mode"))
                if collaboration_mode_raw is not None:
                    collaboration_mode_name = _string_field(collaboration_mode_raw, "mode")
                    if collaboration_mode_name:
                        tc_payload["collaboration_mode"] = collaboration_mode_name
                # Emit agent_policy event when policy fields are present.
                # Payload keys match what _write_session_events expects via
                # _payload_string(event.payload, "approval_policy") etc.
                approval_policy = _string_field(normalized_turn_context, "approval_policy")
                sandbox_raw = normalized_turn_context.get("sandbox_policy")
                if approval_policy or sandbox_raw is not None:
                    policy_payload: dict[str, object] = {}
                    if approval_policy:
                        policy_payload["approval_policy"] = approval_policy
                    if isinstance(sandbox_raw, dict):
                        # Older Codex CLI builds key the sandbox kind as
                        # "mode" (workspace-write, read-only); newer builds
                        # use "type" (e.g. danger-full-access). Both are the
                        # same fact under different wire spellings.
                        mode = _string_field(sandbox_raw, "mode", "type")
                        if mode:
                            policy_payload["sandbox_policy"] = mode
                        network_val = sandbox_raw.get("network_access")
                        if network_val is not None:
                            policy_payload["network_policy"] = str(network_val).lower()
                        for flag_key in ("exclude_slash_tmp", "exclude_tmpdir_env_var"):
                            flag_val = sandbox_raw.get(flag_key)
                            if isinstance(flag_val, bool):
                                policy_payload[flag_key] = flag_val
                    elif isinstance(sandbox_raw, str) and sandbox_raw:
                        policy_payload["sandbox_policy"] = sandbox_raw
                    if policy_payload:
                        session_events.append(
                            ParsedSessionEvent(
                                event_type="agent_policy",
                                timestamp=timestamp,
                                payload=policy_payload,
                            )
                        )
                # Truncation policy and structured-output schema are small,
                # bounded turn-scoped config -- safe to carry on every
                # turn_context event (unlike the large instruction texts
                # below, they legitimately vary turn to turn).
                truncation_policy = _dict_record(normalized_turn_context.get("truncation_policy"))
                if truncation_policy:
                    tc_payload["truncation_policy"] = dict(truncation_policy)
                final_output_schema = _dict_record(normalized_turn_context.get("final_output_json_schema"))
                if final_output_schema:
                    tc_payload["final_output_json_schema"] = dict(final_output_schema)
                # `user_instructions` is the session-level system prompt
                # (CLAUDE.md/AGENTS.md-style content), re-declared on every
                # turn in this record generation. The first value fills the
                # same slot the legacy per-session `instructions` field uses,
                # so an unchanged prompt is stored once rather than per turn;
                # a value distinct from every one already seen is a real edit
                # and becomes its own event.
                user_instructions = _string_value(normalized_turn_context.get("user_instructions"))
                if user_instructions:
                    if not session_instructions:
                        session_instructions = user_instructions
                    elif user_instructions != session_instructions and (
                        user_instructions not in changed_user_instructions
                    ):
                        changed_user_instructions.add(user_instructions)
                        session_events.append(
                            _codex_instructions_changed_event(
                                kind="user_instructions",
                                instructions=user_instructions,
                                revision=len(changed_user_instructions) + 1,
                                timestamp=timestamp,
                                source_index=idx,
                                effective_from_message_position=message_position,
                            )
                        )
                # `developer_instructions` is a distinct, usually
                # subagent-role-specific prompt (e.g. "You are an awaiter.").
                # Its first value rides the one-time identity event alongside
                # agent_role/agent_nickname below; later distinct values are
                # conserved the same way as the user prompt.
                developer_instructions = _string_value(normalized_turn_context.get("developer_instructions"))
                if developer_instructions:
                    if not session_developer_instructions:
                        session_developer_instructions = developer_instructions
                    elif developer_instructions != session_developer_instructions and (
                        developer_instructions not in changed_developer_instructions
                    ):
                        changed_developer_instructions.add(developer_instructions)
                        session_events.append(
                            _codex_instructions_changed_event(
                                kind="developer_instructions",
                                instructions=developer_instructions,
                                revision=len(changed_developer_instructions) + 1,
                                timestamp=timestamp,
                                source_index=idx,
                                effective_from_message_position=message_position,
                            )
                        )
            session_events.append(
                ParsedSessionEvent(
                    event_type="turn_context",
                    timestamp=timestamp,
                    payload=tc_payload,
                )
            )
            continue

        legacy_response = _legacy_response_record(record)
        if legacy_response is not None or _record_type(record) in {"response_item", "event_msg"}:
            inner = legacy_response if legacy_response is not None else _payload_record(record)
            if inner is not None and not _is_message(inner):
                event_payload = _compact_response_payload(
                    inner,
                    index=idx,
                    current_model_name=current_model_name,
                    current_model_effort=current_model_effort,
                )
                event_type = _codex_response_item_event_type(_record_type(inner), _record_type(record))
                if event_type == _CODEX_UNCLASSIFIED_RESPONSE_ITEM_TYPE:
                    event_payload["wire_type"] = _record_type(inner) or _record_type(record)
                timestamp_fallback = _record_timestamp(record)
                tool_message = _codex_tool_message(
                    inner,
                    index=idx,
                    position=message_position,
                    timestamp_fallback=timestamp_fallback,
                    exec_envelope=code_mode_envelopes.get(idx),
                )
                # The event names the message parsed from the same record, so
                # a call and its output never share one event owner.
                response_event = ParsedSessionEvent(
                    event_type=event_type,
                    timestamp=_iso_or_none(_record_timestamp(inner) or _record_timestamp(record)),
                    payload=event_payload,
                    source_message_provider_id=(
                        tool_message.provider_message_id or None
                        if tool_message is not None
                        else _string_value(inner.get("client_id") or inner.get("id") or inner.get("call_id"))
                    ),
                )
                session_events.append(response_event)
                # Usage lowering keeps numeric counters in its typed table.
                # Quota windows have their own timeline event so that lowering
                # the token_count row cannot discard this independent evidence.
                if event_type == "token_count" and event_payload.get("rate_limits"):
                    session_events.append(
                        ParsedSessionEvent(
                            event_type="rate_limits",
                            timestamp=response_event.timestamp,
                            payload={"source_index": idx, "rate_limits": event_payload["rate_limits"]},
                        )
                    )
                # `task_complete.last_agent_message` is the turn's final
                # assistant text repeated on the completion marker. Measured
                # over 270 real rollout files, all 1,565 occurrences were
                # already stored from the same file's live stream -- so it is
                # registered as a candidate and only stored when that does not
                # hold, rather than trusted to be a duplicate.
                if _record_type(inner) == "task_complete":
                    last_agent_message = inner.get("last_agent_message")
                    if isinstance(last_agent_message, str) and last_agent_message:
                        conservation.add_task_completion(last_agent_message, len(session_events) - 1)
                if tool_message is not None:
                    append_message(tool_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, tool_message.timestamp)
                    # code_mode_envelopes maps BOTH the call record's index
                    # and the matching output record's index to the same
                    # (results-enriched) envelope object -- only emit the
                    # child-result evidence once, on the output/tool_result
                    # message, not again on the call/tool_use message.
                    if _record_type(inner) in {
                        "function_call_output",
                        "custom_tool_call_output",
                        "tool_search_output",
                        "web_search_output",
                    }:
                        exec_envelope_for_message = code_mode_envelopes.get(idx)
                        if exec_envelope_for_message is not None and exec_envelope_for_message.results:
                            session_events.extend(
                                _code_mode_child_result_evidence_events(
                                    exec_envelope_for_message,
                                    source_message_provider_id=tool_message.provider_message_id,
                                    timestamp=tool_message.timestamp,
                                )
                            )
                event_message = _codex_event_message(
                    inner,
                    index=idx,
                    position=message_position,
                    echo_index=code_mode_envelopes,
                    timestamp_fallback=timestamp_fallback,
                )
                if event_message is not None:
                    append_message(event_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, event_message.timestamp)
                reasoning_message = _codex_reasoning_message(
                    inner,
                    index=idx,
                    position=message_position,
                    timestamp_fallback=timestamp_fallback,
                )
                if reasoning_message is not None:
                    append_message(reasoning_message)
                    message_position += 1
                    latest_message_timestamp = _newer_timestamp(latest_message_timestamp, reasoning_message.timestamp)
                mcp_messages = _codex_mcp_tool_call_messages(
                    inner,
                    index=idx,
                    position=message_position,
                    timestamp_fallback=timestamp_fallback,
                )
                if mcp_messages is not None:
                    for mcp_message in mcp_messages:
                        append_message(mcp_message)
                    message_position += len(mcp_messages)
                    for mcp_message in mcp_messages:
                        latest_message_timestamp = _newer_timestamp(latest_message_timestamp, mcp_message.timestamp)
                cwd = _extract_cwd(event_payload)
                if cwd:
                    _index.add_workdir(cwd)
                continue

        # These newer producer records are top-level envelopes rather than
        # response_item/event_msg payloads. Keep them on the same normalized
        # session_events route, with the bounded compactor and no raw dump.
        if _record_type(record) in {"inter_agent_communication_metadata", "token_usage_record"}:
            top_level_payload = _payload_record(record) or _record_payload(record)
            event_payload = _compact_response_payload(
                {"type": _record_type(record), **top_level_payload},
                index=idx,
            )
            session_events.append(
                ParsedSessionEvent(
                    event_type=_record_type(record) or "codex_unknown_outer_record",
                    timestamp=_iso_or_none(_record_timestamp(record) or _record_timestamp(top_level_payload)),
                    payload=event_payload,
                    source_message_provider_id=_string_value(top_level_payload.get("id")),
                )
            )
            continue

        if _record_type(record) in {"retained_context", "realtime_item"}:
            retained_event = _codex_retained_or_realtime_event(record, index=idx)
            if retained_event is not None:
                session_events.append(retained_event)
                continue
            # A payload type outside the declared vocabulary falls through to
            # the typed unknown-record event below; the admission ledger
            # already recorded it as an unrecognized type.

        # World-state snapshots (full or delta) report ambient runtime
        # context -- most notably the live subagent roster
        # (`state.environments.subagents`) -- outside the
        # session_meta/turn_context/response_item shapes handled above, so
        # they previously fell through the whole dispatch chain unrecorded.
        # Every `state` key is carried except the declared context-file text
        # in ``_WORLD_STATE_INSTRUCTION_TEXT_KEYS`` (polylogue-w54q3).
        if _record_type(record) == "world_state":
            world_payload = _payload_record(record) or {}
            state = _dict_record(world_payload.get("state"))
            retained = (
                {key: value for key, value in state.items() if key not in _WORLD_STATE_INSTRUCTION_TEXT_KEYS}
                if state
                else {}
            )
            if retained:
                session_events.append(
                    ParsedSessionEvent(
                        event_type="world_state",
                        timestamp=_iso_or_none(_record_timestamp(record)),
                        payload={"source_index": idx, **retained},
                    )
                )
            continue

        session_meta = _session_meta_record(record)
        if session_meta is not None:
            meta_id = _record_id(session_meta)
            if meta_id and len(session_metas_seen) < 2 and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                if len(session_metas_seen) == 1:
                    session_id = meta_id
                    session_timestamp_pair = parse_timestamp_pair(_record_timestamp(session_meta))
                    session_timestamp = session_timestamp_pair[1] if session_timestamp_pair is not None else None
                    # Lineage markers live on the child's own (first) meta only.
                    forked_val = session_meta.get("forked_from_id")
                    if isinstance(forked_val, str) and forked_val.strip():
                        forked_from_id = forked_val.strip()
                    source_val = session_meta.get("source")
                    if isinstance(source_val, dict) and isinstance(source_val.get("subagent"), dict):
                        is_subagent_spawn = True
                        thread_spawn = source_val["subagent"].get("thread_spawn")
                        if isinstance(thread_spawn, dict):
                            spawn_parent = thread_spawn.get("parent_thread_id")
                            if isinstance(spawn_parent, str) and spawn_parent.strip():
                                spawn_parent_thread_id = spawn_parent.strip()
                    meta_parent = session_meta.get("parent_thread_id")
                    if isinstance(meta_parent, str) and meta_parent.strip():
                        meta_parent_thread_id = meta_parent.strip()
                    cwd_val = session_meta.get("cwd")
                    if isinstance(cwd_val, str) and cwd_val.strip():
                        first_meta_cwd = cwd_val.strip()
                    first_meta_git = _git_context(session_meta)
                    if first_meta_git is not None:
                        repo_val = first_meta_git.get("repository_url")
                        if isinstance(repo_val, str) and repo_val.strip():
                            first_meta_repo_url = repo_val.strip()
                elif len(session_metas_seen) == 2:
                    # The second distinct session_meta is the legacy-fallback
                    # candidate parent (see the CONTINUATION classification
                    # below) -- capture its own facts independently of
                    # `session_git`/`session_timestamp`, which track the
                    # first (child) meta only.
                    second_meta_timestamp_pair = parse_timestamp_pair(_record_timestamp(session_meta))
                    cwd_val = session_meta.get("cwd")
                    if isinstance(cwd_val, str) and cwd_val.strip():
                        second_meta_cwd = cwd_val.strip()
                    second_meta_git = _git_context(session_meta)
                    if second_meta_git is not None:
                        repo_val = second_meta_git.get("repository_url")
                        if isinstance(repo_val, str) and repo_val.strip():
                            second_meta_repo_url = repo_val.strip()
            git_context = _git_context(session_meta)
            if git_context and not session_git:
                session_git = git_context
            instructions = _record_instructions(session_meta)
            if not instructions:
                # Newer session_meta records carry `base_instructions` as a
                # {"text": ...} wrapper instead of the legacy flat string.
                base_instructions = _dict_record(session_meta.get("base_instructions"))
                if base_instructions:
                    instructions = _string_value(base_instructions.get("text"))
            if instructions and not session_instructions:
                session_instructions = instructions
            if not session_agent_role:
                session_agent_role = _string_field(session_meta, "agent_role")
            if not session_agent_nickname:
                session_agent_nickname = _string_field(session_meta, "agent_nickname")
            if not session_model_provider:
                session_model_provider = _string_field(session_meta, "model_provider")
            continue

        message_record = _message_record(record)
        if message_record is not None:
            raw_role = _effective_role(message_record)
            content = _effective_content(message_record)
            text = extract_codex_text(content)
            inline_image_blocks = _codex_inline_image_blocks(content)
            timestamp_pair = parse_timestamp_pair(_message_timestamp(record, message_record))
            timestamp = timestamp_pair[1] if timestamp_pair is not None else None

            content_blocks = content_blocks_from_segments(
                content,
                admission=admission,
                lower_transport_text=True,
            )
            content_blocks.extend(inline_image_blocks)
            if inline_image_blocks:
                admission.expect(AdmissionUnit.BLOCK, len(inline_image_blocks))
                for inline_block in inline_image_blocks:
                    admission.materialized(
                        AdmissionUnit.BLOCK,
                        admission.next_ordinal(AdmissionUnit.BLOCK),
                        inline_block.type.value,
                    )
            if not raw_role or raw_role == "unknown":
                continue
            if not text and not content_blocks:
                continue
            role = Role.normalize(raw_role)

            msg_id = _record_id(message_record) or ""
            if not content_blocks and text:
                content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            token_usage = _token_usage(message_record)
            model_name = _string_field(message_record, "model", "model_name") or current_model_name
            model_effort = _string_field(message_record, "effort", "model_effort") or current_model_effort
            duration_ms = _optional_int_field(message_record, "duration_ms", "durationMs", "elapsed_ms")

            message_type = _message_type_from_codex_message(message_record, text)
            append_message(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    blocks=content_blocks,
                    message_type=message_type,
                    material_origin=_codex_material_origin(role, message_type, text),
                    position=message_position,
                    variant_index=0,
                    is_active_path=True,
                    input_tokens=token_usage["input_tokens"],
                    output_tokens=token_usage["output_tokens"],
                    cache_read_tokens=token_usage["cache_read_tokens"],
                    cache_write_tokens=token_usage["cache_write_tokens"],
                    model_name=model_name,
                    model_effort=model_effort,
                    duration_ms=duration_ms,
                )
            )
            message_semantics = _compact_response_payload(message_record, index=idx)
            # A response_item message is lowered into the message tree, but
            # phase/turn/image-reference evidence belongs to session_events so
            # it survives archive writes and public event reads as well.
            retained_message_keys = {
                key: value
                for key, value in message_semantics.items()
                if key
                in {
                    "phase",
                    "turn_id",
                    "turn_id_source",
                    "turn_id_conflict",
                    "local_images",
                    "text_elements",
                }
            }
            if retained_message_keys:
                session_events.append(
                    ParsedSessionEvent(
                        event_type="response_item",
                        timestamp=timestamp,
                        payload={"source_index": idx, **retained_message_keys},
                        source_message_provider_id=msg_id or None,
                    )
                )
            message_position += 1
            latest_message_timestamp = _newer_timestamp_pair(latest_message_timestamp, timestamp_pair)
            continue

        # ``state`` is a supported non-conversational marker. Every other
        # outer envelope that reaches this point is still an input unit: keep
        # a typed record instead of silently accepting a session with an
        # unaccounted suffix (polylogue-vslhb).
        if _is_state(record):
            continue
        session_events.append(
            ParsedSessionEvent(
                event_type="codex_unknown_outer_record",
                timestamp=_iso_or_none(_record_timestamp(record)),
                payload={
                    "source_index": idx,
                    "wire_type": _record_type(record) or "unknown",
                    "record": _record_payload(record),
                },
            )
        )

    # Emit the deduped subagent/session identity facts (agent_role,
    # agent_nickname, model_provider from session_meta; developer_instructions
    # from turn_context) once, if any were observed, instead of once per
    # session_meta/turn_context occurrence.
    identity_payload: dict[str, object] = {}
    if session_agent_role:
        identity_payload["agent_role"] = session_agent_role
    if session_agent_nickname:
        identity_payload["agent_nickname"] = session_agent_nickname
    if session_model_provider:
        identity_payload["model_provider"] = session_model_provider
    if session_developer_instructions:
        identity_payload["developer_instructions"] = session_developer_instructions
    if identity_payload:
        session_events.append(
            ParsedSessionEvent(
                event_type="codex_agent_identity",
                timestamp=session_timestamp,
                payload=identity_payload,
            )
        )

    # Resolve the re-embedded text candidates now that every message and event
    # this session stores exists. A candidate the session already retains is
    # dropped -- linking it again would duplicate content; the rest is stored
    # exactly once, under the compaction or completion record that carried it.
    conservation.resolve(
        messages=messages,
        events=session_events,
        instructions_text=session_instructions,
    )
    conservation.finish_task_completions(session_events)
    conservation.finish_replacement_contexts(session_events)

    # Lineage: prefer the explicit markers on the child's own session_meta.
    #   - `source.subagent.thread_spawn` → spawned subagent (positive evidence
    #     of a subagent relationship): assign SUBAGENT. The parent id comes
    #     from whichever carrier is present -- `forked_from_id`,
    #     `thread_spawn.parent_thread_id`, or a top-level `parent_thread_id`;
    #     a spawn block alone is the common shape and names no other parent.
    #   - `forked_from_id` (no subagent block) → the child shares the parent's
    #     leading context prefix, but Codex sets this field for BOTH a divergent
    #     user fork AND a plain resume of the same thread. The marker proves a
    #     parent, not the relationship *type*. Assigning FORK here over-claimed:
    #     a resume was recorded as a fork. Leave the type unclassified
    #     (`None` → generic topology link, no `sessions.branch_type`) rather
    #     than fabricate FORK from absent evidence. The prefix-sharing
    #     normalization still records the branch point + `inheritance`, so the
    #     shared-prefix fact is preserved.
    # Fall back to the legacy heuristic (older exports with no `forked_from_id`
    # field at all) when no explicit marker is present: a second distinct
    # session_meta *can* be the replayed parent header of a plain resume, but
    # only when it carries the structural evidence of that -- see
    # `_has_continuation_evidence`. A second session_meta id with none of that
    # evidence is not proof of any relationship (e.g. two structurally
    # unrelated session_metas concatenated in one payload), so it stays fully
    # unclassified rather than fabricating CONTINUATION from a bare count.
    explicit_parent_id = forked_from_id or spawn_parent_thread_id or meta_parent_thread_id
    if explicit_parent_id is not None:
        parent_id: str | None = explicit_parent_id
        branch_type = BranchType.SUBAGENT if is_subagent_spawn else None
    elif len(session_metas_seen) > 1 and _has_continuation_evidence(
        first_timestamp=session_timestamp_pair,
        second_timestamp=second_meta_timestamp_pair,
        first_cwd=first_meta_cwd,
        second_cwd=second_meta_cwd,
        first_repo_url=first_meta_repo_url,
        second_repo_url=second_meta_repo_url,
    ):
        parent_id = session_metas_seen[1]
        branch_type = BranchType.CONTINUATION
    else:
        parent_id = None
        branch_type = None

    updated_at_pair = _newer_timestamp_pair(session_timestamp_pair, latest_message_timestamp)

    git_branch_typed: str | None = None
    git_repo_url_typed: str | None = None
    git_commit_hash_typed: str | None = None
    if session_git is not None:
        branch_val = session_git.get("branch")
        if isinstance(branch_val, str) and branch_val.strip():
            git_branch_typed = branch_val.strip()
        repo_val = session_git.get("repository_url")
        if isinstance(repo_val, str) and repo_val.strip():
            git_repo_url_typed = repo_val.strip()
        # commit_hash pins the session to an exact commit — the strongest
        # attribution signal codex provides. Previously kept only inside
        # provider_meta.git where downstream readers had to JSON-extract;
        # now graduated to a typed top-level field.
        commit_val = session_git.get("commit_hash")
        if isinstance(commit_val, str) and commit_val.strip():
            git_commit_hash_typed = commit_val.strip()
    unit_accounting = admission.close()

    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title=session_id,
        created_at=session_timestamp,
        updated_at=updated_at_pair[1] if updated_at_pair is not None else None,
        messages=cast(list[ParsedMessage], messages) if message_sink is None else [],
        active_leaf_message_provider_id=None,
        session_events=cast(list[ParsedSessionEvent], session_events) if event_sink is None else [],
        parent_session_provider_id=parent_id,
        branch_type=branch_type,
        instructions_text=session_instructions,
        working_directories=_index.working_directories(),
        git_branch=git_branch_typed,
        git_repository_url=git_repo_url_typed,
        git_commit_hash=git_commit_hash_typed,
        unit_accounting=unit_accounting,
    )
    updates: dict[str, object] = {}
    if message_sink is not None:
        updates["messages"] = messages
    if event_sink is not None:
        updates["session_events"] = session_events
    if updates:
        session = session.model_copy(update=updates)
    return finalize_codex_session(
        session,
        messages=messages,
        session_events=session_events,
        updated_at=updated_at_pair[1] if updated_at_pair is not None else None,
        unit_accounting=unit_accounting,
        mark_active_leaf=True,
    )


@parser_admission("codex", scan=codex_unknown_wire_type)
def parse(payload: Sequence[object], fallback_id: str) -> ParsedSession:
    return _parse_records(payload, fallback_id)


def parse_stream(
    records: Iterable[object],
    fallback_id: str,
    *,
    message_sink: MutableSequence[ParsedMessage] | None = None,
    event_sink: MutableSequence[ParsedSessionEvent] | None = None,
) -> ParsedSession:
    return _parse_records(records, fallback_id, message_sink=message_sink, event_sink=event_sink)


def detection_projection() -> DetectorProjection:
    """Preserve typed record admission without retaining payload or block bodies."""
    scalar = DetectorProjection()
    content = DetectorProjection(item=scalar, array_fold="all", array_predicate=lambda item: isinstance(item, dict))
    return DetectorProjection(
        fields={
            **dict.fromkeys(("type", "record_type", "role", "id", "instructions", "payload", "timestamp"), scalar),
            "content": content,
            "git": DetectorProjection(fields=dict.fromkeys(("commit_hash", "branch", "repository_url"), scalar)),
            "message": None,
        }
    )
