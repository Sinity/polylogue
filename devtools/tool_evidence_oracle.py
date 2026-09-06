"""Read tool calls and results straight off provider wire records.

An independent statement of what a source declares, used to judge the archive
rather than to build it. It reads the raw payload shapes directly and shares
no code with the parsers, so a parser that loses a result, mislabels its
owner, duplicates it, or rewrites its outcome disagrees with this reading
instead of moving both sides of the comparison together.

Each declared call carries the wire's own identity where the provider supplies
one and a positional key where it does not, so a source whose calls are
anonymous is still countable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

#: The origins whose wire shapes this module reads. An origin outside the set
#: raises rather than returning a silently empty reading.
SUPPORTED_ORIGINS: frozenset[str] = frozenset(
    {
        "chatgpt-export",
        "claude-ai-export",
        "claude-code-session",
        "codex-session",
    }
)

#: Outcome vocabulary, matching ``polylogue.core.enums.ToolOutcome`` without
#: importing it -- the oracle must not inherit the production enum's meaning.
OK = "ok"
ERROR = "error"
UNKNOWN = "unknown"
NO_RESULT = "no_result"


@dataclass(frozen=True, slots=True)
class DeclaredCall:
    """One tool invocation the source states, with the answer it states for it."""

    key: str
    tool_name: str | None
    outcome: str
    result_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceToolEvidence:
    """Every call and physical result record one source declares."""

    origin: str
    calls: tuple[DeclaredCall, ...] = ()
    #: Result records that name no call the source also declares.
    orphan_results: int = 0

    @property
    def result_records(self) -> int:
        return sum(call.result_count for call in self.calls) + self.orphan_results

    def by_key(self) -> dict[str, DeclaredCall]:
        return {call.key: call for call in self.calls}


@dataclass(frozen=True, slots=True)
class ArchivedAction:
    """One row of the archive's canonical actions relation."""

    tool_use_block_id: str
    tool_id: str | None
    tool_name: str | None
    result_state: str
    tool_result_block_id: str | None


@dataclass(frozen=True, slots=True)
class ConservationVerdict:
    """How the archive's actions differ from what the source declares."""

    origin: str
    declared_calls: int
    archived_calls: int
    declared_results: int
    archived_results: int
    lost_results: tuple[str, ...] = field(default=())
    synthesized_results: tuple[str, ...] = field(default=())
    duplicated_results: tuple[str, ...] = field(default=())
    changed_owners: tuple[str, ...] = field(default=())
    outcome_disagreements: tuple[str, ...] = field(default=())
    missing_calls: tuple[str, ...] = field(default=())

    @property
    def conserved(self) -> bool:
        return not (
            self.lost_results
            or self.synthesized_results
            or self.duplicated_results
            or self.changed_owners
            or self.outcome_disagreements
            or self.missing_calls
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "declared_calls": self.declared_calls,
            "archived_calls": self.archived_calls,
            "declared_results": self.declared_results,
            "archived_results": self.archived_results,
            "conserved": self.conserved,
            "lost_results": list(self.lost_results),
            "synthesized_results": list(self.synthesized_results),
            "duplicated_results": list(self.duplicated_results),
            "changed_owners": list(self.changed_owners),
            "outcome_disagreements": list(self.outcome_disagreements),
            "missing_calls": list(self.missing_calls),
        }


def declare_tool_evidence(payload: object, *, origin: str) -> SourceToolEvidence:
    """Read one source payload's declared tool calls and results."""
    if origin not in SUPPORTED_ORIGINS:
        raise ValueError(f"tool evidence oracle has no reading for origin {origin!r}")
    if origin == "chatgpt-export":
        return _chatgpt_evidence(payload)
    if origin == "claude-ai-export":
        return _claude_web_evidence(payload)
    if origin == "claude-code-session":
        return _claude_code_evidence(payload)
    return _codex_evidence(payload)


# --------------------------------------------------------------------------
# ChatGPT export: a mapping tree. An assistant node addressed to a recipient
# other than ``all`` -- or carrying code-interpreter input -- is the call; the
# ``role: tool`` nodes hanging below it are its answers, chained through each
# other when the tool returned more than one record.
# --------------------------------------------------------------------------


def _chatgpt_evidence(payload: object) -> SourceToolEvidence:
    mapping = payload.get("mapping") if isinstance(payload, Mapping) else None
    if not isinstance(mapping, Mapping):
        return SourceToolEvidence(origin="chatgpt-export")
    calls: dict[str, tuple[str | None, list[Mapping[str, object]]]] = {}
    for node_id, node in mapping.items():
        message = _node_message(node)
        if message is None:
            continue
        if _chatgpt_is_call(message):
            recipient = message.get("recipient")
            calls[str(node_id)] = (
                str(recipient) if isinstance(recipient, str) and recipient != "all" else "code_interpreter",
                [],
            )
    orphans = 0
    for node in mapping.values():
        message = _node_message(node)
        if message is None or not _chatgpt_is_tool_role(message):
            continue
        owner = _chatgpt_owning_call(mapping, node, calls)
        if owner is None:
            orphans += 1
            continue
        calls[owner][1].append(message)
    declared = tuple(
        DeclaredCall(
            key=key,
            tool_name=name,
            outcome=_chatgpt_outcome(results),
            result_count=len(results),
        )
        for key, (name, results) in sorted(calls.items())
    )
    return SourceToolEvidence(origin="chatgpt-export", calls=declared, orphan_results=orphans)


def _node_message(node: object) -> Mapping[str, object] | None:
    if not isinstance(node, Mapping):
        return None
    message = node.get("message")
    return message if isinstance(message, Mapping) else None


def _chatgpt_role(message: Mapping[str, object]) -> str | None:
    author = message.get("author")
    role = author.get("role") if isinstance(author, Mapping) else None
    return str(role) if isinstance(role, str) else None


def _chatgpt_is_tool_role(message: Mapping[str, object]) -> bool:
    return _chatgpt_role(message) == "tool"


def _chatgpt_is_call(message: Mapping[str, object]) -> bool:
    if _chatgpt_role(message) == "tool":
        return False
    content = message.get("content")
    if isinstance(content, Mapping) and content.get("content_type") == "code":
        return True
    recipient = message.get("recipient")
    return isinstance(recipient, str) and bool(recipient) and recipient != "all"


def _chatgpt_owning_call(
    mapping: Mapping[str, object],
    node: Mapping[str, object],
    calls: Mapping[str, object],
) -> str | None:
    seen: set[str] = set()
    parent = node.get("parent")
    current = str(parent) if parent else None
    while current and current not in seen:
        seen.add(current)
        if current in calls:
            return current
        ancestor = mapping.get(current)
        message = _node_message(ancestor)
        if message is None or not _chatgpt_is_tool_role(message):
            return None
        parent = ancestor.get("parent") if isinstance(ancestor, Mapping) else None
        current = str(parent) if parent else None
    return None


def _chatgpt_outcome(results: Sequence[Mapping[str, object]]) -> str:
    if not results:
        return NO_RESULT
    for message in results:
        content = message.get("content")
        if isinstance(content, Mapping) and content.get("content_type") == "system_error":
            return ERROR
    statuses = {message.get("status") for message in results}
    if "finished_partial_completion" in statuses:
        return ERROR
    if statuses == {"finished_successfully"}:
        return OK
    return UNKNOWN


# --------------------------------------------------------------------------
# Claude web export: one conversation, each message a ``content`` array. A
# ``tool_use`` segment is answered by the next ``tool_result`` segment of the
# same name inside the same message; neither side carries an id.
# --------------------------------------------------------------------------


def _claude_web_evidence(payload: object) -> SourceToolEvidence:
    messages = payload.get("chat_messages") if isinstance(payload, Mapping) else None
    if not isinstance(messages, Sequence):
        return SourceToolEvidence(origin="claude-ai-export")
    calls: list[DeclaredCall] = []
    orphans = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        segments = message.get("content")
        if not isinstance(segments, Sequence):
            continue
        pending: list[int] = []
        local: list[list[object]] = []
        message_key = str(message.get("uuid") or message.get("id") or index)
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            segment_type = segment.get("type")
            if segment_type == "tool_use":
                pending.append(len(local))
                local.append([str(segment.get("name") or "") or None, segment.get("id"), None])
            elif segment_type == "tool_result":
                name = segment.get("name")
                match = next(
                    (
                        candidate
                        for candidate in reversed(pending)
                        if not (name and local[candidate][0]) or name == local[candidate][0]
                    ),
                    None,
                )
                if match is None:
                    orphans += 1
                    continue
                pending.remove(match)
                local[match][2] = segment
        for ordinal, (name, wire_id, result) in enumerate(local):
            key = str(wire_id) if isinstance(wire_id, str) and wire_id else f"{message_key}:{ordinal}"
            calls.append(
                DeclaredCall(
                    key=key,
                    tool_name=name if isinstance(name, str) else None,
                    outcome=_claude_web_outcome(result),
                    result_count=0 if result is None else 1,
                )
            )
    return SourceToolEvidence(origin="claude-ai-export", calls=tuple(calls), orphan_results=orphans)


def _claude_web_outcome(result: object) -> str:
    if not isinstance(result, Mapping):
        return NO_RESULT
    is_error = result.get("is_error")
    if isinstance(is_error, bool):
        return ERROR if is_error else OK
    return UNKNOWN


# --------------------------------------------------------------------------
# Claude Code session JSONL: the Anthropic protocol on the wire. An assistant
# record carries ``tool_use`` segments; a following user record carries the
# ``tool_result`` segments that name them by ``tool_use_id``.
# --------------------------------------------------------------------------


def _claude_code_evidence(payload: object) -> SourceToolEvidence:
    records = _record_sequence(payload)
    calls: dict[str, tuple[str | None, list[Mapping[str, object]]]] = {}
    pending_results: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        message = record.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for segment in content:
            if not isinstance(segment, Mapping):
                continue
            if segment.get("type") == "tool_use":
                tool_id = segment.get("id")
                if isinstance(tool_id, str) and tool_id:
                    name = segment.get("name")
                    calls[tool_id] = (str(name) if isinstance(name, str) else None, [])
            elif segment.get("type") == "tool_result":
                tool_id = segment.get("tool_use_id")
                if isinstance(tool_id, str) and tool_id:
                    pending_results.setdefault(tool_id, []).append(segment)
    orphans = 0
    for tool_id, results in pending_results.items():
        if tool_id in calls:
            calls[tool_id][1].extend(results)
        else:
            orphans += len(results)
    declared = tuple(
        DeclaredCall(
            key=key,
            tool_name=name,
            outcome=_anthropic_outcome(results),
            result_count=len(results),
        )
        for key, (name, results) in sorted(calls.items())
    )
    return SourceToolEvidence(origin="claude-code-session", calls=declared, orphan_results=orphans)


def _anthropic_outcome(results: Sequence[Mapping[str, object]]) -> str:
    if not results:
        return NO_RESULT
    for segment in results:
        is_error = segment.get("is_error")
        if isinstance(is_error, bool):
            return ERROR if is_error else OK
        exit_code = segment.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            return ERROR if exit_code else OK
    return UNKNOWN


# --------------------------------------------------------------------------
# Codex rollout JSONL: Responses-API items. ``function_call`` /
# ``custom_tool_call`` name their answer by ``call_id``; ``web_search_call``
# has no answer item at all.
# --------------------------------------------------------------------------

_CODEX_CALL_TYPES = frozenset({"function_call", "custom_tool_call", "tool_search_call", "local_shell_call"})
_CODEX_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output", "tool_search_output"})
#: A call item the wire answers inline; no output item is ever emitted for it.
_CODEX_RESULTLESS_CALL_TYPES = frozenset({"web_search_call"})


def _codex_evidence(payload: object) -> SourceToolEvidence:
    calls: dict[str, tuple[str | None, list[Mapping[str, object]]]] = {}
    resultless: list[DeclaredCall] = []
    outputs: dict[str, list[Mapping[str, object]]] = {}
    for index, record in enumerate(_record_sequence(payload)):
        item = record.get("payload") if isinstance(record.get("payload"), Mapping) else record
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type in _CODEX_RESULTLESS_CALL_TYPES:
            raw_id = item.get("call_id") or item.get("id")
            resultless.append(
                DeclaredCall(
                    key=str(raw_id) if raw_id else f"web_search_call:{index}",
                    tool_name=str(item_type),
                    outcome=NO_RESULT,
                )
            )
        elif item_type in _CODEX_CALL_TYPES:
            raw_id = item.get("call_id") or item.get("id")
            if not raw_id:
                continue
            name = item.get("name") or item.get("execution")
            calls[str(raw_id)] = (str(name) if isinstance(name, str) else None, [])
        elif item_type in _CODEX_OUTPUT_TYPES:
            raw_id = item.get("call_id") or item.get("id")
            if raw_id:
                outputs.setdefault(str(raw_id), []).append(item)
    orphans = 0
    for call_id, records in outputs.items():
        if call_id in calls:
            calls[call_id][1].extend(records)
        else:
            orphans += len(records)
    declared = tuple(
        DeclaredCall(
            key=key,
            tool_name=name,
            outcome=_codex_outcome(records),
            result_count=len(records),
        )
        for key, (name, records) in sorted(calls.items())
    )
    return SourceToolEvidence(
        origin="codex-session",
        calls=declared + tuple(resultless),
        orphan_results=orphans,
    )


def _codex_outcome(records: Sequence[Mapping[str, object]]) -> str:
    if not records:
        return NO_RESULT
    for item in records:
        for candidate in (item, item.get("output"), item.get("result")):
            outcome = _codex_outcome_field(candidate)
            if outcome is not None:
                return outcome
    return UNKNOWN


def _codex_outcome_field(value: object) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(value, Mapping):
        return None
    success = value.get("success")
    if isinstance(success, bool):
        return OK if success else ERROR
    is_error = value.get("is_error")
    if isinstance(is_error, bool):
        return ERROR if is_error else OK
    exit_code = value.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return ERROR if exit_code else OK
    return None


def _record_sequence(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [record for record in payload if isinstance(record, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("records", "items", "events"):
            candidate = payload.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                return [record for record in candidate if isinstance(record, Mapping)]
    return []


def judge_conservation(
    evidence: SourceToolEvidence,
    actions: Iterable[ArchivedAction],
    *,
    compare_outcomes: bool = True,
    physical_results: int | None = None,
) -> ConservationVerdict:
    """Compare one source's declared tool evidence to the archive's actions.

    Matching is by tool name and multiplicity, not by identifier: the archive
    is free to name an anonymous call however it likes, and the comparison
    still has to account for every declared answer.

    ``physical_results`` is the archive's own count of stored tool_result
    records. The actions relation shows one answer per call, so an answer
    stored twice is invisible there; comparing the physical count to the
    source's declared record count is what makes duplication and silent
    synthesis visible.
    """
    archived = list(actions)
    declared_by_name: dict[str | None, list[DeclaredCall]] = {}
    for call in evidence.calls:
        declared_by_name.setdefault(call.tool_name, []).append(call)
    archived_by_name: dict[str | None, list[ArchivedAction]] = {}
    for action in archived:
        archived_by_name.setdefault(action.tool_name, []).append(action)

    lost: list[str] = []
    synthesized: list[str] = []
    duplicated: list[str] = []
    outcome_disagreements: list[str] = []
    missing_calls: list[str] = []
    changed_owners: list[str] = []

    for name in sorted(set(declared_by_name) | set(archived_by_name), key=lambda value: value or ""):
        declared = declared_by_name.get(name, [])
        rows = archived_by_name.get(name, [])
        if len(rows) < len(declared):
            missing_calls.append(f"{name}: declared {len(declared)} calls, archived {len(rows)}")
        elif len(rows) > len(declared):
            changed_owners.append(f"{name}: archived {len(rows)} calls, declared {len(declared)}")
        declared_answered = sum(1 for call in declared if call.outcome != NO_RESULT)
        declared_records = sum(call.result_count for call in declared)
        archived_answered = sum(1 for row in rows if row.result_state != "no_result")
        if archived_answered < declared_answered:
            lost.append(f"{name}: source answers {declared_answered} calls, archive answers {archived_answered}")
        elif archived_answered > declared_answered:
            synthesized.append(f"{name}: archive answers {archived_answered} calls, source answers {declared_answered}")
        if declared_records and archived_answered > declared_records:
            duplicated.append(f"{name}: archive pairs {archived_answered} results, source declares {declared_records}")
        if compare_outcomes:
            outcome_disagreements.extend(_outcome_disagreements(name, declared, rows))

    if physical_results is not None:
        if physical_results > evidence.result_records:
            duplicated.append(
                f"archive stores {physical_results} result records, source declares {evidence.result_records}"
            )
        elif physical_results < evidence.result_records:
            lost.append(f"archive stores {physical_results} result records, source declares {evidence.result_records}")

    return ConservationVerdict(
        origin=evidence.origin,
        declared_calls=len(evidence.calls),
        archived_calls=len(archived),
        declared_results=evidence.result_records,
        archived_results=(
            physical_results
            if physical_results is not None
            else sum(1 for row in archived if row.tool_result_block_id is not None)
        ),
        lost_results=tuple(lost),
        synthesized_results=tuple(synthesized),
        duplicated_results=tuple(duplicated),
        changed_owners=tuple(changed_owners),
        outcome_disagreements=tuple(outcome_disagreements),
        missing_calls=tuple(missing_calls),
    )


_ARCHIVE_STATE_BY_OUTCOME = {
    OK: "outcome_success",
    ERROR: "outcome_error",
    UNKNOWN: "outcome_unknown",
    NO_RESULT: "no_result",
}


def _outcome_disagreements(
    name: str | None,
    declared: Sequence[DeclaredCall],
    rows: Sequence[ArchivedAction],
) -> list[str]:
    """Compare the multiset of outcomes, which does not depend on call order."""
    disagreements: list[str] = []
    expected: dict[str, int] = {}
    for call in declared:
        state = _ARCHIVE_STATE_BY_OUTCOME[call.outcome]
        expected[state] = expected.get(state, 0) + 1
    observed: dict[str, int] = {}
    for row in rows:
        observed[row.result_state] = observed.get(row.result_state, 0) + 1
    for state in sorted(set(expected) | set(observed)):
        if expected.get(state, 0) != observed.get(state, 0):
            disagreements.append(
                f"{name}: {state} declared {expected.get(state, 0)}, archived {observed.get(state, 0)}"
            )
    return disagreements


__all__ = [
    "ERROR",
    "NO_RESULT",
    "OK",
    "SUPPORTED_ORIGINS",
    "UNKNOWN",
    "ArchivedAction",
    "ConservationVerdict",
    "DeclaredCall",
    "SourceToolEvidence",
    "declare_tool_evidence",
    "judge_conservation",
]
