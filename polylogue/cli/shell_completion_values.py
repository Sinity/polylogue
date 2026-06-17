"""Shell-completion helpers for archive-backed CLI values.

All archive-backed completions read the ``index.db``
through :class:`~polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore`.
Session-id, tag, repo-name, and tool-name values come from native
session/tag/repo/action read models; cwd-prefix has no archive source
yet and degrades to an empty completion list.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import click
from click.shell_completion import CompletionItem

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.expression import (
    COUNT_QUERY_FIELD_REGISTRY,
    DATE_QUERY_FIELD_REGISTRY,
    EXPRESSION_FIELD_REGISTRY,
    STRUCTURAL_QUERY_UNIT_REGISTRY,
    count_query_fields,
    count_query_operators,
    date_query_fields,
    date_query_operators,
    structural_query_fields,
    structural_query_units,
)
from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS, CompletionSource
from polylogue.archive.query.spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES, QUERY_SEQUENCE_ACTION_TYPES
from polylogue.cli.action_contracts import ACTION_CONTRACTS, CliActionContract
from polylogue.paths import active_index_db_path

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

ArchiveCompletionAction = Callable[["ArchiveStore"], list[CompletionItem]]

_ORIGIN_DESCRIPTIONS: Final[dict[str, str]] = {
    "chatgpt-export": "ChatGPT web exports (lab: OpenAI)",
    "claude-ai-export": "Claude web exports (lab: Anthropic)",
    "claude-code-session": "Claude Code local sessions (lab: Anthropic)",
    "codex-session": "Codex CLI local sessions (lab: OpenAI)",
    "aistudio-drive": "Google AI Studio / Drive exports (lab: Google)",
    "gemini-cli-session": "Gemini CLI local sessions (lab: Google)",
    "hermes-session": "Hermes agent sessions",
    "antigravity-session": "Antigravity brain artifacts",
}

_MAX_ID_COMPLETIONS = 24
_MAX_VALUE_COMPLETIONS = 32
CompletionCallback = Callable[[click.Context, click.Parameter, str], list[CompletionItem]]


@dataclass(frozen=True)
class QueryCompletionCandidate:
    """Structured query-completion candidate shared by shell adapters."""

    value: str
    insert: str
    display: str
    kind: str
    group: str
    description: str
    source: str
    replace_start: int | None = None
    replace_end: int | None = None
    stale: bool = False
    danger: bool = False
    score: float = 1.0
    unsupported_reason: str | None = None
    preview_command: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the structured candidate payload for non-shell consumers."""

        return {
            "value": self.value,
            "insert": self.insert,
            "replace_start": self.replace_start,
            "replace_end": self.replace_end,
            "display": self.display,
            "kind": self.kind,
            "group": self.group,
            "description": self.description,
            "score": self.score,
            "source": self.source,
            "stale": self.stale,
            "danger": self.danger,
            "unsupported_reason": self.unsupported_reason,
            "preview_command": self.preview_command,
        }

    def to_click_item(self) -> CompletionItem:
        help_text = _trim_help(self.description)
        if self.danger:
            help_text = f"DANGER: {help_text}" if help_text else "DANGER"
        return CompletionItem(
            self.insert,
            type="plain",
            help=help_text,
        )


def _split_csv_incomplete(incomplete: str) -> tuple[str, str]:
    if "," not in incomplete:
        return "", incomplete.strip()
    parts = incomplete.split(",")
    prefix_parts = [part.strip() for part in parts[:-1] if part.strip()]
    prefix = ",".join(prefix_parts)
    if prefix:
        prefix += ","
    return prefix, parts[-1].strip()


def _with_csv_prefix(items: list[CompletionItem], prefix: str) -> list[CompletionItem]:
    if not prefix:
        return items
    return [CompletionItem(f"{prefix}{item.value}", type=item.type, help=item.help) for item in items]


def _db_exists() -> bool:
    return active_index_db_path().exists()


def _run_completion(action: ArchiveCompletionAction) -> list[CompletionItem]:
    """Open the archive, run *action*, and return its items.

    Any failure (missing/locked database, unexpected schema) degrades to an
    empty list so completion never raises into the shell.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    index_db = active_index_db_path()
    if not index_db.exists():
        return []
    try:
        with ArchiveStore.open_existing(index_db.parent, read_only=True) as archive:
            return list(action(archive))
    except Exception:
        return []


def _stats_by_items(group_by: str, prefix: str, *, unit: str) -> ArchiveCompletionAction:
    """Archive completion action over ``ArchiveStore.stats_by`` group counts."""

    prefix_lower = prefix.lower()

    def action(archive: ArchiveStore) -> list[CompletionItem]:
        grouped = archive.stats_by(group_by)
        ordered = sorted(grouped.items(), key=lambda pair: (-pair[1], pair[0]))
        items: list[CompletionItem] = []
        for value, count in ordered:
            if prefix_lower and not value.lower().startswith(prefix_lower):
                continue
            items.append(CompletionItem(value, help=f"{count} {unit}"))
            if len(items) >= _MAX_VALUE_COMPLETIONS:
                break
        return items

    return action


def _trim_help(value: str, *, limit: int = 72) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + chr(0x2026)


def _static_completion_items(
    values: tuple[str, ...],
    incomplete: str,
    *,
    csv: bool = False,
) -> list[CompletionItem]:
    prefix, current = _split_csv_incomplete(incomplete) if csv else ("", incomplete.strip())
    current_lower = current.lower()
    items = [CompletionItem(value) for value in values if not current_lower or value.lower().startswith(current_lower)]
    return _with_csv_prefix(items, prefix) if csv else items


def query_field_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return DSL field candidates from the shared expression registry."""

    current = incomplete.strip().lstrip("-").lower()
    if ":" in current:
        return []
    candidates: list[QueryCompletionCandidate] = []
    emitted: set[str] = set()
    for field_name, info in sorted(EXPRESSION_FIELD_REGISTRY.items()):
        if current and not field_name.startswith(current):
            continue
        insert = f"{field_name}:"
        description = info.get("description", "")
        example = info.get("example")
        if example:
            description = f"{description} Example: {example}" if description else f"Example: {example}"
        source = "EXPRESSION_FIELD_REGISTRY"
        count_info = COUNT_QUERY_FIELD_REGISTRY.get(field_name)
        if count_info is not None:
            operators = ", ".join((*count_info.operators, count_info.range_keyword))
            description = (
                f"{description} Readable operators: {operators}. Example: {count_info.example}"
                if description
                else f"Readable operators: {operators}. Example: {count_info.example}"
            )
            source = "EXPRESSION_FIELD_REGISTRY/COUNT_QUERY_FIELD_REGISTRY"
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=insert,
                display=insert,
                kind="query-field",
                group="query fields",
                description=description,
                source=source,
            )
        )
        emitted.add(field_name)
    for field_name, date_info in sorted(DATE_QUERY_FIELD_REGISTRY.items()):
        if field_name in emitted:
            continue
        if current and not field_name.startswith(current):
            continue
        operators = ", ".join((*date_info.operators, date_info.range_keyword))
        description = f"{date_info.description} Readable operators: {operators}. Example: {date_info.example}"
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=f"{field_name} ",
                display=f"{field_name} ",
                kind="query-date-field",
                group="query readable fields",
                description=description,
                source="DATE_QUERY_FIELD_REGISTRY",
            )
        )
    return candidates


def query_structural_unit_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return ``exists <unit>(...)`` candidates from the grammar registry."""

    current = incomplete.strip().lower()
    candidates: list[QueryCompletionCandidate] = []
    for unit in structural_query_units():
        if current and not unit.startswith(current):
            continue
        info = STRUCTURAL_QUERY_UNIT_REGISTRY[unit]
        description = info.description
        if info.example:
            description = f"{description} Example: {info.example}" if description else f"Example: {info.example}"
        candidates.append(
            QueryCompletionCandidate(
                value=unit,
                insert=f"{unit}(",
                display=f"{unit}(",
                kind="query-structural-unit",
                group="query structural units",
                description=description,
                source="STRUCTURAL_QUERY_UNIT_REGISTRY",
            )
        )
    return candidates


def query_structural_field_candidates(unit: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return field candidates accepted inside ``exists <unit>(...)``."""

    current = incomplete.strip().lstrip("-").lower()
    if ":" in current:
        return []
    candidates: list[QueryCompletionCandidate] = []
    for field_name in structural_query_fields(unit):
        if current and not field_name.startswith(current):
            continue
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=f"{field_name}:",
                display=f"{field_name}:",
                kind="query-structural-field",
                group=f"{unit} structural fields",
                description=f"Field accepted inside exists {unit}(...).",
                source="STRUCTURAL_QUERY_UNIT_REGISTRY",
            )
        )
    return candidates


def query_count_operator_candidates(field: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return readable count operators accepted by the query grammar."""

    field_name = field.lower()
    if field_name not in count_query_fields():
        return []
    current = incomplete.strip().lower()
    info = COUNT_QUERY_FIELD_REGISTRY[field_name]
    candidates: list[QueryCompletionCandidate] = []
    for operator in count_query_operators(field_name):
        if current and not operator.startswith(current):
            continue
        insert = f"{operator} " if operator == info.range_keyword else operator
        candidates.append(
            QueryCompletionCandidate(
                value=operator,
                insert=insert,
                display=insert,
                kind="query-count-operator",
                group=f"{field_name} count operators",
                description=f"{info.description} Example: {info.example}",
                source="COUNT_QUERY_FIELD_REGISTRY",
            )
        )
    return candidates


def query_date_operator_candidates(field: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return readable date operators accepted by the query grammar."""

    field_name = field.lower()
    if field_name not in date_query_fields():
        return []
    current = incomplete.strip().lower()
    info = DATE_QUERY_FIELD_REGISTRY[field_name]
    candidates: list[QueryCompletionCandidate] = []
    for operator in date_query_operators(field_name):
        if current and not operator.startswith(current):
            continue
        insert = f"{operator} " if operator == info.range_keyword else operator
        candidates.append(
            QueryCompletionCandidate(
                value=operator,
                insert=insert,
                display=insert,
                kind="query-date-operator",
                group=f"{field_name} date operators",
                description=f"{info.description} Example: {info.example}",
                source="DATE_QUERY_FIELD_REGISTRY",
            )
        )
    return candidates


def _action_description(contract: CliActionContract) -> str:
    guards = ", ".join(contract.guards)
    detail = f"{contract.effect}; input={contract.input_unit}; cardinality={contract.cardinality}"
    return f"{detail}; guards={guards}" if guards else detail


def query_action_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return root query/action candidates from public action contracts."""

    current = incomplete.strip().lower()
    candidates: list[QueryCompletionCandidate] = []
    for contract in ACTION_CONTRACTS:
        if len(contract.path) != 1:
            continue
        name = contract.path[0]
        if current and not name.startswith(current):
            continue
        candidates.append(
            QueryCompletionCandidate(
                value=name,
                insert=name,
                display=name,
                kind="query-action",
                group="query actions",
                description=_action_description(contract),
                source="ACTION_CONTRACTS",
                danger=contract.effect == "destructive",
            )
        )
    return candidates


def _completion_source_for_expression_field(field_name: str) -> CompletionSource | None:
    info = EXPRESSION_FIELD_REGISTRY.get(field_name)
    if info is None:
        return None
    spec_fields = tuple(part.strip() for part in info["spec_field"].split("/") if part.strip())
    for spec_field in spec_fields:
        for descriptor in QUERY_FIELD_DESCRIPTORS:
            if spec_field in {descriptor.name, descriptor.spec_attr} and descriptor.completion_source is not None:
                return descriptor.completion_source
    return None


def _prefixed_query_value_items(
    items: list[CompletionItem],
    *,
    prefix: str,
) -> list[CompletionItem]:
    return [
        CompletionItem(
            f"{prefix}{item.value}",
            type=item.type,
            help=item.help,
        )
        for item in items
    ]


def _complete_query_expression_values(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    negated = incomplete.startswith("-")
    token = incomplete[1:] if negated else incomplete
    field_name, value_prefix = token.split(":", 1)
    field_name = field_name.lower()
    if not field_name or value_prefix.startswith("("):
        return []
    source = _completion_source_for_expression_field(field_name)
    if source is None:
        return []
    value_param = param or click.Option(["--query-value"])
    items = complete_query_source(source)(ctx, value_param, value_prefix)
    prefix = f"{'-' if negated else ''}{field_name}:"
    return _prefixed_query_value_items(items, prefix=prefix)


def _completion_words() -> tuple[str, ...]:
    raw_words = os.environ.get("COMP_WORDS", "")
    words = tuple(part for part in raw_words.split() if part)
    if words and words[0] == "polylogue":
        return words[1:]
    return words


def _structural_completion_context(incomplete: str) -> tuple[str, str] | None:
    stripped = incomplete.strip()
    lower = stripped.lower()
    if lower.startswith("exists "):
        after_exists = stripped[len("exists ") :].lstrip()
        if "(" not in after_exists:
            return "unit", after_exists
        unit, field_prefix = after_exists.split("(", 1)
        unit = unit.strip().lower()
        if unit in structural_query_units() and ")" not in field_prefix:
            return unit, field_prefix.rsplit(" ", 1)[-1]
        return None
    if lower.startswith("exists"):
        return "unit", stripped[len("exists") :].lstrip()
    for unit in structural_query_units():
        prefix = f"{unit}("
        if lower.startswith(prefix):
            return unit, stripped[len(prefix) :].rsplit(" ", 1)[-1]

    words = _completion_words()
    if words:
        previous = words[-2].lower() if len(words) >= 2 else ""
        if previous == "exists":
            return "unit", stripped
        for word in reversed(words[:-1]):
            word_lower = word.lower()
            for unit in structural_query_units():
                if word_lower == f"{unit}(" or word_lower.startswith(f"{unit}("):
                    return unit, stripped
    return None


def _complete_structural_query_context(incomplete: str) -> list[CompletionItem] | None:
    context = _structural_completion_context(incomplete)
    if context is None:
        return None
    unit_or_kind, prefix = context
    if unit_or_kind == "unit":
        return [candidate.to_click_item() for candidate in query_structural_unit_candidates(prefix)]
    return [candidate.to_click_item() for candidate in query_structural_field_candidates(unit_or_kind, prefix)]


def complete_query_expression_fields(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete query DSL field tokens from the canonical grammar registry."""

    if incomplete.startswith("--"):
        return []
    structural_items = _complete_structural_query_context(incomplete)
    if structural_items is not None:
        return structural_items
    if ":" in incomplete:
        return _complete_query_expression_values(ctx, param, incomplete)
    del ctx, param
    return [candidate.to_click_item() for candidate in query_field_candidates(incomplete)]


def complete_query_actions(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete root query actions from public action contracts."""

    del ctx, param
    return [candidate.to_click_item() for candidate in query_action_candidates(incomplete)]


def complete_origin_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    current_lower = current.lower()
    items = [
        CompletionItem(name, help=_ORIGIN_DESCRIPTIONS.get(name))
        for name in _ORIGIN_DESCRIPTIONS
        if not current_lower or name.startswith(current_lower)
    ]
    return _with_csv_prefix(items, prefix)


def complete_action_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(QUERY_ACTION_TYPES, incomplete)


def complete_action_sequence_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(QUERY_SEQUENCE_ACTION_TYPES, incomplete, csv=True)


def complete_message_type_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(tuple(message_type.value for message_type in MessageType), incomplete)


def complete_retrieval_lane_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(QUERY_RETRIEVAL_LANES, incomplete)


def complete_session_ids(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip()
    current_lower = current.lower()

    if not _db_exists():
        return []

    def _query(archive: ArchiveStore) -> list[CompletionItem]:
        summaries = archive.list_summaries(limit=100)
        items: list[CompletionItem] = []
        for summary in summaries:
            cid = str(summary.session_id)
            title = summary.title or ""
            source_name = summary.provider.value
            display = title or cid
            if current and not (
                cid.startswith(current) or (":" in cid and current in cid) or (title and current_lower in title.lower())
            ):
                continue
            items.append(
                CompletionItem(
                    cid,
                    help=f"{source_name} \u00b7 {_trim_help(display)}",
                )
            )
        items.sort(key=lambda item: item.value)
        return items[:_MAX_ID_COMPLETIONS]

    return _run_completion(_query)


def complete_tag_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    current_lower = current.lower()

    if not _db_exists():
        return []

    def _query(archive: ArchiveStore) -> list[CompletionItem]:
        tags = archive.list_user_tags()
        sorted_tags = sorted(tags.items(), key=lambda x: (-x[1], x[0]))
        items: list[CompletionItem] = []
        for name, cnt in sorted_tags:
            if current_lower and not name.lower().startswith(current_lower):
                continue
            items.append(CompletionItem(name, help=f"{cnt} sessions"))
            if len(items) >= _MAX_VALUE_COMPLETIONS:
                break
        return items

    items = _run_completion(_query)
    return _with_csv_prefix(items, prefix)


def complete_repo_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    if not _db_exists():
        return []

    items = _run_completion(_stats_by_items("repo", current, unit="sessions"))
    return _with_csv_prefix(items, prefix)


def complete_cwd_prefix_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param, incomplete
    # The read models do not expose a session-cwd aggregate
    # yet, so cwd-prefix completion has no source to draw from. Returning an
    # empty list keeps the completer well-behaved (no traceback) until a
    # cwd projection lands.
    return []


def complete_tool_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip().lower()
    if not _db_exists():
        return []

    return _run_completion(_stats_by_items("tool", current, unit="actions"))


COMPLETION_SOURCE_HANDLERS: Final[Mapping[CompletionSource, CompletionCallback]] = {
    # Ordered to match the deterministic (sorted) order produced by
    # ``query_completion_sources()`` so the registry-coverage contract holds.
    "action": complete_action_values,
    "action_sequence": complete_action_sequence_values,
    "cwd_prefix": complete_cwd_prefix_values,
    "message_type": complete_message_type_values,
    "origin": complete_origin_values,
    "repo": complete_repo_values,
    "retrieval_lane": complete_retrieval_lane_values,
    "session_id": complete_session_ids,
    "tag": complete_tag_values,
    "tool": complete_tool_values,
}


def complete_query_source(source: CompletionSource) -> CompletionCallback:
    return COMPLETION_SOURCE_HANDLERS[source]


__all__ = [
    "COMPLETION_SOURCE_HANDLERS",
    "complete_action_sequence_values",
    "complete_action_values",
    "complete_query_actions",
    "complete_query_expression_fields",
    "complete_session_ids",
    "complete_cwd_prefix_values",
    "complete_message_type_values",
    "complete_origin_values",
    "complete_query_source",
    "complete_repo_values",
    "complete_retrieval_lane_values",
    "complete_tag_values",
    "complete_tool_values",
    "query_action_candidates",
    "query_count_operator_candidates",
    "query_date_operator_candidates",
    "query_field_candidates",
    "query_structural_field_candidates",
    "query_structural_unit_candidates",
    "QueryCompletionCandidate",
]
