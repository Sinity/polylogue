"""Shell-completion helpers for archive-backed CLI values.

Archive-backed completions are one declared ``completion`` operation through
the kernel, so a resident daemon answers a TAB press from its already-open
snapshot instead of the shell-completion process opening the archive itself.
Session-id, tag, repo-name and tool-name values come from native
session/tag/repo/action read models; cwd-prefix has no archive source yet and
degrades to an empty completion list.

The daemon answers or nobody does. Completion runs on the coldest path the CLI
has, and executing the read locally means building the execution graph and
opening the archive for one keystroke -- measured in seconds, which is not a
completion. So the archive-backed sources dispatch ``daemon_only`` and a
missing daemon becomes a displayed refusal naming how to start it, not a
silently empty list that reads as "no matches".

Declared vocabularies are answered here and never leave the process: an origin
is a declaration, not archive content, so ``--origin`` completes on a fresh
install with no daemon and no archive.

Completion must never raise into the shell: every other failure degrades to an
empty list.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import click
from click.shell_completion import CompletionItem

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.completions import (
    QueryCompletionCandidate,
    query_action_candidates,
    query_count_operator_candidates,
    query_date_operator_candidates,
    query_field_candidates,
    query_numeric_operator_candidates,
    query_pipeline_stage_candidates,
    query_session_field_candidates,
    query_structural_field_candidates,
    query_structural_unit_candidates,
    query_terminal_field_candidates,
    query_terminal_source_candidates,
)
from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS, CompletionSource
from polylogue.archive.query.metadata import (
    EXPRESSION_FIELD_REGISTRY,
    structural_query_units,
    terminal_query_sources,
)
from polylogue.archive.query.spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES, QUERY_SEQUENCE_ACTION_TYPES
from polylogue.cli.shell_completion_classes import MESSAGE_COMPLETION_TYPE, completion_message
from polylogue.cli.shell_words import completion_words
from polylogue.core.enums import MaterialOrigin
from polylogue.sources.origin_specs import public_origin_descriptions
from polylogue.surfaces.action_affordances import InputUnit

_MAX_ID_COMPLETIONS = 24
_MAX_VALUE_COMPLETIONS = 32

#: How long a TAB press may wait on a daemon before the completer gives up and
#: renders nothing. A shell prompt that stops responding is worse than a
#: missing completion, so this bounds the socket call rather than inheriting the
#: operation's ordinary read deadline.
_COMPLETION_DEADLINE_MS = 1000

#: Shown, not inserted, when an archive-backed completion has no daemon to ask.
#: It names the remedy because "unsupported without a daemon" and "broken" look
#: identical from a shell prompt otherwise.
DAEMON_REQUIRED_COMPLETION_MESSAGE = (
    "polylogue: no daemon — archive-backed completion needs `polylogued run` (systemctl --user start polylogued)"
)
CompletionCallback = Callable[[click.Context, click.Parameter, str], list[CompletionItem]]


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
    return [
        # A message is displayed, never inserted, so carrying the CSV prefix
        # into it would render the diagnosis as a candidate value.
        item
        if item.type == MESSAGE_COMPLETION_TYPE
        else CompletionItem(f"{prefix}{item.value}", type=item.type, help=item.help)
        for item in items
    ]


def completion_values(source: str, incomplete: str, *, limit: int) -> list[CompletionItem]:
    """Ask the declared ``completion`` operation for one value vocabulary.

    The resident daemon answers from its open snapshot or nothing does: the
    dispatch is ``daemon_only`` precisely so that a TAB press can never fall
    through to the local reader, which would open the archive and take seconds.
    A missing daemon returns one displayed :func:`completion_message` instead,
    because a bare empty list reads as "the archive has no matching values".

    Every other failure degrades to an empty list: a completer has no channel
    to report on, and a traceback printed into a shell prompt is strictly worse
    than no completion. The bound travels in the request -- the shell wants a
    short list quickly, not a complete one.
    """

    try:
        from polylogue.cli.lowering import lower_completion
        from polylogue.cli.operation_kernel import OperationUnavailableError, dispatch
        from polylogue.config import get_config
    except Exception:
        return []

    try:
        result = dispatch(
            get_config(),
            lower_completion(source, incomplete, limit=limit),
            deadline_ms=_COMPLETION_DEADLINE_MS,
            daemon_only=True,
        )
    except OperationUnavailableError:
        return [completion_message(DAEMON_REQUIRED_COMPLETION_MESSAGE)]
    except Exception:
        # Deliberately broad: see the docstring. Any other typed refusal or
        # transport failure is rendered as no completion rather than a
        # traceback in the prompt.
        return []
    return render_completion_values(result.value)


def render_completion_values(value: object) -> list[CompletionItem]:
    """Render a typed ``completion`` result as shell candidates.

    The renderer half of the ``completion`` binding: it consumes the declared
    operation result and never queries the archive, so a daemon-served and a
    directly-executed result render identically.  A result that carries the
    grammar vocabulary rather than a value vocabulary produces nothing here —
    it is a different question, and guessing which was asked is exactly what
    the two-field result contract exists to prevent.
    """

    body = value if isinstance(value, Mapping) else {}
    completions = body.get("value_completions")
    if not isinstance(completions, Mapping):
        return []
    rows = completions.get("values")
    if not isinstance(rows, list):
        return []
    items: list[CompletionItem] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_value = row.get("value")
        if not isinstance(row_value, str) or not row_value:
            continue
        help_text = row.get("help")
        items.append(CompletionItem(row_value, help=help_text if isinstance(help_text, str) else None))
    return items


def _trim_help(value: str, *, limit: int = 120) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + chr(0x2026)


def query_completion_candidate_to_click_item(candidate: QueryCompletionCandidate) -> CompletionItem:
    """Convert shared query-completion metadata into a Click shell item."""

    help_text = _trim_help(candidate.description)
    if candidate.danger:
        help_text = f"DANGER: {help_text}" if help_text else "DANGER"
    return CompletionItem(candidate.insert, type="plain", help=help_text)


def _static_completion_items(
    values: tuple[str, ...],
    incomplete: str,
    *,
    csv: bool = False,
    hyphen_aliases: bool = False,
) -> list[CompletionItem]:
    prefix, current = _split_csv_incomplete(incomplete) if csv else ("", incomplete.strip())
    current_lower = current.lower().replace("-", "_") if hyphen_aliases else current.lower()
    items = [CompletionItem(value) for value in values if not current_lower or value.lower().startswith(current_lower)]
    return _with_csv_prefix(items, prefix) if csv else items


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
    return completion_words()


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
        return [
            query_completion_candidate_to_click_item(candidate)
            for candidate in query_structural_unit_candidates(prefix)
        ]
    return [
        query_completion_candidate_to_click_item(candidate)
        for candidate in query_structural_field_candidates(unit_or_kind, prefix)
    ]


def _terminal_completion_context(incomplete: str) -> tuple[str, str] | None:
    stripped = incomplete.strip()
    lower = stripped.lower()
    sources = terminal_query_sources()
    for source in sources:
        prefix = f"{source} where "
        if lower.startswith(prefix):
            return source, stripped[len(prefix) :].rsplit(" ", 1)[-1]
        if lower == f"{source} where":
            return source, ""

    words = _completion_words()
    if not words:
        return None
    if words[0] == "find":
        words = words[1:]
    if not words:
        return None
    lowered_words = tuple(word.lower() for word in words)
    for index, word in enumerate(lowered_words[:-1]):
        if lowered_words[index + 1] == "where" and word in sources:
            source = word
            return source, stripped
    return None


def _complete_terminal_query_context(incomplete: str) -> list[CompletionItem] | None:
    context = _terminal_completion_context(incomplete)
    if context is None:
        return None
    source, prefix = context
    return [
        query_completion_candidate_to_click_item(candidate)
        for candidate in query_terminal_field_candidates(source, prefix)
    ]


def _pipeline_completion_context(incomplete: str) -> tuple[str, str] | None:
    stripped = incomplete.strip()
    if "|" not in stripped:
        return None
    before_pipe, stage_prefix = stripped.rsplit("|", 1)
    terminal_context = _terminal_completion_context(before_pipe.strip())
    if terminal_context is None:
        return None
    source, _field_prefix = terminal_context
    return source, stage_prefix.lstrip()


def _complete_pipeline_query_context(incomplete: str) -> list[CompletionItem] | None:
    context = _pipeline_completion_context(incomplete)
    if context is None:
        return None
    source, prefix = context
    return [
        query_completion_candidate_to_click_item(candidate)
        for candidate in query_pipeline_stage_candidates(source, prefix)
    ]


def _session_boolean_completion_prefix(incomplete: str) -> str | None:
    stripped = incomplete.strip()
    lower = stripped.lower()
    prefix = "sessions where "
    if lower.startswith(prefix):
        return stripped[len(prefix) :].rsplit(" ", 1)[-1]
    if lower == "sessions where":
        return ""

    words = _completion_words()
    if words and words[0] == "find":
        words = words[1:]
    lowered_words = tuple(word.lower() for word in words)
    for index, word in enumerate(lowered_words[:-1]):
        if word == "sessions" and lowered_words[index + 1] == "where":
            return stripped
    return None


def _complete_session_boolean_query_context(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem] | None:
    prefix = _session_boolean_completion_prefix(incomplete)
    if prefix is None:
        return None
    if ":" in prefix:
        return _complete_query_expression_values(ctx, param, prefix)
    return [query_completion_candidate_to_click_item(candidate) for candidate in query_session_field_candidates(prefix)]


def complete_query_expression_context_fields(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem] | None:
    """Complete only when the cursor is already inside query syntax."""

    if incomplete.startswith("--"):
        return None
    pipeline_items = _complete_pipeline_query_context(incomplete)
    if pipeline_items is not None:
        return pipeline_items
    terminal_items = _complete_terminal_query_context(incomplete)
    if terminal_items is not None:
        return terminal_items
    structural_items = _complete_structural_query_context(incomplete)
    if structural_items is not None:
        return structural_items
    session_boolean_items = _complete_session_boolean_query_context(ctx, param, incomplete)
    if session_boolean_items is not None:
        return session_boolean_items
    if ":" in incomplete:
        return _complete_query_expression_values(ctx, param, incomplete)
    return None


def complete_query_expression_fields(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete query DSL field tokens from the canonical grammar registry."""

    context_items = complete_query_expression_context_fields(ctx, param, incomplete)
    if context_items is not None:
        return context_items
    del ctx, param
    candidates = [
        *query_terminal_source_candidates(incomplete),
        *query_field_candidates(incomplete),
    ]
    return [query_completion_candidate_to_click_item(candidate) for candidate in candidates]


def complete_query_actions(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
    *,
    input_unit: InputUnit | None = None,
) -> list[CompletionItem]:
    """Complete query actions from public action contracts."""

    del ctx, param
    if _terminal_completion_context(incomplete) is not None or _structural_completion_context(incomplete) is not None:
        return []
    return [
        query_completion_candidate_to_click_item(candidate)
        for candidate in query_action_candidates(incomplete, input_unit=input_unit)
    ]


def complete_query_result_actions(
    ctx: click.Context,
    param: click.Parameter | None,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete actions that accept the current query result set after ``then``."""

    return complete_query_actions(ctx, param, incomplete, input_unit="query_result_set")


def complete_origin_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    current_lower = current.lower()
    # Read locally, and deliberately so: an origin is a declared vocabulary,
    # not archive content. Routing it through the ``completion`` operation would
    # make it depend on an openable archive -- so the one completion that is
    # always answerable would start returning nothing on a fresh install, which
    # is the opposite of what an archive-absent fallback is for.
    descriptions = public_origin_descriptions()
    items = [
        CompletionItem(name, help=descriptions.get(name))
        for name in sorted(descriptions)
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


def complete_material_origin_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(
        tuple(origin.value for origin in MaterialOrigin),
        incomplete,
        csv=True,
        hyphen_aliases=True,
    )


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
    return [
        CompletionItem(item.value, type=item.type, help=_trim_help(item.help) if item.help else None)
        for item in completion_values("session_id", incomplete.strip(), limit=_MAX_ID_COMPLETIONS)
    ]


def complete_tag_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    return _with_csv_prefix(completion_values("tag", current, limit=_MAX_VALUE_COMPLETIONS), prefix)


def complete_repo_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    return _with_csv_prefix(completion_values("repo", current, limit=_MAX_VALUE_COMPLETIONS), prefix)


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
    return completion_values("tool", incomplete.strip().lower(), limit=_MAX_VALUE_COMPLETIONS)


COMPLETION_SOURCE_HANDLERS: Final[Mapping[CompletionSource, CompletionCallback]] = {
    # Ordered to match the deterministic (sorted) order produced by
    # ``query_completion_sources()`` so the registry-coverage contract holds.
    "action": complete_action_values,
    "action_sequence": complete_action_sequence_values,
    "cwd_prefix": complete_cwd_prefix_values,
    "material_origin": complete_material_origin_values,
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
    "DAEMON_REQUIRED_COMPLETION_MESSAGE",
    "completion_values",
    "complete_action_sequence_values",
    "complete_action_values",
    "complete_query_actions",
    "complete_query_expression_context_fields",
    "complete_query_expression_fields",
    "complete_query_result_actions",
    "complete_session_ids",
    "complete_cwd_prefix_values",
    "complete_material_origin_values",
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
    "query_numeric_operator_candidates",
    "query_field_candidates",
    "query_completion_candidate_to_click_item",
    "query_pipeline_stage_candidates",
    "query_session_field_candidates",
    "query_structural_field_candidates",
    "query_structural_unit_candidates",
    "query_terminal_field_candidates",
    "query_terminal_source_candidates",
    "render_completion_values",
    "QueryCompletionCandidate",
]
