"""Hypothesis strategies and helpers for CLI query/run contract tests."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from tests.infra.strategies.summaries import (
    ConversationSummarySpec,
    conversation_summary_batch_strategy,
)

_META_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_"
_META_VALUE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;?!-_/"
_FIELD_NAMES = ("id", "provider", "title", "date", "tags", "summary", "messages")


@dataclass(frozen=True)
class QueryMutationCase:
    """Generated input for _apply_modifiers contracts."""

    summaries: tuple[ConversationSummarySpec, ...]
    set_meta: tuple[tuple[str, str], ...]
    add_tags: tuple[str, ...]
    dry_run: bool
    force: bool
    confirm: bool


@dataclass(frozen=True)
class QueryDeleteCase:
    """Generated input for _delete_conversations contracts."""

    summaries: tuple[ConversationSummarySpec, ...]
    dry_run: bool
    force: bool
    confirm: bool
    delete_results: tuple[bool, ...]


@dataclass(frozen=True)
class SummaryOutputCase:
    """Generated summary output request for JSON/YAML field selection."""

    summaries: tuple[ConversationSummarySpec, ...]
    output_format: str
    selected_fields: tuple[str, ...] | None


@dataclass(frozen=True)
class SendOutputCase:
    """Generated destination routing case for _send_output."""

    to_stdout: bool
    to_file: bool
    to_browser: bool
    to_clipboard: bool
    output_format: str


@st.composite
def query_mutation_case_strategy(draw: st.DrawFn) -> QueryMutationCase:
    """Generate a modifier case with explicit dry-run/confirm branches."""
    summaries = draw(conversation_summary_batch_strategy(min_size=1, max_size=12))
    set_meta = tuple(
        draw(
            st.lists(
                st.tuples(
                    st.text(alphabet=_META_KEY_ALPHABET, min_size=1, max_size=12),
                    st.text(alphabet=_META_VALUE_ALPHABET, min_size=1, max_size=20),
                ),
                min_size=0,
                max_size=3,
                unique=True,
            )
        )
    )
    add_tags = tuple(
        draw(
            st.lists(
                st.text(alphabet=_META_KEY_ALPHABET, min_size=1, max_size=12),
                min_size=0,
                max_size=3,
                unique=True,
            )
        )
    )
    if not set_meta and not add_tags:
        add_tags = ("review",)
    return QueryMutationCase(
        summaries=summaries,
        set_meta=set_meta,
        add_tags=add_tags,
        dry_run=draw(st.booleans()),
        force=draw(st.booleans()),
        confirm=draw(st.booleans()),
    )


@st.composite
def query_delete_case_strategy(draw: st.DrawFn) -> QueryDeleteCase:
    """Generate a delete case covering dry-run, confirm, and partial success."""
    summaries = draw(conversation_summary_batch_strategy(min_size=1, max_size=12))
    return QueryDeleteCase(
        summaries=summaries,
        dry_run=draw(st.booleans()),
        force=draw(st.booleans()),
        confirm=draw(st.booleans()),
        delete_results=tuple(
            draw(
                st.lists(
                    st.booleans(),
                    min_size=len(summaries),
                    max_size=len(summaries),
                )
            )
        ),
    )


@st.composite
def summary_output_case_strategy(draw: st.DrawFn) -> SummaryOutputCase:
    """Generate a structured summary-output request."""
    selected = draw(
        st.one_of(
            st.none(),
            st.lists(st.sampled_from(_FIELD_NAMES), min_size=1, max_size=4, unique=True).map(tuple),
        )
    )
    return SummaryOutputCase(
        summaries=draw(conversation_summary_batch_strategy(min_size=1, max_size=5)),
        output_format=draw(st.sampled_from(("json", "yaml"))),
        selected_fields=selected,
    )


@st.composite
def send_output_case_strategy(draw: st.DrawFn) -> SendOutputCase:
    """Generate one destination-routing case with at least one sink."""
    to_stdout = draw(st.booleans())
    to_file = draw(st.booleans())
    to_browser = draw(st.booleans())
    to_clipboard = draw(st.booleans())
    if not any((to_stdout, to_file, to_browser, to_clipboard)):
        to_stdout = True
    return SendOutputCase(
        to_stdout=to_stdout,
        to_file=to_file,
        to_browser=to_browser,
        to_clipboard=to_clipboard,
        output_format=draw(st.sampled_from(("text", "html"))),
    )

