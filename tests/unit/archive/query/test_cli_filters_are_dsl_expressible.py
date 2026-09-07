"""Every root filter option has an equivalent the query DSL compiles.

A filter reachable only as a Click option is a filter the DSL, the MCP
expression surface, and every saved query cannot express, so the CLI has to
keep its own selection code to apply it.  The binding checked here is the
lowered filter set: an option and its DSL spelling must produce the same
:func:`spec_session_filter_kwargs` entry — the one kwarg set every archive
reader accepts — rather than merely parsing without error.

Two anti-vacuity conditions:

*Per filter* — :data:`_EQUIVALENCES` compares the option route against the DSL
route on the lowered key.  Deleting the DSL token's compilation leaves the key
empty on one side and the comparison red; applying it in Python after the read
instead of lowering it leaves the key empty on both sides, which
:func:`test_every_dsl_spelling_lowers_to_a_non_default_filter` rejects.

*Per option* — :func:`test_every_filter_option_is_dsl_expressible` reads the
option names off :data:`FILTER_OPTION_DECORATORS`, so a newly added filter
option that nothing in :data:`_EQUIVALENCES` covers is red until it is either
given a DSL spelling or declared in :data:`_NOT_A_PREDICATE` with its reason.
"""

from __future__ import annotations

import click
import pytest

from polylogue.archive.query.expression import compile_expression
from polylogue.archive.query.filter_kwargs import spec_session_filter_kwargs
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.click_option_groups import FILTER_OPTION_DECORATORS

#: ``(filter key, option params, DSL expression)`` — the option route and the
#: DSL route must lower to the same value for that key.
_EQUIVALENCES: tuple[tuple[str, dict[str, object], str], ...] = (
    ("action_sequence", {"action_sequence": "file_edit,shell"}, "action_sequence:file_edit>shell"),
    ("action_text_terms", {"action_text": ("pytest",)}, "action_text:pytest"),
    (
        "since_session_id",
        {"since_session_id": "claude-code-session:abc123"},
        "since_session:claude-code-session:abc123",
    ),
    ("action_terms", {"action": ("file_edit",)}, "action:file_edit"),
    ("excluded_action_terms", {"exclude_action": ("file_edit",)}, "-action:file_edit"),
    ("tool_terms", {"tool": ("bash",)}, "tool:bash"),
    ("excluded_tool_terms", {"exclude_tool": ("bash",)}, "-tool:bash"),
    ("tags", {"tag": "review"}, "tag:review"),
    ("excluded_tags", {"exclude_tag": "stale"}, "-tag:stale"),
    ("repo_names", {"repo": "polylogue"}, "repo:polylogue"),
    ("project_refs", {"project": "g-p-6a40343a"}, "project:g-p-6a40343a"),
    ("referenced_paths", {"referenced_path": ("polylogue/cli",)}, "path:polylogue/cli"),
    ("cwd_prefix", {"cwd_prefix": "/realm/project"}, "cwd:/realm/project"),
    ("title", {"title": "refactor"}, "title:refactor"),
    ("has_types", {"has_type": "summary"}, "has:summary"),
    ("has_tool_use", {"filter_has_tool_use": True}, "has:tools"),
    ("has_thinking", {"filter_has_thinking": True}, "has:thinking"),
    ("has_paste", {"filter_has_paste": True}, "has:paste"),
    ("min_messages", {"min_messages": 10}, "messages:>=10"),
    ("max_messages", {"max_messages": 10}, "messages:<=10"),
    ("min_words", {"min_words": 200}, "words:>=200"),
    ("root", {"root": False}, "root:false"),
    ("origins", {"origin": "claude-code-session"}, "origin:claude-code-session"),
    ("excluded_origins", {"exclude_origin": "claude-code-session"}, "-origin:claude-code-session"),
)

#: Filter-group options that select no rows on their own, with the reason.
_NOT_A_PREDICATE: dict[str, str] = {
    "conv_id": "session scope, carried outside the filter set as session_id",
    "contains": "free text; the DSL spells it as a bare term or contains:",
    "cursor": "pagination position",
    "exclude_text": "free text negation; the DSL spells it as a leading-dash term",
    "latest": "ordering shortcut",
    "lexical": "retrieval lane shortcut, spelled lane:dialogue",
    "limit": "page size",
    "offset": "page offset",
    "retrieval_lane": "retrieval lane, spelled lane:",
    "reverse": "ordering direction",
    "sample": "sampling size",
    "semantic": "retrieval lane shortcut, spelled near:",
    "similar_text": "similarity seed, spelled near:",
    "since": "date bound, spelled since: (covered by the date-field tests)",
    "sort": "ordering key",
    "typed_only": "spelled as the count predicate `sessions where paste_messages = 0`",
    "until": "date bound, spelled until: (covered by the date-field tests)",
}


def _filter_option_names() -> frozenset[str]:
    """Return the parameter names the root filter option group declares."""

    def _target() -> None: ...

    decorated: object = _target
    for decorator in FILTER_OPTION_DECORATORS:
        decorated = decorator(decorated)
    params: list[click.Parameter] = decorated.__click_params__  # type: ignore[attr-defined]
    return frozenset(str(param.name) for param in params if param.name)


def _option_lowering(params: dict[str, object]) -> dict[str, object]:
    return dict(spec_session_filter_kwargs(SessionQuerySpec.from_params(params)))


def _dsl_lowering(expression: str) -> dict[str, object]:
    return dict(spec_session_filter_kwargs(compile_expression(expression)))


@pytest.mark.parametrize(
    ("key", "params", "expression"),
    _EQUIVALENCES,
    ids=[key for key, _, _ in _EQUIVALENCES],
)
def test_option_and_dsl_lower_to_the_same_filter(key: str, params: dict[str, object], expression: str) -> None:
    """The DSL spelling reaches the readers' filter set, not CLI-side code."""
    assert _dsl_lowering(expression)[key] == _option_lowering(params)[key]


@pytest.mark.parametrize(
    ("key", "expression"),
    [(key, expression) for key, _, expression in _EQUIVALENCES],
    ids=[key for key, _, _ in _EQUIVALENCES],
)
def test_every_dsl_spelling_lowers_to_a_non_default_filter(key: str, expression: str) -> None:
    """A token that compiles but lowers to the default selects nothing."""
    lowered = _dsl_lowering(expression)[key]
    default = _dsl_lowering("")[key]
    assert lowered != default, f"{expression!r} left {key} at its default {default!r}"


def test_every_filter_option_is_dsl_expressible() -> None:
    """No root filter option is reachable only through Click."""
    covered = {name for _, params, _ in _EQUIVALENCES for name in params}
    uncovered = sorted(_filter_option_names() - covered - set(_NOT_A_PREDICATE))
    assert uncovered == [], f"filter options with no DSL spelling: {uncovered}"


def test_the_not_a_predicate_table_names_only_live_options() -> None:
    """A declared exemption for an option that no longer exists is dead text."""
    stale = sorted(set(_NOT_A_PREDICATE) - _filter_option_names())
    assert stale == []
