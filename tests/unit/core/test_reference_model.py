"""The reference model's own semantics, stated without an archive.

These pin what the oracle claims before any surface is compared to it: the
production Boolean AST drives evaluation, counts aggregate at session grain,
lineage recomposes on request, and ``seq(...)`` walks declared actions.  A
model that quietly answered ``False`` for an unknown field would agree with
every surface, so the closed vocabulary is pinned here too.
"""

from __future__ import annotations

import pytest

from tests.infra.reference_model import (
    ModelBlock,
    ModelMessage,
    ModelRequest,
    ModelSession,
    ReferenceArchive,
    UnsupportedByModelError,
)


def _session(
    key: str,
    *,
    origin: str = "claude-code-session",
    provider: str = "claude-code",
    messages: tuple[ModelMessage, ...] = (),
    tags: tuple[str, ...] = (),
    parent_key: str | None = None,
    minute: int = 0,
) -> ModelSession:
    return ModelSession(
        key=key,
        origin=origin,
        provider=provider,
        title=f"Session {key}",
        created_at=f"2026-01-01T00:{minute:02d}:00+00:00",
        updated_at=f"2026-01-01T01:{minute:02d}:00+00:00",
        messages=messages,
        tags=tags,
        parent_key=parent_key,
    )


def _message(message_id: str, *, text: str = "ship it", minute: int = 0, **kwargs: object) -> ModelMessage:
    return ModelMessage(
        message_id=message_id,
        role=kwargs.pop("role", "user"),  # type: ignore[arg-type]
        text=text,
        timestamp=f"2026-01-01T00:{minute:02d}:00+00:00",
        blocks=kwargs.pop("blocks", (ModelBlock(type="text", text=text),)),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _tool_message(message_id: str, tool_name: str, semantic_type: str, *, minute: int) -> ModelMessage:
    return ModelMessage(
        message_id=message_id,
        role="assistant",
        text=tool_name,
        timestamp=f"2026-01-01T00:{minute:02d}:00+00:00",
        blocks=(
            ModelBlock(
                type="tool_use",
                tool_name=tool_name,
                tool_id=message_id,
                semantic_type=semantic_type,
            ),
        ),
    )


def test_reference_model_reuses_boolean_ast_and_aggregates_at_session_grain() -> None:
    archive = ReferenceArchive.from_model_sessions(
        (
            _session("a", origin="codex-session", provider="codex", messages=(_message("m1"),), minute=1),
            _session("b", messages=(_message("m1"), _message("m2")), minute=2),
        )
    )

    result = archive.query("sessions where origin:codex-session OR messages:>=2")

    assert result.session_ids == ("claude-code-session:ext-b", "codex-session:ext-a")
    assert result.total == 2
    assert result.origin_facets == (("claude-code-session", 1), ("codex-session", 1))


def test_reference_model_totals_the_match_set_and_returns_only_the_window() -> None:
    """``total`` is the pre-window count, which is the grain surfaces report.

    A model that totalled its page instead would agree with a surface making
    exactly the count-grain error this differential exists to find.
    """
    archive = ReferenceArchive.from_model_sessions(
        _session(f"s{index}", messages=(_message("m1"),), minute=index) for index in range(4)
    )
    expression = "sessions where messages:>=1"

    full = archive.query(ModelRequest(name="full", expression=expression))
    page = archive.query(ModelRequest(name="page", expression=expression, limit=2, offset=1))

    assert full.total == 4
    assert len(full.session_ids) == 4
    assert page.total == 4
    assert page.session_ids == full.session_ids[1:3]


def test_reference_model_lineage_orders_ancestors_first() -> None:
    archive = ReferenceArchive.from_model_sessions(
        (
            _session("parent", messages=(_message("m1", text="context"),), minute=1),
            _session("child", messages=(_message("m1", text="tail"),), parent_key="parent", minute=2),
        )
    )

    assert tuple(session.key for session in archive.lineage("child")) == ("parent", "child")
    assert archive.query("contains:tail").session_ids == ("claude-code-session:ext-child",)


def test_reference_model_recomposes_parent_prefix_only_when_asked() -> None:
    """Recomposition is opt-in because the writer stores the divergent tail.

    A child's row holds its own messages plus a parent reference, so the model
    answers for the stored session by default and composes the inherited prefix
    only for a read that recomposes.
    """
    archive = ReferenceArchive.from_model_sessions(
        (
            _session("parent", messages=(_message("m1", text="inherited context"),), minute=1),
            _session("child", messages=(_message("m1", text="divergent tail"),), parent_key="parent", minute=2),
        )
    )

    assert archive.query("contains:inherited").session_ids == ("claude-code-session:ext-parent",)
    assert archive.query("messages:>=2").session_ids == ()

    assert archive.query("contains:inherited", recompose=True).session_ids == (
        "claude-code-session:ext-child",
        "claude-code-session:ext-parent",
    )
    assert archive.query("messages:>=2", recompose=True).session_ids == ("claude-code-session:ext-child",)


def test_reference_model_uses_structural_action_sequence_semantics() -> None:
    archive = ReferenceArchive.from_model_sessions(
        (
            _session(
                "ordered",
                messages=(
                    _tool_message("edit", "Edit", "file_edit", minute=0),
                    _tool_message("shell", "Bash", "shell", minute=1),
                ),
                minute=1,
            ),
            _session(
                "interleaved",
                messages=(
                    _tool_message("edit", "Edit", "file_edit", minute=0),
                    _tool_message("noise", "Read", "file_read", minute=1),
                    _tool_message("shell", "Bash", "shell", minute=2),
                ),
                minute=2,
            ),
        )
    )

    assert archive.query("seq(action:file_edit -> action:shell)").session_ids == (
        "claude-code-session:ext-interleaved",
        "claude-code-session:ext-ordered",
    )
    assert archive.query("seq(action:file_edit ->[next] action:shell)").session_ids == (
        "claude-code-session:ext-ordered",
    )


def test_reference_model_matches_tools_case_folded_and_requires_every_value() -> None:
    """``tool:`` follows production: case-folded, and multi-valued means all.

    ``matches_tool_terms`` lowercases both sides and requires the named tools
    to be a subset of the session's, so a model comparing case-sensitively or
    by any-of would disagree with the archive on the Boolean spelling, which
    lowercases the written value.
    """
    archive = ReferenceArchive.from_model_sessions(
        (
            _session("reader", messages=(_tool_message("m1", "Read", "file_read", minute=0),), minute=1),
            _session(
                "both",
                messages=(
                    _tool_message("m1", "Read", "file_read", minute=0),
                    _tool_message("m2", "Bash", "shell", minute=1),
                ),
                minute=2,
            ),
        )
    )

    assert archive.query("sessions where tool:Read").session_ids == (
        "claude-code-session:ext-both",
        "claude-code-session:ext-reader",
    )
    assert archive.query("sessions where tool:read").session_ids == (
        "claude-code-session:ext-both",
        "claude-code-session:ext-reader",
    )
    assert archive.query("sessions where tool:(Read|Bash)").session_ids == ("claude-code-session:ext-both",)


def test_reference_model_refuses_a_field_it_does_not_declare() -> None:
    """An undeclared field raises rather than answering ``False``.

    Answering ``False`` would let a differential agree with a surface by
    evaluating nothing, which is the one way an oracle fails silently.
    """
    archive = ReferenceArchive.from_model_sessions((_session("a", messages=(_message("m1"),)),))

    with pytest.raises(UnsupportedByModelError):
        archive.query("sessions where semantic:widgets")
