from __future__ import annotations

from datetime import UTC, datetime

from polylogue.archive.models import Message
from tests.infra.builders import make_conv, make_msg
from tests.infra.reference_model import ReferenceArchive


def test_reference_model_reuses_boolean_ast_and_aggregates_at_session_grain() -> None:
    archive = ReferenceArchive()
    archive.add(make_conv(id="a", provider="codex", messages=[make_msg(role="user", text="ship it")]))
    archive.add(
        make_conv(id="b", provider="claude-code", messages=[make_msg(role="user", text="ship it"), make_msg(id="m2")])
    )

    result = archive.query("sessions where origin:codex-session OR messages:>=2")

    assert result.session_ids == ("a", "b")
    assert result.count == 2
    assert result.facets == (("claude-code-session", 1), ("codex-session", 1))


def test_reference_model_lineage_and_text_queries() -> None:
    archive = ReferenceArchive()
    archive.add(make_conv(id="parent", messages=[make_msg(text="context")]))
    archive.add(make_conv(id="child", messages=[make_msg(text="tail")]), parent_id="parent")

    assert archive.lineage("child")[0].id == "parent"
    assert archive.query("text:tail").session_ids == ("child",)


def test_reference_model_recomposes_parent_prefix_for_child_queries() -> None:
    archive = ReferenceArchive()
    archive.add(make_conv(id="parent", messages=[make_msg(text="inherited context")]))
    archive.add(make_conv(id="child", messages=[make_msg(text="divergent tail")]), parent_id="parent")

    assert archive.query("text:inherited").session_ids == ("child", "parent")
    assert archive.query("messages:>=2").session_ids == ("child",)


def test_reference_model_uses_structural_action_sequence_semantics() -> None:
    def action_message(message_id: str, tool_name: str, semantic_type: str) -> Message:
        return make_msg(
            id=message_id,
            text=tool_name,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": tool_name,
                    "tool_id": message_id,
                    "input": {},
                    "semantic_type": semantic_type,
                }
            ],
        )

    archive = ReferenceArchive()
    archive.add(
        make_conv(
            id="ordered",
            messages=(
                action_message("edit", "Edit", "file_edit"),
                action_message("shell", "Bash", "shell"),
            ),
        )
    )
    archive.add(
        make_conv(
            id="interleaved",
            messages=(
                action_message("edit", "Edit", "file_edit"),
                action_message("noise", "Read", "file_read"),
                action_message("shell", "Bash", "shell"),
            ),
        )
    )

    assert archive.query("seq(action:file_edit -> action:shell)").session_ids == ("interleaved", "ordered")
    assert archive.query("seq(action:file_edit ->[next] action:shell)").session_ids == ("ordered",)
