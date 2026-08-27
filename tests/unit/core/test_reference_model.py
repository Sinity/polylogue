from __future__ import annotations

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
