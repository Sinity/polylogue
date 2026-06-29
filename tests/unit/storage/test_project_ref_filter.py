"""Coverage for the ``project:`` query filter on ``sessions.provider_project_ref``.

The ChatGPT project ref (``g-p-<id>``) is materialized on
``sessions.provider_project_ref`` (schema v17) and surfaced through the read
models. This proves it is both queryable (``project:<ref>`` selects by exact
value) and visible (the summary carries the value through the read path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.expression import compile_expression
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.spec import query_spec_to_plan
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder, db_setup

_PROJECT_REF = "g-p-6a40343a"


def _seed(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "conv-in-project")
        .provider("chatgpt")
        .provider_project_ref(_PROJECT_REF)
        .add_message("m1", role="user", text="in project")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-other-project")
        .provider("chatgpt")
        .provider_project_ref("g-p-deadbeef")
        .add_message("m2", role="user", text="other project")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-no-project")
        .provider("chatgpt")
        .add_message("m3", role="user", text="no project")
        .save()
    )


def test_project_field_compiles_into_spec() -> None:
    spec = compile_expression(f"project:{_PROJECT_REF}")
    assert spec.project_refs == (_PROJECT_REF,)


@pytest.mark.asyncio
async def test_project_plan_selects_by_provider_project_ref(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    _seed(db_path)

    plan = SessionQueryPlan(project_refs=(_PROJECT_REF,), limit=10)
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()

    assert [str(summary.id) for summary in summaries] == [
        native_session_id_for("chatgpt", "conv-in-project")
    ]
    # The matched summary also exposes the project ref on the read model.
    assert summaries[0].provider_project_ref == _PROJECT_REF


@pytest.mark.asyncio
async def test_project_dsl_roundtrip_selects_session(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    _seed(db_path)

    plan = query_spec_to_plan(compile_expression(f"project:{_PROJECT_REF}"))
    summaries = await SessionFilter(archive_root=archive_root, query_plan=plan).list_summaries()

    assert [str(summary.id) for summary in summaries] == [
        native_session_id_for("chatgpt", "conv-in-project")
    ]
