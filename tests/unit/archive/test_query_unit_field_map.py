"""One descriptor map serves the aggregate group lowerer and bracket predicates.

``QueryUnitDescriptor.row_field_attributes`` replaced two hand-maintained
restatements of "DSL field name -> the unit's own row attribute": the terminal
executor's aggregate field map and the projection clause's bracket field map.
They had already diverged in membership with nothing keeping them in step.

Anti-vacuity for this module: delete ``message``'s ``type`` entry from
``row_field_attributes`` and both tests below go red from that single
deletion. If only one goes red, the two paths are not reading one map.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.query.attached_units import fetch_attached_units
from polylogue.archive.query.expression import (
    ExpressionCompileError,
    WithUnitWindow,
    parse_unit_source_expression,
)
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
from tests.infra.storage_records import SessionBuilder

_SESSION_ID = "claude-code-session:ext-field-map"


def _seed(index_db: Path) -> None:
    (
        SessionBuilder(index_db, "field-map")
        .provider("claude-code")
        .add_message("m1", role="user", text="one two")
        .add_message("m2", role="assistant", text="three four five")
        .save()
    )


def test_aggregate_group_by_reads_the_descriptor_row_field_map(workspace_env: dict[str, Path]) -> None:
    """``group by type`` resolves ``type`` to the message row's ``message_type``."""

    index_db = workspace_env["archive_root"] / "index.db"
    _seed(index_db)

    source = parse_unit_source_expression("messages where text:one OR text:three | group by type | agg count")
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="field-map-agg", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert {row.group_key: row.count for row in envelope.items} == {"message": 2}


def test_bracket_predicate_reads_the_descriptor_row_field_map(workspace_env: dict[str, Path]) -> None:
    """``with messages[type:message]`` resolves ``type`` through the same map."""

    index_db = workspace_env["archive_root"] / "index.db"
    _seed(index_db)

    with ArchiveStore.open_existing(index_db.parent) as archive:
        attached = fetch_attached_units(
            archive,
            [_SESSION_ID],
            ["message"],
            unit_windows={"message": WithUnitWindow(predicates=(("type", "message"),))},
        )
        empty = fetch_attached_units(
            archive,
            [_SESSION_ID],
            ["message"],
            unit_windows={"message": WithUnitWindow(predicates=(("type", "summary"),))},
        )

    assert [row["role"] for row in attached["message"][_SESSION_ID]] == ["user", "assistant"]
    assert empty["message"] == {}

    # The bracket grammar's accepted vocabulary is the same map's key set.
    from polylogue.archive.query.expression import compile_expression

    assert compile_expression("repo:polylogue with messages[type:message]").with_unit_windows is not None
    with pytest.raises(ExpressionCompileError):
        compile_expression("repo:polylogue with messages[not_a_declared_field:x]")
