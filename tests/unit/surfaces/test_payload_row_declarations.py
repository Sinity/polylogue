"""Declaration parity and row-conversion contracts for surface payloads."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveActionQueryRow,
    ArchiveAssertionQueryRow,
    ArchiveBlockQueryRow,
    ArchiveDelegationAncestryRow,
    ArchiveDelegationQueryRow,
    ArchiveDelegationSubtreeRow,
    ArchiveFileQueryRow,
    ArchiveMessageQueryRow,
    ArchiveQueryUnitAggregateRow,
)
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    AssertionQueryRowPayload,
    BlockQueryRowPayload,
    DelegationAncestryNodePayload,
    DelegationAttemptPayload,
    DelegationSubtreeNodePayload,
    FileQueryRowPayload,
    MessageQueryRowPayload,
    QueryUnitAggregateRowPayload,
    SurfacePayloadModel,
)

_GENERIC_ROW_DECLARATIONS: tuple[tuple[type[SurfacePayloadModel], type[Any], frozenset[str], frozenset[str]], ...] = (
    (MessageQueryRowPayload, ArchiveMessageQueryRow, frozenset({"blocks"}), frozenset({"unit"})),
    (ActionQueryRowPayload, ArchiveActionQueryRow, frozenset(), frozenset({"unit"})),
    (BlockQueryRowPayload, ArchiveBlockQueryRow, frozenset(), frozenset({"unit"})),
    (FileQueryRowPayload, ArchiveFileQueryRow, frozenset(), frozenset({"unit"})),
    (AssertionQueryRowPayload, ArchiveAssertionQueryRow, frozenset(), frozenset({"unit"})),
    (DelegationAttemptPayload, ArchiveDelegationQueryRow, frozenset(), frozenset({"unit"})),
    (DelegationAncestryNodePayload, ArchiveDelegationAncestryRow, frozenset(), frozenset()),
    (DelegationSubtreeNodePayload, ArchiveDelegationSubtreeRow, frozenset(), frozenset()),
    (QueryUnitAggregateRowPayload, ArchiveQueryUnitAggregateRow, frozenset(), frozenset({"metrics"})),
)


@pytest.mark.parametrize(
    ("payload_model", "row_type", "expected_row_only", "expected_model_only"),
    _GENERIC_ROW_DECLARATIONS,
    ids=[payload_model.__name__ for payload_model, *_ in _GENERIC_ROW_DECLARATIONS],
)
def test_generic_row_declarations_have_explicit_field_parity(
    payload_model: type[SurfacePayloadModel],
    row_type: type[Any],
    expected_row_only: frozenset[str],
    expected_model_only: frozenset[str],
) -> None:
    """A row/model drift must be reviewed instead of hidden by generic copying."""
    row_fields = {field.name for field in fields(row_type)}
    model_fields = set(payload_model.model_fields)

    assert row_fields - model_fields == expected_row_only
    assert model_fields - row_fields == expected_model_only


def test_delegation_node_from_rows_preserves_declared_wire_fields() -> None:
    ancestry_row = ArchiveDelegationAncestryRow(
        session_id="session-leaf",
        depth=2,
        child_session_id="session-mid",
        mapping_state="resolved",
        instruction_tool_use_block_id="block-dispatch",
        link_confidence=0.75,
        link_method="provider-id",
    )
    subtree_row = ArchiveDelegationSubtreeRow(
        session_id="session-mid",
        depth=1,
        parent_session_id="session-root",
        mapping_state="edge_only",
        instruction_tool_use_block_id=None,
        link_confidence=None,
        link_method="topology-edge",
    )

    ancestry = DelegationAncestryNodePayload.from_row(ancestry_row)
    subtree = DelegationSubtreeNodePayload.from_row(subtree_row)

    assert ancestry.model_dump() == {
        "session_id": "session-leaf",
        "depth": 2,
        "child_session_id": "session-mid",
        "mapping_state": "resolved",
        "instruction_tool_use_block_id": "block-dispatch",
        "link_confidence": 0.75,
        "link_method": "provider-id",
    }
    assert subtree.model_dump() == {
        "session_id": "session-mid",
        "depth": 1,
        "parent_session_id": "session-root",
        "mapping_state": "edge_only",
        "instruction_tool_use_block_id": None,
        "link_confidence": None,
        "link_method": "topology-edge",
    }
