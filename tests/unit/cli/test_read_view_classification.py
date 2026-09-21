"""The read-view execution classification (design D3) is total and executable."""

from __future__ import annotations

from polylogue.archive.query.metadata import structural_query_units
from polylogue.archive.viewport import read_view_choices
from polylogue.cli.read_view_registry import (
    READ_VIEW_HANDLER_METADATA,
    read_views_by_execution_kind,
)
from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS


def test_every_read_view_carries_one_execution_kind() -> None:
    """A view without a classification has no declared route to durable evidence.

    Anti-vacuity: adding a read-view profile without a metadata row, or a
    metadata row without a profile, turns this red.
    """

    assert set(READ_VIEW_HANDLER_METADATA) == set(read_view_choices())
    assert all(metadata.execution_kind for metadata in READ_VIEW_HANDLER_METADATA.values())


def test_classified_operations_are_declared_and_agree_with_their_kind() -> None:
    """Each row names operations the protocol declares, consistent with its kind.

    Anti-vacuity: naming an operation the protocol does not declare, a
    ``session.read`` projection that reaches for another operation, or a
    renderer composed from nothing each turns this red.
    """

    declared_operations = {spec.name for spec in DAEMON_OPERATION_SPECS}
    units = set(structural_query_units())
    for view_id, metadata in READ_VIEW_HANDLER_METADATA.items():
        unknown = set(metadata.operations) - declared_operations
        assert not unknown, f"{view_id} names undeclared operations: {sorted(unknown)}"
        if metadata.execution_kind == "query-units-projection":
            # A unit projection must name a unit the grammar actually declares,
            # so the refuted hypothesis (§2.3) cannot be reintroduced silently.
            assert view_id in units, f"{view_id} is not a structural query unit"
            assert metadata.operations == ("query.units",)
        elif metadata.execution_kind == "session-read-projection":
            assert metadata.operations == ("session.read",)
        elif metadata.execution_kind == "renderer":
            assert metadata.operations, f"{view_id} is a renderer over nothing"
        elif metadata.execution_kind == "in-process":
            # Not migrated yet: the handler reads the archive in this process
            # and reaches no operation, so it must claim none.
            assert metadata.operations == ()
        else:
            assert metadata.execution_kind == "distinct-operation"
            # ``session.lineage`` and ``context.compile`` land with S9.
            assert metadata.operations == ()


def test_the_decided_classification_partitions_every_view() -> None:
    """The D3 decision itself, pinned so a migration cannot drift it unremarked.

    These are the *executed* routes, not the intended ones: polylogue-dutav
    found this partition claiming ten ``session.read`` projections where only
    ``hooks`` reached ``session.read``, and claiming ``neighbors`` as a
    ``cli.query`` renderer where it calls ``polylogue.neighbor_candidates``
    directly.  ``messages`` joined ``hooks`` when polylogue-fko9.3 moved it
    onto the ``session.read`` ``messages`` window kind; the ratchet in
    ``read_view_registry`` is what keeps that direction one-way.
    ``tests/unit/cli/test_read_view_execution_routes.py`` proves each row by
    dispatching it; this one pins the resulting shape.

    Anti-vacuity: moving any view between kinds -- notably reclassifying one of
    the per-session evidence views onto ``query.units``, which the relation
    evidence refutes (``Session.session_events`` is not the materialized
    relation the ``observed-event`` unit reads, and ``file_edits`` diffs are not
    the affected paths the ``file`` unit reads) -- turns this red.
    """

    assert read_views_by_execution_kind("session-read-projection") == ("hooks", "messages")
    assert read_views_by_execution_kind("query-units-projection") == ()
    assert read_views_by_execution_kind("distinct-operation") == (
        "context",
        "context-image",
        "lineage",
        "topology",
    )
    assert read_views_by_execution_kind("renderer") == ("summary", "transcript")
    assert read_views_by_execution_kind("in-process") == (
        "agent-policies",
        "chronicle",
        "correlation",
        "dialogue",
        "effective_context",
        "events",
        "file-edits",
        "neighbors",
        "raw",
        "temporal",
        "web-content",
    )
    assert sum(
        len(read_views_by_execution_kind(kind))
        for kind in (
            "session-read-projection",
            "query-units-projection",
            "distinct-operation",
            "renderer",
            "in-process",
        )
    ) == len(READ_VIEW_HANDLER_METADATA)
