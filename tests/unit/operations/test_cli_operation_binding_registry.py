"""Every declared operation is joined to CLI code, or explicitly is not.

This is the registry diff the CLI-unification design names: a declaration step
that adds operations no surface can build a request for or render a result
from has added nothing, and without this test that is invisible.
"""

from __future__ import annotations

import pytest

from polylogue.cli.operation_bindings import (
    CLI_EXTERNAL_OPERATIONS,
    CLI_OPERATION_BINDINGS,
    CLI_PENDING_ADOPTION,
    OperationBindingError,
    binding_for,
    resolve_reference,
    unclassified_operations,
)
from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS

#: Operations that are declared but not yet reachable from a CLI route.  This
#: set is closed on purpose: a new declaration is red until someone either
#: binds it or adds it here with the step that adopts it.  It is empty --
#: every declared operation the CLI serves has a lowering and a renderer.
EXPECTED_PENDING_ADOPTION: frozenset[str] = frozenset()


def test_every_declared_operation_is_classified_exactly_once() -> None:
    """Mutation: declare an operation and skip the registry -- it lands here
    first, before it can be mistaken for a served surface."""

    assert unclassified_operations() == ()
    names = [*CLI_OPERATION_BINDINGS, *CLI_PENDING_ADOPTION, *CLI_EXTERNAL_OPERATIONS]
    assert len(names) == len(set(names)), "an operation is classified in more than one table"
    declared = {spec.name for spec in DAEMON_OPERATION_SPECS}
    assert set(names) <= declared, "the registry classifies an operation that is not declared"


def test_pending_adoption_stays_closed() -> None:
    """Mutation: park a new declaration in CLI_PENDING_ADOPTION to quiet the
    totality check -- this assertion is what makes that an explicit decision."""

    assert set(CLI_PENDING_ADOPTION) == EXPECTED_PENDING_ADOPTION
    for operation, reason in CLI_PENDING_ADOPTION.items():
        assert reason.strip(), operation


def test_completion_is_bound_to_a_lowering_and_a_renderer() -> None:
    """The archive-backed completer is a served route, not a deferred one.

    Mutation: return ``completion`` to ``CLI_PENDING_ADOPTION`` and the
    lookup raises instead of naming the two callables -- which is the state
    the registry described while the code had already moved on, because a
    parked reason string is never checked against the code it describes.
    """

    binding = binding_for("completion")
    assert binding.lowering == "polylogue.cli.lowering:lower_completion"
    assert binding.renderers == ("polylogue.cli.shell_completion_values:render_completion_values",)
    assert callable(resolve_reference(binding.lowering))
    assert callable(resolve_reference(binding.renderers[0]))


def test_completion_lowering_clamps_into_the_declared_request_bound() -> None:
    """Mutation: forward the caller's ``limit`` verbatim and the declared
    request model refuses it, which reaches a shell as an empty candidate list
    indistinguishable from "no matching values"."""

    from polylogue.cli.lowering import COMPLETION_LIMIT_BOUNDS, lower_completion
    from polylogue.operations.daemon_protocol import CompletionRequest

    low, high = COMPLETION_LIMIT_BOUNDS
    assert (low, high) == (
        CompletionRequest.model_fields["limit"].metadata[0].ge,
        CompletionRequest.model_fields["limit"].metadata[1].le,
    )
    assert lower_completion("tag", "", limit=32).payload["limit"] == 32
    for out_of_range in (0, -5, high + 1, 10_000):
        payload = lower_completion("tag", "pre", limit=out_of_range).payload
        bound = payload["limit"]
        assert isinstance(bound, int)
        assert low <= bound <= high
        CompletionRequest.model_validate(payload)


@pytest.mark.parametrize("operation", sorted(CLI_OPERATION_BINDINGS))
def test_each_binding_resolves_to_real_callables(operation: str) -> None:
    """Mutation: rename or delete a bound CLI function and the registry stops
    describing the code that exists."""

    binding = binding_for(operation)
    for reference in (binding.lowering, *binding.renderers):
        assert callable(resolve_reference(reference)), reference


def test_read_operations_declared_for_the_cli_are_bound_not_pending() -> None:
    """Mutation: declare a root-query read and wire nothing -- the point of the
    step is that each arrives with a Seam A lowering and a real renderer."""

    for operation in ("cli.query", "query.units", "query.aggregate", "session.read", "session.reference"):
        binding = binding_for(operation)
        assert binding.lowering.startswith("polylogue.cli.lowering:"), (
            f"{operation} must lower through Seam A, not through a surface-local helper"
        )
        assert binding.renderers


def test_an_unclassified_operation_names_itself() -> None:
    """Mutation: return a bare None from binding_for and the caller cannot tell
    a deliberate exclusion from a missing one."""

    with pytest.raises(OperationBindingError, match="not classified"):
        binding_for("operation.that.is.not.declared")
    with pytest.raises(OperationBindingError, match="transport-owned"):
        binding_for("operation.await")


class TestSeamALowerings:
    """The real Seam A lowerings the registry now names."""

    def test_aggregate_mode_reads_the_mode_from_the_root_flags(self) -> None:
        """Mutation: hardcode a mode and ``analyze --count`` and ``analyze --by``
        become the same request."""

        from polylogue.cli.lowering import aggregate_mode

        assert aggregate_mode({"count_only": True, "origin": "codex-session"}) == "count"
        assert aggregate_mode({"stats_by": "origin"}) == "stats_by"
        assert aggregate_mode({"stats_only": True}) == "stats"
        # ``--by`` is checked first: the root callback allows both, and the
        # grouped answer is the specific one.
        assert aggregate_mode({"stats_only": True, "stats_by": "origin"}) == "stats_by"
        assert aggregate_mode({"origin": "codex-session"}) is None

    def test_aggregate_lowering_carries_the_selection_and_the_grouping(self) -> None:
        """Mutation: drop ``group_by`` from the ``stats_by`` payload and the
        handler has no field to group on; drop the selection projection and the
        aggregate summarises a different set than the page it belongs to."""

        import click

        from polylogue.cli.lowering import lower_query_aggregate
        from polylogue.cli.root_request import RootModeRequest

        request = RootModeRequest(params={"origin": "codex-session", "stats_by": "origin"}, query_terms=("retry",))
        lowered = lower_query_aggregate(request, mode="stats_by")
        assert lowered.operation == "query.aggregate"
        assert lowered.payload["mode"] == "stats_by"
        assert lowered.payload["group_by"] == "origin"
        assert lowered.payload["params"] == {"origin": "codex-session", "query": ["retry"]}

        with pytest.raises(click.UsageError):
            lower_query_aggregate(RootModeRequest(params={}, query_terms=()), mode="stats_by")

    def test_session_read_lowering_refuses_a_continuation_with_a_window(self) -> None:
        """Mutation: accept both and the supplied offset is silently ignored in
        favour of the token's, or worse, applied to it."""

        import click

        from polylogue.cli.lowering import lower_session_read

        assert lower_session_read("session:x", limit=10, offset=4).payload == {
            "ref": "session:x",
            "limit": 10,
            "offset": 4,
        }
        assert lower_session_read("session:x", continuation="q2.token").payload == {
            "ref": "session:x",
            "continuation": "q2.token",
        }
        with pytest.raises(click.UsageError):
            lower_session_read("session:x", limit=10, continuation="q2.token")

    def test_session_reference_lowering_carries_only_a_positive_bound(self) -> None:
        """Mutation: forward a negative ``--limit`` and the handler is asked for
        a window no reference membership can satisfy."""

        from polylogue.cli.lowering import lower_session_reference

        assert lower_session_reference("from tag:x").payload == {"expression": "from tag:x"}
        assert lower_session_reference("from tag:x", limit=5).payload["limit"] == 5
        assert "limit" not in lower_session_reference("from tag:x", limit=-1).payload
