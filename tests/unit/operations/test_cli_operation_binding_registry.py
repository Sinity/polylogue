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
#: binds it or adds it here with the step that adopts it.
EXPECTED_PENDING_ADOPTION = frozenset({"completion", "ingest"})


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


@pytest.mark.parametrize("operation", sorted(CLI_OPERATION_BINDINGS))
def test_each_binding_resolves_to_real_callables(operation: str) -> None:
    """Mutation: rename or delete a bound CLI function and the registry stops
    describing the code that exists."""

    binding = binding_for(operation)
    for reference in (binding.lowering, *binding.renderers):
        assert callable(resolve_reference(reference)), reference


def test_read_operations_declared_for_the_cli_are_bound_not_pending() -> None:
    """Mutation: declare these three and wire nothing -- the point of the step
    is that each arrives with a client-side lowering and a renderer."""

    for operation in ("query.aggregate", "session.read", "session.reference"):
        binding = binding_for(operation)
        assert binding.lowering.startswith("polylogue.cli.operation_bindings:")
        assert binding.renderers


def test_an_unclassified_operation_names_itself() -> None:
    """Mutation: return a bare None from binding_for and the caller cannot tell
    a deliberate exclusion from a missing one."""

    with pytest.raises(OperationBindingError, match="not classified"):
        binding_for("operation.that.is.not.declared")
    with pytest.raises(OperationBindingError, match="transport-owned"):
        binding_for("operation.await")


class TestNewReadLowerings:
    def test_aggregate_lowering_reads_the_mode_from_the_root_flags(self) -> None:
        """Mutation: hardcode a mode and `analyze count` and `analyze by` become
        the same request."""

        from polylogue.cli.operation_bindings import lower_query_aggregate

        assert lower_query_aggregate({"count_only": True, "origin": "codex-session"}) == {
            "mode": "count",
            "params": {"origin": "codex-session"},
        }
        assert lower_query_aggregate({"stats_by": "origin"})["mode"] == "stats_by"
        assert lower_query_aggregate({"stats_only": True})["mode"] == "stats"
        with pytest.raises(OperationBindingError):
            lower_query_aggregate({"origin": "codex-session"})

    def test_session_read_lowering_refuses_a_continuation_with_a_window(self) -> None:
        """Mutation: accept both and the supplied offset is silently ignored in
        favour of the token's, or worse, applied to it."""

        from polylogue.cli.operation_bindings import lower_session_read

        assert lower_session_read("session:x", limit=10, offset=4) == {
            "ref": "session:x",
            "limit": 10,
            "offset": 4,
        }
        with pytest.raises(OperationBindingError):
            lower_session_read("session:x", limit=10, continuation="q2.token")

    def test_session_read_renderer_states_an_incomplete_window(self) -> None:
        """Mutation: drop the trailing notice and a bounded window renders
        exactly like a whole transcript."""

        from polylogue.cli.operation_bindings import render_session_read

        lines = render_session_read(
            {
                "session": {"messages": [{"role": "user", "text": "hello"}]},
                "offset": 0,
                "next_offset": 1,
                "total": 9,
                "complete": False,
            }
        )
        assert lines[0].endswith("hello")
        assert "remain" in lines[-1]

    def test_reference_renderer_states_truncation(self) -> None:
        """Mutation: drop the notice and a truncated member list reads as the
        complete membership of the reference."""

        from polylogue.cli.operation_bindings import render_session_reference

        lines = render_session_reference({"members": ["a"], "member_count": 5, "truncated": True})
        assert lines == ["a", "... 5 members in total"]
