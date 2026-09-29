"""The CLI's request lowerings build exactly the declared operation payloads.

Each lowering is the one place a CLI verb turns its syntax into an operation
request, so a lowering that forwards a value the declared request model refuses
reaches the operator as a daemon refusal instead of a usage error.
"""

from __future__ import annotations

import pytest


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


class TestSeamALowerings:
    """The Seam A lowerings the root query verbs call."""

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
        """A continuation carries its own offset, so an explicit one is refused.

        A limit is forwarded beside the token: it may narrow the next page, and
        the daemon refuses one that widens it. Mutation: accept an offset with
        a continuation and it is silently ignored in favour of the token's, or
        worse, applied to it."""

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
        assert lower_session_read("session:x", limit=10, continuation="q2.token").payload == {
            "ref": "session:x",
            "limit": 10,
            "continuation": "q2.token",
        }
        with pytest.raises(click.UsageError):
            lower_session_read("session:x", offset=4, continuation="q2.token")

    def test_session_reference_lowering_carries_only_a_positive_bound(self) -> None:
        """Mutation: forward a negative ``--limit`` and the handler is asked for
        a window no reference membership can satisfy."""

        from polylogue.cli.lowering import lower_session_reference

        assert lower_session_reference("from tag:x").payload == {"expression": "from tag:x"}
        assert lower_session_reference("from tag:x", limit=5).payload["limit"] == 5
        assert "limit" not in lower_session_reference("from tag:x", limit=-1).payload
