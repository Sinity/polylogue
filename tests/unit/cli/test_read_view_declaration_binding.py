"""A read view's static contract has exactly one edit site.

``read_view_registry`` declares what a read view accepts; ``read_view_handlers``
binds the callable that runs it.  The handler table used to spell the session
policy, accepted option names and query-set admission a second time, with a
``metadata_mismatch`` check comparing the two copies -- a check that can only
fire after someone has already made the same edit twice, and that is dead once
there is nothing to disagree.  These tests hold the derivation in place.
"""

from __future__ import annotations

from dataclasses import replace

import click
import pytest

from polylogue.cli.query_verbs import read_verb
from polylogue.cli.read_view_handlers import (
    READ_VIEW_EXECUTION,
    READ_VIEW_HANDLERS,
    ReadViewExecution,
    build_read_view_handler,
)
from polylogue.cli.read_view_registry import (
    READ_VIEW_GLOBAL_OPTION_NAMES,
    READ_VIEW_HANDLER_METADATA,
    ReadViewHandlerMetadata,
    ReadViewOptionDeclaration,
    read_view_option_names,
)
from polylogue.cli.read_views.base import ReadViewInvocation
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv

_SYNTHETIC_VIEW = "synthetic-declared-view"


def test_every_view_option_is_bound_only_from_its_declaration() -> None:
    """The real Click adapter gains view options through the registry.

    A new declaration on an existing executable view is the mutation: a
    hand-bound adapter would omit it, or demand a second option edit.
    """

    context = click.Context(read_verb)
    static_names = {param.name for param in click.Command.get_params(read_verb, context)}
    assert not (static_names & (read_view_option_names() - READ_VIEW_GLOBAL_OPTION_NAMES))

    original = READ_VIEW_HANDLER_METADATA["messages"]
    synthetic = ReadViewOptionDeclaration("synthetic_window", ("--synthetic-window",), "Synthetic window marker.")
    READ_VIEW_HANDLER_METADATA["messages"] = replace(original, declared_options=(*original.declared_options, synthetic))
    try:
        parameters = {param.name: param for param in read_verb.get_params(context)}
        assert "synthetic_window" in parameters
        read_verb.parse_args(context, ["--view", "messages", "--synthetic-window", "marker"])
        assert context.params["synthetic_window"] == "marker"
    finally:
        READ_VIEW_HANDLER_METADATA["messages"] = original


def test_view_option_refusal_names_a_next_action() -> None:
    context = click.Context(read_verb)
    with pytest.raises(click.UsageError, match="remove that option or use `read --views`"):
        read_verb.parse_args(context, ["--view", "summary", "--at-position", "3"])


def _never_runs(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    raise AssertionError("validation must refuse before the handler executes")


def _invocation(*, session_id: str | None, explicit_options: frozenset[str] = frozenset()) -> ReadViewInvocation:
    return ReadViewInvocation(
        view=_SYNTHETIC_VIEW,
        session_id=session_id,
        output_format=None,
        destination="stdout",
        out_path=None,
        explicit_options=explicit_options,
    )


def test_a_handler_enforces_the_contract_its_declaration_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared row, bound to a callable, yields a handler that enforces the row.

    A synthetic declaration is the honest probe: it exists nowhere in the
    handler module, so anything the handler enforces about it can only have come
    from the declaration.

    Anti-vacuity: EXECUTED -- making ``build_read_view_handler`` spell its own
    ``session_policy="optional"`` instead of reading the declaration's
    ``"required"`` makes the first refusal below never raise.
    """

    declaration = ReadViewHandlerMetadata(
        _SYNTHETIC_VIEW,
        "required",
        frozenset({"window_hours"}),
        accepts_query_set=True,
        execution_kind="in-process",
    )
    monkeypatch.setitem(READ_VIEW_HANDLER_METADATA, _SYNTHETIC_VIEW, declaration)

    handler = build_read_view_handler(_SYNTHETIC_VIEW, ReadViewExecution(_never_runs))
    request = RootModeRequest.from_params({})

    assert handler.session_policy == "required"
    assert handler.accepted_options == frozenset({"window_hours"})
    assert handler.accepts_query_set is True

    with pytest.raises(click.UsageError, match="requires a session ID"):
        handler.validate(_invocation(session_id=None), request)

    with pytest.raises(click.UsageError, match="does not use --at-position"):
        handler.validate(
            _invocation(session_id="codex-session:native-1", explicit_options=frozenset({"at_position"})),
            request,
        )

    # The declared option is admitted by the same derivation that refused the
    # undeclared one, so the refusal above is a contract and not a blanket no.
    handler.validate(
        _invocation(session_id="codex-session:native-1", explicit_options=frozenset({"window_hours"})),
        request,
    )


def test_an_executable_binding_without_a_declaration_is_refused() -> None:
    """Binding a callable to an undeclared view has no contract to enforce.

    Anti-vacuity: EXECUTED -- returning a ``ReadViewHandler`` with default
    static fields instead of raising makes this pass with a view that accepts
    nothing and requires nothing, which is how an unclassified view used to
    reach the CLI.
    """

    with pytest.raises(RuntimeError, match="no declaration"):
        build_read_view_handler("view-that-was-never-declared", ReadViewExecution(_never_runs))


def test_a_borrowed_declaration_carries_the_lenders_contract_under_the_borrowers_id() -> None:
    """A session-list projection dispatches under its own name, on a declared contract.

    ``SESSION_LIST_PROJECTIONS`` names the declared CLI handler that serves a
    projection, so the projection's own name is not required to be a declared
    read view -- but the contract it runs under still has to come from one.

    Anti-vacuity: EXECUTED -- reading the declaration for the borrower's id
    instead of the lender's makes this raise ``RuntimeError: ... has an
    executable handler but no declaration`` for a projection whose name is not
    itself a declared view, which is how the MCP projection-table contract
    (``tests/unit/mcp/test_session_projection_table.py``) breaks.
    """

    lender = READ_VIEW_HANDLER_METADATA["events"]
    handler = build_read_view_handler(
        "borrowed-projection",
        ReadViewExecution(_never_runs),
        declared_as="events",
    )

    assert handler.view_id == "borrowed-projection"
    assert handler.session_policy == lender.session_policy
    assert handler.accepted_options == lender.accepted_options


def test_every_executable_binding_names_a_declared_view() -> None:
    """The two tables cover the same views, by id.

    Anti-vacuity: adding an entry to ``READ_VIEW_EXECUTION`` without its
    declaration turns this red -- and also fails at import, which is the point.
    """

    assert set(READ_VIEW_EXECUTION) <= set(READ_VIEW_HANDLER_METADATA)
    assert set(READ_VIEW_HANDLERS) == set(READ_VIEW_HANDLER_METADATA)
