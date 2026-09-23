"""The transcript window has one execution route, and surfaces cannot re-split it.

polylogue-ijbwq unified the transcript window
(``polylogue/operations/transcript_window.py``). The cross-surface behaviour is
proved by the differential and continuation-parity suites; what those cannot
catch is a *new* surface, or a future edit to an existing one, quietly calling
``Polylogue.get_messages_paginated`` again with its own offset arithmetic. Such
a call would still return the right ids for a first page -- so the differential
would stay green while that surface silently lost snapshot binding.

This module is that guard. It is an AST census over the production surface
packages, in the same shape as the controlled-read boundary guard
(``tests/unit/archive/query/test_read_surface_control.py``).

Anti-vacuity: re-split the route -- point the CLI, MCP or HTTP transcript
window back at ``get_messages_paginated`` -- and
``test_no_public_surface_reads_the_transcript_window_directly`` names the file
and line. Delete the epoch validation from ``bind_snapshot`` and
``test_the_route_validates_the_epoch_on_every_resume`` goes red, because a
resumed transaction would no longer be checked against the current snapshot.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]

#: The one module allowed to perform the transcript-window storage read.
ROUTE = "polylogue/operations/transcript_window.py"

#: Production packages that serve the four public surfaces.
SURFACE_PACKAGES = ("polylogue/api", "polylogue/cli", "polylogue/mcp", "polylogue/daemon", "polylogue/operations")

#: Call sites that are not transcript windows, each with the reason it is not.
DECLARED_NON_WINDOW_CALLS: dict[str, str] = {
    # A mechanical sync mirror of that same facade method, not a surface window.
    "polylogue/api/sync/sessions.py": "sync mirror of the facade storage read; the bound window is read_transcript_window",
}


def _paginated_read_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_messages_paginated"
    ]


def test_no_public_surface_reads_the_transcript_window_directly() -> None:
    """Only the one route reads a transcript window; everything else is declared."""

    route_calls = 0
    violations: list[str] = []
    for package in SURFACE_PACKAGES:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            lines = _paginated_read_lines(path)
            if not lines:
                continue
            if relative == ROUTE:
                route_calls += len(lines)
                continue
            if relative in DECLARED_NON_WINDOW_CALLS:
                continue
            violations.extend(f"{relative}:{line}" for line in lines)

    assert route_calls == 1, (
        f"the single transcript-window route must perform exactly one storage read, found {route_calls}"
    )
    assert not violations, (
        "these call sites read a transcript window outside the one route; either route them through "
        "polylogue.operations.transcript_window or declare why they are not a window: " + ", ".join(violations)
    )


def test_every_declared_exception_still_exists() -> None:
    """A declaration that no longer matches real code would silently widen the census."""

    for relative in DECLARED_NON_WINDOW_CALLS:
        path = REPO_ROOT / relative
        assert path.exists(), f"declared exception {relative} no longer exists"
        assert _paginated_read_lines(path), (
            f"{relative} is declared as a non-window caller but no longer calls get_messages_paginated; "
            "remove the declaration rather than leaving the census wider than the code"
        )


def test_the_route_validates_the_epoch_on_every_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed window is checked against the current snapshot, never assumed fresh."""

    import polylogue.archive.query.transaction as transaction_module
    from polylogue.archive.query.transaction import QueryContinuationStaleError, QueryTransactionRequest
    from polylogue.operations.transcript_window import bind_snapshot

    # The reader the route is handed reports a snapshot that moved since the
    # token was issued.
    monkeypatch.setattr(transaction_module, "archive_snapshot_epoch", lambda archive, **_: "archive:v1:current")

    transaction = QueryTransactionRequest(
        operation="sessions.read",
        arguments={"ref": "session:x"},
        page_size=2,
        offset=2,
        projection="session-owner-v1",
        stable_order="position",
    ).with_archive_epoch("archive:v1:stale")

    with pytest.raises(QueryContinuationStaleError):
        bind_snapshot(object(), transaction)


def test_a_fresh_window_is_bound_to_the_snapshot_it_was_composed_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A window with no epoch yet is stamped, so its continuation is bindable."""

    import polylogue.operations.transcript_window as route
    from polylogue.archive.query.transaction import QueryTransactionRequest
    from polylogue.operations.transcript_window import bind_snapshot, window_result

    monkeypatch.setattr(route, "archive_snapshot_epoch", lambda archive, **_: "archive:v1:current")

    transaction = QueryTransactionRequest(
        operation="sessions.read",
        arguments={"ref": "session:x"},
        page_size=2,
        offset=0,
        projection="session-owner-v1",
        stable_order="position",
    )
    framed = bind_snapshot(object(), transaction)

    assert framed.archive_epoch == "archive:v1:current"
    window = window_result(["a", "b"], 4, framed)
    assert window.next_offset == 2
    assert window.continuation is not None
    assert not window.complete
    # The last window mints no token: nothing is left to resume.
    assert window_result(["c", "d"], 4, framed.next(offset=2)).continuation is None
