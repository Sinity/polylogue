"""Resolve a single session id from root filters (#1626, #1642).

Used by single-session commands (``export``, ``messages``, ``raw``,
``neighbors``, ``analyze turns``) so root-level filters like
``--latest`` and ``--origin codex-session`` pick a session without forcing the
operator to also pass an ``--id`` or positional.

The query-verb tree's ``_resolve_target_session_id`` in
``polylogue/cli/query_verbs.py`` is the verb-tree adapter that wraps
this with a ``RootModeRequest``. Top-level commands (``export``,
``neighbors``, ``analyze turns``) call the param-dict variant
directly because they don't run under the query group's typed request.

The resolution itself is the declared ``cli.query`` operation through the
kernel. It used to open a writable-capable ``Polylogue`` facade in the CLI
process for what is a one-row read, which meant ``--latest`` answered from a
second executor that no daemon ever saw.

Only ``--latest`` and ``--first`` ask for the top row. A filter that matches
several sessions is ambiguous: it goes to the TTY chooser or refuses with the
candidates, never to whichever row the read happened to rank first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.shared.types import AppEnv
    from polylogue.config import Config


def resolve_single_session_id(
    request: RootModeRequest,
    *,
    env: AppEnv | None = None,
    operation: str = "read",
    first_only: bool = False,
) -> str | None:
    """Resolve the one session ``request``'s filters select.

    Returns ``None`` when nothing narrows the selection or nothing matched.
    ``--latest`` and ``first_only`` take the top row by request. Otherwise
    several matches go through :func:`resolve_ambiguous_selection`, which runs
    the chooser only on a terminal and raises a typed refusal elsewhere.
    """
    from polylogue.cli.contextual_errors import AMBIGUITY_CANDIDATE_LIMIT
    from polylogue.cli.select import resolve_ambiguous_selection
    from polylogue.cli.session_rows import query_session_ids, query_session_rows

    spec = request.query_spec()
    if not spec.latest and not spec.has_filters():
        return None

    config = cast("Config", request.config())
    if spec.latest or first_only:
        session_ids = query_session_ids(config, request, limit=1)
        return session_ids[0] if session_ids else None

    rows = query_session_rows(config, request, limit=AMBIGUITY_CANDIDATE_LIMIT + 1)
    if len(rows) <= 1:
        return rows[0].session_id if rows else None
    return resolve_ambiguous_selection(
        env,
        [row.session_id for row in rows],
        operation=operation,
        rows_loader=lambda: rows,
    )


def resolve_session_id_from_root_params(
    root_params: Mapping[str, object],
    *,
    env: AppEnv | None = None,
    operation: str = "read",
    first_only: bool = False,
) -> str | None:
    """Resolve to one conv id by consulting an explicit id, then filters.

    Order:

    1. Returns the value at ``root_params["conv_id"]`` if set (explicit
       ``--id`` or positional from the calling command).
    2. Otherwise, if ``--latest`` is set or any narrowing filter
       (``--origin``, ``--tag``, ``--since`` etc.) is present, resolves
       through :func:`resolve_single_session_id`.
    3. Returns ``None`` when no explicit id and no narrowing filters —
       the caller should surface its existing "missing id" error.
    """
    from polylogue.cli.operation_kernel import OperationUnavailableError
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.shared.helper_support import mutation_refusal

    explicit = cast("str | None", root_params.get("conv_id"))
    if explicit:
        return explicit

    try:
        return resolve_single_session_id(
            RootModeRequest.from_params(dict(root_params)), env=env, operation=operation, first_only=first_only
        )
    except OperationUnavailableError as exc:
        raise mutation_refusal(exc, exc.operation or "cli.query") from None


__all__ = ["resolve_session_id_from_root_params", "resolve_single_session_id"]
