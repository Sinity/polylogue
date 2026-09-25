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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def resolve_session_id_from_root_params(root_params: Mapping[str, object]) -> str | None:
    """Resolve to one conv id by consulting an explicit id, then filters.

    Order:

    1. Returns the value at ``root_params["conv_id"]`` if set (explicit
       ``--id`` or positional from the calling command).
    2. Otherwise, if ``--latest`` is set or any narrowing filter
       (``--origin``, ``--tag``, ``--since`` etc.) is present,
       executes the active spec with ``limit=1`` and returns the top
       match's id.
    3. Returns ``None`` when no explicit id and no narrowing filters —
       the caller should surface its existing "missing id" error.
    """
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cli.operation_kernel import OperationUnavailableError
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.session_rows import query_session_ids
    from polylogue.cli.shared.helper_support import mutation_refusal
    from polylogue.config import Config, get_config

    explicit = cast("str | None", root_params.get("conv_id"))
    if explicit:
        return explicit

    spec = SessionQuerySpec.from_params(dict(root_params))
    if not spec.latest and not spec.has_filters():
        return None

    pinned = root_params.get("_config")
    config = pinned if isinstance(pinned, Config) else get_config()
    try:
        session_ids = query_session_ids(config, RootModeRequest.from_params(dict(root_params)), limit=1)
    except OperationUnavailableError as exc:
        raise mutation_refusal(exc, exc.operation or "cli.query") from None
    return session_ids[0] if session_ids else None


__all__ = ["resolve_session_id_from_root_params"]
