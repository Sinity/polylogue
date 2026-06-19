"""Resolve a single session id from root filters (#1626, #1642).

Used by single-session commands (``export``, ``messages``, ``raw``,
``neighbors``, ``ops diagnostics turns``) so root-level filters like
``--latest`` and ``--origin codex-session`` pick a session without forcing the
operator to also pass an ``--id`` or positional.

The query-verb tree's ``_resolve_target_session_id`` in
``polylogue/cli/query_verbs.py`` is the verb-tree adapter that wraps
this with a ``RootModeRequest``. Top-level commands (``export``,
``neighbors``, ``ops diagnostics turns``) call the param-dict variant
directly because they don't run under the query group's typed request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync


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

    explicit = cast("str | None", root_params.get("conv_id"))
    if explicit:
        return explicit

    spec = SessionQuerySpec.from_params(dict(root_params))
    if not spec.latest and not spec.has_filters():
        return None

    one_match_spec = replace(spec, limit=1)

    async def _resolve() -> str | None:
        from polylogue.api import Polylogue
        from polylogue.config import Config

        async with Polylogue.open(config=cast("Config | None", root_params.get("_config"))) as api:
            summaries = await one_match_spec.list_summaries(api.config)
        return str(summaries[0].id) if summaries else None

    return run_coroutine_sync(_resolve())


__all__ = ["resolve_session_id_from_root_params"]
