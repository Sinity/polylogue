"""Synchronous helper to load saved view params for the CLI --view flag."""

from __future__ import annotations

import json
from typing import cast

from polylogue.api.sync.bridge import run_coroutine_sync


def load_view_params_sync(view_name: str) -> dict[str, object]:
    """Load a saved view's query_json into a params dict for SessionQuerySpec.from_params()."""

    async def _load() -> dict[str, object]:
        from polylogue.api import Polylogue

        async with Polylogue() as poly:
            result = await poly.get_view_by_name(view_name)
            if result is None:
                raise KeyError(f"Saved view not found: {view_name!r}")
            return cast(dict[str, object], json.loads(result["query_json"]))

    return run_coroutine_sync(_load())


__all__ = ["load_view_params_sync"]
