"""Synchronous bridge re-export.

The real implementation lives in ``polylogue.core.async_bridge`` (dependency-
free: only ``asyncio``/``threading``) so that CLI hot paths which only need to
drive a coroutine -- and never touch the ``Polylogue`` facade -- can import it
without pulling in the whole ``polylogue.api`` package. This module keeps the
historical import path working for existing callers (Lynchpin, MCP, other API
consumers) that already depend on the ``Polylogue`` facade being available.
"""

from __future__ import annotations

from polylogue.core.async_bridge import run_coroutine_sync

__all__ = ["run_coroutine_sync"]
