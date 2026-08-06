"""Small import/call graph fixture with one live and one unreachable helper."""

from __future__ import annotations


def production_entrypoint() -> str:
    return live_helper()


def live_helper() -> str:
    return "live"


def dead_helper() -> str:
    return "dead"
