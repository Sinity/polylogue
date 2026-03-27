"""Shared support helpers for plain check output."""

from __future__ import annotations

from polylogue.health import VerifyStatus


def status_icon(status: VerifyStatus, *, plain: bool) -> str:
    if plain:
        return {
            VerifyStatus.OK: "OK",
            VerifyStatus.WARNING: "WARN",
            VerifyStatus.ERROR: "ERR",
        }.get(status, "?")
    return {
        VerifyStatus.OK: "[green]✓[/green]",
        VerifyStatus.WARNING: "[yellow]![/yellow]",
        VerifyStatus.ERROR: "[red]✗[/red]",
    }.get(status, "?")


__all__ = ["status_icon"]
