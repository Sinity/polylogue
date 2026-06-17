"""Inspect executable read-view profiles."""

from __future__ import annotations

import click

from polylogue.archive.viewport import READ_VIEW_PROFILES, read_view_profile_payloads
from polylogue.cli.shared.machine_errors import emit_success


def _render_plain() -> str:
    lines = ["Read views:"]
    for profile in READ_VIEW_PROFILES:
        handoff = " handoff" if profile.successor_handoff else ""
        lines.append(
            f"  {profile.view_id:<12} {profile.lossiness:<10} evidence={profile.evidence_policy:<10}"
            f" formats={','.join(profile.formats)}{handoff}"
        )
        lines.append(f"      {profile.purpose}")
    return "\n".join(lines)


@click.command("read-views")
@click.option("--format", "-f", "output_format", type=click.Choice(["plain", "json"]), default="plain")
def read_views_command(output_format: str) -> None:
    """List executable read-view profile metadata."""

    if output_format == "json":
        emit_success({"read_views": read_view_profile_payloads()})
        return
    click.echo(_render_plain())


__all__ = ["read_views_command"]
