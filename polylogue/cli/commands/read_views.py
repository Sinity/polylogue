"""Inspect executable read-view profiles."""

from __future__ import annotations

import click

from polylogue.archive.viewport import READ_VIEW_PROFILES, SessionViewProfile
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.core.json import JSONDocument


def _profile_payload(profile: SessionViewProfile) -> JSONDocument:
    return {
        "view_id": profile.view_id,
        "label": profile.label,
        "owner": profile.owner,
        "purpose": profile.purpose,
        "input_scope": profile.input_scope,
        "included_kinds": list(profile.included_kinds),
        "lossiness": profile.lossiness,
        "evidence_policy": profile.evidence_policy,
        "privacy_policy": profile.privacy_policy,
        "formats": list(profile.formats),
        "machine_payload": profile.machine_payload,
        "degraded_states": list(profile.degraded_states),
        "successor_handoff": profile.successor_handoff,
    }


def _profile_payloads() -> list[JSONDocument]:
    return [_profile_payload(profile) for profile in READ_VIEW_PROFILES]


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
        emit_success({"read_views": _profile_payloads()})
        return
    click.echo(_render_plain())


__all__ = ["read_views_command"]
