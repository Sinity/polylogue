"""Cold-start manual and native agent-client integration commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click

from polylogue.agent_integration.assets import agent_asset_metadata, read_agent_asset
from polylogue.agent_integration.installer import (
    AgentIntegrationError,
    AgentIntegrationManager,
    InstallOptions,
    claude_session_start_payload,
)
from polylogue.agent_integration.spec import CLIENTS, GUIDANCE_MODES, ROLES, AgentClient, GuidanceMode
from polylogue.mcp.declarations import MCPRole

_FORMAT = click.Choice(["plain", "json"])


def _emit(payload: dict[str, object], *, output_format: str) -> None:
    if output_format == "json":
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    action = payload.get("action")
    if action:
        click.echo(f"Polylogue agent {action}: {'ok' if payload.get('ok') else 'attention required'}")
    elif "installed" in payload:
        click.echo(f"Polylogue agent integration: {'ok' if payload.get('ok') else 'attention required'}")
    else:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if payload.get("state_path"):
        click.echo(f"  State: {payload['state_path']}")
    clients = payload.get("clients")
    if isinstance(clients, list):
        for item in clients:
            if not isinstance(item, dict):
                continue
            name = item.get("client", "unknown")
            state = "ok"
            if item.get("native_ok") is False or item.get("retained_drift"):
                state = "attention"
            click.echo(f"  {name}: {state}")
    problems = payload.get("problems")
    if isinstance(problems, list):
        for problem in problems:
            click.echo(f"  - {problem}")


def _manager() -> AgentIntegrationManager:
    return AgentIntegrationManager()


@click.group("agent")
def agent_command() -> None:
    """Install and inspect executable cold-start guidance."""


@agent_command.command("manual")
@click.option("--kind", type=click.Choice(["standing", "reference"]), default="standing", show_default=True)
@click.option("-f", "--format", "output_format", type=_FORMAT, default="plain", show_default=True)
def manual_command(kind: str, output_format: str) -> None:
    """Print the packaged standing manual or deeper reference."""
    asset = "standing-manual.md" if kind == "standing" else "deep-reference.md"
    text = read_agent_asset(asset)
    if output_format == "json":
        click.echo(
            json.dumps(
                {"kind": kind, "content": text, "asset": agent_asset_metadata()},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        click.echo(text, nl=False)


@agent_command.command("manifest")
@click.option("--role", type=click.Choice(list(ROLES)), default="read", show_default=True)
@click.option("-f", "--format", "output_format", type=_FORMAT, default="json", show_default=True)
def manifest_command(role: str, output_format: str) -> None:
    """Report the role-scoped runtime and six-tool target surfaces."""
    try:
        from polylogue.agent_integration.manifest import build_live_manifest

        payload = build_live_manifest(cast(MCPRole, role))
    except ImportError as exc:
        raise click.ClickException(f"MCP runtime dependencies are unavailable: {exc}") from exc
    _emit(payload, output_format=output_format)


@agent_command.command("install")
@click.option("--client", "clients", type=click.Choice(list(CLIENTS)), multiple=True, required=True)
@click.option("--role", type=click.Choice(list(ROLES)), default="read", show_default=True)
@click.option("--guidance", type=click.Choice(list(GUIDANCE_MODES)), default="full", show_default=True)
@click.option("--reference/--no-reference", "include_reference", default=True, show_default=True)
@click.option("--mcp/--no-mcp", "install_mcp", default=True, show_default=True)
@click.option("--archive-root", type=click.Path(path_type=Path), default=None)
@click.option("--config-path", type=click.Path(path_type=Path), default=None)
@click.option("--server-command", default="polylogue-mcp", show_default=True)
@click.option("--polylogue-command", default="polylogue", show_default=True)
@click.option("--replace-clients", is_flag=True, help="Uninstall exact owned operations for clients not selected here.")
@click.option("-f", "--format", "output_format", type=_FORMAT, default="plain", show_default=True)
def install_command(
    clients: tuple[str, ...],
    role: str,
    guidance: str,
    include_reference: bool,
    install_mcp: bool,
    archive_root: Path | None,
    config_path: Path | None,
    server_command: str,
    polylogue_command: str,
    replace_clients: bool,
    output_format: str,
) -> None:
    """Install user-scoped MCP and standing guidance for native clients."""
    from polylogue.agent_integration.manifest import target_surface_is_registered

    typed_role = cast(MCPRole, role)
    if not target_surface_is_registered(typed_role):
        raise click.ClickException(
            "six-tool agent guidance is staged but target tool-name registration and generated-schema verification "
            "have not both completed; rebase after the t46.8 cutover, verify live signatures, and regenerate before "
            "installing"
        )
    options = InstallOptions(
        clients=cast(tuple[AgentClient, ...], clients),
        role=typed_role,
        guidance=cast(GuidanceMode, guidance),
        include_reference=include_reference,
        install_mcp=install_mcp,
        archive_root=archive_root,
        config_path=config_path,
        server_command=server_command,
        polylogue_command=polylogue_command,
        replace_clients=replace_clients,
    )
    try:
        payload = _manager().install(options)
    except AgentIntegrationError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit(payload, output_format=output_format)
    if payload.get("ok") is not True:
        raise click.exceptions.Exit(1)


@agent_command.command("status")
@click.option("-f", "--format", "output_format", type=_FORMAT, default="plain", show_default=True)
def status_command(output_format: str) -> None:
    """Inspect ownership state and native configuration without mutation."""
    payload = _manager().status()
    _emit(payload, output_format=output_format)
    if payload.get("ok") is not True:
        raise click.exceptions.Exit(1)


@agent_command.command("doctor")
@click.option("-f", "--format", "output_format", type=_FORMAT, default="plain", show_default=True)
def doctor_command(output_format: str) -> None:
    """Run blocking native syntax, ownership, executable, and identity checks."""
    payload = _manager().doctor()
    _emit(payload, output_format=output_format)
    if payload.get("ok") is not True:
        raise click.exceptions.Exit(1)


@agent_command.command("uninstall")
@click.option("--client", "clients", type=click.Choice(list(CLIENTS)), multiple=True)
@click.option("-f", "--format", "output_format", type=_FORMAT, default="plain", show_default=True)
def uninstall_command(clients: tuple[str, ...], output_format: str) -> None:
    """Remove only exact operations recorded as Polylogue-owned."""
    selected = cast(tuple[AgentClient, ...], clients) if clients else None
    try:
        payload = _manager().uninstall(selected)
    except AgentIntegrationError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit(payload, output_format=output_format)
    if payload.get("ok") is not True:
        raise click.exceptions.Exit(1)


@agent_command.command("session-start", hidden=True)
@click.option("--client", type=click.Choice(["claude-code"]), required=True)
def session_start_command(client: str) -> None:
    """Emit native hook output for a supported client."""
    if client != "claude-code":
        raise click.ClickException(f"unsupported SessionStart client: {client}")
    click.echo(json.dumps(claude_session_start_payload(), ensure_ascii=False, separators=(",", ":")))


__all__ = ["agent_command"]
