"""Get/set/list durable ``user_settings`` rows (polylogue-at44 liveness slice).

This is deliberately a closed, typed registry -- see
``polylogue.storage.sqlite.archive_tiers.user_settings_write`` for the key
registry and validators. The full scope x actor x override resolver design
belongs to the w8db epic; this command is only the liveness surface.
"""

from __future__ import annotations

import json

import click

from polylogue.cli.operation_kernel import OperationRequest
from polylogue.cli.read_dispatch import dispatch_read
from polylogue.cli.shared.types import AppEnv


def _print_setting(row: dict[str, object], *, output_format: str) -> None:
    """Render one setting row, however it was obtained.

    The read paths hold a storage envelope and the write path holds the
    daemon's JSON result; both reduce to these four fields, so the rendering
    is one function rather than one per transport.
    """
    if output_format == "json":
        click.echo(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return
    click.echo(
        f"{row['setting_key']} = {row['value']!r} (author={row['author_ref']}, updated_at_ms={row['updated_at_ms']})"
    )


@click.group("setting")
def setting_command() -> None:
    """Get, set, and list durable user settings (e.g. ``subscription_tier``)."""


@setting_command.command("get")
@click.argument("setting_key")
@click.option("-f", "--format", "output_format", type=click.Choice(("text", "json")), default="text", show_default=True)
@click.pass_obj
def setting_get_command(env: AppEnv, setting_key: str, output_format: str) -> None:
    """Print one durable setting through the resident User-only read."""
    result, _served_by = dispatch_read(env.config, OperationRequest("user.settings.get", {"setting_key": setting_key}))
    row = result.get("item")
    if not isinstance(row, dict):
        if output_format == "json":
            click.echo(json.dumps({"setting_key": setting_key, "value": None}))
        else:
            click.echo(f"{setting_key} is unset")
        return
    _print_setting(row, output_format=output_format)


@setting_command.command("set")
@click.argument("setting_key")
@click.argument("value")
@click.option("-f", "--format", "output_format", type=click.Choice(("text", "json")), default="text", show_default=True)
@click.pass_obj
def setting_set_command(env: AppEnv, setting_key: str, value: str, output_format: str) -> None:
    """Insert-or-update one typed setting row (rejects unknown keys/values).

    ``user.db`` is the archive's one irreplaceable tier and the daemon is its
    sole writer, so this lowers to the declared ``mutation.user.setting.set``
    operation instead of opening a writable store in the CLI process
    (polylogue-gjwto / polylogue-r29bv). With no daemon the command refuses
    rather than becoming a second writer.
    """

    from polylogue.cli.archive_query import submit_cli_mutation

    written = submit_cli_mutation(
        env,
        "mutation.user.setting.set",
        {"setting_key": setting_key, "value": value},
    )
    row = written.get("result")
    if not isinstance(row, dict):
        raise click.ClickException("daemon accepted the setting write but returned no row")
    _print_setting(row, output_format=output_format)


@setting_command.command("list")
@click.option("-f", "--format", "output_format", type=click.Choice(("text", "json")), default="text", show_default=True)
@click.pass_obj
def setting_list_command(env: AppEnv, output_format: str) -> None:
    """List stored settings through the resident User-only read."""
    result, _served_by = dispatch_read(env.config, OperationRequest("user.settings.list", {}))
    rows = result["items"]
    assert isinstance(rows, list)
    if output_format == "json":
        click.echo(json.dumps(rows, ensure_ascii=False, sort_keys=True))
        return
    if not rows:
        click.echo("no settings stored")
        return
    for row in rows:
        assert isinstance(row, dict)
        _print_setting(row, output_format="text")


__all__ = ["setting_command"]
