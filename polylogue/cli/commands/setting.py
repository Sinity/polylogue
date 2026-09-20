"""Get/set/list durable ``user_settings`` rows (polylogue-at44 liveness slice).

This is deliberately a closed, typed registry -- see
``polylogue.storage.sqlite.archive_tiers.user_settings_write`` for the key
registry and validators. The full scope x actor x override resolver design
belongs to the w8db epic; this command is only the liveness surface.
"""

from __future__ import annotations

import asyncio
import json

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_root


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


def _print_envelope(env: object, *, output_format: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope

    assert isinstance(env, ArchiveUserSettingEnvelope)
    _print_setting(
        {
            "setting_key": env.setting_key,
            "value": env.value,
            "updated_at_ms": env.updated_at_ms,
            "author_ref": env.author_ref,
        },
        output_format=output_format,
    )


@click.group("setting")
def setting_command() -> None:
    """Get, set, and list durable user settings (e.g. ``subscription_tier``)."""


@setting_command.command("get")
@click.argument("setting_key")
@click.option("-f", "--format", "output_format", type=click.Choice(("text", "json")), default="text", show_default=True)
def setting_get_command(setting_key: str, output_format: str) -> None:
    """Print one setting's stored value, or report it as unset."""

    from polylogue.api import Polylogue

    async def run() -> object | None:
        async with Polylogue(archive_root=archive_root()) as poly:
            return await poly.get_setting(setting_key)

    envelope = asyncio.run(run())
    if envelope is None:
        if output_format == "json":
            click.echo(json.dumps({"setting_key": setting_key, "value": None}))
        else:
            click.echo(f"{setting_key} is unset")
        return
    _print_envelope(envelope, output_format=output_format)


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
def setting_list_command(output_format: str) -> None:
    """List every stored setting row."""

    from polylogue.api import Polylogue

    async def run() -> list[object]:
        async with Polylogue(archive_root=archive_root()) as poly:
            return list(await poly.list_settings())

    envelopes = asyncio.run(run())
    if output_format == "json":
        from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope

        payload = []
        for env in envelopes:
            assert isinstance(env, ArchiveUserSettingEnvelope)
            payload.append(
                {
                    "setting_key": env.setting_key,
                    "value": env.value,
                    "updated_at_ms": env.updated_at_ms,
                    "author_ref": env.author_ref,
                }
            )
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    if not envelopes:
        click.echo("no settings stored")
        return
    for env in envelopes:
        _print_envelope(env, output_format="text")


__all__ = ["setting_command"]
