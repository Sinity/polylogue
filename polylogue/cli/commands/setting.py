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

from polylogue.paths import archive_root


def _print_envelope(env: object, *, output_format: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope

    assert isinstance(env, ArchiveUserSettingEnvelope)
    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "setting_key": env.setting_key,
                    "value": env.value,
                    "updated_at_ms": env.updated_at_ms,
                    "author_ref": env.author_ref,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    click.echo(f"{env.setting_key} = {env.value!r} (author={env.author_ref}, updated_at_ms={env.updated_at_ms})")


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
def setting_set_command(setting_key: str, value: str, output_format: str) -> None:
    """Insert-or-update one typed setting row (rejects unknown keys/values)."""

    from polylogue.api import Polylogue

    async def run() -> object:
        async with Polylogue(archive_root=archive_root()) as poly:
            return await poly.set_setting(setting_key, value)

    try:
        envelope = asyncio.run(run())
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_envelope(envelope, output_format=output_format)


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
