"""Health command."""

from __future__ import annotations

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.health import get_health


@click.command("health")
@click.pass_obj
def health_command(env: AppEnv) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("health", str(exc))
    payload = get_health(config)
    cached = payload.get("cached")
    age = payload.get("age_seconds")
    header = f"Health (cached={cached}, age={age}s)" if cached is not None else "Health"
    checks = payload.get("checks", [])
    lines = []
    for check in checks:
        name = check.get("name")
        status = check.get("status")
        detail = check.get("detail")
        lines.append(f"{name}: {status} - {detail}")
    env.ui.summary(header, lines)
