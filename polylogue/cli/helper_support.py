"""Shared CLI helper primitives."""

from __future__ import annotations

from typing import NoReturn

import click

from polylogue.cli.types import AppEnv
from polylogue.config import Config


def fail(command: str, message: str) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(message)


def load_effective_config(env: AppEnv) -> Config:
    """Return the effective runtime configuration."""
    return env.config


__all__ = ["fail", "load_effective_config"]
