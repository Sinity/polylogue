"""Shared CLI helper primitives."""

from __future__ import annotations

from typing import NoReturn

from polylogue.cli.types import AppEnv
from polylogue.config import Config


def fail(command: str, message: str) -> NoReturn:
    raise SystemExit(f"{command}: {message}")


def load_effective_config(env: AppEnv) -> Config:
    """Return the effective runtime configuration."""
    return env.config


__all__ = ["fail", "load_effective_config"]
