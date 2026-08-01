"""``AppEnv.config``/``.runtime`` must not force ``.services`` when a
``ResolvedRuntimeConfig`` is already available (polylogue-g3jk).

``AppEnv.services`` (``_lazy_services`` -> ``polylogue.services`` ->
``storage.repository``/``storage.sqlite``) is expensive to import and
constructs backend/repository machinery this property never needs: a
``RuntimeServices.get_config()`` call just returns the same
``runtime.as_config()`` projection ``AppEnv.config`` can compute directly.
``polylogue/cli/click_app.py`` always constructs ``AppEnv(runtime=runtime,
...)`` (no explicit ``services=``), so this is the real production shape the
daemon fast path hits on every query invocation, not a synthetic corner case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config, resolve_runtime_config


def test_config_returns_runtime_projection_without_building_services(workspace_env: dict[str, Path]) -> None:
    runtime = resolve_runtime_config()
    env = AppEnv(runtime=runtime, plain=True)

    config = env.config

    assert isinstance(config, Config)
    assert config.archive_root == runtime.as_config().archive_root
    assert config.db_path == runtime.as_config().db_path
    # The fast path must not have built (or cached) a RuntimeServices object.
    assert env._services is None


def test_runtime_property_returns_the_same_object_without_building_services(
    workspace_env: dict[str, Path],
) -> None:
    runtime = resolve_runtime_config()
    env = AppEnv(runtime=runtime, plain=True)

    assert env.runtime is runtime
    assert env._services is None


def test_config_still_works_when_services_is_explicitly_supplied(workspace_env: dict[str, Path]) -> None:
    """Backward compatibility: explicit ``services=`` (tests/library callers) still wins."""
    from polylogue.services import RuntimeServices

    explicit_root = workspace_env["archive_root"].parent / "explicit"
    explicit_root.mkdir()
    explicit_config = Config(archive_root=explicit_root, render_root=explicit_root, sources=[])
    services = RuntimeServices(config=explicit_config)

    # Also pass a *different* runtime to prove `.config` prefers the already-
    # constructed `services` object once one exists, exactly as before this
    # change (the fast path only applies when `services` was never built).
    runtime = resolve_runtime_config()
    env = AppEnv(runtime=runtime, services=services, plain=True)

    assert env.config is explicit_config
    assert env.config.archive_root == explicit_root


def test_config_raises_when_neither_runtime_nor_services_supplied() -> None:
    """No silent success: an AppEnv with nothing resolvable still errors clearly."""
    from polylogue.config import ConfigError

    env = AppEnv(plain=True)
    with pytest.raises(ConfigError):
        _ = env.config
