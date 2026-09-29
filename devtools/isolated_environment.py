"""Environments that isolate a scratch daemon from the host's sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def isolated_home_environment(inherited: Mapping[str, str], *, home: Path) -> dict[str, str]:
    """``inherited`` with every source-discovery root pointed into ``home``.

    Sources are acquired only from canonical locations under ``HOME``, so a
    scratch daemon run must replace ``HOME`` and every XDG root, and drop the
    Polylogue path overrides: any one of them inherited would add the
    operator's real sources to the run. ``_user_config_path`` also falls
    through to ``<cwd>/polylogue.toml`` when no ``POLYLOGUE_CONFIG`` is set,
    so an untracked project-local config in the launch checkout's cwd (the
    dev-loop proof deliberately runs the daemon from the repository root)
    would still be discovered even with HOME and every XDG root replaced.
    ``POLYLOGUE_CONFIG`` is pointed at a nonexistent path under the isolated
    home instead of dropped: an explicit override -- present or not --
    disables that cwd fallback outright.
    """
    env = dict(inherited)
    env["HOME"] = str(home)
    for variable, relative in (
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_STATE_HOME", ".local/state"),
        ("XDG_CACHE_HOME", ".cache"),
    ):
        env[variable] = str(home / relative)
    env["POLYLOGUE_SITE_CONFIG"] = ""
    env["POLYLOGUE_CONFIG"] = str(home / "unconfigured-polylogue.toml")
    for variable in (
        "POLYLOGUE_HERMES_ROOT",
        "POLYLOGUE_BROWSER_CAPTURE_SPOOL_PATH",
        "POLYLOGUE_HOOK_SIDECAR_DIR",
        "POLYLOGUE_CREDENTIAL_PATH",
        "POLYLOGUE_TOKEN_PATH",
    ):
        env.pop(variable, None)
    return env
