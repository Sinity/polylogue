"""Neutral packaged resources for native client upgrade controls."""

from pathlib import Path

from polylogue.agent_integration.assets import ALL_ASSETS, read_agent_asset


def copy_agent_package(root: Path) -> Path:
    root.mkdir()
    for name in ALL_ASSETS:
        (root / name).write_text(read_agent_asset(name), encoding="utf-8")
    return root
