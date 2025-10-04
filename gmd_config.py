from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_ENV = "GMD_CONFIG"
DEFAULT_PATHS = [
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gmd" / "config.json",
    Path.home() / ".gmdconfig",
]


@dataclass
class OutputDirs:
    render: Path = Path("gmd_out")
    sync_drive: Path = Path("gemini_synced")
    sync_codex: Path = Path("codex_synced")
    sync_claude_code: Path = Path("claude_code_synced")


@dataclass
class Defaults:
    collapse_threshold: int = 25
    html_previews: bool = False
    html_theme: str = "light"
    output_dirs: OutputDirs = field(default_factory=OutputDirs)


@dataclass
class Config:
    defaults: Defaults = field(default_factory=Defaults)


def _load_config_dict() -> Dict[str, Any]:
    candidates: list[Path] = []
    env_path = os.environ.get(CONFIG_ENV)
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(DEFAULT_PATHS)
    for path in candidates:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _load_defaults(data: Dict[str, Any]) -> Defaults:
    defaults = data.get("defaults") if isinstance(data, dict) else None
    if not isinstance(defaults, dict):
        return Defaults()
    collapse = defaults.get("collapse_threshold")
    html_previews = defaults.get("html_previews")
    html_theme = defaults.get("html_theme")
    out_dirs_cfg = defaults.get("output_dirs")
    output_dirs = OutputDirs()
    if isinstance(out_dirs_cfg, dict):
        if out_dirs_cfg.get("render"):
            output_dirs.render = Path(out_dirs_cfg["render"]).expanduser()
        if out_dirs_cfg.get("sync_drive"):
            output_dirs.sync_drive = Path(out_dirs_cfg["sync_drive"]).expanduser()
        if out_dirs_cfg.get("sync_codex"):
            output_dirs.sync_codex = Path(out_dirs_cfg["sync_codex"]).expanduser()
        if out_dirs_cfg.get("sync_claude_code"):
            output_dirs.sync_claude_code = Path(out_dirs_cfg["sync_claude_code"]).expanduser()
    return Defaults(
        collapse_threshold=int(collapse) if isinstance(collapse, (int, float)) else 25,
        html_previews=bool(html_previews) if html_previews is not None else False,
        html_theme=str(html_theme) if isinstance(html_theme, str) else "light",
        output_dirs=output_dirs,
    )


def load_config() -> Config:
    data = _load_config_dict()
    return Config(defaults=_load_defaults(data))


CONFIG = load_config()
