from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..paths import CONFIG_HOME, DATA_HOME
from .config_validation import validate_config_payload

CONFIG_ENV = "POLYLOGUE_CONFIG"
DEFAULT_CONFIG_LOCATIONS = [
    CONFIG_HOME / "config.json",
    Path.home() / ".polylogueconfig",
]


@dataclass
class OutputPaths:
    render: Path
    sync_drive: Path
    sync_codex: Path
    sync_claude_code: Path
    import_chatgpt: Path
    import_claude: Path

    @classmethod
    def default(cls) -> "OutputPaths":
        markdown_root = DATA_HOME / "archive" / "markdown"
        return cls(
            render=markdown_root / "gemini-render",
            sync_drive=markdown_root / "gemini-sync",
            sync_codex=markdown_root / "codex",
            sync_claude_code=markdown_root / "claude-code",
            import_chatgpt=markdown_root / "chatgpt",
            import_claude=markdown_root / "claude",
        )


@dataclass
class Defaults:
    collapse_threshold: int = 25
    html_previews: bool = False
    html_theme: str = "light"
    output_dirs: OutputPaths = field(default_factory=OutputPaths.default)


@dataclass
class AppConfig:
    defaults: Defaults = field(default_factory=Defaults)
    path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_config_sources() -> Tuple[Dict[str, Any], Optional[Path]]:
    candidates: list[Path] = []
    env_path = os.environ.get(CONFIG_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(DEFAULT_CONFIG_LOCATIONS)

    for path in candidates:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data, path
        except Exception:
            continue
    return {}, None


def _load_defaults(data: Dict[str, Any]) -> Defaults:
    defaults_raw = data.get("defaults") if isinstance(data, dict) else None
    defaults = Defaults()
    if not isinstance(defaults_raw, dict):
        return defaults

    collapse = defaults_raw.get("collapse_threshold")
    if isinstance(collapse, (int, float)) and collapse > 0:
        defaults.collapse_threshold = int(collapse)

    html_previews = defaults_raw.get("html_previews")
    if isinstance(html_previews, bool):
        defaults.html_previews = html_previews

    html_theme = defaults_raw.get("html_theme")
    if isinstance(html_theme, str) and html_theme.strip():
        defaults.html_theme = html_theme.strip()

    out_dirs_cfg = defaults_raw.get("output_dirs")
    if isinstance(out_dirs_cfg, dict):
        paths = defaults.output_dirs
        if out_dirs_cfg.get("render"):
            paths.render = Path(out_dirs_cfg["render"]).expanduser()
        if out_dirs_cfg.get("sync_drive"):
            paths.sync_drive = Path(out_dirs_cfg["sync_drive"]).expanduser()
        if out_dirs_cfg.get("sync_codex"):
            paths.sync_codex = Path(out_dirs_cfg["sync_codex"]).expanduser()
        if out_dirs_cfg.get("sync_claude_code"):
            paths.sync_claude_code = Path(out_dirs_cfg["sync_claude_code"]).expanduser()
        if out_dirs_cfg.get("import_chatgpt"):
            paths.import_chatgpt = Path(out_dirs_cfg["import_chatgpt"]).expanduser()
        if out_dirs_cfg.get("import_claude"):
            paths.import_claude = Path(out_dirs_cfg["import_claude"]).expanduser()

    return defaults


def load_configuration() -> AppConfig:
    data, path = _load_config_sources()
    if data:
        validate_config_payload(data)
    defaults = _load_defaults(data)
    return AppConfig(defaults=defaults, path=path, raw=data if isinstance(data, dict) else {})
