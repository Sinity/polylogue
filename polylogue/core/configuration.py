from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..paths import CONFIG_HOME, DATA_HOME
from .config_validation import validate_config_payload
from ..util import DEFAULT_CODEX_HOME, DEFAULT_CLAUDE_CODE_HOME

CONFIG_ENV = "POLYLOGUE_CONFIG"
DEFAULT_CONFIG_LOCATIONS = [
    CONFIG_HOME / "config.json",
    Path.home() / ".polylogueconfig",
]
DEFAULT_EXPORTS_CHATGPT = DATA_HOME / "exports" / "chatgpt"
DEFAULT_EXPORTS_CLAUDE = DATA_HOME / "exports" / "claude"


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
    drive: Optional["DriveConfig"] = None
    index: Optional["IndexConfig"] = None
    exports: Optional["ExportsConfig"] = None
    path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriveConfig:
    credentials_path: Optional[Path] = None
    token_path: Optional[Path] = None
    retries: Optional[int] = None
    retry_base: Optional[float] = None


@dataclass
class IndexConfig:
    backend: str = "sqlite"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "polylogue"
    qdrant_vector_size: Optional[int] = None


@dataclass
class ExportsConfig:
    chatgpt: Path
    claude: Path


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


def _load_drive(data: Dict[str, Any]) -> Optional[DriveConfig]:
    drive_raw = data.get("drive") if isinstance(data, dict) else None
    if not isinstance(drive_raw, dict):
        return None
    cfg = DriveConfig()
    if drive_raw.get("credentials_path"):
        cfg.credentials_path = Path(drive_raw["credentials_path"]).expanduser()
    if drive_raw.get("token_path"):
        cfg.token_path = Path(drive_raw["token_path"]).expanduser()
    retries = drive_raw.get("retries")
    if isinstance(retries, int) and retries >= 1:
        cfg.retries = retries
    retry_base = drive_raw.get("retry_base")
    if isinstance(retry_base, (int, float)) and retry_base >= 0:
        cfg.retry_base = float(retry_base)
    return cfg


def _load_index(data: Dict[str, Any]) -> IndexConfig:
    index_raw = data.get("index") if isinstance(data, dict) else None
    cfg = IndexConfig()
    if not isinstance(index_raw, dict):
        return cfg
    backend = index_raw.get("backend")
    if isinstance(backend, str) and backend.strip():
        cfg.backend = backend.strip().lower()
    qdrant_raw = index_raw.get("qdrant")
    if isinstance(qdrant_raw, dict):
        if qdrant_raw.get("url"):
            cfg.qdrant_url = str(qdrant_raw["url"]).strip()
        if qdrant_raw.get("api_key"):
            cfg.qdrant_api_key = str(qdrant_raw["api_key"]).strip()
        if qdrant_raw.get("collection"):
            cfg.qdrant_collection = str(qdrant_raw["collection"]).strip()
        vector_size = qdrant_raw.get("vector_size")
        if isinstance(vector_size, int) and vector_size >= 1:
            cfg.qdrant_vector_size = vector_size
    return cfg


def _load_exports(data: Dict[str, Any]) -> ExportsConfig:
    exports_raw = data.get("exports") if isinstance(data, dict) else None
    chatgpt = DEFAULT_EXPORTS_CHATGPT
    claude = DEFAULT_EXPORTS_CLAUDE
    if isinstance(exports_raw, dict):
        if exports_raw.get("chatgpt"):
            chatgpt = Path(exports_raw["chatgpt"]).expanduser()
        if exports_raw.get("claude"):
            claude = Path(exports_raw["claude"]).expanduser()
    return ExportsConfig(chatgpt=chatgpt, claude=claude)


def load_configuration() -> AppConfig:
    data, path = _load_config_sources()
    if data:
        validate_config_payload(data)
    defaults = _load_defaults(data)
    drive = _load_drive(data)
    index = _load_index(data)
    exports = _load_exports(data)
    return AppConfig(
        defaults=defaults,
        drive=drive,
        index=index,
        exports=exports,
        path=path,
        raw=data if isinstance(data, dict) else {},
    )
