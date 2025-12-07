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
]
DEFAULT_INPUT_ROOT = DATA_HOME / "inbox"
DEFAULT_OUTPUT_ROOT = DATA_HOME / "archive"


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
        output_root = DEFAULT_OUTPUT_ROOT
        return cls(
            render=output_root / "render",
            sync_drive=output_root / "gemini",
            sync_codex=output_root / "codex",
            sync_claude_code=output_root / "claude-code",
            import_chatgpt=output_root / "chatgpt",
            import_claude=output_root / "claude",
        )


@dataclass
class Defaults:
    collapse_threshold: int = 25
    html_previews: bool = True
    html_theme: str = "dark"
    output_dirs: OutputPaths = field(default_factory=OutputPaths.default)


@dataclass
class AppConfig:
    defaults: Defaults = field(default_factory=Defaults)
    index: Optional["IndexConfig"] = None
    exports: Optional["ExportsConfig"] = None
    path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)


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


def _load_paths(data: Dict[str, Any]) -> Tuple[Path, Path]:
    paths_raw = data.get("paths") if isinstance(data, dict) else None
    input_root = DEFAULT_INPUT_ROOT
    output_root = DEFAULT_OUTPUT_ROOT
    if isinstance(paths_raw, dict):
        raw_input = paths_raw.get("input_root")
        if isinstance(raw_input, str) and raw_input.strip():
            input_root = Path(raw_input).expanduser()
        raw_output = paths_raw.get("output_root")
        if isinstance(raw_output, str) and raw_output.strip():
            output_root = Path(raw_output).expanduser()
    return input_root, output_root


def _load_defaults(data: Dict[str, Any], *, output_root: Path) -> Defaults:
    ui_raw = None
    if isinstance(data, dict):
        ui_raw = data.get("ui") or data.get("defaults")
    defaults = Defaults(
        output_dirs=OutputPaths(
            render=output_root / "render",
            sync_drive=output_root / "gemini",
            sync_codex=output_root / "codex",
            sync_claude_code=output_root / "claude-code",
            import_chatgpt=output_root / "chatgpt",
            import_claude=output_root / "claude",
        )
    )
    if not isinstance(ui_raw, dict):
        return defaults

    collapse = ui_raw.get("collapse_threshold")
    if isinstance(collapse, (int, float)) and collapse > 0:
        defaults.collapse_threshold = int(collapse)

    html_enabled = ui_raw.get("html")
    if isinstance(html_enabled, bool):
        defaults.html_previews = html_enabled

    html_theme = ui_raw.get("theme")
    if isinstance(html_theme, str) and html_theme.strip():
        defaults.html_theme = html_theme.strip()

    return defaults


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


def _load_exports(input_root: Path) -> ExportsConfig:
    return ExportsConfig(chatgpt=input_root, claude=input_root)


def load_configuration() -> AppConfig:
    data, path = _load_config_sources()
    if data:
        validate_config_payload(data)
    input_root, output_root = _load_paths(data)
    defaults = _load_defaults(data, output_root=output_root)
    index = _load_index(data)
    exports = _load_exports(input_root)
    return AppConfig(
        defaults=defaults,
        index=index,
        exports=exports,
        path=path,
        raw=data if isinstance(data, dict) else {},
    )
