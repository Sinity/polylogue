from __future__ import annotations

import os
import json
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .core.configuration import (
    CONFIG_ENV,
    DEFAULT_CONFIG_LOCATIONS,
    AppConfig as CoreAppConfig,
    Defaults as CoreDefaults,
    OutputPaths as CoreOutputPaths,
    IndexConfig as CoreIndexConfig,
    ExportsConfig as CoreExportsConfig,
    load_configuration,
)
from .paths import CONFIG_HOME, DATA_HOME

# Public aliases for config dataclasses used outside the core configuration module.
IndexConfig = CoreIndexConfig
ExportsConfig = CoreExportsConfig

_ENV_CREDENTIAL_PATH = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
_ENV_TOKEN_PATH = os.environ.get("POLYLOGUE_TOKEN_PATH")

DEFAULT_CREDENTIALS = Path(_ENV_CREDENTIAL_PATH).expanduser() if _ENV_CREDENTIAL_PATH else CONFIG_HOME / "credentials.json"
DEFAULT_TOKEN = Path(_ENV_TOKEN_PATH).expanduser() if _ENV_TOKEN_PATH else CONFIG_HOME / "token.json"


@dataclass
class DriveConfig:
    credentials_path: Path = DEFAULT_CREDENTIALS
    token_path: Path = DEFAULT_TOKEN
    retries: int = 3
    retry_base: float = 0.5

CONFIG_DIR = CONFIG_HOME
DEFAULT_PATHS = list(DEFAULT_CONFIG_LOCATIONS)

DEFAULT_INPUT_ROOT = DATA_HOME / "inbox"
DEFAULT_OUTPUT_ROOT = DATA_HOME / "archive"
DEFAULT_EXPORTS_CHATGPT = DEFAULT_INPUT_ROOT
DEFAULT_EXPORTS_CLAUDE = DEFAULT_INPUT_ROOT
_DECLARATIVE_FLAG = os.environ.get("POLYLOGUE_DECLARATIVE")


@dataclass
class OutputDirs:
    render: Path = DEFAULT_OUTPUT_ROOT / "render"
    sync_drive: Path = DEFAULT_OUTPUT_ROOT / "gemini"
    sync_codex: Path = DEFAULT_OUTPUT_ROOT / "codex"
    sync_claude_code: Path = DEFAULT_OUTPUT_ROOT / "claude-code"
    import_chatgpt: Path = DEFAULT_OUTPUT_ROOT / "chatgpt"
    import_claude: Path = DEFAULT_OUTPUT_ROOT / "claude"


@dataclass
class Defaults:
    collapse_threshold: int = 25
    html_previews: bool = True
    html_theme: str = "dark"
    output_dirs: OutputDirs = field(default_factory=OutputDirs)
    roots: dict[str, OutputDirs] = field(default_factory=dict)

    @property
    def render(self) -> Path:
        return self.output_dirs.render

    @property
    def sync_drive(self) -> Path:
        return self.output_dirs.sync_drive

    @property
    def sync_codex(self) -> Path:
        return self.output_dirs.sync_codex

    @property
    def sync_claude_code(self) -> Path:
        return self.output_dirs.sync_claude_code

    @property
    def import_chatgpt(self) -> Path:
        return self.output_dirs.import_chatgpt

    @property
    def import_claude(self) -> Path:
        return self.output_dirs.import_claude


@dataclass
class Config:
    defaults: Defaults = field(default_factory=Defaults)
    index: Optional[IndexConfig] = None
    exports: ExportsConfig = field(default_factory=lambda: ExportsConfig(chatgpt=DEFAULT_EXPORTS_CHATGPT, claude=DEFAULT_EXPORTS_CLAUDE))
    drive: DriveConfig = field(default_factory=DriveConfig)


CONFIG_PATH: Optional[Path] = None


def _truthy(val: Optional[str]) -> bool:
    return bool(val) and str(val).strip().lower() in {"1", "true", "yes", "on"}


def is_config_declarative(path: Optional[Path] = None) -> Tuple[bool, str, Path]:
    """Detect declarative/immutable configs (e.g., NixOS module) to avoid runtime edits."""
    target = path or CONFIG_PATH or (CONFIG_HOME / "config.json")
    resolved = target
    try:
        resolved = target.resolve()
    except Exception:
        pass

    if _truthy(_DECLARATIVE_FLAG):
        return True, "POLYLOGUE_DECLARATIVE is set", target

    if str(resolved).startswith("/nix/store/"):
        return True, f"config file is in nix store ({resolved})", target

    # If file exists but is not writable, treat as declarative
    if target.exists():
        writable = os.access(target, os.W_OK)
        try:
            mode = target.stat().st_mode
            writable = writable and bool(mode & stat.S_IWUSR)
        except Exception:
            pass
        if not writable:
            return True, "config file is read-only", target
    else:
        parent = target.parent
        existing_parent = parent
        try:
            while not existing_parent.exists() and existing_parent != existing_parent.parent:
                existing_parent = existing_parent.parent
        except Exception:
            existing_parent = parent
        try:
            parent_resolved = existing_parent.resolve()
        except Exception:
            parent_resolved = existing_parent
        if str(parent_resolved).startswith("/nix/store/"):
            return True, f"config parent is in nix store ({parent_resolved})", target
        if existing_parent.exists() and not os.access(existing_parent, os.W_OK):
            return True, "config parent directory is read-only", target

    return False, "", target


def persist_config(
    *,
    input_root: Path,
    output_root: Path,
    collapse_threshold: int,
    html_previews: bool,
    html_theme: str,
    index: Optional[IndexConfig] = None,
    roots: Optional[Dict[str, object]] = None,
    path: Optional[Path] = None,
) -> Path:
    """Write a config.json that mirrors the sample schema.

    Existing index settings are preserved via the supplied IndexConfig to avoid
    silently dropping Qdrant settings when re-initializing.
    """
    target = path or CONFIG_HOME / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    index_cfg = index or IndexConfig()
    payload = {
        "paths": {
            "input_root": str(input_root.expanduser()),
            "output_root": str(output_root.expanduser()),
        },
        "ui": {
            "collapse_threshold": collapse_threshold,
            "html": bool(html_previews),
            "theme": html_theme,
        },
        "index": {
            "backend": index_cfg.backend,
            "qdrant": {
                "url": index_cfg.qdrant_url,
                "api_key": index_cfg.qdrant_api_key,
                "collection": index_cfg.qdrant_collection,
                "vector_size": index_cfg.qdrant_vector_size,
            },
        },
    }
    if roots:
        mapped: Dict[str, Dict[str, str]] = {}
        for label, base in roots.items():
            if isinstance(base, OutputDirs):
                root_path = base.render.parent.expanduser()
            else:
                root_path = Path(base).expanduser()
            mapped[label] = {
                "render": str(root_path / "render"),
                "sync_drive": str(root_path / "gemini"),
                "sync_codex": str(root_path / "codex"),
                "sync_claude_code": str(root_path / "claude-code"),
                "import_chatgpt": str(root_path / "chatgpt"),
                "import_claude": str(root_path / "claude"),
            }
        payload["paths"]["roots"] = mapped
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _convert_output_dirs(paths: CoreOutputPaths) -> OutputDirs:
    return OutputDirs(
        render=paths.render,
        sync_drive=paths.sync_drive,
        sync_codex=paths.sync_codex,
        sync_claude_code=paths.sync_claude_code,
        import_chatgpt=paths.import_chatgpt,
        import_claude=paths.import_claude,
    )


def _convert_defaults(core: CoreDefaults, output_paths: CoreOutputPaths) -> Defaults:
    roots: dict[str, OutputDirs] = {}
    for label, paths in getattr(core, "roots", {}).items():
        roots[label] = _convert_output_dirs(paths)
    return Defaults(
        collapse_threshold=core.collapse_threshold,
        html_previews=core.html_previews,
        html_theme=core.html_theme,
        output_dirs=_convert_output_dirs(output_paths),
        roots=roots,
    )


def _convert_index(core: Optional[CoreIndexConfig]) -> Optional[IndexConfig]:
    if not core:
        return None
    from .core.configuration import IndexConfig as CoreIndex
    if isinstance(core, IndexConfig):
        return core
    if isinstance(core, CoreIndex):
        return IndexConfig(
            backend=core.backend,
            qdrant_url=core.qdrant_url,
            qdrant_api_key=core.qdrant_api_key,
            qdrant_collection=core.qdrant_collection,
            qdrant_vector_size=core.qdrant_vector_size,
        )
    return None


def _convert_exports(core: CoreExportsConfig) -> ExportsConfig:
    return ExportsConfig(chatgpt=core.chatgpt, claude=core.claude)


def load_config() -> Config:
    global CONFIG_PATH
    app_config: CoreAppConfig = load_configuration()
    CONFIG_PATH = app_config.config_path

    # Get exports, use default if not present
    from .core.configuration import ExportsConfig as CoreExports
    exports_config = getattr(app_config, 'exports', None) or CoreExports()

    # Get output paths from app config
    output_paths_config = app_config.get_output_paths()

    return Config(
        defaults=_convert_defaults(app_config.defaults, output_paths_config),
        index=_convert_index(app_config.index),
        exports=_convert_exports(exports_config),
        drive=DriveConfig(),
    )


CONFIG = load_config()
