from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

DEFAULT_CREDENTIALS = CONFIG_HOME / "credentials.json"
DEFAULT_TOKEN = CONFIG_HOME / "token.json"


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


def _convert_output_dirs(paths: CoreOutputPaths) -> OutputDirs:
    return OutputDirs(
        render=paths.render,
        sync_drive=paths.sync_drive,
        sync_codex=paths.sync_codex,
        sync_claude_code=paths.sync_claude_code,
        import_chatgpt=paths.import_chatgpt,
        import_claude=paths.import_claude,
    )


def _convert_defaults(core: CoreDefaults) -> Defaults:
    return Defaults(
        collapse_threshold=core.collapse_threshold,
        html_previews=core.html_previews,
        html_theme=core.html_theme,
        output_dirs=_convert_output_dirs(core.output_dirs),
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
    CONFIG_PATH = app_config.path
    return Config(
        defaults=_convert_defaults(app_config.defaults),
        index=_convert_index(app_config.index),
        exports=_convert_exports(app_config.exports),
        drive=DriveConfig(),
    )


CONFIG = load_config()
