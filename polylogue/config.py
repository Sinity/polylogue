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
    load_configuration,
)
from .paths import CONFIG_HOME, DATA_HOME

CONFIG_DIR = CONFIG_HOME
DEFAULT_PATHS = list(DEFAULT_CONFIG_LOCATIONS)

DEFAULT_ARCHIVE_ROOT = Path(
    os.environ.get("POLYLOGUE_ARCHIVE_ROOT", str(DATA_HOME / "archive"))
).expanduser()
ARCHIVE_ROOT = DEFAULT_ARCHIVE_ROOT
MARKDOWN_ROOT = ARCHIVE_ROOT / "markdown"


@dataclass
class OutputDirs:
    render: Path = MARKDOWN_ROOT / "gemini-render"
    sync_drive: Path = MARKDOWN_ROOT / "gemini-sync"
    sync_codex: Path = MARKDOWN_ROOT / "codex"
    sync_claude_code: Path = MARKDOWN_ROOT / "claude-code"
    import_chatgpt: Path = MARKDOWN_ROOT / "chatgpt"
    import_claude: Path = MARKDOWN_ROOT / "claude"


@dataclass
class Defaults:
    collapse_threshold: int = 25
    html_previews: bool = False
    html_theme: str = "light"
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


def load_config() -> Config:
    global CONFIG_PATH
    app_config: CoreAppConfig = load_configuration()
    CONFIG_PATH = app_config.path
    return Config(defaults=_convert_defaults(app_config.defaults))


CONFIG = load_config()
