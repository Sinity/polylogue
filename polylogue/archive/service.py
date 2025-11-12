from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ..config import Config, Defaults, OutputDirs


@dataclass
class Archive:
    """High-level view of archive directories derived from configuration."""

    config: Config
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        defaults = self.config.defaults
        render_root = defaults.output_dirs.render
        self.root = render_root.parent if render_root.parent else render_root

    @property
    def defaults(self) -> Defaults:
        return self.config.defaults

    def markdown_root(self) -> Path:
        return self.defaults.output_dirs.render.parent

    def provider_root(self, provider: str) -> Path:
        mapping: Dict[str, Path] = {
            "render": self.defaults.output_dirs.render,
            "drive": self.defaults.output_dirs.sync_drive,
            "codex": self.defaults.output_dirs.sync_codex,
            "claude-code": self.defaults.output_dirs.sync_claude_code,
            "chatgpt": self.defaults.output_dirs.import_chatgpt,
            "claude": self.defaults.output_dirs.import_claude,
        }
        if provider in mapping:
            return mapping[provider]
        return self.markdown_root() / provider

    def conversation_dir(self, provider: str, slug: str) -> Path:
        return self.provider_root(provider) / slug

    def conversation_markdown(self, provider: str, slug: str) -> Path:
        return self.conversation_dir(provider, slug) / "conversation.md"

    def ensure_provider_root(self, provider: str) -> Path:
        path = self.provider_root(provider)
        path.mkdir(parents=True, exist_ok=True)
        return path
