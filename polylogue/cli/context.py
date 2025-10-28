from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from ..config import CONFIG
from ..settings import SETTINGS, Settings
from ..drive_client import DEFAULT_FOLDER_NAME


def _resolve_settings(settings: Optional[Settings]) -> Settings:
    return settings or SETTINGS


def default_sync_namespace(provider: str, settings: Optional[Settings] = None) -> argparse.Namespace:
    active = _resolve_settings(settings)
    return argparse.Namespace(
        provider=provider,
        out=None,
        links_only=False,
        dry_run=False,
        force=False,
        prune=False,
        collapse_threshold=None,
        html_mode=default_html_mode(active),
        diff=False,
        json=False,
        base_dir=None,
        all=False,
        folder_name=DEFAULT_FOLDER_NAME,
        folder_id=None,
        since=None,
        until=None,
        name_filter=None,
        list_only=False,
    )


def default_import_namespace(
    *,
    provider: str,
    sources: list[str],
    base_dir: Optional[Path],
    all_flag: bool,
    conversation_ids: list[str],
    settings: Optional[Settings] = None,
) -> argparse.Namespace:
    active = _resolve_settings(settings)
    return argparse.Namespace(
        provider=provider,
        source=sources,
        out=None,
        collapse_threshold=None,
        html_mode=default_html_mode(active),
        force=False,
        all=all_flag,
        conversation_ids=conversation_ids,
        base_dir=base_dir,
        json=False,
        to_clipboard=False,
    )


def merge_with_defaults(defaults: argparse.Namespace, overrides: argparse.Namespace) -> argparse.Namespace:
    merged = argparse.Namespace(**vars(defaults))
    for key, value in vars(overrides).items():
        setattr(merged, key, value)
    return merged

DEFAULT_COLLAPSE = CONFIG.defaults.collapse_threshold
DEFAULT_RENDER_OUT = CONFIG.defaults.output_dirs.render
DEFAULT_SYNC_OUT = CONFIG.defaults.output_dirs.sync_drive
DEFAULT_CODEX_SYNC_OUT = CONFIG.defaults.output_dirs.sync_codex
DEFAULT_CLAUDE_CODE_SYNC_OUT = CONFIG.defaults.output_dirs.sync_claude_code
DEFAULT_CHATGPT_OUT = CONFIG.defaults.output_dirs.import_chatgpt
DEFAULT_CLAUDE_OUT = CONFIG.defaults.output_dirs.import_claude

DEFAULT_OUTPUT_ROOTS = list(
    dict.fromkeys(
        [
            DEFAULT_RENDER_OUT,
            DEFAULT_SYNC_OUT,
            DEFAULT_CODEX_SYNC_OUT,
            DEFAULT_CLAUDE_CODE_SYNC_OUT,
            DEFAULT_CHATGPT_OUT,
            DEFAULT_CLAUDE_OUT,
        ]
    )
)


def ensure_settings_defaults(settings: Optional[Settings] = None) -> Settings:
    active = _resolve_settings(settings)
    active.html_previews = CONFIG.defaults.html_previews
    active.html_theme = CONFIG.defaults.html_theme
    return active


ensure_settings_defaults()


def default_html_mode(settings: Optional[Settings] = None) -> str:
    active = _resolve_settings(settings)
    return "on" if active.html_previews else "off"


def resolve_html_settings(args: argparse.Namespace, settings: Optional[Settings] = None) -> Tuple[bool, bool]:
    active = _resolve_settings(settings)
    mode = getattr(args, "html_mode", "auto") or "auto"
    mode = mode.lower()
    if mode == "on":
        return True, True
    if mode == "off":
        return False, True
    return active.html_previews, False


def resolve_html_enabled(args: argparse.Namespace, settings: Optional[Settings] = None) -> bool:
    return resolve_html_settings(args, settings)[0]


def resolve_output_path(path_value: Optional[str], fallback: Path) -> Path:
    if path_value:
        return Path(path_value).expanduser()
    return fallback


def resolve_collapse_value(value: Optional[int], default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default
