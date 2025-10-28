from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from ..config import CONFIG
from ..settings import SETTINGS
from ..drive_client import DEFAULT_FOLDER_NAME


def default_sync_namespace(provider: str) -> argparse.Namespace:
    return argparse.Namespace(
        provider=provider,
        out=None,
        links_only=False,
        dry_run=False,
        force=False,
        prune=False,
        collapse_threshold=None,
        html_mode=default_html_mode(),
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
) -> argparse.Namespace:
    return argparse.Namespace(
        provider=provider,
        source=sources,
        out=None,
        collapse_threshold=None,
        html_mode=default_html_mode(),
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


def ensure_settings_defaults() -> None:
    SETTINGS.html_previews = CONFIG.defaults.html_previews
    SETTINGS.html_theme = CONFIG.defaults.html_theme


ensure_settings_defaults()


def default_html_mode() -> str:
    return "on" if SETTINGS.html_previews else "off"


def resolve_html_settings(args: argparse.Namespace) -> Tuple[bool, bool]:
    mode = getattr(args, "html_mode", "auto") or "auto"
    mode = mode.lower()
    if mode == "on":
        return True, True
    if mode == "off":
        return False, True
    return SETTINGS.html_previews, False


def resolve_html_enabled(args: argparse.Namespace) -> bool:
    return resolve_html_settings(args)[0]
