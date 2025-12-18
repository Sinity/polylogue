from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

from ..config import CONFIG
from ..settings import SETTINGS, Settings, ensure_settings_defaults
from ..drive_client import DEFAULT_FOLDER_NAME


def _resolve_settings(settings: Optional[Settings]) -> Settings:
    return settings or SETTINGS


def default_sync_namespace(provider: str, settings: Optional[Settings] = None) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        out=None,
        links_only=False,
        dry_run=False,
        force=False,
        prune=False,
        resume_from=None,
        collapse_threshold=None,
        html_mode="auto",
        diff=False,
        json=False,
        chat_ids=None,
        sessions=None,
        base_dir=None,
        all=False,
        folder_name=DEFAULT_FOLDER_NAME,
        folder_id=None,
        since=None,
        until=None,
        name_filter=None,
        list_only=False,
        root=None,
    )


def default_import_namespace(
    *,
    provider: str,
    sources: list[str],
    base_dir: Optional[Path],
    all_flag: bool,
    conversation_ids: list[str],
    settings: Optional[Settings] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        source=sources,
        out=None,
        collapse_threshold=None,
        html_mode="auto",
        force=False,
        all=all_flag,
        conversation_ids=conversation_ids,
        base_dir=base_dir,
        json=False,
        to_clipboard=False,
    )


def merge_with_defaults(defaults: SimpleNamespace, overrides: SimpleNamespace) -> SimpleNamespace:
    merged = SimpleNamespace(**vars(defaults))
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


def _collect_output_dirs(dirs) -> list[Path]:
    return [
        dirs.render,
        dirs.sync_drive,
        dirs.sync_codex,
        dirs.sync_claude_code,
        dirs.import_chatgpt,
        dirs.import_claude,
    ]


_base_output_dirs = _collect_output_dirs(CONFIG.defaults.output_dirs)
_labeled_output_dirs: list[Path] = []
for paths in getattr(CONFIG.defaults, "roots", {}).values():
    _labeled_output_dirs.extend(_collect_output_dirs(paths))

DEFAULT_OUTPUT_ROOTS = list(dict.fromkeys(_base_output_dirs + _labeled_output_dirs))


ensure_settings_defaults()


def default_html_mode(settings: Optional[Settings] = None) -> str:
    # Preserve the documented "auto" default so settings/env can drive the fallback.
    return "auto"


def resolve_html_settings(args: object, settings: Optional[Settings] = None) -> Tuple[bool, bool]:
    active = _resolve_settings(settings)
    mode = getattr(args, "html_mode", "auto") or "auto"
    mode = mode.lower()
    if mode == "on":
        return True, True
    if mode == "off":
        return False, True
    return active.html_previews, False


def resolve_html_enabled(args: object, settings: Optional[Settings] = None) -> bool:
    return resolve_html_settings(args, settings)[0]


def resolve_output_path(path_value: Optional[str], fallback: Path) -> Path:
    if path_value:
        return Path(path_value).expanduser()
    return fallback


def resolve_collapse_value(value: Optional[int], settings: Optional[Settings] = None) -> int:
    """Resolve collapse threshold from args, settings, then CONFIG.

    Priority order:
    1. Explicit command-line argument
    2. User settings (from 'polylogue config set --collapse-threshold')
    3. CONFIG defaults

    Passing 0 disables collapsing entirely.
    """
    if isinstance(value, int):
        if value >= 0:
            return value

    active = _resolve_settings(settings)
    if active.collapse_threshold is not None and active.collapse_threshold >= 0:
        return active.collapse_threshold

    return DEFAULT_COLLAPSE


def resolve_collapse_thresholds(args: object, settings: Optional[Settings] = None) -> Dict[str, int]:
    """Return per-type collapse thresholds with sensible fallbacks."""

    def _clean(val: Optional[int]) -> Optional[int]:
        if isinstance(val, int) and val >= 0:
            return val
        return None

    base = resolve_collapse_value(getattr(args, "collapse_threshold", None), settings)
    msg = _clean(getattr(args, "collapse_threshold_message", None))
    tool = _clean(getattr(args, "collapse_threshold_tool", None))
    return {
        "message": msg if msg is not None else base,
        "tool": tool if tool is not None else base,
    }


def parse_meta_items(items: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeatable key=value pairs (e.g., Click's multiple=True option)."""

    meta: Dict[str, str] = {}
    if not items:
        return meta
    for item in items:
        raw = str(item)
        if "=" not in raw:
            raise SystemExit(f"Invalid --meta value {raw!r} (expected key=value).")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --meta value {raw!r} (empty key).")
        meta[key] = value
    return meta
