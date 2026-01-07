"""CLI output formatting utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from polylogue.config import Source
from polylogue.core.timestamps import format_timestamp


def should_use_plain(*, plain: bool, interactive: bool) -> bool:
    if plain:
        return True
    if interactive:
        return False
    env_force = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    if env_force and env_force.lower() not in {"0", "false", "no"}:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())


def announce_plain_mode() -> None:
    sys.stderr.write("Plain output active (non-TTY). Use --interactive from a TTY to re-enable prompts.\n")


def format_cursors(cursors: dict) -> str | None:
    if not cursors:
        return None
    parts: list[str] = []
    for name, cursor in cursors.items():
        detail_bits: list[str] = []
        file_count = cursor.get("file_count")
        if isinstance(file_count, int):
            detail_bits.append(f"{file_count} files")
        error_count = cursor.get("error_count")
        if isinstance(error_count, int) and error_count:
            detail_bits.append(f"{error_count} errors")
        latest_mtime = cursor.get("latest_mtime")
        latest_label = None
        if isinstance(latest_mtime, (int, float)):
            latest_label = format_timestamp(int(latest_mtime))
        else:
            latest_name = cursor.get("latest_file_name")
            latest_path = cursor.get("latest_path")
            if isinstance(latest_name, str):
                latest_label = latest_name
            elif isinstance(latest_path, str):
                latest_label = Path(latest_path).name
        if latest_label:
            detail_bits.append(f"latest {latest_label}")
        detail = ", ".join(detail_bits) if detail_bits else "unknown"
        parts.append(f"{name} ({detail})")
    return "; ".join(parts)


def format_counts(counts: dict) -> str:
    parts = [
        f"{counts.get('conversations', 0)} conv",
        f"{counts.get('messages', 0)} msg",
    ]
    rendered = counts.get("rendered", 0)
    if rendered:
        parts.append(f"{rendered} rendered")
    return ", ".join(parts)


def format_index_status(stage: str, indexed: bool, index_error: str | None) -> str:
    if stage in {"ingest", "render"}:
        return "Index: skipped"
    if index_error:
        return "Index: error"
    if indexed:
        return "Index: ok"
    return "Index: up-to-date"


def format_source_label(source_name: str | None, provider_name: str) -> str:
    if source_name and source_name != provider_name:
        return f"{source_name}/{provider_name}"
    return source_name or provider_name


def format_sources_summary(sources: list[Source]) -> str:
    if not sources:
        return "none"
    labels: list[str] = []
    for source in sources:
        if source.folder:
            labels.append(f"{source.name} (drive)")
        elif source.path:
            labels.append(source.name)
        else:
            labels.append(f"{source.name} (missing)")
    if len(labels) > 8:
        extra = len(labels) - 8
        labels = labels[:8] + [f"+{extra} more"]
    return ", ".join(labels)
