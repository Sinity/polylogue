"""Sample loading from polylogue database and session files.

Provides ProviderConfig and functions to load raw data samples for
schema inference. Knows how to locate and decode provider data, but
performs no schema manipulation.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import CORE_RUNTIME_PROVIDERS
from polylogue.lib.raw_payload import (
    build_raw_payload_envelope,
    collect_limited_samples,
    extract_payload_samples,
)
from polylogue.paths import db_path as default_db_path


@dataclass
class ProviderConfig:
    """Configuration for a provider's schema generation."""

    name: str
    description: str
    db_provider_name: str | None = None  # Provider name in polylogue DB
    session_dir: Path | None = None  # For JSONL session-based providers
    max_sessions: int | None = None
    sample_granularity: str = "document"  # "document" | "record"
    record_type_key: str | None = None  # best-effort stratification key


# Provider configurations
PROVIDERS: dict[str, ProviderConfig] = {
    "chatgpt": ProviderConfig(
        name="chatgpt",
        description="ChatGPT message format",
        db_provider_name="chatgpt",
        sample_granularity="document",
    ),
    "claude-code": ProviderConfig(
        name="claude-code",
        description="Claude Code message format",
        db_provider_name="claude-code",
        sample_granularity="record",
        record_type_key="type",
    ),
    "claude-ai": ProviderConfig(
        name="claude-ai",
        description="Claude AI web message format",
        db_provider_name="claude",  # DB uses "claude"
        sample_granularity="document",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        description="Gemini AI Studio message format",
        db_provider_name="gemini",
        sample_granularity="document",
    ),
    "codex": ProviderConfig(
        name="codex",
        description="OpenAI Codex CLI session format",
        session_dir=Path.home() / ".codex/sessions",
        max_sessions=100,
        sample_granularity="record",
        record_type_key="type",
    ),
}


def _resolve_provider_config(provider_name: str) -> ProviderConfig:
    config = next((c for c in PROVIDERS.values() if c.db_provider_name == provider_name), None)
    if config is not None:
        return config
    return ProviderConfig(
        name=provider_name,
        description=f"{provider_name} export format",
        db_provider_name=provider_name,
        sample_granularity="document",
    )


def _sample_provider_where_clause(provider_name: str) -> tuple[str, tuple[Any, ...]]:
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        "payload_provider = ? "
        "OR (payload_provider IS NULL AND provider_name = ?) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[Any, ...] = (
        provider_name,
        provider_name,
        *CORE_RUNTIME_PROVIDERS,
    )
    return clause, params


def _iter_samples_from_db(
    provider_name: str,
    *,
    db_path: Path,
    config: ProviderConfig,
) -> Any:
    conn = sqlite3.connect(db_path)
    try:
        where_clause, where_params = _sample_provider_where_clause(provider_name)
        cursor = conn.execute(
            f"""
            SELECT raw_content, source_path, provider_name, payload_provider
            FROM raw_conversations
            WHERE {where_clause}
            ORDER BY acquired_at DESC
            """,
            where_params,
        )
        while True:
            rows = cursor.fetchmany(250)
            if not rows:
                break
            for row in rows:
                try:
                    envelope = build_raw_payload_envelope(
                        row[0],
                        source_path=row[1],
                        fallback_provider=row[2],
                        payload_provider=row[3],
                        jsonl_dict_only=True,
                    )
                except Exception:
                    continue
                if envelope.provider != provider_name:
                    continue
                yield from extract_payload_samples(
                    envelope.payload,
                    sample_granularity=config.sample_granularity,
                    max_samples=None,
                    record_type_key=config.record_type_key,
                )
    finally:
        conn.close()


def _iter_samples_from_sessions(
    session_dir: Path,
    *,
    max_sessions: int | None,
) -> Any:
    if not session_dir.exists():
        return

    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(jsonl_files) > max_sessions:
        step = len(jsonl_files) // max_sessions
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            yield parsed
        except OSError:
            continue


def load_samples_from_db(
    provider_name: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw samples from polylogue database."""
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return []

    config = _resolve_provider_config(provider_name)
    if max_samples is None:
        return list(_iter_samples_from_db(provider_name, db_path=db_path, config=config))
    return collect_limited_samples(
        lambda: _iter_samples_from_db(provider_name, db_path=db_path, config=config),
        limit=max_samples,
        stratify=config.sample_granularity == "record",
        record_type_key=config.record_type_key,
    )


def load_samples_from_sessions(
    session_dir: Path,
    max_sessions: int | None = None,
    max_samples: int | None = None,
    record_type_key: str | None = None,
) -> list[dict[str, Any]]:
    """Load samples from JSONL session files."""
    if max_samples is None:
        return list(_iter_samples_from_sessions(session_dir, max_sessions=max_sessions))
    return collect_limited_samples(
        lambda: _iter_samples_from_sessions(session_dir, max_sessions=max_sessions),
        limit=max_samples,
        stratify=True,
        record_type_key=record_type_key,
    )


def get_sample_count_from_db(
    provider_name: str,
    db_path: Path | None = None,
) -> int:
    """Get total message count for a provider in database."""
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("""
            SELECT COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ?
        """, (provider_name,)).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


__all__ = [
    "ProviderConfig",
    "PROVIDERS",
    "_resolve_provider_config",
    "_sample_provider_where_clause",
    "_iter_samples_from_db",
    "_iter_samples_from_sessions",
    "load_samples_from_db",
    "load_samples_from_sessions",
    "get_sample_count_from_db",
]
