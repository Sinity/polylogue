"""Tooling-side sample loading from the polylogue database and session files."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import (
    CORE_RUNTIME_PROVIDERS,
    canonical_runtime_provider,
    canonical_schema_provider,
)
from polylogue.lib.raw_payload import (
    build_raw_payload_envelope,
    collect_limited_samples,
    extract_record_samples_from_raw_content,
)
from polylogue.paths import db_path as default_db_path
from polylogue.schemas.observation import (
    ProviderConfig,
    extract_schema_units_from_payload,
    resolve_provider_config,
)
from polylogue.types import Provider


def _sample_provider_where_clause(provider_name: str | Provider) -> tuple[str, tuple[Any, ...]]:
    provider_token = str(Provider.from_string(provider_name))
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        "payload_provider = ? "
        "OR (payload_provider IS NULL AND provider_name = ?) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[Any, ...] = (
        provider_token,
        provider_token,
        *CORE_RUNTIME_PROVIDERS,
    )
    return clause, params


def _iter_schema_units_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> Any:
    """Yield clusterable schema units from raw_conversations."""
    provider_name = Provider.from_string(provider_name)
    conn = sqlite3.connect(db_path)
    try:
        query_provider = config.db_provider_name or provider_name
        where_clause, where_params = _sample_provider_where_clause(query_provider)
        cursor = conn.execute(
            f"""
            SELECT raw_content, source_path, provider_name, payload_provider, raw_id, file_mtime, acquired_at
            FROM raw_conversations
            WHERE {where_clause}
            """,
            where_params,
        )
        batch_size = 1 if config.sample_granularity == "record" else 100
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                if config.sample_granularity == "record":
                    runtime_provider = canonical_runtime_provider(row[3] or row[2])
                    if canonical_schema_provider(runtime_provider) != str(provider_name):
                        continue

                    sample_limit = max_samples
                    if sample_limit is None:
                        sample_limit = config.schema_sample_cap or 128

                    try:
                        samples = extract_record_samples_from_raw_content(
                            row[0],
                            max_samples=sample_limit,
                            record_type_key=config.record_type_key,
                        )
                    except Exception:
                        samples = []

                    if samples:
                        yield from extract_schema_units_from_payload(
                            samples,
                            provider_name=provider_name,
                            source_path=row[1],
                            raw_id=row[4],
                            observed_at=row[5] or row[6],
                            config=config,
                            max_samples=max_samples,
                        )
                        continue

                try:
                    envelope = build_raw_payload_envelope(
                        row[0],
                        source_path=row[1],
                        fallback_provider=row[2],
                        payload_provider=row[3],
                        jsonl_dict_only=config.sample_granularity == "record",
                    )
                except Exception:
                    continue
                if canonical_schema_provider(envelope.provider) != str(provider_name):
                    continue
                yield from extract_schema_units_from_payload(
                    envelope.payload,
                    provider_name=provider_name,
                    source_path=row[1],
                    raw_id=row[4],
                    observed_at=row[5] or row[6],
                    config=config,
                    max_samples=max_samples,
                )
    finally:
        conn.close()


def _iter_schema_units_from_sessions(
    provider_name: Provider,
    session_dir: Path,
    *,
    max_sessions: int | None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> Any:
    """Yield clusterable schema units from filesystem session files."""
    provider_name = Provider.from_string(provider_name)
    if not session_dir.exists():
        return

    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(jsonl_files) > max_sessions:
        step = max(1, len(jsonl_files) // max_sessions)
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            records.append(parsed)
        except OSError:
            continue

        if not records:
            continue

        yield from extract_schema_units_from_payload(
            records,
            provider_name=provider_name,
            source_path=path,
            raw_id=path.stem,
            observed_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            config=config,
            max_samples=max_samples,
        )


def iter_schema_units(
    provider_name: str | Provider,
    *,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> Any:
    """Yield schema units for a provider from DB, with session fallback."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()

    config = resolve_provider_config(provider_name)
    yielded_any = False

    if config.db_provider_name and db_path.exists():
        for unit in _iter_schema_units_from_db(
            provider_name,
            db_path=db_path,
            config=config,
            max_samples=max_samples,
        ):
            yielded_any = True
            yield unit

    if yielded_any or config.session_dir is None:
        return

    yield from _iter_schema_units_from_sessions(
        provider_name,
        config.session_dir,
        max_sessions=config.max_sessions,
        config=config,
        max_samples=max_samples,
    )


def _iter_samples_from_db(
    provider_name: Provider,
    *,
    db_path: Path,
    config: ProviderConfig,
    with_conv_ids: bool = False,
) -> Any:
    """Yield individual sample dicts from the database."""
    provider_name = Provider.from_string(provider_name)
    for unit in _iter_schema_units_from_db(provider_name, db_path=db_path, config=config):
        for sample in unit.schema_samples:
            if with_conv_ids:
                yield sample, unit.conversation_id
            else:
                yield sample


def _iter_samples_from_sessions(
    session_dir: Path,
    *,
    max_sessions: int | None,
) -> Any:
    """Yield individual sample dicts from session files."""
    if not session_dir.exists():
        return

    jsonl_files = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(jsonl_files) > max_sessions:
        step = max(1, len(jsonl_files) // max_sessions)
        jsonl_files = jsonl_files[::step][:max_sessions]

    for path in jsonl_files:
        try:
            with path.open(encoding="utf-8") as handle:
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
    provider_name: str | Provider,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw samples from the polylogue database."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return []

    config = resolve_provider_config(provider_name)
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
    provider_name: str | Provider,
    db_path: Path | None = None,
) -> int:
    """Get total message count for a provider in the database."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = default_db_path()
    if not db_path.exists():
        return 0

    config = resolve_provider_config(provider_name)
    provider_tokens = [str(provider_name)]
    if config.db_provider_name and str(config.db_provider_name) not in provider_tokens:
        provider_tokens.append(str(config.db_provider_name))

    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in provider_tokens)
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name IN ({placeholders})
            """,
            provider_tokens,
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


__all__ = [
    "_iter_samples_from_db",
    "_iter_samples_from_sessions",
    "_iter_schema_units_from_db",
    "_iter_schema_units_from_sessions",
    "_sample_provider_where_clause",
    "get_sample_count_from_db",
    "iter_schema_units",
    "load_samples_from_db",
    "load_samples_from_sessions",
]
