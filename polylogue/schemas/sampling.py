"""Tooling-side sample loading from the polylogue database and session files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.lib.raw_payload import build_raw_payload_envelope, collect_limited_samples
from polylogue.paths import db_path as archive_db_path
from polylogue.schemas.observation import resolve_provider_config
from polylogue.schemas.sampling_db import (
    _iter_samples_from_db,
    _iter_schema_units_from_db,
    _sample_provider_where_clause,
    get_sample_count_from_db,
)
from polylogue.schemas.sampling_sessions import (
    _iter_samples_from_sessions,
    _iter_schema_units_from_sessions,
)
from polylogue.types import Provider


def iter_schema_units(
    provider_name: str | Provider,
    *,
    db_path: Path | None = None,
    max_samples: int | None = None,
    full_corpus: bool = False,
):
    """Yield schema units for a provider from DB, with session fallback."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = archive_db_path()

    config = resolve_provider_config(provider_name)
    yielded_any = False

    if config.db_provider_name and db_path.exists():
        for unit in _iter_schema_units_from_db(
            provider_name,
            db_path=db_path,
            config=config,
            max_samples=max_samples,
            full_corpus=full_corpus,
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


def load_samples_from_db(
    provider_name: str | Provider,
    db_path: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw samples from the polylogue database."""
    provider_name = Provider.from_string(provider_name)
    if db_path is None:
        db_path = archive_db_path()
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


__all__ = [
    "_iter_samples_from_db",
    "_iter_samples_from_sessions",
    "_iter_schema_units_from_db",
    "_iter_schema_units_from_sessions",
    "_sample_provider_where_clause",
    "build_raw_payload_envelope",
    "get_sample_count_from_db",
    "iter_schema_units",
    "load_samples_from_db",
    "load_samples_from_sessions",
]
