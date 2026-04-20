"""Filesystem-backed session sampling for schema tooling."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from polylogue.lib.json import JSONDocument, json_document, loads
from polylogue.schemas.observation import ProviderConfig, extract_schema_units_from_payload
from polylogue.schemas.observation_models import SchemaUnit
from polylogue.types import Provider


def _iter_session_json_documents(session_dir: Path, *, max_sessions: int | None) -> Iterator[tuple[Path, JSONDocument]]:
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
                    with contextlib.suppress(ValueError):
                        if record := json_document(loads(line)):
                            yield path, record
        except OSError:
            continue


def _iter_schema_units_from_sessions(
    provider_name: Provider,
    session_dir: Path,
    *,
    max_sessions: int | None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> Iterator[SchemaUnit]:
    """Yield clusterable schema units from filesystem session files."""
    provider_name = Provider.from_string(provider_name)
    records_by_path: dict[Path, list[JSONDocument]] = {}
    for path, record in _iter_session_json_documents(session_dir, max_sessions=max_sessions):
        records_by_path.setdefault(path, []).append(record)

    for path, records in records_by_path.items():
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


def _iter_samples_from_sessions(
    session_dir: Path,
    *,
    max_sessions: int | None,
) -> Iterator[JSONDocument]:
    """Yield individual sample dicts from session files."""
    for _path, record in _iter_session_json_documents(session_dir, max_sessions=max_sessions):
        yield record


__all__ = [
    "_iter_samples_from_sessions",
    "_iter_schema_units_from_sessions",
]
