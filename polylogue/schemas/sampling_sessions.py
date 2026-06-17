"""Filesystem-backed session sampling for schema tooling."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, json_document, loads
from polylogue.schemas.observation import ProviderConfig, extract_schema_units_from_payload
from polylogue.schemas.observation_models import SchemaUnit

_SESSION_SAMPLE_SUFFIXES = (".jsonl", ".json")


def _iter_session_json_documents(session_dir: Path, *, max_sessions: int | None) -> Iterator[tuple[Path, JSONDocument]]:
    if not session_dir.exists():
        return

    session_files = sorted(
        (path for path in session_dir.rglob("*") if path.is_file() and path.suffix.lower() in _SESSION_SAMPLE_SUFFIXES),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if max_sessions and len(session_files) > max_sessions:
        step = max(1, len(session_files) // max_sessions)
        session_files = session_files[::step][:max_sessions]

    for path in session_files:
        try:
            if path.suffix.lower() == ".json":
                with contextlib.suppress(ValueError):
                    if record := json_document(loads(path.read_text(encoding="utf-8"))):
                        yield path, record
                continue
            with path.open(encoding="utf-8") as handle:
                yield from _iter_jsonl_records(path, handle)
        except OSError:
            continue


def _iter_jsonl_records(path: Path, handle: Iterator[str]) -> Iterator[tuple[Path, JSONDocument]]:
    for line in handle:
        if not line.strip():
            continue
        with contextlib.suppress(ValueError):
            if record := json_document(loads(line)):
                yield path, record


def _iter_schema_units_from_sessions(
    source_name: Provider,
    session_dir: Path,
    *,
    max_sessions: int | None,
    config: ProviderConfig,
    max_samples: int | None = None,
) -> Iterator[SchemaUnit]:
    """Yield clusterable schema units from filesystem session files."""
    source_name = Provider.from_string(source_name)
    current_path: Path | None = None
    current_records: list[JSONDocument] = []
    for path, record in _iter_session_json_documents(session_dir, max_sessions=max_sessions):
        if current_path is not None and path != current_path:
            yield from extract_schema_units_from_payload(
                current_records,
                source_name=source_name,
                source_path=current_path,
                raw_id=current_path.stem,
                observed_at=datetime.fromtimestamp(current_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                config=config,
                max_samples=max_samples,
            )
            current_records = []
        current_path = path
        current_records.append(record)

    if current_path is not None and current_records:
        yield from extract_schema_units_from_payload(
            current_records,
            source_name=source_name,
            source_path=current_path,
            raw_id=current_path.stem,
            observed_at=datetime.fromtimestamp(current_path.stat().st_mtime, tz=timezone.utc).isoformat(),
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
