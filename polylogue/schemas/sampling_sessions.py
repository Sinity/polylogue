"""Filesystem-backed session sampling for schema tooling."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.schemas.observation import ProviderConfig, extract_schema_units_from_payload
from polylogue.types import Provider


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


__all__ = [
    "_iter_samples_from_sessions",
    "_iter_schema_units_from_sessions",
]
