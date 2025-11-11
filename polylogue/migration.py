"""Utilities for migrating legacy JSON caches into SQLite."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .paths import STATE_HOME
from .persistence.database import ConversationDatabase
from .persistence.state import ConversationStateRepository
from .db import open_connection
from .util import add_run


@dataclass
class LegacyMigrationReport:
    conversations_migrated: int = 0
    runs_migrated: int = 0
    state_source: Optional[Path] = None
    runs_source: Optional[Path] = None
    dry_run: bool = False
    runs_skipped: bool = False
    missing_state: bool = False
    missing_runs: bool = False
    errors: List[str] = field(default_factory=list)


def perform_legacy_migration(
    *,
    state_path: Optional[Path] = None,
    runs_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    dry_run: bool = False,
    force_runs: bool = False,
) -> LegacyMigrationReport:
    """Migrate legacy state.json and runs.json files into SQLite."""

    state_path = (state_path or (STATE_HOME / "state.json")).expanduser()
    runs_path = (runs_path or (STATE_HOME / "runs.json")).expanduser()
    report = LegacyMigrationReport(
        state_source=state_path if state_path.exists() else None,
        runs_source=runs_path if runs_path.exists() else None,
        dry_run=dry_run,
    )

    database = ConversationDatabase(path=db_path)
    state_repo = ConversationStateRepository(database=database)

    if not state_path.exists():
        report.missing_state = True
    else:
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupt files
            report.errors.append(f"Failed to read {state_path}: {exc}")
        else:
            conversations = data.get("conversations")
            if isinstance(conversations, dict):
                for provider, provider_map in conversations.items():
                    if not isinstance(provider_map, dict):
                        continue
                    for conversation_id, payload in provider_map.items():
                        if not isinstance(payload, dict):
                            continue
                        report.conversations_migrated += 1
                        if dry_run:
                            continue
                        state_repo.upsert(provider, conversation_id, dict(payload))

    if not runs_path.exists():
        report.missing_runs = True
        runs_records: List[dict] = []
    else:
        try:
            records = json.loads(runs_path.read_text(encoding="utf-8"))
            runs_records = records if isinstance(records, list) else []
        except Exception as exc:  # pragma: no cover - corrupt files
            report.errors.append(f"Failed to read {runs_path}: {exc}")
            runs_records = []

    if runs_records:
        existing_runs = 0
        with open_connection(database.resolve_path()) as conn:
            existing_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            if existing_runs and force_runs and not dry_run:
                conn.execute("DELETE FROM runs")
                conn.commit()
        if existing_runs and not force_runs:
            report.runs_skipped = True
        else:
            for entry in runs_records:
                if not isinstance(entry, dict):
                    continue
                report.runs_migrated += 1
                if dry_run:
                    continue
                add_run(dict(entry))

    return report
