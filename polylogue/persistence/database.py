from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .. import db as db_module


@dataclass
class ConversationDatabase:
    """Thin convenience wrapper around the SQLite archive."""

    path: Optional[Path] = None

    def query(self, sql: str, params: Sequence[object] = ()) -> Iterable[dict]:
        with db_module.open_connection(self.resolve_path()) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
        return rows

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        with db_module.open_connection(self.resolve_path()) as conn:
            conn.execute(sql, params)
            conn.commit()

    def resolve_path(self) -> Optional[Path]:
        return self.path or db_module.default_db_path()
