"""Private, disposable reduced-evidence cache for source schema inference."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.json import JSONDocument


class SourceContributionCacheError(RuntimeError):
    """The private source-evidence cache cannot be read or written safely."""


@dataclass(frozen=True, slots=True)
class CachedContribution:
    """One recipe-bound reduced contribution.

    ``evidence`` is deliberately the field-evidence projection.  Raw source
    payloads, paths, logical source identifiers, and representative samples do
    not belong in this cache.
    """

    cache_key: str
    evidence: JSONDocument
    input_bytes: int
    record_count: int


class SourceContributionCache:
    """SQLite owner for immutable, reduced source contributions.

    The caller owns source inventory and reduction.  This class only provides
    atomic cache rows; it never becomes an acquisition or revision authority.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> SourceContributionCache:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS source_evidence_contributions (
                cache_key TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                evidence_json TEXT NOT NULL,
                input_bytes INTEGER NOT NULL CHECK (input_bytes >= 0),
                record_count INTEGER NOT NULL CHECK (record_count >= 0)
            ) STRICT
            """
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()

    def get(self, cache_key: str) -> CachedContribution | None:
        row = self._connection.execute(
            """
            SELECT evidence_json, input_bytes, record_count
            FROM source_evidence_contributions
            WHERE cache_key = ? AND schema_version = ?
            """,
            (cache_key, self._SCHEMA_VERSION),
        ).fetchone()
        if row is None:
            return None
        try:
            evidence = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceContributionCacheError("cached source evidence is not valid JSON") from exc
        if not isinstance(evidence, dict):
            raise SourceContributionCacheError("cached source evidence must be a JSON object")
        return CachedContribution(cache_key, evidence, int(row[1]), int(row[2]))

    def put(self, contribution: CachedContribution) -> None:
        payload = json.dumps(contribution.evidence, sort_keys=True, separators=(",", ":"))
        self._connection.execute(
            """
            INSERT INTO source_evidence_contributions
                (cache_key, schema_version, evidence_json, input_bytes, record_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                schema_version = excluded.schema_version,
                evidence_json = excluded.evidence_json,
                input_bytes = excluded.input_bytes,
                record_count = excluded.record_count
            """,
            (
                contribution.cache_key,
                self._SCHEMA_VERSION,
                payload,
                contribution.input_bytes,
                contribution.record_count,
            ),
        )


__all__ = [
    "CachedContribution",
    "SourceContributionCache",
    "SourceContributionCacheError",
]
