"""Private, disposable reduced-evidence cache for source schema inference."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
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
    metadata: JSONDocument


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
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self.path.chmod(0o600)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS source_evidence_contributions (
                cache_key TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                evidence_json TEXT NOT NULL,
                input_bytes INTEGER NOT NULL CHECK (input_bytes >= 0),
                record_count INTEGER NOT NULL CHECK (record_count >= 0),
                metadata_json TEXT NOT NULL DEFAULT '{}'
            ) STRICT
            """
        )
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(source_evidence_contributions)")}
        if "metadata_json" not in columns:
            self._connection.execute(
                "ALTER TABLE source_evidence_contributions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_evidence_recipes (
                fingerprint TEXT NOT NULL,
                phase TEXT NOT NULL,
                contract_json TEXT NOT NULL,
                PRIMARY KEY (fingerprint, phase)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS source_evidence_lookup (
                cache_key TEXT PRIMARY KEY REFERENCES source_evidence_contributions(cache_key),
                provider TEXT NOT NULL,
                source_context TEXT NOT NULL,
                revision_sha256 TEXT NOT NULL,
                phase TEXT NOT NULL,
                recipe_fingerprint TEXT NOT NULL,
                FOREIGN KEY (recipe_fingerprint, phase) REFERENCES source_evidence_recipes(fingerprint, phase)
            ) STRICT;
            CREATE INDEX IF NOT EXISTS source_evidence_lookup_by_source
                ON source_evidence_lookup(provider, source_context, revision_sha256, phase);
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
            SELECT evidence_json, input_bytes, record_count, metadata_json
            FROM source_evidence_contributions
            WHERE cache_key = ? AND schema_version = ?
            """,
            (cache_key, self._SCHEMA_VERSION),
        ).fetchone()
        if row is None:
            return None
        return self._decode(cache_key, row)

    @staticmethod
    def _decode(cache_key: str, row: tuple[object, ...]) -> CachedContribution:
        raw_evidence, input_bytes, record_count, raw_metadata = row
        if not (
            isinstance(raw_evidence, str)
            and isinstance(raw_metadata, str)
            and isinstance(input_bytes, int)
            and isinstance(record_count, int)
        ):
            raise SourceContributionCacheError("cached source evidence has invalid storage types")
        try:
            evidence = json.loads(raw_evidence)
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceContributionCacheError("cached source evidence is not valid JSON") from exc
        if not isinstance(evidence, dict) or not isinstance(metadata, dict):
            raise SourceContributionCacheError("cached source evidence must be a JSON object")
        return CachedContribution(cache_key, evidence, input_bytes, record_count, metadata)

    def register_recipe(self, fingerprint: str, phase: str, contract: JSONDocument) -> None:
        payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        prior = self._connection.execute(
            "SELECT contract_json FROM source_evidence_recipes WHERE fingerprint=? AND phase=?",
            (fingerprint, phase),
        ).fetchone()
        if prior is not None and prior[0] != payload:
            raise SourceContributionCacheError("a source recipe fingerprint has conflicting contracts")
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO source_evidence_recipes VALUES (?, ?, ?)", (fingerprint, phase, payload)
            )

    def recipes(self, phase: str) -> Iterator[tuple[str, JSONDocument]]:
        for fingerprint, payload in self._connection.execute(
            "SELECT fingerprint, contract_json FROM source_evidence_recipes WHERE phase=? ORDER BY fingerprint",
            (phase,),
        ):
            yield fingerprint, json.loads(payload)

    def iter_contributions(self) -> Iterator[CachedContribution]:
        for key, *row in self._connection.execute(
            "SELECT cache_key, evidence_json, input_bytes, record_count, metadata_json FROM source_evidence_contributions"
        ):
            yield self._decode(key, tuple(row))

    def index_contributions(self, entries: Iterable[tuple[str, str, str, str, str, str]]) -> None:
        """Index verified existing evidence without changing its recipe or payload."""
        with self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO source_evidence_lookup VALUES (?, ?, ?, ?, ?, ?)", entries
            )

    def find_contributions(
        self, provider: str, source_context: str, revision_sha256: str, phase: str
    ) -> Iterator[tuple[CachedContribution, JSONDocument]]:
        for key, evidence, byte_count, records, metadata, contract in self._connection.execute(
            """
            SELECT c.cache_key, c.evidence_json, c.input_bytes, c.record_count, c.metadata_json, r.contract_json
            FROM source_evidence_lookup l
            JOIN source_evidence_contributions c ON c.cache_key=l.cache_key
            JOIN source_evidence_recipes r ON r.fingerprint=l.recipe_fingerprint AND r.phase=l.phase
            WHERE l.provider=? AND l.source_context=? AND l.revision_sha256=? AND l.phase=?
              AND c.schema_version=?
            ORDER BY c.rowid DESC
            """,
            (provider, source_context, revision_sha256, phase, self._SCHEMA_VERSION),
        ):
            yield self._decode(key, (evidence, byte_count, records, metadata)), json.loads(contract)

    def put(self, contribution: CachedContribution) -> None:
        payload = json.dumps(contribution.evidence, sort_keys=True, separators=(",", ":"))
        metadata = json.dumps(contribution.metadata, sort_keys=True, separators=(",", ":"))
        with self._connection:
            self._connection.execute(
                """
            INSERT INTO source_evidence_contributions
                (cache_key, schema_version, evidence_json, input_bytes, record_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                schema_version = excluded.schema_version,
                evidence_json = excluded.evidence_json,
                input_bytes = excluded.input_bytes,
                record_count = excluded.record_count,
                metadata_json = excluded.metadata_json
            """,
                (
                    contribution.cache_key,
                    self._SCHEMA_VERSION,
                    payload,
                    contribution.input_bytes,
                    contribution.record_count,
                    metadata,
                ),
            )
            address = contribution.metadata.get("address")
            if isinstance(address, dict):
                values = tuple(
                    address.get(name) for name in ("provider", "source_context", "revision", "phase", "recipe")
                )
                if not all(isinstance(value, str) for value in values):
                    raise SourceContributionCacheError("cached contribution address is invalid")
                self._connection.execute(
                    "INSERT OR REPLACE INTO source_evidence_lookup VALUES (?, ?, ?, ?, ?, ?)",
                    (contribution.cache_key, *values),
                )


__all__ = [
    "CachedContribution",
    "SourceContributionCache",
    "SourceContributionCacheError",
]
