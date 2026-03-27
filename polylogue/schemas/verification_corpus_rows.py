"""Row iteration helpers for schema verification."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

from polylogue.lib.provider_identity import CORE_RUNTIME_PROVIDERS

from .verification_support import bounded_window


def verification_provider_clause(providers: list[str]) -> tuple[str, tuple[Any, ...]]:
    provider_placeholders = ",".join("?" for _ in providers)
    runtime_placeholders = ",".join("?" for _ in CORE_RUNTIME_PROVIDERS)
    clause = (
        f"payload_provider IN ({provider_placeholders}) "
        f"OR (payload_provider IS NULL AND provider_name IN ({provider_placeholders})) "
        f"OR (payload_provider IS NULL AND provider_name NOT IN ({runtime_placeholders}))"
    )
    params: tuple[Any, ...] = (*providers, *providers, *CORE_RUNTIME_PROVIDERS)
    return clause, params


def iter_verification_rows(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None,
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int, Iterator[sqlite3.Row]]:
    bounded_limit, bounded_offset = bounded_window(record_limit, record_offset)
    provider_where = ""
    where_params: tuple[Any, ...] = ()
    if providers:
        provider_where, where_params = verification_provider_clause(providers)

    def rows() -> Iterator[sqlite3.Row]:
        batch_size_limit = 50
        last_rowid = 0

        if bounded_offset > 0:
            offset_query = "SELECT rowid FROM raw_conversations "
            if provider_where:
                offset_query += f"WHERE {provider_where} "
            offset_query += "ORDER BY rowid LIMIT 1 OFFSET ?"
            row = conn.execute(offset_query, (*where_params, bounded_offset - 1)).fetchone()
            if row is None:
                return
            last_rowid = row[0]

        base_query = (
            "SELECT rowid, raw_id, provider_name, payload_provider, source_path, raw_content "
            "FROM raw_conversations "
        )
        records_fetched = 0
        while True:
            if bounded_limit is not None:
                remaining = bounded_limit - records_fetched
                if remaining <= 0:
                    break
                batch_size = min(batch_size_limit, remaining)
            else:
                batch_size = batch_size_limit

            if provider_where:
                query = base_query + f"WHERE rowid > ? AND ({provider_where}) ORDER BY rowid LIMIT ?"
                params: tuple[Any, ...] = (last_rowid, *where_params, batch_size)
            else:
                query = base_query + "WHERE rowid > ? ORDER BY rowid LIMIT ?"
                params = (last_rowid, batch_size)

            batch = conn.execute(query, params).fetchall()
            if not batch:
                break

            last_rowid = batch[-1]["rowid"]
            records_fetched += len(batch)
            for row in batch:
                yield row

    return bounded_limit, bounded_offset, rows()


def candidate_provider(row: sqlite3.Row) -> tuple[str, str | None]:
    raw_provider = str(row["provider_name"])
    stored_payload_provider = row["payload_provider"]
    return str(stored_payload_provider or raw_provider), stored_payload_provider


__all__ = ["candidate_provider", "iter_verification_rows", "verification_provider_clause"]
