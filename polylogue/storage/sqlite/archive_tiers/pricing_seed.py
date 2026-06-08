"""Idempotent price-catalog seed for the archive index tier.

Loads the in-memory curated PRICING dict (polylogue.archive.semantic.pricing)
into the ``price_catalogs`` / ``model_prices`` SQLite tables so that cost
computations written to ``session_model_usage`` have a durable FK target and
the DB-backed rates match the in-memory estimates exactly.

Design constraints:
- The in-memory PRICING dict is the authoritative source.  This seeder
  populates the DB FROM it; rates are never duplicated by hand here.
- The catalog_id is a deterministic slug so re-seeding is idempotent.
- INSERT … ON CONFLICT DO NOTHING on both tables: re-running after a
  schema-version bump (wipe-and-reinit) or daemon restart is safe.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time


def _catalog_id() -> str:
    from polylogue.archive.semantic.pricing import CATALOG_EFFECTIVE_DATE, CATALOG_PROVENANCE

    return f"{CATALOG_PROVENANCE}-{CATALOG_EFFECTIVE_DATE}"


def _catalog_hash() -> str:
    """Stable hash of the current PRICING dict for change-detection."""
    from polylogue.archive.semantic.pricing import PRICING

    parts: list[str] = []
    for model_name in sorted(PRICING):
        p = PRICING[model_name]
        parts.append(
            f"{model_name}:{p.input_usd_per_1m}:{p.output_usd_per_1m}"
            f":{p.cache_read_usd_per_1m}:{p.cache_write_usd_per_1m}"
        )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _effective_at_ms() -> int | None:
    """Parse CATALOG_EFFECTIVE_DATE to epoch-ms, return None on failure."""
    from polylogue.archive.semantic.pricing import CATALOG_EFFECTIVE_DATE

    try:
        import datetime

        d = datetime.datetime.strptime(CATALOG_EFFECTIVE_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        return int(d.timestamp() * 1000)
    except ValueError:
        return None


def seed_price_catalog(conn: sqlite3.Connection) -> str:
    """Seed price_catalogs + model_prices from the curated PRICING dict.

    Idempotent: uses INSERT … ON CONFLICT DO NOTHING so calling this at
    every archive-init or daemon-start has no effect when the catalog row
    already exists.

    Returns the catalog_id that was (or already was) seeded.
    """
    from polylogue.archive.semantic.pricing import CATALOG_PROVENANCE, PRICING

    catalog_id = _catalog_id()
    catalog_hash = _catalog_hash()
    effective_at = _effective_at_ms()
    loaded_at = int(time.time() * 1000)

    conn.execute(
        """
        INSERT INTO price_catalogs (
            catalog_id, catalog_hash, source_name, effective_at_ms, loaded_at_ms
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(catalog_id) DO NOTHING
        """,
        (catalog_id, catalog_hash, CATALOG_PROVENANCE, effective_at, loaded_at),
    )

    for model_name, pricing in PRICING.items():
        conn.execute(
            """
            INSERT INTO model_prices (
                catalog_id, model_name, price_unit,
                input_cost_per_million, output_cost_per_million,
                cache_read_cost_per_million, cache_write_cost_per_million,
                effective_from_ms
            ) VALUES (?, ?, 'tokens', ?, ?, ?, ?, 0)
            ON CONFLICT(catalog_id, model_name, price_unit, effective_from_ms) DO NOTHING
            """,
            (
                catalog_id,
                model_name,
                pricing.input_usd_per_1m,
                pricing.output_usd_per_1m,
                pricing.cache_read_usd_per_1m,
                pricing.cache_write_usd_per_1m,
            ),
        )

    return catalog_id


__all__ = [
    "seed_price_catalog",
]
