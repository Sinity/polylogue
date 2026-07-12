"""Idempotent price-catalog seed for the archive index tier.

Loads the in-memory curated PRICING dict (polylogue.archive.semantic.pricing)
into the ``price_catalogs`` / ``model_prices`` SQLite tables so that cost
computations written to ``session_model_usage`` have a durable FK target and
the DB-backed rates match the in-memory estimates exactly.

Design constraints:
- The in-memory PRICING dict is the authoritative source.  This seeder
  populates the DB FROM it; rates are never duplicated by hand here.
- A changed catalog hash creates a versioned catalog row, so existing usage
  records retain the rates under which they were priced.
- Re-seeding an unchanged catalog is idempotent.

Writer module: index.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time


def _catalog_id() -> str:
    from polylogue.archive.semantic.pricing import CATALOG_EFFECTIVE_DATE, CATALOG_PROVENANCE

    return f"{CATALOG_PROVENANCE}-{CATALOG_EFFECTIVE_DATE}"


def _catalog_hash() -> str:
    """Stable identity of the current PRICING dict."""
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


def _catalog_id_for_hash(conn: sqlite3.Connection, catalog_hash: str) -> str:
    """Return the persisted version identity for the current catalog hash."""
    from polylogue.archive.semantic.pricing import CATALOG_PROVENANCE

    effective_at = _effective_at_ms()
    matching = conn.execute(
        """
        SELECT catalog_id
        FROM price_catalogs
        WHERE catalog_hash = ?
          AND source_name = ?
          AND effective_at_ms IS ?
        ORDER BY catalog_id
        LIMIT 1
        """,
        (catalog_hash, CATALOG_PROVENANCE, effective_at),
    ).fetchone()
    if matching is not None:
        return str(matching[0])

    base_catalog_id = _catalog_id()
    existing_base = conn.execute(
        "SELECT catalog_hash FROM price_catalogs WHERE catalog_id = ?",
        (base_catalog_id,),
    ).fetchone()
    if existing_base is None:
        return base_catalog_id
    return f"{base_catalog_id}-{catalog_hash}"


def active_price_catalog_id(conn: sqlite3.Connection) -> str | None:
    """Find the persisted catalog whose hash matches the active PRICING dict."""
    from polylogue.archive.semantic.pricing import CATALOG_PROVENANCE

    row = conn.execute(
        """
        SELECT catalog_id
        FROM price_catalogs
        WHERE catalog_hash = ?
          AND source_name = ?
          AND effective_at_ms IS ?
        ORDER BY catalog_id
        LIMIT 1
        """,
        (_catalog_hash(), CATALOG_PROVENANCE, _effective_at_ms()),
    ).fetchone()
    return str(row[0]) if row is not None else None


def seed_price_catalog(conn: sqlite3.Connection) -> str:
    """Seed price_catalogs + model_prices from the curated PRICING dict.

    A changed hash gets a new catalog ID and its own immutable price rows;
    an unchanged hash reuses its existing catalog ID.

    Returns the catalog_id that was (or already was) seeded.
    """
    from polylogue.archive.semantic.pricing import CATALOG_PROVENANCE, PRICING

    catalog_hash = _catalog_hash()
    catalog_id = _catalog_id_for_hash(conn, catalog_hash)
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
    "active_price_catalog_id",
    "seed_price_catalog",
]
