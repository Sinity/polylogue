"""Index-tier same-version benign-DDL convergence registry.

Operator ruling (polylogue-jc1b, 2026-07-19): benign DDL changes -- dropping
zero-consumer tables, adding indexes/tables no current code reads -- must not
force an ``INDEX_SCHEMA_VERSION`` bump and the full rebuild that follows from
one. The version guard exists to protect semantic code<->index compatibility
(consumer-visible columns/behavior, reparse-required content); changes outside
any consumer contract are not version events.

The ops tier already proved this pattern (``bootstrap.py``'s
``_ensure_ops_*_columns`` helpers: idempotent additive DDL re-applied on every
same-version open, zero bumps ever). This module extends the regime to the
index tier with a narrow, explicitly registered statement list rather than a
blanket DDL re-exec, so each entry can be policy-checked
(``devtools lab policy schema-versioning``) for the three properties that make
it safe to apply on every open of an already-populated, same-version archive:

- **Idempotent**: ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT
  EXISTS`` / ``DROP TABLE IF EXISTS`` only. A second application must be a
  no-op.
- **Data-non-transforming**: no ``INSERT`` / ``UPDATE`` / ``DELETE`` / row
  rewriting. This registry only ever adds or removes empty-of-consequence
  schema objects, never touches row content.
- **Bidirectionally safe at the same version**: older same-version code must
  have zero consumers of anything a registered entry drops, and newer code
  must not depend on anything a registered entry has not yet created (so an
  archive can be opened by either an older or newer same-``INDEX_SCHEMA_VERSION``
  build without breaking).

Applied from both the fresh-init path (harmless no-op: a fresh archive never
had the dropped tables, and any additive entry lands identically to canonical
DDL) and the same-version reopen path (`storage/sqlite/archive_tiers/bootstrap.py`
and its `schema.py` sync/async twins), so fresh-init and converged-live
archives agree schema-wise by construction rather than by an ad hoc parity
check.

First application: polylogue-v2mg drops ``model_prices`` and
``session_reported_costs`` (zero-consumer tables -- see each entry's
``reason``).

Second application: polylogue-resk drops ``price_catalogs``. v2mg's original
justification for keeping it ("session_model_usage.priced_with FK,
active_price_catalog_id" are genuine reads) measured false in all three
particulars on a 2026-07-31 audit -- see that entry's ``reason``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import aiosqlite


@dataclass(frozen=True, slots=True)
class BenignDDLEntry:
    """One registered idempotent, data-non-transforming DDL statement.

    ``devtools lab policy schema-versioning`` validates every registry entry's
    ``sql`` against the allowed idempotent-DDL shapes; see
    ``devtools/verify_schema_upgrade_lane.py``.
    """

    name: str
    sql: str
    reason: str


INDEX_BENIGN_DDL_REGISTRY: tuple[BenignDDLEntry, ...] = (
    BenignDDLEntry(
        name="drop_model_prices",
        sql="DROP TABLE IF EXISTS model_prices",
        reason=(
            "polylogue-v2mg: zero-consumer table. Written by "
            "pricing_seed.seed_price_catalog() but never read back -- cost "
            "computation (write.py._aggregate_message_tokens_into_model_usage) "
            "resolves per-model rates from the in-process "
            "polylogue.archive.semantic.pricing.PRICING catalog, never "
            "round-tripping through this DB-backed mirror."
        ),
    ),
    BenignDDLEntry(
        name="drop_session_reported_costs",
        sql="DROP TABLE IF EXISTS session_reported_costs",
        reason=(
            "polylogue-v2mg: zero-consumer table. Written by write.py's "
            "session-cost writer (an INSERT OR REPLACE keyed on "
            "session.reported_cost_usd) and cleared by "
            "_clear_session_projection_rows on full-replace, but never read "
            "by any query surface, insight, or CLI/MCP payload."
        ),
    ),
    BenignDDLEntry(
        name="drop_price_catalogs",
        sql="DROP TABLE IF EXISTS price_catalogs",
        reason=(
            "polylogue-resk: v2mg kept price_catalogs on the claim it is "
            "genuinely read via session_model_usage.priced_with FK / "
            "active_price_catalog_id. A 2026-07-31 audit measured that "
            "claim false in all three particulars: priced_with and "
            "priced_at_ms had zero production SELECTs (every reference was "
            "an INSERT/UPDATE in write.py, the DDL FK line, or prose -- the "
            "only SELECTs were in "
            "tests/unit/storage/test_pricing_chain_roundtrip.py); "
            "active_price_catalog_id's only non-test caller was "
            "pricing_seed.py itself (a seeded-already check, not a genuine "
            "read); live data confirmed the column carried no information "
            "(18,655 session_model_usage rows, 10,222 with priced_with set, "
            "exactly 1 distinct value). Actual pricing resolution is "
            "in-process via polylogue.archive.semantic.pricing.PRICING -- "
            "the same defect the v2mg bead was closing for model_prices. "
            "The referencing columns (session_model_usage.priced_with/"
            "priced_at_ms) and pricing_seed.py's seed_price_catalog()/"
            "active_price_catalog_id() are removed in the same change "
            "(INDEX_SCHEMA_VERSION v61's CONSTRAINT_ONLY delta). Does NOT "
            "cover session_profiles.priced_with/priced_at_ms -- a different, "
            "unrelated pair of columns (no FK to price_catalogs) with a "
            "real production reader (daemon/http.py's session-insights "
            "panel) -- those are kept."
        ),
    ),
)


def apply_index_benign_ddl_convergence(conn: sqlite3.Connection) -> None:
    """Apply every registered benign DDL statement (idempotent, no-op-safe).

    Safe to call on a brand-new fresh-init connection (nothing to converge)
    and on every same-version reopen of an existing archive alike.
    """
    for entry in INDEX_BENIGN_DDL_REGISTRY:
        conn.execute(entry.sql)


async def apply_index_benign_ddl_convergence_async(conn: aiosqlite.Connection) -> None:
    """Async twin of :func:`apply_index_benign_ddl_convergence`."""
    for entry in INDEX_BENIGN_DDL_REGISTRY:
        await conn.execute(entry.sql)


__all__ = [
    "INDEX_BENIGN_DDL_REGISTRY",
    "BenignDDLEntry",
    "apply_index_benign_ddl_convergence",
    "apply_index_benign_ddl_convergence_async",
]
