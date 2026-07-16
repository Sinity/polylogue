---
created: "2026-06-28T00:00:00Z"
purpose: "Design for in-place archive re-price (no source re-read) when the price catalog changes"
status: "complete"
project: "polylogue"
---

# Re-price-in-place: design + perf estimate

## TL;DR
- Cost is **fully recomputable from already-materialized token columns** — no source
  re-read, no message re-hydration needed. The four priced lanes
  (input / output / cache_read / cache_write) are all persisted per
  `(session_id, model_name)` in `session_model_usage`.
- The "scary" table — `session_provider_usage_events` (~2.97M rows) — **has no cost
  column and is NOT touched by a reprice.** It's raw token-event evidence only.
- Only **two** tables carry derived `cost_usd` and must be UPDATEd:
  `session_model_usage` (~15.7K rows) and `session_profiles` (~16.4K rows).
- Total work ≈ **~32K small REAL updates, single transaction, sub-second, a few MB WAL.**
  No chunking required (well under the 40 MiB autocheckpoint / 160 MiB journal cap).
- Recommend **Python-driven, set-based per-model UPDATEs** (not pure-SQL JOIN, not
  row-by-row), because the model-name→catalog-key match needs
  `_normalize_model`'s prefix logic, which is not SQL-expressible.

---

## 1. How cost is computed and stored today

Two distinct modules:

- `polylogue/archive/semantic/pricing.py` — the catalog + math.
  - `PRICING: dict[str, ModelPricing]` = vendored LiteLLM map **+** `_CURATED_PRICING`
    overrides. Keys are **normalized** model ids (e.g. `claude-opus-4-8`).
  - `estimate_cost(input, output, model, cache_read, cache_write)` →
    `_normalize_model(model)` lookup, then
    `tokens * usd_per_1m / 1_000_000` summed over the **four lanes only**
    (`_cost_components`). There is **no separate reasoning lane** in pricing —
    reasoning tokens are folded into `output` per provider semantics.
  - `_normalize_model` does: casefold, strip `openai/|anthropic/|google/|gemini/`
    prefixes, strip trailing `-YYYY-MM-DD` / `-YYYYMMDD` date snapshots, exact
    match, then **longest-prefix match** against catalog keys. This prefix step is
    why a pure-SQL `JOIN ON model_name = model_prices.model_name` is unsafe.
- `polylogue/archive/semantic/cost_compute.py` — `compute_session_cost(session)`
  walks `session.messages` (a **hydrated** session) to build per-model breakdowns,
  calls `estimate_cost`. This is the **source/ingest path** the operator wants to
  avoid for reprice — it requires re-reading messages.

### Stored columns (index.db DDL, `storage/sqlite/archive_tiers/index.py`)

`session_model_usage`  (PK `(session_id, model_name)`, ~15.7K rows) — **canonical priced layer**:
| column | role |
|---|---|
| `input_tokens, output_tokens, cache_read_tokens, cache_write_tokens` | the 4 priced lanes (persisted ✓) |
| `model_name` | **RAW provider name** as written (NOT normalized) — see write.py `_aggregate_message_tokens_into_model_usage` / `_write_reported_costs` |
| `cost_usd` | derived; recomputed by reprice |
| `cost_credits` | subscription credits (optional) |
| `cost_provenance` | `'origin_reported' | 'priced' | 'estimated'` (NULL ok) |
| `priced_with` | FK → `price_catalogs(catalog_id)` (which catalog priced this) |
| `priced_at_ms` | when priced |
| `message_count` | informational |

`session_profiles` (PK `session_id`, ~16.4K rows) — **aggregate**:
- `cost_usd`, `total_cost_usd`, `cost_credits`, `cost_is_estimated`,
  `cost_provenance`, `priced_with`, `priced_at_ms`.
- Written via `upsert_session_profile_costs` (write.py). The session aggregate is
  the SUM of its `session_model_usage.cost_usd`.

`session_provider_usage_events` (PK `(session_id, position)`, **~2.97M rows**):
- Holds `last_*` and `total_*` token lanes incl. `total_reasoning_output_tokens`,
  `total_cached_input_tokens`, `total_cache_write_tokens`.
- **Has NO cost column at all.** Pure token evidence → reprice ignores it entirely.

`session_reported_costs` (PK `(session_id, cost_kind, source)`):
- Provider-`origin_reported` exact USD/credits. **Authoritative — never repriced.**

`price_catalogs` / `model_prices` — the **price table already lives in the DB**:
- `model_prices(catalog_id, model_name, price_unit='tokens',
  input_cost_per_million, output_cost_per_million,
  cache_read_cost_per_million, cache_write_cost_per_million, ...)`.
- Seeded idempotently from `PRICING` by
  `storage/sqlite/archive_tiers/pricing_seed.py:seed_price_catalog`
  under a deterministic `catalog_id = f"{CATALOG_PROVENANCE}-{CATALOG_EFFECTIVE_DATE}"`.
- model_prices keys are **normalized** (same as PRICING). So price injection needs
  no temp table and no CASE — it's a real, queryable table.

## 2. Minimal UPDATE set + recompute-from-tokens confirmation

- **Recomputable purely from stored tokens + catalog?** YES. cost is a pure
  function of `(input, output, cache_read, cache_write)` × per-million rates, all
  four lanes persisted in `session_model_usage`. No message re-read needed.
- **Reasoning lane:** persisted in `session_provider_usage_events` only, but is
  **not a priced lane** (no reasoning rate exists; folded into output). So its
  absence from `session_model_usage` is correct and irrelevant to reprice.
- **Tables to UPDATE:**
  1. `session_model_usage` — `cost_usd` (+ `cost_credits`, `priced_with`,
     `priced_at_ms`) for rows with `cost_provenance IN ('priced','estimated')`.
     **Skip `'origin_reported'` rows** (provider-exact; their cost_usd is NULL/given).
  2. `session_profiles` — re-aggregate `cost_usd`/`total_cost_usd`/`cost_credits`
     from the updated `session_model_usage`, restamp `priced_with`/`priced_at_ms`.
- **NOT touched:** `session_provider_usage_events` (no cost col),
  `session_reported_costs` (authoritative), `messages` (source — that's the point).

## 3. The in-place reprice (recommended approach)

**Driver: Python, set-based per distinct model. Single transaction.**

Rationale for Python over pure-SQL JOIN: `session_model_usage.model_name` is the
**raw** provider string; matching it to a catalog rate requires
`_normalize_model`'s casefold/prefix/date-strip logic, which SQLite cannot
express. The distinct model count is small (tens), so we resolve rates in Python
once and issue a handful of **set-based** UPDATEs (each covering all rows for one
model), never row-by-row.

```python
def reprice_in_place(conn):                      # one sqlite3 connection
    seed_price_catalog(conn)                     # idempotent: ensure new catalog_id rows exist
    catalog_id = _catalog_id()                   # new f"{PROVENANCE}-{EFFECTIVE_DATE}"
    now_ms = int(time.time() * 1000)

    models = [r[0] for r in conn.execute(
        "SELECT DISTINCT model_name FROM session_model_usage "
        "WHERE cost_provenance IN ('priced','estimated')")]

    with conn:                                   # ATOMIC: single transaction
        for raw in models:
            norm = _normalize_model(raw)
            p = PRICING.get(norm)
            if p is None:
                # unknown model: clear cost (no fabrication), matches ingest behavior
                conn.execute(
                    "UPDATE session_model_usage SET cost_usd=NULL, priced_with=NULL, "
                    "priced_at_ms=NULL WHERE model_name=? AND cost_provenance IN ('priced','estimated')",
                    (raw,))
                continue
            ir, orr = p.input_usd_per_1m, p.output_usd_per_1m
            crr, cwr = p.cache_read_usd_per_1m, p.cache_write_usd_per_1m
            conn.execute(
                """
                UPDATE session_model_usage SET
                  cost_usd = round(
                      input_tokens*?/1e6 + output_tokens*?/1e6
                    + cache_read_tokens*?/1e6 + cache_write_tokens*?/1e6, 6),
                  priced_with = ?, priced_at_ms = ?
                WHERE model_name = ?
                  AND cost_provenance IN ('priced','estimated')
                  AND (input_tokens+output_tokens+cache_read_tokens+cache_write_tokens) > 0
                """,
                (ir, orr, crr, cwr, catalog_id, now_ms, raw))

        # re-aggregate session_profiles from the refreshed per-model rows
        conn.execute(
            """
            UPDATE session_profiles AS sp SET
              total_cost_usd = COALESCE(agg.s, sp.total_cost_usd),
              cost_usd       = COALESCE(agg.s, sp.cost_usd),
              priced_with    = ?, priced_at_ms = ?
            FROM (SELECT session_id, SUM(cost_usd) AS s
                  FROM session_model_usage
                  WHERE cost_usd IS NOT NULL
                  GROUP BY session_id) AS agg
            WHERE sp.session_id = agg.session_id
              AND sp.cost_provenance IN ('priced','estimated','mixed')
            """,
            (catalog_id, now_ms))
```

Notes:
- Keep arithmetic identical to `estimate_cost` (sum of four lanes / 1e6, `round(...,6)`)
  so in-place result == a from-source rebuild bit-for-bit.
- `cost_credits` / subscription-equivalent reprice follows the same shape using
  `subscription_pricing.compute_credit_cost`; same per-model loop.
- Wrap in one `with conn:` so the archive is never half-repriced.

### Alternatives rejected
- **Pure-SQL `UPDATE ... FROM model_prices` JOIN on model_name** — breaks on dated/
  prefixed raw names (`gpt-4o-2024-08-06`, `anthropic/claude-...`) because the
  catalog is keyed by normalized ids. Would require either a `normalized_model`
  column on `session_model_usage` (schema bump → violates the no-in-place-upgrade
  policy and isn't needed) or pre-expanding `model_prices` with every raw variant.
  Not worth it at this scale.
- **CASE expression / temp price table** — unnecessary; `model_prices` is already a
  real table and the Python loop already has the rates in hand.
- **Row-by-row UPDATE** — 15.7K statements; pointless when one set-based UPDATE per
  distinct model (tens of statements) covers everything.

## 4. Cost / perf estimate

| metric | value |
|---|---|
| `session_model_usage` rows updated | ≤ 15,739 (only `priced`/`estimated`) |
| `session_profiles` rows updated | current baseline ≤ 2,390; historical pre-dedup ceiling was 16,410 |
| `session_provider_usage_events` (2.97M) | **0 rows touched** |
| UPDATE statements issued | ~tens (one per distinct model) + 1 aggregate |
| WAL growth | a few MB (≈32K dirtied small REAL rows + index leaf pages) |
| vs `WAL_AUTOCHECKPOINT_PAGES`=10000 (40 MiB) | well under → 0 or 1 passive checkpoint |
| vs `journal_size_limit` 160 MiB | never approached |
| RAM | negligible (SQLite streams; page cache single-digit MB) |
| IO | WAL write + one TRUNCATE/passive checkpoint, single-digit MB |
| wall time | sub-second to low single-digit seconds |

**Chunking: not needed.** The dataset that would justify chunking
(`session_provider_usage_events`, 2.97M) is never written. A single transaction is
both safe (atomic reprice) and cheap. If a future schema ever adds a per-row cost
to the 2.97M-row table, revisit with batched UPDATEs of ~50–100K rows/commit to
keep WAL under the 40 MiB autocheckpoint band; today that's moot.

**Driver: Python, not pure-SQL** — required for `_normalize_model`, and it keeps
`PRICING` (in-memory) as the single source of truth while `seed_price_catalog`
mirrors it into `price_catalogs`/`model_prices` for the FK/provenance stamp.

## 5. Open items / caveats for the implementer
- Confirm `session_profiles.cost_provenance` value set used at ingest
  (`'mixed'` appears in `compute_session_cost`); include all non-`origin_reported`
  states in the aggregate WHERE so estimated sessions get refreshed.
- Decide whether a price change should bump `CATALOG_EFFECTIVE_DATE` (new
  `catalog_id`) — recommended, so `priced_with` distinguishes pre/post reprice and
  `seed_price_catalog`'s ON-CONFLICT-DO-NOTHING inserts the new rate rows.
- Provider `origin_reported` costs are deliberately untouched — verify no
  `session_model_usage` row mixes provenance per `(session,model)` (PK guarantees one row).
