//! Stored message costs follow the price catalog compiled into this build
//! (architecture 5.6: a derived value carries the version of what it was
//! computed from). The writer prices each message as it writes it; when the
//! archive was priced under another catalog version, every message is
//! priced again before the writer starts.

use crate::Result;
use polylogue_model::{Catalog, Usage};
use rusqlite::{Connection, OptionalExtension, params};

/// The `meta` key holding the catalog version the stored costs use.
const KEY: &str = "price_catalog";

/// Prices every message again when the stored costs were computed under a
/// catalog other than `catalog`. Returns how many messages were repriced.
pub fn reprice(index: &Connection, catalog: &Catalog) -> Result<u64> {
    let stored: Option<String> = index
        .query_row("SELECT value FROM meta WHERE key = ?1", [KEY], |r| r.get(0))
        .optional()?;
    if stored.as_deref() == Some(catalog.version.as_str()) {
        return Ok(0);
    }
    let tx = index.unchecked_transaction()?;
    let mut changed = 0;
    {
        let mut read = tx.prepare(
            "SELECT pk, model, input_tokens, output_tokens, cache_read_tokens,
                    cache_write_tokens, reasoning_tokens, cache_write_1h_tokens, service_tier,
                    speed
             FROM message WHERE model IS NOT NULL",
        )?;
        let mut write = tx.prepare("UPDATE message SET cost_nusd = ?2 WHERE pk = ?1")?;
        let rows = read.query_map([], |r| {
            let n = |i| r.get::<_, Option<i64>>(i).map(|v| v.map(|v| v as u64));
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                Usage {
                    input: n(2)?,
                    output: n(3)?,
                    cached_input: n(4)?,
                    cache_write: n(5)?,
                    reasoning: n(6)?,
                    cache_write_1h: n(7)?,
                    service_tier: r.get(8)?,
                    speed: r.get(9)?,
                },
            ))
        })?;
        for row in rows {
            let (pk, model, usage) = row?;
            write.execute(params![pk, catalog.cost_nusd(Some(&model), &usage)])?;
            changed += 1;
        }
    }
    tx.execute(
        "INSERT INTO meta(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![KEY, catalog.version],
    )?;
    tx.commit()?;
    Ok(changed)
}

/// The tool's own figure for what a response cost, in nano-USD.
pub fn reported_nusd(usd: Option<&serde_json::Number>) -> Option<i64> {
    let usd = usd?.as_f64()?;
    usd.is_finite().then(|| (usd * 1e9).round() as i64)
}
