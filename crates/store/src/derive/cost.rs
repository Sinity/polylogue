//! API-equivalent message cost from retained usage and the compiled price catalog.

use super::{Derivation, Version};
use crate::Result;
use polylogue_model::{Catalog, Usage};
use rusqlite::{Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "cost",
    version: Version::Catalog(1),
    after: &[],
    run,
    finish: None,
};

fn run(tx: &Transaction<'_>, session_pk: i64, catalog: &Catalog) -> Result<()> {
    {
        let mut read = tx.prepare_cached(
            "SELECT pk, model, input_tokens, output_tokens, cache_read_tokens,
                    cache_write_tokens, reasoning_tokens, cache_write_1h_tokens, service_tier,
                    speed
             FROM message WHERE session_pk = ?1",
        )?;
        let mut write = tx.prepare_cached(
            "UPDATE message SET cost_nusd = ?2 WHERE pk = ?1 AND cost_nusd IS NOT ?2",
        )?;
        let rows = read.query_map([session_pk], |r| {
            let n = |i| r.get::<_, Option<i64>>(i).map(|v| v.map(|v| v as u64));
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?,
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
            write.execute(params![pk, catalog.cost_nusd(model.as_deref(), &usage)])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
