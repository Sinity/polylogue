//! API-equivalent message cost from recorded usage and the compiled price catalog.

use super::{Class, Derivation, Output, Version};
use crate::Result;
use polylogue_model::{Catalog, Usage};
use rusqlite::{Connection, Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "cost",
    class: Class::Row,
    products: &[],
    version: Version::Catalog(1),
    after: &[],
    dirtied_by: &[],
    compute: |conn, pk, inputs| compute(conn, pk, inputs.catalog).map(Output::Cost),
};

/// The session's messages whose cost changes, with their new cost.
fn compute(
    conn: &Connection,
    session_pk: i64,
    catalog: &Catalog,
) -> Result<Vec<(i64, Option<i64>)>> {
    let mut read = conn.prepare_cached(
        "SELECT pk, model, input_tokens, output_tokens, cache_read_tokens,
                cache_write_tokens, reasoning_tokens, cache_write_1h_tokens, service_tier,
                speed, cost_nusd
         FROM message WHERE session_pk = ?1",
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
            r.get::<_, Option<i64>>(10)?,
        ))
    })?;
    let mut changed = Vec::new();
    for row in rows {
        let (pk, model, usage, stored) = row?;
        let cost = catalog.cost_nusd(model.as_deref(), &usage);
        if cost != stored {
            changed.push((pk, cost));
        }
    }
    Ok(changed)
}

pub(super) fn apply(tx: &Transaction<'_>, costs: &[(i64, Option<i64>)]) -> Result<()> {
    let mut write = tx.prepare_cached("UPDATE message SET cost_nusd = ?2 WHERE pk = ?1")?;
    for (pk, cost) in costs {
        write.execute(params![pk, cost])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
