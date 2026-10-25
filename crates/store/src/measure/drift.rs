//! Format drift: what each format's parser did not know.

use crate::{Reader, Result};
use serde::Serialize;

/// Something a format's parser did not know, across every input of that
/// format. New record types and fields appear here first.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Drift {
    pub format: String,
    /// `record:<type>`, `field:<json path>` or `undecodable:<reason>`.
    pub what: String,
    pub count: u64,
    pub inputs: u64,
    /// Kept only as a count by design: known, not drift.
    pub by_design: bool,
}

/// Every unrecognized item, most frequent first; items kept as counts by
/// design come after the rest.
pub fn drift(reader: &Reader<'_>) -> Result<Vec<Drift>> {
    let mut stmt = reader.conn()?.prepare(
        "SELECT format, what, sum(count), count(DISTINCT input_pk), max(counted_only)
         FROM unrecognized GROUP BY format, what
         ORDER BY max(counted_only), sum(count) DESC, format, what",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(Drift {
            format: r.get(0)?,
            what: r.get(1)?,
            count: r.get::<_, i64>(2)? as u64,
            inputs: r.get::<_, i64>(3)? as u64,
            by_design: r.get::<_, i64>(4)? == 1,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}
