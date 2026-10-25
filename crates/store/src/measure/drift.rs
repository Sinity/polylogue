//! Format drift: what each format's parser did not know.

use crate::{Reader, Result};
use serde::Serialize;

/// Something a format's parser did not know, across every input of that
/// format. New record types and fields appear here first.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Drift {
    pub format: String,
    /// `record:<type>`, `field:<json path>`, `user:<tag>`, `invalid_utf8`,
    /// `time_unparsed` or `undecodable`.
    pub what: String,
    pub count: u64,
    pub inputs: u64,
    /// When it was first and last seen, by record time.
    pub first_at: Option<i64>,
    pub last_at: Option<i64>,
    /// Kept only as a count by design: known, not drift.
    pub by_design: bool,
}

/// Every unrecognized item, newest first; items kept as counts by design
/// come after the rest.
pub fn drift(reader: &Reader<'_>) -> Result<Vec<Drift>> {
    let mut stmt = reader.conn()?.prepare(
        "SELECT format, what, sum(count), count(DISTINCT input_pk), min(first_at), max(last_at),
                max(counted_only)
         FROM unrecognized GROUP BY format, what
         ORDER BY max(counted_only), min(first_at) DESC NULLS LAST, format, what",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(Drift {
            format: r.get(0)?,
            what: r.get(1)?,
            count: r.get::<_, i64>(2)? as u64,
            inputs: r.get::<_, i64>(3)? as u64,
            first_at: r.get(4)?,
            last_at: r.get(5)?,
            by_design: r.get::<_, i64>(6)? == 1,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}
