//! Functions over stored source times, without zone I/O: the latest of a
//! set of times, a time's quality and precision, and the order relations.
use crate::query::fields::TimeAttribute;
use polylogue_model::Precision;
use rusqlite::Connection;

use super::FLAGS;

type Failure = Box<dyn std::error::Error + Send + Sync + 'static>;

mod order;
mod source_time;
use source_time::read as source_time;

pub(super) fn register(conn: &Connection) -> rusqlite::Result<()> {
    conn.create_aggregate_function("pl_time_latest", 1, FLAGS, source_time::Latest)?;
    order::register(conn)?;
    conn.create_scalar_function("pl_time_attribute", 3, FLAGS, |ctx| {
        let bytes: Option<Vec<u8>> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let attribute: String = ctx.get(2)?;
        let attribute = TimeAttribute::from_name(&attribute)
            .ok_or_else(|| rusqlite::Error::UserFunctionError("unknown time attribute".into()))?;
        let time =
            source_time(bytes.as_deref(), &storage).map_err(rusqlite::Error::UserFunctionError)?;
        // No related tuple is distinct from an explicitly unknown quality.
        if bytes.is_none() {
            return Ok(None);
        }
        Ok(match attribute {
            TimeAttribute::Quality => time.quality.map(|quality| quality.as_str().to_owned()),
            TimeAttribute::Precision => time
                .precision
                .map(|precision| precision.as_str().to_owned()),
        })
    })?;
    conn.create_scalar_function("pl_time_is_instant", 2, FLAGS, |ctx| {
        let bytes: Option<Vec<u8>> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let time =
            source_time(bytes.as_deref(), &storage).map_err(rusqlite::Error::UserFunctionError)?;
        Ok(time.at.is_some() && time.precision == Some(Precision::Instant))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests;
