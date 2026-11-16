//! Temporal predicates over stored source times, without zone I/O.
use crate::query::{fields::TimeAttribute, time::TimePredicate};
use polylogue_model::{Precision, Timestamp};
use query_language::semantics::Truth;
use query_language::time::{Time, TimeValue};
use rusqlite::{Connection, functions::FunctionFlags};

type Failure = Box<dyn std::error::Error + Send + Sync + 'static>;

mod order;
mod source_time;
use source_time::read as source_time;

fn period(
    at: Option<Timestamp>,
    precision: Option<Precision>,
    until: Option<Timestamp>,
) -> Option<Time> {
    let at: jiff::Timestamp = at?.into();
    if precision == Some(Precision::Instant) {
        return Some(Time::Instant { at });
    }
    let end = match until {
        Some(end) => end.into(),
        None => match precision {
            Some(Precision::Second | Precision::Minute) => {
                let seconds = if precision == Some(Precision::Second) {
                    1
                } else {
                    60
                };
                at.checked_add(jiff::SignedDuration::from_secs(seconds))
                    .ok()?
            }
            // A calendar grain needs the source's boundary or upper bound.
            // The operation's own zone cannot stand in for the source's.
            _ => return None,
        },
    };
    Some(Time::Span { start: at, end })
}

fn test(
    bytes: Option<&[u8]>,
    storage: &str,
    predicate: &TimePredicate,
) -> Result<Option<bool>, Failure> {
    let time = source_time(bytes, storage)?;
    if matches!(predicate, TimePredicate::Known) {
        return Ok(Some(time.at.is_some()));
    }
    let Some(period) = time.period() else {
        return Ok(None);
    };
    match predicate {
        TimePredicate::Known => unreachable!("handled known predicate"),
        TimePredicate::Overlap { since, until } => Ok(Some(query_language::time::overlaps_bounds(
            period, *since, *until,
        ))),
        TimePredicate::Compare { condition, invert } => {
            // An explicit source period supplies its bounds independently of
            // a calendar label. Whole-period comparisons are the shared rule.
            let value = match period {
                Time::Instant { at } => TimeValue::instant(at),
                period => TimeValue::new(period, Precision::Bounded, None)?,
            };
            let truth = value.compare(condition);
            let truth = if *invert { truth.not() } else { truth };
            Ok(match truth {
                Truth::True => Some(true),
                Truth::False => Some(false),
                Truth::Unknown => None,
            })
        }
    }
}

pub(super) fn register(conn: &Connection) -> rusqlite::Result<()> {
    let flags = FunctionFlags::SQLITE_UTF8.union(FunctionFlags::SQLITE_DETERMINISTIC);
    conn.create_aggregate_function("pl_time_latest", 1, flags, source_time::Latest)?;
    order::register(conn)?;
    conn.create_scalar_function("pl_time_attribute", 3, flags, |ctx| {
        let bytes: Option<Vec<u8>> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let attribute: String = ctx.get(2)?;
        let attribute = TimeAttribute::from_code(&attribute)
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
    conn.create_scalar_function("pl_time_is_point", 2, flags, |ctx| {
        let bytes: Option<Vec<u8>> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let time =
            source_time(bytes.as_deref(), &storage).map_err(rusqlite::Error::UserFunctionError)?;
        Ok(time.at.is_some() && time.precision == Some(Precision::Instant))
    })?;
    conn.create_scalar_function("pl_time_test", 3, flags, |ctx| {
        let bytes: Option<Vec<u8>> = ctx.get(0)?;
        let storage: String = ctx.get(1)?;
        let predicate: String = ctx.get(2)?;
        let predicate: TimePredicate = serde_json::from_str(&predicate)
            .map_err(|error| rusqlite::Error::UserFunctionError(Box::new(error)))?;
        test(bytes.as_deref(), &storage, &predicate).map_err(rusqlite::Error::UserFunctionError)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests;
