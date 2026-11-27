//! Shared time mechanics with explicit conversion at the millisecond SQL seam:
//! stored times are whole milliseconds, so a query's bound is rounded to the
//! millisecond that decides the same comparison.
use polylogue_query::QueryError;
pub use query_language::time::{Clock, ResolvedTime, Time, duration_ms, resolve};

pub trait Milliseconds {
    fn start_ms(self) -> i64;
    fn end_ms(self) -> Result<i64, QueryError>;
}
impl Milliseconds for Time {
    fn start_ms(self) -> i64 {
        crate::calendar::boundary_ms(self.start())
    }
    fn end_ms(self) -> Result<i64, QueryError> {
        match self {
            Time::Span { end, .. } => Ok(crate::calendar::boundary_ms(end)),
            Time::Instant { at } => i64::try_from(at.as_nanosecond().div_euclid(1_000_000))
                .expect("jiff timestamps fit the millisecond index")
                .checked_add(1)
                .ok_or_else(|| QueryError::Invalid {
                    message: "time boundary cannot be represented by the index".into(),
                    position: 0,
                }),
        }
    }
}

#[cfg(test)]
mod tests;

/// A time predicate before it is compiled to column comparisons
/// ([`super::compile`]'s `time_test`).
#[derive(Debug, Clone)]
pub(crate) enum TimePredicate {
    Known,
    Compare {
        condition: query_language::time::TimeCondition,
        invert: bool,
    },
    Overlap {
        since: Option<jiff::Timestamp>,
        until: Option<jiff::Timestamp>,
    },
}

/// The first whole millisecond at or after `at`: a stored millisecond `t`
/// is at or after `at` exactly when `t >= ceil_ms(at)`.
pub(crate) fn ceil_ms(at: jiff::Timestamp) -> i64 {
    let ns = at.as_nanosecond();
    i64::try_from(ns.div_euclid(1_000_000) + i128::from(ns.rem_euclid(1_000_000) != 0))
        .expect("jiff timestamps fit the millisecond index")
}

/// The last whole millisecond at or before `at`.
pub(crate) fn floor_ms(at: jiff::Timestamp) -> i64 {
    i64::try_from(at.as_nanosecond().div_euclid(1_000_000))
        .expect("jiff timestamps fit the millisecond index")
}
