//! Shared time mechanics with explicit conversion at the millisecond SQL seam.
//! Bounds are ceiled for integer-index comparisons.
//! Native temporal predicates compare canonical evidence at nanosecond precision.
//! Millisecond conversions remain for derived point fields and broad candidates.
use polylogue_query::QueryError;
pub use polylogue_query::time::money_nusd;
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

/// Serialized once by lowering and interpreted by the deterministic SQL adapter.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum NativePredicate {
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
impl NativePredicate {
    pub(crate) fn candidates(&self, cache: &str) -> String {
        use query_language::time::TimeCondition::*;
        let lower = |at: jiff::Timestamp| {
            format!(
                "({cache}) >= {}",
                at.as_nanosecond().div_euclid(1_000_000) - 1
            )
        };
        let upper =
            |at: jiff::Timestamp| format!("({cache}) <= {}", crate::calendar::boundary_ms(at) + 1);
        match self {
            Self::Known | Self::Compare { invert: true, .. } => "1".into(),
            Self::Compare { condition, .. } => match *condition {
                Before { at } | AtOrBefore { at } => upper(at),
                After { at } | AtOrAfter { at } => lower(at),
                Equal { at } => format!("({} AND {})", lower(at), upper(at)),
                Within { range } => format!("({} AND {})", lower(range.start), upper(range.end)),
                NotEqual { .. } => "1".into(),
            },
            Self::Overlap { since, until } => {
                let parts: Vec<_> = since
                    .map(lower)
                    .into_iter()
                    .chain(until.map(upper))
                    .collect();
                if parts.is_empty() {
                    "1".into()
                } else {
                    format!("({})", parts.join(" AND "))
                }
            }
        }
    }
}
