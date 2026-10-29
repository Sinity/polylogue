//! Shared time mechanics with explicit conversion at the millisecond SQL seam.
//! Bounds are ceiled for integer-index comparisons. This does not recover
//! submillisecond source precision absent from the timestamp index.
use polylogue_query::QueryError;
pub use query_language::time::{Clock, ResolvedTime, Time, duration_ms, resolve};

/// Dollars to the product's integer nano-dollar storage unit.
pub fn money_nusd(text: &str) -> Option<i64> {
    let text = text.trim().to_ascii_lowercase();
    let text = text
        .strip_suffix("usd")
        .unwrap_or(&text)
        .trim_start_matches('$');
    let n: f64 = text.parse().ok()?;
    n.is_finite().then(|| (n * 1e9).round() as i64)
}

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
