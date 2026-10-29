//! Shared time mechanics with explicit conversion at the millisecond SQL seam.
pub use query_language::time::{Clock, Time, ResolvedTime, resolve, duration_ms};
use crate::QueryError;
use polylogue_model::LogicalDay;

pub fn clock(now: i64, day: LogicalDay) -> Result<Clock, QueryError> {
    let invalid = |message: String| QueryError::Invalid { message, position: 0 };
    let now = jiff::Timestamp::from_millisecond(now).map_err(|e| invalid(e.to_string()))?;
    let start = jiff::civil::Time::new((day.start_minutes / 60) as i8, (day.start_minutes % 60) as i8, 0, 0)
        .map_err(|e| invalid(e.to_string()))?;
    let day = query_language::time::DaySetting::new(day.zone, start).map_err(|e| QueryError::language(e, ""))?;
    Ok(Clock::new(now, day))
}

pub fn product_day(clock: &Clock) -> LogicalDay {
    LogicalDay { zone: clock.day.zone().clone(), start_minutes: u16::from(clock.day.start().hour() as u8) * 60 + u16::from(clock.day.start().minute() as u8) }
}

/// Dollars to the product's integer nano-dollar storage unit.
pub fn money_nusd(text: &str) -> Option<i64> {
    let text = text.trim().to_ascii_lowercase();
    let text = text.strip_suffix("usd").unwrap_or(&text).trim_start_matches('$');
    let n: f64 = text.parse().ok()?;
    n.is_finite().then(|| (n * 1e9).round() as i64)
}

pub trait Milliseconds { fn start_ms(self) -> i64; fn end_ms(self) -> Result<i64, QueryError>; }
impl Milliseconds for Time {
    fn start_ms(self) -> i64 { self.start().as_millisecond() }
    fn end_ms(self) -> Result<i64, QueryError> {
        match self {
            Time::Span { end, .. } => Ok(end.as_millisecond()),
            Time::Instant { at } => at.as_millisecond().checked_add(1).ok_or_else(|| QueryError::Invalid { message: "time boundary cannot be represented by the index".into(), position: 0 }),
        }
    }
}

#[cfg(test)]
mod tests;
