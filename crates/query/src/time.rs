//! Times, durations and money in queries (spec query.md, Times). Each query
//! resolves its times once, against one `now` and one logical-day setting,
//! and the result records both.

use polylogue_model::LogicalDay;
use serde::Serialize;

/// What a query's times are resolved against.
#[derive(Debug, Clone)]
pub struct Clock {
    /// Unix ms.
    pub now: i64,
    pub day: LogicalDay,
}

impl Clock {
    pub fn new(now: i64, day: LogicalDay) -> Self {
        Self { now, day }
    }
}

/// A resolved time: one instant, or a half-open span such as a logical day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "time", rename_all = "snake_case")]
pub enum Time {
    Instant { ms: i64 },
    Span { start: i64, end: i64 },
}

impl Time {
    /// The first instant it covers.
    pub fn start(self) -> i64 {
        match self {
            Time::Instant { ms } => ms,
            Time::Span { start, .. } => start,
        }
    }

    /// The first instant after it: a span's end, or just after an instant.
    pub fn end(self) -> i64 {
        match self {
            Time::Instant { ms } => ms + 1,
            Time::Span { end, .. } => end,
        }
    }
}

const HOUR: i64 = 3_600_000;

/// `90s`, `5m`, `1.5h`, `3d`, `1w`, `250ms` in milliseconds.
pub fn duration_ms(text: &str) -> Option<i64> {
    let t = text.trim().to_ascii_lowercase();
    let (number, unit) = if let Some(n) = t.strip_suffix("ms") {
        (n, 1.0)
    } else {
        let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
        let (n, u) = t.split_at(split);
        let unit = match u {
            "s" => 1_000.0,
            "m" => 60_000.0,
            "h" => HOUR as f64,
            "d" => 24.0 * HOUR as f64,
            "w" => 168.0 * HOUR as f64,
            _ => return None,
        };
        (n, unit)
    };
    let n: f64 = number.parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    Some((n * unit).round() as i64)
}

/// `2.90` or `2.90usd` in nano-dollars, the unit costs are stored in.
pub fn money_nusd(text: &str) -> Option<i64> {
    let t = text.trim().to_ascii_lowercase();
    let t = t.strip_suffix("usd").unwrap_or(&t).trim_start_matches('$');
    let n: f64 = t.parse().ok()?;
    n.is_finite().then(|| (n * 1e9).round() as i64)
}

/// Resolves a time value: relative (`7d`, `3h`, `2w`: exact hours back from
/// now), a date or logical span (`2026-10-05`, `2026-10`, `today`,
/// `yesterday`, `this week`, `last month`), a time of day (`09:00`, today
/// at that time) or a date-time, with or without an offset.
pub fn resolve(text: &str, clock: &Clock) -> Option<Time> {
    let t = text.trim().to_ascii_lowercase();
    let day = &clock.day;
    let today = day.date_of(clock.now);
    let span = |(start, end): (i64, i64)| Some(Time::Span { start, end });
    match t.as_str() {
        "today" => return span(day.day(today)),
        "yesterday" => return span(day.day(today.yesterday().ok()?)),
        "this week" => return span(day.week(today)),
        "last week" => {
            let (start, _) = day.week(today);
            return span(day.week(day.date_of(start - 1)));
        }
        "this month" => return span(day.month(today)),
        "last month" => {
            let (start, _) = day.month(today);
            return span(day.month(day.date_of(start - 1)));
        }
        _ => {}
    }
    // Relative: hours, days and weeks back from now.
    if let Some(unit) = t.chars().last()
        && matches!(unit, 'h' | 'd' | 'w')
        && t[..t.len() - 1].chars().all(|c| c.is_ascii_digit())
        && t.len() > 1
    {
        let n: i64 = t[..t.len() - 1].parse().ok()?;
        let hours = match unit {
            'h' => n,
            'd' => n * 24,
            _ => n * 168,
        };
        return Some(Time::Instant {
            ms: clock.now - hours * HOUR,
        });
    }
    // A date-time first: the date parser would take its date and drop the
    // time.
    if t.contains('t') {
        if let Ok(ts) = text.trim().parse::<jiff::Timestamp>() {
            return Some(Time::Instant {
                ms: ts.as_millisecond(),
            });
        }
        if let Ok(dt) = t.parse::<jiff::civil::DateTime>() {
            let zoned = day.zone.to_ambiguous_zoned(dt).compatible().ok()?;
            return Some(Time::Instant {
                ms: zoned.timestamp().as_millisecond(),
            });
        }
        return None;
    }
    if let Ok(date) = t.parse::<jiff::civil::Date>() {
        return span(day.day(date));
    }
    if let Some((y, m)) = t.split_once('-')
        && y.len() == 4
        && m.len() == 2
        && let (Ok(y), Ok(m)) = (y.parse::<i16>(), m.parse::<i8>())
        && let Ok(first) = jiff::civil::Date::new(y, m, 1)
    {
        return span(day.month(first));
    }
    if let Ok(time) = t.parse::<jiff::civil::Time>()
        && t.contains(':')
        && !t.contains('t')
    {
        // Today at that time; before the day's start it is the calendar
        // day after the logical date.
        let mut date = today;
        let minutes = i32::from(time.hour()) * 60 + i32::from(time.minute());
        if minutes < i32::from(day.start_minutes) {
            date = date.tomorrow().ok()?;
        }
        let zoned = day
            .zone
            .to_ambiguous_zoned(date.to_datetime(time))
            .compatible()
            .ok()?;
        return Some(Time::Instant {
            ms: zoned.timestamp().as_millisecond(),
        });
    }
    if let Ok(ts) = text.trim().parse::<jiff::Timestamp>() {
        return Some(Time::Instant {
            ms: ts.as_millisecond(),
        });
    }
    if let Ok(dt) = t.parse::<jiff::civil::DateTime>() {
        let zoned = day.zone.to_ambiguous_zoned(dt).compatible().ok()?;
        return Some(Time::Instant {
            ms: zoned.timestamp().as_millisecond(),
        });
    }
    None
}

#[cfg(test)]
mod tests;
