use serde::{Deserialize, Serialize};
use std::fmt;

/// A point in time as a source stated it. Unknown times are `None` at the
/// use site, never a zero or a guess.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct Timestamp(#[schemars(with = "String")] jiff::Timestamp);

impl Timestamp {
    pub fn from_unix_ms(ms: i64) -> Option<Self> {
        jiff::Timestamp::from_millisecond(ms).ok().map(Self)
    }

    pub fn from_unix_ns(ns: i128) -> Option<Self> {
        // The current jiff nanosecond constructor checks i64 seconds but
        // reaches its internal unchecked constructor outside timestamp bounds.
        if !(jiff::Timestamp::MIN.as_nanosecond()..=jiff::Timestamp::MAX.as_nanosecond())
            .contains(&ns)
        {
            return None;
        }
        jiff::Timestamp::from_nanosecond(ns).ok().map(Self)
    }

    /// RFC 3339 with an offset (`2026-01-02T03:04:05.678Z`). A text without
    /// an offset is refused: its zone is unknown.
    pub fn parse_rfc3339(text: &str) -> Option<Self> {
        text.parse::<jiff::Timestamp>().ok().map(Self)
    }

    pub fn unix_ms(self) -> i64 {
        self.0.as_millisecond()
    }

    pub fn unix_ns(self) -> i128 {
        self.0.as_nanosecond()
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timestamp({})", self.0)
    }
}

/// The operator's day as a person reads it (joint decision 15): date D runs
/// from D at `start` to D+1 at the same wall-clock time, in `zone`, so work
/// at 01:30 belongs to the day before. A DST day is 23 or 25 hours long. A
/// start time in a DST gap means the first instant after the gap; in a fold,
/// the earlier instant.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalDay {
    pub zone: jiff::tz::TimeZone,
    /// Minutes after midnight; 240 is 04:00.
    pub start_minutes: u16,
}

impl LogicalDay {
    pub const DEFAULT_START_MINUTES: u16 = 4 * 60;

    /// The machine's zone and the default 04:00 start.
    pub fn system() -> Self {
        Self {
            zone: jiff::tz::TimeZone::system(),
            start_minutes: Self::DEFAULT_START_MINUTES,
        }
    }

    /// A named IANA zone, or `None` when the name is unknown.
    pub fn named(zone: &str, start_minutes: u16) -> Option<Self> {
        Some(Self {
            zone: jiff::tz::TimeZone::get(zone).ok()?,
            start_minutes,
        })
    }

    /// The zone's IANA name, when it has one.
    pub fn zone_name(&self) -> Option<&str> {
        self.zone.iana_name()
    }

    fn start_time(&self) -> jiff::civil::Time {
        jiff::civil::Time::new(
            (self.start_minutes / 60).min(23) as i8,
            (self.start_minutes % 60) as i8,
            0,
            0,
        )
        .unwrap_or(jiff::civil::Time::midnight())
    }

    /// The instant logical date `date` starts.
    pub fn start_of(&self, date: jiff::civil::Date) -> i64 {
        // `compatible`: a gap resolves to the instant after it, a fold to
        // the earlier instant.
        self.zone
            .to_ambiguous_zoned(date.to_datetime(self.start_time()))
            .compatible()
            .map_or(0, |z| z.timestamp().as_millisecond())
    }

    /// The logical date an instant belongs to.
    /// The local hour (0-23) of an instant.
    pub fn local_hour(&self, ms: i64) -> u8 {
        let ts = jiff::Timestamp::from_millisecond(ms).unwrap_or(jiff::Timestamp::UNIX_EPOCH);
        ts.to_zoned(self.zone.clone()).hour() as u8
    }

    pub fn date_of(&self, ms: i64) -> jiff::civil::Date {
        let ts = jiff::Timestamp::from_millisecond(ms).unwrap_or(jiff::Timestamp::UNIX_EPOCH);
        let local = ts.to_zoned(self.zone.clone()).date();
        if ms < self.start_of(local) {
            local.yesterday().unwrap_or(local)
        } else {
            local
        }
    }

    /// `[start, end)` of logical date `date`, in Unix ms.
    pub fn day(&self, date: jiff::civil::Date) -> (i64, i64) {
        let next = date.tomorrow().unwrap_or(date);
        (self.start_of(date), self.start_of(next))
    }

    /// `[start, end)` of the logical ISO week (Monday first) holding `date`.
    pub fn week(&self, date: jiff::civil::Date) -> (i64, i64) {
        let back = i64::from(date.weekday().to_monday_zero_offset());
        let monday = date
            .checked_sub(jiff::Span::new().days(back))
            .unwrap_or(date);
        let next = monday
            .checked_add(jiff::Span::new().days(7))
            .unwrap_or(monday);
        (self.start_of(monday), self.start_of(next))
    }

    /// `[start, end)` of the logical month holding `date`.
    pub fn month(&self, date: jiff::civil::Date) -> (i64, i64) {
        let first = date.first_of_month();
        let next = first
            .checked_add(jiff::Span::new().months(1))
            .unwrap_or(first);
        (self.start_of(first), self.start_of(next))
    }

    /// Group labels: the logical date `2026-10-05`, the ISO week
    /// `2026-W41`, the month `2026-10`.
    pub fn day_label(&self, ms: i64) -> String {
        self.date_of(ms).to_string()
    }

    pub fn week_label(&self, ms: i64) -> String {
        let d = self.date_of(ms).iso_week_date();
        format!("{:04}-W{:02}", d.year(), d.week())
    }

    pub fn month_label(&self, ms: i64) -> String {
        let d = self.date_of(ms);
        format!("{:04}-{:02}", d.year(), d.month())
    }
}

#[cfg(test)]
mod logical_day;

#[cfg(test)]
mod tests;
