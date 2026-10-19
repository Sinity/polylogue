use serde::{Deserialize, Serialize};
use std::fmt;

/// A point in time as a source stated it. Unknown times are `None` at the
/// use site, never a zero or a guess.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(jiff::Timestamp);

impl Timestamp {
    pub fn from_unix_ms(ms: i64) -> Option<Self> {
        jiff::Timestamp::from_millisecond(ms).ok().map(Self)
    }

    pub fn from_unix_ns(ns: i128) -> Option<Self> {
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
mod logical_day {
    use super::LogicalDay;
    use jiff::civil::date;

    fn warsaw() -> LogicalDay {
        LogicalDay::named("Europe/Warsaw", LogicalDay::DEFAULT_START_MINUTES).unwrap()
    }

    fn ms(text: &str) -> i64 {
        text.parse::<jiff::Timestamp>().unwrap().as_millisecond()
    }

    #[test]
    fn work_after_midnight_belongs_to_the_day_before() {
        let d = warsaw();
        // 01:30 local on Oct 6 (CEST, +02:00) is logical Oct 5. Clocks change
        // at 02:00, before the 04:00 start, so the short and long logical
        // days are the Saturdays before the change.
        assert_eq!(d.date_of(ms("2026-10-05T23:30:00Z")), date(2026, 10, 5));
        assert_eq!(d.date_of(ms("2026-10-06T02:00:00Z")), date(2026, 10, 6));
        assert_eq!(d.day_label(ms("2026-10-06T01:59:59Z")), "2026-10-05");
    }

    #[test]
    fn dst_days_are_23_and_25_hours() {
        let d = warsaw();
        let (s, e) = d.day(date(2026, 3, 28));
        assert_eq!(e - s, 23 * 3_600_000);
        let (s, e) = d.day(date(2026, 10, 24));
        assert_eq!(e - s, 25 * 3_600_000);
        let (s, e) = d.day(date(2026, 10, 5));
        assert_eq!(e - s, 24 * 3_600_000);
        assert_eq!(s, ms("2026-10-05T02:00:00Z"));
    }

    #[test]
    fn weeks_start_on_monday_and_months_on_the_first() {
        let d = warsaw();
        let (s, e) = d.week(date(2026, 10, 8));
        assert_eq!(s, d.day(date(2026, 10, 5)).0);
        assert_eq!(e, d.day(date(2026, 10, 12)).0);
        assert_eq!(d.week_label(ms("2026-10-08T12:00:00Z")), "2026-W41");
        let (s, _) = d.month(date(2026, 10, 18));
        assert_eq!(s, d.day(date(2026, 10, 1)).0);
        assert_eq!(d.month_label(ms("2026-10-31T23:00:00Z")), "2026-10");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rfc3339_with_offset_only() {
        let t = Timestamp::parse_rfc3339("2026-01-02T03:04:05.678Z").unwrap();
        assert_eq!(t.unix_ms(), 1_767_323_045_678);
        assert!(Timestamp::parse_rfc3339("2026-01-02T03:04:05").is_none());
        assert_eq!(
            Timestamp::parse_rfc3339("2026-01-02T04:04:05.678+01:00"),
            Some(t)
        );
    }

    #[test]
    fn serializes_as_rfc3339() {
        let t = Timestamp::from_unix_ms(0).unwrap();
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            "\"1970-01-01T00:00:00Z\""
        );
    }
}
