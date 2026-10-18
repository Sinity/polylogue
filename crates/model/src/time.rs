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
