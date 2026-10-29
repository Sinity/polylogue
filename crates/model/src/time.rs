use serde::{Deserialize, Serialize};
use std::fmt;

pub use query_language::time::{Precision, TimeQuality};

/// Evidence accompanying a source time. The timestamp stays in its domain
/// field; this value validates its quality, precision and optional upper bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TimeEvidence {
    pub quality: TimeQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<Precision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<Timestamp>,
}

impl Default for TimeEvidence {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

impl TimeEvidence {
    pub const UNKNOWN: Self = Self {
        quality: TimeQuality::Unknown,
        precision: None,
        until: None,
    };

    pub const fn intrinsic(precision: Precision) -> Self {
        Self {
            quality: TimeQuality::Intrinsic,
            precision: Some(precision),
            until: None,
        }
    }

    pub fn validate(self, at: Option<Timestamp>) -> Result<(), TimeEvidenceError> {
        let Some(at) = at else {
            return if self == Self::UNKNOWN {
                Ok(())
            } else {
                Err(TimeEvidenceError::MissingTime)
            };
        };
        if self.quality == TimeQuality::Unknown {
            return Err(TimeEvidenceError::UnknownQuality);
        }
        let precision = self.precision.ok_or(TimeEvidenceError::MissingPrecision)?;
        if self.until.is_some_and(|until| until <= at) {
            return Err(TimeEvidenceError::InvalidBounds);
        }
        match precision {
            Precision::Instant if self.until.is_some() => Err(TimeEvidenceError::InstantHasBounds),
            Precision::Bounded if self.until.is_none() => Err(TimeEvidenceError::MissingBounds),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TimeEvidenceError {
    #[error("time evidence requires a timestamp")]
    MissingTime,
    #[error("a known timestamp requires known quality")]
    UnknownQuality,
    #[error("a known timestamp requires precision")]
    MissingPrecision,
    #[error("a time upper bound must follow its timestamp")]
    InvalidBounds,
    #[error("an instant cannot carry a time upper bound")]
    InstantHasBounds,
    #[error("bounded precision requires a time upper bound")]
    MissingBounds,
}

pub(crate) const fn unknown_quality() -> TimeQuality {
    TimeQuality::Unknown
}

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

#[cfg(test)]
mod tests;
