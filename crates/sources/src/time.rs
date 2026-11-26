//! Decode each source's declared timestamp encoding together with its accuracy.
//! Units and precision come from the representation the source wrote, never the clock.

/// Version of the rows this module produces.
pub const VERSION: u32 = 2;

parser_modules! {}

use polylogue_model::{Json, Precision, TimeAccuracy, TimeAccuracyError, TimeQuality, Timestamp};
use serde::{Deserialize, Serialize};

/// A decoded source time remains paired with its accuracy through parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "TimeWire", into = "TimeWire")]
pub struct SourceTime {
    at: Timestamp,
    accuracy: TimeAccuracy,
}
#[derive(Serialize, Deserialize)]
struct TimeWire {
    at: Timestamp,
    accuracy: TimeAccuracy,
}
impl TryFrom<TimeWire> for SourceTime {
    type Error = TimeAccuracyError;
    fn try_from(value: TimeWire) -> Result<Self, Self::Error> {
        Self::new(value.at, value.accuracy)
    }
}
impl From<SourceTime> for TimeWire {
    fn from(value: SourceTime) -> Self {
        Self {
            at: value.at,
            accuracy: value.accuracy,
        }
    }
}
impl SourceTime {
    pub fn new(at: Timestamp, accuracy: TimeAccuracy) -> Result<Self, TimeAccuracyError> {
        accuracy.validate(Some(at))?;
        Ok(Self { at, accuracy })
    }
    pub fn timestamp(self) -> Timestamp {
        self.at
    }
    pub fn accuracy(self) -> TimeAccuracy {
        self.accuracy
    }
    /// A source-declared interval end, never a guessed rounding boundary.
    pub fn with_until(mut self, until: Timestamp) -> Option<Self> {
        self.accuracy.precision = Some(Precision::Bounded);
        self.accuracy.until = Some(until);
        Self::new(self.at, self.accuracy).ok()
    }
    pub fn unix_ms(self) -> i64 {
        self.at.unix_ms()
    }
    pub fn unix_ns(self) -> i128 {
        self.at.unix_ns()
    }
    /// Only call for a field explicitly defined as a capture observation.
    pub fn captured(mut self) -> Self {
        self.accuracy.quality = TimeQuality::Captured;
        self
    }
    /// Only call for a field explicitly defined as source file metadata time.
    pub fn file_mtime(mut self) -> Self {
        self.accuracy.quality = TimeQuality::FileMtime;
        self
    }
}
impl std::fmt::Display for SourceTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.at.fmt(f)
    }
}

pub fn parts(time: Option<SourceTime>) -> (Option<Timestamp>, TimeAccuracy) {
    time.map_or((None, TimeAccuracy::UNKNOWN), |time| {
        (Some(time.at), time.accuracy)
    })
}
fn intrinsic(at: Timestamp, precision: Precision) -> SourceTime {
    SourceTime {
        at,
        accuracy: TimeAccuracy::intrinsic(precision),
    }
}

/// RFC 3339/ISO time with an explicit offset, preserving its stated resolution.
pub fn rfc3339(value: &Json) -> Option<SourceTime> {
    let text = value.as_str()?;
    let at = Timestamp::parse_rfc3339(text)?;
    let clock = text.split_once(['T', 't', ' '])?.1;
    let clock = clock.split(['Z', 'z', '+', '-', '[']).next()?;
    let precision = if clock.contains(['.', ',']) {
        Precision::Instant
    } else if clock.matches(':').count() >= 2 {
        Precision::Second
    } else if clock.contains(':') {
        Precision::Minute
    } else {
        let until = Timestamp::from_unix_ns(at.unix_ns().checked_add(3_600_000_000_000)?)?;
        return SourceTime::new(
            at,
            TimeAccuracy {
                quality: TimeQuality::Intrinsic,
                precision: Some(Precision::Bounded),
                until: Some(until),
            },
        )
        .ok();
    };
    Some(intrinsic(at, precision))
}

/// Integer epoch seconds state whole-second precision; a fractional encoding
/// states a subsecond instant. Decimal arithmetic keeps the nanoseconds as written.
pub fn epoch_seconds(value: &Json) -> Option<SourceTime> {
    let number = value.as_number()?;
    let text = number.to_string();
    let nanos = query_language::number::Decimal::parse(&text)?.exact_scaled_i128(9)?;
    let precision = if number.as_i64().is_some() {
        Precision::Second
    } else {
        Precision::Instant
    };
    Some(intrinsic(Timestamp::from_unix_ns(nanos)?, precision))
}

/// Integer epoch milliseconds are subsecond instants, including exact seconds.
pub fn epoch_millis(value: &Json) -> Option<SourceTime> {
    Some(intrinsic(
        Timestamp::from_unix_ms(value.as_number()?.as_i64()?)?,
        Precision::Instant,
    ))
}

/// Integer nanoseconds as written, including OTLP's decimal string representation.
pub fn epoch_nanos(value: &Json) -> Option<SourceTime> {
    let text = match value {
        Json::Number(number) => number.to_string(),
        Json::String(text) => text.clone(),
        _ => return None,
    };
    Some(intrinsic(
        Timestamp::from_unix_ns(text.parse().ok()?)?,
        Precision::Instant,
    ))
}

#[cfg(test)]
mod tests;
