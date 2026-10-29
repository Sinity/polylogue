//! Decode each source's declared timestamp encoding together with its evidence.
//! Units and precision come from the native representation, never the clock.

/// Version of the rows this module produces.
pub const VERSION: u32 = 2;

use polylogue_model::{Json, Precision, TimeEvidence, TimeEvidenceError, TimeQuality, Timestamp};
use serde::{Deserialize, Serialize};

/// A decoded native time remains paired with its evidence through parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "TimeWire", into = "TimeWire")]
pub struct SourceTime {
    at: Timestamp,
    evidence: TimeEvidence,
}
#[derive(Serialize, Deserialize)]
struct TimeWire {
    at: Timestamp,
    evidence: TimeEvidence,
}
impl TryFrom<TimeWire> for SourceTime {
    type Error = TimeEvidenceError;
    fn try_from(value: TimeWire) -> Result<Self, Self::Error> {
        Self::new(value.at, value.evidence)
    }
}
impl From<SourceTime> for TimeWire {
    fn from(value: SourceTime) -> Self {
        Self {
            at: value.at,
            evidence: value.evidence,
        }
    }
}
impl SourceTime {
    pub fn new(at: Timestamp, evidence: TimeEvidence) -> Result<Self, TimeEvidenceError> {
        evidence.validate(Some(at))?;
        Ok(Self { at, evidence })
    }
    pub fn timestamp(self) -> Timestamp {
        self.at
    }
    pub fn evidence(self) -> TimeEvidence {
        self.evidence
    }
    /// A source-declared interval end, never a guessed rounding boundary.
    pub fn with_until(mut self, until: Timestamp) -> Option<Self> {
        self.evidence.precision = Some(Precision::Bounded);
        self.evidence.until = Some(until);
        Self::new(self.at, self.evidence).ok()
    }
    pub fn unix_ms(self) -> i64 {
        self.at.unix_ms()
    }
    pub fn unix_ns(self) -> i128 {
        self.at.unix_ns()
    }
    /// Only call for a field explicitly defined as a capture observation.
    pub fn captured(mut self) -> Self {
        self.evidence.quality = TimeQuality::Captured;
        self
    }
    /// Only call for a field explicitly defined as source file metadata time.
    pub fn file_mtime(mut self) -> Self {
        self.evidence.quality = TimeQuality::FileMtime;
        self
    }
}
impl std::fmt::Display for SourceTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.at.fmt(f)
    }
}

pub fn parts(time: Option<SourceTime>) -> (Option<Timestamp>, TimeEvidence) {
    time.map_or((None, TimeEvidence::UNKNOWN), |time| {
        (Some(time.at), time.evidence)
    })
}
fn intrinsic(at: Timestamp, precision: Precision) -> SourceTime {
    SourceTime {
        at,
        evidence: TimeEvidence::intrinsic(precision),
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
            TimeEvidence {
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
/// states a subsecond instant. Decimal arithmetic retains native nanoseconds.
pub fn epoch_seconds(value: &Json) -> Option<SourceTime> {
    let number = value.as_number()?;
    let text = number.to_string();
    let nanos = scaled_decimal(&text, 9)?;
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

/// Native integer nanoseconds, including OTLP's decimal string representation.
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

fn scaled_decimal(text: &str, places: i64) -> Option<i128> {
    let (negative, text) = text
        .strip_prefix('-')
        .map_or((false, text), |text| (true, text));
    let (mantissa, exponent) = text
        .split_once(['e', 'E'])
        .map_or(Some((text, 0i64)), |(mantissa, exponent)| {
            Some((mantissa, exponent.parse().ok()?))
        })?;
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits = format!("{whole}{fraction}");
    let mut scale = places
        .checked_add(exponent)?
        .checked_sub(i64::try_from(fraction.len()).ok()?)?;
    while scale < 0 && digits.ends_with('0') {
        digits.pop();
        scale += 1;
    }
    if scale < 0 {
        return None;
    }
    let magnitude = digits
        .parse::<i128>()
        .ok()?
        .checked_mul(10i128.checked_pow(scale.try_into().ok()?)?)?;
    if negative {
        magnitude.checked_neg()
    } else {
        Some(magnitude)
    }
}

#[cfg(test)]
mod tests;
