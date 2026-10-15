use serde::{Deserialize, Serialize};

/// UTC milliseconds since Unix epoch. No machine-local timezone conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(pub i64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeQuality { Intrinsic, FileMtime, Unknown }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision { Instant, Second, Minute, Day, Month, Year, Bounded }

/// Unknown provider time has no invented epoch timestamp or precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "time_quality", rename_all = "snake_case")]
pub enum ProviderTime {
    Intrinsic { at: Timestamp, precision: Precision, until: Option<Timestamp> },
    FileMtime { at: Timestamp, precision: Precision, until: Option<Timestamp> },
    Unknown,
}
impl ProviderTime {
    pub fn quality(self) -> TimeQuality {
        match self {
            Self::Intrinsic { .. } => TimeQuality::Intrinsic,
            Self::FileMtime { .. } => TimeQuality::FileMtime,
            Self::Unknown => TimeQuality::Unknown,
        }
    }
    pub fn at(self) -> Option<Timestamp> {
        match self {
            Self::Intrinsic { at, .. } | Self::FileMtime { at, .. } => Some(at),
            Self::Unknown => None,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_is_not_epoch_or_default_precision() {
        let unknown: ProviderTime = serde_json::from_str(r#"{"time_quality":"unknown"}"#).unwrap();
        assert_eq!(unknown.at(), None);
        assert_eq!(unknown.quality(), TimeQuality::Unknown);
        let epoch = ProviderTime::Intrinsic { at: Timestamp(0), precision: Precision::Instant, until: None };
        assert_ne!(epoch, unknown);
        assert!(serde_json::from_str::<ProviderTime>(r#"{"time_quality":"intrinsic","precision":"instant"}"#).is_err());
    }
}
