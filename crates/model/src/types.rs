use serde::{Deserialize, Serialize};

pub use crate::identity::Origin;

/// Money in integer nano-USD. Credits have a different type and no conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Usd(pub i64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Credits(pub i64);

/// Unknown denominators are represented by None, never by a zero-valued ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RatioWire", into = "RatioWire")]
pub struct Ratio {
    num: u64,
    den: u64,
}
#[derive(Serialize, Deserialize)]
struct RatioWire {
    num: u64,
    den: u64,
}
impl Ratio {
    pub fn new(num: u64, den: u64) -> Option<Self> {
        (den != 0).then_some(Self { num, den })
    }
    pub fn with_known_denominator(num: u64, den: Option<u64>) -> Option<Self> {
        Self::new(num, den?)
    }
    pub fn numerator(self) -> u64 {
        self.num
    }
    pub fn denominator(self) -> u64 {
        self.den
    }
}
impl TryFrom<RatioWire> for Ratio {
    type Error = &'static str;
    fn try_from(value: RatioWire) -> Result<Self, Self::Error> {
        Self::new(value.num, value.den).ok_or("ratio denominator must be nonzero")
    }
}
impl From<Ratio> for RatioWire {
    fn from(value: Ratio) -> Self {
        Self {
            num: value.num,
            den: value.den,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ResultOutcome {
    Ok,
    Empty,
    Degraded { missing: Vec<String> },
    Error { kind: ErrorKind },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    InvalidQuery,
    NotYetAvailable,
    NotFound,
    Ambiguous,
    Conflict,
    Refused { reason: String },
    Timeout,
    NotRunning,
    Io,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    pub kind: ErrorKind,
    pub message: String,
    pub position: Option<usize>,
    pub suggestion: Option<String>,
    pub hint: Option<String>,
    pub retryable: bool,
}
impl Failure {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            position: None,
            suggestion: None,
            hint: None,
            retryable: false,
        }
    }
    pub fn outcome(&self) -> ResultOutcome {
        ResultOutcome::Error {
            kind: self.kind.clone(),
        }
    }
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Failure {}
impl ErrorKind {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidQuery | Self::NotYetAvailable | Self::Ambiguous => 2,
            Self::Refused { .. } => 3,
            Self::NotFound | Self::Conflict | Self::Io => 4,
            Self::NotRunning => 6,
            Self::Timeout => 7,
        }
    }
}
impl ResultOutcome {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Ok => 0,
            Self::Empty => 1,
            Self::Degraded { .. } => 5,
            Self::Error { kind } => kind.exit_code(),
        }
    }
}

/// A parser accounts for each record independently of query or tool outcomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordOutcome<I> {
    Used,
    Ignored(I),
    Unrecognized(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ratio_validates_wire_and_preserves_integer_extremes() {
        assert!(Ratio::new(10, 0).is_none());
        assert!(Ratio::with_known_denominator(10, None).is_none());
        for wire in [
            r#"{"num":1,"den":0}"#,
            r#"{"num":1}"#,
            r#"{"num":1,"den":null}"#,
        ] {
            assert!(serde_json::from_str::<Ratio>(wire).is_err());
        }
        for (num, den) in [(0, 1), (u64::MAX, 1), (u64::MAX, u64::MAX)] {
            let ratio = Ratio::new(num, den).unwrap();
            let decoded: Ratio =
                serde_json::from_slice(&serde_json::to_vec(&ratio).unwrap()).unwrap();
            assert_eq!((decoded.numerator(), decoded.denominator()), (num, den));
        }
    }
    #[test]
    fn failures_have_non_success_outcomes_and_preserve_details() {
        let mut failure = Failure::new(
            ErrorKind::Refused {
                reason: "unsupported".into(),
            },
            "search is unavailable",
        );
        failure.hint = Some("foundation diagnostics are available".into());
        let decoded: Failure =
            serde_json::from_slice(&serde_json::to_vec(&failure).unwrap()).unwrap();
        assert_eq!(decoded, failure);
        assert_eq!(decoded.outcome().exit_code(), 3);
        assert_eq!(ResultOutcome::Empty.exit_code(), 1);
        assert_eq!(
            ResultOutcome::Degraded {
                missing: vec!["input".into()]
            }
            .exit_code(),
            5
        );
        for (kind, code) in [
            (ErrorKind::InvalidQuery, 2),
            (ErrorKind::NotYetAvailable, 2),
            (ErrorKind::NotFound, 4),
            (ErrorKind::Ambiguous, 2),
            (ErrorKind::Conflict, 4),
            (ErrorKind::Timeout, 7),
            (ErrorKind::NotRunning, 6),
            (ErrorKind::Io, 4),
        ] {
            assert_eq!(Failure::new(kind, "failed").outcome().exit_code(), code);
        }
    }
}
