use serde::{Deserialize, Serialize};

/// A tool result's recorded status. Unknown never defaults to success.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcome {
    Ok,
    Error { reason: Option<ErrorReason> },
    Unknown { reason: UnknownReason },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownReason {
    NoResult,
    PassedWithoutResult,
    BackgroundNeverReported,
    ResultWithoutStatus,
    OutputNotRecorded,
    StatusFieldUnreadable,
    JsonMarkerNotDecoded,
    Other(String),
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorReason {
    Denied(String),
    Interrupted,
    TimedOut,
    BlockedByHook,
    Other(String),
}
