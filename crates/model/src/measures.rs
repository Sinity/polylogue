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

impl ErrorReason {
    pub fn code(&self) -> String {
        match self {
            ErrorReason::Denied(kind) => format!("denied:{kind}"),
            ErrorReason::Interrupted => "interrupted".into(),
            ErrorReason::TimedOut => "timed_out".into(),
            ErrorReason::BlockedByHook => "blocked_by_hook".into(),
            ErrorReason::Other(s) => s.clone(),
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "interrupted" => ErrorReason::Interrupted,
            "timed_out" => ErrorReason::TimedOut,
            "blocked_by_hook" => ErrorReason::BlockedByHook,
            other => match other.strip_prefix("denied:") {
                Some(kind) => ErrorReason::Denied(kind.to_owned()),
                None => ErrorReason::Other(other.to_owned()),
            },
        }
    }
}

impl UnknownReason {
    pub fn as_str(&self) -> &str {
        match self {
            UnknownReason::NoResult => "no_result",
            UnknownReason::PassedWithoutResult => "passed_without_result",
            UnknownReason::BackgroundNeverReported => "background_never_reported",
            UnknownReason::ResultWithoutStatus => "result_without_status",
            UnknownReason::OutputNotRecorded => "output_not_recorded",
            UnknownReason::StatusFieldUnreadable => "status_field_unreadable",
            UnknownReason::JsonMarkerNotDecoded => "json_marker_not_decoded",
            UnknownReason::Other(s) => s,
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "no_result" => UnknownReason::NoResult,
            "passed_without_result" => UnknownReason::PassedWithoutResult,
            "background_never_reported" => UnknownReason::BackgroundNeverReported,
            "result_without_status" => UnknownReason::ResultWithoutStatus,
            "output_not_recorded" => UnknownReason::OutputNotRecorded,
            "status_field_unreadable" => UnknownReason::StatusFieldUnreadable,
            "json_marker_not_decoded" => UnknownReason::JsonMarkerNotDecoded,
            other => UnknownReason::Other(other.to_owned()),
        }
    }
}

impl ToolOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolOutcome::Ok => "ok",
            ToolOutcome::Error { .. } => "error",
            ToolOutcome::Unknown { .. } => "unknown",
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self, ToolOutcome::Ok)
    }
}
