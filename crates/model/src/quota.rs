use crate::Timestamp;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// One of a subscription's usage limits: a window of fixed length.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaLimit {
    /// The tool's name for the limit (`codex`, `five_hour`, `seven_day`).
    pub limit: Cow<'static, str>,
    pub minutes: i64,
    /// The window starts with the first response after the previous one
    /// ended, not on a fixed schedule.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub from_first_response: bool,
}

/// The state of one limit as the tool reported it with a response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaWindow {
    #[serde(flatten)]
    pub limit: QuotaLimit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resets_at: Option<Timestamp>,
    /// The tool's own word for the state (`allowed`, `rejected`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// The subscription quota a tool reported with a response, read from its
/// own wire format by the origin's parser.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QuotaReading {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    /// Prepaid credits left, as the tool wrote the number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub windows: Vec<QuotaWindow>,
}
