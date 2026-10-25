//! Normalized readings whose source does not identify an owning response.
//! The enclosing Event time is when the record was emitted, not necessarily
//! when the provider measured usage or quota.
use crate::{MessageId, QuotaReading, Timestamp, Usage};
use serde::{Deserialize, Serialize};

pub const USAGE_REPORT: &str = "usage_report";
pub const QUOTA_REPORT: &str = "quota_report";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageReport {
    /// Identity of the original reading, shared by cloned copies.
    pub report_id: String,
    pub usage: Usage,
    pub model: Option<String>,
    /// None leaves response membership unknown.
    pub response: Option<MessageId>,
    /// A measured completion report, not an entire response object or a
    /// count of all API attempts. None means this is not established.
    pub measured_completion: Option<bool>,
    /// Provider time, only when the source supplies it independently.
    pub measured_at: Option<Timestamp>,
    /// Inclusive totals remain known when missing component counters
    /// prevent deriving uncached input or non-reasoning output.
    pub input_including_cache: Option<u64>,
    pub output_including_reasoning: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaReport {
    pub report_id: String,
    pub reading: QuotaReading,
    pub measured_at: Option<Timestamp>,
}
