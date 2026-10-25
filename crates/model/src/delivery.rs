//! The closed sets a delivery record is stored and queried by.

use serde::{Deserialize, Serialize};

/// What a delivery answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryEvent {
    SessionStart,
    Resume,
    Compaction,
    Takeover,
    Recall,
    Signal,
}

stored_names!(DeliveryEvent {
    SessionStart => "session_start",
    Resume => "resume",
    Compaction => "compaction",
    Takeover => "takeover",
    Recall => "recall",
    Signal => "signal",
});

/// What prepared a delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliverySender {
    Hook,
    Mcp,
    Signal,
}

stored_names!(DeliverySender {
    Hook => "hook",
    Mcp => "mcp",
    Signal => "signal",
});

/// What happened to a delivery after it was prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryOutcome {
    Delivered,
    Failed,
    NotSent,
    Used,
}

stored_names!(DeliveryOutcome {
    Delivered => "delivered",
    Failed => "failed",
    NotSent => "not_sent",
    Used => "used",
});

/// A delivery's state: `prepared` until an outcome is recorded.
pub const DELIVERY_STATES: &[&str] = &["prepared", "delivered", "failed", "not_sent", "used"];

#[cfg(test)]
mod tests;
