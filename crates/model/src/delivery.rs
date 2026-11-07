//! The closed sets a delivery record is stored and queried by.

use serde::{Deserialize, Serialize};

/// What a delivery answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryEvent {
    SessionStart,
    Resume,
    Compaction,
    Takeover,
    Recall,
    Signal,
    /// A hint just before a tool runs.
    Advice,
    /// An addressed message to a session.
    Ping,
    /// A signal rule's notice to a session.
    Notice,
    /// A live-session act: an answer to a pending request, an interrupt.
    Control,
    /// The first turn of a session Polylogue starts: a role's text, a
    /// hand-off's context, the text it was started with.
    Pack,
}

stored_names!(DeliveryEvent {
    SessionStart => "session_start",
    Resume => "resume",
    Compaction => "compaction",
    Takeover => "takeover",
    Recall => "recall",
    Signal => "signal",
    Advice => "advice",
    Ping => "ping",
    Notice => "notice",
    Control => "control",
    Pack => "pack",
});

/// What prepared a delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeliverySender {
    Hook,
    Mcp,
    Signal,
    /// A ping, from whichever surface its author used.
    Ping,
    /// A live-session act from the command line, MCP or the web.
    Act,
}

stored_names!(DeliverySender {
    Hook => "hook",
    Mcp => "mcp",
    Signal => "signal",
    Ping => "ping",
    Act => "act",
});

/// What happened to a delivery after it was prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
