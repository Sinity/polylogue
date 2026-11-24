//! The closed sets a delivery record is stored and queried by.

use serde::{Deserialize, Serialize};

/// What a delivery answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryEvent {
    SessionStart,
    Resume,
    Compaction,
    SessionBrief,
    SearchBrief,
    Signal,
    /// A hint just before a tool runs.
    Advice,
    /// An addressed message to a session.
    Message,
    /// A signal rule's notice to a session.
    Notice,
    /// A live-session act: an answer to a pending request, an interrupt.
    Control,
    /// The first turn of a session Polylogue starts: a role's text, a
    /// hand-off's context, the text it was started with.
    FirstTurn,
}

stored_names!(DeliveryEvent {
    SessionStart => "session_start",
    Resume => "resume",
    Compaction => "compaction",
    SessionBrief => "session_brief",
    SearchBrief => "search_brief",
    Signal => "signal",
    Advice => "advice",
    Message => "message",
    Notice => "notice",
    Control => "control",
    FirstTurn => "first_turn",
});

/// What prepared a delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DeliverySender {
    Hook,
    Mcp,
    Signal,
    /// A message, from whichever surface its author used.
    Send,
    /// A live-session act from the command line, MCP or the web.
    Act,
}

stored_names!(DeliverySender {
    Hook => "hook",
    Mcp => "mcp",
    Signal => "signal",
    Send => "send",
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
