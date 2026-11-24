//! The closed sets a delivery record is stored and queried by.

stored_names! {
    /// What a delivery answered.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DeliveryEvent {
        SessionStart => "session_start",
        Resume => "resume",
        Compaction => "compaction",
        SessionBrief => "session_brief",
        SearchBrief => "search_brief",
        Signal => "signal",
        /// A hint just before a tool runs.
        Advice => "advice",
        /// An addressed message to a session.
        Message => "message",
        /// A signal rule's notice to a session.
        Notice => "notice",
        /// A live-session act: an answer to a pending request, an interrupt.
        Control => "control",
        /// The first turn of a session Polylogue starts: a role's text, a
        /// hand-off's context, the text it was started with.
        FirstTurn => "first_turn",
    }
}

stored_names! {
    /// What prepared a delivery.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DeliverySender {
        Hook => "hook",
        Mcp => "mcp",
        Signal => "signal",
        /// A message, from whichever surface its author used.
        Send => "send",
        /// A live-session act from the command line, MCP or the web.
        Act => "act",
    }
}

stored_names! {
    /// What happened to a delivery after it was prepared.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DeliveryOutcome {
        Delivered => "delivered",
        Failed => "failed",
        NotSent => "not_sent",
        Used => "used",
    }
}

/// A delivery's state: `prepared` until an outcome is recorded.
pub const DELIVERY_STATES: &[&str] = &["prepared", "delivered", "failed", "not_sent", "used"];

#[cfg(test)]
mod tests;
