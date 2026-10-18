use crate::{Block, MessageId, SessionId, Timestamp, Usage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// What kind of user message this is, so counts of "what the operator
/// typed" exclude what the harness wrote in the user's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptKind {
    /// Prose a person typed.
    #[default]
    Prose,
    /// A slash command, shell escape or memory line.
    Command,
    /// Text the harness wrote: tool results, interruption notices, local
    /// command output, injected reminders.
    Harness,
}

/// One message, stored once, in the session where it originated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub session: SessionId,
    /// The message this one replies to, when the source records it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<MessageId>,
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_kind: Option<PromptKind>,
    pub blocks: Vec<Block>,
    #[serde(default, skip_serializing_if = "Usage::is_empty")]
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Cost the tool itself reported for this response, as written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_cost_usd: Option<serde_json::Number>,
    /// How long the response took, as the tool reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl Message {
    /// Prose text of the message, blocks joined by blank lines.
    pub fn text(&self) -> String {
        let parts: Vec<&str> = self.blocks.iter().filter_map(|b| b.text()).collect();
        parts.join("\n\n")
    }
}
