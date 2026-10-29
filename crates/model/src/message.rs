use crate::{Block, MessageId, SessionId, Timestamp, Usage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

stored_names!(Role {
    User => "user",
    Assistant => "assistant",
    System => "system",
});

/// A user prompt's category. `Harness` also marks runtime-written notices
/// carried under another native role; the other kinds remain user categories.
/// This keeps harness material out of counts of what a person or model wrote.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PromptKind {
    /// Prose a person typed.
    #[default]
    Prose,
    /// A slash command, shell escape or memory line.
    Command,
    /// Text the harness wrote: tool results, interruption notices, local
    /// command output, injected reminders or assistant-shaped runtime notices.
    Harness,
    /// Text another model wrote as this session's prompt: a subagent's
    /// task, a coordinator's follow-up, a peer agent's message.
    Model,
}

stored_names!(PromptKind {
    Prose => "prose",
    Command => "command",
    Harness => "harness",
    Model => "model",
});

/// Who wrote a message's text, from its role and prompt kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WrittenBy {
    Human,
    Model,
    Harness,
    Unknown,
}

stored_names!(WrittenBy {
    Human => "human",
    Model => "model",
    Harness => "harness",
    Unknown => "unknown",
});

impl WrittenBy {
    pub fn of(role: Role, prompt_kind: Option<PromptKind>) -> Self {
        match (role, prompt_kind) {
            (_, Some(PromptKind::Harness)) | (Role::System, _) => Self::Harness,
            (Role::Assistant, _) | (Role::User, Some(PromptKind::Model)) => Self::Model,
            (Role::User, Some(_)) => Self::Human,
            (Role::User, None) => Self::Unknown,
        }
    }
}

/// One message, stored once, in the session where it originated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// How the harness produced the message, beyond its content.
    #[serde(default, skip_serializing_if = "Production::is_empty")]
    pub production: Production,
    /// Provider fields indexed as given, including nested citations and content details.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub attrs: serde_json::Map<String, serde_json::Value>,
}

/// Server tools are billed separately from model tokens.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ServerToolUse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_fetch_requests: Option<u64>,
}

/// What the harness recorded about producing a message: the settings it
/// ran with, who asked for it, and the API's own diagnostics. Measurement
/// reads these; none of it is content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Production {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u64>,
    /// Provider response channel, such as commentary or final.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// These recorded items have no evidenced model-response grouping.
    /// None is an absent update; a supplied false remains distinct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_boundary_unknown: Option<bool>,

    /// Reasoning effort the request asked for (`low`, `medium`, `high`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// The model the request named, when the one that answered differs
    /// (a fallback) or carries a variant suffix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model: Option<String>,
    /// The model consulted as an advisor for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advisor_model: Option<String>,
    #[serde(default, skip_serializing_if = "Attribution::is_empty")]
    pub attribution: Attribution,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_ms: Option<u64>,
    /// Time to the first token of the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_ms: Option<u64>,
    /// The API error this message reports, instead of a response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_error: Option<ApiError>,
    /// Why the prompt cache missed, as the API reported it, with the
    /// tokens it cost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_miss: Option<serde_json::Value>,
    /// The refusal category and explanation, when the model refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<serde_json::Value>,
    /// The permission decision for the tool call this message answers:
    /// decision, its source and the kind of reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission: Option<serde_json::Value>,
    /// The harness's ID for the API request that produced this message,
    /// which its request records carry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    /// The model API's own ID for the request (Anthropic `requestId`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// What started the turn, when not a person at the terminal (`sdk`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_origin: Option<String>,
    /// The subscription's quota state the tool reported with this
    /// response: which windows, how used, when they reset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quota: Option<crate::QuotaReading>,
}

impl Production {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Takes what a later record of the same message states; facts it is
    /// silent about stay as they were.
    pub fn merge(&mut self, later: Production) {
        fn take<T>(into: &mut Option<T>, from: Option<T>) {
            if from.is_some() {
                *into = from;
            }
        }
        take(
            &mut self.response_boundary_unknown,
            later.response_boundary_unknown,
        );
        if let Some(server) = later.server_tool_use {
            let current = self
                .server_tool_use
                .get_or_insert_with(ServerToolUse::default);
            take(&mut current.web_search_requests, server.web_search_requests);
            take(&mut current.web_fetch_requests, server.web_fetch_requests);
        }
        take(&mut self.iterations, later.iterations);
        take(&mut self.phase, later.phase);
        take(&mut self.effort, later.effort);
        take(&mut self.requested_model, later.requested_model);
        take(&mut self.advisor_model, later.advisor_model);
        let (a, b) = (&mut self.attribution, later.attribution);
        take(&mut a.agent, b.agent);
        take(&mut a.skill, b.skill);
        take(&mut a.plugin, b.plugin);
        take(&mut a.mcp_server, b.mcp_server);
        take(&mut a.mcp_tool, b.mcp_tool);
        take(&mut self.thinking_ms, later.thinking_ms);
        take(&mut self.first_token_ms, later.first_token_ms);
        take(&mut self.api_error, later.api_error);
        take(&mut self.cache_miss, later.cache_miss);
        take(&mut self.refusal, later.refusal);
        take(&mut self.permission, later.permission);
        take(&mut self.request, later.request);
        take(&mut self.request_id, later.request_id);
        take(&mut self.turn_origin, later.turn_origin);
        take(&mut self.quota, later.quota);
    }
}

/// Which agent, skill, plugin or MCP tool the work was done for.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Attribution {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_server: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_tool: Option<String>,
}

impl Attribution {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ApiError {
    /// The harness's label (`rate_limit`, `invalid_request`).
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
}

impl Message {
    /// Prose text of the message, blocks joined by blank lines.
    pub fn text(&self) -> String {
        let parts: Vec<&str> = self.blocks.iter().filter_map(|b| b.text()).collect();
        parts.join("\n\n")
    }
}

/// A user message written by neither the harness nor another model.
pub fn is_prompt(role: Role, kind: Option<PromptKind>) -> bool {
    role == Role::User && !matches!(kind, Some(PromptKind::Harness | PromptKind::Model))
}

#[cfg(test)]
mod tests;
