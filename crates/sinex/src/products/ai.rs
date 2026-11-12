//! AI facts exposed to Sinex readers independently of the local row transport.
use crate::kinds::{GitIdentity, Objective, RecordRef, TerminalIdentity};
use polylogue_model::{
    Block, Message, MessageId, Production, PromptKind, Role, Session, SessionId, SessionKind,
    TimeEvidence, Timestamp, Usage, WrittenBy,
};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
pub(super) struct MessageFields<'a> {
    message: &'a MessageId,
    session: &'a SessionId,
    parent: &'a Option<MessageId>,
    role: Role,
    written_by: WrittenBy,
    prompt_kind: Option<PromptKind>,
    at: Option<Timestamp>,
    at_evidence: TimeEvidence,
    model: &'a Option<String>,
    text: String,
    blocks: &'a [Block],
    usage: &'a Usage,
    stop_reason: &'a Option<String>,
    reported_cost_usd: &'a Option<serde_json::Number>,
    duration_ms: Option<u64>,
    production: &'a Production,
    attrs: &'a serde_json::Map<String, serde_json::Value>,
}

impl<'a> From<&'a Message> for MessageFields<'a> {
    fn from(message: &'a Message) -> Self {
        Self {
            message: &message.id,
            session: &message.session,
            parent: &message.parent,
            role: message.role,
            written_by: WrittenBy::of(message),
            prompt_kind: message.prompt_kind,
            at: message.at,
            at_evidence: message.time_evidence(),
            model: &message.model,
            text: message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            blocks: &message.blocks,
            usage: &message.usage,
            stop_reason: &message.stop_reason,
            reported_cost_usd: &message.reported_cost_usd,
            duration_ms: message.duration_ms,
            production: &message.production,
            attrs: &message.attrs,
        }
    }
}

#[derive(Serialize, JsonSchema)]
pub(super) struct SessionFields<'a> {
    session: &'a SessionId,
    origin: &'a str,
    kind: SessionKind,
    title: Option<&'a str>,
    cwd: &'a [String],
    repo: &'a Option<String>,
    branch: &'a Option<String>,
    git_commit: &'a Option<String>,
    model_provider: &'a Option<String>,
    instructions: &'a Option<String>,
    tool_version: &'a Option<String>,
    mode: &'a Option<String>,
    started: Option<Timestamp>,
    started_evidence: TimeEvidence,
    machine: Option<&'a str>,
    objectives: Vec<Objective>,
    git_identity: Option<GitIdentity>,
    terminal: Option<TerminalIdentity>,
    boot_id: Option<String>,
    native_sources: Vec<RecordRef>,
    native_unavailable: Vec<String>,
}

impl<'a> From<&'a Session> for SessionFields<'a> {
    fn from(session: &'a Session) -> Self {
        Self {
            session: &session.id,
            origin: session.id.origin().as_str(),
            kind: session.kind,
            title: session.title.as_ref().map(|title| title.text.as_str()),
            cwd: &session.cwd,
            repo: &session.repo,
            branch: &session.git_branch,
            git_commit: &session.git_commit,
            model_provider: &session.model_provider,
            instructions: &session.instructions,
            tool_version: &session.tool_version,
            mode: &session.mode,
            started: session.started,
            started_evidence: session.time_evidence(),
            machine: None,
            objectives: Vec::new(),
            git_identity: None,
            terminal: None,
            boot_id: None,
            native_sources: Vec::new(),
            native_unavailable: Vec::new(),
        }
    }
}

pub(super) fn schema(product: &str) -> Option<serde_json::Value> {
    match product {
        "message" => Some(serde_json::to_value(schemars::schema_for!(MessageFields)).unwrap()),
        "session" => Some(serde_json::to_value(schemars::schema_for!(SessionFields)).unwrap()),
        "tool_call" => Some(serde_json::to_value(schemars::schema_for!(ToolFields)).unwrap()),
        _ => None,
    }
}

#[derive(Serialize, JsonSchema)]
pub(super) struct ToolFields<'a> {
    pub record_key: String,
    pub source: &'a polylogue_store::primary::Reading,
    #[serde(flatten)]
    pub call: &'a polylogue_store::primary::products::ToolCall,
    pub context: crate::native::Facts,
}
