//! Source body values. Presence/rank facts are supplied separately by source mappers.
use crate::identity::{BlobHash, BlockId, BlockSlot, IdError, MessageId, Origin, SessionId};
use crate::measures::{ErrorReason, ToolOutcome, UnknownReason};
use crate::time::ProviderTime;
use crate::types::Usd;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    #[error("{0} must not be empty")]
    Empty(&'static str),
    #[error(transparent)]
    Identity(#[from] IdError),
    #[error("block belongs to a different message")]
    BlockMessage,
    #[error("duplicate block ID or position")]
    DuplicateBlock,
    #[error("message belongs to a different session")]
    MessageSession,
    #[error("duplicate message ID")]
    DuplicateMessage,
    #[error("duplicate session ID")]
    DuplicateSession,
    #[error("link parent and child must differ")]
    SelfLink,
    #[error("link kind and how disagree")]
    LinkKind,
}
fn nonempty<'de, D: Deserializer<'de>>(de: D) -> Result<String, D::Error> {
    let value = String::deserialize(de)?;
    if value.is_empty() {
        return Err(serde::de::Error::custom("value must not be empty"));
    }
    Ok(value)
}
fn present_json<'de, D: Deserializer<'de>>(de: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(de).map(Some)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionKind {
    Main,
    /// Has an incoming `subagent` link. Continuations are links, not a kind.
    Subagent,
    Chat,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    pub origin: Origin,
    #[serde(deserialize_with = "nonempty")]
    pub native_id: String,
    pub kind: SessionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_at: Option<ProviderTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<ProviderTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_activity_at: Option<ProviderTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// The repository's root commit hash: its identity across clones and worktrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_root: Option<String>,
    /// The worktree path, when it differs from the repository's main checkout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkout: Option<String>,
    /// The provider's grouping (ChatGPT project or custom GPT, claude.ai project).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<Collection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_version: Option<String>,
    /// Agent team (Claude Code `teamName`) and the session's name in it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    /// Host the session ran on, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
    /// Model name to the number of messages it produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<BTreeMap<String, u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporary: Option<bool>,
}

/// A provider's grouping of conversations.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Collection {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Session {
    /// Required source identity is supplied explicitly, never decoded from `id`.
    pub fn new(
        id: SessionId,
        origin: Origin,
        native_id: impl Into<String>,
        kind: SessionKind,
    ) -> Result<Self, ModelError> {
        let native_id = native_id.into();
        if native_id.is_empty() {
            return Err(ModelError::Empty("native_id"));
        }
        Ok(Self {
            id,
            origin,
            native_id,
            kind,
            title: None,
            title_at: None,
            started_at: None,
            last_activity_at: None,
            cwd: None,
            repo: None,
            branch: None,
            commit: None,
            repo_root: None,
            checkout: None,
            collection: None,
            harness: None,
            harness_version: None,
            team: None,
            agent_name: None,
            machine: None,
            models: None,
            temporary: None,
        })
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.native_id.is_empty() {
            return Err(ModelError::Empty("native_id"));
        }
        Ok(())
    }
}

/// The message's role: the provider's role mapped by its section's table to
/// one of four values (`spec/model.md`, Message); the raw value is kept in
/// `attrs` when it differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// Who actually produced a message's content, independent of its role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrittenBy {
    Human,
    Model,
    Harness,
    /// A record matching no form the provider's section lists; counted per
    /// input under unrecognized (`spec/model.md`, Message).
    Unknown,
}

/// For a human message: typed prose, or a command (slash command, `!` shell
/// input, `#` memory input). "Human prompt" everywhere means `Prose`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptKind {
    Prose,
    Command,
}

/// Tokens as the provider reported them. Each key only when reported.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
    /// The part of `cache_write` written with a 5-minute lifetime, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_5m: Option<u64>,
    /// The part of `cache_write` written with a 1-hour lifetime, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cost_nusd: Option<Usd>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdKind {
    Provider,
    Provisional,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CopyOrigin {
    Known,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub id_kind: IdKind,
    /// Filled only from complete source evidence, never normalized content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_hash: Option<BlobHash>,
    pub role: Role,
    pub written_by: WrittenBy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_kind: Option<PromptKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<ProviderTime>,
    /// For a message assembled from several timed records: the last record's time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<ProviderTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<MessageId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_active_path: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_progress: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy_origin: Option<CopyOrigin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counted_as: Option<MessageId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<MessageId>,
    /// For a message one agent sent to another: the sending session when
    /// resolved, and the sender's name as the provider wrote it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_session: Option<SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Provider fields kept as given that have no field of their own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "present_json")]
    pub attrs: Option<Value>,
    #[serde(default)]
    pub blocks: Vec<Block>,
}

impl Message {
    /// Defaults here are unknown body values, not source-presence evidence.
    pub fn new(id: MessageId, id_kind: IdKind, role: Role, written_by: WrittenBy) -> Self {
        Self {
            id,
            id_kind,
            version_hash: None,
            role,
            written_by,
            prompt_kind: None,
            model: None,
            at: None,
            completed_at: None,
            parent: None,
            on_active_path: None,
            in_progress: None,
            copy_origin: None,
            counted_as: None,
            inherited_from: None,
            from_session: None,
            from_name: None,
            usage: None,
            stop_reason: None,
            duration_ms: None,
            attrs: None,
            blocks: Vec::new(),
        }
    }
    /// The source mapper owns the separate ordinary and reasoning counters.
    pub fn push(&mut self, slot: BlockSlot, content: BlockContent) -> Result<&BlockId, ModelError> {
        let id = match slot {
            BlockSlot::Item(n) => BlockId::new(self.id.clone(), n),
            BlockSlot::Reasoning(n) => BlockId::reasoning(self.id.clone(), n),
            BlockSlot::Attachment(native) => BlockId::attachment(self.id.clone(), &native)?,
        };
        self.insert_block(Block { id, content })
    }
    pub fn insert_block(&mut self, block: Block) -> Result<&BlockId, ModelError> {
        if block.id.message() != &self.id {
            return Err(ModelError::BlockMessage);
        }
        if self
            .blocks
            .iter()
            .any(|b| b.id.as_str() == block.id.as_str() || b.id.slot() == block.id.slot())
        {
            return Err(ModelError::DuplicateBlock);
        }
        self.blocks.push(block);
        Ok(&self.blocks[self.blocks.len() - 1].id)
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        let mut ids = std::collections::HashSet::new();
        let mut slots = std::collections::HashSet::new();
        for block in &self.blocks {
            if block.id.message() != &self.id {
                return Err(ModelError::BlockMessage);
            }
            if !ids.insert(block.id.as_str()) || !slots.insert(block.id.slot()) {
                return Err(ModelError::DuplicateBlock);
            }
        }
        if self
            .parent
            .as_ref()
            .is_some_and(|id| id.session() != self.id.session())
        {
            return Err(ModelError::MessageSession);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    #[serde(flatten)]
    pub content: BlockContent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockContent {
    Text {
        text: String,
    },
    Thinking(Thinking),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Image(Attachment),
    File(Attachment),
    Other {
        #[serde(rename = "type")]
        type_name: String,
        data: Value,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thinking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted: Option<bool>,
    /// The encrypted payload as stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// Arguments exactly as given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "present_json")]
    pub input: Option<Value>,
    /// For shell-like tools, the command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

/// Large output lives in a side file stored as a blob.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolOutput {
    Text(String),
    Blob(BlobHash),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "ToolResultWire", try_from = "ToolResultWire")]
pub struct ToolResult {
    pub call_id: Option<String>,
    pub outcome: ToolOutcome,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<u64>,
    pub output: Option<ToolOutput>,
}

/// The export shape of a tool result: `outcome` plus `outcome_reason`, and
/// `text` or `blob`.
#[derive(Serialize, Deserialize)]
struct ToolResultWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    call_id: Option<String>,
    outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    outcome_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exit_code: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    blob: Option<BlobHash>,
}

impl From<ToolResult> for ToolResultWire {
    fn from(r: ToolResult) -> Self {
        let (text, blob) = match r.output {
            Some(ToolOutput::Text(t)) => (Some(t), None),
            Some(ToolOutput::Blob(h)) => (None, Some(h)),
            None => (None, None),
        };
        let outcome = r.outcome.as_str().to_owned();
        let outcome_reason = match r.outcome {
            ToolOutcome::Unknown { reason } => Some(reason.as_str().to_owned()),
            ToolOutcome::Error { reason } => reason.map(|r| r.code()),
            _ => None,
        };
        Self {
            call_id: r.call_id,
            outcome,
            outcome_reason,
            exit_code: r.exit_code,
            duration_ms: r.duration_ms,
            text,
            blob,
        }
    }
}

impl TryFrom<ToolResultWire> for ToolResult {
    type Error = String;

    fn try_from(w: ToolResultWire) -> Result<Self, String> {
        let outcome = match (w.outcome.as_str(), w.outcome_reason) {
            ("ok", None) => ToolOutcome::Ok,
            ("error", reason) => ToolOutcome::Error {
                reason: reason.as_deref().map(ErrorReason::parse),
            },
            ("unknown", Some(reason)) => ToolOutcome::Unknown {
                reason: UnknownReason::parse(&reason),
            },
            ("unknown", None) => return Err("outcome \"unknown\" needs outcome_reason".into()),
            (o, Some(_)) => {
                return Err(format!(
                    "outcome_reason is only allowed with \"unknown\" or \"error\", not {o:?}"
                ));
            }
            (o, None) => return Err(format!("unknown outcome {o:?}")),
        };
        let output = match (w.text, w.blob) {
            (Some(t), None) => Some(ToolOutput::Text(t)),
            (None, Some(h)) => Some(ToolOutput::Blob(h)),
            (None, None) => None,
            (Some(_), Some(_)) => return Err("a tool result has text or blob, not both".into()),
        };
        Ok(Self {
            call_id: w.call_id,
            outcome,
            exit_code: w.exit_code,
            duration_ms: w.duration_ms,
            output,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentDirection {
    Input,
    Output,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    #[serde(default)]
    pub direction: AttachmentDirection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produced_by: Option<BlockId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "present_json")]
    pub attrs: Option<Value>,
    /// Bytes in the blob store; absent when they were never obtained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<BlobHash>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// Started by a tool call in the parent.
    Subagent,
    /// The conversation continued in a new session.
    Continuation,
    /// Text in the child names the parent.
    Reference,
}

/// How the child got the parent's context (`model.md`, Link).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkHow {
    // subagent
    Fresh,
    Copied,
    Summarized,
    // continuation
    Resume,
    Fork,
    AfterCompaction,
    // reference
    Briefing,
    Ping,
    Mention,
}

/// A relationship between two sessions, only ever from evidence in the data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn_evidence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_of: Option<BlockId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_use: Option<ResultUse>,
    pub kind: LinkKind,
    pub how: LinkHow,
    pub parent: SessionId,
    pub child: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_point: Option<MessageId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawned_by: Option<BlockId>,
    /// The input field that established the link, e.g. `meta.toolUseId`.
    pub evidence: String,
    /// Provider facts about the relationship as given (agent type, role, depth, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "present_json")]
    pub attrs: Option<Value>,
}

/// Something in the timeline that is not a message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    /// `compaction`, `model_switch`, `permission`, `cwd_change`, `hook`, …
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<ProviderTime>,
    /// The message this event sits after, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_message: Option<MessageId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_saw: Option<bool>,
    /// The provider's payload as given.
    pub data: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultUse {
    Returned,
    Referenced,
    Unknown,
}

impl Link {
    pub fn new(
        kind: LinkKind,
        how: LinkHow,
        parent: SessionId,
        child: SessionId,
        evidence: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let link = Self {
            kind,
            how,
            parent,
            child,
            evidence: evidence.into(),
            branch_point: None,
            spawned_by: None,
            attrs: None,
            spawn_evidence: None,
            retry_of: None,
            result_use: None,
        };
        link.validate()?;
        Ok(link)
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.parent == self.child {
            return Err(ModelError::SelfLink);
        }
        if self.evidence.is_empty() {
            return Err(ModelError::Empty("evidence"));
        }
        if !matches!(
            (self.kind, self.how),
            (
                LinkKind::Subagent,
                LinkHow::Fresh | LinkHow::Copied | LinkHow::Summarized
            ) | (
                LinkKind::Continuation,
                LinkHow::Resume | LinkHow::Fork | LinkHow::AfterCompaction
            ) | (
                LinkKind::Reference,
                LinkHow::Briefing | LinkHow::Ping | LinkHow::Mention
            )
        ) {
            return Err(ModelError::LinkKind);
        }
        if self
            .branch_point
            .as_ref()
            .is_some_and(|id| id.session() != &self.parent)
            || self
                .spawned_by
                .as_ref()
                .is_some_and(|id| id.message().session() != &self.parent)
            || self
                .retry_of
                .as_ref()
                .is_some_and(|id| id.message().session() != &self.parent)
        {
            return Err(ModelError::MessageSession);
        }
        Ok(())
    }
}

/// A record type or field a parser did not understand, counted per input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unrecognized {
    /// `record:<type>` or `field:<json path>`.
    pub what: String,
    pub count: u64,
    /// A known record type kept only as a count, by design (e.g. `progress`).
    pub counted_only: bool,
}

/// One session as a parser produced it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedSession {
    pub session: Session,
    pub messages: Vec<Message>,
    pub events: Vec<Event>,
}

/// Everything parsed from one input.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parsed {
    pub sessions: Vec<ParsedSession>,
    pub links: Vec<Link>,
    pub unrecognized: Vec<Unrecognized>,
}

impl ParsedSession {
    pub fn validate(&self) -> Result<(), ModelError> {
        self.session.validate()?;
        let mut ids = std::collections::HashSet::new();
        for message in &self.messages {
            if message.id.session() != &self.session.id {
                return Err(ModelError::MessageSession);
            }
            if !ids.insert(message.id.as_str()) {
                return Err(ModelError::DuplicateMessage);
            }
            message.validate()?;
        }
        for event in &self.events {
            if event.kind.is_empty() {
                return Err(ModelError::Empty("event kind"));
            }
            if event
                .after_message
                .as_ref()
                .is_some_and(|id| id.session() != &self.session.id)
            {
                return Err(ModelError::MessageSession);
            }
        }
        Ok(())
    }
}
impl Parsed {
    pub fn validate(&self) -> Result<(), ModelError> {
        let mut ids = std::collections::HashSet::new();
        for session in &self.sessions {
            if !ids.insert(session.session.id.as_str()) {
                return Err(ModelError::DuplicateSession);
            }
            session.validate()?;
        }
        for link in &self.links {
            link.validate()?;
        }
        Ok(())
    }
}
