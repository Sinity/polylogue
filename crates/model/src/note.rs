use crate::{Json, MessageId, SessionId, Timestamp};
use serde::{Deserialize, Serialize};

/// What a note is attached to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum Target {
    Session(SessionId),
    Message(MessageId),
    /// A tool call, by its session and call ID.
    ToolCall(SessionId, String),
    /// A work-graph node.
    Work(i64),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "session", rename_all = "snake_case")]
pub enum Author {
    Operator,
    /// An agent, by the session it wrote from when known. Agent notes are
    /// shown with their author and never injected into context
    /// automatically.
    Agent(Option<SessionId>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteKind {
    Tag,
    Comment,
    Outcome,
    Correction,
    Judgment,
    Hidden,
    Lesson,
    Handoff,
}

/// A note or assertion. Revisions are new notes that replace an earlier one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: i64,
    pub target: Target,
    pub kind: NoteKind,
    pub author: Author,
    pub at: Timestamp,
    pub body: String,
    #[serde(default, skip_serializing_if = "Json::is_null")]
    pub value: Json,
    /// The note this one revises.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaces: Option<i64>,
}
