//! Notes and assertions: everything people and agents add to the archive.
//! One store with revisions and relations. A revision is
//! a new note in the same family (`first_note`); the current revision of a
//! family is its newest note that is not withdrawn.

use crate::{IdError, MessageId, SessionId, Timestamp};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// What a note is about. Stored as a reference string (`session:<id>`),
/// so a note can name things the index has not seen yet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Target {
    Session(SessionId),
    Message(MessageId),
    Document(crate::DocumentId),
    /// A tool call, by its session and the call ID the harness gave it.
    ToolCall(SessionId, String),
    /// A work-graph node.
    Work(i64),
    /// A kept query result.
    Result(String),
    /// Another note.
    Note(String),
    /// A repository, by name or URL: lessons and handoffs that apply to
    /// every session working in it.
    Repo(String),
    /// A working directory.
    Cwd(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TargetError {
    #[error(
        "reference {0:?} has no kind prefix (session:, message:, document:, tool:, work:, result:, note:, repo:, cwd:)"
    )]
    NoKind(String),
    #[error("reference {0:?} has an unknown kind")]
    UnknownKind(String),
    #[error("reference {0:?}: {1}")]
    Id(String, IdError),
    #[error("reference {0:?} is incomplete")]
    Incomplete(String),
}

impl Target {
    /// Parses a reference: `session:claude-code:abc`, `message:codex:x`,
    /// `tool:claude-code:abc#toolu_1`, `work:12`, `result:<id>`,
    /// `note:<id>`, `repo:<name>`, `cwd:<path>`.
    pub fn parse(text: &str) -> Result<Self, TargetError> {
        let (kind, rest) = text
            .split_once(':')
            .ok_or_else(|| TargetError::NoKind(text.to_owned()))?;
        let id_err = |e| TargetError::Id(text.to_owned(), e);
        let nonempty = |s: &str| {
            if s.is_empty() {
                Err(TargetError::Incomplete(text.to_owned()))
            } else {
                Ok(s.to_owned())
            }
        };
        Ok(match kind {
            "session" => Self::Session(SessionId::parse(rest).map_err(id_err)?),
            "document" => Self::Document(crate::DocumentId::parse(rest).map_err(id_err)?),
            "message" => Self::Message(MessageId::parse(rest).map_err(id_err)?),
            "tool" => {
                let (session, call) = rest
                    .rsplit_once('#')
                    .ok_or_else(|| TargetError::Incomplete(text.to_owned()))?;
                Self::ToolCall(SessionId::parse(session).map_err(id_err)?, nonempty(call)?)
            }
            "work" => Self::Work(
                rest.parse()
                    .map_err(|_| TargetError::Incomplete(text.to_owned()))?,
            ),
            "result" => Self::Result(nonempty(rest)?),
            "note" => Self::Note(nonempty(rest)?),
            "repo" => Self::Repo(nonempty(rest)?),
            "cwd" => Self::Cwd(nonempty(rest)?),
            _ => return Err(TargetError::UnknownKind(text.to_owned())),
        })
    }

    /// The session this target is in, when it is in one.
    pub fn session(&self) -> Option<&SessionId> {
        match self {
            Self::Session(s) | Self::ToolCall(s, _) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(s) => write!(f, "session:{s}"),
            Self::Document(d) => write!(f, "document:{d}"),
            Self::Message(m) => write!(f, "message:{m}"),
            Self::ToolCall(s, c) => write!(f, "tool:{s}#{c}"),
            Self::Work(n) => write!(f, "work:{n}"),
            Self::Result(r) => write!(f, "result:{r}"),
            Self::Note(n) => write!(f, "note:{n}"),
            Self::Repo(r) => write!(f, "repo:{r}"),
            Self::Cwd(c) => write!(f, "cwd:{c}"),
        }
    }
}

impl Serialize for Target {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_newtype_struct("polylogue_reference", &self.to_string())
    }
}

impl<'de> Deserialize<'de> for Target {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Who wrote a note. Agent notes are shown with their author and are
/// pulled on request only: nothing injects them into context
/// automatically.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Author {
    Operator,
    /// An agent, by the session it wrote from when known.
    Agent(Option<SessionId>),
    /// A program acting for the operator, by name.
    Script(String),
}

impl Author {
    pub fn is_agent(&self) -> bool {
        matches!(self, Self::Agent(_))
    }

    /// The stored form: `operator`, `agent`, `agent:<session>`,
    /// `script:<name>`.
    pub fn parse(text: &str) -> Result<Self, TargetError> {
        match text.split_once(':') {
            None if text == "operator" => Ok(Self::Operator),
            None if text == "agent" => Ok(Self::Agent(None)),
            Some(("agent", s)) => Ok(Self::Agent(Some(
                SessionId::parse(s).map_err(|e| TargetError::Id(text.to_owned(), e))?,
            ))),
            Some(("script", name)) if !name.is_empty() => Ok(Self::Script(name.to_owned())),
            _ => Err(TargetError::UnknownKind(text.to_owned())),
        }
    }
}

impl fmt::Display for Author {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator => f.write_str("operator"),
            Self::Agent(None) => f.write_str("agent"),
            Self::Agent(Some(s)) => write!(f, "agent:{s}"),
            Self::Script(n) => write!(f, "script:{n}"),
        }
    }
}

impl Serialize for Author {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Author {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteKind {
    /// `key` is the tag; `value` optionally carries membership, affinity
    /// and confidence as JSON.
    Tag,
    Note,
    Title,
    Decision,
    Claim,
    Lesson,
    Blocker,
    Question,
    Finding,
    Judgment,
    Outcome,
    Hidden,
    Correction,
    Handoff,
    Goal,
    Problem,
    Forecast,
    SavedQuery,
    Ping,
    /// A watch with a delivery rule.
    Signal,
}

impl NoteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::Note => "note",
            Self::Title => "title",
            Self::Decision => "decision",
            Self::Claim => "claim",
            Self::Lesson => "lesson",
            Self::Blocker => "blocker",
            Self::Question => "question",
            Self::Finding => "finding",
            Self::Judgment => "judgment",
            Self::Outcome => "outcome",
            Self::Hidden => "hidden",
            Self::Correction => "correction",
            Self::Handoff => "handoff",
            Self::Goal => "goal",
            Self::Problem => "problem",
            Self::Forecast => "forecast",
            Self::SavedQuery => "saved_query",
            Self::Ping => "ping",
            Self::Signal => "signal",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(text.to_owned())).ok()
    }

    /// Kinds a session-start brief carries when the operator wrote them.
    pub fn briefed(self) -> bool {
        matches!(
            self,
            Self::Lesson | Self::Handoff | Self::Decision | Self::Blocker | Self::Goal
        )
    }
}

/// How one note relates to something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    Answers,
    Challenges,
    DerivedFrom,
    Blocks,
    Corrects,
    MemberOf,
}

impl Relation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Answers => "answers",
            Self::Challenges => "challenges",
            Self::DerivedFrom => "derived_from",
            Self::Blocks => "blocks",
            Self::Corrects => "corrects",
            Self::MemberOf => "member_of",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(text.to_owned())).ok()
    }
}

/// A note to write. The writing request's ID becomes the note's ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewNote {
    pub kind: NoteKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    pub author: Author,
    pub targets: Vec<Target>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<(Relation, String)>,
    /// The note this one revises; the new note joins its family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revises: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<Timestamp>,
}

/// A stored note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    /// The first note of its revisions; its own ID for a first note.
    pub first_note: String,
    pub kind: NoteKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    pub author: Author,
    pub created_at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<Timestamp>,
    pub targets: Vec<Target>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<(Relation, String)>,
}

impl Note {
    /// Expired notes stay readable but leave briefings.
    pub fn expired_at(&self, now: Timestamp) -> bool {
        self.valid_until.is_some_and(|until| until < now)
    }
}

#[cfg(test)]
mod tests;
