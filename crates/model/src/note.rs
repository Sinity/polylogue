//! Notes and assertions: everything people and agents add to the archive.
//! One store with revisions and relations. A revision is
//! a new note in the same family (`first_note`); the current revision of a
//! family is its newest note that is not withdrawn.

use crate::{Reference, ReferenceError, SessionId, Timestamp};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

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
    pub fn parse(text: &str) -> Result<Self, ReferenceError> {
        match text.split_once(':') {
            None if text == "operator" => Ok(Self::Operator),
            None if text == "agent" => Ok(Self::Agent(None)),
            Some(("agent", s)) => Ok(Self::Agent(Some(
                SessionId::parse(s).map_err(|e| ReferenceError::Id(text.to_owned(), e))?,
            ))),
            Some(("script", name)) if !name.is_empty() => Ok(Self::Script(name.to_owned())),
            _ => Err(ReferenceError::UnknownKind(text.to_owned())),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
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
}

/// How one note relates to something else.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NewNote {
    pub kind: NoteKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    pub author: Author,
    pub targets: Vec<Reference>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relations: Vec<(Relation, String)>,
    /// The note this one revises; the new note joins its family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revises: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<Timestamp>,
}

/// A stored note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
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
    pub targets: Vec<Reference>,
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
