//! Notes and assertions: everything people and agents add to the archive.
//! One store with revisions and relations. A revision is
//! a new note in the same family (`first_note`); the current revision of a
//! family is its newest note that is not withdrawn.

#[cfg(test)]
use crate::ReferenceError;
use crate::{Author, Reference, Timestamp};
use serde::{Deserialize, Serialize};

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
