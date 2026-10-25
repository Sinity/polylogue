//! Stored provider text that is not a conversation turn.
use crate::{Block, DocumentId, Json, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentPurpose {
    Instruction,
    Context,
    Memory,
    Transcript,
}

stored_names!(DocumentPurpose {
    Instruction => "instruction",
    Context => "context",
    Memory => "memory",
    Transcript => "transcript",
});

/// A purpose, project creator or upload does not by itself establish a writer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentAuthorship {
    #[default]
    Unknown,
    Human,
    Model,
    Harness,
}

stored_names!(DocumentAuthorship {
    Unknown => "unknown",
    Human => "human",
    Model => "model",
    Harness => "harness",
});

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub purpose: DocumentPurpose,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The provider's time for the source record, not an inferred writing time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_time: Option<Timestamp>,
    #[serde(default)]
    pub authorship: DocumentAuthorship,
    pub blocks: Vec<Block>,
    /// Source descriptors. Projected text is held only in blocks.
    #[serde(default, skip_serializing_if = "Json::is_null")]
    pub metadata: Json,
}

impl Document {
    pub fn new(id: DocumentId, purpose: DocumentPurpose) -> Self {
        Self {
            id,
            purpose,
            title: None,
            source_time: None,
            authorship: DocumentAuthorship::Unknown,
            blocks: Vec::new(),
            metadata: Json::Null,
        }
    }
}
