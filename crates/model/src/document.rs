//! Stored provider text that is not a conversation turn.
use crate::{Block, DocumentId, Json, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Document {
    pub id: DocumentId,
    pub purpose: DocumentPurpose,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The provider's time for the source record, not an inferred writing time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_time: Option<Timestamp>,
    #[serde(default = "crate::time::unknown_quality")]
    pub source_time_quality: crate::TimeQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_time_precision: Option<crate::Precision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_time_until: Option<Timestamp>,
    #[serde(default)]
    pub authorship: DocumentAuthorship,
    pub blocks: Vec<Block>,
    /// Source descriptors. Projected text is held only in blocks.
    #[serde(default, skip_serializing_if = "Json::is_null")]
    pub metadata: Json,
}

impl Document {
    pub fn time_evidence(&self) -> crate::TimeEvidence {
        crate::TimeEvidence {
            quality: self.source_time_quality,
            precision: self.source_time_precision,
            until: self.source_time_until,
        }
    }

    pub fn set_time(
        &mut self,
        at: Option<Timestamp>,
        evidence: crate::TimeEvidence,
    ) -> Result<(), crate::TimeEvidenceError> {
        evidence.validate(at)?;
        self.source_time = at;
        self.source_time_quality = evidence.quality;
        self.source_time_precision = evidence.precision;
        self.source_time_until = evidence.until;
        Ok(())
    }

    pub fn new(id: DocumentId, purpose: DocumentPurpose) -> Self {
        Self {
            id,
            purpose,
            title: None,
            source_time: None,
            source_time_quality: crate::TimeQuality::Unknown,
            source_time_precision: None,
            source_time_until: None,
            authorship: DocumentAuthorship::Unknown,
            blocks: Vec::new(),
            metadata: Json::Null,
        }
    }
}
