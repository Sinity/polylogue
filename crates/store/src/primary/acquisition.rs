//! Private acquisition coordinates. They preserve custody and compare actual
//! predecessors; they never assign a source revision or an admission receipt.
use super::{CapturedInput, PendingCapture, PreparedBody};
use crate::{Result, StoreError};
use polylogue_model::batch::ParserCheckpoint;
use serde::{Deserialize, Serialize};

/// The declared input format and exact native location. Display paths and
/// host labels cannot identify a filesystem observation.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct SourceSelector {
    pub origin: String,
    pub machine: String,
    pub path_bytes: Vec<u8>,
}

/// Ready is identity evidence, independent of network currency. An absent
/// cache row proves absence only after authoritative source-scope coverage.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SourceStatus {
    Ready,
    #[default]
    MissingEvidence,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceBasis {
    pub source: SourceSelector,
    pub status: SourceStatus,
    pub confirmed: Option<String>,
    pub pending: Option<String>,
}

/// Original decoded source interval physically acquired, before overlap trim.
/// None on CapturedInput means observation metadata, not an empty raw source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawRange {
    pub start: u64,
    pub end: u64,
}
impl RawRange {
    pub fn length(self) -> Result<u64> {
        self.end
            .checked_sub(self.start)
            .ok_or_else(|| StoreError::Invalid("reversed acquired source range".into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    Immutable,
    Mutable,
}

/// Immutable packages deduplicate content intent. Mutable identity binds the
/// intent to its ordered durable predecessors under the SDK's custody lock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CapturePlan {
    pub intent: String,
    pub bases: Vec<SourceBasis>,
    pub count: u64,
    pub mode: CaptureMode,
}
impl CapturePlan {
    pub fn validate(&self) -> Result<()> {
        if self.intent.is_empty() || self.count == 0 {
            return Err(StoreError::Invalid(
                "an acquisition group needs its content intent and members".into(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for basis in &self.bases {
            if !seen.insert(&basis.source) {
                return Err(StoreError::Invalid(
                    "an acquisition group repeats a source basis".into(),
                ));
            }
        }
        Ok(())
    }
}

/// The actual control capture remains explicit when newer raw acquisitions
/// have no parser state. The small header has an empty parsed vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PendingCheckpoint {
    pub capture: String,
    pub header: CapturedInput,
    pub checkpoint: ParserCheckpoint,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PendingSource {
    pub status: SourceStatus,
    pub raw: Option<PendingCapture>,
    pub continuation: Option<PendingCheckpoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConfirmedSource {
    pub input: String,
    pub observation: crate::input_rows::ObservationRow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceSnapshot {
    pub basis: SourceBasis,
    pub confirmed: Option<ConfirmedSource>,
    pub pending: PendingSource,
}

/// Enumerates owned encoded bodies one member at a time. The SDK stages every
/// body and complete private facts before atomically publishing the group.
pub type CaptureEmitter<'a> = dyn FnMut(CapturedInput, PreparedBody) -> Result<()> + 'a;
pub type CaptureMembers<'a> = dyn FnMut(&mut CaptureEmitter<'_>) -> Result<()> + 'a;
