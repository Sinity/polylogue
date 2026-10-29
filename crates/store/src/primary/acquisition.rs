//! Private acquisition coordinates. They preserve custody and compare actual
//! predecessors; they never assign a source revision or an admission receipt.
use super::{CapturedInput, PendingCapture, PreparedBody};
use crate::{Result, StoreError};
use polylogue_model::batch::ParserCheckpoint;
use serde::{Deserialize, Serialize};

/// Receiver-minted IO evidence. Native field observation does not identify
/// Polylogue as its producer; producer proof is a separate post-emit receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionRoute {
    NativeCapture,
    ExplicitImport,
}

/// Original body acquisition, before overlap trimming and final source chain
/// assignment. Historical absence is unknown. Payload declarations cannot mint it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionReceipt {
    pub version: u32,
    pub route: AcquisitionRoute,
    pub source: SourceSelector,
    pub range: RawRange,
    pub hash: polylogue_model::BlobHash,
    pub observed_at: polylogue_model::Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<super::SourceGeneration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_predecessor: Option<String>,
}
impl AcquisitionReceipt {
    pub const VERSION: u32 = 1;

    pub fn validate_for(&self, input: &crate::InputKey) -> Result<()> {
        let invalid = || {
            StoreError::Invalid(
                "acquisition evidence does not match its generation or input".into(),
            )
        };
        if self.version != Self::VERSION || self.source.origin != input.format {
            return Err(invalid());
        }
        self.range.length()?;
        let Some(generation) = &self.generation else {
            return if self.raw_total.is_none() && self.raw_predecessor.is_none() {
                Ok(())
            } else {
                Err(invalid())
            };
        };
        generation.validate()?;
        let stream = generation.stream();
        let window = generation.window();
        if stream.source != self.source
            || stream.input.id != input.id
            || stream.input.format != input.format
            || self.range.start < window.start
            || self.range.end > window.end
            || self
                .raw_total
                .is_some_and(|end| end != window.end || end != self.range.end)
            || self
                .raw_predecessor
                .as_ref()
                .is_some_and(|body| body.is_empty() || self.range.start == window.start)
            || matches!(generation,
                super::SourceGeneration::File { stream: super::RawStream { role: super::RawStreamRole::ProviderBytes, .. }, observed, .. }
                if self.raw_total.is_some_and(|end| end != observed.size))
        {
            return Err(invalid());
        }
        Ok(())
    }
}

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
pub struct SourceSubject {
    pub input: String,
    pub observation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceBasis {
    /// Missing authority permits private custody, not source-coordinate claims.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary: Option<super::SourceBoundary>,
    /// A native-ID alias resolved strictly within the pre-acquisition cut.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved: Option<SourceSubject>,
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
    /// Receiver constraint on new custody only. Exact retained retries keep
    /// their original disposition; this cannot promote unresolved work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_disposition: Option<CaptureDisposition>,
}
impl CapturePlan {
    pub fn validate(&self) -> Result<()> {
        if self.intent.is_empty() || self.count == 0 {
            return Err(StoreError::Invalid(
                "an acquisition group needs its content intent and members".into(),
            ));
        }
        if self.required_disposition == Some(CaptureDisposition::Prepared) {
            return Err(StoreError::Invalid(
                "an acquisition constraint cannot require acceptance".into(),
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
    /// Latest durable acquisition of any kind, including observation-only records.
    pub head: Option<PendingCapture>,
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

/// Durable acquisition preparation is distinct from committed source semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CaptureReceipt {
    Accepted { write: super::PreparedWrite },
    UnresolvedBasisConflict { custody: RawCustodyReceipt },
}

/// Existing group identity and original count. Bodies are resolved one page at
/// a time through group headers and capture_body, never by provider paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawCustodyReceipt {
    pub identity: String,
    pub count: u64,
}

/// A private custody disposition cannot acknowledge a byte cursor or revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CaptureDisposition {
    Prepared,
    UnresolvedBasisConflict,
}

/// One bounded page of retained original group members. Exhaustion says
/// nothing about provider EOF or missing members already applied elsewhere.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PendingGroupPage {
    /// None means no retained group disposition is known, not accepted EOF.
    pub disposition: Option<CaptureDisposition>,
    pub headers: Vec<PendingCapture>,
    /// Original ordinal, exclusive, for the next page.
    pub next: Option<u64>,
}

/// Enumerates owned encoded bodies one member at a time. The SDK stages every
/// body and complete private facts before atomically publishing the group.
pub type CaptureEmitter<'a> = dyn FnMut(CapturedInput, PreparedBody) -> Result<()> + 'a;
pub type CaptureMembers<'a> = dyn FnMut(&mut CaptureEmitter<'_>) -> Result<()> + 'a;
