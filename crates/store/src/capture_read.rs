//! Where and how a capture read happened: the source it read, the byte range
//! and the observed file or database state, recorded with the extents it
//! stored (`extent.capture_read`) and used to plan the next read of the source.
use crate::OptionExt as _;
use crate::{InputKey, Result, StoreError};
use serde::{Deserialize, Serialize};
mod input;

/// Receiver-minted IO evidence. Native field observation does not identify
/// Polylogue as its producer; producer proof is a separate post-emit receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReadRoute {
    NativeCapture,
    ExplicitImport,
}

/// Original body acquisition, before overlap trimming and final source chain
/// assignment. Historical absence is unknown. Payload declarations cannot mint it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptureRead {
    pub version: u32,
    pub route: ReadRoute,
    pub source: SourceSelector,
    pub range: RawRange,
    pub hash: polylogue_model::BlobHash,
    pub observed_at: polylogue_model::Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<ObservedState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_revision: Option<String>,
    /// The body's content-defined chunks, in order, when it is stored as
    /// chunks; empty when it is one blob.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<crate::Chunk>,
}
impl CaptureRead {
    pub const VERSION: u32 = 1;

    pub fn validate_for(&self, input: &crate::InputKey) -> Result<()> {
        let inconsistent =
            || StoreError::internal("capture read does not match its observed state or input");
        if self.version != Self::VERSION || self.source.origin != input.format {
            return Err(inconsistent());
        }
        self.range.length()?;
        let Some(generation) = &self.generation else {
            return if self.raw_total.is_none() && self.previous_revision.is_none() {
                Ok(())
            } else {
                Err(inconsistent())
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
                .previous_revision
                .as_ref()
                .is_some_and(|body| body.is_empty() || self.range.start == window.start)
            || matches!(generation,
                ObservedState::File { stream: RawStream { role: RawStreamRole::ProviderBytes, .. }, observed, .. }
                if self.raw_total.is_some_and(|end| end != observed.size))
        {
            return Err(inconsistent());
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

/// Ready means the source's identity is known: one stored input observed at
/// the location, with a commit position.
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

/// Original decoded source interval physically acquired, before overlap trim.
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
            .or_internal("reversed captured source range")
    }
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
    pub status: SourceStatus,
    pub confirmed: Option<ConfirmedSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawStream {
    pub source: SourceSelector,
    #[serde(with = "input")]
    #[schemars(with = "input::Identity")]
    pub input: InputKey,
    pub role: RawStreamRole,
    pub namespace: Option<String>,
}

impl PartialEq for RawStream {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.input.id == other.input.id
            && self.input.format == other.input.format
            && self.role == other.role
            && self.namespace == other.namespace
    }
}
impl Eq for RawStream {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum RawStreamRole {
    ProviderBytes,
    DecodedMember {
        codec: String,
        member_index: u64,
        path_bytes: Vec<u8>,
    },
    DatabaseDump,
    DatabaseTable {
        table: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FileTime {
    pub seconds: i64,
    pub nanos: u32,
}

/// Read from the opened provider FD before I/O, then checked against that FD
/// and its current path after acquisition. Encoded scratch-file metadata is not
/// provider identity. Missing or changed evidence yields no generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FileSnapshot {
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub modified: FileTime,
    pub changed: FileTime,
}

/// Chosen once while an actual consistent provider read is held. This nonce is
/// custody-only, not a provider revision, arrival order or data_version. A new
/// transaction, including after restart, always has a new nonce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DatabaseSnapshotToken {
    pub nonce: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedState {
    File {
        stream: RawStream,
        observed: FileSnapshot,
        window: RawRange,
    },
    Database {
        stream: RawStream,
        snapshot: DatabaseSnapshotToken,
        window: RawRange,
    },
}

impl ObservedState {
    pub fn stream(&self) -> &RawStream {
        match self {
            Self::File { stream, .. } | Self::Database { stream, .. } => stream,
        }
    }
    pub fn window(&self) -> RawRange {
        match self {
            Self::File { window, .. } | Self::Database { window, .. } => *window,
        }
    }
    pub fn validate(&self) -> Result<()> {
        let stream = self.stream();
        self.window().length()?;
        if stream.source.origin != stream.input.format
            || stream.input.id.is_empty()
            || stream.source.machine.is_empty()
            || stream.source.path_bytes.is_empty()
            || stream.namespace.as_ref().is_some_and(String::is_empty)
        {
            return Err(StoreError::internal(
                "byte cursor stream identity is incomplete",
            ));
        }
        match self {
            Self::File {
                observed, stream, ..
            } => {
                if observed.modified.nanos >= 1_000_000_000
                    || observed.changed.nanos >= 1_000_000_000
                {
                    return Err(StoreError::internal(
                        "provider file time has invalid nanoseconds",
                    ));
                }
                match &stream.role {
                    RawStreamRole::ProviderBytes if self.window().end <= observed.size => {}
                    RawStreamRole::DecodedMember { codec, .. } if !codec.is_empty() => {}
                    _ => {
                        return Err(StoreError::internal(
                            "observed file state has an incompatible stream window or role",
                        ));
                    }
                }
            }
            Self::Database {
                snapshot, stream, ..
            } => {
                if uuid::Uuid::parse_str(&snapshot.nonce)
                    .ok()
                    .is_none_or(|id| id.is_nil())
                {
                    return Err(StoreError::internal("consistent-read nonce is missing"));
                }
                match &stream.role {
                    RawStreamRole::DatabaseDump => {}
                    RawStreamRole::DatabaseTable { table } if !table.is_empty() => {}
                    _ => {
                        return Err(StoreError::internal(
                            "observed database state has an incompatible stream role",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
