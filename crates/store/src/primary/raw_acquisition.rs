//! Physical acquisition progress is independent of source acceptance and parsing.
use super::{CaptureDisposition, RawRange, SourceSelector};
use crate::{InputKey, Result, StoreError};
use polylogue_model::BlobHash;
use serde::{Deserialize, Serialize};
mod input;

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
pub enum SourceGeneration {
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

/// An actual retained body identifies one immutable predecessor-linked chain.
/// Without it the owner must prove there is exactly one original root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawAcquisitionQuery {
    pub generation: SourceGeneration,
    pub anchor: Option<String>,
}
impl RawAcquisitionQuery {
    pub fn validate(&self) -> Result<()> {
        self.generation.validate()?;
        if self.anchor.as_ref().is_some_and(String::is_empty) {
            return Err(StoreError::Invalid(
                "raw acquisition anchor is empty".into(),
            ));
        }
        Ok(())
    }
}

impl SourceGeneration {
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
            return Err(StoreError::Invalid(
                "raw acquisition stream identity is incomplete".into(),
            ));
        }
        match self {
            Self::File {
                observed, stream, ..
            } => {
                if observed.modified.nanos >= 1_000_000_000
                    || observed.changed.nanos >= 1_000_000_000
                {
                    return Err(StoreError::Invalid(
                        "provider file time has invalid nanoseconds".into(),
                    ));
                }
                match &stream.role {
                    RawStreamRole::ProviderBytes if self.window().end <= observed.size => {}
                    RawStreamRole::DecodedMember { codec, .. } if !codec.is_empty() => {}
                    _ => {
                        return Err(StoreError::Invalid(
                            "file generation has an incompatible stream window or role".into(),
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
                    return Err(StoreError::Invalid(
                        "consistent-read custody nonce is missing".into(),
                    ));
                }
                match &stream.role {
                    RawStreamRole::DatabaseDump => {}
                    RawStreamRole::DatabaseTable { table } if !table.is_empty() => {}
                    _ => {
                        return Err(StoreError::Invalid(
                            "database generation has an incompatible stream role".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// The owner supplies a range only after its exact body reference is durably
/// committed and its decoded hash/length agree with the immutable header.
#[derive(Debug, Clone)]
pub struct RawBodyRange {
    pub generation: SourceGeneration,
    pub body: String,
    pub predecessor: Option<String>,
    pub disposition: Option<CaptureDisposition>,
    pub range: RawRange,
    pub hash: BlobHash,
    pub decoded_length: u64,
    pub known_total: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawAcquisitionCursor {
    generation: SourceGeneration,
    range: RawRange,
    known_total: Option<u64>,
    last_body: String,
    root_body: String,
    #[serde(default)]
    disposition: Option<CaptureDisposition>,
}

impl RawAcquisitionCursor {
    pub fn generation(&self) -> &SourceGeneration {
        &self.generation
    }
    pub fn range(&self) -> RawRange {
        self.range
    }
    pub fn known_total(&self) -> Option<u64> {
        self.known_total
    }
    pub fn root_body(&self) -> &str {
        &self.root_body
    }
    pub fn last_body(&self) -> &str {
        &self.last_body
    }
    /// Prepared describes private custody only, never logical source acceptance.
    pub fn disposition(&self) -> Option<CaptureDisposition> {
        self.disposition
    }
    pub fn is_complete_window(&self) -> bool {
        let window = self.generation.window();
        self.range == window && self.known_total == Some(window.end)
    }
}

/// One bounded verifier serves catalog and journal adapters. Feed owner-verified
/// body ranges in stream order; a hole, changed generation or unproved overlap
/// poisons the result instead of manufacturing a maximum-end cursor.
pub struct RawAcquisitionProof {
    generation: SourceGeneration,
    range: RawRange,
    total: Option<u64>,
    disposition: Option<CaptureDisposition>,
    root_body: Option<String>,
    last: Option<(RawRange, BlobHash, String, Option<String>)>,
    valid: bool,
}

impl RawAcquisitionProof {
    pub fn new(generation: SourceGeneration) -> Result<Self> {
        generation.validate()?;
        let start = generation.window().start;
        Ok(Self {
            generation,
            range: RawRange { start, end: start },
            total: None,
            disposition: Some(CaptureDisposition::Prepared),
            root_body: None,
            last: None,
            valid: true,
        })
    }
    pub fn push(&mut self, body: RawBodyRange) -> Result<()> {
        if body.body.is_empty() || body.range.length()? != body.decoded_length {
            return Err(StoreError::Invalid(
                "raw custody body reference or decoded range is invalid".into(),
            ));
        }
        self.disposition = match (self.disposition, body.disposition) {
            (Some(CaptureDisposition::UnresolvedBasisConflict), _)
            | (_, Some(CaptureDisposition::UnresolvedBasisConflict)) => {
                Some(CaptureDisposition::UnresolvedBasisConflict)
            }
            (None, _) | (_, None) => None,
            _ => Some(CaptureDisposition::Prepared),
        };
        if body.generation != self.generation
            || body.range.start < self.generation.window().start
            || body.range.end > self.generation.window().end
            || body
                .known_total
                .is_some_and(|end| end != self.generation.window().end)
            || matches!(&self.generation,
                SourceGeneration::File { stream: RawStream { role: RawStreamRole::ProviderBytes, .. }, observed, .. }
                if body.known_total.is_some_and(|end| end != observed.size))
            || self
                .total
                .zip(body.known_total)
                .is_some_and(|(a, b)| a != b)
        {
            self.valid = false;
        }
        if let Some((range, hash, previous_body, predecessor)) = &self.last {
            if *range == body.range && *hash == body.hash {
                // Equal bytes under another original acquisition are a separate
                // chain. A generation-only lookup cannot choose between them.
                if body.range.start == self.generation.window().start
                    && body.predecessor.is_none()
                    && *previous_body != body.body
                {
                    self.valid = false;
                }
                if *predecessor != body.predecessor {
                    self.valid = false;
                }
                if self.valid {
                    self.total = body.known_total.or(self.total);
                    self.last = Some((body.range, body.hash, body.body, body.predecessor));
                }
                return Ok(());
            }
            if body.predecessor.as_deref() != Some(previous_body.as_str()) {
                self.valid = false;
            }
        } else if body.predecessor.is_some() {
            self.valid = false;
        }
        if body.range.start != self.range.end {
            self.valid = false;
        }
        if !self.valid {
            return Ok(());
        }
        if self.root_body.is_none() {
            self.root_body = Some(body.body.clone());
        }
        self.range.end = body.range.end;
        self.total = body.known_total.or(self.total);
        self.last = Some((body.range, body.hash, body.body, body.predecessor));
        Ok(())
    }
    pub fn finish(self) -> Option<RawAcquisitionCursor> {
        let (_, _, last_body, _) = self.last?;
        let root_body = self.root_body?;
        self.valid.then_some(RawAcquisitionCursor {
            generation: self.generation,
            range: self.range,
            known_total: self.total,
            last_body,
            root_body,
            disposition: self.disposition,
        })
    }
}

#[cfg(test)]
mod tests;
