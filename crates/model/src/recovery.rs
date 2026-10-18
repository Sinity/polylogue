//! The record a recovered archive copy carries permanently.
use crate::BlobHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadCoverage {
    Unknown,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryReason {
    UncertainGenerationTail,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurableTier {
    #[serde(rename = "archive.db")]
    Archive,
    #[serde(rename = "embeddings.db")]
    Embeddings,
}
impl DurableTier {
    pub fn file(&self) -> &'static str {
        match self {
            Self::Archive => "archive.db",
            Self::Embeddings => "embeddings.db",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredWal {
    pub tier: DurableTier,
    pub committed_frame: u64,
    pub first_previous_generation_frame: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryRecord {
    pub version: u32,
    pub latest_acknowledged_head: HeadCoverage,
    pub reason: RecoveryReason,
    pub affected: Vec<RecoveredWal>,
    pub source_manifest_blake3: BlobHash,
}
impl RecoveryRecord {
    pub const NOTE: &'static str = "Recovered copy: the last state committed before the damage is unknown. Keep the recovery receipt to compare with other copies; this note never clears.";
    pub fn valid(&self) -> bool {
        self.version == 1
            && self.affected.len() == 1
            && self.affected.iter().all(|wal| {
                wal.committed_frame > 0
                    && wal.committed_frame.checked_add(1)
                        == Some(wal.first_previous_generation_frame)
            })
    }
}
