//! Export callback identity, separate from native byte parser checkpoints.
//! These claims carry no input, receipt, or native observation authority.
use crate::BlobHash;
use serde::{Deserialize, Serialize};

const VERSION: u32 = 1;
const SEED: &[u8] = b"polylogue.export-delivery.prefix.v1";
const LINK: &[u8] = b"polylogue.export-delivery.link.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExportDeliverySource {
    ContainerHeader,
    Item {
        start: u64,
        end: u64,
    },
    Complete {
        length: u64,
        /// Actual item callbacks, excluding the optional container header.
        item_count: u64,
        /// Preceding header and item callbacks; equal to the final ordinal.
        delivery_count: u64,
    },
}
impl ExportDeliverySource {
    pub fn validate(&self) -> Result<(), ExportDeliveryError> {
        match self {
            Self::Item { start, end } if start > end => {
                Err(ExportDeliveryError::InvalidSourceRange)
            }
            Self::Complete {
                item_count,
                delivery_count,
                ..
            } if delivery_count
                .checked_sub(*item_count)
                .is_none_or(|headers| headers > 1) =>
            {
                Err(ExportDeliveryError::InvalidCompletionCounts)
            }
            _ => Ok(()),
        }
    }
}

/// The bytes a native reading covered, or the actual source of an export callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadingFootprint {
    Bytes { start: u64, end: u64 },
    Export { source: ExportDeliverySource },
}
impl ReadingFootprint {
    pub fn validate(&self) -> Result<(), ExportDeliveryError> {
        match self {
            Self::Bytes { start, end } if start > end => {
                Err(ExportDeliveryError::InvalidSourceRange)
            }
            Self::Export { source } => source.validate(),
            _ => Ok(()),
        }
    }
}

/// One immutable callback body and its position in the delivery prefix.
/// `ordinal` counts actual callbacks, independently of message display order or
/// raw item position. The caller computes body_hash from the delivered semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportDelivery {
    pub version: u32,
    pub ordinal: u64,
    pub source: ExportDeliverySource,
    pub body_hash: BlobHash,
    pub previous_prefix: BlobHash,
    pub prefix_hash: BlobHash,
}
impl ExportDelivery {
    pub const VERSION: u32 = VERSION;

    pub fn seed() -> BlobHash {
        BlobHash::of(SEED)
    }

    /// The single versioned prefix codec, shared by capture and receiving control.
    pub fn link_hash(
        version: u32,
        ordinal: u64,
        source: &ExportDeliverySource,
        body_hash: BlobHash,
        previous_prefix: BlobHash,
    ) -> BlobHash {
        let bytes = serde_json::to_vec(&(version, ordinal, source, body_hash, previous_prefix))
            .expect("primitive export delivery headers serialize");
        let mut hash = blake3::Hasher::new();
        hash.update(LINK);
        hash.update(&bytes);
        BlobHash(*hash.finalize().as_bytes())
    }

    pub fn new(
        ordinal: u64,
        source: ExportDeliverySource,
        body_hash: BlobHash,
        previous_prefix: BlobHash,
    ) -> Result<Self, ExportDeliveryError> {
        let delivery = Self {
            version: Self::VERSION,
            ordinal,
            source,
            body_hash,
            previous_prefix,
            prefix_hash: Self::link_hash(
                Self::VERSION,
                ordinal,
                &source,
                body_hash,
                previous_prefix,
            ),
        };
        delivery.validate()?;
        Ok(delivery)
    }

    /// Validates envelope shape and link integrity, not the body's truth, an
    /// earlier replayed prefix, or the receiving input/reading/receipt binding.
    pub fn validate(&self) -> Result<(), ExportDeliveryError> {
        if self.version != Self::VERSION {
            return Err(ExportDeliveryError::UnsupportedVersion {
                version: self.version,
            });
        }
        self.source.validate()?;
        match self.source {
            ExportDeliverySource::ContainerHeader if self.ordinal != 0 => {
                return Err(ExportDeliveryError::InvalidOrdinal);
            }
            ExportDeliverySource::Complete { delivery_count, .. }
                if delivery_count != self.ordinal =>
            {
                return Err(ExportDeliveryError::InvalidOrdinal);
            }
            _ => {}
        }
        if (self.ordinal == 0 && self.previous_prefix != Self::seed())
            || self.prefix_hash
                != Self::link_hash(
                    self.version,
                    self.ordinal,
                    &self.source,
                    self.body_hash,
                    self.previous_prefix,
                )
        {
            return Err(ExportDeliveryError::InvalidPrefix);
        }
        Ok(())
    }

    /// Retains only this delivery's cursor metadata. Callers establish admission
    /// and verify replayed prefixes before persisting or extending this cursor.
    pub fn cursor(&self) -> Result<ExportCursor, ExportDeliveryError> {
        self.validate()?;
        Ok(ExportCursor {
            version: ExportCursor::VERSION,
            last_ordinal: self.ordinal,
            prefix_hash: self.prefix_hash,
        })
    }
}

/// One retained cursor; absence is represented by Option<ExportCursor>, not zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportCursor {
    pub version: u32,
    pub last_ordinal: u64,
    pub prefix_hash: BlobHash,
}
impl ExportCursor {
    pub const VERSION: u32 = VERSION;

    pub fn validate(&self) -> Result<(), ExportDeliveryError> {
        if self.version != Self::VERSION {
            return Err(ExportDeliveryError::UnsupportedVersion {
                version: self.version,
            });
        }
        Ok(())
    }

    pub fn from_first(delivery: &ExportDelivery) -> Result<Self, ExportDeliveryError> {
        if delivery.ordinal != 0 {
            return Err(ExportDeliveryError::InvalidOrdinal);
        }
        delivery.cursor()
    }

    pub fn next_ordinal(&self) -> Result<u64, ExportDeliveryError> {
        self.validate()?;
        self.last_ordinal
            .checked_add(1)
            .ok_or(ExportDeliveryError::OrdinalOverflow)
    }

    /// Checks one extension. Verifying an archived replay through this retained
    /// prefix remains the receiving caller's job; no earlier item ledger exists.
    pub fn advance(&self, delivery: &ExportDelivery) -> Result<Self, ExportDeliveryError> {
        let expected = self.next_ordinal()?;
        delivery.validate()?;
        if delivery.ordinal != expected {
            return Err(ExportDeliveryError::InvalidOrdinal);
        }
        if delivery.previous_prefix != self.prefix_hash {
            return Err(ExportDeliveryError::InvalidPrefix);
        }
        delivery.cursor()
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    thiserror::Error,
)]
#[serde(tag = "error", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExportDeliveryError {
    #[error("unsupported export delivery protocol version {version}")]
    UnsupportedVersion { version: u32 },
    #[error("export reading source range is reversed")]
    InvalidSourceRange,
    #[error("export completion counts disagree with the optional single header")]
    InvalidCompletionCounts,
    #[error("export delivery ordinal is not the expected callback position")]
    InvalidOrdinal,
    #[error("export delivery prefix does not match its envelope or predecessor")]
    InvalidPrefix,
    #[error("export delivery ordinal cannot advance beyond u64::MAX")]
    OrdinalOverflow,
}

#[cfg(test)]
mod tests;
