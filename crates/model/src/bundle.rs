//! Portable readings over the canonical domain records. Provider bytes stay
//! in their archive; imported source claims never become local byte extents.

use crate::Event;
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "polylogue-bundle";
pub const VERSION: u32 = 1;

/// One original source claim carried through a bundle. The receiving
/// archive keeps its actual local bundle input and chain separately.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct EventSourceClaim {
    pub input: String,
    pub source: String,
    pub source_chain: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end: Option<u64>,
}

/// A stable event identity and all original claims for the selected
/// source reading. Equal imports union claims rather than dropping one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportedEventIdentity {
    pub id: String,
    pub sources: Vec<EventSourceClaim>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputEventRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: u64,
    pub identity: ImportedEventIdentity,
    pub event: Event,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BundleError {
    #[error("invalid event identity or original source claim")]
    InvalidEventIdentity,
    #[error("event reference does not name its imported identity")]
    EventReferenceMismatch,
    #[error("event after a message has no session")]
    InvalidEventAssociation,
}

impl EventSourceClaim {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.input.is_empty()
            || self.source.is_empty()
            || self.source_chain < 1
            || self.source_start.is_some() != self.source_end.is_some()
            || self
                .source_start
                .zip(self.source_end)
                .is_some_and(|(a, b)| a > b)
        {
            return Err(BundleError::InvalidEventIdentity);
        }
        Ok(())
    }
}

impl ImportedEventIdentity {
    pub fn validate(&self) -> Result<(), BundleError> {
        if self.id.is_empty() || self.sources.is_empty() {
            return Err(BundleError::InvalidEventIdentity);
        }
        self.sources.iter().try_for_each(EventSourceClaim::validate)
    }

    pub fn canonicalize(&mut self) {
        self.sources.sort();
        self.sources.dedup();
    }
}

impl InputEventRecord {
    pub fn validate(&self) -> Result<(), BundleError> {
        self.identity.validate()?;
        if self.reference != format!("event:{}", self.identity.id) {
            return Err(BundleError::EventReferenceMismatch);
        }
        if self.event.session.is_none() && self.event.after.is_some() {
            return Err(BundleError::InvalidEventAssociation);
        }
        Ok(())
    }
}
