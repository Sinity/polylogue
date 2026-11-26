//! Replacing an `index.db` written for another schema. The index is derived,
//! so an index this build cannot open is replaced and its reason recorded in
//! the new one. The new index is built privately and installed over the old
//! one ([`super::index_build`]).
//! `archive.db` is never handled here.

use crate::ConnExt as _;
use crate::{Result, StoreError};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

/// Why an index was set aside.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IndexRebuildReason {
    /// Its tables, indexes or triggers differ from this build's.
    SchemaDiffers,
    /// It is at another index format version.
    FormatVersion { found: u32, wanted: u32 },
    /// It carries no Polylogue format identity: written before the baseline.
    OlderFormat,
}

/// What `status` reports about an index that replaced one from another
/// schema. Progress is the rest of the same answer: the chains by parse
/// state against the stored inputs, and the read completeness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexRebuild {
    pub reason: IndexRebuildReason,
    /// The schema identity of the index that was set aside.
    pub previous_schema: String,
    /// The schema identity of this build's index.
    pub schema: String,
    /// Whether the build of the new index is still running.
    pub building: bool,
}

/// The record kept in the new index's `meta`, under [`META_KEY`].
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    reason: IndexRebuildReason,
    previous_schema: String,
    schema: String,
}

pub(crate) const META_KEY: &str = "index_rebuild";

/// The reason an opened index must be replaced, for the refusals that mean
/// "another schema". Any other error is not a reason to set it aside.
pub(crate) fn reason(error: &StoreError) -> Option<IndexRebuildReason> {
    match error {
        StoreError::OtherFormat {
            database: "index", ..
        } => Some(IndexRebuildReason::SchemaDiffers),
        StoreError::IndexVersion { found, wanted } => Some(IndexRebuildReason::FormatVersion {
            found: *found,
            wanted: *wanted,
        }),
        StoreError::OlderFormat { database: "index" } => Some(IndexRebuildReason::OlderFormat),
        _ => None,
    }
}

/// Why an index a build set aside was replaced, and its schema
/// identity, read from the file without changing it; `None` when this build
/// could open it after all.
pub(crate) fn set_aside_reason(
    previous: &Connection,
) -> Result<Option<(IndexRebuildReason, String)>> {
    let Err(error) = crate::schema::check_standalone_index(previous) else {
        return Ok(None);
    };
    let Some(reason) = reason(&error) else {
        return Err(error);
    };
    Ok(Some((reason, crate::schema::index_identity(previous)?)))
}

/// Records in the new index why it replaced the previous one.
pub(crate) fn record(
    index: &Connection,
    reason: IndexRebuildReason,
    previous_schema: String,
    schema: String,
) -> Result<()> {
    let record = Record {
        reason,
        previous_schema,
        schema,
    };
    crate::meta_set(index, META_KEY, serde_json::to_string(&record)?)
}

/// The rebuild this index records, with its current state, if it replaced
/// an index from another schema. An index is read only once built, or,
/// while a build replaces it, as the previous index.
pub(crate) fn status(index: &Connection, archive: &super::Archive) -> Result<Option<IndexRebuild>> {
    let value: Option<String> =
        index.scalar_optional("SELECT value FROM main.meta WHERE key = ?1", [META_KEY])?;
    let Some(value) = value else {
        return Ok(None);
    };
    let record: Record = serde_json::from_str(&value)?;
    Ok(Some(IndexRebuild {
        building: archive.index_build.building(),
        reason: record.reason,
        previous_schema: record.previous_schema,
        schema: record.schema,
    }))
}

#[cfg(test)]
mod tests;
