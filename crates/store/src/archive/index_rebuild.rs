//! Replacing an `index.db` written for another schema. The index is derived,
//! so an index this build cannot open is replaced and its reason recorded in
//! the new one. A standalone index is built privately and installed over the
//! old one ([`super::index_build`]); the primary cache is renamed beside
//! itself under its schema identity, rebuilt from the confirmed stream, and
//! the renamed file removed once the rebuild is complete ([`built`]: the
//! stream's latest position applied, every stored chain parsed, no event
//! unresolved).
//! `archive.db` is never handled here.

use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::{Result, StoreError};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::Path;

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
    /// Primary mode: the set-aside cache beside `index.db`, until the
    /// rebuild reaches the stream's latest_position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set_aside: Option<String>,
}

/// The record kept in the new index's `meta`, under [`META_KEY`].
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    reason: IndexRebuildReason,
    previous_schema: String,
    schema: String,
}

pub(crate) const META_KEY: &str = "index_rebuild";
const PREFIX: &str = "index-set-aside-";

fn file_name(identity: &str) -> String {
    format!("{PREFIX}{identity}.db")
}

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

/// Why a standalone index a build set aside was replaced, and its schema
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

/// Renames the primary cache (closed by the caller after reading `previous`)
/// and any write-ahead files it left beside it, under its schema identity.
pub(crate) fn set_aside(index: &Path, previous: &str) -> Result<()> {
    let dir = index.parent().or_internal("index.db has no directory")?;
    let target = dir.join(file_name(previous));
    for suffix in ["-wal", "-shm", ""] {
        let mut from = index.as_os_str().to_owned();
        from.push(suffix);
        let mut to = target.as_os_str().to_owned();
        to.push(suffix);
        match std::fs::rename(&from, &to) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
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

/// Whether the primary cache has finished its rebuild: it has applied Sinex's
/// latest position, committed parser progress for every stored chain, and has no
/// unresolved Sinex events.
pub(crate) fn built(index: &Connection) -> Result<bool> {
    let state = crate::primary::confirmed::replay_state(index)?;
    if !matches!(
        (state.applied_position, state.latest_position),
        (Some(applied), Some(latest_position)) if applied >= latest_position
    ) {
        return Ok(false);
    }
    let unresolved: bool = index.scalar(
        "WITH chains AS (SELECT input,chain,max(length) AS length FROM source_revision GROUP BY input,chain)
         SELECT EXISTS(SELECT 1 FROM chains c
             LEFT JOIN input_ref i ON i.id=c.input
             LEFT JOIN parse_state p ON p.input_pk=i.pk AND p.chain=c.chain
             WHERE c.length>0 AND NOT COALESCE((
                 (p.parse_version IS NOT NULL AND p.parse_version<>'' AND
                    ((p.status IN ('parsing','done') AND p.parsed_upto>=c.length) OR
                     (p.status='waiting' AND p.at_length=c.length) OR
                     (p.status IN ('refused','failed') AND p.at_length=c.length AND p.stopped_attempt IS NOT NULL)))
                 OR (p.status='no_parser_yet' AND p.at_length=c.length)),0))
         OR EXISTS(SELECT 1 FROM primary_refused r LEFT JOIN confirmed_event f ON f.event=r.event
                   WHERE f.event IS NULL OR f.state='live')",
        [])?;
    Ok(!unresolved)
}

/// Removes every set-aside cache beside `index.db` once the primary cache
/// has finished its rebuild. Matching by name also removes one a crash left
/// before its record was written.
pub(crate) fn remove_when_built(index: &Connection, path: &Path) -> Result<()> {
    let Some(dir) = path.parent() else {
        return Ok(());
    };
    let set_aside_files = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with(PREFIX))
        })
        .collect::<Vec<_>>();
    if set_aside_files.is_empty() || !built(index)? {
        return Ok(());
    }
    let search: Vec<_> = set_aside_files
        .iter()
        .filter_map(|path| crate::search::files::named_on_disk(path))
        .collect();
    for path in set_aside_files {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    for files in search {
        crate::search::files::remove(&files)?;
    }
    Ok(())
}

/// The rebuild this index records, with its current state, if it replaced
/// an index from another schema. A standalone index is read only once built,
/// or, while a build replaces it, as the previous index.
pub(crate) fn status(index: &Connection, archive: &super::Archive) -> Result<Option<IndexRebuild>> {
    let value: Option<String> =
        index.scalar_optional("SELECT value FROM main.meta WHERE key = ?1", [META_KEY])?;
    let Some(value) = value else {
        return Ok(None);
    };
    let record: Record = serde_json::from_str(&value)?;
    let primary = archive.is_primary();
    let building = if primary {
        !built(index)?
    } else {
        archive.index_build.building()
    };
    Ok(Some(IndexRebuild {
        set_aside: (primary && building).then(|| file_name(&record.previous_schema)),
        reason: record.reason,
        previous_schema: record.previous_schema,
        schema: record.schema,
        building,
    }))
}

#[cfg(test)]
mod tests;
