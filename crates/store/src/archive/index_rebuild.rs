//! Replacing an `index.db` written in another format. An unknown current
//! index shape is refused unchanged; an index at another format version is
//! renamed beside it, under its schema identity, and a new one is built in
//! its place. The renamed file is removed once the new index has finished
//! its first build. `archive.db` is never handled here.

use crate::{Result, StoreError, schema};
use rusqlite::{Connection, OptionalExtension};
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
    /// Whether the first build of the new index is still running.
    pub building: bool,
    /// The set-aside file beside `index.db`, until the first build ends.
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
        StoreError::IndexVersion { found, wanted } => Some(IndexRebuildReason::FormatVersion {
            found: *found,
            wanted: *wanted,
        }),
        StoreError::OlderFormat { database: "index" } => Some(IndexRebuildReason::OlderFormat),
        _ => None,
    }
}

/// Renames `index.db` (closed by the caller after reading `previous`) and
/// any write-ahead files it left beside it, under its schema identity.
pub(crate) fn set_aside(index: &Path, previous: &str) -> Result<()> {
    let dir = index
        .parent()
        .ok_or_else(|| StoreError::Invalid("index.db has no directory".into()))?;
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
    index.execute(
        "INSERT INTO meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [META_KEY, &serde_json::to_string(&record)?],
    )?;
    Ok(())
}

/// Whether the index has finished its first build: standalone, the build
/// marker is cleared. Primary also needs its actual native head, admitted local
/// parser progress for every stored chain, and no unresolved native facts.
pub(crate) fn built(index: &Connection, primary: bool) -> Result<bool> {
    if !primary {
        return Ok(!schema::building(index)?);
    }
    let state = crate::primary::projection::replay_state(index)?;
    if !matches!(
        (state.applied_position, state.head),
        (Some(applied), Some(head)) if applied >= head
    ) {
        return Ok(false);
    }
    let unresolved: bool = index.query_row(
        "WITH chains AS (SELECT input,chain,max(length) AS length FROM source_revision GROUP BY input,chain)
         SELECT EXISTS(SELECT 1 FROM chains c
             LEFT JOIN input_ref i ON i.id=c.input
             LEFT JOIN parse_state p ON p.input_pk=i.pk AND p.chain=c.chain
             WHERE c.length>0 AND NOT COALESCE(p.native_publication IS NOT NULL AND (
                 (p.reading_version IS NOT NULL AND p.reading_version<>'' AND
                    ((p.status IN ('parsing','done') AND p.parsed_upto>=c.length) OR
                     (p.status='waiting' AND p.at_length=c.length) OR
                     (p.status IN ('refused','failed') AND p.at_length=c.length AND p.stopped_attempt IS NOT NULL)))
                 OR (p.status='no_parser_yet' AND p.at_length=c.length)),0))
         OR EXISTS(SELECT 1 FROM primary_refused r LEFT JOIN primary_fact f ON f.event=r.event
                   WHERE f.event IS NULL OR f.state='live')",
        [], |row| row.get(0),
    )?;
    Ok(!unresolved)
}

/// Removes every set-aside index beside `index.db` once the index has
/// finished its first build. Matching by name also removes one a crash left
/// before its record was written.
pub(crate) fn remove_when_built(index: &Connection, path: &Path, primary: bool) -> Result<()> {
    let Some(dir) = path.parent() else {
        return Ok(());
    };
    let retained = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with(PREFIX))
        })
        .collect::<Vec<_>>();
    if retained.is_empty() || !built(index, primary)? {
        return Ok(());
    }
    for path in retained {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// The rebuild this index records, with its current state, if it replaced
/// an index from another schema.
pub(crate) fn status(index: &Connection, primary: bool) -> Result<Option<IndexRebuild>> {
    let value: Option<String> = index
        .query_row(
            "SELECT value FROM main.meta WHERE key = ?1",
            [META_KEY],
            |r| r.get(0),
        )
        .optional()?;
    let Some(value) = value else {
        return Ok(None);
    };
    let record: Record = serde_json::from_str(&value)?;
    let building = !built(index, primary)?;
    Ok(Some(IndexRebuild {
        set_aside: building.then(|| file_name(&record.previous_schema)),
        reason: record.reason,
        previous_schema: record.previous_schema,
        schema: record.schema,
        building,
    }))
}

#[cfg(test)]
mod tests;
