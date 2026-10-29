use crate::{Archive, Result, StoreError};
use std::os::unix::ffi::OsStrExt;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

/// A pre-acquisition cut of source authority. It is not an index write counter.
/// Primary cuts require the owner's full projection certificate; a raw-input
/// certificate or an intermediate stream checkpoint cannot construct one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceBoundary {
    generation: String,
    scope_root: [u8; 32],
    authority: Authority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Authority {
    Standalone { through: u64 },
}

impl SourceBoundary {
    pub(crate) fn read(
        index: &Connection,
        source: &Connection,
        archive: &Archive,
    ) -> Result<Option<Self>> {
        if archive.is_primary() {
            return Ok(None);
        }
        let Some(through) = source_position(source)? else {
            return Ok(None);
        };
        let generation = index.query_row(
            "SELECT value FROM main.meta WHERE key='generation'",
            [],
            |row| row.get(0),
        )?;
        Ok(Some(Self {
            generation,
            scope_root: scope_root(archive),
            authority: Authority::Standalone { through },
        }))
    }

    pub(crate) fn through(&self, index: &Connection, archive: &Archive) -> Result<u64> {
        let generation: String = index.query_row(
            "SELECT value FROM main.meta WHERE key='generation'",
            [],
            |row| row.get(0),
        )?;
        if archive.is_primary() || generation != self.generation || scope_root(archive) != self.scope_root {
            return Err(StoreError::ScopeChanged);
        }
        match self.authority {
            Authority::Standalone { through } => Ok(through),
        }
    }
}

fn source_position(conn: &Connection) -> Result<Option<u64>> {
    let source = super::Source::reading(conn)?;
    let position: Option<String> = conn.query_row(
        &source.sql("SELECT value FROM @source.meta WHERE key='write_counter'"),
        [],
        |row| row.get(0),
    ).optional()?;
    if let Some(position) = position {
        return position.parse::<u64>().map(Some).map_err(|_| {
            StoreError::Invalid("source admission fence is not a nonnegative integer".into())
        });
    }
    let empty: bool = conn.query_row(
        &source.sql("SELECT NOT EXISTS(SELECT 1 FROM @source.input)"),
        [],
        |row| row.get(0),
    )?;
    Ok(empty.then_some(0))
}

// This binds the existing scope registry key; it does not create an archive identity.
fn scope_root(archive: &Archive) -> [u8; 32] {
    *blake3::hash(Archive::resolved_root(&archive.root).as_os_str().as_bytes()).as_bytes()
}
