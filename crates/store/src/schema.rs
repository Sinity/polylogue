//! Table declarations. `archive.db` and `embeddings.db` evolve by numbered
//! additive migrations applied when the daemon opens the archive.
//! `index.db` is never migrated: a new version builds a new file and swaps
//! it in.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;

pub const ARCHIVE: &str = include_str!("schema/archive.sql");
pub const INDEX: &str = include_str!("schema/index.sql");
pub const EMBEDDINGS: &str = include_str!("schema/embeddings.sql");

/// Bumped whenever `INDEX` changes; a different stored version means the
/// index is rebuilt into a new file.
pub const INDEX_VERSION: u32 = 26;

/// Migrations for `archive.db`, applied in order after the base schema.
/// Each is additive; an archive at a higher number than this list is refused.
pub const ARCHIVE_MIGRATIONS: &[&str] = &[
    include_str!("schema/migrations/002-excision-revision.sql"),
    include_str!("schema/migrations/003-vector-excision.sql"),
    include_str!("schema/migrations/004-local-blob-excision.sql"),
    include_str!("schema/migrations/005-append-facts.sql"),
];

/// The `user_version` of a database: 0 for an empty file.
pub fn version(conn: &Connection) -> Result<u32> {
    Ok(conn.pragma_query_value(None, "user_version", |r| r.get(0))?)
}

fn set_version(conn: &Connection, v: u32) -> Result<()> {
    conn.pragma_update(None, "user_version", v)?;
    Ok(())
}

/// Creates `archive.db` at the base version (1) and applies migrations.
pub fn create_archive(conn: &Connection) -> Result<()> {
    conn.execute_batch(ARCHIVE)?;
    set_version(conn, 1)?;
    Ok(())
}

pub fn create_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(INDEX)?;
    set_version(conn, INDEX_VERSION)?;
    Ok(())
}

pub fn create_embeddings(conn: &Connection) -> Result<()> {
    conn.execute_batch(EMBEDDINGS)?;
    set_version(conn, 1)?;
    Ok(())
}

/// Creates the index in an empty file. An index at another version is an
/// error here; [`crate::Archive::open_or_create`] moves such a file aside
/// first, so the loop rebuilds the index from `archive.db`.
pub fn ensure_index(conn: &Connection) -> Result<()> {
    match version(conn)? {
        0 => create_index(conn),
        v if v == INDEX_VERSION => Ok(()),
        found => Err(StoreError::IndexVersion {
            found,
            wanted: INDEX_VERSION,
        }),
    }
}

/// Whether the index was being built without syncs and never settled
/// (`meta.unsynced = 1`): after a crash such a file may be damaged, so it is
/// rebuilt rather than trusted.
pub fn unsynced(conn: &Connection) -> Result<bool> {
    let v: Option<String> = conn
        .query_row("SELECT value FROM meta WHERE key = 'unsynced'", [], |r| {
            r.get(0)
        })
        .optional()?;
    Ok(v.as_deref() == Some("1"))
}

/// Applies pending archive migrations inside one transaction.
pub fn migrate_archive(conn: &mut Connection) -> Result<()> {
    let latest = 1 + ARCHIVE_MIGRATIONS.len() as u32;
    let found = version(conn)?;
    if found > latest {
        return Err(StoreError::TooNew {
            found,
            supported: latest,
        });
    }
    let tx = conn.transaction()?;
    for (i, sql) in ARCHIVE_MIGRATIONS.iter().enumerate() {
        let target = 2 + i as u32;
        if found < target {
            tx.execute_batch(sql)?;
            set_version(&tx, target)?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Ordinary tables in `index.db` and their column names.
pub(crate) fn index_columns(
    i: &rusqlite::Transaction<'_>,
) -> Result<Vec<(String, HashSet<String>)>> {
    let names: Vec<String> = i
        .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' AND name NOT LIKE 'fts_%'
             AND name NOT LIKE 'tool_call_trigram%'",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for name in names {
        let cols: HashSet<String> = i
            .prepare(&format!("SELECT name FROM pragma_table_info('{name}')"))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        out.push((name, cols));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
