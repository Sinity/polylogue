//! Table declarations. `archive.db` and `embeddings.db` are created at
//! version 1, the only version. A file whose version or tables differ from
//! what this build creates is refused: its data crosses to a new schema only
//! by an export with the build that wrote it and an import with this one.
//! `index.db` is never migrated: a new version or a different schema builds a
//! new file and swaps it in.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;

pub const ARCHIVE: &str = include_str!("schema/archive.sql");
pub const INDEX: &str = include_str!("schema/index.sql");
pub const EMBEDDINGS: &str = include_str!("schema/embeddings.sql");

/// A different version rebuilds the index into a new file. Raise it when the
/// index schema, the row encoding, or what any parser produces from the same
/// bytes changes.
pub const INDEX_VERSION: u32 = 42;

/// The `user_version` of a database: 0 for an empty file.
pub fn version(conn: &Connection) -> Result<u32> {
    Ok(conn.pragma_query_value(None, "user_version", |r| r.get(0))?)
}

fn set_version(conn: &Connection, v: u32) -> Result<()> {
    conn.pragma_update(None, "user_version", v)?;
    Ok(())
}

pub fn create_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(INDEX)?;
    crate::archive::generation::create(conn)?;
    set_version(conn, INDEX_VERSION)?;
    Ok(())
}

/// Creates the index in an empty file. An index at another version is an
/// error here; [`crate::Archive::open_or_create`] deletes such a file, or
/// one with another schema, first, so the loop rebuilds the index from
/// `archive.db`.
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

/// Creates `archive.db` in an empty file; refuses a file with another schema.
pub fn ensure_archive(conn: &Connection) -> Result<()> {
    ensure(conn, "archive", &[ARCHIVE])
}

/// The tables of `sinex.db`, primary mode's register: the archive's tables
/// and the Sinex material of each part.
pub(crate) const SINEX: [&str; 2] = [ARCHIVE, crate::primary::SCHEMA];

/// Creates `sinex.db` in an empty file; refuses a file with another schema.
pub fn ensure_sinex(conn: &Connection) -> Result<()> {
    ensure(conn, "sinex", &SINEX)
}

/// Creates `embeddings.db` in an empty file; refuses a file with another
/// schema.
pub(crate) fn ensure_embeddings(conn: &Connection) -> Result<()> {
    ensure(conn, "embeddings", &[EMBEDDINGS])
}

fn ensure(conn: &Connection, database: &'static str, sql: &[&str]) -> Result<()> {
    match version(conn)? {
        0 => {
            for part in sql {
                conn.execute_batch(part)?;
            }
            set_version(conn, 1)
        }
        1 if same_schema(conn, sql)? => Ok(()),
        _ => Err(StoreError::OtherSchema { database }),
    }
}

/// Whether the database holds exactly the tables, indexes and triggers that
/// `sql` creates. Comments and spacing inside the statements do not count.
pub(crate) fn same_schema(conn: &Connection, sql: &[&str]) -> Result<bool> {
    let expected = Connection::open_in_memory()?;
    for part in sql {
        expected.execute_batch(part)?;
    }
    Ok(shape(conn)? == shape(&expected)?)
}

fn shape(conn: &Connection) -> Result<Vec<(String, String, String)>> {
    Ok(conn
        .prepare(
            "SELECT type, name, sql FROM main.sqlite_master
             WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite~_%' ESCAPE '~'
             ORDER BY type, name",
        )?
        .query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, statement(&r.get::<_, String>(2)?)))
        })?
        .collect::<rusqlite::Result<_>>()?)
}

/// A statement without its comments, and with space only where it separates
/// two words. Quoted text is kept as written.
fn statement(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut chars = sql.chars().peekable();
    let mut space = false;
    let word = |c: char| c.is_alphanumeric() || c == '_';
    while let Some(c) = chars.next() {
        match c {
            '-' if chars.peek() == Some(&'-') => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
                space = true;
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut star = false;
                for c in chars.by_ref() {
                    if star && c == '/' {
                        break;
                    }
                    star = c == '*';
                }
                space = true;
            }
            c if c.is_whitespace() => space = true,
            '\'' | '"' | '`' | '[' => {
                if space && out.chars().last().is_some_and(word) {
                    out.push(' ');
                }
                space = false;
                let close = if c == '[' { ']' } else { c };
                out.push(c);
                for c in chars.by_ref() {
                    out.push(c);
                    if c == close {
                        break;
                    }
                }
            }
            c => {
                if space && word(c) && out.chars().last().is_some_and(word) {
                    out.push(' ');
                }
                space = false;
                out.push(c);
            }
        }
    }
    out
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
