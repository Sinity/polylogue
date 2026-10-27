//! Table declarations. `archive.db` holds what cannot be recomputed, so an
//! older one is brought forward by numbered migrations
//! ([`ARCHIVE_MIGRATIONS`]) when the archive opens. An archive newer than this
//! build, or one whose tables match no version this build knows, is refused.
//! `embeddings.db` has one version and no migrations yet. `index.db` is never
//! migrated: a new version or a different schema builds a new file and swaps
//! it in.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;

pub const ARCHIVE: &str = include_str!("schema/archive.sql");
pub const INDEX: &str = include_str!("schema/index.sql");
pub const EMBEDDINGS: &str = include_str!("schema/embeddings.sql");

/// A different version rebuilds the index into a new file. Raise it when the
/// index schema, the row encoding, or what any parser produces from the same
/// bytes changes.
pub const INDEX_VERSION: u32 = 47;

/// The steps that bring `archive.db` forward: the first takes version 1 to
/// version 2, and so on. A change to `archive.sql` appends one step, which
/// must leave exactly the tables `archive.sql` creates (where `ALTER TABLE`
/// cannot, the step creates the new table, copies the rows and renames it).
/// The schema tests check that version 1 and these steps give `archive.sql`.
pub(crate) const ARCHIVE_MIGRATIONS: &[&str] = &[
    // 2: backfill jobs.
    include_str!("schema/archive_2.sql"),
    // 3: requests kept whole.
    include_str!("schema/archive_3.sql"),
    // 4: row counts kept by triggers.
    include_str!("schema/archive_4.sql"),
];

/// The version `archive.sql` creates.
pub const ARCHIVE_VERSION: u32 = 1 + ARCHIVE_MIGRATIONS.len() as u32;

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

/// Whether the index was being built without syncs and never finished
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

/// Creates `archive.db` in an empty file and brings an older one forward.
pub fn ensure_archive(conn: &Connection) -> Result<()> {
    ensure(conn, "archive", &[ARCHIVE], ARCHIVE_MIGRATIONS)
}

/// The tables of `sinex.db`, primary mode's register: the archive's tables
/// and the Sinex material of each part.
pub(crate) const SINEX: [&str; 2] = [ARCHIVE, crate::primary::SCHEMA];

/// Creates `sinex.db` in an empty file and brings an older one forward with
/// the archive's migrations.
pub fn ensure_sinex(conn: &Connection) -> Result<()> {
    ensure(conn, "sinex", &SINEX, ARCHIVE_MIGRATIONS)
}

/// Creates `embeddings.db` in an empty file; refuses one with other tables.
pub(crate) fn ensure_embeddings(conn: &Connection) -> Result<()> {
    ensure(conn, "embeddings", &[EMBEDDINGS], &[])
}

/// Creates the database from `sql` in an empty file, or applies the
/// migrations it lacks in one transaction. A database newer than the
/// migrations, or whose tables then differ from `sql`, is refused unchanged.
pub(crate) fn ensure(
    conn: &Connection,
    database: &'static str,
    sql: &[&str],
    migrations: &[&str],
) -> Result<()> {
    let current = 1 + migrations.len() as u32;
    let found = version(conn)?;
    if found > current {
        return Err(StoreError::NewerSchema {
            database,
            found,
            current,
        });
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    if found == 0 {
        for part in sql {
            tx.execute_batch(part)?;
        }
    } else {
        for step in &migrations[found as usize - 1..] {
            tx.execute_batch(step)?;
        }
        if !same_schema(&tx, sql)? {
            return Err(StoreError::NoMigration { database, found });
        }
    }
    if found != current {
        set_version(&tx, current)?;
    }
    tx.commit()?;
    Ok(())
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
/// two words. Quoted text is kept as written; quoted names lose their quotes.
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
                // Text keeps its quotes; a quoted name (SQLite quotes a
                // renamed table's) compares as the bare name.
                let text = c == '\'';
                let close = if c == '[' { ']' } else { c };
                if text {
                    out.push(c);
                }
                for c in chars.by_ref() {
                    if c == close {
                        if text {
                            out.push(c);
                        }
                        break;
                    }
                    out.push(c);
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
