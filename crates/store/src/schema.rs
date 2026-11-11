//! Table declarations. Each database has one baseline: an empty file is
//! created at version 1 with exactly the tables declared here, and a file at
//! version 1 with these tables opens. A newer file, or one whose tables differ
//! (an archive in another format), is refused unchanged.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;

pub const ARCHIVE: &str = concat!(
    include_str!("schema/archive.sql"),
    "\n",
    include_str!("schema/client_requests.sql"),
    "\n",
    include_str!("schema/assertions.sql"),
    "\n",
    include_str!("schema/variants.sql"),
    "\n",
    include_str!("schema/query_objects.sql"),
    "\n",
    include_str!("schema/query_runs.sql"),
    "\n",
    include_str!("schema/signals.sql")
);
pub const INDEX: &str = concat!(
    include_str!("schema/index.sql"),
    "\n",
    include_str!("schema/wire_shapes.sql"),
    "\n",
    include_str!("schema/injection_fields.sql"),
    "\n",
    include_str!("schema/attachment_bindings.sql"),
    "\n",
    include_str!("schema/text_fragments.sql"),
    "\n",
    include_str!("schema/session_cwd.sql"),
    "\n",
    include_str!("schema/session_items.sql"),
    "\n",
    include_str!("schema/session_values.sql")
);
#[cfg(test)]
pub(crate) const QUERY_OBJECTS: &str = concat!(
    include_str!("schema/query_objects.sql"),
    "\n",
    include_str!("schema/query_runs.sql")
);
pub(crate) const READ_INDEXES: &str = include_str!("schema/index_reads.sql");
pub const EMBEDDINGS: &str = include_str!("schema/embeddings.sql");

/// The version every baseline database is created at.
pub const ARCHIVE_VERSION: u32 = 1;

/// The index schema and row encoding version. Reading versions are independent.
pub const INDEX_VERSION: u32 = 1;

/// The SQLite `application_id` of an `archive.db` or `index.db` written at or
/// after the baseline ("PLG1"). A non-empty file without it was written by an
/// earlier Polylogue and is an older format, read only by the importer.
pub const FORMAT_ID: i32 = 0x504c_4731;

fn format_id(conn: &Connection) -> Result<i32> {
    Ok(conn.pragma_query_value(None, "application_id", |r| r.get(0))?)
}

/// Refuses a versioned database that carries no baseline format identity,
/// and an unversioned file that already holds tables: only an empty file is
/// created into.
fn check_format(conn: &Connection, database: &'static str) -> Result<()> {
    let found = version(conn)?;
    if found != 0 && format_id(conn)? != FORMAT_ID {
        return Err(StoreError::OlderFormat { database });
    }
    if found == 0 && !shape(conn)?.is_empty() {
        return Err(StoreError::NoMigration { database, found });
    }
    Ok(())
}

/// The `user_version` of a database: 0 for an empty file.
pub fn version(conn: &Connection) -> Result<u32> {
    Ok(conn.pragma_query_value(None, "user_version", |r| r.get(0))?)
}

fn set_version(conn: &Connection, v: u32) -> Result<()> {
    conn.pragma_update(None, "user_version", v)?;
    Ok(())
}

pub fn create_index(conn: &Connection) -> Result<()> {
    create_index_in(conn, false)
}

fn add_primary(conn: &Connection) -> Result<()> {
    conn.execute_batch(&crate::source::schema()?.ddl)?;
    conn.execute_batch(crate::primary::projection::SCHEMA)?;
    Ok(())
}

fn base_index(conn: &Connection) -> Result<()> {
    conn.execute_batch(INDEX)?;
    conn.execute_batch(crate::coverage::SCHEMA)?;
    conn.execute_batch(crate::variants::SCHEMA)?;
    conn.execute_batch(crate::parser_context::SCHEMA)?;
    Ok(())
}

fn create_index_in(conn: &Connection, primary: bool) -> Result<()> {
    base_index(conn)?;
    conn.execute_batch(READ_INDEXES)?;
    if primary {
        add_primary(conn)?;
        conn.execute(
            "INSERT INTO meta(key,value) VALUES ('primary_cache',?1)",
            [uuid::Uuid::now_v7().to_string()],
        )?;
    }
    if !primary {
        conn.execute("INSERT INTO meta(key,value) VALUES ('building','1')", [])?;
    }
    crate::archive::generation::create(conn)?;
    set_version(conn, INDEX_VERSION)?;
    conn.pragma_update(None, "application_id", FORMAT_ID)?;
    Ok(())
}

/// Creates the index in an empty file, or opens one at this version with the
/// declared tables, adding any read index it lacks. Any other index is
/// refused without removing its rows; rebuilding it is a deliberate act.
pub fn ensure_index(conn: &Connection) -> Result<()> {
    ensure_index_in(conn, false)
}
pub fn ensure_primary_index(conn: &Connection) -> Result<()> {
    ensure_index_in(conn, true)
}
fn ensure_index_in(conn: &Connection, primary: bool) -> Result<()> {
    // ALTER TABLE validates views that read packed bodies through these functions.
    crate::sql_functions::register(conn)?;
    check_format(conn, "index")?;
    let found = version(conn)?;
    if found > INDEX_VERSION {
        return Err(StoreError::IndexVersion {
            found,
            wanted: INDEX_VERSION,
        });
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    if found == 0 {
        create_index_in(&tx, primary)?;
    } else {
        // Read indexes are added before the shape check, so an index missing
        // only those is completed rather than refused. A first build still in
        // progress leaves out the ones it defers: it builds each once from a
        // sorted scan when it ends, and building one here would read the
        // whole table at every restart.
        let deferring = building(&tx)?;
        let statements = READ_INDEXES
            .lines()
            .filter(|line| !line.trim_start().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");
        for statement in statements.split(';').map(str::trim) {
            // `CREATE INDEX IF NOT EXISTS <name> ON ...`
            let name = statement.split_whitespace().nth(5);
            if !statement.is_empty()
                && !(deferring
                    && name.is_some_and(|name| {
                        crate::writer::deferred_index_names().any(|deferred| deferred == name)
                    }))
            {
                tx.execute_batch(statement)?;
            }
        }
        if primary {
            tx.execute_batch(include_str!("primary/projection/read_indexes.sql"))?;
        }
        let matches = found == INDEX_VERSION
            && if building(&tx)? {
                same_build_schema_in(&tx, primary)?
            } else {
                same_index_schema_in(&tx, primary)?
            };
        if !matches {
            return Err(StoreError::NoMigration {
                database: "index",
                found,
            });
        }
    }
    tx.commit()?;
    Ok(())
}

/// Whether the first build still has search and secondary indexes to finish.
pub fn building(conn: &Connection) -> Result<bool> {
    let value: Option<String> = conn
        .query_row("SELECT value FROM meta WHERE key = 'building'", [], |row| {
            row.get(0)
        })
        .optional()?;
    Ok(value.as_deref() == Some("1"))
}

/// Creates `archive.db` in an empty file, or opens one at the baseline.
pub fn ensure_archive(conn: &Connection) -> Result<()> {
    crate::sql_functions::register(conn)?;
    check_format(conn, "archive")?;
    // An empty file takes the identity before its tables, so no crash can
    // leave a versioned archive without it.
    if version(conn)? == 0 {
        conn.pragma_update(None, "application_id", FORMAT_ID)?;
    }
    ensure(conn, "archive", &[ARCHIVE])
}

/// Creates `embeddings.db`, or opens one at the baseline.
pub(crate) fn ensure_embeddings(conn: &Connection) -> Result<()> {
    ensure(conn, "embeddings", &[EMBEDDINGS])
}

/// Creates the database from `sql` in an empty file, or opens one at the
/// baseline whose tables match `sql`. A newer database, or one whose tables
/// differ, is refused unchanged.
pub(crate) fn ensure(conn: &Connection, database: &'static str, sql: &[&str]) -> Result<()> {
    let current = 1;
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
        set_version(&tx, current)?;
    } else if !same_schema(&tx, sql)? {
        return Err(StoreError::NoMigration { database, found });
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

/// Missing read indexes can be created without rebuilding parsed rows.
fn same_index_schema_in(conn: &Connection, primary: bool) -> Result<bool> {
    let expected = Connection::open_in_memory()?;
    base_index(&expected)?;
    if primary {
        add_primary(&expected)?;
    }
    let base = shape(&expected)?;
    expected.execute_batch(READ_INDEXES)?;
    let expected = shape(&expected)?;
    let read_indexes: HashSet<_> = expected
        .iter()
        .filter(|item| item.0 == "index" && !base.contains(item))
        .map(|item| item.1.clone())
        .collect();
    let found = shape(conn)?;
    Ok(found.iter().all(|item| expected.contains(item))
        && expected.iter().all(|item| {
            (item.0 == "index" && read_indexes.contains(&item.1)) || found.contains(item)
        }))
}

/// A pending first build may lack only the indexes the writer deferred.
fn same_build_schema_in(conn: &Connection, primary: bool) -> Result<bool> {
    let expected = Connection::open_in_memory()?;
    base_index(&expected)?;
    expected.execute_batch(READ_INDEXES)?;
    if primary {
        add_primary(&expected)?;
    }
    let found = shape(conn)?;
    let mut wanted = shape(&expected)?;
    wanted.retain(|(kind, name, _)| {
        kind != "index"
            || !crate::writer::deferred_index_names().any(|deferred| deferred == name)
            || found.iter().any(|(_, present, _)| present == name)
    });
    Ok(found == wanted)
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
