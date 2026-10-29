//! Table declarations. `archive.db` holds what cannot be recomputed, so an
//! older one is brought forward by numbered migrations
//! ([`ARCHIVE_MIGRATIONS`]) when the archive opens. An archive newer than this
//! build, or one whose tables match no version this build knows, is refused.
//! `embeddings.db` has one version and no migrations yet. An incompatible
//! index is refused without removing its rows.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;

pub const ARCHIVE: &str = concat!(
    include_str!("schema/archive.sql"),
    "\n",
    include_str!("schema/variants.sql"),
    "\n",
    include_str!("schema/query_objects.sql"),
    "\n",
    include_str!("schema/signals.sql")
);
pub const INDEX: &str = concat!(
    include_str!("schema/index.sql"),
    "\n",
    include_str!("schema/wire_shapes.sql"),
    "\n",
    include_str!("schema/index_9.sql"),
    "\n",
    include_str!("schema/attachment_bindings.sql"),
    "\n",
    include_str!("schema/index_12.sql")
);
pub(crate) const READ_INDEXES: &str = include_str!("schema/index_reads.sql");
pub const EMBEDDINGS: &str = include_str!("schema/embeddings.sql");

/// Published target versions are explicit. Unpublished reservations have no
/// migration and cannot be mistaken for a database this build understands.
struct IndexMigration {
    version: u32,
    target: IndexTarget,
}
#[derive(Clone, Copy)]
enum IndexTarget {
    Index(&'static str),
    IndexAndPrimary(&'static str, &'static str),
    PrimarySource(&'static str),
}
const INDEX_MIGRATIONS: &[IndexMigration] = &[
    IndexMigration { version: 2, target: IndexTarget::Index(include_str!("schema/index_2.sql")) },
    IndexMigration { version: 3, target: IndexTarget::Index(include_str!("schema/index_3.sql")) },
    IndexMigration { version: 4, target: IndexTarget::Index(include_str!("schema/index_4.sql")) },
    IndexMigration { version: 5, target: IndexTarget::Index(include_str!("schema/index_5.sql")) },
    IndexMigration { version: 6, target: IndexTarget::PrimarySource(include_str!("schema/archive_2.sql")) },
    IndexMigration { version: 8, target: IndexTarget::PrimarySource(concat!(
        include_str!("schema/variant_3.sql"), "\n", include_str!("schema/index_8.sql")
    )) },
    IndexMigration { version: 9, target: IndexTarget::Index(include_str!("schema/index_9.sql")) },
    IndexMigration { version: 10, target: IndexTarget::IndexAndPrimary(
        include_str!("schema/attachment_bindings.sql"), crate::primary::projection::REFERENCE_INDEX
    ) },
    IndexMigration { version: 11, target: IndexTarget::PrimarySource(include_str!("schema/index_11.sql")) },
    IndexMigration { version: 12, target: IndexTarget::Index(include_str!("schema/index_12.sql")) },
];
pub(crate) const ARCHIVE_MIGRATIONS: &[&str] = &[
    include_str!("schema/archive_2.sql"),
    concat!(include_str!("schema/variant_3.sql"), "\n", include_str!("schema/archive_3.sql")),
    include_str!("schema/archive_4.sql"),
];

/// The index schema and row encoding version. Reading versions are independent.
pub const INDEX_VERSION: u32 = INDEX_MIGRATIONS[INDEX_MIGRATIONS.len() - 1].version;
/// The version of the source declarations, including their numbered upgrades.
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
    Ok(())
}

/// Applies numbered index migrations without replacing committed rows.
pub fn ensure_index(conn: &Connection) -> Result<()> {
    ensure_index_in(conn, false)
}
pub fn ensure_primary_index(conn: &Connection) -> Result<()> {
    ensure_index_in(conn, true)
}
fn ensure_index_in(conn: &Connection, primary: bool) -> Result<()> {
    legacy_variant_time(conn)?;
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
        if found != 1 && !INDEX_MIGRATIONS.iter().any(|step| step.version == found) {
            return Err(StoreError::NoMigration { database: "index", found });
        }
        for step in INDEX_MIGRATIONS.iter().filter(|step| step.version > found) {
            match step.target {
                IndexTarget::Index(sql) => tx.execute_batch(sql)?,
                IndexTarget::IndexAndPrimary(sql, primary_sql) => {
                    tx.execute_batch(sql)?;
                    if primary { tx.execute_batch(primary_sql)?; }
                }
                IndexTarget::PrimarySource(sql) if primary => {
                    tx.execute_batch(&crate::source::Source::Cache.sql(sql))?;
                }
                IndexTarget::PrimarySource(_) => {}
            }
        }
        tx.execute_batch(READ_INDEXES)?;
        let matches = if building(&tx)? {
            if primary {
                same_build_schema_in(&tx, true)?
            } else {
                same_build_schema(&tx)?
            }
        } else {
            if primary {
                same_index_schema_in(&tx, true)?
            } else {
                same_index_schema(&tx)?
            }
        };
        if !matches {
            return Err(StoreError::NoMigration {
                database: "index",
                found,
            });
        }
        set_version(&tx, INDEX_VERSION)?;
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

/// Creates `archive.db` in an empty file and brings an older one forward.
pub fn ensure_archive(conn: &Connection) -> Result<()> {
    legacy_variant_time(conn)?;
    ensure(conn, "archive", &[ARCHIVE], ARCHIVE_MIGRATIONS)
}

/// Migration uses the same timestamp parser and formatter as current variant rows.
fn legacy_variant_time(conn: &Connection) -> Result<()> {
    conn.create_scalar_function(
        "polylogue_legacy_variant_time",
        1,
        rusqlite::functions::FunctionFlags::SQLITE_UTF8
            | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
        |context| {
            let millis: i64 = context.get(0)?;
            polylogue_model::Timestamp::from_unix_ms(millis)
                .map(|at| at.to_string())
                .ok_or_else(|| rusqlite::Error::UserFunctionError(Box::new(
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "legacy variant time is out of range"),
                )))
        },
    )?;
    Ok(())
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
            tx.execute_batch(&crate::source::Source::Ledger.sql(step))?;
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

/// Missing read indexes can be created without rebuilding parsed rows.
pub(crate) fn same_index_schema(conn: &Connection) -> Result<bool> {
    same_index_schema_in(conn, false)
}
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
pub(crate) fn same_build_schema(conn: &Connection) -> Result<bool> {
    same_build_schema_in(conn, false)
}
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
