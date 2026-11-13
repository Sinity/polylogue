//! Current table declarations. The durable register converts its exact
//! immediate predecessor at exclusive open; other differences are refused.
//! Standalone Index1 additions retain parsed progress. Primary cache3 records
//! native original-material admission; its exact public Index1 predecessor rebuilds.
//! Unknown current shapes refuse.
//! Indexes from another format are set aside and rebuilt. Paid-vector
//! stores retain their bytes and gain declared additions at normal open.

use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;

mod index_transition;
mod request_hash_transition;

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
    include_str!("schema/session_values.sql"),
    "\n",
    include_str!("schema/lineage.sql")
);
#[cfg(test)]
pub(crate) const QUERY_OBJECTS: &str = concat!(
    include_str!("schema/query_objects.sql"),
    "\n",
    include_str!("schema/query_runs.sql")
);
/// Physical indexes created at open; the writer defers only its query-only subset.
pub(crate) const OPEN_INDEXES: &str = include_str!("schema/open_indexes.sql");
#[cfg(test)]
pub(crate) const INDEX_SCHEMA: &[&str] = &[
    INDEX,
    crate::coverage::SCHEMA,
    crate::variants::SCHEMA,
    crate::parser_context::SCHEMA,
    crate::parser_context::VERSION_INDEX,
    OPEN_INDEXES,
];
/// The vectors and failures tables of `embeddings.db`, as the first build at
/// the baseline created them.
const EMBEDDINGS_BASE: &str = include_str!("schema/embeddings.sql");
/// Each table `embeddings.db` gained after its first build, in the order they
/// were added. A released addition is never edited, since a file is matched
/// against the base and each prefix of this list to find what it lacks; a
/// schema change appends one. Only new tables, indexes and triggers can be
/// added: the vectors table is never rewritten, so a change to an existing
/// table is not an addition and needs its own plan.
const EMBEDDINGS_ADDITIONS: &[&str] = &[
    include_str!("schema/embedding_revision.sql"),
    include_str!("schema/query_vector.sql"),
];

/// The current durable register version.
pub const ARCHIVE_VERSION: u32 = 3;

/// The index schema and row encoding version. Reading versions are independent.
pub const INDEX_VERSION: u32 = 1;
/// The primary cache indexes original materials, native admission and canonical request hashes.
pub const PRIMARY_INDEX_VERSION: u32 = 3;

pub const fn index_version(primary: bool) -> u32 {
    if primary {
        PRIMARY_INDEX_VERSION
    } else {
        INDEX_VERSION
    }
}

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
    conn.execute_batch(crate::parser_context::VERSION_INDEX)?;
    Ok(())
}

fn create_index_in(conn: &Connection, primary: bool) -> Result<()> {
    base_index(conn)?;
    conn.execute_batch(OPEN_INDEXES)?;
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
    set_version(conn, index_version(primary))?;
    conn.pragma_update(None, "application_id", FORMAT_ID)?;
    Ok(())
}

/// Creates the index in an empty file, or opens one at this version with the
/// declared tables, adding any read index it lacks. Any other index is
/// refused unchanged. The declared Index1 additions retain parsed rows and
/// progress; an unknown Index1 shape is never reset at archive open.
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
    let wanted = index_version(primary);
    if found > wanted {
        return Err(StoreError::IndexVersion { found, wanted });
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    if found == 0 {
        create_index_in(&tx, primary)?;
    } else {
        // This reverse reference is needed by the writer, including a resumed
        // first build. Adding the index preserves rows and table identity.
        tx.execute_batch(crate::parser_context::VERSION_INDEX)?;
        // Physical indexes are added before the shape check, so an index missing
        // only those is completed rather than refused. A first build still in
        // progress leaves out the ones it defers: it builds each once from a
        // sorted scan when it ends, and building one here would read the
        // whole table at every restart.
        let deferring = building(&tx)?;
        let statements = OPEN_INDEXES
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
        let current = |conn: &Connection| {
            if building(conn)? {
                same_build_schema_in(conn, primary)
            } else {
                same_index_schema_in(conn, primary)
            }
        };
        if primary && found == 1 && index_transition::before_native(&tx)? {
            // Its native contexts lack material location and owner record hash.
            // The archive open preserves this exact predecessor and rebuilds
            // the cache from the confirmed originals, spool and paid stores.
            return Err(StoreError::IndexVersion { found, wanted });
        }
        if !current(&tx)? {
            index_transition::apply(&tx, primary)?;
            if !current(&tx)? {
                return Err(StoreError::Invalid(
                    "Index1 additions did not produce the current schema; the transaction was rolled back".into(),
                ));
            }
        }
        if primary && found < wanted {
            request_hash_transition::apply(&tx, true)?;
            set_version(&tx, wanted)?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// The schema identity of an index file: a hash of its version, its format
/// identity and the tables, views and triggers it holds, as statements
/// without comments or spacing. Indexes are left out: opening a file can add them,
/// and a first build defers query-only indexes, so a file keeps its identity through both.
pub(crate) fn index_identity(conn: &Connection) -> Result<String> {
    let mut hash = blake3::Hasher::new();
    hash.update(format!("version {} format {}\n", version(conn)?, format_id(conn)?).as_bytes());
    for (kind, name, sql) in shape(conn)? {
        if kind != "index" {
            for part in [kind, name, sql] {
                hash.update(part.as_bytes());
                hash.update(b"\0");
            }
        }
    }
    Ok(hash.finalize().to_hex()[..16].to_owned())
}

/// The schema identity of the index this build creates.
pub(crate) fn build_index_identity(primary: bool) -> Result<String> {
    let expected = Connection::open_in_memory()?;
    crate::sql_functions::register(&expected)?;
    base_index(&expected)?;
    if primary {
        add_primary(&expected)?;
    }
    set_version(&expected, index_version(primary))?;
    expected.pragma_update(None, "application_id", FORMAT_ID)?;
    index_identity(&expected)
}

#[cfg(any(test, feature = "fixture"))]
pub(crate) fn primary_before_originals() -> Result<Connection> {
    let conn = index_transition::predecessor_in(true, true, true, false)?;
    set_version(&conn, 1)?;
    conn.pragma_update(None, "application_id", FORMAT_ID)?;
    crate::archive::generation::create(&conn)?;
    conn.execute(
        "INSERT INTO meta VALUES ('primary_cache',?1)",
        [uuid::Uuid::now_v7().to_string()],
    )?;
    Ok(conn)
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

/// Creates the current durable register, or opens it under the archive's
/// exclusive lock. Exact predecessors gain unknown assertion times, source
/// memberships and validated canonical request hashes in one transaction.
pub fn ensure_archive(conn: &Connection) -> Result<()> {
    crate::sql_functions::register(conn)?;
    check_format(conn, "archive")?;
    let found = version(conn)?;
    if found > ARCHIVE_VERSION {
        return Err(StoreError::NewerSchema {
            database: "archive",
            found,
            current: ARCHIVE_VERSION,
        });
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    match found {
        0 => {
            tx.pragma_update(None, "application_id", FORMAT_ID)?;
            tx.execute_batch(ARCHIVE)?;
        }
        1 => {
            let previous = ARCHIVE.replacen(
                crate::assertions::SCHEMA,
                include_str!("schema/assertions_before_unknown_time.sql"),
                1,
            );
            if !same_schema(&tx, &[&previous])? {
                return Err(StoreError::NoMigration {
                    database: "archive",
                    found,
                });
            }
            relax_assertion_time(&tx)?;
        }
        _ if same_schema(&tx, &[ARCHIVE])? => {}
        _ => {
            return Err(StoreError::NoMigration {
                database: "archive",
                found,
            });
        }
    }
    if found != 0 && found < ARCHIVE_VERSION {
        request_hash_transition::apply(&tx, false)?;
    }
    if found != ARCHIVE_VERSION {
        set_version(&tx, ARCHIVE_VERSION)?;
    }
    tx.commit()?;
    Ok(())
}

fn relax_assertion_time(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    // Relax the constraint without rewriting assertion rows or their inbound
    // references. The register remains under the same immediate transaction.
    tx.execute_batch("ALTER TABLE assertion ALTER COLUMN created_at DROP NOT NULL")?;
    let current = Connection::open_in_memory()?;
    current.execute_batch(ARCHIVE)?;
    let provenance: String = current.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='table' AND name='assertion_import'",
        [],
        |r| r.get(0),
    )?;
    tx.execute_batch(&provenance)?;
    let file: String = tx.query_row(
        "SELECT file FROM pragma_database_list WHERE name='main'",
        [],
        |r| r.get(0),
    )?;
    crate::fault::check("archive.authored_time_transition", &file)?;
    if shape(tx)? != shape(&current)? {
        return Err(StoreError::Invalid(
            "authored-time transition did not produce the current register".into(),
        ));
    }
    Ok(())
}

/// Creates `embeddings.db` in an empty file, or opens one at the baseline.
/// A file whose schema is the base and an earlier prefix of
/// [`EMBEDDINGS_ADDITIONS`] gains the remaining additions in place, in one
/// transaction; its rows are not rewritten. Any other schema is refused
/// unchanged, naming how it differs from the nearest schema a build created.
pub(crate) fn ensure_embeddings(conn: &Connection) -> Result<()> {
    let current = 1;
    let found = version(conn)?;
    if found > current {
        return Err(StoreError::NewerSchema {
            database: "embeddings",
            found,
            current,
        });
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    if found == 0 {
        tx.execute_batch(EMBEDDINGS_BASE)?;
        for addition in EMBEDDINGS_ADDITIONS {
            tx.execute_batch(addition)?;
        }
        set_version(&tx, current)?;
    } else {
        let stored = shape(&tx)?;
        let earlier = embeddings_schemas()?;
        let Some(added) = earlier.iter().position(|schema| *schema == stored) else {
            return Err(StoreError::EmbeddingsSchemaDiffers {
                difference: nearest_difference(&stored, &earlier),
            });
        };
        for addition in &EMBEDDINGS_ADDITIONS[added..] {
            tx.execute_batch(addition)?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// The schema of `embeddings.db` after the base and each prefix of
/// [`EMBEDDINGS_ADDITIONS`], the last being the current one.
fn embeddings_schemas() -> Result<Vec<Vec<(String, String, String)>>> {
    let expected = Connection::open_in_memory()?;
    expected.execute_batch(EMBEDDINGS_BASE)?;
    let mut schemas = vec![shape(&expected)?];
    for addition in EMBEDDINGS_ADDITIONS {
        expected.execute_batch(addition)?;
        schemas.push(shape(&expected)?);
    }
    Ok(schemas)
}

/// How `stored` differs from the schema in `schemas` it is closest to, as
/// "table embedded differs; trigger t is missing; table x is not declared".
fn nearest_difference(
    stored: &[(String, String, String)],
    schemas: &[Vec<(String, String, String)>],
) -> String {
    let differences = |schema: &Vec<(String, String, String)>| {
        let mut out = Vec::new();
        for (kind, name, sql) in schema {
            match stored
                .iter()
                .find(|item| item.0 == *kind && item.1 == *name)
            {
                None => out.push(format!("{kind} {name} is missing")),
                Some(item) if item.2 != *sql => out.push(format!("{kind} {name} differs")),
                Some(_) => {}
            }
        }
        for (kind, name, _) in stored {
            if !schema.iter().any(|item| item.0 == *kind && item.1 == *name) {
                out.push(format!("{kind} {name} is not declared"));
            }
        }
        out
    };
    schemas
        .iter()
        .map(differences)
        .min_by_key(Vec::len)
        .unwrap_or_default()
        .join("; ")
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

/// Physical indexes created at open do not change the parsed table format.
fn same_index_schema_in(conn: &Connection, primary: bool) -> Result<bool> {
    let expected = Connection::open_in_memory()?;
    base_index(&expected)?;
    if primary {
        add_primary(&expected)?;
    }
    let base = shape(&expected)?;
    expected.execute_batch(OPEN_INDEXES)?;
    let expected = shape(&expected)?;
    let open_indexes: HashSet<_> = expected
        .iter()
        .filter(|item| item.0 == "index" && !base.contains(item))
        .map(|item| item.1.clone())
        .collect();
    let found = shape(conn)?;
    Ok(found.iter().all(|item| expected.contains(item))
        && expected.iter().all(|item| {
            (item.0 == "index" && open_indexes.contains(&item.1)) || found.contains(item)
        }))
}

/// A pending first build may lack only the indexes the writer deferred.
fn same_build_schema_in(conn: &Connection, primary: bool) -> Result<bool> {
    let expected = Connection::open_in_memory()?;
    base_index(&expected)?;
    expected.execute_batch(OPEN_INDEXES)?;
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
