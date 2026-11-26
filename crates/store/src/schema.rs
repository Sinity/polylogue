//! Table declarations. Each database has one current format: an empty file
//! is created at the current version with exactly the tables declared here,
//! and a file at that version with these tables opens. Any other file is
//! refused unchanged; nothing converts it. An index refused this way is
//! replaced by the archive's open, since it is derived from stored bytes.
//! `embeddings.db` holds paid vectors and is never replaced: a file written by
//! an earlier build gains the tables added since ([`ensure_embeddings`]).

use crate::{Result, StoreError};
use rusqlite::{Connection, fallible_iterator::FallibleIterator as _};
use std::collections::HashSet;
use std::sync::LazyLock;

pub static ARCHIVE: LazyLock<String> = LazyLock::new(|| {
    with_names(concat!(
        include_str!("schema/archive.sql"),
        "\n",
        include_str!("schema/archive_result_index.sql"),
        "\n",
        include_str!("schema/client_requests.sql"),
        "\n",
        include_str!("schema/annotations.sql"),
        "\n",
        include_str!("schema/alternates.sql"),
        "\n",
        include_str!("schema/query_objects.sql"),
        "\n",
        include_str!("schema/query_runs.sql"),
        "\n",
        include_str!("schema/signals.sql")
    ))
});
pub static INDEX: LazyLock<String> = LazyLock::new(|| {
    with_names(concat!(
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
        include_str!("schema/transcript_lines.sql"),
        "\n",
        include_str!("schema/summary.sql"),
        "\n",
        include_str!("schema/lineage.sql")
    ))
});
#[cfg(test)]
pub(crate) static QUERY_OBJECTS: LazyLock<String> = LazyLock::new(|| {
    with_names(concat!(
        include_str!("schema/query_objects.sql"),
        "\n",
        include_str!("schema/query_runs.sql")
    ))
});

/// The closed sets the schema text lists. `@names(Type)` in a schema file
/// becomes that type's `NAMES` as a quoted SQL list when the text is
/// assembled, so a `CHECK` list and the enum it holds are one declaration.
const NAMED_SETS: &[(&str, &[&str])] = &[
    ("Produced", crate::input_rows::Produced::NAMES),
    (
        "ObservationClass",
        crate::input_rows::ObservationClass::NAMES,
    ),
    ("DeliveryOutcome", polylogue_model::DeliveryOutcome::NAMES),
    ("RunsHere", polylogue_model::authored::RunsHere::NAMES),
    (
        "QueryEdgeKind",
        polylogue_model::query_objects::QueryEdgeKind::NAMES,
    ),
    ("Precision", &PRECISIONS),
    ("Route", polylogue_model::Route::NAMES),
    ("IdKind", polylogue_model::batch::IdKind::NAMES),
    ("Completion", polylogue_model::Completion::NAMES),
    ("RoutedFieldGroup", &ROUTED_FIELD_GROUPS),
    ("Party", crate::measure::time::Party::NAMES),
    ("SessionKind", polylogue_model::SessionKind::NAMES),
    ("Role", polylogue_model::Role::NAMES),
    ("WrittenBy", polylogue_model::WrittenBy::NAMES),
    ("PromptKind", polylogue_model::PromptKind::NAMES),
    ("BlockKind", polylogue_model::BlockKind::NAMES),
    ("ToolAction", polylogue_model::ToolAction::NAMES),
    ("CallOutcome", polylogue_model::CallOutcome::NAMES),
    ("LinkKind", polylogue_model::LinkKind::NAMES),
    ("ReferenceReason", polylogue_model::ReferenceReason::NAMES),
    ("ChainFailureKind", polylogue_model::ChainFailureKind::NAMES),
];

/// The field groups `message_route` records: every group but the content,
/// whose route is the message's main route.
const ROUTED_FIELD_GROUPS: [&str; polylogue_model::FieldGroup::NAMES.len() - 1] = {
    let all = polylogue_model::FieldGroup::ALL;
    let mut names = [""; polylogue_model::FieldGroup::NAMES.len() - 1];
    let (mut i, mut n) = (0, 0);
    while i < all.len() {
        if !matches!(all[i], polylogue_model::FieldGroup::Content) {
            names[n] = all[i].as_str();
            n += 1;
        }
        i += 1;
    }
    names
};

/// The precisions a stored time states, from the shared query language's
/// closed set.
const PRECISIONS: [&str; polylogue_model::Precision::ALL.len()] = {
    let mut names = [""; polylogue_model::Precision::ALL.len()];
    let mut i = 0;
    while i < names.len() {
        names[i] = polylogue_model::Precision::ALL[i].as_str();
        i += 1;
    }
    names
};

/// Schema text with each `@names(Type)` replaced by that type's names.
pub(crate) fn with_names(sql: &str) -> String {
    const OPEN: &str = "@names(";
    let mut out = String::with_capacity(sql.len());
    let mut rest = sql;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        let tail = &rest[start + OPEN.len()..];
        let end = tail
            .find(')')
            .expect("a schema names placeholder is closed");
        let set = &tail[..end];
        let (_, names) = NAMED_SETS
            .iter()
            .find(|(name, _)| *name == set)
            .unwrap_or_else(|| panic!("the schema names an undeclared set {set}"));
        let quoted: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
        out.push_str(&quoted.join(","));
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}
/// The triggers that keep search membership and the trigram index with the
/// shown rows. A build file is created without them; the build creates them
/// once its composition and search entries are written.
pub(crate) const SEARCH_TRIGGERS: &str = include_str!("schema/search_triggers.sql");

fn search_trigger_names() -> impl Iterator<Item = &'static str> {
    SEARCH_TRIGGERS
        .lines()
        .filter_map(|line| line.strip_prefix("CREATE TRIGGER "))
        .filter_map(|tail| tail.split_whitespace().next())
}

/// Read indexes added at open, or once the private build ends.
pub(crate) const OPEN_INDEXES: &str = include_str!("schema/open_indexes.sql");

/// Indexes written once from sorted rows when a private build ends. The
/// natural-key indexes the writer looks rows up by stay throughout.
pub(crate) const BUILT_AT_END: &[&str] = &[
    "text_normalized",
    "message_time",
    "message_period",
    "message_marked",
    "tool_call_time",
    "event_time",
    "document_time",
    "event_period",
    "session_last_active",
    "session_started",
    "session_cwd",
    "session_repo",
    "session_live",
    "tool_call_action",
    "tool_call_repo_command",
    "event_kind",
    "removed_session_changed",
];
#[cfg(test)]
pub(crate) static INDEX_SCHEMA: LazyLock<[&str; 5]> = LazyLock::new(|| {
    [
        INDEX.as_str(),
        crate::coverage::SCHEMA,
        crate::alternates::SCHEMA,
        SEARCH_TRIGGERS,
        OPEN_INDEXES,
    ]
});
/// The `embedded` (vectors) and `embed_failure` tables of `embeddings.db`, as the first build at
/// the baseline created them.
const EMBEDDINGS_BASE: &str = include_str!("schema/embeddings.sql");
/// Each table `embeddings.db` gained after its first build, in the order they
/// were added. A released addition is never edited, since a file is matched
/// against the base and each prefix of this list to find what it lacks; a
/// schema change appends one. Only new tables, indexes and triggers can be
/// added: the `embedded` table is never rewritten, so a change to an existing
/// table is not an addition and needs its own plan.
const EMBEDDINGS_ADDITIONS: &[&str] = &[
    include_str!("schema/embedding_revision.sql"),
    include_str!("schema/query_vector.sql"),
];

/// The register format version.
pub const ARCHIVE_VERSION: u32 = 9;

/// The index schema and row encoding version. Reading versions are independent.
pub const INDEX_VERSION: u32 = 23;

/// The SQLite `application_id` of an `archive.db` or `index.db` written at or
/// after the baseline ("PLG1"). A versioned file without it was written by an
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
    if found == 0 && !schema_rows(conn)?.is_empty() {
        return Err(StoreError::OtherFormat { database, found });
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

/// Executes complete SQLite statements, leaving only the declared build
/// indexes out when `deferring`. SQLite splits triggers and quoted SQL without another parser.
fn execute_index_sql(conn: &Connection, sql: &str, deferring: bool) -> Result<()> {
    if !deferring {
        conn.execute_batch(sql)?;
        return Ok(());
    }
    let mut batch = rusqlite::Batch::new(conn, sql);
    while let Some(mut prepared) = batch.next()? {
        let sql = prepared
            .expanded_sql()
            .ok_or_else(|| StoreError::internal("a schema statement has no SQL"))?;
        let normalized = statement(&sql);
        let index = normalized
            .strip_prefix("CREATE UNIQUE INDEX ")
            .or_else(|| normalized.strip_prefix("CREATE INDEX "))
            .map(|tail| tail.strip_prefix("IF NOT EXISTS ").unwrap_or(tail))
            .and_then(|tail| tail.split_whitespace().next());
        if !index.is_some_and(|name| BUILT_AT_END.contains(&name)) {
            prepared.execute([])?;
        }
    }
    Ok(())
}

/// The exact declared index statements, in the order a build writes them.
pub(crate) fn built_at_end_sql() -> Result<Vec<String>> {
    let expected = Connection::open_in_memory()?;
    base_index(&expected)?;
    expected.execute_batch(OPEN_INDEXES)?;
    BUILT_AT_END
        .iter()
        .map(|name| {
            let sql: String = expected.query_row(
                "SELECT sql FROM sqlite_schema WHERE type='index' AND name=?1",
                [name],
                |row| row.get(0),
            )?;
            Ok(sql
                .replacen(
                    "CREATE UNIQUE INDEX ",
                    "CREATE UNIQUE INDEX IF NOT EXISTS ",
                    1,
                )
                .replacen("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1))
        })
        .collect()
}

/// Creates the index in an empty database outside a transaction, with its
/// full-text files attached.
pub fn create_index(conn: &Connection) -> Result<()> {
    create_index_in(conn, false)?;
    crate::search::files::attach(conn)
}

fn base_index(conn: &Connection) -> Result<()> {
    base_index_in(conn, false)
}

fn base_index_in(conn: &Connection, deferring: bool) -> Result<()> {
    execute_index_sql(conn, &INDEX, deferring)?;
    conn.execute_batch(crate::coverage::SCHEMA)?;
    conn.execute_batch(crate::alternates::SCHEMA)?;
    if !deferring {
        conn.execute_batch(SEARCH_TRIGGERS)?;
    }
    Ok(())
}

fn create_index_in(conn: &Connection, deferring: bool) -> Result<()> {
    base_index_in(conn, deferring)?;
    crate::search::files::create(conn, deferring)?;
    execute_index_sql(conn, OPEN_INDEXES, deferring)?;
    crate::archive::generation::create(conn)?;
    set_version(conn, INDEX_VERSION)?;
    conn.pragma_update(None, "application_id", FORMAT_ID)?;
    Ok(())
}

/// Creates the index in an empty file, or opens one at this version with the
/// declared tables, adding any read index it lacks. Any other index is
/// refused unchanged; the archive's open builds a new one in its place
/// ([`crate::archive::index_build`]).
pub fn ensure_index(conn: &Connection) -> Result<()> {
    ensure_index_in(conn, false)
}
/// [`ensure_index`] for a build in progress, which leaves out the
/// indexes it builds when it ends ([`crate::archive::index_build`]).
pub fn ensure_build_index(conn: &Connection) -> Result<()> {
    ensure_index_in(conn, true)
}
/// The version of an index this build can open (0 for an empty file), or
/// the refusal of one from another format or version.
fn index_version_of(conn: &Connection) -> Result<u32> {
    check_format(conn, "index")?;
    let found = version(conn)?;
    let wanted = INDEX_VERSION;
    if found != 0 && found != wanted {
        return Err(StoreError::IndexVersion { found, wanted });
    }
    Ok(found)
}

/// What [`ensure_index`] refuses in an index, checked without changing the
/// file: for an index a build set aside, why it was.
pub(crate) fn check_standalone_index(conn: &Connection) -> Result<()> {
    let found = index_version_of(conn)?;
    if found != 0 && !same_index_schema_in(conn)? {
        return Err(StoreError::OtherFormat {
            database: "index",
            found,
        });
    }
    Ok(())
}

fn ensure_index_in(conn: &Connection, deferring: bool) -> Result<()> {
    // ALTER TABLE validates views that read packed bodies through these functions.
    crate::sql_functions::register(conn)?;
    let found = index_version_of(conn)?;
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
    if found == 0 {
        create_index_in(&tx, deferring)?;
    } else {
        // Read indexes are added before the shape check, so an index missing
        // only those is completed rather than refused. A build file is always
        // created fresh, so only an installed index is opened here.
        if deferring {
            return Err(StoreError::Invalid(
                "a build file is created, never opened again".into(),
            ));
        }
        execute_index_sql(&tx, OPEN_INDEXES, false)?;
        if !same_index_schema_in(&tx)? {
            return Err(StoreError::OtherFormat {
                database: "index",
                found,
            });
        }
    }
    tx.commit()?;
    // A build's writer reaches its files only once the build ends.
    if !deferring {
        crate::search::files::attach(conn)?;
    }
    Ok(())
}

/// The schema identity of an index file: a hash of its version, its format
/// identity and the tables, views and triggers it holds, as statements
/// without comments or spacing. Indexes and the search triggers are left out:
/// opening a file can add indexes, and a first build defers query-only
/// indexes and the search triggers, so a file keeps its identity through both.
pub(crate) fn index_identity(conn: &Connection) -> Result<String> {
    let mut hash = blake3::Hasher::new();
    hash.update(format!("version {} format {}\n", version(conn)?, format_id(conn)?).as_bytes());
    for (kind, name, sql) in schema_rows(conn)? {
        if kind != "index" && !(kind == "trigger" && search_trigger_names().any(|n| n == name)) {
            for part in [kind, name, sql] {
                hash.update(part.as_bytes());
                hash.update(b"\0");
            }
        }
    }
    Ok(hash.finalize().to_hex()[..16].to_owned())
}

/// The schema identity of the index this build creates.
pub(crate) fn build_index_identity() -> Result<String> {
    let expected = Connection::open_in_memory()?;
    crate::sql_functions::register(&expected)?;
    base_index(&expected)?;
    set_version(&expected, INDEX_VERSION)?;
    expected.pragma_update(None, "application_id", FORMAT_ID)?;
    index_identity(&expected)
}

/// Creates the register in an empty file, or opens one at this version with
/// the declared tables. Any other register is refused unchanged.
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
    if found == 0 {
        let tx =
            rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
        tx.pragma_update(None, "application_id", FORMAT_ID)?;
        tx.execute_batch(&ARCHIVE)?;
        crate::meta_set(&tx, "archive_id", uuid::Uuid::new_v4().simple().to_string())?;
        set_version(&tx, ARCHIVE_VERSION)?;
        tx.commit()?;
        return Ok(());
    }
    if found != ARCHIVE_VERSION || !same_schema(conn, &[&ARCHIVE])? {
        return Err(StoreError::OtherFormat {
            database: "archive",
            found,
        });
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
        let stored = schema_rows(&tx)?;
        let earlier = embeddings_schemas()?;
        let Some(applied) = earlier.iter().position(|schema| *schema == stored) else {
            return Err(StoreError::EmbeddingsSchemaDiffers {
                difference: nearest_difference(&stored, &earlier),
            });
        };
        for addition in &EMBEDDINGS_ADDITIONS[applied..] {
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
    let mut schemas = vec![schema_rows(&expected)?];
    for addition in EMBEDDINGS_ADDITIONS {
        expected.execute_batch(addition)?;
        schemas.push(schema_rows(&expected)?);
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
    Ok(schema_rows(conn)? == schema_rows(&expected)?)
}

/// Indexes created at open do not change the parsed table format.
fn same_index_schema_in(conn: &Connection) -> Result<bool> {
    let expected = Connection::open_in_memory()?;
    base_index(&expected)?;
    let base = schema_rows(&expected)?;
    expected.execute_batch(OPEN_INDEXES)?;
    let expected = schema_rows(&expected)?;
    let open_indexes: HashSet<_> = expected
        .iter()
        .filter(|item| item.0 == "index" && !base.contains(item))
        .map(|item| item.1.clone())
        .collect();
    let found = schema_rows(conn)?;
    Ok(found.iter().all(|item| expected.contains(item))
        && expected.iter().all(|item| {
            (item.0 == "index" && open_indexes.contains(&item.1)) || found.contains(item)
        }))
}

fn schema_rows(conn: &Connection) -> Result<Vec<(String, String, String)>> {
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

/// Tables in `index.db` and their column names.
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
