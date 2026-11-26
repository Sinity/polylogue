//! Source rows have one schema, read through views over the attached archive.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::sync::OnceLock;
mod confirmed;

/// The link made by the native SQLite decoder after storing its row stream.
/// A generic container, an import format hint or a failed decode does not make
/// the original a kept database.
pub(crate) fn database_rows(link: &str) -> String {
    format!(
        "{link}.kind = 'sqlite' AND {link}.member_index = 0
            AND {link}.path_bytes = CAST('rows.jsonl' AS BLOB)"
    )
}

/// Committed imported row streams and the originals they represent. The
/// caller can restrict the original by `container_input` on its existing index.
pub(crate) fn database_links() -> String {
    let rows = database_rows("member");
    format!(
        "SELECT member.container_input FROM source_observation_container member
         JOIN source_observation child ON child.id = member.observation_id
         WHERE {rows} AND child.class = 'imported' AND child.input <> member.container_input"
    )
}

pub(crate) fn database_original(input: &str) -> String {
    let links = database_links();
    format!("EXISTS({links} AND member.container_input = {input})")
}

/// The format an import read its input as, on an observation aliased `o`.
pub(crate) const IMPORT_FORMAT: &str = "json_extract(o.details_json, '$.import_format')";

/// The observations aliased `o` that state an import format.
pub(crate) const STATES_IMPORT_FORMAT: &str = "o.class = 'imported' AND json_valid(o.details_json)
    AND json_type(o.details_json, '$.import_format') = 'text'";

/// Which of an input's stated import formats its chains are parsed as: a
/// recognized format before the kept-file one, then the newest observation.
/// An input never has two recognized formats (`check_import_format`).
pub(crate) const IMPORT_FORMAT_ORDER: &str = "(json_extract(o.details_json, '$.import_format') = 'export-file'), o.commit_seq DESC, o.id DESC";

/// The format the chains of `input` are parsed as: the one an import read it
/// as, else `own`, the input's own format (SQL expressions).
pub(crate) fn parse_format(input: &str, own: &str) -> String {
    format!(
        "COALESCE((SELECT {IMPORT_FORMAT} FROM source_observation o
            WHERE o.input = {input} AND {STATES_IMPORT_FORMAT}
            ORDER BY {IMPORT_FORMAT_ORDER} LIMIT 1), {own})"
    )
}

pub(crate) struct SourceSchema {
    views: String,
    pub ddl: String,
}

/// SQLite owns identifier rewriting, including foreign keys and trigger bodies.
pub(crate) fn schema() -> Result<&'static SourceSchema> {
    static SCHEMA: OnceLock<SourceSchema> = OnceLock::new();
    if let Some(schema) = SCHEMA.get() {
        return Ok(schema);
    }
    let _ = SCHEMA.set(from_archive(&crate::schema::ARCHIVE)?);
    Ok(SCHEMA.get().expect("source schema initialized"))
}

pub(crate) fn from_archive(archive: &str) -> Result<SourceSchema> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(archive)?;
    let tables:Vec<String>=conn.prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?
        .query_map([],|row|row.get(0))?.collect::<rusqlite::Result<_>>()?;
    for name in &tables {
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(StoreError::internal(
                "source schema contains an invalid identifier",
            ));
        }
        conn.execute_batch(&format!(
            "ALTER TABLE \"{name}\" RENAME TO \"source_{name}\""
        ))?;
    }
    let declarations:Vec<(String,String,String)>=conn.prepare("SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,name")?
        .query_map([],|row| <(_, _, _)>::try_from(row))?.collect::<rusqlite::Result<_>>()?;
    let mut views = String::new();
    for name in &tables {
        let declaration: String = conn.scalar(
            "SELECT sql FROM sqlite_schema WHERE name=?1",
            [format!("source_{name}")],
        )?;
        let rowid = if declaration.to_ascii_uppercase().contains("WITHOUT ROWID") {
            ""
        } else {
            "rowid AS rowid, "
        };
        views.push_str(&format!(
            "CREATE TEMP VIEW source_{name} AS SELECT {rowid}* FROM a.\"{name}\";\n"
        ));
    }
    let mut ddl = String::new();
    for (kind, name, sql) in declarations {
        let sql = match kind.as_str() {
            "index" => sql.replacen(&format!("INDEX {name}"), &format!("INDEX source_{name}"), 1),
            "trigger" => sql.replacen(
                &format!("TRIGGER {name}"),
                &format!("TRIGGER source_{name}"),
                1,
            ),
            _ => sql,
        };
        ddl.push_str(&sql);
        ddl.push_str(";\n");
    }
    if tables.iter().any(|name| name == "row_count") {
        ddl.push_str("INSERT INTO source_row_count(name,n) VALUES ('input',0),('revision',0),('stored_bytes',0);\n");
    }
    Ok(SourceSchema { ddl, views })
}

/// Read SQL uses these logical names: views over the attached archive.
pub(crate) fn views(conn: &Connection) -> Result<()> {
    conn.execute_batch(&schema()?.views)?;
    Ok(())
}

/// The source state a capture read samples before provider I/O: the stored
/// observation at `source`. Rows with no commit position leave it unknown.
pub(crate) fn snapshot(
    conn: &Connection,
    source: &crate::capture_read::SourceSelector,
) -> Result<crate::capture_read::SourceSnapshot> {
    use crate::capture_read::SourceStatus;
    let (mut status, confirmed) = Source::confirmed_source(conn, source)?;
    if commit_position(conn)?.is_none() && status == SourceStatus::Ready {
        status = SourceStatus::MissingEvidence;
    }
    Ok(crate::capture_read::SourceSnapshot { status, confirmed })
}

/// The archive's commit position: its write counter, or 0 when it holds no
/// input yet; none when rows hold no position.
fn commit_position(conn: &Connection) -> Result<Option<u64>> {
    let source = Source::reading(conn)?;
    let position: Option<String> = conn.scalar_optional(
        &source.sql("SELECT value FROM @source.meta WHERE key='write_counter'"),
        [],
    )?;
    if let Some(position) = position {
        return position.parse::<u64>().map(Some).map_err(|_| {
            StoreError::stored_data("source commit position is not a nonnegative integer")
        });
    }
    let empty: bool = conn.scalar(
        &source.sql("SELECT NOT EXISTS(SELECT 1 FROM @source.input)"),
        [],
    )?;
    Ok(empty.then_some(0))
}

/// Explicit source markers distinguish table names from identically named columns.
#[derive(Clone, Copy)]
pub(crate) enum Source {
    Standalone,
    Cache,
}
impl Source {
    pub(crate) fn of(conn: &Connection) -> Result<Self> {
        Ok(if conn.scalar::<bool>("SELECT EXISTS(SELECT 1 FROM main.sqlite_schema WHERE type='table' AND name='source_input')",[])? { Self::Cache } else { Self::Standalone })
    }
    /// The commit position by which every stored extent covering the requested
    /// prefix was committed. A zero-length prefix covers no extent, so it is
    /// committed when the chain's earliest revision was; None when the chain has
    /// no revision, or when its extents do not cover the prefix.
    pub(crate) fn committed_through(
        conn: &Connection,
        input: &str,
        chain: i64,
        through: u64,
    ) -> Result<Option<u64>> {
        let source = Self::of(conn)?;
        let through = i64::try_from(through)
            .map_err(|_| StoreError::internal("source position exceeds SQLite's integer range"))?;
        let mut query = conn.prepare(&source.sql("SELECT offset,length,commit_seq FROM @source.extent WHERE input=?1 AND chain=?2 AND offset<?3 ORDER BY offset"))?;
        let mut rows = query.query(rusqlite::params![input, chain, through])?;
        let mut covered = 0i64;
        let mut committed = 0i64;
        while let Some(row) = rows.next()? {
            let offset: i64 = row.get(0)?;
            let length: i64 = row.get(1)?;
            let position: i64 = row.get(2)?;
            if length < 0 {
                return Err(StoreError::stored(1, "negative source extent length"));
            }
            if position < 0 {
                return Err(StoreError::stored(2, "negative source commit position"));
            }
            if offset != covered {
                return Ok(None);
            }
            covered = offset
                .checked_add(length)
                .or_stored("source extent endpoint overflow")?;
            committed = committed.max(position);
            if covered >= through {
                return Ok(Some(committed as u64));
            }
        }
        if through == 0 {
            let position: Option<i64> = conn.scalar(
                &source.sql(
                    "SELECT min(commit_seq) FROM @source.revision WHERE input=?1 AND chain=?2",
                ),
                rusqlite::params![input, chain],
            )?;
            return position
                .map(|position| {
                    u64::try_from(position).map_err(|_| {
                        crate::StoreError::stored(0, "negative source commit position")
                    })
                })
                .transpose();
        }
        Ok(None)
    }

    pub(crate) fn reading_commit(
        conn: &Connection,
        input: &str,
        chain: i64,
        through: u64,
    ) -> Result<crate::wire_shapes::ReadingCommit> {
        let source = Self::of(conn)?;
        let through = i64::try_from(through)
            .map_err(|_| StoreError::internal("source position exceeds SQLite's integer range"))?;
        let sequence = Self::committed_through(conn, input, chain, through as u64)?
            .ok_or_else(|| StoreError::internal("a reading has no committed source prefix"))?;
        let sequence = i64::try_from(sequence).map_err(|_| {
            StoreError::internal("source commit position exceeds SQLite's integer range")
        })?;
        // The location that captured the prefix: the input's latest
        // observation committed with or before its last byte. An append at an
        // unchanged location records none of its own.
        let observation = conn.query_row(&source.sql("SELECT o.id,o.seen_at FROM @source.observation o WHERE o.input=?1 AND o.commit_seq<=?2 ORDER BY o.commit_seq DESC, o.id DESC LIMIT 1"),rusqlite::params![input,sequence],|row| <(_, _)>::try_from(row)).optional()?;
        let (observation, observed_at_ms) =
            observation.map_or((None, None), |(id, time)| (Some(id), Some(time)));
        Ok(crate::wire_shapes::ReadingCommit {
            sequence,
            id: None,
            observation,
            observed_at_ms,
        })
    }

    /// Readers use the logical source namespace, including standalone TEMP
    /// views over the attached archive. Writers still select through `of`.
    pub(crate) fn reading(conn: &Connection) -> Result<Self> {
        if conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM temp.sqlite_schema WHERE name='source_input')",
            [],
            |row| row.get::<_, bool>(0),
        )? {
            Ok(Self::Cache)
        } else {
            Self::of(conn)
        }
    }

    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::Standalone => "",
            Self::Cache => "source_",
        }
    }

    pub(crate) fn sql(self, sql: &str) -> String {
        sql.replace("@source.", self.prefix())
    }
}

#[cfg(test)]
mod tests;
