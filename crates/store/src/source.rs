//! Source rows have one schema, in the standalone ledger or the primary cache.
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use std::sync::OnceLock;

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
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(crate::schema::ARCHIVE)?;
    let tables:Vec<String>=conn.prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?
        .query_map([],|row|row.get(0))?.collect::<rusqlite::Result<_>>()?;
    for name in &tables {
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(StoreError::Invalid(
                "source schema contains an invalid identifier".into(),
            ));
        }
        conn.execute_batch(&format!(
            "ALTER TABLE \"{name}\" RENAME TO \"source_{name}\""
        ))?;
    }
    let declarations:Vec<(String,String,String)>=conn.prepare("SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,name")?
        .query_map([],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?.collect::<rusqlite::Result<_>>()?;
    let mut views = String::new();
    for name in &tables {
        let declaration: String = conn.query_row(
            "SELECT sql FROM sqlite_schema WHERE name=?1",
            [format!("source_{name}")],
            |row| row.get(0),
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
    ddl.push_str("INSERT INTO source_row_count(name,n) VALUES ('input',0),('revision',0),('stored_bytes',0);\n");
    let _ = SCHEMA.set(SourceSchema { ddl, views });
    Ok(SCHEMA.get().expect("source schema initialized"))
}

/// Read SQL uses these logical names in both modes. Primary tables live in main.
pub(crate) fn views(conn: &Connection) -> Result<()> {
    conn.execute_batch(&schema()?.views)?;
    Ok(())
}

/// Explicit source markers distinguish table names from identically named columns.
#[derive(Clone, Copy)]
pub(crate) enum Source {
    Ledger,
    Cache,
}
impl Source {
    pub(crate) fn of(conn: &Connection) -> Result<Self> {
        Ok(if conn.query_row("SELECT EXISTS(SELECT 1 FROM main.sqlite_schema WHERE type='table' AND name='source_input')",[],|row|row.get::<_,bool>(0))? { Self::Cache } else { Self::Ledger })
    }
    /// Actual source admission for every retained extent covering the requested prefix.
    pub(crate) fn admitted_through(
        conn: &Connection,
        input: &str,
        chain: i64,
        through: u64,
    ) -> Result<Option<u64>> {
        let source = Self::of(conn)?;
        let through = i64::try_from(through).map_err(|_| {
            StoreError::Invalid("source position exceeds SQLite's integer range".into())
        })?;
        let mut query = conn.prepare(&source.sql("SELECT offset,length,commit_seq FROM @source.extent WHERE input=?1 AND chain=?2 AND offset<?3 ORDER BY offset"))?;
        let mut rows = query.query(rusqlite::params![input, chain, through])?;
        let mut covered = 0i64;
        let mut admitted = 0i64;
        while let Some(row) = rows.next()? {
            let offset: i64 = row.get(0)?;
            let length: i64 = row.get(1)?;
            let position: i64 = row.get(2)?;
            if offset != covered || length < 0 || position < 0 {
                return Ok(None);
            }
            covered = offset
                .checked_add(length)
                .ok_or_else(|| StoreError::Invalid("source extent endpoint overflow".into()))?;
            admitted = admitted.max(position);
            if covered >= through {
                return Ok(Some(admitted as u64));
            }
        }
        if through == 0 {
            let position: Option<i64> = conn.query_row(
                &source.sql(
                    "SELECT min(commit_seq) FROM @source.revision WHERE input=?1 AND chain=?2",
                ),
                rusqlite::params![input, chain],
                |row| row.get(0),
            )?;
            return position
                .map(|position| {
                    u64::try_from(position).map_err(|_| {
                        StoreError::Invalid("negative source admission position".into())
                    })
                })
                .transpose();
        }
        Ok(None)
    }

    pub(crate) fn reading_receipt(
        conn: &Connection,
        input: &str,
        chain: i64,
        through: u64,
    ) -> Result<crate::wire_shapes::ReadingReceipt> {
        let source = Self::of(conn)?;
        let through = i64::try_from(through).map_err(|_| {
            StoreError::Invalid("source position exceeds SQLite's integer range".into())
        })?;
        let sequence =
            Self::admitted_through(conn, input, chain, through as u64)?.ok_or_else(|| {
                StoreError::Invalid("a reading has no committed source prefix".into())
            })?;
        let sequence = i64::try_from(sequence).map_err(|_| {
            StoreError::Invalid("source admission exceeds SQLite's integer range".into())
        })?;
        let observation = conn.query_row(&source.sql("SELECT o.id,o.seen_at FROM @source.observation o LEFT JOIN @source.revision r ON r.input=o.input AND r.number=o.revision WHERE o.input=?1 AND coalesce(o.chain,r.chain)=?2 AND o.length>=?3 ORDER BY o.commit_seq DESC,o.rowid DESC LIMIT 1"),rusqlite::params![input,chain,through],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
        let (observation, observed_at_ms) =
            observation.map_or((None, None), |(id, time)| (Some(id), Some(time)));
        Ok(crate::wire_shapes::ReadingReceipt {
            sequence,
            id: None,
            observation,
            observed_at_ms,
        })
    }

    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::Ledger => "",
            Self::Cache => "source_",
        }
    }

    pub(crate) fn sql(self, sql: &str) -> String {
        sql.replace(
            "@source.",
            match self {
                Self::Ledger => "",
                Self::Cache => "source_",
            },
        )
    }
}

#[cfg(test)]
mod tests;
