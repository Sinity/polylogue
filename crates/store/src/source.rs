//! Source rows have one schema, in the standalone ledger or the primary cache.
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::sync::OnceLock;

pub(crate) struct SourceSchema {
    pub tables: Vec<String>,
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
    let _ = SCHEMA.set(SourceSchema { tables, ddl, views });
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
