//! The declared Index1 additions, without replacing parsed rows or their cursors.
use super::*;

const LINEAGE: &str = include_str!("lineage.sql");

/// Construct only the known shapes that preceded the current declarations.
fn predecessor(primary: bool, lineage: bool, current_authored: bool) -> Result<Connection> {
    predecessor_in(primary, lineage, current_authored, true)
}

pub(super) fn predecessor_in(
    primary: bool,
    lineage: bool,
    current_authored: bool,
    native: bool,
) -> Result<Connection> {
    let expected = Connection::open_in_memory()?;
    crate::sql_functions::register(&expected)?;
    let index = if lineage {
        INDEX.to_owned()
    } else {
        INDEX.replacen(LINEAGE, "", 1).replacen(
            "  lineage           INTEGER NOT NULL DEFAULT 0,\n",
            "",
            1,
        )
    };
    expected.execute_batch(&index)?;
    expected.execute_batch(crate::coverage::SCHEMA)?;
    expected.execute_batch(crate::variants::SCHEMA)?;
    expected.execute_batch(crate::parser_context::SCHEMA)?;
    expected.execute_batch(crate::parser_context::VERSION_INDEX)?;
    expected.execute_batch(OPEN_INDEXES)?;
    if primary {
        let archive = if current_authored {
            ARCHIVE.to_owned()
        } else {
            ARCHIVE.replacen(
                crate::assertions::SCHEMA,
                include_str!("assertions_before_unknown_time.sql"),
                1,
            )
        };
        expected.execute_batch(&crate::source::from_archive(&archive)?.ddl)?;
        let projection = if native {
            crate::primary::projection::SCHEMA.to_owned()
        } else {
            crate::primary::projection::SCHEMA
                .replacen("ALTER TABLE parse_state ADD COLUMN native_publication BLOB;\n", "", 1)
                .replacen(", value_hash BLOB, material TEXT, record_hash TEXT, native_kind TEXT", "", 1)
                .replacen("CREATE INDEX primary_fact_live_material ON primary_fact(material,event) WHERE state='live';\n", "", 1)
                .replacen("CREATE INDEX primary_fact_value ON primary_fact(input,chain,reading_version,reference,value_hash,state);\n", "", 1)
                .replacen("CREATE INDEX primary_fact_reading_progress ON primary_fact(input,chain,reading_version,step_start,step_end) WHERE kind='reading_progress';\n", "", 1)
        };
        expected.execute_batch(&projection)?;
    }
    Ok(expected)
}

fn matches(conn: &Connection, expected: &Connection) -> Result<bool> {
    let found = shape(conn)?;
    let wanted = shape(expected)?;
    let building = building(conn)?;
    let optional: HashSet<_> = OPEN_INDEXES
        .split(';')
        .filter_map(|statement| {
            let line = statement
                .lines()
                .filter(|line| !line.trim_start().starts_with("--"))
                .collect::<Vec<_>>()
                .join(" ");
            line.split_whitespace().nth(5).map(str::to_owned)
        })
        .collect();
    Ok(found.iter().all(|item| wanted.contains(item))
        && wanted.iter().all(|item| {
            found.contains(item)
                || (item.0 == "index"
                    && (optional.contains(&item.1)
                        || building
                            && crate::writer::deferred_index_names().any(|name| name == item.1)))
        }))
}

/// Exact public primary Index1 shapes before original-material interpretation.
pub(super) fn before_native(conn: &Connection) -> Result<bool> {
    for lineage in [false, true] {
        for authored in [false, true] {
            if matches(conn, &predecessor_in(true, lineage, authored, false)?)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The caller's immediate transaction owns all changes. Unknown shapes are
/// refused; their saved progress is not grounds for an automatic rebuild.
pub(super) fn apply(conn: &Connection, primary: bool) -> Result<()> {
    let lineage: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='session_lineage' AND type='table')",
        [],
        |row| row.get(0),
    )?;
    let old_authored = matches(conn, &predecessor(primary, lineage, false)?)?;
    if !old_authored && !(primary && matches(conn, &predecessor(primary, lineage, true)?)?) {
        return Err(StoreError::NoMigration {
            database: "index",
            found: version(conn)?,
        });
    }
    if primary && old_authored {
        add_primary_authored(conn)?;
    }
    if !lineage {
        add_lineage(conn)?;
    }
    crate::fault::check(
        "index.additions_transition",
        conn.path().unwrap_or("memory"),
    )?;
    Ok(())
}

fn add_primary_authored(conn: &Connection) -> Result<()> {
    conn.execute_batch("ALTER TABLE source_assertion ALTER COLUMN created_at DROP NOT NULL")?;
    let current = Connection::open_in_memory()?;
    current.execute_batch(&crate::source::schema()?.ddl)?;
    let imports: String = current.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='table' AND name='source_assertion_import'",
        [],
        |row| row.get(0),
    )?;
    conn.execute_batch(&imports)?;
    Ok(())
}

fn add_lineage(conn: &Connection) -> Result<()> {
    // Only this small queue needs its column order brought to the current
    // declaration. Its integer primary keys are also its row identities.
    let expected = predecessor(false, true, false)?;
    let ddl: String = expected.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='table' AND name='derive_due'",
        [],
        |row| row.get(0),
    )?;
    let triggers = conn
        .prepare(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND instr(sql,'derive_due')>0",
        )?
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (name, _) in &triggers {
        conn.execute_batch(&format!("DROP TRIGGER \"{}\"", name.replace('"', "\"\"")))?;
    }
    conn.execute_batch("ALTER TABLE derive_due RENAME TO derive_due_before_lineage")?;
    conn.execute_batch(&ddl)?;
    conn.execute_batch(
        "INSERT INTO derive_due(session_pk,rows_changed,repo_lookup,rewritten,marked)
         SELECT session_pk,rows_changed,repo_lookup,rewritten,marked FROM derive_due_before_lineage;
         DROP TABLE derive_due_before_lineage;",
    )?;
    for (_, sql) in triggers {
        conn.execute_batch(&sql)?;
    }
    conn.execute_batch(LINEAGE)?;
    conn.execute_batch(
        "INSERT INTO derive_due(session_pk,rows_changed,lineage) SELECT pk,0,1 FROM session WHERE true
         ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
