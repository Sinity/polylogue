//! Grant changes replace cached facts while private producer intents keep their custody.
use crate::{Archive, Result, StoreError};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};

pub(crate) fn current(index: &Connection) -> Result<Option<String>> {
    Ok(index
        .query_row(
            "SELECT value FROM meta WHERE key='confirmed_scope'",
            [],
            |row| row.get(0),
        )
        .optional()?)
}

pub(crate) fn require_readable(index: &Connection) -> Result<()> {
    if index
        .query_row(
            "SELECT 1 FROM meta WHERE key='confirmed_scope_pending'",
            [],
            |_| Ok(()),
        )
        .optional()?
        .is_some()
    {
        return Err(StoreError::ScopeChanged);
    }
    Ok(())
}

fn quoted(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

pub(crate) fn reset(
    index: &mut Connection,
    archive: &Archive,
    resolver: &crate::writer::resolve::Resolver,
    scope: &str,
) -> Result<bool> {
    if scope.is_empty() {
        return Err(StoreError::Invalid(
            "a confirmed scope needs its owner identity".into(),
        ));
    }
    if current(index)?.as_deref() == Some(scope) && require_readable(index).is_ok() {
        return Ok(false);
    }
    // Cancellation precedes visibility. A failed transaction leaves reads blocked
    // until the same authoritative reset can be completed.
    archive.read_scope.begin_reset()?;
    // The owner control must survive a failed clear and restart. This is cached
    // grant metadata, independent of pending producer intents or input custody.
    let previous: i64 = index.pragma_query_value(None, "synchronous", |row| row.get(0))?;
    index.pragma_update(None, "synchronous", previous.max(2))?;
    let result = (|| {
        index.execute("INSERT INTO meta(key,value) VALUES ('confirmed_scope_pending',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [scope])?;
        clear(index, archive, resolver, scope)
    })();
    let restored = index.pragma_update(None, "synchronous", previous);
    result.and_then(|changed| restored.map(|_| changed).map_err(Into::into))
}

fn clear(
    index: &mut Connection,
    archive: &Archive,
    resolver: &crate::writer::resolve::Resolver,
    scope: &str,
) -> Result<bool> {
    resolver.discard_cached()?;
    index.flush_prepared_statement_cache();
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.pragma_update(None, "defer_foreign_keys", true)?;
    let temporary: Vec<(String,String)> = tx.prepare("SELECT name,type FROM temp.sqlite_schema WHERE type IN ('view','table') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC")?.query_map([], |row| Ok((row.get(0)?,row.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
    for (name, kind) in temporary {
        tx.execute_batch(&format!("DROP {kind} {}", quoted(&name)))?;
    }
    let tables: Vec<String> = tx.prepare("SELECT name FROM pragma_table_list WHERE schema='main' AND type='table' AND name NOT LIKE 'sqlite_%' AND name NOT IN ('meta','row_count','source_row_count') ORDER BY name")?.query_map([], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?;
    for name in &tables {
        tx.execute(&format!("DELETE FROM {}", quoted(name)), [])?;
    }
    // Deletion triggers queue derived garbage. The whole cache is being reset,
    // so clear their queues after their producers, retaining the table seeds.
    for name in tables.iter().filter(|name| name.starts_with("changed_")) {
        tx.execute(&format!("DELETE FROM {}", quoted(name)), [])?;
    }
    let fts: Vec<String> = tx.prepare("SELECT name FROM main.sqlite_schema WHERE type='table' AND lower(sql) LIKE '%using fts5(%'")?.query_map([], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?;
    for name in fts {
        tx.execute(
            &format!(
                "INSERT INTO {}({}) VALUES ('delete-all')",
                quoted(&name),
                quoted(&name)
            ),
            [],
        )?;
    }
    tx.execute("UPDATE row_count SET n=0", [])?;
    tx.execute("UPDATE source_row_count SET n=0", [])?;
    tx.execute(
        "DELETE FROM meta WHERE key NOT IN ('primary_cache','boot_id','generation','cursor_key')",
        [],
    )?;
    crate::archive::generation::rotate(&tx)?;
    tx.execute(
        "INSERT INTO meta(key,value) VALUES ('confirmed_scope',?1)",
        [scope],
    )?;
    crate::fault::check("primary.scope_reset_commit", "")?;
    tx.commit()?;
    archive.read_scope.finish_reset();
    Ok(true)
}

#[cfg(test)]
mod tests;
