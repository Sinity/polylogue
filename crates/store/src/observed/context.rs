//! The stored context (role and name) of an observed Claude input. It lives in
//! the archive, so it does not depend on parser state or the provider's
//! current directory layout.
use crate::{Error, InputFormat, Result, db};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;

fn validate(context: &[String]) -> Result<()> {
    InputFormat::ClaudeCodeJsonl
        .validate_context(&context.iter().map(String::as_str).collect::<Vec<_>>())
}
pub(crate) fn read(conn: &Connection, file: &Path, input: &str) -> Result<Vec<String>> {
    let row: Option<(i64, Option<String>)> = db(file, || {
        conn.query_row(
        "SELECT version,CASE WHEN length(CAST(context_json AS BLOB))<=65536 THEN context_json END FROM observed_context WHERE input=?1",
        [input],|r|Ok((r.get(0)?,r.get(1)?))).optional()
    })?;
    let Some((1, Some(text))) = row else {
        return Err(Error::ObservedSourceRefused(
            "observed context is missing or has an unknown version".into(),
        ));
    };
    let context: Vec<String> = serde_json::from_str(&text)
        .map_err(|_| Error::ObservedSourceRefused("invalid observed context".into()))?;
    validate(&context)?;
    Ok(context)
}
pub(crate) fn insert(
    conn: &Connection,
    file: &Path,
    input: &str,
    context: &[String],
) -> Result<()> {
    validate(context)?;
    let text = serde_json::to_string(context).map_err(|e| Error::Invariant(e.to_string()))?;
    db(file, || {
        conn.execute(
            "INSERT INTO observed_context(input,version,context_json) VALUES(?1,1,?2)",
            params![input, text],
        )
    })?;
    Ok(())
}
/// Archive upgrade only: derive each context from the input's first observed
/// path, as older versions did. Never a fallback for a missing context.
pub(crate) fn migrate(conn: &Connection, file: &Path) -> Result<()> {
    let mut q = db(file, || {
        conn.prepare("SELECT i.id,(SELECT path FROM observation o WHERE o.input=i.id AND o.class='first' ORDER BY o.rowid LIMIT 1) FROM input i WHERE i.original_hash IS NULL AND i.format='claude-code-jsonl' ORDER BY i.id")
    })?;
    let mut rows = db(file, || q.query([]))?;
    while let Some(row) = db(file, || rows.next())? {
        let id: String = db(file, || row.get(0))?;
        let path: Option<String> = db(file, || row.get(1))?;
        let Some(context) = path
            .as_deref()
            .and_then(|p| Path::new(p).file_stem())
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .map(|s| vec!["claude-main".into(), s.into()])
        else {
            continue;
        };
        // An input whose context cannot be derived keeps none.
        if validate(&context).is_ok() {
            insert(conn, file, &id, &context)?;
        }
    }
    Ok(())
}
