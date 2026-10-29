//! The one mapping between the model's [`Session`] and its `session` row,
//! both ways. The derive step writes a session's merged facts through it,
//! and reads and exports read them back through it, so a `Session` field
//! is stored and read in one place.

use crate::Result;
use crate::read::ts;
use polylogue_model::{Session, SessionId, Timestamp, TitleSource};
use rusqlite::{Connection, Row, params};

/// The columns [`read`] expects, in order, for a `SELECT`.
pub(crate) const COLUMNS: &str = "kind, title, title_source, cwd_json, branch, record_repo, harness_version, started_at, mode, git_commit, model_provider, instructions";

/// Writes `s`'s facts into the row `pk`; `cwd` is the last directory.
pub(crate) fn write(conn: &Connection, pk: i64, s: &Session) -> Result<()> {
    conn.prepare_cached(
        "UPDATE session SET kind = ?2, title = ?3, title_source = ?4, cwd_json = ?5, cwd = ?6,
             branch = ?7, record_repo = ?8, harness_version = ?9, started_at = ?10, mode = ?11, git_commit = ?12, model_provider = ?13, instructions = ?14
         WHERE pk = ?1",
    )?
    .execute(params![
        pk,
        s.kind.as_str(),
        s.title.as_ref().map(|t| t.text.as_str()),
        s.title.as_ref().map(|t| name(&t.source)),
        serde_json::to_string(&s.cwd)?,
        s.cwd.last(),
        s.git_branch,
        s.repo,
        s.tool_version,
        s.started.map(Timestamp::unix_ms),
        s.mode,
        s.git_commit,
        s.model_provider,
        s.instructions,
    ])?;
    Ok(())
}

/// The session `id` from a row selecting [`COLUMNS`].
pub(crate) fn read(id: SessionId, r: &Row<'_>) -> rusqlite::Result<Session> {
    let mut s = Session::new(id, decode_name(&r.get::<_, String>(0)?, 0)?);
    let title: Option<String> = r.get(1)?;
    let source: Option<String> = r.get(2)?;
    if let (Some(text), Some(source)) = (
        title,
        source
            .as_deref()
            .map(|v| decode_name::<TitleSource>(v, 2))
            .transpose()?,
    ) {
        s.offer_title(&text, source);
    }
    s.cwd = r
        .get::<_, Option<String>>(3)?
        .map(|c| serde_json::from_str(&c).map_err(|e| invalid(3, e)))
        .transpose()?
        .unwrap_or_default();
    s.git_branch = r.get(4)?;
    s.repo = r.get(5)?;
    s.tool_version = r.get(6)?;
    s.started = ts(r.get(7)?);
    s.mode = r.get(8)?;
    s.git_commit = r.get(9)?;
    s.model_provider = r.get(10)?;
    s.instructions = r.get(11)?;
    Ok(s)
}

fn name<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::new(),
    }
}

fn invalid(column: usize, error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(column, rusqlite::types::Type::Text, Box::new(error))
}

fn decode_name<T: serde::de::DeserializeOwned>(s: &str, column: usize) -> rusqlite::Result<T> {
    serde_json::from_value(serde_json::Value::String(s.to_owned())).map_err(|e| invalid(column, e))
}

mod remove;
pub(crate) use remove::remove;
