//! The one mapping between the model's [`Session`] and its `session` row,
//! both ways. The derive step writes a session's merged facts through it,
//! and reads and exports read them back through it, so a `Session` field
//! is stored and read in one place.

use crate::Result;
use crate::read::{parse_name, ts};
use polylogue_model::{Session, SessionId, SessionKind, Timestamp, TitleSource};
use rusqlite::{Connection, Row, params};

/// The columns [`read`] expects, in order, for a `SELECT`.
pub(crate) const COLUMNS: &str =
    "kind, title, title_source, cwd_json, branch, record_repo, harness_version, started_at, mode";

/// Writes `s`'s facts into the row `pk`; `cwd` is the last directory.
pub(crate) fn write(conn: &Connection, pk: i64, s: &Session) -> Result<()> {
    conn.prepare_cached(
        "UPDATE session SET kind = ?2, title = ?3, title_source = ?4, cwd_json = ?5, cwd = ?6,
             branch = ?7, record_repo = ?8, harness_version = ?9, started_at = ?10, mode = ?11
         WHERE pk = ?1",
    )?
    .execute(params![
        pk,
        name(&s.kind),
        s.title.as_ref().map(|t| t.text.as_str()),
        s.title.as_ref().map(|t| name(&t.source)),
        serde_json::to_string(&s.cwd)?,
        s.cwd.last(),
        s.git_branch,
        s.repo,
        s.tool_version,
        s.started.map(Timestamp::unix_ms),
        s.mode,
    ])?;
    Ok(())
}

/// The session `id` from a row selecting [`COLUMNS`].
pub(crate) fn read(id: SessionId, r: &Row<'_>) -> rusqlite::Result<Session> {
    let mut s = Session::new(
        id,
        parse_name(&r.get::<_, String>(0)?).unwrap_or(SessionKind::Main),
    );
    let title: Option<String> = r.get(1)?;
    let source: Option<String> = r.get(2)?;
    if let (Some(text), Some(source)) = (title, source.as_deref().and_then(parse_name::<TitleSource>))
    {
        s.offer_title(&text, source);
    }
    s.cwd = r
        .get::<_, Option<String>>(3)?
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_default();
    s.git_branch = r.get(4)?;
    s.repo = r.get(5)?;
    s.tool_version = r.get(6)?;
    s.started = ts(r.get(7)?);
    s.mode = r.get(8)?;
    Ok(s)
}

fn name<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::new(),
    }
}
