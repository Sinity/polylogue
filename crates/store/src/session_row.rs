//! The one mapping between the model's [`Session`] and its `session` row,
//! both ways. The derive step writes a session's merged facts through it,
//! and reads and exports read them back through it, so a `Session` field
//! is stored and read in one place.

use crate::Result;
use crate::encoding::Columns;
use polylogue_model::{Session, SessionId, SessionKind, Title, TitleSource};
use rusqlite::{Connection, Row, params};

/// The columns [`read`] expects, in order, for a `SELECT`.
pub(crate) const COLUMNS: &str = "kind, title, title_source, cwd_json, branch, record_repo, harness_version, started_time, mode, git_commit, model_provider, instructions";

/// Writes `s`'s facts into the row `pk`; `cwd` is the last directory.
pub(crate) fn write(conn: &Connection, pk: i64, s: &Session) -> Result<()> {
    let started = crate::encoding::TimeColumns::of(s.started, s.time_accuracy());
    conn.prepare_cached(
        "UPDATE session SET kind = ?2, title = ?3, title_source = ?4, cwd_json = ?5, cwd = ?6,
             branch = ?7, record_repo = ?8, harness_version = ?9, started_at = ?10, mode = ?11, git_commit = ?12, model_provider = ?13, instructions = ?14, started_time = ?15,
             started_until = ?16, started_precision = ?17
         WHERE pk = ?1",
    )?
    .execute(params![
        pk,
        s.kind.as_str(),
        s.title.as_ref().map(|t| t.text.as_str()),
        s.title.as_ref().map(|t| t.source.as_str()),
        serde_json::to_string(&s.cwd)?,
        s.cwd.last(),
        s.git_branch,
        s.repo,
        s.tool_version,
        started.at,
        s.mode,
        s.git_commit,
        s.model_provider,
        s.instructions,
        crate::encoding::time_bytes(s.started,s.time_accuracy())?,
        started.until,
        started.precision_name(),
    ])?;
    Ok(())
}

/// The session `id` from a row selecting [`COLUMNS`].
pub(crate) fn read(id: SessionId, r: &Row<'_>) -> rusqlite::Result<Session> {
    let mut s = Session::new(id, r.name::<SessionKind>(0)?);
    let title: Option<String> = r.get(1)?;
    let source = r.optional_name::<TitleSource>(2)?;
    if let (Some(text), Some(source)) = (title, source) {
        s.title = Some(Title { text, source });
    }
    s.cwd = r
        .get::<_, Option<String>>(3)?
        .map(|c| serde_json::from_str(&c).map_err(|e| unreadable_json(3, e)))
        .transpose()?
        .unwrap_or_default();
    s.git_branch = r.get(4)?;
    s.repo = r.get(5)?;
    s.tool_version = r.get(6)?;
    let (at, accuracy) = crate::encoding::read_time(r, 7)?;
    s.started = at;
    s.started_quality = accuracy.quality;
    s.started_precision = accuracy.precision;
    s.started_until = accuracy.until;
    s.mode = r.get(8)?;
    s.git_commit = r.get(9)?;
    s.model_provider = r.get(10)?;
    s.instructions = r.get(11)?;
    Ok(s)
}

fn unreadable_json(column: usize, error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(column, rusqlite::types::Type::Text, Box::new(error))
}

mod remove;
pub(crate) use remove::remove;
