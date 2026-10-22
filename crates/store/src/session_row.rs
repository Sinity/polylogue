//! The one mapping between the model's [`Session`] and its `session` row,
//! both ways. The derive step writes a session's merged facts through it,
//! and reads and exports read them back through it, so a `Session` field
//! is stored and read in one place.

use crate::Result;
use crate::read::ts;
use polylogue_model::{Session, SessionId, Timestamp, TitleSource};
use rusqlite::{Connection, Row, Transaction, params};

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

/// Remove a session's derived rows. Raw inputs, authored values and embeddings
/// are owned by other databases and are not changed here. Returns text and blob
/// references for callers performing explicit content removal.
pub(crate) fn remove(
    i: &Transaction<'_>,
    pk: i64,
    id: &str,
    counter: i64,
) -> Result<(Vec<i64>, Vec<Vec<u8>>)> {
    i.execute("DELETE FROM snapshot WHERE pk IN (SELECT snapshot_pk FROM snapshot_session WHERE session_pk=?1)", [pk])?;
    let text_ids: Vec<i64> = i
        .prepare(
            "SELECT DISTINCT text_id FROM block WHERE session_pk = ?1 AND text_id IS NOT NULL
             UNION SELECT v.text_id FROM version_block v JOIN message_version mv ON mv.pk=v.version_pk JOIN message_source ms ON ms.version_hash=mv.hash JOIN message m ON m.pk=ms.message_pk WHERE (m.session_pk=?1 OR ms.session_pk=?1) AND v.text_id IS NOT NULL",
        )?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let blobs: Vec<Vec<u8>> = i
        .prepare(
            "SELECT DISTINCT blob_hash FROM block WHERE session_pk = ?1 AND blob_hash IS NOT NULL
             UNION SELECT v.blob_hash FROM version_block v JOIN message_version mv ON mv.pk=v.version_pk JOIN message_source ms ON ms.version_hash=mv.hash JOIN message m ON m.pk=ms.message_pk WHERE (m.session_pk=?1 OR ms.session_pk=?1) AND v.blob_hash IS NOT NULL",
        )?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let copies:Vec<i64>=i.prepare("SELECT DISTINCT s.message_pk FROM message_source s JOIN message m ON m.pk=s.message_pk WHERE s.session_pk=?1 AND m.session_pk<>?1")?.query_map([pk],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
    i.execute("DELETE FROM message_source WHERE session_pk=?1 OR message_pk IN (SELECT pk FROM message WHERE session_pk=?1)",[pk])?;
    // Trigram entries before their tool calls.
    let calls: Vec<(i64, Option<String>, Option<String>)> = i
        .prepare("SELECT pk, command, path FROM tool_call WHERE session_pk = ?1")?
        .query_map([pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (cpk, command, path) in &calls {
        i.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
             VALUES ('delete', ?1, ?2, ?3)",
            params![cpk, command, path],
        )?;
    }
    // Every table that names the session, its messages or its tool calls,
    // whichever component added it.
    let tables = crate::schema::index_columns(i)?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("tool_call_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE tool_call_pk IN
                     (SELECT pk FROM tool_call WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("message_pk") && !cols.contains("session_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE message_pk IN
                     (SELECT pk FROM message WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    i.execute("DELETE FROM tool_call WHERE session_pk = ?1", [pk])?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" {
            continue;
        }
        if cols.contains("session_pk") {
            i.execute(&format!("DELETE FROM {table} WHERE session_pk = ?1"), [pk])?;
        }
    }
    i.execute(
        "DELETE FROM link WHERE parent_id = ?1 OR child_id = ?1",
        [id],
    )?;
    i.execute("DELETE FROM message WHERE session_pk = ?1", [pk])?;
    i.execute("DELETE FROM session WHERE pk = ?1", [pk])?;
    crate::writer::promote_current(i, &copies, counter)?;
    crate::versions::prune(i)?;
    Ok((text_ids, blobs))
}
