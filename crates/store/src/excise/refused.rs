use crate::{Result, StoreError};
use polylogue_model::SessionId;
use rusqlite::Connection;
use serde::Deserialize;
use std::collections::HashSet;

mod load;

/// Omission is historical unknown; a present value must be an ID array.
pub(super) fn session_selection<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> std::result::Result<Option<Vec<SessionId>>, D::Error> {
    Vec::<SessionId>::deserialize(de).map(Some)
}

pub(super) fn legacy_empty(value: &str) -> bool {
    value
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .eq(b"[]".iter().copied())
}

#[derive(Debug)]
struct State {
    conn: Connection,
    _file: tempfile::NamedTempFile,
}

/// Reconstructed writer lookup; canonical excision rows own the policy.
#[derive(Debug, Default)]
pub struct Refused(Option<State>);

/// Only the excluded identities present in the current bounded writer group.
#[derive(Debug, Default)]
pub(crate) struct RefusedIds {
    pub sessions: HashSet<String>,
    pub messages: HashSet<String>,
}

impl RefusedIds {
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty() && self.messages.is_empty()
    }
}

impl Refused {
    /// A complete replacement, built before it can become the writer's lookup.
    pub fn load(archive: &Connection) -> Result<Self> {
        let mut out = Self::default();
        let mut stmt =
            archive.prepare("SELECT scope,message_ids_json,refused_json,id FROM excision")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let scope: String = row.get(0)?;
            let excision: String = row.get(3)?;
            if let Some(id) = scope.strip_prefix("session:") {
                out.insert("sessions", id)?;
            }
            let policy = row.get_ref(2)?;
            let policy = policy
                .as_str()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            let known = load::policy(&mut out, policy)?;
            if scope.starts_with("input:") && !known {
                return Err(StoreError::Invalid(format!(
                    "input excision {excision} has no retained session selection; recover its exclusion policy before collecting new inputs"
                )));
            }
            let ids = row.get_ref(1)?;
            load::messages(
                &mut out,
                ids.as_str()
                    .map_err(|e| StoreError::Invalid(e.to_string()))?,
            )?;
        }
        if let Some(state) = &out.0 {
            state.conn.execute_batch("COMMIT")?;
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub(crate) fn matching(&self, sessions: &[String], messages: &[String]) -> Result<RefusedIds> {
        let mut out = RefusedIds::default();
        if let Some(state) = &self.0 {
            Self::find(&state.conn, "sessions", sessions, &mut out.sessions)?;
            Self::find(&state.conn, "messages", messages, &mut out.messages)?;
        }
        Ok(out)
    }

    fn find(
        conn: &Connection,
        table: &str,
        ids: &[String],
        out: &mut HashSet<String>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        crate::fault::check("exclusion_cache.lookup", table)?;
        let mut stmt = conn.prepare_cached(&format!(
            "SELECT id FROM {table} WHERE id IN (SELECT value FROM json_each(?1))"
        ))?;
        for chunk in ids.chunks(512) {
            let json = serde_json::to_string(chunk)?;
            for id in stmt.query_map([json], |r| r.get::<_, String>(0))? {
                out.insert(id?);
            }
        }
        Ok(())
    }

    fn insert(&mut self, table: &str, id: &str) -> Result<()> {
        crate::fault::check("exclusion_cache.insert", id)?;
        if self.0.is_none() {
            let file = tempfile::NamedTempFile::new()?;
            let conn = Connection::open(file.path())?;
            conn.execute_batch(
                "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
                 PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                 CREATE TABLE sessions(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE messages(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 BEGIN;",
            )?;
            self.0 = Some(State { conn, _file: file });
        }
        self.0
            .as_ref()
            .expect("created exclusion lookup")
            .conn
            .prepare_cached(&format!("INSERT OR IGNORE INTO {table} VALUES(?1)"))?
            .execute([id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
