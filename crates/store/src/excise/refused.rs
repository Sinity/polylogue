use crate::{Result, StoreError};
use polylogue_model::SessionId;
use rusqlite::Connection;
use serde::Deserialize;
use std::collections::HashSet;

mod load;

/// A present value must be an ID array.
pub(super) fn session_selection<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> std::result::Result<Option<Vec<SessionId>>, D::Error> {
    Vec::<SessionId>::deserialize(de).map(Some)
}

/// The writer's lookup of refused IDs, rebuilt from the excision rows into
/// an in-memory database.
#[derive(Debug, Default)]
pub struct Refused(Option<Connection>);

/// Only the excluded identities present in the current bounded writer group.
#[derive(Debug, Default)]
pub(crate) struct RefusedIds {
    pub sessions: HashSet<String>,
    pub messages: HashSet<String>,
    pub documents: HashSet<String>,
}

impl RefusedIds {
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty() && self.messages.is_empty() && self.documents.is_empty()
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
            if let Some(id) = scope.strip_prefix("input:") {
                out.insert("inputs", id)?;
            }
            let policy = row.get_ref(2)?;
            let policy = policy
                .as_str()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            let known = load::policy(&mut out, policy)?;
            if scope.starts_with("input:") && !known {
                return Err(StoreError::Invalid(format!(
                    "input excision {excision} has no list of the sessions it removed"
                )));
            }
            let ids = row.get_ref(1)?;
            load::messages(
                &mut out,
                ids.as_str()
                    .map_err(|e| StoreError::Invalid(e.to_string()))?,
            )?;
        }
        let mut selected =
            archive.prepare("SELECT document_id FROM excision_document ORDER BY document_id")?;
        for id in selected.query_map([], |r| r.get::<_, String>(0))? {
            let id = id?;
            polylogue_model::DocumentId::parse(&id)
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            out.insert("documents", &id)?;
        }
        if let Some(conn) = &out.0 {
            conn.execute_batch("COMMIT")?;
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub(crate) fn raw_import(
        &self,
        input: &str,
        hash: &polylogue_model::BlobHash,
        observation: &crate::Observation,
    ) -> Result<bool> {
        let Some(conn) = &self.0 else {
            return Ok(false);
        };
        let location = serde_json::to_string(&(&observation.machine, &observation.path))?;
        let hash = hash.to_string();
        Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM inputs WHERE id=?1)
            OR EXISTS(SELECT 1 FROM blobs WHERE id=?2) OR EXISTS(SELECT 1 FROM revisions WHERE id=?2)
            OR EXISTS(SELECT 1 FROM locations WHERE id=?3)",rusqlite::params![input,hash,location],|row|row.get(0))?)
    }

    pub(crate) fn matching(&self, sessions: &[String], messages: &[String]) -> Result<RefusedIds> {
        let mut out = RefusedIds::default();
        if let Some(conn) = &self.0 {
            Self::find(conn, "sessions", sessions, &mut out.sessions)?;
            Self::find(conn, "messages", messages, &mut out.messages)?;
        }
        Ok(out)
    }

    pub(crate) fn matching_documents(&self, documents: &[String]) -> Result<HashSet<String>> {
        let mut selected = HashSet::new();
        if let Some(conn) = &self.0 {
            Self::find(conn, "documents", documents, &mut selected)?;
        }
        Ok(selected)
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
            let conn = Connection::open_in_memory()?;
            conn.execute_batch(
                "PRAGMA journal_mode=OFF;
                 CREATE TABLE sessions(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE messages(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE documents(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE inputs(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE blobs(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE revisions(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 CREATE TABLE locations(id TEXT PRIMARY KEY) WITHOUT ROWID;
                 BEGIN;",
            )?;
            self.0 = Some(conn);
        }
        self.0
            .as_ref()
            .expect("created exclusion lookup")
            .prepare_cached(&format!("INSERT OR IGNORE INTO {table} VALUES(?1)"))?
            .execute([id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
