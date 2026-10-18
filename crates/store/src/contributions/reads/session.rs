//! Session snapshots: the catalog of all sessions and one session's messages
//! in sequence order.
use super::{
    PinnedRead, SharedRead, locked, message::MessageView, metadata_admission, ordered_extras,
};
use crate::contributions::directory as d;
use crate::{Error, ReadHandle, Result, db};
use polylogue_model::{MessageId, Origin, SessionId, SessionKind};
use rusqlite::{OptionalExtension, params};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

/// Session metadata from the view's snapshot. Missing values stay `None`.
#[derive(Clone, Debug)]
pub struct SessionHeader {
    pub id: SessionId,
    pub origin: Origin,
    pub native_id: String,
    pub native_id_encoding: Option<String>,
    pub kind: SessionKind,
    pub title: Option<String>,
    pub cwd: Option<String>,
    pub repo: Option<String>,
    pub checkout: Option<String>,
    pub branch: Option<String>,
    pub commit: Option<String>,
    pub harness: Option<String>,
    pub harness_version: Option<String>,
    pub machine: Option<String>,
    pub team: Option<String>,
    pub agent_name: Option<String>,
    pub collection_id: Option<String>,
    pub collection_name: Option<String>,
    pub attrs: Option<polylogue_model::RawJson>,
    pub started_at: Option<i64>,
    pub last_activity_at: Option<i64>,
    pub stub: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionCoverage {
    /// Only messages already in the index were visited. The session may have
    /// more that are not prepared yet.
    KnownCatalogOnly,
}
pub struct SessionPosition<'a> {
    pub generation: &'a str,
    pub counter: u64,
}
pub struct SessionView {
    pub(super) read: SharedRead,
    pub(super) path: PathBuf,
    header: SessionHeader,
    pk: i64,
    generation: String,
    counter: u64,
    pub(super) archive_counter: u64,
}
/// A snapshot of all sessions, pinned together with the archive's input
/// catalog.
#[derive(Clone)]
pub struct SessionCatalog {
    pub(super) read: SharedRead,
    pub(super) path: PathBuf,
    generation: String,
    counter: u64,
    archive_counter: u64,
}
#[derive(Clone)]
pub struct SessionCatalogCursor {
    read: SharedRead,
    after: String,
}
impl SessionCatalog {
    pub fn recovery(&self) -> Result<Option<polylogue_model::RecoveryRecord>> {
        Ok(locked(&self.read)?.recovery.clone())
    }
    pub(super) fn visible(&self) -> Result<()> {
        let read = locked(&self.read)?;
        let archive_path = self.path.with_file_name("archive.db");
        if db(&archive_path, || {
            read.archive_visibility
                .as_ref()
                .unwrap()
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)")?
                .query_row([], |r| r.get::<_, bool>(0))
        })? {
            return Err(Error::ExcisionCheckRequired);
        }
        if !db(&self.path, || {
            read.visibility
                .prepare_cached(
                    "SELECT EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?1)",
                )?
                .query_row([&self.generation], |r| r.get::<_, bool>(0))
        })? {
            return Err(Error::StaleMessage);
        }
        Ok(())
    }
    pub fn position(&self) -> SessionPosition<'_> {
        SessionPosition {
            generation: &self.generation,
            counter: self.counter,
        }
    }
    pub fn source_archive_counter(&self) -> u64 {
        self.archive_counter
    }
    /// The next session in id order. A cursor from another catalog is refused.
    pub fn next(
        &self,
        after: Option<&SessionCatalogCursor>,
        allowance: usize,
    ) -> Result<Option<(SessionCatalogCursor, SessionView)>> {
        self.visible()?;
        let key = match after {
            None => "",
            Some(c) if Arc::ptr_eq(&c.read, &self.read) => &c.after,
            Some(_) => return Err(Error::StaleMessage),
        };
        let read = locked(&self.read)?;
        let row: Option<(u64, Option<String>)> = db(&self.path, || {
            read.connection.query_row(
            "SELECT length(CAST(id AS BLOB)),CASE WHEN length(CAST(id AS BLOB))<=?2 THEN id END FROM session WHERE id>?1 ORDER BY id LIMIT 1",
            params![key, i64::try_from(allowance).unwrap_or(i64::MAX)], |r| Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?))).optional()
        })?;
        drop(read);
        let Some((length, id)) = row else {
            return Ok(None);
        };
        let id = id.ok_or_else(|| metadata_admission("Session catalog identity", length))?;
        let cursor = SessionCatalogCursor {
            read: Arc::clone(&self.read),
            after: id.clone(),
        };
        let id = SessionId::from_stored(id).map_err(|e| Error::Invariant(e.to_string()))?;
        let session = self.session(&id, allowance)?.ok_or(Error::StaleMessage)?;
        Ok(Some((cursor, session)))
    }
}
/// Position in one session view. Messages are ordered by sequence; the row
/// key only lets the cursor resume and never decides order.
#[derive(Clone)]
pub struct SessionCursor {
    read: SharedRead,
    session: i64,
    sequence: i64,
    message: i64,
}
pub enum SessionStep {
    Ready {
        cursor: SessionCursor,
        sequence: u64,
        message: MessageView,
    },
    Pending {
        cursor: SessionCursor,
        sequence: u64,
        id: MessageId,
    },
    UnresolvedOrder {
        sequence: u64,
    },
    End {
        coverage: SessionCoverage,
    },
}
const SESSION_TEXT: [&str; 18] = [
    "origin",
    "native_id",
    "native_id_encoding",
    "kind",
    "title",
    "cwd",
    "repo",
    "checkout",
    "branch",
    "\"commit\"",
    "harness",
    "harness_version",
    "machine",
    "team",
    "agent_name",
    "collection_id",
    "collection_name",
    "attrs_json",
];
fn session_header_sql() -> String {
    let mut query = String::from("SELECT pk,required");
    for name in SESSION_TEXT {
        query.push_str(&format!(",CASE WHEN required<=?2 THEN {name} END"))
    }
    query.push_str(",started_at,last_activity_at,stub FROM (SELECT *,length(CAST(id AS BLOB))");
    for name in SESSION_TEXT {
        query.push_str(&format!("+COALESCE(length(CAST({name} AS BLOB)),0)"))
    }
    query.push_str(" AS required FROM session WHERE id=?1)");
    query
}
impl ReadHandle {
    pub fn selected_session(
        &self,
        id: &SessionId,
        allowance: usize,
    ) -> Result<Option<SessionView>> {
        self.session_catalog(allowance)?.session(id, allowance)
    }
    /// Open a catalog snapshot. The archive is pinned before the index, and
    /// if the archive changed in between the call fails with `StaleMessage`,
    /// so inputs stored but not yet registered in the index are never missed.
    pub fn session_catalog(&self, allowance: usize) -> Result<SessionCatalog> {
        let archive_path = self.root.join("archive.db");
        let archive = crate::open(&archive_path, true, true)?;
        db(&archive_path, || archive.execute_batch("BEGIN DEFERRED"))?;
        let archive_counter = crate::counter(&archive, &archive_path)?;
        let recovery = crate::recovery::record(&archive, &archive_path)?;
        let path = self.root.join("index.db");
        let connection = crate::open(&path, true, false)?;
        ordered_extras::configure(&connection, &path)?;
        db(&path, || connection.execute_batch("BEGIN DEFERRED"))?;
        let (generation_length, generation): (u64, Option<String>) = db(&path, || {
            connection.query_row("SELECT length(CAST(value AS BLOB)),CASE WHEN length(CAST(value AS BLOB))<=?1 THEN value END FROM meta WHERE key='generation'",[i64::try_from(allowance).unwrap_or(i64::MAX)],|row|Ok((crate::parser_state::unsigned(row,0)?,row.get(1)?)))
        })?;
        let generation = generation
            .ok_or_else(|| metadata_admission("session read generation", generation_length))?;
        let counter = crate::counter(&connection, &path)?;
        if crate::counter(&self.archive, &archive_path)? != archive_counter {
            return Err(Error::StaleMessage);
        }
        Ok(SessionCatalog {
            read: Arc::new(Mutex::new(PinnedRead {
                connection,
                visibility: crate::open(&path, true, false)?,
                archive: Some(archive),
                archive_visibility: Some(crate::open(&archive_path, true, true)?),
                recovery,
                extra_orders: 0,
                _lock: Arc::clone(&self._lock),
            })),
            path,
            generation,
            counter,
            archive_counter,
        })
    }
}
impl SessionCatalog {
    pub fn session(&self, id: &SessionId, allowance: usize) -> Result<Option<SessionView>> {
        if id.as_str().len() > allowance {
            return Err(metadata_admission(
                "session read identity",
                id.as_str().len() as u64,
            ));
        }
        self.visible()?;
        let read = locked(&self.read)?;
        let connection = &read.connection;
        let path = &self.path;
        let generation = &self.generation;
        let remaining = allowance.checked_sub(generation.len()).ok_or_else(|| {
            metadata_admission("session read generation", generation.len() as u64)
        })?;
        let row: Option<(
            i64,
            u64,
            [Option<String>; 18],
            Option<i64>,
            Option<i64>,
            bool,
        )> = db(&path, || {
            connection
                .query_row(
                    &session_header_sql(),
                    params![id.as_str(), i64::try_from(remaining).unwrap_or(i64::MAX)],
                    |r| {
                        Ok((
                            r.get(0)?,
                            crate::parser_state::unsigned(r, 1)?,
                            [
                                r.get(2)?,
                                r.get(3)?,
                                r.get(4)?,
                                r.get(5)?,
                                r.get(6)?,
                                r.get(7)?,
                                r.get(8)?,
                                r.get(9)?,
                                r.get(10)?,
                                r.get(11)?,
                                r.get(12)?,
                                r.get(13)?,
                                r.get(14)?,
                                r.get(15)?,
                                r.get(16)?,
                                r.get(17)?,
                                r.get(18)?,
                                r.get(19)?,
                            ],
                            r.get(20)?,
                            r.get(21)?,
                            r.get(22)?,
                        ))
                    },
                )
                .optional()
        })?;
        let Some((pk, required, text, started_at, last_activity_at, stub)) = row else {
            return Ok(None);
        };
        let [
            origin,
            native_id,
            native_id_encoding,
            kind,
            title,
            cwd,
            repo,
            checkout,
            branch,
            commit,
            harness,
            harness_version,
            machine,
            team,
            agent_name,
            collection_id,
            collection_name,
            attrs,
        ] = text;
        let origin = origin.ok_or_else(|| {
            metadata_admission(
                "session read metadata",
                required.saturating_add(generation.len() as u64),
            )
        })?;
        let origin = Origin::new(origin).map_err(|e| Error::Invariant(e.to_string()))?;
        let native_id =
            native_id.ok_or_else(|| Error::Invariant("session native_id missing".into()))?;
        if native_id.is_empty() {
            return Err(Error::Invariant("session native_id empty".into()));
        }
        polylogue_model::NativeId::from_wire(&native_id, native_id_encoding.as_deref())
            .map_err(|e| Error::Invariant(e.to_string()))?;
        let kind = match kind.as_deref() {
            Some("main") => SessionKind::Main,
            Some("subagent") => SessionKind::Subagent,
            Some("chat") => SessionKind::Chat,
            _ => return Err(Error::Invariant("selected Session kind unknown".into())),
        };
        let attrs = attrs
            .map(|text| {
                serde_json::from_str(&text)
                    .map_err(|e| Error::Invariant(format!("selected Session attrs JSON: {e}")))
            })
            .transpose()?;
        let header = SessionHeader {
            id: id.clone(),
            origin,
            native_id,
            native_id_encoding,
            kind,
            title,
            cwd,
            repo,
            checkout,
            branch,
            commit,
            harness,
            harness_version,
            machine,
            team,
            agent_name,
            collection_id,
            collection_name,
            attrs,
            started_at,
            last_activity_at,
            stub,
        };
        Ok(Some(SessionView {
            read: Arc::clone(&self.read),
            path: self.path.clone(),
            header,
            pk,
            generation: self.generation.clone(),
            counter: self.counter,
            archive_counter: self.archive_counter,
        }))
    }
}
impl SessionView {
    pub fn recovery(&self) -> Result<Option<polylogue_model::RecoveryRecord>> {
        Ok(locked(&self.read)?.recovery.clone())
    }
    pub fn catalog(&self) -> SessionCatalog {
        SessionCatalog {
            read: Arc::clone(&self.read),
            path: self.path.clone(),
            generation: self.generation.clone(),
            counter: self.counter,
            archive_counter: self.archive_counter,
        }
    }
    pub(super) fn visible(&self) -> Result<()> {
        let read = locked(&self.read)?;
        if let Some(archive) = &read.archive_visibility {
            let archive_path = self.path.with_file_name("archive.db");
            let excision: bool = db(&archive_path, || {
                archive
                    .prepare_cached("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)")?
                    .query_row([], |r| r.get(0))
            })?;
            if excision {
                return Err(Error::ExcisionCheckRequired);
            }
        }

        let visible: bool = db(&self.path, || {
            read.visibility.prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk=?1 AND id=?2) AND EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?3)")?.query_row(params![self.pk,self.header.id.as_str(),self.generation],|r|r.get(0))
        })?;
        if visible {
            Ok(())
        } else {
            Err(Error::StaleMessage)
        }
    }
    pub fn header(&self) -> Result<&SessionHeader> {
        self.visible()?;
        Ok(&self.header)
    }
    pub fn position(&self) -> SessionPosition<'_> {
        SessionPosition {
            generation: &self.generation,
            counter: self.counter,
        }
    }
    pub fn next(&self, after: Option<&SessionCursor>, allowance: usize) -> Result<SessionStep> {
        self.visible()?;
        let (sequence, message) = match after {
            None => (-1, 0),
            Some(cursor) if Arc::ptr_eq(&self.read, &cursor.read) && cursor.session == self.pk => {
                (cursor.sequence, cursor.message)
            }
            Some(_) => return Err(Error::StaleMessage),
        };
        let remaining = allowance
            .checked_sub(self.header.id.as_str().len())
            .ok_or_else(|| {
                metadata_admission(
                    "session Message relation",
                    self.header.id.as_str().len() as u64,
                )
            })?;
        let read = locked(&self.read)?;
        let row: Option<(
            i64,
            u64,
            u64,
            Option<String>,
            i64,
            Option<i64>,
            Option<i64>,
            Option<bool>,
            Option<i64>,
            Option<i64>,
        )> = db(&self.path, || {
            read.connection.query_row("SELECT m.pk,m.seq,length(CAST(m.id AS BLOB)),CASE WHEN length(CAST(m.id AS BLOB))<=?4 THEN m.id END,m.removal_epoch,m.selected_body_pk,b.pk,b.ready,b.block_root_pk,b.removal_epoch FROM message m LEFT JOIN message_body b ON b.pk=m.selected_body_pk AND b.message_pk=m.pk AND b.session_pk=m.session_pk WHERE m.session_pk=?1 AND (m.seq,m.pk)>(?2,?3) AND m.hidden=0 ORDER BY m.seq,m.pk LIMIT 1",params![self.pk,sequence,message,i64::try_from(remaining).unwrap_or(i64::MAX)],|r|Ok((r.get(0)?,crate::parser_state::unsigned(r,1)?,crate::parser_state::unsigned(r,2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?,r.get(9)?))).optional()
        })?;
        let Some((
            message,
            sequence,
            length,
            id,
            removal_epoch,
            selected,
            body,
            ready,
            root,
            body_epoch,
        )) = row
        else {
            return Ok(SessionStep::End {
                coverage: SessionCoverage::KnownCatalogOnly,
            });
        };
        let sequence_i64 = i64::try_from(sequence).map_err(|_| Error::Range)?;
        let ambiguous: bool = db(&self.path, || {
            read.connection.query_row("SELECT EXISTS(SELECT 1 FROM message WHERE session_pk=?1 AND seq=?2 AND pk<>?3 AND hidden=0)",params![self.pk,sequence_i64,message],|r|r.get(0))
        })?;
        if ambiguous {
            return Ok(SessionStep::UnresolvedOrder { sequence });
        }
        let id = id.ok_or_else(|| {
            metadata_admission(
                "session Message identity",
                length.saturating_add(self.header.id.as_str().len() as u64),
            )
        })?;
        let id = MessageId::from_stored(id, self.header.id.clone())
            .map_err(|e| Error::Invariant(e.to_string()))?;
        let visible: bool = db(&self.path, || {
            read.visibility.query_row("SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND session_pk=?2 AND id=?3 AND hidden=0 AND removal_epoch=?4) AND EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?5)",params![message,self.pk,id.as_str(),removal_epoch,self.generation],|r|r.get(0))
        })?;
        if !visible {
            return Err(Error::StaleMessage);
        }
        if selected.is_some() && body.is_none() {
            return Err(Error::Invariant(
                "selected Session Message has an invalid body relation".into(),
            ));
        }
        let cursor = SessionCursor {
            read: Arc::clone(&self.read),
            session: self.pk,
            sequence: i64::try_from(sequence).map_err(|_| Error::Range)?,
            message,
        };
        if let Some(body) =
            body.filter(|_| ready == Some(true) && body_epoch == Some(removal_epoch))
        {
            Ok(SessionStep::Ready {
                cursor,
                sequence,
                message: MessageView {
                    read: Arc::clone(&self.read),
                    path: self.path.clone(),
                    id,
                    owner: d::Owner {
                        message,
                        session: self.pk,
                    },
                    body,
                    removal_epoch,
                    root,
                    generation: self.generation.clone(),
                },
            })
        } else {
            Ok(SessionStep::Pending {
                cursor,
                sequence,
                id,
            })
        }
    }
}
