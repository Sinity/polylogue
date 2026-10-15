//! One SQLite snapshot selects the whole immutable Message incarnation. Sparse
//! order-statistic reads and payload ranges never switch to a newer head midway.
use super::directory as d;
use crate::{Error, ReadHandle, Result, db, values};
use polylogue_model::{BlockSlot, MessageId, Origin, Role, SessionId, SessionKind, WrittenBy};
use rusqlite::{Connection, OptionalExtension, params};
use std::{
    fs::File,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

/// Pending is distinct from an absent reference: retained parseable input may
/// not yet have a selected whole Message body.
pub enum MessageLookup {
    Absent,
    Pending,
    Ready(MessageView),
}
pub struct MessageHeader {
    pub role: Role,
    pub written_by: WrittenBy,
    pub in_progress: Option<bool>,
}
struct PinnedRead {
    connection: Connection,
    visibility: Connection,
    archive: Option<Connection>,
    archive_visibility: Option<Connection>,
    _lock: Arc<File>,
}
type SharedRead = Arc<Mutex<PinnedRead>>;
mod dependencies;
pub use dependencies::{SessionDependencyCursor, SessionInputDependency};
fn locked(read: &SharedRead) -> Result<MutexGuard<'_, PinnedRead>> {
    read.lock()
        .map_err(|_| Error::Invariant("selected read snapshot mutex poisoned".into()))
}
pub struct MessageView {
    read: SharedRead,
    path: PathBuf,
    id: MessageId,
    owner: d::Owner,
    body: i64,
    removal_epoch: i64,
    root: Option<i64>,
    generation: String,
}
/// A payload reference is private and usable only through its originating
/// Message snapshot. It cannot be substituted into a newer Message body.
pub struct BlockView<'a> {
    message: &'a MessageView,
    pk: i64,
    pub slot: BlockSlot,
    pub group: u64,
    pub local: u64,
    pub kind: String,
    pub encrypted: Option<bool>,
    field_root: Option<i64>,
    value: Option<values::ValueId>,
}
pub enum FieldRead<'a> {
    Value(FieldView<'a>),
    /// Occupied unresolved conflict, never a missing slot available to fill.
    /// Detailed witness traversal/export remains a separate required consumer.
    Conflict,
}
pub struct FieldView<'a> {
    message: &'a MessageView,
    value: values::ValueId,
    origin: Option<values::ValueId>,
    pub supplied: bool,
    pub source_time: Option<i64>,
}
impl ReadHandle {
    /// Resolve the opaque Message key through its stored Session relation.
    /// Native colons never acquire parsing authority.
    pub fn message_reference(&self, id: &str, allowance: usize) -> Result<MessageLookup> {
        if id.len() > allowance {
            return Err(Error::ParserAdmission {
                resource: "message read reference",
                required: id.len() as u64,
            });
        }
        let path = self.root.join("index.db");
        let row: Option<(u64, Option<String>, bool)> = db(&path, || {
            self.index.query_row(
            "SELECT length(CAST(s.id AS BLOB)),CASE WHEN length(CAST(s.id AS BLOB))<=?2 THEN s.id ELSE NULL END,EXISTS(SELECT 1 FROM message_body b WHERE b.pk=m.selected_body_pk AND b.ready=1) FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.id=?1 AND m.hidden=0",
            params![id,i64::try_from(allowance.saturating_sub(id.len())).unwrap_or(i64::MAX)],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?,r.get(2)?))).optional()
        })?;
        let Some((length, session, ready)) = row else {
            return Ok(MessageLookup::Absent);
        };
        let session = session.ok_or(Error::ParserAdmission {
            resource: "message read relation",
            required: length.saturating_add(id.len() as u64),
        })?;
        if !ready {
            return Ok(MessageLookup::Pending);
        }
        let session =
            SessionId::from_stored(session).map_err(|e| Error::Invariant(e.to_string()))?;
        let id =
            MessageId::from_stored(id, session).map_err(|e| Error::Invariant(e.to_string()))?;
        self.selected_message(&id)?
            .map(MessageLookup::Ready)
            .ok_or(Error::MessagePredecessorChanged)
    }
    pub fn selected_message(&self, id: &MessageId) -> Result<Option<MessageView>> {
        let path = self.root.join("index.db");
        let connection = crate::open(&path, true, false)?;
        db(&path, || connection.execute_batch("BEGIN DEFERRED"))?;
        // This read establishes the pinned snapshot before any body traversal.
        let generation = db(&path, || {
            connection.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
                r.get(0)
            })
        })?;
        let selected: Option<(i64, i64, i64, Option<i64>, i64)> = db(&path, || {
            connection.query_row(
            "SELECT m.pk,m.session_pk,b.pk,b.block_root_pk,m.removal_epoch FROM message m JOIN session s ON s.pk=m.session_pk JOIN message_body b ON b.pk=m.selected_body_pk AND b.message_pk=m.pk AND b.session_pk=m.session_pk AND b.ready=1 WHERE m.id=?1 AND s.id=?2 AND m.hidden=0",
            params![id.as_str(), id.session().as_str()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).optional()
        })?;
        let Some((message, session, body, root, removal_epoch)) = selected else {
            return Ok(None);
        };
        Ok(Some(MessageView {
            read: Arc::new(Mutex::new(PinnedRead {
                connection,
                visibility: crate::open(&path, true, false)?,
                archive: None,
                archive_visibility: None,
                _lock: Arc::clone(&self._lock),
            })),
            path,
            id: id.clone(),
            owner: d::Owner { message, session },
            body,
            removal_epoch,
            root,
            generation,
        }))
    }
}
impl MessageView {
    pub fn provider_time(&self) -> Result<Option<i64>> {
        search_time(self)
    }
    // Content replacement preserves this pinned old incarnation. Privacy/removal
    // invalidates access immediately through the current scalar epoch instead.
    fn visible(&self) -> Result<()> {
        let read = locked(&self.read)?;
        let visible: bool = db(&self.path, || {
            read.visibility.query_row(
            "SELECT EXISTS(SELECT 1 FROM message m WHERE m.pk=?1 AND m.session_pk=?2 AND m.id=?3 AND m.hidden=0 AND m.removal_epoch=?4) AND EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?5)",
            params![self.owner.message,self.owner.session,self.id.as_str(),self.removal_epoch,self.generation],|r|r.get(0))
        })?;
        if !visible {
            return Err(Error::MessagePredecessorChanged);
        }
        Ok(())
    }

    pub fn id(&self) -> &MessageId {
        &self.id
    }
    pub fn body_id(&self) -> super::bodies::BodyId {
        super::bodies::BodyId {
            path: self.path.clone(),
            generation: self.generation.clone(),
            pk: self.body,
            message: self.owner.message,
            session: self.owner.session,
        }
    }
    pub fn header(&self) -> Result<MessageHeader> {
        self.visible()?;
        let read = locked(&self.read)?;
        let (role, by, in_progress): (String, String, Option<bool>) = db(&self.path, || {
            read.connection.query_row(
                "SELECT role,written_by,in_progress FROM message_body WHERE pk=?1 AND ready=1",
                [self.body],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
        })?;
        let role = match role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "tool" => Role::Tool,
            _ => return Err(Error::Invariant("unknown stored Message role".into())),
        };
        let written_by = match by.as_str() {
            "human" => WrittenBy::Human,
            "model" => WrittenBy::Model,
            "harness" => WrittenBy::Harness,
            "unknown" => WrittenBy::Unknown,
            _ => return Err(Error::Invariant("unknown stored Message author".into())),
        };
        Ok(MessageHeader {
            role,
            written_by,
            in_progress,
        })
    }
    pub fn body_pk(&self) -> i64 {
        self.body
    }
    pub fn field(
        &self,
        slot: &polylogue_model::merge::Slot,
        allowance: usize,
    ) -> Result<Option<FieldRead<'_>>> {
        self.visible()?;
        use polylogue_model::merge::Slot;
        if let Slot::Block { block, .. }
        | Slot::Extra {
            block: Some(block), ..
        } = slot
        {
            let requested = match block {
                BlockSlot::Item(i) => BlockSlot::Item(*i),
                BlockSlot::Reasoning(i) => BlockSlot::Reasoning(*i),
                BlockSlot::Attachment(_) => {
                    return Err(Error::Invariant(
                        "attachment selection is not implemented".into(),
                    ));
                }
            };
            return match self.block(requested)? {
                Some(block) => block.field(slot, allowance),
                None => Ok(None),
            };
        }
        let key = super::writer::field_key(slot, allowance)?;
        let read = locked(&self.read)?;
        let root: Option<i64> = db(&self.path, || {
            read.connection.query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1 AND ready=1",
                [self.body],
                |r| r.get(0),
            )
        })?;
        drop(read);
        self.field_at(root, &key)
    }
    fn field_at(&self, root: Option<i64>, key: &[u8]) -> Result<Option<FieldRead<'_>>> {
        let read = locked(&self.read)?;
        let slot: Option<i64> = db(&self.path, || {
            read.connection
                .query_row(
                    "SELECT pk FROM message_field_slot WHERE message_pk=?1 AND semantic_key=?2",
                    params![self.owner.message, key],
                    |r| r.get(0),
                )
                .optional()
        })?;
        let Some(slot) = slot else { return Ok(None) };
        let edge = d::lookup(
            &read.connection,
            &self.path,
            self.owner,
            d::Kind::Fields,
            root,
            u64::try_from(slot).map_err(|_| Error::Range)?,
            &mut d::Work::default(),
        )?;
        match edge.map(|e| e.link) {
            None => Ok(None),
            Some(d::Link::Conflict { slot: stored, .. }) if stored == slot => {
                Ok(Some(FieldRead::Conflict))
            }
            Some(d::Link::Field(fact)) => {
                let (value, origin, supplied, source_time): (i64, Option<i64>, bool, Option<i64>) =
                    db(&self.path, || {
                        read.connection.query_row("SELECT value_pk,origin_value_pk,supplied,source_time FROM message_field_fact WHERE pk=?1 AND slot_pk=?2 AND message_pk=?3 AND session_pk=?4",params![fact,slot,self.owner.message,self.owner.session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
                    })?;
                Ok(Some(FieldRead::Value(FieldView {
                    message: self,
                    value: values::ValueId {
                        pk: value,
                        index_generation: self.generation.clone(),
                        index_path: self.path.clone(),
                    },
                    origin: origin.map(|pk| values::ValueId {
                        pk,
                        index_generation: self.generation.clone(),
                        index_path: self.path.clone(),
                    }),
                    supplied,
                    source_time,
                })))
            }
            _ => Err(Error::Invariant("selected field directory shape".into())),
        }
    }
    /// At most32 fixed16-edge nodes are read, regardless of sibling/prefix count.
    /// A source group/local address is distinct from the public block ordinal.
    pub fn block(&self, slot: BlockSlot) -> Result<Option<BlockView<'_>>> {
        let (index, selection) = match &slot {
            BlockSlot::Item(i) => (u64::from(*i), d::Ordinal::Ordinary),
            BlockSlot::Reasoning(i) => (u64::from(*i), d::Ordinal::Reasoning),
            BlockSlot::Attachment(_) => {
                return Err(Error::Invariant(
                    "attachment selection is not implemented".into(),
                ));
            }
        };
        let result = self.select_block(index, selection)?;
        if result.as_ref().is_some_and(|b| b.slot != slot) {
            return Err(Error::Invariant(
                "selected public block ordinal disagrees".into(),
            ));
        }
        Ok(result)
    }
    /// Read one block in the canonical source order, preserving interleaved
    /// separate reasoning IDs. The selected root's summaries locate the block.
    pub fn block_at(&self, index: u64) -> Result<Option<BlockView<'_>>> {
        self.select_block(index, d::Ordinal::All)
    }
    fn select_block(&self, index: u64, selection: d::Ordinal) -> Result<Option<BlockView<'_>>> {
        self.visible()?;
        let read = locked(&self.read)?;
        let mut work = d::Work::default();
        let Some(selected_group) = d::nth(
            &read.connection,
            &self.path,
            self.owner,
            d::Kind::Groups,
            self.root,
            index,
            selection,
            &mut work,
        )?
        else {
            return Ok(None);
        };
        let group = selected_group.key;
        let local_index = selected_group.remainder;
        let d::Link::Node {
            pk: inner,
            kind: d::Kind::Blocks,
            height: 15,
        } = selected_group.edge.link
        else {
            return Err(Error::Invariant("selected Message group shape".into()));
        };
        let Some(selected_block) = d::nth(
            &read.connection,
            &self.path,
            self.owner,
            d::Kind::Blocks,
            Some(inner),
            local_index,
            selection,
            &mut work,
        )?
        else {
            return Err(Error::Invariant(
                "selected Message block count disagrees".into(),
            ));
        };
        let local = selected_block.key;
        let edge = selected_block.edge;
        let reasoning = edge.counts.reasoning == 1;
        if selected_block.remainder != 0
            || edge.counts.ordinary != i64::from(!reasoning)
            || edge.counts.reasoning != i64::from(reasoning)
        {
            return Err(Error::Invariant("block leaf count is not one".into()));
        }
        let skipped = selected_group.skipped.add(selected_block.skipped)?;
        let ordinal = u32::try_from(if reasoning {
            skipped.reasoning
        } else {
            skipped.ordinary
        })
        .map_err(|_| Error::Range)?;
        let slot = if reasoning {
            BlockSlot::Reasoning(ordinal)
        } else {
            BlockSlot::Item(ordinal)
        };
        let d::Link::Block { pk, .. } = edge.link else {
            return Err(Error::Invariant("selected block leaf shape".into()));
        };
        let (stored_group, stored_local, stored_slot, kind, encrypted, value, field_root): (
            u64,
            u64,
            String,
            String,
            Option<bool>,
            Option<i64>,
            Option<i64>,
        ) = db(&self.path, || {
            read.connection.query_row(
            "SELECT b.group_ordinal,b.local_ordinal,b.slot_kind,b.kind,b.encrypted,t.value_pk,b.field_root_pk FROM block b LEFT JOIN text t ON t.id=b.text_id WHERE b.pk=?1 AND b.message_pk=?2 AND b.session_pk=?3",
            params![pk,self.owner.message,self.owner.session], |r| Ok((crate::parser_state::unsigned(r,0)?,crate::parser_state::unsigned(r,1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?)))
        })?;
        if stored_group != group
            || stored_local != local
            || stored_slot != if reasoning { "reasoning" } else { "ordinary" }
        {
            return Err(Error::Invariant(
                "block metadata differs from selected directory".into(),
            ));
        }
        Ok(Some(BlockView {
            message: self,
            pk,
            slot,
            group,
            local,
            kind,
            encrypted,
            field_root,
            value: value.map(|pk| values::ValueId {
                pk,
                index_generation: self.generation.clone(),
                index_path: self.path.clone(),
            }),
        }))
    }
}
impl FieldView<'_> {
    /// Read the retained source-owned JSON descriptor, when supplied, from this
    /// same immutable Message snapshot. Readiness proves integrity, not the
    /// selector/transform's semantic validity or raw-span membership.
    pub fn origin_range(&self, offset: u64, allowance: usize) -> Result<Option<values::Range>> {
        self.message.visible()?;
        let read = locked(&self.message.read)?;
        self.origin
            .as_ref()
            .map(|id| values::range(&read.connection, &self.message.path, id, offset, allowance))
            .transpose()
    }

    pub fn codec(&self) -> Result<values::Codec> {
        self.message.visible()?;
        let read = locked(&self.message.read)?;
        Ok(values::head(&read.connection, &self.message.path, &self.value)?.codec)
    }
    pub fn range(&self, offset: u64, allowance: usize) -> Result<values::Range> {
        self.message.visible()?;
        let read = locked(&self.message.read)?;
        values::range(
            &read.connection,
            &self.message.path,
            &self.value,
            offset,
            allowance,
        )
    }
}
/// Recorded result metadata; pairing and call-side enrichment are separate.
pub struct ToolResultHeader {
    pub outcome: polylogue_model::ToolOutcome,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<u64>,
}
impl<'a> BlockView<'a> {
    /// Read the immutable result scalar columns, including statuses which did
    /// not originate in a supplied/ranking field. Unknown never becomes OK.
    pub fn tool_result(&self, allowance: usize) -> Result<Option<ToolResultHeader>> {
        self.message.visible()?;
        if self.kind != "tool_result" {
            return Ok(None);
        }
        let read = locked(&self.message.read)?;
        let (outcome, reason_length, reason, exit_code, duration): (
            Option<String>,
            u64,
            Option<String>,
            Option<i64>,
            Option<i64>,
        ) = db(&self.message.path, || {
            read.connection.query_row(
                "SELECT CASE WHEN result_outcome IN('ok','error','unknown') THEN result_outcome END,COALESCE(length(CAST(result_reason AS BLOB)),0),CASE WHEN COALESCE(length(CAST(result_reason AS BLOB)),0)<=?4 THEN result_reason END,result_exit_code,result_duration_ms FROM block WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND kind='tool_result'",
                params![self.pk,self.message.owner.message,self.message.owner.session,i64::try_from(allowance).unwrap_or(i64::MAX)],
                |r|Ok((r.get(0)?,crate::parser_state::unsigned(r,1)?,r.get(2)?,r.get(3)?,r.get(4)?)))
        })?;
        if reason_length > allowance as u64 {
            return Err(metadata_admission("tool result reason", reason_length));
        }
        let outcome = match (outcome.as_deref(), reason.as_deref()) {
            (Some("ok"), None) => polylogue_model::ToolOutcome::Ok,
            (Some("error"), reason) => polylogue_model::ToolOutcome::Error {
                reason: reason.map(polylogue_model::ErrorReason::parse),
            },
            (Some("unknown"), Some(reason)) => polylogue_model::ToolOutcome::Unknown {
                reason: polylogue_model::UnknownReason::parse(reason),
            },
            _ => return Err(Error::Invariant("invalid stored tool result status".into())),
        };
        Ok(Some(ToolResultHeader {
            outcome,
            exit_code,
            duration_ms: duration
                .map(|n| u64::try_from(n).map_err(|_| Error::Range))
                .transpose()?,
        }))
    }
    /// Resolve a field only inside this selected block's immutable root.
    /// The requested ordinal is checked against this view, never used to infer
    /// correspondence with a different source body's block.
    pub fn field(
        &self,
        slot: &polylogue_model::merge::Slot,
        allowance: usize,
    ) -> Result<Option<FieldRead<'a>>> {
        use polylogue_model::merge::Slot;
        self.message.visible()?;
        let block = match slot {
            Slot::Block { block, .. }
            | Slot::Extra {
                block: Some(block), ..
            } => block,
            _ => {
                return Err(Error::Invariant(
                    "block field requires block-local slot".into(),
                ));
            }
        };
        if block != &self.slot {
            return Err(Error::Invariant(
                "field slot names another selected block".into(),
            ));
        }
        let key = super::writer::block_field_key(slot, allowance)?;
        self.message.field_at(self.field_root, &key)
    }
    pub fn text_range(&self, offset: u64, allowance: usize) -> Result<Option<values::Range>> {
        self.message.visible()?;
        let read = locked(&self.message.read)?;
        self.value
            .as_ref()
            .map(|id| values::range(&read.connection, &self.message.path, id, offset, allowance))
            .transpose()
    }
}

/// Metadata from this exact selected index snapshot. Optional values stay
/// unknown when absent; stored millisecond aggregates imply no time quality.
#[derive(Clone, Debug)]
pub struct SessionHeader {
    pub id: SessionId,
    pub origin: Origin,
    pub native_id: String,
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
    pub attrs: Option<serde_json::Value>,
    pub started_at: Option<i64>,
    pub last_activity_at: Option<i64>,
    pub stub: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionCoverage {
    /// Only registered Message rows are traversable here. Provider pending
    /// cells can name further unprepared members; End never proves completeness.
    KnownCatalogOnly,
}
pub struct SessionPosition<'a> {
    pub generation: &'a str,
    pub counter: u64,
}
pub struct SessionView {
    read: SharedRead,
    path: PathBuf,
    header: SessionHeader,
    pk: i64,
    generation: String,
    counter: u64,
    archive_counter: u64,
}
/// The query and exact Session readers share this pinned admission/index view.
#[derive(Clone)]
pub struct SessionCatalog {
    read: SharedRead,
    path: PathBuf,
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
    fn visible(&self) -> Result<()> {
        let read = locked(&self.read)?;
        let archive_path = self.path.with_file_name("archive.db");
        if db(&archive_path, || {
            read.archive_visibility.as_ref().unwrap().query_row(
                "SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)",
                [],
                |r| r.get::<_, bool>(0),
            )
        })? {
            return Err(Error::ExcisionCheckRequired);
        }
        if !db(&self.path, || {
            read.visibility.query_row(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?1)",
                [&self.generation],
                |r| r.get::<_, bool>(0),
            )
        })? {
            return Err(Error::MessagePredecessorChanged);
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
    /// One complete identity in stable key order; the cursor cannot cross views.
    pub fn next(
        &self,
        after: Option<&SessionCatalogCursor>,
        allowance: usize,
    ) -> Result<Option<(SessionCatalogCursor, SessionView)>> {
        self.visible()?;
        let key = match after {
            None => "",
            Some(c) if Arc::ptr_eq(&c.read, &self.read) => &c.after,
            Some(_) => return Err(Error::MessagePredecessorChanged),
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
        let session = self
            .session(&id, allowance)?
            .ok_or(Error::MessagePredecessorChanged)?;
        Ok(Some((cursor, session)))
    }
}
/// Work cursor bound to one exact read snapshot. Its PK is a tie cursor only,
/// never an ordering authority for distinct Messages with the same sequence.
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
fn metadata_admission(resource: &'static str, required: u64) -> Error {
    Error::ParserAdmission { resource, required }
}
const SESSION_TEXT: [&str; 17] = [
    "origin",
    "native_id",
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
    /// Pin retained admission before the selected index. No matching-row or
    /// newest-row restriction is applied to this catalog.
    pub fn session_catalog(&self, allowance: usize) -> Result<SessionCatalog> {
        // Admission metadata is immutable, but its catalog can grow before
        // disposable registration. Pin it FIRST, then the index, and refuse an
        // archive change across that interval rather than hide a crash gap.
        let archive_path = self.root.join("archive.db");
        let archive = crate::open(&archive_path, true, true)?;
        db(&archive_path, || archive.execute_batch("BEGIN DEFERRED"))?;
        let archive_counter = crate::counter(&archive, &archive_path)?;
        let path = self.root.join("index.db");
        let connection = crate::open(&path, true, false)?;
        db(&path, || connection.execute_batch("BEGIN DEFERRED"))?;
        let (generation_length, generation): (u64, Option<String>) = db(&path, || {
            connection.query_row("SELECT length(CAST(value AS BLOB)),CASE WHEN length(CAST(value AS BLOB))<=?1 THEN value END FROM meta WHERE key='generation'",[i64::try_from(allowance).unwrap_or(i64::MAX)],|row|Ok((crate::parser_state::unsigned(row,0)?,row.get(1)?)))
        })?;
        let generation = generation
            .ok_or_else(|| metadata_admission("session read generation", generation_length))?;
        let counter = crate::counter(&connection, &path)?;
        if crate::counter(&self.archive, &archive_path)? != archive_counter {
            return Err(Error::MessagePredecessorChanged);
        }
        Ok(SessionCatalog {
            read: Arc::new(Mutex::new(PinnedRead {
                connection,
                visibility: crate::open(&path, true, false)?,
                archive: Some(archive),
                archive_visibility: Some(crate::open(&archive_path, true, true)?),
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
            [Option<String>; 17],
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
                            ],
                            r.get(19)?,
                            r.get(20)?,
                            r.get(21)?,
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
        let native_id = native_id
            .ok_or_else(|| Error::Invariant("selected Session native identity missing".into()))?;
        if native_id.is_empty() {
            return Err(Error::Invariant(
                "selected Session native identity empty".into(),
            ));
        }
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
    pub fn catalog(&self) -> SessionCatalog {
        SessionCatalog {
            read: Arc::clone(&self.read),
            path: self.path.clone(),
            generation: self.generation.clone(),
            counter: self.counter,
            archive_counter: self.archive_counter,
        }
    }
    fn visible(&self) -> Result<()> {
        let read = locked(&self.read)?;
        if let Some(archive) = &read.archive_visibility {
            let archive_path = self.path.with_file_name("archive.db");
            let excision: bool = db(&archive_path, || {
                archive.query_row("SELECT EXISTS(SELECT 1 FROM excision LIMIT 1)", [], |r| {
                    r.get(0)
                })
            })?;
            if excision {
                return Err(Error::ExcisionCheckRequired);
            }
        }

        let visible: bool = db(&self.path, || {
            read.visibility.query_row("SELECT EXISTS(SELECT 1 FROM session WHERE pk=?1 AND id=?2) AND EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?3)",params![self.pk,self.header.id.as_str(),self.generation],|r|r.get(0))
        })?;
        if visible {
            Ok(())
        } else {
            Err(Error::MessagePredecessorChanged)
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
            Some(_) => return Err(Error::MessagePredecessorChanged),
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
            return Err(Error::MessagePredecessorChanged);
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

/// Exact retained citation. It is not a byte-availability or per-range excision
/// verdict; those require the archive's excision authority, outside this index.
pub struct SourceSpan {
    pub input: String,
    pub chain: u64,
    pub start: u64,
    pub end: u64,
    pub preferred: bool,
}
#[derive(Clone, Copy)]
struct SpanKey {
    input: i64,
    chain: i64,
    start: i64,
    end: i64,
}
#[derive(Clone)]
enum SourcePosition {
    Preferred {
        after: Option<SpanKey>,
    },
    Candidate {
        after: i64,
    },
    Spans {
        candidate: i64,
        after: Option<SpanKey>,
    },
    CheckPreferred {
        candidate: i64,
        key: SpanKey,
    },
    CheckEarlier {
        candidate: i64,
        key: SpanKey,
        after: i64,
    },
}
#[derive(Clone)]
pub struct SourceCursor {
    read: SharedRead,
    body: i64,
    position: SourcePosition,
}
pub enum SourceStep {
    Span {
        cursor: SourceCursor,
        source: SourceSpan,
    },
    Continue {
        cursor: SourceCursor,
    },
    End,
}
impl MessageView {
    /// Preferred source first, then retained historical citations. Each turn
    /// visits one candidate/span or one dedup probe; no citation list is built.
    pub fn next_source(
        &self,
        after: Option<&SourceCursor>,
        allowance: usize,
    ) -> Result<SourceStep> {
        self.visible()?;
        let position = match after {
            None => SourcePosition::Preferred { after: None },
            Some(cursor) if Arc::ptr_eq(&self.read, &cursor.read) && cursor.body == self.body => {
                cursor.position.clone()
            }
            Some(_) => return Err(Error::MessagePredecessorChanged),
        };
        let read = locked(&self.read)?;
        let preferred: i64 = db(&self.path, || {
            read.connection.query_row("SELECT preferred_candidate_pk FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND ready=1",params![self.body,self.owner.message,self.owner.session],|r|r.get(0))
        })?;
        let cursor = |position| SourceCursor {
            read: Arc::clone(&self.read),
            body: self.body,
            position,
        };
        let progress = |position| {
            Ok(SourceStep::Continue {
                cursor: cursor(position),
            })
        };
        match position {
            SourcePosition::Preferred { after } => {
                match self.source_key(&read.connection, preferred, after)? {
                    Some(key) => self.source_span(
                        &read.connection,
                        key,
                        true,
                        cursor(SourcePosition::Preferred { after: Some(key) }),
                        allowance,
                    ),
                    None => progress(SourcePosition::Candidate { after: 0 }),
                }
            }
            SourcePosition::Candidate { after } => {
                let candidate: Option<i64> = db(&self.path, || {
                    read.connection.query_row("SELECT pk FROM message_candidate WHERE message_pk=?1 AND session_pk=?2 AND pk>?3 AND pk<>?4 ORDER BY pk LIMIT 1",params![self.owner.message,self.owner.session,after,preferred],|r|r.get(0)).optional()
                })?;
                match candidate {
                    Some(candidate) => progress(SourcePosition::Spans {
                        candidate,
                        after: None,
                    }),
                    None => Ok(SourceStep::End),
                }
            }
            SourcePosition::Spans { candidate, after } => {
                match self.source_key(&read.connection, candidate, after)? {
                    Some(key) => progress(SourcePosition::CheckPreferred { candidate, key }),
                    None => progress(SourcePosition::Candidate { after: candidate }),
                }
            }
            SourcePosition::CheckPreferred { candidate, key } => {
                if source_has(&read.connection, &self.path, preferred, key)? {
                    progress(SourcePosition::Spans {
                        candidate,
                        after: Some(key),
                    })
                } else {
                    progress(SourcePosition::CheckEarlier {
                        candidate,
                        key,
                        after: 0,
                    })
                }
            }
            SourcePosition::CheckEarlier {
                candidate,
                key,
                after,
            } => {
                let earlier: Option<i64> = db(&self.path, || {
                    read.connection.query_row("SELECT pk FROM message_candidate WHERE message_pk=?1 AND session_pk=?2 AND pk>?3 AND pk<?4 AND pk<>?5 ORDER BY pk LIMIT 1",params![self.owner.message,self.owner.session,after,candidate,preferred],|r|r.get(0)).optional()
                })?;
                match earlier {
                    Some(earlier) if source_has(&read.connection, &self.path, earlier, key)? => {
                        progress(SourcePosition::Spans {
                            candidate,
                            after: Some(key),
                        })
                    }
                    Some(earlier) => progress(SourcePosition::CheckEarlier {
                        candidate,
                        key,
                        after: earlier,
                    }),
                    None => self.source_span(
                        &read.connection,
                        key,
                        false,
                        cursor(SourcePosition::Spans {
                            candidate,
                            after: Some(key),
                        }),
                        allowance,
                    ),
                }
            }
        }
    }
    fn source_key(
        &self,
        connection: &Connection,
        candidate: i64,
        after: Option<SpanKey>,
    ) -> Result<Option<SpanKey>> {
        let row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<(i64, i64, SpanKey)> {
            Ok((
                r.get(0)?,
                r.get(1)?,
                SpanKey {
                    input: r.get(2)?,
                    chain: r.get(3)?,
                    start: r.get(4)?,
                    end: r.get(5)?,
                },
            ))
        };
        let found = match after {
            None => db(&self.path, || {
                connection.query_row("SELECT message_pk,session_pk,input_pk,chain,source_start,source_end FROM message_source WHERE candidate_pk=?1 ORDER BY input_pk,chain,source_start,source_end LIMIT 1",[candidate],row).optional()
            })?,
            Some(key) => db(&self.path, || {
                connection.query_row("SELECT message_pk,session_pk,input_pk,chain,source_start,source_end FROM message_source WHERE candidate_pk=?1 AND (input_pk,chain,source_start,source_end)>(?2,?3,?4,?5) ORDER BY input_pk,chain,source_start,source_end LIMIT 1",params![candidate,key.input,key.chain,key.start,key.end],row).optional()
            })?,
        };
        match found {
            Some((message, session, key))
                if message == self.owner.message
                    && session == self.owner.session
                    && key.input > 0
                    && key.chain > 0
                    && key.start >= 0
                    && key.end >= key.start =>
            {
                Ok(Some(key))
            }
            Some(_) => Err(Error::Invariant(
                "retained source citation owner or range invalid".into(),
            )),
            None => Ok(None),
        }
    }
    fn source_span(
        &self,
        connection: &Connection,
        key: SpanKey,
        preferred: bool,
        cursor: SourceCursor,
        allowance: usize,
    ) -> Result<SourceStep> {
        let (length, input): (u64, Option<String>) = db(&self.path, || {
            connection.query_row("SELECT length(CAST(id AS BLOB)),CASE WHEN length(CAST(id AS BLOB))<=?2 THEN id END FROM input_ref WHERE pk=?1",params![key.input,i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?)))
        })?;
        let input =
            input.ok_or_else(|| metadata_admission("Message source natural identity", length))?;
        Ok(SourceStep::Span {
            cursor,
            source: SourceSpan {
                input,
                chain: u64::try_from(key.chain).map_err(|_| Error::Range)?,
                start: u64::try_from(key.start).map_err(|_| Error::Range)?,
                end: u64::try_from(key.end).map_err(|_| Error::Range)?,
                preferred,
            },
        })
    }
}
fn source_has(
    connection: &Connection,
    path: &std::path::Path,
    candidate: i64,
    key: SpanKey,
) -> Result<bool> {
    db(path, || {
        connection.query_row("SELECT EXISTS(SELECT 1 FROM message_source WHERE candidate_pk=?1 AND input_pk=?2 AND chain=?3 AND source_start=?4 AND source_end=?5)",params![candidate,key.input,key.chain,key.start,key.end],|r|r.get(0))
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldScope {
    Message,
    Block { group: u64, local: u64 },
}
#[derive(Clone)]
pub struct FieldCursor {
    read: SharedRead,
    body: i64,
    root: Option<i64>,
    scope: FieldScope,
    after: u64,
}
pub struct MessageFieldEntry<'a> {
    pub cursor: FieldCursor,
    pub slot: polylogue_model::merge::Slot,
    pub value: FieldRead<'a>,
}
pub enum BlockFieldAddress {
    Field(polylogue_model::merge::BlockField),
    Extra(String),
}
pub struct BlockFieldEntry<'a> {
    pub cursor: FieldCursor,
    pub address: BlockFieldAddress,
    pub value: FieldRead<'a>,
}
impl MessageView {
    /// Traverse only installed leaves in this selected Message root. The store
    /// slot key is a work address, not a cross-candidate correspondence proof.
    pub fn next_field(
        &self,
        after: Option<&FieldCursor>,
        allowance: usize,
    ) -> Result<Option<MessageFieldEntry<'_>>> {
        self.visible()?;
        let read = locked(&self.read)?;
        let root: Option<i64> = db(&self.path, || {
            read.connection.query_row("SELECT field_root_pk FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND ready=1",params![self.body,self.owner.message,self.owner.session],|r|r.get(0))
        })?;
        let found = self.field_successor(
            &read.connection,
            root,
            FieldScope::Message,
            after,
            allowance,
        )?;
        drop(read);
        let Some((cursor, key)) = found else {
            return Ok(None);
        };
        let slot: polylogue_model::merge::Slot = serde_json::from_slice(&key)
            .map_err(|e| Error::Invariant(format!("selected Message semantic key: {e}")))?;
        if matches!(
            slot,
            polylogue_model::merge::Slot::Block { .. }
                | polylogue_model::merge::Slot::Extra { block: Some(_), .. }
        ) {
            return Err(Error::Invariant(
                "block-local address in Message field root".into(),
            ));
        }
        let value = self
            .field_at(root, &key)?
            .ok_or_else(|| Error::Invariant("selected Message field membership changed".into()))?;
        Ok(Some(MessageFieldEntry {
            cursor,
            slot,
            value,
        }))
    }
    fn field_successor(
        &self,
        connection: &Connection,
        root: Option<i64>,
        scope: FieldScope,
        after: Option<&FieldCursor>,
        allowance: usize,
    ) -> Result<Option<(FieldCursor, Vec<u8>)>> {
        let after = match after {
            None => None,
            Some(cursor)
                if Arc::ptr_eq(&self.read, &cursor.read)
                    && cursor.body == self.body
                    && cursor.root == root
                    && cursor.scope == scope =>
            {
                Some(cursor.after)
            }
            Some(_) => return Err(Error::MessagePredecessorChanged),
        };
        let Some((key, edge)) = d::next(
            connection,
            &self.path,
            self.owner,
            d::Kind::Fields,
            root,
            after,
            &mut d::Work::default(),
        )?
        else {
            return Ok(None);
        };
        match edge.link {
            d::Link::Field(_) => {}
            d::Link::Conflict { slot, .. } if u64::try_from(slot).ok() == Some(key) => {}
            _ => {
                return Err(Error::Invariant(
                    "selected field successor has wrong leaf kind".into(),
                ));
            }
        }
        let slot = i64::try_from(key).map_err(|_| Error::Range)?;
        let (length, semantic): (u64, Option<Vec<u8>>) = db(&self.path, || {
            connection.query_row("SELECT length(semantic_key),CASE WHEN length(semantic_key)<=?3 THEN semantic_key END FROM message_field_slot WHERE pk=?1 AND message_pk=?2",params![slot,self.owner.message,i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?)))
        })?;
        let semantic =
            semantic.ok_or_else(|| metadata_admission("selected field semantic key", length))?;
        Ok(Some((
            FieldCursor {
                read: Arc::clone(&self.read),
                body: self.body,
                root,
                scope,
                after: key,
            },
            semantic,
        )))
    }
}
impl<'a> BlockView<'a> {
    pub fn next_field(
        &self,
        after: Option<&FieldCursor>,
        allowance: usize,
    ) -> Result<Option<BlockFieldEntry<'a>>> {
        self.message.visible()?;
        let read = locked(&self.message.read)?;
        let found = self.message.field_successor(
            &read.connection,
            self.field_root,
            FieldScope::Block {
                group: self.group,
                local: self.local,
            },
            after,
            allowance,
        )?;
        drop(read);
        let Some((cursor, key)) = found else {
            return Ok(None);
        };
        let address = match serde_json::from_slice::<super::writer::BlockFieldKey<'_>>(&key)
            .map_err(|e| Error::Invariant(format!("selected Block semantic key: {e}")))?
        {
            super::writer::BlockFieldKey::Field(field) => BlockFieldAddress::Field(field),
            super::writer::BlockFieldKey::Extra(key) => BlockFieldAddress::Extra(key.into_owned()),
        };
        let value = self
            .message
            .field_at(self.field_root, &key)?
            .ok_or_else(|| Error::Invariant("selected Block field membership changed".into()))?;
        Ok(Some(BlockFieldEntry {
            cursor,
            address,
            value,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ImportObservation, InputFormat, Writer,
        contributions::{
            BlockKind, BlockPresence, Identity, ProjectionStep, Source, SourceProjection,
            SourceRange,
        },
        parser_state::{self, Accounting, Basis, Budget, Head, Phase, Progress},
    };
    use polylogue_model::{
        BlobHash,
        merge::{BlockField, FinishState, MessageField, RouteKind, Slot},
    };
    fn budget() -> Budget {
        Budget {
            cells: 1000,
            encoded_bytes: 1 << 20,
        }
    }
    // Synthetic typed source certificates exercise the store reader; this is
    // not provider interpretation, election or complete Session qualification.
    struct Fixture {
        _dir: tempfile::TempDir,
        writer: Writer,
        head: Head,
        session: SessionId,
        origin: Origin,
    }
    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let mut writer = Writer::open(dir.path()).unwrap();
            let hash = writer
                .put_file_blob(&b"synthetic retained source bytes\n"[..])
                .unwrap();
            let input = writer
                .retain_immutable_input(
                    hash,
                    InputFormat::CodexRollout,
                    &[],
                    &ImportObservation {
                        id: "synthetic-reader",
                        machine: "synthetic",
                        host: "synthetic",
                        path: "synthetic/source.jsonl",
                        seen_at: 0,
                    },
                    4096,
                )
                .unwrap();
            let head = writer
                .begin_parser(&Basis {
                    input_pk: input.input_pk,
                    chain: input.chain,
                    parser_version: 1,
                    digest: [1; 32],
                    acquired_upto: input.length,
                    companions_json: None,
                    excisions_seen: 0,
                })
                .unwrap();
            let origin = Origin::new("synthetic").unwrap();
            let session = SessionId::new(&origin, "session:opaque").unwrap();
            let mut f = Self {
                _dir: dir,
                writer,
                head,
                session,
                origin,
            };
            f.advance();
            f
        }
        fn advance(&mut self) {
            let path = self.writer.root.join("index.db");
            self.head = parser_state::write_step(
                &mut self.writer._index,
                &path,
                &self.head,
                Progress {
                    inspected_upto: self.head.acquired_upto,
                    parsed_upto: self.head.acquired_upto,
                    phase: Phase::Finalizing,
                    finalization_pending: true,
                    counts: Accounting::default(),
                },
                &[],
                budget(),
                |_| Ok(()),
            )
            .unwrap();
        }
        fn prepare(&mut self, name: &str, seq: u64, key: u8) -> super::super::Preparation {
            let id = MessageId::new(self.session.clone(), name).unwrap();
            self.writer
                .prepare_message(
                    Identity {
                        session: &self.session,
                        origin: &self.origin,
                        native_session: "session:opaque",
                        session_kind: SessionKind::Main,
                        message: &id,
                        sequence: seq,
                        role: Role::Assistant,
                        written_by: WrittenBy::Model,
                    },
                    Source {
                        predecessor: &self.head.token,
                        revision: 1,
                        evidence_revision: self.head.token.step as u64,
                        logical_key: [key; 32],
                        route: RouteKind::Native,
                    },
                    65536,
                )
                .unwrap()
        }
        fn ready(
            &mut self,
            name: &str,
            seq: u64,
            key: u8,
            model: &str,
            old_extra: bool,
            spans: &[(u64, u64)],
        ) -> SourceProjection {
            self.ready_with_block(
                name,
                seq,
                key,
                model,
                old_extra,
                spans,
                BlockKind::Thinking { encrypted: None },
            )
        }
        fn ready_with_block(
            &mut self,
            name: &str,
            seq: u64,
            key: u8,
            model: &str,
            old_extra: bool,
            spans: &[(u64, u64)],
            kind: BlockKind<'_>,
        ) -> SourceProjection {
            let thinking = matches!(kind, BlockKind::Thinking { .. });
            let previous = self
                .writer
                .current_source_projection(&self.head.token, &[key; 32], 65536)
                .unwrap();
            let build = self.prepare(name, seq, key).build;
            let mut sequence = 0;
            for &(start, end) in spans {
                self.writer
                    .add_message_source(
                        &build,
                        SourceRange { start, end },
                        &ProjectionStep {
                            expected_sequence: sequence,
                            next_cursor: b"source",
                        },
                        65536,
                    )
                    .unwrap();
                sequence += 1;
            }
            self.json(&build, &serde_json::to_vec(model).unwrap());
            self.writer
                .write_message_field(
                    &build,
                    &Slot::Message(MessageField::Model),
                    None,
                    true,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"model",
                    },
                    65536,
                )
                .unwrap();
            sequence += 1;
            if old_extra {
                self.json(&build, b"false");
                self.writer
                    .write_message_field(
                        &build,
                        &Slot::Extra {
                            block: None,
                            key: "old:extra".into(),
                        },
                        None,
                        true,
                        &ProjectionStep {
                            expected_sequence: sequence,
                            next_cursor: b"extra",
                        },
                        65536,
                    )
                    .unwrap();
                sequence += 1;
            }
            self.writer
                .begin_block(
                    &build,
                    0,
                    0,
                    kind,
                    BlockPresence {
                        source_block: true,
                        source_text: false,
                    },
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"block",
                    },
                    65536,
                )
                .unwrap();
            sequence += 1;
            if thinking {
                self.json(&build, b"\"retained-signature\"");
                self.writer
                    .write_block_field(
                        &build,
                        &Slot::Block {
                            block: BlockSlot::Item(0),
                            field: BlockField::ThinkingSignature,
                        },
                        None,
                        false,
                        &ProjectionStep {
                            expected_sequence: sequence,
                            next_cursor: b"signature",
                        },
                        65536,
                    )
                    .unwrap();
                sequence += 1;
            }
            self.writer
                .install_block(
                    &build,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"installed",
                    },
                    65536,
                )
                .unwrap();
            sequence += 1;
            let prepared = self
                .writer
                .finish_message_projection(
                    &build,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"finished",
                    },
                    BlobHash::from_bytes([key.wrapping_add(self.head.token.step as u8); 32]),
                    FinishState::Finished,
                    65536,
                )
                .unwrap();
            let source = self
                .writer
                .publish_source_projection(
                    &prepared,
                    previous.as_ref(),
                    b"synthetic-reader-proof",
                    65536,
                )
                .unwrap();
            self.writer
                .select_uncontested_source_message(&source, 65536)
                .unwrap();
            source
        }
        fn json(&mut self, build: &super::super::Build, bytes: &[u8]) {
            let value = self
                .writer
                .begin_message_value(build, values::Codec::Json, bytes.len() as u64)
                .unwrap();
            self.writer
                .append_value_chunk(&value.id, 0, bytes, bytes.len())
                .unwrap();
            let reader = self.writer.read_handle().unwrap();
            let mut proof = reader.validate_value(&value.id).unwrap();
            while !proof.progress().complete {
                proof.advance(&reader, 65536).unwrap();
            }
            drop(reader);
            self.writer.seal_message_value(build, &proof).unwrap();
        }
        fn session(&self) -> SessionView {
            self.writer
                .read_handle()
                .unwrap()
                .selected_session(&self.session, 65536)
                .unwrap()
                .unwrap()
        }
    }
    fn model(view: &MessageView) -> String {
        let FieldRead::Value(field) = view
            .field(&Slot::Message(MessageField::Model), 65536)
            .unwrap()
            .unwrap()
        else {
            panic!("conflict")
        };
        serde_json::from_slice(&field.range(0, 65536).unwrap().bytes).unwrap()
    }
    #[test]
    fn dependency_catalog_pins_admission_identity_and_survives_index_registration_gap() {
        let mut f = Fixture::new();
        f.ready("catalog", 0, 1, "model", false, &[(0, 1)]);
        let old = f.session();
        let old_archive_counter = old.source_archive_counter();
        let hash = f.writer.put_file_blob(b"{malformed}\n").unwrap();
        let input = f
            .writer
            .retain_immutable_input(
                hash,
                InputFormat::ClaudeCodeJsonl,
                &["claude-main", "context-only"],
                &ImportObservation {
                    id: "catalog-gap",
                    machine: "synthetic",
                    host: "synthetic",
                    path: "synthetic/context.jsonl",
                    seen_at: 0,
                },
                65536,
            )
            .unwrap();
        assert!(old.next_dependency(None, 1).is_err());
        let (old_cursor, old_input) = old.next_dependency(None, 65536).unwrap().unwrap();
        assert_ne!(old_input.id(), input.id);
        assert!(
            old.next_dependency(Some(&old_cursor), 65536)
                .unwrap()
                .is_none()
        );
        // The native crash-repair gap is visible from archive admission even
        // before a disposable input_ref or any Message exists for this input.
        f.writer
            ._index
            .execute("DELETE FROM input_ref WHERE pk=?1", [input.input_pk])
            .unwrap();
        let fresh = f.session();
        assert!(fresh.source_archive_counter() > old_archive_counter);
        assert!(matches!(
            fresh.next_dependency(Some(&old_cursor), 65536),
            Err(Error::MessagePredecessorChanged)
        ));
        let mut after = None;
        let mut found = false;
        while let Some((cursor, dependency)) = fresh.next_dependency(after.as_ref(), 65536).unwrap()
        {
            if dependency.id() == input.id {
                assert_eq!(dependency.context(), ["claude-main", "context-only"]);
                assert!(dependency.registered().is_none());
                assert!(dependency.head().is_none());
                found = true;
            }
            after = Some(cursor);
        }
        assert!(found);
        // Mutating admitted identity does not redirect a pinned old catalog;
        // a fresh reader refuses the hash/context inconsistency.
        f.writer
            .archive
            .execute(
                "UPDATE input SET context_json='[\"claude-main\",\"changed\"]' WHERE id=?1",
                [&input.id],
            )
            .unwrap();
        let newer = f.session();
        let mut after = None;
        let mut refused = false;
        loop {
            match newer.next_dependency(after.as_ref(), 65536) {
                Ok(Some((cursor, _))) => after = Some(cursor),
                Ok(None) => break,
                Err(Error::Invariant(_)) => {
                    refused = true;
                    break;
                }
                Err(e) => panic!("unexpected: {e}"),
            }
        }
        assert!(refused);
        // Previously pinned metadata still has its original validated identity.
        let mut after = None;
        let mut original = false;
        while let Some((cursor, dependency)) = fresh.next_dependency(after.as_ref(), 65536).unwrap()
        {
            if dependency.id() == input.id {
                assert_eq!(dependency.context(), ["claude-main", "context-only"]);
                original = true;
            }
            after = Some(cursor);
        }
        assert!(original);
    }
    #[test]
    fn session_snapshot_pins_all_heads_and_reports_pending_and_catalog_incomplete() {
        fn assert_send<T: Send>() {}
        assert_send::<MessageView>();
        assert_send::<SessionView>();
        assert_send::<SessionCursor>();
        assert_send::<FieldCursor>();
        assert_send::<SourceCursor>();
        let mut f = Fixture::new();
        let a = f.ready("a:opaque", 0, 1, "old-a", false, &[(0, 1)]);
        let b = f.ready("b:opaque", 1, 2, "old-b", false, &[(2, 3)]);
        let _pending = f.prepare("pending:opaque", 2, 3);
        let view = f.session();
        let counter = view.position().counter;
        assert_eq!(view.header().unwrap().id, f.session);
        assert_eq!(view.header().unwrap().native_id, "session:opaque");
        let SessionStep::Ready {
            cursor,
            sequence,
            message: first,
        } = view.next(None, 65536).unwrap()
        else {
            panic!("first")
        };
        assert_eq!(sequence, 0);
        assert_eq!(first.id(), a.message());
        assert_eq!(first.header().unwrap().in_progress, Some(false));
        assert_eq!(model(&first), "old-a");
        f.advance();
        f.ready("b:opaque", 1, 2, "new-b", false, &[(2, 3)]);
        let SessionStep::Ready {
            cursor: second_cursor,
            message: second,
            ..
        } = view.next(Some(&cursor), 65536).unwrap()
        else {
            panic!("second")
        };
        assert_eq!(second.id(), b.message());
        assert_eq!(model(&second), "old-b");
        assert_eq!(
            model(
                &f.writer
                    .read_handle()
                    .unwrap()
                    .selected_message(b.message())
                    .unwrap()
                    .unwrap()
            ),
            "new-b"
        );
        assert_eq!(view.position().counter, counter);
        assert!(f.session().position().counter > counter);
        let SessionStep::Pending {
            cursor,
            id,
            sequence,
        } = view.next(Some(&second_cursor), 65536).unwrap()
        else {
            panic!("pending member")
        };
        assert_eq!(sequence, 2);
        assert_eq!(
            id,
            MessageId::new(f.session.clone(), "pending:opaque").unwrap()
        );
        assert!(matches!(
            view.next(Some(&cursor), 65536).unwrap(),
            SessionStep::End {
                coverage: SessionCoverage::KnownCatalogOnly
            }
        ));
        let other = f.session();
        assert!(matches!(
            other.next(Some(&cursor), 65536),
            Err(Error::MessagePredecessorChanged)
        ));
    }
    #[test]
    fn session_duplicate_sequence_does_not_use_pk_as_canonical_order() {
        let mut f = Fixture::new();
        f.ready("a", 4, 1, "a", false, &[]);
        f.prepare("b", 4, 2);
        assert!(matches!(
            f.session().next(None, 65536).unwrap(),
            SessionStep::UnresolvedOrder { sequence: 4 }
        ));
    }
    #[test]
    fn session_metadata_admission_absence_and_current_privacy_are_explicit() {
        let mut f = Fixture::new();
        let source = f.ready("a", 0, 1, "old", false, &[]);
        let reader = f.writer.read_handle().unwrap();
        let missing = SessionId::from_stored("opaque:absent:session").unwrap();
        assert!(reader.selected_session(&missing, 65536).unwrap().is_none());
        assert!(matches!(
            reader.selected_session(&f.session, 1),
            Err(Error::ParserAdmission { .. })
        ));
        drop(reader);
        let view = f.session();
        let SessionStep::Ready { message, .. } = view.next(None, 65536).unwrap() else {
            panic!("ready")
        };
        f.writer
            ._index
            .execute(
                "UPDATE message SET hidden=1,removal_epoch=removal_epoch+1 WHERE id=?1",
                [source.message().as_str()],
            )
            .unwrap();
        assert!(matches!(
            message.header(),
            Err(Error::MessagePredecessorChanged)
        ));
        assert!(matches!(
            view.next(None, 65536),
            Err(Error::MessagePredecessorChanged)
        ));
        f.writer
            ._index
            .execute("UPDATE session SET title=?1", ["large".repeat(1000)])
            .unwrap();
        assert!(matches!(
            f.writer
                .read_handle()
                .unwrap()
                .selected_session(&f.session, 256),
            Err(Error::ParserAdmission { .. })
        ));
    }
    #[test]
    fn field_successors_follow_selected_roots_and_include_unranked_leaves() {
        let mut f = Fixture::new();
        let first = f.ready("a", 0, 1, "old", true, &[]);
        let old = f
            .writer
            .read_handle()
            .unwrap()
            .selected_message(first.message())
            .unwrap()
            .unwrap();
        let old_entry = old.next_field(None, 65536).unwrap().unwrap();
        let mut count = 1;
        let mut cursor = old_entry.cursor;
        while let Some(entry) = old.next_field(Some(&cursor), 65536).unwrap() {
            count += 1;
            cursor = entry.cursor
        }
        assert_eq!(count, 2);
        let block = old.block_at(0).unwrap().unwrap();
        let auxiliary = block.next_field(None, 65536).unwrap().unwrap();
        assert!(matches!(
            auxiliary.address,
            BlockFieldAddress::Field(BlockField::ThinkingSignature)
        ));
        let FieldRead::Value(value) = auxiliary.value else {
            panic!("auxiliary")
        };
        assert!(!value.supplied);
        assert_eq!(
            value.range(0, 65536).unwrap().bytes,
            b"\"retained-signature\""
        );
        assert!(
            block
                .next_field(Some(&auxiliary.cursor), 65536)
                .unwrap()
                .is_none()
        );
        f.advance();
        let second = f.ready("a", 0, 1, "new", false, &[]);
        let new = f
            .writer
            .read_handle()
            .unwrap()
            .selected_message(second.message())
            .unwrap()
            .unwrap();
        let entry = new.next_field(None, 65536).unwrap().unwrap();
        assert!(matches!(entry.slot, Slot::Message(MessageField::Model)));
        assert!(
            new.next_field(Some(&entry.cursor), 65536)
                .unwrap()
                .is_none()
        );
        assert!(matches!(
            new.next_field(Some(&cursor), 65536),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(model(&old), "old");
        assert_eq!(model(&new), "new");
        assert!(matches!(
            new.next_field(None, 1),
            Err(Error::ParserAdmission { .. })
        ));
    }
    #[test]
    fn source_spans_are_deduped_preferred_first_and_pin_historical_view() {
        let mut f = Fixture::new();
        let first = f.ready("a", 0, 1, "old", false, &[(0, 1), (4, 5)]);
        let old = f
            .writer
            .read_handle()
            .unwrap()
            .selected_message(first.message())
            .unwrap()
            .unwrap();
        f.advance();
        let second = f.ready("a", 0, 1, "new", false, &[(0, 1), (2, 3)]);
        let new = f
            .writer
            .read_handle()
            .unwrap()
            .selected_message(second.message())
            .unwrap()
            .unwrap();
        assert!(matches!(
            new.next_source(None, 1),
            Err(Error::ParserAdmission { .. })
        ));
        fn collect(view: &MessageView) -> Vec<(u64, u64, bool, String)> {
            let mut cursor = None;
            let mut spans = Vec::new();
            for _ in 0..100 {
                match view.next_source(cursor.as_ref(), 65536).unwrap() {
                    SourceStep::Continue { cursor: next } => cursor = Some(next),
                    SourceStep::Span {
                        cursor: next,
                        source,
                    } => {
                        spans.push((source.start, source.end, source.preferred, source.input));
                        cursor = Some(next)
                    }
                    SourceStep::End => return spans,
                }
            }
            panic!("source traversal did not end")
        }
        let spans = collect(&new);
        assert_eq!(
            spans.iter().map(|s| (s.0, s.1, s.2)).collect::<Vec<_>>(),
            vec![(0, 1, true), (2, 3, true), (4, 5, false)]
        );
        assert!(spans.iter().all(|s| s.3 == spans[0].3));
        let old_spans = collect(&old);
        assert_eq!(
            old_spans
                .iter()
                .map(|s| (s.0, s.1, s.2))
                .collect::<Vec<_>>(),
            vec![(0, 1, true), (4, 5, true)]
        );
        let SourceStep::Span { cursor, .. } = old.next_source(None, 65536).unwrap() else {
            panic!("preferred")
        };
        assert!(matches!(
            new.next_source(Some(&cursor), 65536),
            Err(Error::MessagePredecessorChanged)
        ));
    }
    #[test]
    fn result_headers_preserve_unknown_error_and_zero_without_supplied_fields() {
        use polylogue_model::{ErrorReason, ToolOutcome, UnknownReason};
        let mut f = Fixture::new();
        for (sequence, outcome) in [
            ToolOutcome::Unknown {
                reason: UnknownReason::OutputNotRecorded,
            },
            ToolOutcome::Error {
                reason: Some(ErrorReason::Denied("permission".into())),
            },
        ]
        .iter()
        .enumerate()
        {
            let source = f.ready_with_block(
                &format!("result-{sequence}"),
                sequence as u64,
                sequence as u8 + 1,
                "model",
                false,
                &[],
                BlockKind::ToolResult {
                    outcome,
                    exit_code: Some(0),
                    duration_ms: Some(0),
                },
            );
            let view = f
                .writer
                .read_handle()
                .unwrap()
                .selected_message(source.message())
                .unwrap()
                .unwrap();
            let block = view.block_at(0).unwrap().unwrap();
            assert!(block.next_field(None, 65536).unwrap().is_none());
            let header = block.tool_result(65536).unwrap().unwrap();
            assert_eq!(&header.outcome, outcome);
            assert_eq!(header.exit_code, Some(0));
            assert_eq!(header.duration_ms, Some(0));
            assert!(matches!(
                block.tool_result(1),
                Err(Error::ParserAdmission { .. })
            ));
        }
        let source = f.ready("thinking", 2, 3, "model", false, &[]);
        let view = f
            .writer
            .read_handle()
            .unwrap()
            .selected_message(source.message())
            .unwrap()
            .unwrap();
        assert!(
            view.block_at(0)
                .unwrap()
                .unwrap()
                .tool_result(0)
                .unwrap()
                .is_none()
        );
    }
}

// The existing selected body owns branch/copy membership and provider time.
pub(crate) fn search_membership(message: &MessageView) -> Result<bool> {
    message.visible()?;
    let read = locked(&message.read)?;
    db(&message.path, || {
        read.connection.query_row(
            "SELECT on_active_path IS NOT 0 AND copy_origin IS NULL FROM message_body WHERE pk=?1",
            [message.body],
            |r| r.get(0),
        )
    })
}
pub(crate) fn search_time(message: &MessageView) -> Result<Option<i64>> {
    use polylogue_model::merge::{MessageField, Slot};
    match message.field(&Slot::Message(MessageField::At), 65536)? {
        Some(FieldRead::Value(value)) => {
            let range = value.range(0, 65536)?;
            if !range.eof {
                return Err(metadata_admission(
                    "search provider time",
                    range.next_offset,
                ));
            }
            let time: serde_json::Value = serde_json::from_slice(&range.bytes)
                .map_err(|e| Error::Invariant(e.to_string()))?;
            Ok(time.get("at").and_then(serde_json::Value::as_i64))
        }
        _ => Ok(None),
    }
}
