//! One SQLite snapshot selects the whole immutable Message incarnation. Sparse
//! order-statistic reads and payload ranges never switch to a newer head midway.
use super::directory as d;
use crate::{Error, ReadHandle, Result, db, values};
use polylogue_model::{BlockSlot, MessageId, Role, SessionId, WrittenBy};
use rusqlite::{Connection, OptionalExtension, params};
use std::{fs::File, path::PathBuf, sync::Arc};

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
}
pub struct MessageView {
    connection: Connection,
    visibility: Connection,
    path: PathBuf,
    id: MessageId,
    owner: d::Owner,
    body: i64,
    removal_epoch: i64,
    root: Option<i64>,
    generation: String,
    _lock: Arc<File>,
}
/// A payload reference is private and usable only through its originating
/// Message snapshot. It cannot be substituted into a newer Message body.
pub struct BlockView<'a> {
    message: &'a MessageView,
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
            connection,
            visibility: crate::open(&path, true, false)?,
            path,
            id: id.clone(),
            owner: d::Owner { message, session },
            body,
            removal_epoch,
            root,
            generation,
            _lock: Arc::clone(&self._lock),
        }))
    }
}
impl MessageView {
    // Content replacement preserves this pinned old incarnation. Privacy/removal
    // invalidates access immediately through the current scalar epoch instead.
    fn visible(&self) -> Result<()> {
        let visible: bool = db(&self.path, || {
            self.visibility.query_row(
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
        let (role, by): (String, String) = db(&self.path, || {
            self.connection.query_row(
                "SELECT role,written_by FROM message_body WHERE pk=?1 AND ready=1",
                [self.body],
                |r| Ok((r.get(0)?, r.get(1)?)),
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
        Ok(MessageHeader { role, written_by })
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
        let root: Option<i64> = db(&self.path, || {
            self.connection.query_row(
                "SELECT field_root_pk FROM message_body WHERE pk=?1 AND ready=1",
                [self.body],
                |r| r.get(0),
            )
        })?;
        self.field_at(root, &key)
    }
    fn field_at(&self, root: Option<i64>, key: &[u8]) -> Result<Option<FieldRead<'_>>> {
        let slot: Option<i64> = db(&self.path, || {
            self.connection
                .query_row(
                    "SELECT pk FROM message_field_slot WHERE message_pk=?1 AND semantic_key=?2",
                    params![self.owner.message, key],
                    |r| r.get(0),
                )
                .optional()
        })?;
        let Some(slot) = slot else { return Ok(None) };
        let edge = d::lookup(
            &self.connection,
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
                        self.connection.query_row("SELECT value_pk,origin_value_pk,supplied,source_time FROM message_field_fact WHERE pk=?1 AND slot_pk=?2 AND message_pk=?3 AND session_pk=?4",params![fact,slot,self.owner.message,self.owner.session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
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
        let mut work = d::Work::default();
        let Some(selected_group) = d::nth(
            &self.connection,
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
            &self.connection,
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
            self.connection.query_row(
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
        self.origin
            .as_ref()
            .map(|id| {
                values::range(
                    &self.message.connection,
                    &self.message.path,
                    id,
                    offset,
                    allowance,
                )
            })
            .transpose()
    }

    pub fn codec(&self) -> Result<values::Codec> {
        self.message.visible()?;
        Ok(values::head(&self.message.connection, &self.message.path, &self.value)?.codec)
    }
    pub fn range(&self, offset: u64, allowance: usize) -> Result<values::Range> {
        self.message.visible()?;
        values::range(
            &self.message.connection,
            &self.message.path,
            &self.value,
            offset,
            allowance,
        )
    }
}
impl<'a> BlockView<'a> {
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
        self.value
            .as_ref()
            .map(|id| {
                values::range(
                    &self.message.connection,
                    &self.message.path,
                    id,
                    offset,
                    allowance,
                )
            })
            .transpose()
    }
}
