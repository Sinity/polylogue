//! The selected body of one message: header, fields, blocks and a content
//! digest.
use super::{
    PinnedRead, SharedRead, fields::BlockFieldAddress, locked, metadata_admission, ordered_extras,
};
use crate::contributions::directory as d;
use crate::{Error, ReadHandle, Result, db, values};
use polylogue_model::{BlockSlot, MessageId, Role, SessionId, WrittenBy};
use rusqlite::{OptionalExtension, params};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

/// `Pending`: the message exists but has no ready selected body yet.
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
pub struct MessageView {
    pub(super) read: SharedRead,
    pub(super) path: PathBuf,
    pub(super) id: MessageId,
    pub(super) owner: d::Owner,
    pub(super) body: i64,
    pub(super) removal_epoch: i64,
    pub(super) root: Option<i64>,
    pub(super) generation: String,
}
/// One block of a message view, read through that view's snapshot.
pub struct BlockView<'a> {
    pub(super) message: &'a MessageView,
    pk: i64,
    pub slot: BlockSlot,
    pub group: u64,
    pub local: u64,
    pub kind: String,
    pub encrypted: Option<bool>,
    pub(super) field_root: Option<i64>,
    value: Option<values::ValueId>,
}
pub enum FieldRead<'a> {
    Value(FieldView<'a>),
    /// The slot holds an unresolved conflict. It is occupied, not missing.
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
    /// Look up a message by its stored id, through its session row. Colons in
    /// the id are not parsed.
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
            .ok_or(Error::StaleMessage)
    }
    pub fn selected_message(&self, id: &MessageId) -> Result<Option<MessageView>> {
        let archive_path = self.root.join("archive.db");
        let archive = crate::open(&archive_path, true, true)?;
        db(&archive_path, || archive.execute_batch("BEGIN DEFERRED"))?;
        let archive_counter = crate::counter(&archive, &archive_path)?;
        let recovery = crate::recovery::record(&archive, &archive_path)?;
        let path = self.root.join("index.db");
        let connection = crate::open(&path, true, false)?;
        ordered_extras::configure(&connection, &path)?;
        db(&path, || connection.execute_batch("BEGIN DEFERRED"))?;
        // The first read starts the snapshot.
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
        if crate::counter(&self.archive, &archive_path)? != archive_counter {
            return Err(Error::StaleMessage);
        }
        Ok(Some(MessageView {
            read: Arc::new(Mutex::new(PinnedRead {
                connection,
                visibility: crate::open(&path, true, false)?,
                archive: Some(archive),
                archive_visibility: None,
                recovery,
                extra_orders: 0,
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
    pub fn version_hash(&self) -> Result<polylogue_model::BlobHash> {
        self.visible()?;
        let read = locked(&self.read)?;
        let bytes: Vec<u8> = db(&self.path, || {
            read.connection.query_row(
                "SELECT version_hash FROM message_body WHERE pk=?1 AND ready=1",
                [self.body],
                |r| r.get(0),
            )
        })?;
        Ok(polylogue_model::BlobHash::from_bytes(
            bytes
                .try_into()
                .map_err(|_| Error::Invariant("ready Message version unavailable".into()))?,
        ))
    }
    pub fn recovery(&self) -> Result<Option<polylogue_model::RecoveryRecord>> {
        Ok(locked(&self.read)?.recovery.clone())
    }
    pub fn provider_time(&self) -> Result<Option<i64>> {
        use polylogue_model::merge::{MessageField, Slot};
        match self.field(&Slot::Message(MessageField::At), 65536)? {
            Some(FieldRead::Value(value)) => {
                let range = value.range(0, 65536)?;
                if !range.eof {
                    return Err(metadata_admission(
                        "Message provider time",
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
    // A newer body leaves this view alone, but hiding or removing the message
    // ends it at once: check the live row on the separate visibility connection.
    pub(super) fn visible(&self) -> Result<()> {
        let read = locked(&self.read)?;
        let visible: bool = db(&self.path, || {
            read.visibility.prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM message m WHERE m.pk=?1 AND m.session_pk=?2 AND m.id=?3 AND m.hidden=0 AND m.removal_epoch=?4) AND EXISTS(SELECT 1 FROM meta WHERE key='generation' AND value=?5)")?.query_row(
            params![self.owner.message,self.owner.session,self.id.as_str(),self.removal_epoch,self.generation],|r|r.get(0))
        })?;
        if !visible {
            return Err(Error::StaleMessage);
        }
        Ok(())
    }

    pub fn id(&self) -> &MessageId {
        &self.id
    }
    pub fn body_id(&self) -> crate::contributions::BodyId {
        crate::contributions::BodyId {
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
    /// Digest of the whole normalized body and field origins, built from stored
    /// value hashes. Row ids are not enough: selection can swap in an equal body
    /// from another source. Origins are checked against the stored companion
    /// bytes and their selectors are hashed, so equal text with different
    /// attribution gets a different digest.
    pub fn read_certificate(&self) -> Result<[u8; 32]> {
        self.read_certificate_with_cancel(&mut || false)
    }
    pub fn read_certificate_with_cancel(
        &self,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<[u8; 32]> {
        self.visible()?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"native-model-body-origin-v1\0");
        for bytes in [
            self.path.as_os_str().as_encoded_bytes(),
            self.generation.as_bytes(),
            self.id.as_str().as_bytes(),
        ] {
            hash.update(&(bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        hash.update(&self.removal_epoch.to_le_bytes());
        {
            let read = locked(&self.read)?;
            crate::contributions::compatibility::hash_row(
                &read.connection,
                &self.path,
                "SELECT layout,role,written_by,prompt_kind,at,completed_at,at_until,id_kind,in_progress,on_active_path,copy_origin,usage_present,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,cache_write_5m_tokens,cache_write_1h_tokens,reasoning_tokens,web_requests,provider_cost_nusd,cost_nusd,duration_ms,words,parser_version FROM message_body WHERE pk=?1",
                self.body,
                &mut hash,
                65536,
            )?;
        }
        let mut cursor = None;
        while let Some(entry) = self.next_field(cursor.as_ref(), 65536)? {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let key =
                serde_json::to_vec(&entry.slot).map_err(|e| Error::Invariant(e.to_string()))?;
            hash.update(&(key.len() as u64).to_le_bytes());
            hash.update(&key);
            self.hash_field(&entry.value, &mut hash)?;
            cursor = Some(entry.cursor);
        }
        hash.update(b"end-message-fields\0");
        let mut i = 0;
        while let Some(block) = self.block_at(i)? {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            hash.update(&i.to_le_bytes());
            let read = locked(&self.read)?;
            crate::contributions::compatibility::hash_row(
                &read.connection,
                &self.path,
                "SELECT b.slot_kind,b.attach_key,b.kind,b.encrypted,b.blob_hash,b.direction,b.size,b.result_outcome,b.result_reason,b.result_exit_code,b.result_duration_ms,v.codec,v.length,v.hash FROM block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN body_value v ON v.pk=t.value_pk AND v.state='ready' WHERE b.pk=?1",
                block.pk,
                &mut hash,
                65536,
            )?;
            drop(read);
            let mut cursor = None;
            while let Some(entry) = block.next_field(cursor.as_ref(), 65536)? {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let key = match entry.address {
                    BlockFieldAddress::Field(kind) => {
                        hash.update(b"field\0");
                        serde_json::to_vec(&kind)
                    }
                    BlockFieldAddress::Extra(key) => {
                        hash.update(b"extra\0");
                        serde_json::to_vec(&key)
                    }
                }
                .map_err(|e| Error::Invariant(e.to_string()))?;
                hash.update(&(key.len() as u64).to_le_bytes());
                hash.update(&key);
                self.hash_field(&entry.value, &mut hash)?;
                cursor = Some(entry.cursor);
            }
            hash.update(b"end-block-fields\0");
            i += 1;
        }
        Ok(*hash.finalize().as_bytes())
    }
    fn hash_field(&self, field: &FieldRead<'_>, hash: &mut blake3::Hasher) -> Result<()> {
        let FieldRead::Value(field) = field else {
            return Err(Error::MessagePreparationPending);
        };
        let read = locked(&self.read)?;
        for value in [Some(&field.value), field.origin.as_ref()] {
            match value {
                None => {
                    hash.update(&[0]);
                }
                Some(value) => {
                    let head = values::head(&read.connection, &self.path, value)?;
                    hash.update(&[1]);
                    hash.update(head.codec.as_str().as_bytes());
                    hash.update(&head.length.to_le_bytes());
                    hash.update(&head.hash.ok_or(Error::ValueNotReady)?);
                }
            }
        }
        if let Some(origin) = &field.origin {
            let range = values::range(&read.connection, &self.path, origin, 0, 65536)?;
            if !range.eof {
                return Err(metadata_admission("Message field origin", 65537));
            }
            let archive = read
                .archive
                .as_ref()
                .ok_or_else(|| Error::Invariant("Message archive snapshot missing".into()))?;
            crate::claude_bodies::verify_origin(
                archive,
                self.path.parent().ok_or(Error::Range)?,
                &range.bytes,
            )?;
        }
        hash.update(&[u8::from(field.supplied)]);
        hash.update(&[u8::from(field.source_time.is_some())]);
        hash.update(&field.source_time.unwrap_or_default().to_le_bytes());
        Ok(())
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
        let key = crate::contributions::writer::field_key(slot, allowance)?;
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
    pub(super) fn field_at(&self, root: Option<i64>, key: &[u8]) -> Result<Option<FieldRead<'_>>> {
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
    /// The block at a public slot (the nth item or nth reasoning block). Reads
    /// at most 32 directory nodes. The slot is not the source group/local
    /// address.
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
    /// The `index`-th block in source order, counting items and reasoning
    /// blocks together.
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
    /// The field's stored origin descriptor (JSON), if any. A ready value is
    /// intact; whether its selector is valid is for the caller to check.
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
/// A tool result block's stored outcome, exit code and duration.
pub struct ToolResultHeader {
    pub outcome: polylogue_model::ToolOutcome,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<u64>,
}
impl<'a> BlockView<'a> {
    /// The tool result columns, or `None` for other block kinds. They are read
    /// even when no field was supplied. An unknown outcome stays unknown.
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
    /// A field of this block. The slot must name this block.
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
        let key = crate::contributions::writer::block_field_key(slot, allowance)?;
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

// Search indexes only messages on the active path that are not copies.
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
