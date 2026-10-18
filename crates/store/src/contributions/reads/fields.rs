//! Walking the fields of a selected message or block in slot-key order.
use super::{
    SharedRead, locked,
    message::{BlockView, FieldRead, MessageView},
    metadata_admission,
};
use crate::contributions::{directory as d, writer::BlockFieldKey};
use crate::{Error, Result, db};
use rusqlite::{Connection, params};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FieldScope {
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
    /// The next field of the selected body, in slot-key order. The slot key
    /// only orders the walk; it does not match fields across candidates.
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
    pub(super) fn field_successor(
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
            Some(_) => return Err(Error::StaleMessage),
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
        let address = match serde_json::from_slice::<BlockFieldKey<'_>>(&key)
            .map_err(|e| Error::Invariant(format!("selected Block semantic key: {e}")))?
        {
            BlockFieldKey::Field(field) => BlockFieldAddress::Field(field),
            BlockFieldKey::Extra(key) => BlockFieldAddress::Extra(key.into_owned()),
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
