//! One selected call and its recorded counterpart, inside the Reader snapshot.
//! The indexed foreign-key relation decides pairing; neither arrival order nor
//! a matching name elsewhere in the archive supplies a result.
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    Block, MessageId, SessionId,
    bundle::MessageProjection,
    read::{Source, SourceBlock, ToolPair, ValueState},
};
use rusqlite::{Row, params};

struct BlockFact {
    pk: i64,
    index: usize,
    ordinal: u64,
    message: MessageId,
    session: SessionId,
    call_id: Option<String>,
    version: Option<Vec<u8>>,
    layout: Option<Vec<u8>>,
    blocks: usize,
}
impl BlockFact {
    fn read(row: &Row<'_>, offset: usize) -> Result<Self> {
        let ordinal: String = row.get(offset + 2)?;
        Ok(Self {
            pk: row.get(offset)?,
            index: row.get(offset + 1)?,
            ordinal: ordinal.parse().map_err(|_| {
                StoreError::stored(
                    offset + 2,
                    format!("block ordinal {ordinal:?} is not a number"),
                )
            })?,
            message: crate::encoding::Columns::id(row, offset + 3)?,
            session: crate::encoding::Columns::id(row, offset + 4)?,
            call_id: row.get(offset + 5)?,
            version: row.get(offset + 6)?,
            layout: row.get(offset + 7)?,
            blocks: row.get(offset + 8)?,
        })
    }
    fn projection(&self) -> Result<Option<MessageProjection>> {
        self.version
            .as_ref()
            .map(|version| {
                crate::versions::message_projection(version, self.layout.as_deref(), self.blocks)
            })
            .transpose()
    }
    fn visible(&self) -> Result<bool> {
        Ok(self.projection()?.is_some_and(|projection| {
            projection.block_ordinals.get(self.index) == Some(&self.ordinal)
        }))
    }
    fn source(&self) -> Source {
        Source::block(&self.message, &self.session, self.ordinal, false)
    }
}
struct PairFact {
    call: BlockFact,
    result: Option<BlockFact>,
    matched: bool,
}
impl Reader<'_> {
    /// Exactly one selected original call or result position. An unavailable
    /// or orphan result stays absent; only current shown blocks are read.
    /// The caller releases this pair before preparing another call's payload.
    pub fn message_tool_pair(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<ToolPair>> {
        self.message_tool_pair_detail(message, original_block_ordinal, true)
    }

    /// Collapsed evidence restores the recorded block header only. Remote
    /// output text, structured data and attachment bodies remain unread.
    pub fn message_tool_pair_header(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<ToolPair>> {
        self.message_tool_pair_detail(message, original_block_ordinal, false)
    }

    /// The result's emitted shape, without restoring output or structured
    /// payloads. Data presence checks SQL nullness; attachments come only from
    /// the packed canonical header before collapsed evidence clears them.
    pub fn message_tool_pair_result_shape(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<(bool, u64)>> {
        self.check()?;
        let Some(fact) = self.tool_pair_fact(message, original_block_ordinal)? else {
            return Ok(None);
        };
        let Some(result) = fact.result.filter(|_| fact.matched) else {
            return Ok(None);
        };
        self.check()?;
        let (body, data): (Vec<u8>, bool) = self.conn()?.query_row(
            "SELECT body,data IS NOT NULL FROM block WHERE pk=?1",
            [result.pk],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let Block::ToolResult {
            call_id,
            attachments,
            ..
        } = crate::encoding::unpack_block(&body)?
        else {
            return Err(invalid("indexed result is not a tool result"));
        };
        if call_id != fact.call.call_id {
            return Err(invalid(
                "indexed result identity differs from its canonical block",
            ));
        }
        let count = u64::try_from(attachments.len())
            .map_err(|_| invalid("attachment count exceeds its address space"))?;
        self.check()?;
        Ok(Some((data, count)))
    }

    /// The indexed sources only, for the caller's existing selection-membership
    /// check before any off-selected counterpart payload is read.
    pub fn message_tool_pair_sources(
        &self,
        message: &MessageId,
        original_block_ordinal: u64,
    ) -> Result<Option<(Source, Option<Source>)>> {
        self.check()?;
        let pair = self
            .tool_pair_fact(message, original_block_ordinal)?
            .map(|fact| {
                (
                    fact.call.source(),
                    fact.result
                        .filter(|_| fact.matched)
                        .map(|result| result.source()),
                )
            });
        self.check()?;
        Ok(pair)
    }

    fn message_tool_pair_detail(
        &self,
        message: &MessageId,
        ordinal: u64,
        full: bool,
    ) -> Result<Option<ToolPair>> {
        self.check()?;
        let Some(fact) = self.tool_pair_fact(message, ordinal)? else {
            return Ok(None);
        };
        if !fact.call.visible()? {
            return Ok(None);
        }
        let call = self.tool_pair_block(&fact.call, full)?;
        let Block::ToolCall { call_id, .. } = &call.block else {
            return Err(invalid("indexed call is not a tool call"));
        };
        if call_id != &fact.call.call_id {
            return Err(invalid(
                "indexed call identity differs from its canonical block",
            ));
        }
        let result = if fact.matched {
            fact.result
                .as_ref()
                .map(|result| self.tool_pair_block(result, full))
                .transpose()?
        } else {
            None
        };
        if let Some(result) = &result {
            let Block::ToolResult { call_id, .. } = &result.block else {
                return Err(invalid("indexed result is not a tool result"));
            };
            if call_id != &fact.call.call_id {
                return Err(invalid(
                    "indexed result identity differs from its canonical block",
                ));
            }
        }
        let duration_ms = match result.as_ref().map(|result| &result.block) {
            Some(Block::ToolResult {
                reported_duration_ms: Some(duration),
                ..
            }) => ValueState::known(*duration),
            _ => ValueState::Unknown {
                reason: Some("no canonical reported duration".into()),
            },
        };
        let pair = ToolPair {
            emit_at: call.source.reference.clone(),
            call,
            result,
            running: false,
            duration_ms,
            error_line: None,
            child: None,
            edit: None,
        };
        pair.validate()
            .map_err(|error| invalid(&error.to_string()))?;
        self.check()?;
        Ok(Some(pair))
    }

    fn tool_pair_fact(&self, message: &MessageId, ordinal: u64) -> Result<Option<PairFact>> {
        let canonical = crate::parser_context::canonical_message(self.conn()?, message)?;
        let message = &canonical;

        let conn = self.conn()?;
        let mut input = conn.prepare_cached(
            "SELECT b.pk,b.idx,coalesce(b.ordinal,cast(b.idx AS TEXT)),m.id,s.id,b.call_id,
                    (SELECT version_hash FROM message_version WHERE pk=shown.version_pk),projection.layout,(SELECT count(*) FROM block WHERE message_pk=m.pk)
             FROM message m JOIN session s ON s.pk=m.session_pk
             JOIN block b ON b.message_pk=m.pk AND b.session_pk=m.session_pk
             LEFT JOIN message_source shown ON shown.message_pk=m.pk AND shown.input_pk=m.input_pk AND shown.chain=m.source_chain
             LEFT JOIN message_projection projection ON projection.version_pk=shown.version_pk
             WHERE m.id=?1 AND m.duplicate_of IS NULL AND coalesce(b.ordinal,cast(b.idx AS TEXT))=?2",
        )?;
        let mut rows = input.query(params![message.to_string(), ordinal.to_string()])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let input = BlockFact::read(row, 0)?;
        if rows.next()?.is_some() {
            return Err(invalid(
                "one original block ordinal has conflicting current rows",
            ));
        }
        if !input.visible()? {
            return Ok(None);
        }
        self.check()?;
        let mut statement = conn.prepare_cached(
            "SELECT cb.pk,cb.idx,coalesce(cb.ordinal,cast(cb.idx AS TEXT)),cm.id,cs.id,cb.call_id,
                    (SELECT version_hash FROM message_version WHERE pk=shown.version_pk),projection.layout,(SELECT count(*) FROM block WHERE message_pk=cm.pk),
                    rb.pk,rb.idx,coalesce(rb.ordinal,cast(rb.idx AS TEXT)),rm.id,rs.id,rb.call_id,
                    (SELECT version_hash FROM message_version WHERE pk=rshown.version_pk),rprojection.layout,(SELECT count(*) FROM block WHERE message_pk=rm.pk),
                    rb.kind='tool_result' AND rb.session_pk=t.session_pk AND rm.session_pk=t.session_pk
                      AND rm.duplicate_of IS NULL AND cb.call_id IS NOT NULL AND rb.call_id=cb.call_id AS matched
             FROM tool_call t
             JOIN block cb ON cb.pk=t.call_block_pk AND cb.session_pk=t.session_pk AND cb.kind='tool_call'
             JOIN message cm ON cm.pk=cb.message_pk AND cm.session_pk=t.session_pk
             JOIN session cs ON cs.pk=cm.session_pk
             LEFT JOIN message_source shown ON shown.message_pk=cm.pk AND shown.input_pk=cm.input_pk AND shown.chain=cm.source_chain
             LEFT JOIN message_projection projection ON projection.version_pk=shown.version_pk
             LEFT JOIN block rb ON rb.pk=t.result_block_pk
             LEFT JOIN message rm ON rm.pk=rb.message_pk
             LEFT JOIN session rs ON rs.pk=rm.session_pk
             LEFT JOIN message_source rshown ON rshown.message_pk=rm.pk AND rshown.input_pk=rm.input_pk AND rshown.chain=rm.source_chain
             LEFT JOIN message_projection rprojection ON rprojection.version_pk=rshown.version_pk
             WHERE (t.call_block_pk=?1 OR t.result_block_pk=?1) AND cm.duplicate_of IS NULL AND t.call_id IS cb.call_id",
        )?;
        let mut rows = statement.query([input.pk])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let call = BlockFact::read(row, 0)?;
        let result = if row.get::<_, Option<i64>>(9)?.is_some()
            && row.get::<_, Option<String>>(12)?.is_some()
            && row.get::<_, Option<String>>(13)?.is_some()
        {
            Some(BlockFact::read(row, 9)?)
        } else {
            None
        };
        let matched = row.get::<_, Option<bool>>(18)?.unwrap_or(false)
            && result
                .as_ref()
                .is_some_and(|result| result.session == call.session)
            && result
                .as_ref()
                .map(BlockFact::visible)
                .transpose()?
                .unwrap_or(false);
        if rows.next()?.is_some() {
            return Err(invalid(
                "one source block has conflicting current tool relations",
            ));
        }
        if !call.visible()? || (input.pk != call.pk && !matched) {
            return Ok(None);
        }
        Ok(Some(PairFact {
            call,
            result,
            matched,
        }))
    }
    fn tool_pair_block(&self, fact: &BlockFact, full: bool) -> Result<SourceBlock> {
        self.check()?;
        let conn = self.conn()?;
        let mut block = if full {
            crate::ordered_blocks::one(conn, fact.pk)?
        } else {
            let body: Vec<u8> =
                conn.query_row("SELECT body FROM block WHERE pk=?1", [fact.pk], |row| {
                    row.get(0)
                })?;
            crate::encoding::unpack_block(&body)?
        };
        if full && matches!(block, Block::ToolResult { .. }) {
            self.check()?;
            let data: Option<Vec<u8>> =
                conn.query_row("SELECT data FROM block WHERE pk=?1", [fact.pk], |row| {
                    row.get(0)
                })?;
            if let Some(data) = data {
                crate::encoding::restore_data(&mut block, &data)?;
            }
        }
        if !full {
            match &mut block {
                Block::ToolCall { input, .. } => *input = serde_json::Value::Null,
                Block::ToolResult {
                    output,
                    data,
                    attachments,
                    ..
                } => {
                    *output = None;
                    *data = None;
                    attachments.clear();
                }
                _ => {}
            }
        }
        self.check()?;
        Ok(SourceBlock {
            source: fact.source(),
            block,
        })
    }
}
fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

#[cfg(test)]
#[path = "tool_pairs/tests.rs"]
mod tests;
