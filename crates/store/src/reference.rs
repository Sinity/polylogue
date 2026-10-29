//! Original recorded text, before view filtering, masking or rendition. Source
//! hashes stream these same chunks and never materialize a whole session.
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    BlobHash, Block, Reference, SpanBounds, SpanError, SpanUnit, TEXT_BODY_SEPARATOR,
    TextBodyOffsets, text_body,
};
use rusqlite::{OptionalExtension, params};
use std::ops::ControlFlow;

type Visit<'a> = dyn FnMut(&str) -> Result<ControlFlow<()>> + 'a;
struct Text<'a, 'v> {
    visit: &'a mut Visit<'v>,
    offsets: TextBodyOffsets,
    reference: String,
}
impl Text<'_, '_> {
    fn body(&mut self, text: &str) -> Result<ControlFlow<()>> {
        let range = self.offsets.push(text.len() as u64).map_err(span_error)?;
        if range.start != 0 && (self.visit)(TEXT_BODY_SEPARATOR)?.is_break() {
            return Ok(ControlFlow::Break(()));
        }
        (self.visit)(text)
    }
    fn block(&mut self, block: &Block) -> Result<ControlFlow<()>> {
        match text_body(block).map_err(|error| StoreError::ReferenceUnavailable {
            reference: self.reference.clone(),
            reason: error.to_string(),
        })? {
            Some(text) => self.body(text),
            None => Ok(ControlFlow::Continue(())),
        }
    }
}
fn unavailable(reference: &Reference, reason: &str) -> StoreError {
    StoreError::ReferenceUnavailable {
        reference: reference.to_string(),
        reason: reason.into(),
    }
}
fn span_error(error: SpanError) -> StoreError {
    StoreError::Invalid(error.to_string())
}

impl Reader<'_> {
    /// `Some(0)` is accessible empty text; `None` is an absent target. A target
    /// whose text is unavailable returns a typed error and never an empty hash.
    pub fn visit_reference_text(
        &self,
        reference: &Reference,
        mut visit: impl FnMut(&str) -> Result<()>,
    ) -> Result<Option<u64>> {
        let mut bytes = 0u64;
        let found = self.reference_text(reference, true, &mut |text| {
            visit(text)?;
            bytes = bytes.checked_add(text.len() as u64).ok_or_else(|| {
                StoreError::Invalid("reference text length exceeds its address space".into())
            })?;
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(found.then_some(bytes))
    }
    /// Stop before loading another text body. Continue carries the complete
    /// byte length; Break carries bytes visited before successful cancellation.
    /// Completeness checks for full reference identities remain in force.
    pub fn visit_reference_text_while(
        &self,
        reference: &Reference,
        mut visit: impl FnMut(&str) -> Result<ControlFlow<()>>,
    ) -> Result<Option<ControlFlow<u64, u64>>> {
        let mut bytes = 0u64;
        let mut stopped = false;
        let found = self.reference_text(reference, true, &mut |text| {
            bytes = bytes.checked_add(text.len() as u64).ok_or_else(|| {
                StoreError::Invalid("reference text length exceeds its address space".into())
            })?;
            let flow = visit(text)?;
            stopped = flow.is_break();
            Ok(flow)
        })?;
        Ok(found.then_some(if stopped {
            ControlFlow::Break(bytes)
        } else {
            ControlFlow::Continue(bytes)
        }))
    }
    pub fn reference_hash(&self, reference: &Reference) -> Result<Option<BlobHash>> {
        let mut hash = blake3::Hasher::new();
        Ok(self
            .visit_reference_text(reference, |text| {
                hash.update(text.as_bytes());
                Ok(())
            })?
            .map(|_| BlobHash(*hash.finalize().as_bytes())))
    }
    pub(crate) fn source_message_projection(
        &self,
        message: i64,
        reference: &Reference,
    ) -> Result<polylogue_model::bundle::MessageProjection> {
        let conn = self.conn()?;
        let (hash, layout, blocks): (Option<Vec<u8>>, Option<Vec<u8>>, i64) = conn.query_row(
            "SELECT s.version_hash,p.layout,(SELECT count(*) FROM block WHERE message_pk=m.pk) FROM message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.shown=1 LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE m.pk=?1",
            [message], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let hash = hash
            .ok_or_else(|| unavailable(reference, "the canonical source version is unavailable"))?;
        crate::versions::message_projection(
            &hash,
            layout.as_deref(),
            usize::try_from(blocks)
                .map_err(|_| StoreError::Invalid("invalid stored block count".into()))?,
        )
    }
    fn require_complete_message(&self, message: i64, reference: &Reference) -> Result<()> {
        let projection = self.source_message_projection(message, reference)?;
        if projection.block_count != projection.block_ordinals.len() as u64 {
            return Err(unavailable(
                reference,
                "some original source blocks are unavailable",
            ));
        }
        Ok(())
    }
    fn reference_text(
        &self,
        reference: &Reference,
        complete: bool,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let conn = self.conn()?;
        match reference {
            Reference::Session(id) => {
                let pk: Option<i64> = conn
                    .query_row(
                        "SELECT pk FROM session WHERE id=?1",
                        [id.to_string()],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(pk) = pk else { return Ok(false) };
                let mut messages = conn.prepare_cached(
                    "SELECT pk FROM message WHERE session_pk=?1 AND duplicate_of IS NULL",
                )?;
                let mut rows = messages.query([pk])?;
                while let Some(row) = rows.next()? {
                    self.require_complete_message(row.get(0)?, reference)?;
                }
                let mut text = Text {
                    visit,
                    offsets: TextBodyOffsets::default(),
                    reference: reference.to_string(),
                };
                crate::ordered_blocks::visit_text(conn, pk, |block| text.block(&block))?;
                Ok(true)
            }
            Reference::Message(id) => {
                let pk: Option<i64> = conn
                    .query_row(
                        "SELECT pk FROM message WHERE id=?1",
                        [id.to_string()],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(pk) = pk else { return Ok(false) };
                let projection = self.source_message_projection(pk, reference)?;
                if complete {
                    self.require_complete_message(pk, reference)?;
                }
                let mut text = Text {
                    visit,
                    offsets: TextBodyOffsets::default(),
                    reference: reference.to_string(),
                };
                let mut statement = conn
                    .prepare_cached("SELECT pk,idx FROM block WHERE message_pk=?1 ORDER BY idx")?;
                let mut rows = statement.query([pk])?;
                while let Some(row) = rows.next()? {
                    let index = usize::try_from(row.get::<_, i64>(1)?)
                        .map_err(|_| StoreError::Invalid("invalid stored block index".into()))?;
                    if !complete
                        && projection.block_ordinals.get(index).copied() != Some(index as u64)
                    {
                        return Err(unavailable(
                            reference,
                            "an earlier original source block is unavailable",
                        ));
                    }
                    if text
                        .block(&crate::ordered_blocks::one(conn, row.get(0)?)?)?
                        .is_break()
                    {
                        return Ok(true);
                    }
                }
                if !complete && projection.block_count != projection.block_ordinals.len() as u64 {
                    return Err(unavailable(
                        reference,
                        "some original source blocks are unavailable",
                    ));
                }
                Ok(true)
            }
            Reference::Block { message, n } => {
                let message_pk: Option<i64> = conn
                    .query_row(
                        "SELECT pk FROM message WHERE id=?1",
                        [message.to_string()],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(message_pk) = message_pk else {
                    return Ok(false);
                };
                let projection = self.source_message_projection(message_pk, reference)?;
                let pk: Option<i64> = conn.query_row(
                    "SELECT pk FROM block WHERE message_pk=?1 AND coalesce(ordinal,cast(idx AS TEXT))=?2",
                    params![message_pk,n.to_string()], |row| row.get(0),
                ).optional()?;
                let Some(pk) = pk else {
                    return if *n < projection.block_count {
                        Err(unavailable(
                            reference,
                            "the original source block is unavailable",
                        ))
                    } else {
                        Ok(false)
                    };
                };
                let block = crate::ordered_blocks::one(conn, pk)?;
                if !matches!(
                    block,
                    Block::Text { .. }
                        | Block::Thinking { .. }
                        | Block::ToolResult {
                            output: Some(_),
                            ..
                        }
                ) {
                    return Err(unavailable(
                        reference,
                        "the block has no recorded textual body",
                    ));
                }
                let _ = Text {
                    visit,
                    offsets: TextBodyOffsets::default(),
                    reference: reference.to_string(),
                }
                .block(&block)?;
                Ok(true)
            }
            Reference::Document(id) => {
                let Some(document) = self.select_document(id)? else {
                    return Ok(false);
                };
                let mut text = Text {
                    visit,
                    offsets: TextBodyOffsets::default(),
                    reference: reference.to_string(),
                };
                let mut error = None;
                document.visit_text_blocks(|block| match text.block(&block) {
                    Ok(flow) => flow,
                    Err(failed) => {
                        error = Some(failed);
                        ControlFlow::Break(())
                    }
                })?;
                if let Some(error) = error {
                    return Err(error);
                }
                Ok(true)
            }
            Reference::MessageRange { first, last } => self.range_text(first, last, visit),
            Reference::Variant { revision, part } => {
                let variant = if let Some(id) = &revision.revision {
                    self.variant(id)?
                        .filter(|variant| variant.root == revision.root)
                } else {
                    let mut heads = self.current_variants(&revision.root)?;
                    if heads.len() > 1 {
                        return Err(StoreError::Ambiguous {
                            reference: reference.to_string(),
                            candidates: heads
                                .iter()
                                .map(|v| format!("variant:{}@{}", v.root, v.id))
                                .collect(),
                        });
                    }
                    heads.pop()
                };
                let Some(variant) = variant else {
                    return Ok(false);
                };
                let mut text = Text {
                    visit,
                    offsets: TextBodyOffsets::default(),
                    reference: reference.to_string(),
                };
                let mut present = part.is_none();
                self.visit_variant_parts_while(&variant.id, |value| {
                    if part.is_none_or(|n| value.n == n) {
                        present = true;
                        return Ok(text.body(&value.body)?.is_continue());
                    }
                    Ok(true)
                })?;
                Ok(present)
            }
            Reference::Assertion(revision) => {
                let Some(id) = &revision.revision else {
                    return Err(unavailable(
                        reference,
                        "current assertion text needs its canonical revision selection",
                    ));
                };
                let Some(assertion) = self
                    .note(id)?
                    .filter(|assertion| assertion.first_note == revision.root)
                else {
                    return Ok(false);
                };
                let _ = visit(assertion.value.as_deref().unwrap_or(""))?;
                Ok(true)
            }
            Reference::Span {
                container,
                unit,
                start,
                end,
            } => self.span_text(container, *unit, *start, *end, visit),
            Reference::Page(_) => Err(unavailable(
                reference,
                "page text needs its prepared slot snapshot",
            )),
            _ => Err(unavailable(
                reference,
                "this reference has no local recorded text source",
            )),
        }
    }
    fn range_text(
        &self,
        first: &polylogue_model::MessageId,
        last: &polylogue_model::MessageId,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let reference = Reference::MessageRange {
            first: first.clone(),
            last: last.clone(),
        };
        type Position = (i64, String, i64, i64, String, bool);
        let conn = self.conn()?;
        let position = |id: &polylogue_model::MessageId| -> Result<Option<Position>> {
            Ok(conn.query_row("SELECT session_pk,order_input_id,seq,record_position,id,duplicate_of IS NOT NULL FROM message WHERE id=?1",[id.to_string()],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?))).optional()?)
        };
        let (Some(first), Some(last)) = (position(first)?, position(last)?) else {
            return Ok(false);
        };
        if first.0 != last.0 {
            return Err(StoreError::Invalid(
                "a message range must stay within one physical session".into(),
            ));
        }
        if first.5 || last.5 {
            return Err(StoreError::ReferenceUnavailable {
                reference: format!("message:{}..{}", first.4, last.4),
                reason: "range endpoints are outside the physical source selection".into(),
            });
        }
        if (&first.1, first.2, first.3, &first.4) > (&last.1, last.2, last.3, &last.4) {
            return Err(StoreError::Invalid(
                "message range ends before it starts".into(),
            ));
        }
        const RANGE: &str = "m.session_pk=?1 AND m.duplicate_of IS NULL AND (m.order_input_id,m.seq,m.record_position,m.id)>=(?2,?3,?4,?5) AND (m.order_input_id,m.seq,m.record_position,m.id)<=(?6,?7,?8,?9)";
        let mut messages =
            conn.prepare_cached(&format!("SELECT m.pk FROM message m WHERE {RANGE}"))?;
        let mut rows = messages.query(params![
            first.0, first.1, first.2, first.3, first.4, last.1, last.2, last.3, last.4
        ])?;
        while let Some(row) = rows.next()? {
            self.require_complete_message(row.get(0)?, &reference)?;
        }
        let mut statement = conn.prepare_cached(&format!("SELECT b.pk FROM message m JOIN block b ON b.message_pk=m.pk WHERE {RANGE} AND b.kind IN ('text','thinking','tool_result') ORDER BY m.order_input_id,m.seq,m.record_position,m.id,b.idx"))?;
        let mut rows = statement.query(params![
            first.0, first.1, first.2, first.3, first.4, last.1, last.2, last.3, last.4
        ])?;
        let mut text = Text {
            visit,
            offsets: TextBodyOffsets::default(),
            reference: reference.to_string(),
        };
        while let Some(row) = rows.next()? {
            if text
                .block(&crate::ordered_blocks::one(conn, row.get(0)?)?)?
                .is_break()
            {
                return Ok(true);
            }
        }
        Ok(true)
    }
    fn span_text(
        &self,
        container: &Reference,
        unit: SpanUnit,
        start: u64,
        end: u64,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let mut bounds = SpanBounds::new(unit, start, end).map_err(span_error)?;
        let found = self.reference_text(container, true, &mut |text| {
            bounds.push(text).map_err(span_error)?;
            Ok(ControlFlow::Continue(()))
        })?;
        if !found {
            return Ok(false);
        }
        let range = bounds.finish().map_err(span_error)?;
        let mut position = 0u64;
        self.reference_text(container, true, &mut |text| {
            let next = position
                .checked_add(text.len() as u64)
                .ok_or(SpanError::Overflow)
                .map_err(span_error)?;
            let begin = range.start.max(position);
            let end = range.end.min(next);
            if begin < end
                && visit(&text[(begin - position) as usize..(end - position) as usize])?.is_break()
            {
                return Ok(ControlFlow::Break(()));
            }
            position = next;
            Ok(ControlFlow::Continue(()))
        })
    }
}
#[cfg(test)]
mod tests;
