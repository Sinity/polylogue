//! Original recorded text, before view filtering, masking or rendition. Source
//! hashes stream these same chunks and never materialize a whole session.
use crate::{Reader, Result, StoreError};
use polylogue_model::{BlobHash, Block, Reference, SpanUnit};
use rusqlite::{OptionalExtension, params};
use std::ops::ControlFlow;

type Visit<'a> = dyn FnMut(&str) -> Result<()> + 'a;
struct Text<'a, 'v> {
    visit: &'a mut Visit<'v>,
    separated: bool,
    reference: String,
}
impl Text<'_, '_> {
    fn body(&mut self, text: &str) -> Result<()> {
        if self.separated {
            (self.visit)("\n\n")?;
        }
        self.separated = true;
        (self.visit)(text)
    }
    fn block(&mut self, block: &Block) -> Result<()> {
        match block {
            Block::Text { text }
            | Block::Thinking {
                text: Some(text), ..
            } => self.body(text),
            Block::Thinking { text: None, .. } => Err(StoreError::ReferenceUnavailable {
                reference: self.reference.clone(),
                reason: "the thinking text was not recorded".into(),
            }),
            Block::ToolResult {
                output: Some(text), ..
            } => self.body(text),
            _ => Ok(()),
        }
    }
}
fn unavailable(reference: &Reference, reason: &str) -> StoreError {
    StoreError::ReferenceUnavailable {
        reference: reference.to_string(),
        reason: reason.into(),
    }
}
fn width(character: char, unit: SpanUnit) -> u64 {
    match unit {
        SpanUnit::Bytes => character.len_utf8() as u64,
        SpanUnit::CodePoints => 1,
        SpanUnit::Utf16 => character.len_utf16() as u64,
    }
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
        let found = self.reference_text(reference, &mut |text| {
            visit(text)?;
            bytes = bytes.checked_add(text.len() as u64).ok_or_else(|| {
                StoreError::Invalid("reference text length exceeds its address space".into())
            })?;
            Ok(())
        })?;
        Ok(found.then_some(bytes))
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
    fn reference_text(&self, reference: &Reference, visit: &mut Visit<'_>) -> Result<bool> {
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
                let mut text = Text {
                    visit,
                    separated: false,
                    reference: reference.to_string(),
                };
                crate::ordered_blocks::visit_text(conn, pk, |block| {
                    text.block(&block.block)?;
                    Ok(ControlFlow::Continue(()))
                })?;
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
                let mut text = Text {
                    visit,
                    separated: false,
                    reference: reference.to_string(),
                };
                let mut statement =
                    conn.prepare_cached("SELECT pk FROM block WHERE message_pk=?1 AND kind IN ('text','thinking','tool_result') ORDER BY idx")?;
                let mut rows = statement.query([pk])?;
                while let Some(row) = rows.next()? {
                    text.block(&crate::ordered_blocks::one(conn, row.get(0)?)?)?;
                }
                Ok(true)
            }
            Reference::Block { message, n } => {
                let Ok(n) = i64::try_from(*n) else {
                    return Ok(false);
                };
                let pk:Option<i64>=conn.query_row("SELECT b.pk FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.id=?1 AND b.idx=?2",params![message.to_string(),n],|row|row.get(0)).optional()?;
                let Some(pk) = pk else { return Ok(false) };
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
                Text {
                    visit,
                    separated: false,
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
                    separated: false,
                    reference: reference.to_string(),
                };
                let mut error = None;
                document.visit_blocks(|block| {
                    if let Err(failed) = text.block(&block) {
                        error = Some(failed);
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
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
                    separated: false,
                    reference: reference.to_string(),
                };
                let mut present = part.is_none();
                self.visit_variant_parts(&variant.id, |value| {
                    if part.is_none_or(|n| value.n == n) {
                        present = true;
                        text.body(&value.body)?;
                    }
                    Ok(())
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
                visit(assertion.value.as_deref().unwrap_or(""))?;
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
        let mut statement=conn.prepare_cached("SELECT b.pk FROM message m JOIN block b ON b.message_pk=m.pk WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND b.kind IN ('text','thinking','tool_result') AND (m.order_input_id,m.seq,m.record_position,m.id)>=(?2,?3,?4,?5) AND (m.order_input_id,m.seq,m.record_position,m.id)<=(?6,?7,?8,?9) ORDER BY m.order_input_id,m.seq,m.record_position,m.id,b.idx")?;
        let mut rows = statement.query(params![
            first.0, first.1, first.2, first.3, first.4, last.1, last.2, last.3, last.4
        ])?;
        let mut text = Text {
            visit,
            separated: false,
            reference: reference.to_string(),
        };
        while let Some(row) = rows.next()? {
            text.block(&crate::ordered_blocks::one(conn, row.get(0)?)?)?;
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
        if start > end {
            return Err(StoreError::Invalid("span end precedes its start".into()));
        }
        let mut position = 0;
        let mut start_valid = start == 0;
        let mut end_valid = end == 0;
        let found = self.reference_text(container, &mut |text| {
            for character in text.chars() {
                position += width(character, unit);
                start_valid |= position == start;
                end_valid |= position == end;
            }
            Ok(())
        })?;
        if !found {
            return Ok(false);
        }
        if end > position || !start_valid || !end_valid {
            return Err(StoreError::Invalid(
                "span bounds exceed the source text or split a character".into(),
            ));
        }
        let mut position = 0;
        self.reference_text(container, &mut |text| {
            let mut begin = None;
            let mut finish = 0;
            for (byte, character) in text.char_indices() {
                let next = position + width(character, unit);
                if position >= start && next <= end {
                    begin.get_or_insert(byte);
                    finish = byte + character.len_utf8();
                }
                position = next;
            }
            if let Some(begin) = begin {
                visit(&text[begin..finish])?;
            }
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests;
