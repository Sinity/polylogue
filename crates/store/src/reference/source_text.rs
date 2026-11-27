//! The source Message/Block/Document text walker. Full text, bounded spans and
//! resumed spans share these source loaders and the same two-LF body recipe.
use super::{ReferenceText, unavailable};
use crate::ConnExt as _;
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    Block, Reference, TEXT_BODY_SEPARATOR,
    bundle::fragments::seek::{SourceTextBody, SourceTextPiece, SourceTextPosition, TextField},
    text_body,
};
use rusqlite::params;
use std::ops::ControlFlow;

type Visit<'a> = dyn FnMut(SourceTextPiece, &str) -> Result<ControlFlow<()>> + 'a;
struct Text<'a, 'v> {
    visit: &'a mut Visit<'v>,
    previous: Option<SourceTextBody>,
    expected: Option<SourceTextPiece>,
    reference: &'a Reference,
}
impl Text<'_, '_> {
    fn piece(&mut self, piece: SourceTextPiece, text: &str) -> Result<ControlFlow<()>> {
        if self
            .expected
            .take()
            .is_some_and(|expected| expected != piece)
        {
            return Err(unavailable(
                self.reference,
                "the bound source text piece is unavailable",
            ));
        }
        (self.visit)(piece, text)
    }
    fn prime(&mut self, from: Option<&SourceTextPosition>) -> Result<ControlFlow<()>> {
        if let Some(SourceTextPosition {
            piece: SourceTextPiece::Separator { before, after },
            ..
        }) = from
        {
            // The authenticated endpoint was derived by this walker. Seeking a
            // recorded separator uses the two bodies it sits between, not either body's text.
            let flow = (self.visit)(
                SourceTextPiece::Separator {
                    before: *before,
                    after: *after,
                },
                TEXT_BODY_SEPARATOR,
            )?;
            self.expected = Some(SourceTextPiece::Body { body: *after });
            return Ok(flow);
        }
        Ok(ControlFlow::Continue(()))
    }
    fn block(&mut self, ordinal: u64, block: &Block) -> Result<ControlFlow<()>> {
        let Some(text) =
            text_body(block).map_err(|error| unavailable(self.reference, &error.to_string()))?
        else {
            return Ok(ControlFlow::Continue(()));
        };
        let field = match block {
            Block::Text { .. } => TextField::Text,
            Block::Thinking { .. } => TextField::Thinking,
            Block::ToolResult { .. } => TextField::ToolOutput,
            _ => {
                return Err(StoreError::stored_data(
                    "text body has no source text field",
                ));
            }
        };
        let body = SourceTextBody {
            block: ordinal,
            field,
        };
        if let Some(before) = self.previous
            && self
                .piece(
                    SourceTextPiece::Separator {
                        before,
                        after: body,
                    },
                    TEXT_BODY_SEPARATOR,
                )?
                .is_break()
        {
            return Ok(ControlFlow::Break(()));
        }
        self.previous = Some(body);
        self.piece(SourceTextPiece::Body { body }, text)
    }
}
impl Reader<'_> {
    pub(crate) fn source_text_while(
        &self,
        reference: &Reference,
        complete: bool,
        from: Option<&SourceTextPosition>,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let found = ReferenceText::new(self.conn()?, self.conn()?).source_text_while(
            reference,
            complete,
            from,
            &mut |piece, text| {
                self.check()?;
                visit(piece, text)
            },
        )?;
        self.check()?;
        Ok(found)
    }
}
impl ReferenceText<'_> {
    pub(crate) fn source_text_while(
        &self,
        reference: &Reference,
        complete: bool,
        from: Option<&SourceTextPosition>,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let first = match from.map(|position| position.piece) {
            Some(SourceTextPiece::Body { body }) => Some(body.block),
            Some(SourceTextPiece::Separator { after, .. }) => Some(after.block),
            Some(SourceTextPiece::Empty) | None => None,
        };
        let mut text = Text {
            visit,
            previous: None,
            expected: from.map(|position| position.piece),
            reference,
        };
        let conn = self.original;
        match reference {
            Reference::Message(id) => {
                let pk: Option<i64> =
                    conn.scalar_optional("SELECT pk FROM message WHERE id=?1", [id.to_string()])?;
                let Some(pk) = pk else { return Ok(false) };
                let selection = self.source_block_selection(pk, reference)?;
                if complete {
                    self.require_complete_message(pk, reference)?;
                }
                if text.prime(from)?.is_break() {
                    return Ok(true);
                }
                let from = if let Some(first) = first {
                    let index: Option<i64> = conn.scalar_optional("SELECT position FROM block WHERE message_pk=?1 AND coalesce(source_position,cast(position AS TEXT))=?2", params![pk, first.to_string()])?;
                    index.ok_or_else(|| {
                        unavailable(reference, "the bound source block is unavailable")
                    })?
                } else {
                    0
                };
                let mut statement = conn.prepare_cached(
                    "SELECT pk,position,kind FROM block WHERE message_pk=?1 AND position>=?2 ORDER BY position",
                )?;
                let mut rows = statement.query(params![pk, from])?;
                while let Some(row) = rows.next()? {
                    let index = usize::try_from(row.get::<_, i64>(1)?)
                        .map_err(|_| crate::StoreError::stored(1, "invalid stored block index"))?;
                    let ordinal =
                        selection
                            .block_ordinals
                            .get(index)
                            .copied()
                            .ok_or_else(|| {
                                StoreError::stored_data(
                                    "stored block has no original source ordinal",
                                )
                            })?;
                    if !complete && ordinal != index as u64 {
                        return Err(unavailable(
                            reference,
                            "an earlier original source block is unavailable",
                        ));
                    }
                    let kind: String = row.get(2)?;
                    if matches!(kind.as_str(), "text" | "thinking" | "tool_result")
                        && text
                            .block(ordinal, &crate::ordered_blocks::one(conn, row.get(0)?)?)?
                            .is_break()
                    {
                        return Ok(true);
                    }
                }
                if !complete && selection.block_count != selection.block_ordinals.len() as u64 {
                    return Err(unavailable(
                        reference,
                        "some original source blocks are unavailable",
                    ));
                }
            }
            Reference::Block { message, n } => {
                let pk: Option<i64> = conn
                    .scalar_optional("SELECT pk FROM message WHERE id=?1", [message.to_string()])?;
                let Some(pk) = pk else { return Ok(false) };
                let selection = self.source_block_selection(pk, reference)?;
                let block: Option<i64> = conn.scalar_optional("SELECT pk FROM block WHERE message_pk=?1 AND coalesce(source_position,cast(position AS TEXT))=?2", params![pk,n.to_string()])?;
                let Some(block) = block else {
                    return if *n < selection.block_count {
                        Err(unavailable(
                            reference,
                            "the original source block is unavailable",
                        ))
                    } else {
                        Ok(false)
                    };
                };
                let block = crate::ordered_blocks::one(conn, block)?;
                if text_body(&block)
                    .map_err(|error| unavailable(reference, &error.to_string()))?
                    .is_none()
                {
                    return Err(unavailable(
                        reference,
                        "the block has no recorded textual body",
                    ));
                }
                let _ = text.block(*n, &block)?;
            }
            Reference::Document(id) => {
                let Some(document) = crate::documents::text_read(conn, id)? else {
                    return Ok(false);
                };
                if text.prime(from)?.is_break() {
                    return Ok(true);
                }
                let mut result = Ok(ControlFlow::Continue(()));
                document.visit_text_blocks_from(first.unwrap_or(0), |ordinal, block| {
                    result = text.block(ordinal, &block);
                    match &result {
                        Ok(flow) => *flow,
                        Err(_) => ControlFlow::Break(()),
                    }
                })?;
                let _ = result?;
            }
            _ => {
                return Err(unavailable(
                    reference,
                    "source text coordinates require a Message, Block or Document",
                ));
            }
        }
        if text
            .expected
            .is_some_and(|expected| expected != SourceTextPiece::Empty)
        {
            return Err(unavailable(
                reference,
                "the bound source text piece is unavailable",
            ));
        }
        Ok(true)
    }
}
