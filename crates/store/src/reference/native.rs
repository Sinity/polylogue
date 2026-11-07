//! The native Message/Block/Document text walker. Full text, bounded spans and
//! resumed spans share these source loaders and the same two-LF body recipe.
use super::{ReferenceText, unavailable};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    Block, Reference, TEXT_BODY_SEPARATOR,
    bundle::fragments::seek::{NativeTextBody, NativeTextPiece, NativeTextPosition, TextField},
    text_body,
};
use rusqlite::{OptionalExtension, params};
use std::ops::ControlFlow;

type Visit<'a> = dyn FnMut(NativeTextPiece, &str) -> Result<ControlFlow<()>> + 'a;
struct Text<'a, 'v> {
    visit: &'a mut Visit<'v>,
    previous: Option<NativeTextBody>,
    expected: Option<NativeTextPiece>,
    reference: &'a Reference,
}
impl Text<'_, '_> {
    fn piece(&mut self, piece: NativeTextPiece, text: &str) -> Result<ControlFlow<()>> {
        if self
            .expected
            .take()
            .is_some_and(|expected| expected != piece)
        {
            return Err(unavailable(
                self.reference,
                "the bound native text piece is unavailable",
            ));
        }
        (self.visit)(piece, text)
    }
    fn prime(&mut self, from: Option<&NativeTextPosition>) -> Result<ControlFlow<()>> {
        if let Some(NativeTextPosition {
            piece: NativeTextPiece::Separator { before, after },
            ..
        }) = from
        {
            // The authenticated endpoint was derived by this walker. Seeking a
            // recorded separator uses its two native anchors, not either body.
            let flow = (self.visit)(
                NativeTextPiece::Separator {
                    before: *before,
                    after: *after,
                },
                TEXT_BODY_SEPARATOR,
            )?;
            self.expected = Some(NativeTextPiece::Body { body: *after });
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
            _ => return Err(StoreError::Invalid("text body has no native field".into())),
        };
        let body = NativeTextBody {
            block: ordinal,
            field,
        };
        if let Some(before) = self.previous
            && self
                .piece(
                    NativeTextPiece::Separator {
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
        self.piece(NativeTextPiece::Body { body }, text)
    }
}
impl Reader<'_> {
    pub(crate) fn native_text_while(
        &self,
        reference: &Reference,
        complete: bool,
        from: Option<&NativeTextPosition>,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let found = ReferenceText::new(self.conn()?, self.conn()?).native_text_while(
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
    pub(crate) fn native_text_while(
        &self,
        reference: &Reference,
        complete: bool,
        from: Option<&NativeTextPosition>,
        visit: &mut Visit<'_>,
    ) -> Result<bool> {
        let (first, previous) = match from.map(|position| position.piece) {
            Some(NativeTextPiece::Body { body }) => (Some(body.block), None),
            Some(NativeTextPiece::Separator { after, .. }) => (Some(after.block), None),
            Some(NativeTextPiece::Empty) | None => (None, None),
        };
        let mut text = Text {
            visit,
            previous,
            expected: from.map(|position| position.piece),
            reference,
        };
        let conn = self.original;
        match reference {
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
                if text.prime(from)?.is_break() {
                    return Ok(true);
                }
                let from = if let Some(first) = first {
                    let index: Option<i64> = conn.query_row("SELECT idx FROM block WHERE message_pk=?1 AND coalesce(ordinal,cast(idx AS TEXT))=?2", params![pk, first.to_string()], |row| row.get(0)).optional()?;
                    index.ok_or_else(|| {
                        unavailable(reference, "the bound source block is unavailable")
                    })?
                } else {
                    0
                };
                let mut statement = conn.prepare_cached(
                    "SELECT pk,idx,kind FROM block WHERE message_pk=?1 AND idx>=?2 ORDER BY idx",
                )?;
                let mut rows = statement.query(params![pk, from])?;
                while let Some(row) = rows.next()? {
                    let index = usize::try_from(row.get::<_, i64>(1)?)
                        .map_err(|_| StoreError::Invalid("invalid stored block index".into()))?;
                    let ordinal =
                        projection
                            .block_ordinals
                            .get(index)
                            .copied()
                            .ok_or_else(|| {
                                StoreError::Invalid(
                                    "stored block has no original source ordinal".into(),
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
                if !complete && projection.block_count != projection.block_ordinals.len() as u64 {
                    return Err(unavailable(
                        reference,
                        "some original source blocks are unavailable",
                    ));
                }
            }
            Reference::Block { message, n } => {
                let pk: Option<i64> = conn
                    .query_row(
                        "SELECT pk FROM message WHERE id=?1",
                        [message.to_string()],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(pk) = pk else { return Ok(false) };
                let projection = self.source_message_projection(pk, reference)?;
                let block: Option<i64> = conn.query_row("SELECT pk FROM block WHERE message_pk=?1 AND coalesce(ordinal,cast(idx AS TEXT))=?2", params![pk,n.to_string()], |row| row.get(0)).optional()?;
                let Some(block) = block else {
                    return if *n < projection.block_count {
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
                    "native text coordinates require a Message, Block or Document",
                ));
            }
        }
        if text
            .expected
            .is_some_and(|expected| expected != NativeTextPiece::Empty)
        {
            return Err(unavailable(
                reference,
                "the bound native text piece is unavailable",
            ));
        }
        Ok(true)
    }
}
