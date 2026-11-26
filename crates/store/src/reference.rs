//! Original recorded text, before view filtering, masking or alternate. Source
//! hashes stream these same chunks and never load a whole session at once.
mod source_text;

use crate::ConnExt as _;
use crate::OptionExt as _;
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
    reference: &'a Reference,
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
        match text_body(block).map_err(|error| unavailable(self.reference, &error.to_string()))? {
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
    polylogue_model::Reason::BadSpan {
        problem: error.to_string(),
    }
    .into()
}

impl Reader<'_> {
    /// Stop before loading another text body. Continue carries the complete
    /// byte length; Break carries bytes visited before successful cancellation.
    /// Completeness checks for full reference identities remain in force.
    pub fn visit_reference_text(
        &self,
        reference: &Reference,
        mut visit: impl FnMut(&str) -> Result<ControlFlow<()>>,
    ) -> Result<Option<ControlFlow<u64, u64>>> {
        let mut bytes = 0u64;
        let mut stopped = false;
        let found = ReferenceText::new(self.conn()?, self.conn()?).reference_text(
            reference,
            true,
            &mut |text| {
                self.check()?;
                bytes = bytes.checked_add(text.len() as u64).ok_or_else(|| {
                    StoreError::internal("reference text length exceeds its address space")
                })?;
                let flow = visit(text)?;
                stopped = flow.is_break();
                Ok(flow)
            },
        )?;
        self.check()?;
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
                Ok(ControlFlow::Continue(()))
            })?
            .map(|_| BlobHash(*hash.finalize().as_bytes())))
    }
    pub(crate) fn source_block_selection(
        &self,
        message: i64,
        reference: &Reference,
    ) -> Result<polylogue_model::bundle::ShownBlocks> {
        ReferenceText::new(self.conn()?, self.conn()?).source_block_selection(message, reference)
    }
}

/// Source text is read from the pinned original store and authored text from
/// the authored connection (the write transaction, or the preflight overlay's
/// transaction). Recursive spans keep that split.
pub(crate) struct ReferenceText<'a> {
    original: &'a rusqlite::Connection,
    authored: &'a rusqlite::Connection,
}
impl<'a> ReferenceText<'a> {
    pub(crate) fn new(
        original: &'a rusqlite::Connection,
        authored: &'a rusqlite::Connection,
    ) -> Self {
        Self { original, authored }
    }
    pub(crate) fn source_block_selection(
        &self,
        message: i64,
        reference: &Reference,
    ) -> Result<polylogue_model::bundle::ShownBlocks> {
        let conn = self.original;
        let (hash, layout, blocks): (Option<Vec<u8>>, Option<Vec<u8>>, i64) = conn.query_row(
            "SELECT m.version_hash,m.layout,(SELECT count(*) FROM block WHERE message_pk=m.pk) FROM message m WHERE m.pk=?1",
            [message], |row| <(_, _, _)>::try_from(row),
        )?;
        let hash =
            hash.ok_or_else(|| unavailable(reference, "the source version is unavailable"))?;
        crate::encoding::message_block_selection(
            &hash,
            layout.as_deref(),
            usize::try_from(blocks)
                .map_err(|_| crate::StoreError::stored(2, "invalid stored block count"))?,
        )
    }
    pub(crate) fn require_complete_message(
        &self,
        message: i64,
        reference: &Reference,
    ) -> Result<()> {
        let block_selection = self.source_block_selection(message, reference)?;
        if block_selection.block_count != block_selection.block_ordinals.len() as u64 {
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
        let conn = self.original;
        match reference {
            Reference::Session(id) => {
                let pk: Option<i64> =
                    conn.scalar_optional("SELECT pk FROM session WHERE id=?1", [id.to_string()])?;
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
                    reference,
                };
                crate::ordered_blocks::visit_text(conn, pk, |block| text.block(&block))?;
                Ok(true)
            }
            Reference::Message(_) | Reference::Block { .. } | Reference::Document(_) => {
                self.source_text_while(reference, complete, None, &mut |_, text| visit(text))
            }
            Reference::MessageRange { first, last } => self.range_text(first, last, visit),
            Reference::Alternate { revision, part } => {
                let alternate = if let Some(id) = &revision.revision {
                    crate::alternates::get_record(self.authored, id)?
                        .filter(|alternate| alternate.root == revision.root)
                } else {
                    let mut heads = crate::alternates::heads(self.authored, &revision.root)?;
                    if heads.len() > 1 {
                        return Err(StoreError::Ambiguous {
                            reference: reference.to_string(),
                            candidates: heads
                                .iter()
                                .map(|v| crate::alternates::reference_string(&v.root, &v.id))
                                .collect(),
                        });
                    }
                    heads.pop()
                };
                let Some(alternate) = alternate else {
                    return Ok(false);
                };
                let mut text = Text {
                    visit,
                    offsets: TextBodyOffsets::default(),
                    reference,
                };
                let mut present = part.is_none();
                crate::alternates::parts_while(self.authored, &alternate.id, |value| {
                    if part.is_none_or(|n| value.n == n) {
                        present = true;
                        return Ok(text.body(&value.body)?.is_continue());
                    }
                    Ok(true)
                })?;
                Ok(present)
            }
            Reference::Annotation(revision) => match self.annotation_text(revision, visit)? {
                TextState::Missing => Ok(false),
                TextState::Present => Ok(true),
                TextState::NotApplicable => Err(unavailable(
                    reference,
                    "this annotation has no recorded text body",
                )),
            },
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
    fn annotation_text(
        &self,
        revision: &polylogue_model::RevisionReference,
        visit: &mut Visit<'_>,
    ) -> Result<TextState> {
        let reference = Reference::Annotation(revision.clone());
        let id = if let Some(id) = &revision.revision {
            id.clone()
        } else {
            let heads = crate::annotations::heads(self.authored, &reference)?;
            if heads.len() > 1 {
                return Err(StoreError::Ambiguous {
                    reference: reference.to_string(),
                    candidates: heads.iter().map(ToString::to_string).collect(),
                });
            }
            if let Some(Reference::Annotation(head)) = heads.first() {
                head.revision
                    .clone()
                    .or_stored("annotation head names no revision")?
            } else {
                if crate::annotations::family_present(self.authored, &revision.root)? {
                    return Err(unavailable(
                        &reference,
                        "the annotation family has no current text",
                    ));
                }
                return Ok(TextState::Missing);
            }
        };
        let Some(annotation) = crate::annotations::get(self.authored, &id)?
            .filter(|annotation| annotation.fact.root == revision.root)
        else {
            return Ok(TextState::Missing);
        };
        annotation
            .fact
            .validate()
            .map_err(crate::StoreError::stored_data)?;
        let Some(text) = annotation.fact.text() else {
            return Ok(TextState::NotApplicable);
        };
        let _ = visit(text)?;
        Ok(TextState::Present)
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
        type Position = (i64, Vec<u8>, String, bool);
        let conn = self.original;
        let position = |id: &polylogue_model::MessageId| -> Result<Option<Position>> {
            Ok(conn.query_row("SELECT session_pk,session_order,id,duplicate_of IS NOT NULL FROM message WHERE id=?1",[id.to_string()],|row| <(_, _, _, _)>::try_from(row)).optional()?)
        };
        let (Some(first), Some(last)) = (position(first)?, position(last)?) else {
            return Ok(false);
        };
        if first.0 != last.0 {
            return Err(polylogue_model::Reason::RangeAcrossSessions {}.into());
        }
        if first.3 || last.3 {
            return Err(StoreError::ReferenceUnavailable {
                reference: format!("message:{}..{}", first.2, last.2),
                reason: "range endpoints are outside the physical source selection".into(),
            });
        }
        if (&first.1, &first.2) > (&last.1, &last.2) {
            return Err(polylogue_model::Reason::RangeEndsBeforeStart {}.into());
        }
        const RANGE: &str = "m.session_pk=?1 AND m.duplicate_of IS NULL AND (m.session_order,m.id)>=(?2,?3) AND (m.session_order,m.id)<=(?4,?5)";
        let mut messages =
            conn.prepare_cached(&format!("SELECT m.pk FROM message m WHERE {RANGE}"))?;
        let mut rows = messages.query(params![first.0, first.1, first.2, last.1, last.2])?;
        while let Some(row) = rows.next()? {
            self.require_complete_message(row.get(0)?, &reference)?;
        }
        let mut statement = conn.prepare_cached(&format!("SELECT b.pk FROM message m JOIN block b ON b.message_pk=m.pk WHERE {RANGE} AND b.kind IN ('text','thinking','tool_result') ORDER BY m.session_order,m.id,b.idx"))?;
        let mut rows = statement.query(params![first.0, first.1, first.2, last.1, last.2])?;
        let mut text = Text {
            visit,
            offsets: TextBodyOffsets::default(),
            reference: &reference,
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

/// The quote a link to a span keeps: the span's text and up to 32
/// characters on either side, from the container's current text. `None` when
/// the reference is not a span or its container has no recorded text yet.
pub(crate) fn link_quote(
    original: &rusqlite::Connection,
    authored: &rusqlite::Connection,
    reference: &Reference,
) -> Result<Option<polylogue_model::annotation::Quote>> {
    let Reference::Span {
        container,
        unit,
        start,
        end,
    } = reference
    else {
        return Ok(None);
    };
    pinned(original)?;
    let view = ReferenceText::new(original, authored);
    let mut bounds = SpanBounds::new(*unit, *start, *end).map_err(span_error)?;
    let mut text = String::new();
    let found = view.reference_text(container, true, &mut |chunk| {
        bounds.push(chunk).map_err(span_error)?;
        text.push_str(chunk);
        Ok(ControlFlow::Continue(()))
    })?;
    if !found {
        return Ok(None);
    }
    let range = bounds.finish().map_err(span_error)?;
    let span =
        |position: u64| usize::try_from(position).map_err(|_| span_error(SpanError::Overflow));
    Ok(polylogue_model::annotation::Quote::of(
        &text,
        span(range.start)?,
        span(range.end)?,
    ))
}

/// Full-text hashes require the caller's existing original-source snapshot.
/// Authored rows never stand in for source Message/Document text.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LinkHash {
    Missing,
    NotApplicable,
    Text(BlobHash),
}
enum TextState {
    Missing,
    NotApplicable,
    Present,
}
fn hash_source(read: impl FnOnce(&mut Visit<'_>) -> Result<TextState>) -> Result<LinkHash> {
    let mut hasher = blake3::Hasher::new();
    let state = read(&mut |text| {
        hasher.update(text.as_bytes());
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(match state {
        TextState::Missing => LinkHash::Missing,
        TextState::NotApplicable => LinkHash::NotApplicable,
        TextState::Present => LinkHash::Text(BlobHash(*hasher.finalize().as_bytes())),
    })
}
fn pinned(original: &rusqlite::Connection) -> Result<()> {
    if original.is_autocommit() {
        return Err(StoreError::internal(
            "reference hashing requires a pinned original snapshot",
        ));
    }
    Ok(())
}
#[cfg(test)]
fn hash(
    original: &rusqlite::Connection,
    authored: &rusqlite::Connection,
    reference: &Reference,
) -> Result<Option<BlobHash>> {
    pinned(original)?;
    match hash_source(|visit| {
        ReferenceText::new(original, authored)
            .reference_text(reference, true, visit)
            .map(|present| {
                if present {
                    TextState::Present
                } else {
                    TextState::Missing
                }
            })
    })? {
        LinkHash::Text(hash) => Ok(Some(hash)),
        LinkHash::Missing => Ok(None),
        LinkHash::NotApplicable => Err(unavailable(
            reference,
            "this reference has no recorded text body",
        )),
    }
}
/// Link capture distinguishes an absent target from an existing,
/// validated non-text annotation. Neither null state is a future text wildcard.
pub(crate) fn link_hash(
    original: &rusqlite::Connection,
    authored: &rusqlite::Connection,
    reference: &Reference,
) -> Result<LinkHash> {
    pinned(original)?;
    let view = ReferenceText::new(original, authored);
    hash_source(|visit| match reference {
        Reference::Annotation(revision) => view.annotation_text(revision, visit),
        _ => view.reference_text(reference, true, visit).map(|present| {
            if present {
                TextState::Present
            } else {
                TextState::Missing
            }
        }),
    })
}

#[cfg(test)]
mod tests;
