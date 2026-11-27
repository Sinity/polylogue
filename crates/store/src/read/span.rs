//! A portable span observes only selected original bytes. Resolving a known
//! prefix never establishes completeness or hashes the unavailable remainder.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::{Reader, Result, StoreError};
use polylogue_model::bundle::{
    TextSpanOrigin,
    fragments::seek::{
        SourceSpanSeek, SourceTextBody, SourceTextPiece, SourceTextPosition, TextField,
    },
};
use polylogue_model::{
    BlobHash, Reference, SpanBounds,
    bundle::{TextCoverage, TextSourceVersion},
};
use rusqlite::params;
use std::ops::ControlFlow;

/// Keeps the Reader snapshot that resolved both coordinate bounds.
/// The version identifies the original value; selected bytes cannot verify it.
pub struct SpanSource<'reader, 'pool> {
    reader: &'reader Reader<'pool>,
    origin: TextSpanOrigin,
    version: TextSourceVersion,
    coverage: TextCoverage,
    committed: Option<u64>,
    seek: SourceSpanSeek,
}
impl SpanSource<'_, '_> {
    pub fn container(&self) -> &Reference {
        let Reference::Span { container, .. } = &self.origin.reference else {
            unreachable!("a selected span has a span reference")
        };
        container
    }
    /// The original coordinates of the selected source. Tool-output aliases
    /// are bound to their result block here before source evidence is stored.
    pub fn origin(&self) -> &TextSpanOrigin {
        &self.origin
    }
    pub fn version(&self) -> &TextSourceVersion {
        &self.version
    }
    pub fn coverage(&self) -> TextCoverage {
        self.coverage
    }
    pub fn committed(&self) -> Option<u64> {
        self.committed
    }

    pub fn seek(&self) -> SourceSpanSeek {
        self.seek
    }

    /// Seeks directly to the bound source-text piece, never decoding earlier bodies.
    pub fn visit_text(
        &self,
        mut visit: impl FnMut(&str) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        self.reader.check()?;
        if self.seek.start.piece == SourceTextPiece::Empty {
            return Ok(ControlFlow::Continue(()));
        }
        let mut first = true;
        let mut ended = false;
        let mut cancelled = false;
        let mut selected = 0u64;
        self.reader.source_text_while(
            self.container(),
            false,
            Some(&self.seek.start),
            &mut |piece, text| {
                self.reader.check()?;
                let start = if first { self.seek.start.byte } else { 0 };
                first = false;
                let end = if piece == self.seek.end.piece {
                    ended = true;
                    self.seek.end.byte
                } else {
                    text.len() as u64
                };
                let start = usize::try_from(start).map_err(|_| stale(self.container()))?;
                let end = usize::try_from(end).map_err(|_| stale(self.container()))?;
                let text = text
                    .get(start..end)
                    .ok_or_else(|| stale(self.container()))?;
                selected = selected
                    .checked_add(text.len() as u64)
                    .ok_or_else(|| stale(self.container()))?;
                if !text.is_empty() && visit(text)?.is_break() {
                    cancelled = true;
                    return Ok(ControlFlow::Break(()));
                }
                Ok(if ended {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                })
            },
        )?;
        self.reader.check()?;
        if cancelled {
            return Ok(ControlFlow::Break(()));
        }
        if !ended || selected != self.coverage.end - self.coverage.start {
            return Err(stale(self.container()));
        }
        Ok(ControlFlow::Continue(()))
    }
}
impl<'pool> Reader<'pool> {
    /// Resolves absolute byte/codepoint/UTF-16 bounds in the known original
    /// prefix. Missing text before either bound remains unavailable. Text after
    /// the resolved end is not inspected; its bytes and total remain unknown.
    pub fn select_span_source<'reader>(
        &'reader self,
        span: &Reference,
    ) -> Result<Option<SpanSource<'reader, 'pool>>> {
        let Reference::Span {
            container,
            unit,
            start,
            end,
        } = span
        else {
            return Err(StoreError::internal("a span source needs a Span reference"));
        };
        let result;
        let container = if let Reference::ToolCall(session, call) = container.as_ref() {
            let Some(block) = super::tool_result_reference(self.conn()?, session, call)? else {
                return Ok(None);
            };
            result = block;
            &result
        } else {
            container.as_ref()
        };
        let Some((version, committed)) = self.span_identity(container)? else {
            return Ok(None);
        };
        let mut bounds = SpanBounds::new(*unit, *start, *end).map_err(bad_span)?;
        let mut bytes = 0u64;
        let mut stopped = false;
        let mut start_position = None;
        let mut end_position = None;
        let found = self.source_text_while(container, false, None, &mut |piece, text| {
            bounds.push(text).map_err(bad_span)?;
            let next = bytes
                .checked_add(text.len() as u64)
                .or_internal("span text length overflows")?;
            if let Some(bound) = bounds.resolved_start()
                && bound >= bytes
                && bound <= next
                && (start_position.is_none() || bound == bytes)
            {
                start_position = Some(SourceTextPosition {
                    piece,
                    byte: bound - bytes,
                });
            }
            if let Some(bound) = bounds.resolved_end()
                && bound >= bytes
                && bound <= next
                && end_position.is_none()
            {
                end_position = Some(SourceTextPosition {
                    piece,
                    byte: bound - bytes,
                });
            }
            bytes = next;
            stopped = bounds.resolved().is_some();
            Ok(if stopped {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            })
        })?;
        if !found {
            return Ok(None);
        }
        let range = bounds.finish().map_err(bad_span)?;
        let empty = SourceTextPosition {
            piece: SourceTextPiece::Empty,
            byte: 0,
        };
        let seek = SourceSpanSeek {
            start: start_position.unwrap_or(empty),
            end: end_position.unwrap_or(empty),
        };
        Ok(Some(SpanSource {
            reader: self,
            origin: TextSpanOrigin {
                reference: Reference::Span {
                    container: Box::new(container.clone()),
                    unit: *unit,
                    start: *start,
                    end: *end,
                },
                start: range.start,
                end: range.end,
            },
            version,
            coverage: TextCoverage {
                start: range.start,
                end: range.end,
                total_bytes: (!stopped).then_some(bytes),
            },
            committed,
            seek,
        }))
    }
    /// Replays coordinates restored from the existing authenticated Run. The
    /// caller must validate that Run's selection/scope binding before passing
    /// metadata here; a plain DTO does not vouch for its global byte mapping.
    pub fn resume_span_source<'reader>(
        &'reader self,
        origin: &TextSpanOrigin,
        version: &TextSourceVersion,
        seek: SourceSpanSeek,
    ) -> Result<SpanSource<'reader, 'pool>> {
        self.check()?;
        let Reference::Span { container, .. } = &origin.reference else {
            return Err(StoreError::internal(
                "span replay needs original Span coordinates",
            ));
        };
        version
            .validate_container(container)
            .map_err(crate::StoreError::internal)?;
        let coverage = TextCoverage {
            start: origin.start,
            end: origin.end,
            total_bytes: None,
        };
        seek.validate(container, &coverage)
            .map_err(crate::StoreError::internal)?;
        let Some((current, committed)) = self.span_identity(container)? else {
            return Err(stale(&origin.reference));
        };
        if current != *version {
            return Err(stale(&origin.reference));
        }
        for position in [seek.start, seek.end] {
            match position.piece {
                SourceTextPiece::Body { body } => self.require_span_field(container, body)?,
                SourceTextPiece::Separator { before, after } => {
                    self.require_span_field(container, before)?;
                    self.require_span_field(container, after)?;
                }
                SourceTextPiece::Empty => {}
            }
        }
        Ok(SpanSource {
            reader: self,
            origin: origin.clone(),
            version: version.clone(),
            coverage,
            committed,
            seek,
        })
    }
    fn require_span_field(&self, container: &Reference, body: SourceTextBody) -> Result<()> {
        let kind: Option<String> = match container {
            Reference::Message(id) | Reference::Block { message: id, .. } => self.conn()?.scalar_optional("SELECT b.kind FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.id=?1 AND coalesce(b.source_position,cast(b.position AS TEXT))=?2", params![id.to_string(), body.block.to_string()])?,
            Reference::Document(id) => {
                let Ok(ordinal) = i64::try_from(body.block) else { return Err(stale(container)) };
                self.conn()?.scalar_optional("SELECT b.kind FROM document_block b JOIN document d ON d.pk=b.document_pk WHERE d.id=?1 AND b.position=?2", params![id.to_string(), ordinal])?
            }
            _ => return Err(stale(container)),
        };
        let expected = match body.field {
            TextField::Text => "text",
            TextField::Thinking => "thinking",
            TextField::ToolOutput => "tool_result",
        };
        if kind.as_deref() != Some(expected) {
            return Err(stale(container));
        }
        Ok(())
    }
    fn span_identity(
        &self,
        container: &Reference,
    ) -> Result<Option<(TextSourceVersion, Option<u64>)>> {
        match container {
            Reference::Message(id) | Reference::Block { message: id, .. } => {
                let pk: Option<i64> = self
                    .conn()?
                    .scalar_optional("SELECT pk FROM message WHERE id=?1", [id.to_string()])?;
                let Some(pk) = pk else { return Ok(None) };
                let block_selection = self.source_block_selection(pk, container)?;
                Ok(Some((
                    TextSourceVersion::ContentHash {
                        record: Reference::Message(id.clone()),
                        hash: block_selection.source_version,
                    },
                    self.message_commit(id)?,
                )))
            }
            Reference::Document(id) => {
                let Some(document) = self.select_document(id)? else {
                    return Ok(None);
                };
                let hash =
                    BlobHash::parse(&document.version()).map_err(crate::StoreError::stored_data)?;
                Ok(Some((
                    TextSourceVersion::ContentHash {
                        record: container.clone(),
                        hash,
                    },
                    self.document_commit(id)?,
                )))
            }
            _ => Err(StoreError::ReferenceUnavailable {
                reference: container.to_string(),
                reason: "portable span source needs an available source version and its commit"
                    .into(),
            }),
        }
    }
}

fn stale(reference: &Reference) -> StoreError {
    StoreError::Stale {
        reference: reference.to_string(),
        current: Vec::new(),
    }
}

#[cfg(test)]
#[path = "span/tests.rs"]
mod tests;

fn bad_span(error: impl ToString) -> StoreError {
    polylogue_model::Reason::BadSpan {
        problem: error.to_string(),
    }
    .into()
}
