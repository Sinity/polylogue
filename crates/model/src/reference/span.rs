use super::SpanUnit;
use crate::Block;
use std::ops::Range;

pub const TEXT_BODY_SEPARATOR: &str = "\n\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TextBodyError {
    #[error("the thinking text was not recorded")]
    NotRecorded,
}

pub fn text_body(block: &Block) -> Result<Option<&str>, TextBodyError> {
    match block {
        Block::Text { text }
        | Block::Thinking {
            text: Some(text), ..
        } => Ok(Some(text)),
        Block::Thinking { text: None, .. } => Err(TextBodyError::NotRecorded),
        Block::ToolResult { output, .. } => Ok(output.as_deref()),
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpanError {
    #[error("span end precedes its start")]
    Reversed,
    #[error("span bounds split a Unicode character")]
    SplitCharacter,
    #[error("span bounds exceed the source text")]
    OutOfRange,
    #[error("source text exceeds its address space")]
    Overflow,
}

/// Resolves original-source offsets across UTF-8 chunks without retaining the
/// text. Finishing validates both bounds before a caller emits a replacement.
pub struct SpanBounds {
    unit: SpanUnit,
    start: u64,
    end: u64,
    units: u64,
    bytes: u64,
    start_byte: Option<u64>,
    end_byte: Option<u64>,
}
impl SpanBounds {
    pub fn new(unit: SpanUnit, start: u64, end: u64) -> Result<Self, SpanError> {
        if start > end {
            return Err(SpanError::Reversed);
        }
        Ok(Self {
            unit,
            start,
            end,
            units: 0,
            bytes: 0,
            start_byte: (start == 0).then_some(0),
            end_byte: (end == 0).then_some(0),
        })
    }
    pub fn push(&mut self, text: &str) -> Result<(), SpanError> {
        for character in text.chars() {
            let width = match self.unit {
                SpanUnit::Bytes => character.len_utf8() as u64,
                SpanUnit::CodePoints => 1,
                SpanUnit::Utf16 => character.len_utf16() as u64,
            };
            let units = self.units.checked_add(width).ok_or(SpanError::Overflow)?;
            let bytes = self
                .bytes
                .checked_add(character.len_utf8() as u64)
                .ok_or(SpanError::Overflow)?;
            if [self.start, self.end]
                .iter()
                .any(|bound| *bound > self.units && *bound < units)
            {
                return Err(SpanError::SplitCharacter);
            }
            if self.start == units {
                self.start_byte = Some(bytes);
            }
            if self.end == units {
                self.end_byte = Some(bytes);
            }
            self.units = units;
            self.bytes = bytes;
        }
        Ok(())
    }
    /// Both requested bounds have been observed in the original text prefix.
    /// This proves their UTF-8 coordinates, not completeness of the source.
    /// Offsets count verbatim source text; no Unicode normalization is applied.
    pub fn resolved(&self) -> Option<Range<u64>> {
        Some(self.start_byte?..self.end_byte?)
    }
    /// The original start coordinate is known even before the end arrives.
    pub fn resolved_start(&self) -> Option<u64> {
        self.start_byte
    }
    pub fn resolved_end(&self) -> Option<u64> {
        self.end_byte
    }
    pub fn finish(self) -> Result<Range<u64>, SpanError> {
        self.resolved().ok_or(SpanError::OutOfRange)
    }
}

/// Original body positions include separators even around recorded empty
/// bodies. Structured blocks never call this helper.
#[derive(Default)]
pub struct TextBodyOffsets {
    bytes: u64,
    separated: bool,
}
impl TextBodyOffsets {
    pub fn push(&mut self, byte_len: u64) -> Result<Range<u64>, SpanError> {
        let start = self
            .bytes
            .checked_add(if self.separated {
                TEXT_BODY_SEPARATOR.len() as u64
            } else {
                0
            })
            .ok_or(SpanError::Overflow)?;
        let end = start.checked_add(byte_len).ok_or(SpanError::Overflow)?;
        self.bytes = end;
        self.separated = true;
        Ok(start..end)
    }
}

/// Indices enumerate the supplied text bodies, never stored block ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextSpanPiece {
    Body {
        body_index: u64,
        start: u64,
        end: u64,
    },
    Separator {
        after_body: u64,
        start: u8,
        end: u8,
    },
}

/// Splits resolved UTF-8 byte bounds over original bodies and their separators.
/// An empty source with no bodies has no invented body insertion anchor.
pub fn span_bodies(
    span: Range<u64>,
    body_lengths: impl IntoIterator<Item = u64>,
) -> Result<Vec<TextSpanPiece>, SpanError> {
    if span.start > span.end {
        return Err(SpanError::Reversed);
    }
    let mut offsets = TextBodyOffsets::default();
    let mut pieces = Vec::new();
    let mut total = 0;
    let mut index = 0u64;
    for len in body_lengths {
        let body = offsets.push(len)?;
        if index != 0 {
            let begin = span.start.max(total);
            let end = span.end.min(body.start);
            let insertion = span.is_empty() && span.start > total && span.start < body.start;
            if begin < end || insertion {
                pieces.push(TextSpanPiece::Separator {
                    after_body: index - 1,
                    start: (begin - total) as u8,
                    end: (end - total) as u8,
                });
            }
        }
        let begin = span.start.max(body.start);
        let end = span.end.min(body.end);
        let insertion = span.is_empty() && span.start >= body.start && span.start <= body.end;
        if begin < end || insertion {
            pieces.push(TextSpanPiece::Body {
                body_index: index,
                start: begin - body.start,
                end: end - body.start,
            });
        }
        total = body.end;
        index = index.checked_add(1).ok_or(SpanError::Overflow)?;
    }
    if span.end > total {
        return Err(SpanError::OutOfRange);
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests;
