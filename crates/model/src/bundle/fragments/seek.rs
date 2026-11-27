//! Source text coordinates kept in an acknowledged read execution. These are
//! seek instructions, not copied text or a check of a complete source hash.
use super::{TextCoverage, TextFragmentError};
use crate::Reference;
use serde::{Deserialize, Serialize};

crate::stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TextField {
        Text => "text",
        Thinking => "thinking",
        ToolOutput => "tool_output",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceTextBody {
    /// Original Message block ordinal or canonical Document block ordinal.
    /// This is never a text-body sequence index or local database primary key.
    pub block: u64,
    pub field: TextField,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceTextPiece {
    Body {
        body: SourceTextBody,
    },
    Separator {
        before: SourceTextBody,
        after: SourceTextBody,
    },
    /// A complete, accessible container with no text bodies; no block invented.
    Empty,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceTextPosition {
    pub piece: SourceTextPiece,
    /// UTF-8 byte offset inside the exact field or two-LF separator.
    pub byte: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceSpanSeek {
    pub start: SourceTextPosition,
    pub end: SourceTextPosition,
}
impl SourceSpanSeek {
    /// Shape checks only. The Reader validates the fields and boundaries;
    /// the acknowledged Run binds the global resolved coordinate mapping.
    pub fn validate(
        &self,
        container: &Reference,
        coverage: &TextCoverage,
    ) -> Result<(), TextFragmentError> {
        if coverage.start > coverage.end
            || coverage
                .total_bytes
                .is_some_and(|total| coverage.end > total)
            || !matches!(
                container,
                Reference::Message(_) | Reference::Block { .. } | Reference::Document(_)
            )
        {
            return Err(TextFragmentError::Origin);
        }
        for endpoint in [self.start, self.end] {
            match endpoint.piece {
                SourceTextPiece::Body { body } => {
                    if let Reference::Block { n, .. } = container
                        && body.block != *n
                    {
                        return Err(TextFragmentError::Origin);
                    }
                }
                SourceTextPiece::Separator { before, after } => {
                    if matches!(container, Reference::Block { .. })
                        || endpoint.byte > 2
                        || before.block >= after.block
                    {
                        return Err(TextFragmentError::Origin);
                    }
                }
                SourceTextPiece::Empty => {
                    if endpoint.byte != 0
                        || coverage.start != 0
                        || coverage.end != 0
                        || matches!(container, Reference::Block { .. })
                    {
                        return Err(TextFragmentError::Origin);
                    }
                }
            }
        }
        if matches!(self.start.piece, SourceTextPiece::Empty)
            != matches!(self.end.piece, SourceTextPiece::Empty)
        {
            return Err(TextFragmentError::Origin);
        }
        if self.start.piece == self.end.piece
            && self.end.byte.checked_sub(self.start.byte) != Some(coverage.end - coverage.start)
        {
            return Err(TextFragmentError::Origin);
        }
        let start = piece_order(self.start.piece);
        let end = piece_order(self.end.piece);
        if start > end || (start == end && self.start.piece != self.end.piece) {
            return Err(TextFragmentError::Origin);
        }
        Ok(())
    }
}

fn piece_order(piece: SourceTextPiece) -> (u64, u8) {
    match piece {
        SourceTextPiece::Body { body } => (body.block, 0),
        SourceTextPiece::Separator { before, .. } => (before.block, 1),
        SourceTextPiece::Empty => (0, 0),
    }
}

#[cfg(test)]
#[path = "seek/tests.rs"]
mod tests;
