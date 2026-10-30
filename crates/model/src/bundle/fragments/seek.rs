//! Native text coordinates retained in an acknowledged read execution. These
//! are seek instructions, not copied text or proof of a complete source hash.
use super::{TextCoverage, TextFragmentError};
use crate::Reference;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TextField {
    Text,
    Thinking,
    ToolOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeTextBody {
    /// Original Message block ordinal or canonical Document block ordinal.
    /// This is never a text-body sequence index or local database primary key.
    pub block: u64,
    pub field: TextField,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeTextPiece {
    Body {
        body: NativeTextBody,
    },
    Separator {
        before: NativeTextBody,
        after: NativeTextBody,
    },
    /// A complete, accessible container with no text bodies; no block invented.
    Empty,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeTextPosition {
    pub piece: NativeTextPiece,
    /// UTF-8 byte offset inside the exact field or two-LF separator.
    pub byte: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NativeSpanSeek {
    pub start: NativeTextPosition,
    pub end: NativeTextPosition,
}
impl NativeSpanSeek {
    /// Shape checks only. The Reader validates actual fields and boundaries;
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
                NativeTextPiece::Body { body } => {
                    if let Reference::Block { n, .. } = container
                        && body.block != *n
                    {
                        return Err(TextFragmentError::Origin);
                    }
                }
                NativeTextPiece::Separator { before, after } => {
                    if matches!(container, Reference::Block { .. })
                        || endpoint.byte > 2
                        || before.block >= after.block
                    {
                        return Err(TextFragmentError::Origin);
                    }
                }
                NativeTextPiece::Empty => {
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
        if matches!(self.start.piece, NativeTextPiece::Empty)
            != matches!(self.end.piece, NativeTextPiece::Empty)
        {
            return Err(TextFragmentError::Origin);
        }
        if self.start.piece == self.end.piece
            && self.end.byte.checked_sub(self.start.byte) != Some(coverage.end - coverage.start)
        {
            return Err(TextFragmentError::Origin);
        }
        let start = native_order(self.start.piece);
        let end = native_order(self.end.piece);
        if start > end || (start == end && self.start.piece != self.end.piece) {
            return Err(TextFragmentError::Origin);
        }
        Ok(())
    }
}

fn native_order(piece: NativeTextPiece) -> (u64, u8) {
    match piece {
        NativeTextPiece::Body { body } => (body.block, 0),
        NativeTextPiece::Separator { before, .. } => (before.block, 1),
        NativeTextPiece::Empty => (0, 0),
    }
}

#[cfg(test)]
#[path = "seek/tests.rs"]
mod tests;
