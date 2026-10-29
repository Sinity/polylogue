//! Store/service-prepared evidence. Rendering does not resolve parents, fetch
//! blobs, read grants or choose archive rows.
use super::{
    Annotation, Card, DelegationCard, EditView, IndicatorKind, RawRecord, ReadError, ReadyCall,
    Source, SourceBlock, ValueState, VariantMark,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BranchState {
    Active,
    Abandoned {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        n: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total: Option<u64>,
    },
    Unknown {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        versions: Option<u64>,
    },
}
impl Default for BranchState {
    fn default() -> Self {
        Self::Unknown { versions: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreparedIndicator {
    pub source: Source,
    pub kind: IndicatorKind,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<ReadyCall>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolPair {
    /// Emit once at this selected position; the service resolves a call's row
    /// by ID even when it is outside the current page.
    pub emit_at: String,
    pub call: SourceBlock,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<SourceBlock>,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub duration_ms: ValueState<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_line: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child: Option<DelegationCard>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<EditView>,
}
impl ToolPair {
    pub fn validate(&self) -> Result<(), ReadError> {
        if !matches!(self.call.block, crate::Block::ToolCall { .. })
            || self
                .result
                .as_ref()
                .is_some_and(|result| !matches!(result.block, crate::Block::ToolResult { .. }))
        {
            return Err(ReadError::InvalidEvidence(
                "tool pair needs a call and optional result".into(),
            ));
        }
        if let (
            crate::Block::ToolCall {
                call_id: Some(call),
                ..
            },
            Some(SourceBlock {
                block:
                    crate::Block::ToolResult {
                        call_id: Some(result),
                        ..
                    },
                ..
            }),
        ) = (&self.call.block, &self.result)
            && call != result
        {
            return Err(ReadError::InvalidEvidence(
                "tool pair call IDs disagree".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContextEvidence {
    pub selected: bool,
    pub complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default)]
    pub evicted: Vec<String>,
    #[serde(default)]
    pub not_recorded: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantText {
    pub mark: VariantMark,
    pub text: String,
}

/// Evidence is per record/page, never an entire session required to produce
/// the next item. Variants change layout, not the canonical source record.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectionEvidence {
    #[serde(default)]
    pub inherited: bool,
    #[serde(default)]
    pub branch: BranchState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextEvidence>,
    #[serde(default)]
    pub before: Vec<PreparedIndicator>,
    #[serde(default)]
    pub after: Vec<PreparedIndicator>,
    /// Original ordinal -> prepared relationship. No stream-order pairing.
    #[serde(default)]
    pub tool_pairs: BTreeMap<u64, ToolPair>,
    #[serde(default)]
    pub output_calls: BTreeMap<u64, ReadyCall>,
    #[serde(default)]
    pub assertions: Vec<Annotation>,
    #[serde(default)]
    pub variants: BTreeMap<u64, VariantText>,
    #[serde(default)]
    pub cards: Vec<Card>,
    #[serde(default)]
    pub raw: Vec<RawRecord>,
    /// Source block reference -> metadata aligned with its attachment positions.
    #[serde(default)]
    pub attachments: BTreeMap<String, Vec<AttachmentEvidence>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand_inherited: Option<ReadyCall>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentEvidence {
    #[serde(default)]
    pub supplied_by: ValueState<String>,
    #[serde(default)]
    pub availability: ValueState<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    /// The store has resolved access and supplied original content bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_bytes: Option<Vec<u8>>,
}
