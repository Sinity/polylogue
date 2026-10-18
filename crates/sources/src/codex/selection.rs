//! Codex field descriptors: how to recompute one normalized field from its
//! original rollout records.
use super::origins::RecordRange;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

/// Descriptor JSON layout version.
pub const CODEC_VERSION: u32 = 1;
/// Transform semantics version. Stored descriptors must stay readable, so
/// change this only when a transform's meaning changes.
pub const ADAPTER_RECIPE_VERSION: u32 = 2;

pub use crate::selection::{Binding, Cursor, PathPart, SelectedRaw};
use crate::selection::{JsonWriteError, Operation, Recipe, ScalarError};
use polylogue_model::RawJson as Value;
use polylogue_model::{
    BlockContent, BlockSlot, ToolCall, ToolResult,
    merge::{BlockField, Slot},
};
pub type ValueError<E> = crate::selection::ValueError<E, Error>;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transform {
    /// The selected JSON value.
    JsonValueV1,
    /// The selected JSON string.
    JsonStringV1,
    /// The selected strings, in order, joined with LF.
    TextJoinLfV1,
    /// An RFC 3339 timestamp, to the millisecond.
    ProviderTimeV1,
    /// `input_tokens - cached_input_tokens`, saturating at zero; inputs in that
    /// order.
    UncachedInputV1,
    ToolFieldV1 {
        field: BlockField,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactSelection {
    pub codec_version: u32,
    pub adapter_version: u32,
    pub transform: Transform,
    pub inputs: Vec<SelectedRaw>,
}
#[derive(Debug)]
pub enum Error {
    Invalid(&'static str),
    Encoding(String),
    Io(io::Error),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid Codex field selection: {reason}"),
            Self::Encoding(reason) => write!(f, "Codex field selection encoding failed: {reason}"),
            Self::Io(error) => write!(f, "Codex field selection sink failed: {error}"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}
impl FactSelection {
    pub(super) fn new(transform: Transform, inputs: Vec<SelectedRaw>) -> Self {
        Self {
            codec_version: CODEC_VERSION,
            adapter_version: ADAPTER_RECIPE_VERSION,
            transform,
            inputs,
        }
    }
    pub fn validate(&self) -> Result<(), Error> {
        self.validate_header()?;
        crate::selection::validate_inputs(&self.inputs, self.operation()).map_err(Error::Invalid)
    }
    fn validate_header(&self) -> Result<(), Error> {
        if self.codec_version != CODEC_VERSION || self.adapter_version != ADAPTER_RECIPE_VERSION {
            return Err(Error::Invalid("unsupported version"));
        }
        Ok(())
    }
    pub fn cursor(&self) -> Cursor {
        Cursor::default()
    }
    pub fn advance(&self, cursor: &mut Cursor) -> Result<Option<RecordRange>, Error> {
        self.validate_header()?;
        crate::selection::validate_count(&self.inputs, self.operation()).map_err(Error::Invalid)?;
        cursor.advance(&self.inputs).map_err(Error::Invalid)
    }
    /// Recompute the value; see `selection::write_value`.
    pub fn write_value<E, W: Write>(
        &self,
        resolver: impl FnMut(RecordRange) -> Result<Vec<u8>, E>,
        record_allowance: usize,
        writer: &mut W,
    ) -> Result<(), ValueError<E>> {
        crate::selection::write_value(self, resolver, record_allowance, writer)
    }
    /// Write this descriptor with its input binding. On error the output is
    /// partial and must be discarded.
    pub fn write_bound<W: Write>(&self, binding: Binding<'_>, writer: &mut W) -> Result<(), Error> {
        self.validate()?;
        binding.validate().map_err(Error::Invalid)?;
        crate::selection::write_bound("codex-field-origin", self, binding, writer).map_err(
            |error| match error {
                JsonWriteError::Io(error) => Error::Io(error),
                JsonWriteError::Encoding(error) => Error::Encoding(error.to_string()),
            },
        )
    }
}
impl Recipe for FactSelection {
    type Error = Error;
    fn provider(&self) -> &'static str {
        "codex"
    }
    fn validate(&self) -> Result<(), Error> {
        FactSelection::validate(self)
    }
    fn inputs(&self) -> &[SelectedRaw] {
        &self.inputs
    }
    fn operation(&self) -> Operation {
        match self.transform {
            Transform::JsonValueV1 => Operation::JsonValue,
            Transform::JsonStringV1 => Operation::JsonString,
            Transform::TextJoinLfV1 => Operation::TextJoinLf,
            Transform::ProviderTimeV1 => Operation::ProviderTime,
            Transform::UncachedInputV1 => Operation::UnsignedSaturatingDifference,
            Transform::ToolFieldV1 { .. } => Operation::ProviderScalar,
        }
    }
    fn invalid(&self, reason: &'static str) -> Error {
        Error::Invalid(reason)
    }
    fn write_scalar<W: Write>(
        &self,
        value: &Value,
        sink: &mut W,
    ) -> Result<(), ScalarError<Self::Error>> {
        let Transform::ToolFieldV1 { field } = &self.transform else {
            return Err(ScalarError::Type);
        };
        let content = match field {
            BlockField::Tool | BlockField::CallId | BlockField::Input | BlockField::Command => {
                let input =
                    value
                        .get("arguments")
                        .or_else(|| value.get("input"))
                        .map(|v| match v {
                            Value::String(s) => {
                                super::scalar_json(s.text()).unwrap_or_else(|| v.clone())
                            }
                            _ => v.clone(),
                        });
                let command = value
                    .get("command")
                    .cloned()
                    .or_else(|| {
                        input
                            .as_ref()
                            .and_then(|v| v.get("cmd").or_else(|| v.get("command")))
                            .cloned()
                    })
                    .map(|v| match v {
                        Value::String(s) => s.text().to_owned(),
                        Value::Array(v) => v
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(" "),
                        v => v.to_string(),
                    });
                BlockContent::ToolCall(ToolCall {
                    tool: value.get("name").and_then(Value::as_str).map(str::to_owned),
                    call_id: value
                        .get("call_id")
                        .and_then(Value::as_str)
                        .map(polylogue_model::NativeId::unicode),
                    input,
                    command,
                })
            }
            _ => {
                let output = super::recorded_output(
                    value
                        .get("aggregated_output")
                        .or_else(|| value.get("output")),
                );
                let text = match &output {
                    Some(polylogue_model::ToolOutput::Text(s)) => s.as_str(),
                    _ => "",
                };
                let (outcome, exit) = if value.get("status").is_some() {
                    super::native_outcome(value)
                } else {
                    super::output_outcome(
                        if text.starts_with("Script ") {
                            "exec"
                        } else {
                            "exec_command"
                        },
                        value.get("output").unwrap_or(&Value::Null),
                        text,
                    )
                };
                BlockContent::ToolResult(ToolResult {
                    data: None,
                    call_id: value
                        .get("call_id")
                        .and_then(Value::as_str)
                        .map(polylogue_model::NativeId::unicode),
                    outcome,
                    exit_code: exit,
                    duration_ms: super::native_duration(value),
                    output,
                })
            }
        };
        let slot = BlockSlot::Item(0);
        let selected = Slot::Block {
            block: slot.clone(),
            field: *field,
        };
        let status = polylogue_model::source_leaf::write(
            polylogue_model::source_leaf::Subject::Block {
                slot: &slot,
                content: &content,
            },
            &selected,
            sink,
        )
        .map_err(ScalarError::Leaf)?;
        if status != polylogue_model::source_leaf::PresenceStatus::Written {
            return Err(ScalarError::Type);
        }
        Ok(())
    }
}
