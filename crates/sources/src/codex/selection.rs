//! Immutable extraction instructions for one mapper-proved field. These are
//! record selectors, never payload byte extents or content-version evidence.
use super::origins::RecordRange;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

/// Descriptor wire layout, independent of parser checkpoint/cell storage.
pub const CODEC_VERSION: u32 = 1;
/// Source interpretation recipe. Unrelated parser-cell layout changes must not
/// make immutable historical field descriptors unreadable.
pub const ADAPTER_RECIPE_VERSION: u32 = 1;

pub use crate::selection::{Cursor, PathPart, SelectedRaw};
use crate::selection::{JsonWriteError, Operation, Recipe};
pub type ValueError<E> = crate::selection::ValueError<E, Error>;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transform {
    /// Decode as the adapter's serde_json::Value; retain an opaque typed value,
    /// not raw numeric spelling or raw content-version canonical bytes.
    JsonValueV1,
    /// Decode the one selected JSON string.
    JsonStringV1,
    /// Decode selected strings in this exact order and join with LF.
    TextJoinLfV1,
    /// Current adapter: RFC3339 via jiff, milliseconds, Intrinsic/Instant.
    ProviderTimeV1,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactSelection {
    pub codec_version: u32,
    pub adapter_version: u32,
    pub transform: Transform,
    pub inputs: Vec<SelectedRaw>,
}
#[derive(Clone, Copy, Serialize)]
pub struct Binding<'a> {
    pub input: &'a str,
    /// Owning archive revision number (1-based), never a source byte offset.
    pub chain: i64,
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
    /// Uses the shared evaluator without changing this historical Codex recipe.
    pub fn write_value<E, W: Write>(
        &self,
        resolver: impl FnMut(RecordRange) -> Result<Vec<u8>, E>,
        record_allowance: usize,
        writer: &mut W,
    ) -> Result<(), ValueError<E>> {
        crate::selection::write_value(self, resolver, record_allowance, writer)
    }
    /// Stream an already admitted descriptor into the caller's chunk producer.
    /// Binding must come from the pinned source snapshot. The owning writer must
    /// separately validate candidate membership and supported interpretation.
    /// A failed write is not resumable; discard the partial producer or replay.
    pub fn write_bound<W: Write>(&self, binding: Binding<'_>, writer: &mut W) -> Result<(), Error> {
        self.validate()?;
        if binding.input.is_empty() || binding.chain <= 0 {
            return Err(Error::Invalid("invalid binding"));
        }
        #[derive(Serialize)]
        struct Bound<'a> {
            format: &'static str,
            binding: Binding<'a>,
            selection: &'a FactSelection,
        }
        crate::selection::write_json(
            &Bound {
                format: "codex-field-origin",
                binding,
                selection: self,
            },
            writer,
        )
        .map_err(|error| match error {
            JsonWriteError::Io(error) => Error::Io(error),
            JsonWriteError::Encoding(error) => Error::Encoding(error.to_string()),
        })
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
        }
    }
    fn invalid(&self, reason: &'static str) -> Error {
        Error::Invalid(reason)
    }
    fn provider_time(&self, text: &str) -> Option<polylogue_model::ProviderTime> {
        super::parse_timestamp(text).map(super::provider_time)
    }
}
