//! Claude-owned recipes; decoding/path/admission/streaming belongs to the one
//! shared selector evaluator. Original donors never name the effective overlay.
pub use super::background::Leaf as BackgroundLeaf;
pub use super::tool_result::StatusKind;
use super::{
    keyed::Contribution,
    origins::{self, RecordOrigins},
    projection::{Fact, Field},
};
use crate::selection::{self, Operation, Recipe, RecordRange, ScalarError, SelectedRaw};
use polylogue_model::RetainedJson as Value;
use polylogue_model::{
    MessageId, PromptKind, Role, ToolOutput, WrittenBy,
    merge::{BlockField, MessageField},
};
use serde::{Deserialize, Serialize};
use std::io::Write;
pub const CODEC_VERSION: u32 = 1;
pub const ADAPTER_RECIPE_VERSION: u32 = 2;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transform {
    JsonValueV1,
    JsonStringV1,
    NativeIdV2,
    ProviderTimeV1,
    ToolStatusV1 {
        kind: StatusKind,
        reason: bool,
    },
    ToolOutputV1,
    UsdNanosV1,
    BackgroundV1 {
        leaf: BackgroundLeaf,
    },
    ParentReferenceV1 {
        native_uuid: polylogue_model::NativeId,
        resolved: MessageId,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeafSelection {
    pub codec_version: u32,
    pub adapter_version: u32,
    pub transform: Transform,
    pub inputs: Vec<SelectedRaw>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorContext {
    pub role: Role,
    pub first_subagent_instruction: bool,
    pub copied: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorLeaf {
    WrittenBy,
    PromptKind,
    FromName,
}
/// First original record, not a reconstructed record. Author-affecting UUID
/// corrections remain FirstAuthorUpdate obligations until their owner resolves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorSelection {
    pub codec_version: u32,
    pub adapter_version: u32,
    pub input: SelectedRaw,
    pub context: AuthorContext,
    pub leaf: AuthorLeaf,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid Claude field selection: {}", self.0)
    }
}
impl std::error::Error for Error {}
impl LeafSelection {
    fn new(transform: Transform, input: SelectedRaw) -> Self {
        Self {
            codec_version: CODEC_VERSION,
            adapter_version: ADAPTER_RECIPE_VERSION,
            transform,
            inputs: vec![input],
        }
    }
    pub fn write_value<E, W: Write>(
        &self,
        resolver: impl FnMut(RecordRange) -> Result<Vec<u8>, E>,
        record_allowance: usize,
        sink: &mut W,
    ) -> Result<(), selection::ValueError<E, Error>> {
        selection::write_value(self, resolver, record_allowance, sink)
    }
}
impl AuthorSelection {
    pub(super) fn new(record: RecordRange, context: AuthorContext) -> Self {
        Self {
            codec_version: CODEC_VERSION,
            adapter_version: ADAPTER_RECIPE_VERSION,
            input: SelectedRaw {
                record,
                path: Vec::new(),
            },
            context,
            leaf: AuthorLeaf::WrittenBy,
        }
    }
    pub fn for_leaf(&self, leaf: AuthorLeaf) -> Self {
        let mut selection = self.clone();
        selection.leaf = leaf;
        selection
    }
    pub fn write_value<E, W: Write>(
        &self,
        resolver: impl FnMut(RecordRange) -> Result<Vec<u8>, E>,
        record_allowance: usize,
        sink: &mut W,
    ) -> Result<(), selection::ValueError<E, Error>> {
        selection::write_value(self, resolver, record_allowance, sink)
    }
}
fn version(codec: u32, adapter: u32) -> Result<(), Error> {
    if codec != CODEC_VERSION || adapter != ADAPTER_RECIPE_VERSION {
        Err(Error("unsupported version"))
    } else {
        Ok(())
    }
}
impl Recipe for LeafSelection {
    type Error = Error;
    fn provider(&self) -> &'static str {
        "claude-code"
    }
    fn validate(&self) -> Result<(), Error> {
        version(self.codec_version, self.adapter_version)?;
        selection::validate_inputs(&self.inputs, self.operation()).map_err(Error)
    }
    fn inputs(&self) -> &[SelectedRaw] {
        &self.inputs
    }
    fn operation(&self) -> Operation {
        match &self.transform {
            Transform::JsonValueV1 => Operation::JsonValue,
            Transform::JsonStringV1 => Operation::JsonString,
            Transform::NativeIdV2 => Operation::NativeId,
            Transform::ProviderTimeV1 => Operation::ProviderTime,
            Transform::ToolStatusV1 { .. }
            | Transform::ToolOutputV1
            | Transform::UsdNanosV1
            | Transform::BackgroundV1 { .. }
            | Transform::ParentReferenceV1 { .. } => Operation::ProviderScalar,
        }
    }
    fn invalid(&self, reason: &'static str) -> Error {
        Error(reason)
    }
    fn provider_time(&self, text: &str) -> Option<polylogue_model::ProviderTime> {
        super::parse_time_text(text)
    }
    fn write_scalar<W: Write>(
        &self,
        value: &Value,
        sink: &mut W,
    ) -> Result<(), ScalarError<Error>> {
        match &self.transform {
            Transform::ToolStatusV1 { kind, reason } => {
                let outcome = super::tool_result::status(*kind, value).ok_or(ScalarError::Type)?;
                if *reason {
                    let polylogue_model::ToolOutcome::Error {
                        reason: Some(reason),
                    } = outcome
                    else {
                        return Err(ScalarError::Invalid(Error(
                            "status has no supplied error reason",
                        )));
                    };
                    serde_json::to_writer(sink, &reason.code()).map_err(ScalarError::Json)
                } else {
                    serde_json::to_writer(sink, outcome.as_str()).map_err(ScalarError::Json)
                }
            }
            Transform::ParentReferenceV1 {
                native_uuid,
                resolved,
            } => {
                if value.native() != Some(native_uuid) {
                    return Err(ScalarError::Invalid(Error(
                        "parent UUID differs from its proved relation",
                    )));
                }
                polylogue_model::source_leaf::write_message_reference(resolved, sink)
                    .map_err(ScalarError::Leaf)?;
                Ok(())
            }
            Transform::BackgroundV1 { leaf } => {
                let completion = super::background::parse(value.as_str().ok_or(ScalarError::Type)?)
                    .ok_or(ScalarError::Type)?;
                match leaf {
                    BackgroundLeaf::Outcome => {
                        serde_json::to_writer(sink, completion.outcome().as_str())
                    }
                    BackgroundLeaf::ExitCode => {
                        serde_json::to_writer(sink, &completion.exit.ok_or(ScalarError::Type)?)
                    }
                    BackgroundLeaf::Status => serde_json::to_writer(sink, completion.status),
                    BackgroundLeaf::OutputFile => serde_json::to_writer(
                        sink,
                        completion.output_file.ok_or(ScalarError::Type)?,
                    ),
                    BackgroundLeaf::TaskId => serde_json::to_writer(sink, completion.task),
                }
                .map_err(ScalarError::Json)
            }
            Transform::UsdNanosV1 => {
                let nanos = super::usd_nanos(value).ok_or(ScalarError::Type)?;
                serde_json::to_writer(sink, &nanos).map_err(ScalarError::Json)
            }
            Transform::ToolOutputV1 => {
                let text = super::text_of(value).ok_or(ScalarError::Type)?;
                serde_json::to_writer(sink, &ToolOutput::Text(text)).map_err(ScalarError::Json)
            }
            _ => Err(ScalarError::Invalid(Error("unsupported scalar transform"))),
        }
    }
}
impl Recipe for AuthorSelection {
    type Error = Error;
    fn provider(&self) -> &'static str {
        "claude-code"
    }
    fn validate(&self) -> Result<(), Error> {
        version(self.codec_version, self.adapter_version)?;
        if !self.input.path.is_empty() || !matches!(self.context.role, Role::User | Role::Assistant)
        {
            return Err(Error("invalid author root/context"));
        }
        selection::validate_inputs(self.inputs(), self.operation()).map_err(Error)
    }
    fn inputs(&self) -> &[SelectedRaw] {
        std::slice::from_ref(&self.input)
    }
    fn operation(&self) -> Operation {
        Operation::ProviderScalar
    }
    fn invalid(&self, reason: &'static str) -> Error {
        Error(reason)
    }
    fn write_scalar<W: Write>(
        &self,
        value: &Value,
        sink: &mut W,
    ) -> Result<(), ScalarError<Error>> {
        let expected = if self.context.role == Role::Assistant {
            "assistant"
        } else {
            "user"
        };
        if value.get("type").and_then(Value::as_str) != Some(expected) {
            return Err(ScalarError::Type);
        }
        let content = value
            .get("message")
            .and_then(|m| m.get("content"))
            .ok_or(ScalarError::Type)?;
        let (written_by, prompt_kind, from_name) = classify_author(value, content, self.context);
        match self.leaf {
            AuthorLeaf::WrittenBy if written_by != WrittenBy::Unknown => {
                serde_json::to_writer(sink, &written_by).map_err(ScalarError::Json)
            }
            AuthorLeaf::PromptKind => serde_json::to_writer(
                sink,
                &prompt_kind.ok_or(ScalarError::Invalid(Error("prompt kind is unknown")))?,
            )
            .map_err(ScalarError::Json),
            AuthorLeaf::FromName => serde_json::to_writer(
                sink,
                &from_name.ok_or(ScalarError::Invalid(Error("sender name is unknown")))?,
            )
            .map_err(ScalarError::Json),
            _ => Err(ScalarError::Invalid(Error("author is unknown"))),
        }
    }
}
fn transform(fact: &Fact) -> Option<Transform> {
    Some(match &fact.field {
        Field::FirstMessage(MessageField::At)
        | Field::SelectedMessage(MessageField::CompletedAt) => Transform::ProviderTimeV1,
        Field::FirstMessage(MessageField::FromName)
        | Field::SelectedMessage(MessageField::Model | MessageField::StopReason) => {
            Transform::JsonStringV1
        }
        Field::Usage(polylogue_model::merge::UsageField::ProviderCostNusd) => Transform::UsdNanosV1,
        Field::SelectedMessage(MessageField::DurationMs) => Transform::JsonValueV1,
        Field::Usage(_)
        | Field::Extra { .. }
        | Field::SelectedExtra { .. }
        | Field::ResultData { .. } => Transform::JsonValueV1,
        Field::Block {
            field:
                BlockField::Text
                | BlockField::ThinkingText
                | BlockField::ThinkingData
                | BlockField::Tool,
            ..
        } => Transform::JsonStringV1,
        Field::Block {
            field: BlockField::CallId,
            ..
        } => Transform::NativeIdV2,
        Field::Block {
            field: BlockField::Input | BlockField::Encrypted | BlockField::ExitCode,
            ..
        } => Transform::JsonValueV1,
        Field::Block {
            field: BlockField::Outcome | BlockField::OutcomeReason,
            ..
        } => {
            let reason = matches!(
                fact.field,
                Field::Block {
                    field: BlockField::OutcomeReason,
                    ..
                }
            );
            let kind = match fact.source_path.last().map(String::as_str) {
                Some("is_error") => StatusKind::IsError,
                Some("toolDenialKind") => StatusKind::Denied,
                Some("interrupted") => StatusKind::Interrupted,
                Some("timedOutAfterMs") => StatusKind::TimedOut,
                _ => return None,
            };
            Transform::ToolStatusV1 { kind, reason }
        }
        Field::Block {
            field: BlockField::Output,
            ..
        } => Transform::ToolOutputV1,
        // Parent UUID resolution needs the record-address owner, not JSON string
        // equality or parsing an opaque MessageId. Never invent that transform.
        _ => return None,
    })
}
pub(super) fn for_fact(
    fact: &Fact,
    record: &Value,
    origins: &RecordOrigins,
) -> Option<LeafSelection> {
    let path = origins::typed_path(record, &fact.source_path)?;
    let selected = origins.select(&path)?;
    Some(LeafSelection::new(transform(fact)?, selected))
}
pub(super) fn for_selected(
    fact: &Fact,
    index: usize,
    raw: &Value,
    base: &SelectedRaw,
) -> Option<LeafSelection> {
    let prefix = if index >= 3 { 1 } else { 2 };
    let suffix = origins::typed_path(raw, fact.source_path.get(prefix..)?)?;
    let mut selected = base.clone();
    selected.path.extend(suffix);
    Some(LeafSelection::new(transform(fact)?, selected))
}
pub(super) fn signature(record: &Contribution, local: u32) -> Option<LeafSelection> {
    let value: Value = serde_json::from_slice(&record.raw).ok()?;
    let path = vec![
        "message".into(),
        "content".into(),
        local.to_string(),
        "signature".into(),
    ];
    let selected = record
        .origins
        .select(&origins::typed_path(&value, &path)?)?;
    Some(LeafSelection::new(Transform::JsonStringV1, selected))
}

/// The existing source classifier, shared by interpretation and scalar recipe
/// evaluation; context is recorded at first interpretation, never inferred from
/// a replay order, UUID spelling or DTO defaults.
pub(super) fn classify_author(
    v: &Value,
    content: &Value,
    context: AuthorContext,
) -> (WrittenBy, Option<PromptKind>, Option<String>) {
    if context.role == Role::Assistant {
        return (
            if v.get("isApiErrorMessage").and_then(Value::as_bool) == Some(true) {
                WrittenBy::Harness
            } else {
                WrittenBy::Model
            },
            None,
            None,
        );
    }
    if content.as_array().is_some_and(|items| {
        items
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) == Some("tool_result"))
    }) {
        return (WrittenBy::Harness, None, None);
    }
    let origin = v
        .get("origin")
        .and_then(|o| o.get("kind"))
        .and_then(Value::as_str);
    match origin {
        Some("human") => return (WrittenBy::Human, Some(super::prompt_kind_of(content)), None),
        Some("peer") => {
            let name = v
                .get("origin")
                .and_then(|o| o.get("name"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            return (WrittenBy::Model, None, name);
        }
        Some("coordinator") => return (WrittenBy::Model, None, None),
        Some("task-notification" | "auto-continuation") => {
            return (WrittenBy::Harness, None, None);
        }
        _ => {}
    }
    if context.first_subagent_instruction {
        // The instruction the parent model wrote in its Agent or Task call.
        return (WrittenBy::Model, None, None);
    }
    match v.get("promptSource").and_then(Value::as_str) {
        Some("typed" | "suggestion_accepted" | "queued") => {
            return (WrittenBy::Human, Some(super::prompt_kind_of(content)), None);
        }
        Some("sdk" | "system") => return (WrittenBy::Harness, None, None),
        _ => {}
    }
    if v.get("isMeta").and_then(Value::as_bool) == Some(true)
        || v.get("isVisibleInTranscriptOnly").and_then(Value::as_bool) == Some(true)
    {
        return (WrittenBy::Harness, None, None);
    }
    if context.copied {
        // A copy keeps the author it had in its source session; that is resolved at write.
        return (WrittenBy::Unknown, None, None);
    }
    let text = match content {
        Value::String(t) => t.as_str(),
        Value::Array(a) => a
            .iter()
            .find_map(|b| b.get("text").and_then(Value::as_str))
            .unwrap_or(""),
        _ => "",
    };
    let t = text.trim_start();
    // A typed slash command or `!` input: the person typed it (prompt_kind command).
    if t.contains("<command-name>") || t.starts_with("<bash-input>") {
        return (WrittenBy::Human, Some(PromptKind::Command), None);
    }
    // Its expansion and output, and other harness wrappers.
    const HARNESS: [&str; 8] = [
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<bash-stdout>",
        "<bash-stderr>",
        "<command-message>",
        "<system-reminder>",
        "<task-notification>",
        "[Request interrupted by user",
    ];
    if HARNESS.iter().any(|h| t.starts_with(h)) {
        return (WrittenBy::Harness, None, None);
    }
    let has_source = v.get("promptSource").is_some();
    // Versions before the field existed: a remaining prompt is the person's.
    if !has_source
        && v.get("version")
            .and_then(Value::as_str)
            .is_some_and(super::predates_prompt_source)
    {
        return (WrittenBy::Human, Some(super::prompt_kind_of(content)), None);
    }
    // Claude Code gives every submitted prompt a promptId.
    if !has_source
        && v.get("origin").is_none()
        && v.get("promptId").and_then(Value::as_str).is_some()
    {
        return (WrittenBy::Human, Some(super::prompt_kind_of(content)), None);
    }
    (WrittenBy::Unknown, None, None)
}

#[derive(Clone, Copy, Serialize)]
pub struct Binding<'a> {
    pub input: &'a str,
    pub chain: i64,
}
#[derive(Debug)]
pub enum EncodeError {
    Invalid(Error),
    Encoding(serde_json::Error),
    Io(std::io::Error),
}
impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "{e}"),
            Self::Encoding(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for EncodeError {}
fn write_bound<T: Serialize, W: Write>(
    recipe: &T,
    binding: Binding<'_>,
    sink: &mut W,
) -> Result<(), EncodeError> {
    if binding.input.is_empty() || binding.chain <= 0 {
        return Err(EncodeError::Invalid(Error("invalid binding")));
    }
    #[derive(Serialize)]
    struct Bound<'a, T> {
        format: &'static str,
        binding: Binding<'a>,
        selection: &'a T,
    }
    selection::write_json(
        &Bound {
            format: "claude-code-field-origin",
            binding,
            selection: recipe,
        },
        sink,
    )
    .map_err(|e| match e {
        selection::JsonWriteError::Encoding(e) => EncodeError::Encoding(e),
        selection::JsonWriteError::Io(e) => EncodeError::Io(e),
    })
}
impl LeafSelection {
    pub fn write_bound<W: Write>(
        &self,
        binding: Binding<'_>,
        sink: &mut W,
    ) -> Result<(), EncodeError> {
        self.validate().map_err(EncodeError::Invalid)?;
        write_bound(self, binding, sink)
    }
}
impl AuthorSelection {
    pub fn write_bound<W: Write>(
        &self,
        binding: Binding<'_>,
        sink: &mut W,
    ) -> Result<(), EncodeError> {
        self.validate().map_err(EncodeError::Invalid)?;
        write_bound(self, binding, sink)
    }
}

/// The interpretation owner proved this session-scoped record relation. The
/// recipe verifies its original UUID donor; no opaque ID is parsed or guessed.
pub(super) fn parent(
    native_uuid: &polylogue_model::NativeId,
    resolved: MessageId,
    origins: &RecordOrigins,
) -> Option<LeafSelection> {
    let input = origins.select(&[selection::PathPart::Key("parentUuid".into())])?;
    Some(LeafSelection::new(
        Transform::ParentReferenceV1 {
            native_uuid: native_uuid.clone(),
            resolved,
        },
        input,
    ))
}

pub(super) fn background(range: RecordRange, leaf: BackgroundLeaf) -> LeafSelection {
    LeafSelection::new(
        Transform::BackgroundV1 { leaf },
        SelectedRaw {
            record: range,
            path: vec![selection::PathPart::Key("content".into())],
        },
    )
}
