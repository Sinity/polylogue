//! Immutable questions and the exact bindings of their executions.

use crate::bundle::{
    TextCoverage, TextFragmentError, TextSourceVersion, TextSpanOrigin,
    fragments::seek::NativeSpanSeek,
};
use crate::{ChangePosition, Timestamp, authorship::Authorship};
use query_language::{
    ast::Query,
    catalog::{OutputSchema, OutputSource},
    cursor::CursorContext,
    identity::QuestionId,
    time::ClockWire,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod evolution;

/// Immutable format-1 input retained by the numbered representation upgrade.
/// This is evidence for original identity and custody, never an operation input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HistoricalQuerySource {
    pub format: QuerySourceFormat,
    pub record: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum QuerySourceFormat {
    #[serde(rename = "structured_query_v1")]
    StructuredQueryV1,
    #[serde(rename = "retired_read_v2")]
    RetiredReadV2,
}

/// Semantic source declaration captured with a question. Missing proof never
/// authenticates a retained text or JSON matching policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SourceOutput {
    pub source: OutputSource,
    pub schema: OutputSchema,
    /// Kept-result declaration proof: None lacks original evidence, Some([])
    /// captures every field, Some(names) retains the named uncertainties.
    /// Saved sources establish their declarations through the frozen body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unproven_fields: Option<Vec<String>>,
}
impl SourceOutput {
    pub fn catalog_output(&self) -> query_language::catalog::SourceOutput {
        query_language::catalog::SourceOutput {
            source: self.source.clone(),
            schema: self.schema.clone(),
        }
    }
}

/// Semantic identity excludes display names and the clock of an execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryObject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub historical_source: Option<Box<HistoricalQuerySource>>,
    pub id: QuestionId,
    pub normal_form: serde_json::Value,
    pub parameters: BTreeMap<String, String>,
    pub body: QuestionBody,
    pub source_outputs: Vec<SourceOutput>,
    pub definitions: Vec<FrozenQueryDefinition>,
    pub dependencies: Vec<QuestionId>,
    pub edges: Vec<QueryEdge>,
}

/// The complete semantic body of a dependency, addressed by its own hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrozenQueryDefinition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub historical_source: Option<Box<HistoricalQuerySource>>,
    pub question: QuestionId,
    pub normal_form: serde_json::Value,
    pub parameters: BTreeMap<String, String>,
    pub body: QuestionBody,
    pub source_outputs: Vec<SourceOutput>,
}

/// A retained definition has either executable current syntax or a proved
/// historical reason why it cannot execute. Original syntax stays in its source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QuestionBody {
    Executable { query: Query },
    Unavailable { reason: QueryUnavailable },
}
impl QuestionBody {
    pub fn query(&self) -> Result<&Query, QueryUnavailable> {
        match self {
            Self::Executable { query } => Ok(query),
            Self::Unavailable { reason } => Err(reason.clone()),
        }
    }
    pub fn query_mut(&mut self) -> Result<&mut Query, QueryUnavailable> {
        match self {
            Self::Executable { query } => Ok(query),
            Self::Unavailable { reason } => Err(reason.clone()),
        }
    }
    pub fn into_query(self) -> Result<Query, QueryUnavailable> {
        match self {
            Self::Executable { query } => Ok(query),
            Self::Unavailable { reason } => Err(reason),
        }
    }
}
#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema, thiserror::Error,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryUnavailable {
    #[error(
        "the retained definition uses a computed time bound that its original Polylogue backend did not execute"
    )]
    ComputedTimeBound,
    #[error(
        "the retained definition tests whether a computed value exists, which its original Polylogue backend did not execute"
    )]
    ComputedExistence,
    #[error(
        "the retained definition uses span overlap, which its original Polylogue backend did not execute"
    )]
    SpanOverlap,
    #[error(
        "the retained definition uses a temporal stage that its original Polylogue backend did not execute"
    )]
    TemporalStage,
    #[error(
        "the retained definition uses the original {source_kind} source, which Polylogue did not execute"
    )]
    NativeSource { source_kind: HistoricalNativeSource },
    #[error("the retained definition depends on unavailable query:{question}")]
    Dependency { question: QuestionId },
    #[error(
        "the retained definition projects the retired private storage key of {unit}; current queries use stable object identities"
    )]
    RetiredStorageKey { unit: String },
    #[error(
        "the retained definition projects matches of {unit}, which its original Polylogue backend did not execute"
    )]
    MatchUnit { unit: String },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalNativeSource {
    Walk,
    Recall,
    RecallLast,
    RecallMark,
}
impl std::fmt::Display for HistoricalNativeSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Walk => "walk",
            Self::Recall => "recall",
            Self::RecallLast => "recall last",
            Self::RecallMark => "recall mark",
        })
    }
}
impl QueryObject {
    pub fn executable_ast(&self) -> Result<&Query, QueryUnavailable> {
        self.body.query()
    }
}
impl FrozenQueryDefinition {
    pub fn executable_ast(&self) -> Result<&Query, QueryUnavailable> {
        self.body.query()
    }
}

/// A saved definition as it was bound, including its actual syntax.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrozenDefinition {
    pub name: String,
    pub root: String,
    pub revision: String,
    pub question: QuestionId,
    pub ast: Query,
}

/// Execution choices that remain after semantic references have been bound.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryExecutionOptions {
    /// Actual accepted direct presentation, independent of semantic query identity.
    pub direct_reference: Option<crate::Reference>,
    pub fields: Vec<String>,
    pub self_session: Option<crate::SessionId>,
    pub embeddings: bool,
    pub why: bool,
    /// The actual provider, model and vector basis required by this execution.
    pub ranking_identity: Option<String>,
}

/// Rebinding uses these captured inputs, never current names or the wall clock.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReplayBinding {
    pub bound_ast: Query,
    pub source_outputs: Vec<SourceOutput>,
    pub definitions: Vec<FrozenDefinition>,
    pub context: CursorContext<ClockWire>,
    pub scope_fingerprint: Option<String>,
    pub original_text: String,
}

impl PartialEq for ReplayBinding {
    fn eq(&self, other: &Self) -> bool {
        self.bound_ast == other.bound_ast
            && self.source_outputs == other.source_outputs
            && self.definitions == other.definitions
            && self.context.question == other.context.question
            && self.context.clock == other.context.clock
            && self.context.resolved_times == other.context.resolved_times
            && self.context.random_seed == other.context.random_seed
            && self.context.read_positions == other.context.read_positions
            && self.context.versions == other.context.versions
            && self.scope_fingerprint == other.scope_fingerprint
            && self.original_text == other.original_text
    }
}

/// Complete accepted inputs retained before an execution exposes continuation calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBinding {
    pub replay: ReplayBinding,
    pub query_options: QueryExecutionOptions,
    pub presentation: ExecutionPresentation,
    /// Original trusted envelope without credentials; provenance grants no authority.
    /// Internal calls may have none. Replay uses current operation validation.
    pub accepted_request: Option<serde_json::Value>,
}

/// Selected source facts are read under the authenticated snapshot, not copied here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionPresentation {
    Query,
    Read {
        options: crate::read::ReadOptions,
        delivery: crate::read::ReadDelivery,
        format: crate::read::Format,
        /// Logical limits before reserving this response's envelope bytes.
        page_budget: Option<crate::read::PageBudget>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        direct_source: Option<Box<ReadSpanBinding>>,
    },
}

/// An exact object read has no semantic query or invented question identity.
/// Serialized fields record inputs; only a current pinned Reader derives them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObjectExecutionBinding {
    pub reference: crate::Reference,
    pub content_hash: crate::BlobHash,
    pub scope_fingerprint: Option<String>,
    pub versions: BTreeMap<String, String>,
    pub presentation: ExecutionPresentation,
    pub accepted_request: Option<serde_json::Value>,
}
impl ObjectExecutionBinding {
    pub fn validate(&self) -> Result<(), ExecutionBindingError> {
        let target = match &self.reference {
            crate::Reference::Request(id) => !id.is_empty(),
            crate::Reference::Run(_) => matches!(
                &self.presentation,
                ExecutionPresentation::Read { options, .. } if options.view == crate::read::View::Raw
            ),
            _ => false,
        };
        if !target {
            return Err(ExecutionBindingError::ObjectTarget);
        }
        let ExecutionPresentation::Read {
            direct_source: None,
            ..
        } = &self.presentation
        else {
            return Err(ExecutionBindingError::ObjectTarget);
        };
        validate_presentation(&self.presentation, Some(&self.reference))
    }
    pub fn hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        metadata_hash(b"polylogue-object-execution-binding-v1\0", self)
    }
}

/// One actual selected span source, never a native/fragment fallback on replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadSpanBinding {
    Native {
        span: NormalizedReadSpan,
    },
    Fragments {
        origin: TextSpanOrigin,
        version: TextSourceVersion,
        /// Digest of the actual indexed stored-grain metadata, not permissions.
        membership: crate::BlobHash,
    },
}
impl ReadSpanBinding {
    pub fn origin(&self) -> &TextSpanOrigin {
        match self {
            Self::Native { span } => &span.origin,
            Self::Fragments { origin, .. } => origin,
        }
    }
    pub fn version(&self) -> &TextSourceVersion {
        match self {
            Self::Native { span } => &span.version,
            Self::Fragments { version, .. } => version,
        }
    }
    pub fn validate(&self) -> Result<(), TextFragmentError> {
        match self {
            Self::Native { span } => span.validate(),
            Self::Fragments {
                origin, version, ..
            } => version.validate_container(span_container(origin)?),
        }
    }
}

/// Resolver-produced coordinates kept in acknowledged Run custody. Source
/// identity is retained metadata, never a claim that all source text was read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NormalizedReadSpan {
    pub origin: TextSpanOrigin,
    pub version: TextSourceVersion,
    pub seek: NativeSpanSeek,
}
impl NormalizedReadSpan {
    pub fn new(
        origin: TextSpanOrigin,
        version: TextSourceVersion,
        seek: NativeSpanSeek,
    ) -> Result<Self, TextFragmentError> {
        let value = Self {
            origin,
            version,
            seek,
        };
        value.validate()?;
        Ok(value)
    }
    /// Global UTF-8 bounds only. The unavailable remainder stays unknown.
    pub fn coverage(&self) -> TextCoverage {
        TextCoverage {
            start: self.origin.start,
            end: self.origin.end,
            total_bytes: None,
        }
    }
    /// Metadata checks only. The authenticated Reader validates current grant,
    /// source version/content and the indexed field/UTF-8 boundary mapping.
    pub fn validate(&self) -> Result<(), TextFragmentError> {
        let container = span_container(&self.origin)?;
        self.version.validate_container(container)?;
        self.seek.validate(container, &self.coverage())
    }
}

fn span_container(origin: &TextSpanOrigin) -> Result<&crate::Reference, TextFragmentError> {
    let crate::Reference::Span {
        container,
        unit,
        start,
        end,
    } = &origin.reference
    else {
        return Err(TextFragmentError::Origin);
    };
    if crate::Reference::parse(&origin.reference.to_string())
        .ok()
        .as_ref()
        != Some(&origin.reference)
        || origin.start > origin.end
        || matches!(unit, crate::SpanUnit::Bytes) && (*start != origin.start || *end != origin.end)
    {
        return Err(TextFragmentError::Origin);
    }
    Ok(container)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum RunOutcome {
    Ok,
    Error { kind: String },
}

/// One accepted execution. Authorship is computed at the daemon boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryRun {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub historical_source: Option<Box<HistoricalQuerySource>>,
    pub id: crate::RunId,
    pub question: Option<QuestionId>,
    pub at: Timestamp,
    pub read_at: ChangePosition,
    pub machine: String,
    pub authorship: Authorship,
    pub binding: RunBinding,
}

/// The inputs an execution ran with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RunBinding {
    /// An invocation whose original query has no current executable representation.
    Unavailable {
        reason: QueryUnavailable,
    },
    /// A source-versioned invocation of the retired content-position reader.
    /// It remains inspectable evidence and grants no current replay authority.
    RetiredRead {
        invocation: RetiredReadInvocation,
    },
    OriginalObject {
        execution: Box<ObjectExecutionBinding>,
    },
    Original {
        execution: Box<ExecutionBinding>,
    },
}
/// Actual invocation shape retained from the retired source, never a current cursor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RetiredReadInvocation {
    Original {
        binding_hash: crate::BlobHash,
    },
    Continuation {
        root: crate::RunId,
        binding_hash: crate::BlobHash,
        seek: serde_json::Value,
    },
}
impl RunBinding {
    pub fn original(&self) -> Option<&ExecutionBinding> {
        match self {
            Self::Original { execution } => Some(execution),
            Self::OriginalObject { .. } | Self::RetiredRead { .. } | Self::Unavailable { .. } => {
                None
            }
        }
    }
    pub fn original_mut(&mut self) -> Option<&mut ExecutionBinding> {
        match self {
            Self::Original { execution } => Some(execution),
            Self::OriginalObject { .. } | Self::RetiredRead { .. } | Self::Unavailable { .. } => {
                None
            }
        }
    }
    /// Stable action identity belongs to the original execution, not the page.
    pub fn binding_hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        match self {
            Self::Original { execution } => execution.hash(),
            Self::OriginalObject { execution } => execution.hash(),
            Self::Unavailable { .. } => Err(evolution_error(
                "unavailable invocations require original binding custody",
            )),
            Self::RetiredRead { invocation } => Ok(match invocation {
                RetiredReadInvocation::Original { binding_hash }
                | RetiredReadInvocation::Continuation { binding_hash, .. } => *binding_hash,
            }),
        }
    }
}
impl QueryRun {
    /// Preserve an invocation whose content-position reader was explicitly retired.
    pub fn from_retired_read_v2(source: serde_json::Value) -> Result<Self, String> {
        let current = evolution::retired_read_run(&source)?;
        let mut run: Self = serde_json::from_value(current).map_err(|error| error.to_string())?;
        run.historical_source = Some(Box::new(HistoricalQuerySource {
            format: QuerySourceFormat::RetiredReadV2,
            record: source,
        }));
        run.validate_evolution()?;
        Ok(run)
    }
    /// The numbered format-1 upgrade retains the exact JSON used by hash custody.
    pub fn from_structured_v1(source: serde_json::Value, catalog: &str) -> Result<Self, String> {
        let current = evolution::query_run(&source, catalog)?;
        let mut run: Self = serde_json::from_value(current).map_err(|error| error.to_string())?;
        run.historical_source = Some(Box::new(HistoricalQuerySource {
            format: QuerySourceFormat::StructuredQueryV1,
            record: source,
        }));
        run.validate_evolution()?;
        Ok(run)
    }
    /// Check the executable representation before consulting original evidence.
    pub fn validate_evolution(&self) -> Result<(), String> {
        let catalog = self
            .binding
            .original()
            .and_then(|execution| execution.replay.context.versions.get("catalog"))
            .map(String::as_str)
            .unwrap_or("");
        if matches!(self.binding, RunBinding::Unavailable { .. })
            && !matches!(
                self.historical_source.as_deref(),
                Some(HistoricalQuerySource {
                    format: QuerySourceFormat::StructuredQueryV1,
                    ..
                })
            )
        {
            return Err(
                "unavailable invocations require their exact structured source evidence".into(),
            );
        }
        if matches!(self.binding, RunBinding::RetiredRead { .. })
            && !matches!(
                self.historical_source.as_deref(),
                Some(HistoricalQuerySource {
                    format: QuerySourceFormat::RetiredReadV2,
                    ..
                })
            )
        {
            return Err(
                "retired read invocations require their exact versioned source evidence".into(),
            );
        }
        validate_evolution(self, self.historical_source.as_deref(), |source| match self
            .historical_source
            .as_deref()
            .map(|source| source.format)
        {
            Some(QuerySourceFormat::RetiredReadV2) => evolution::retired_read_run(source),
            _ => evolution::query_run(source, catalog),
        })
    }
    pub fn binding_hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        if let Some(source) = &self.historical_source {
            self.validate_evolution().map_err(evolution_error)?;
            let binding = &source.record["binding"];
            if binding["kind"] == "continuation" {
                return serde_json::from_value(binding["binding_hash"].clone());
            }
            let domain: &[u8] = match binding["kind"].as_str() {
                Some("original") => b"polylogue-execution-binding-v1\0",
                Some("original_object") => b"polylogue-object-execution-binding-v1\0",
                _ => return Err(evolution_error("invalid historical binding kind")),
            };
            return metadata_hash(domain, &binding["execution"]);
        }
        self.binding.binding_hash()
    }
    /// Integrity of this exact invocation, including its position and envelope.
    pub fn record_hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        if let Some(source) = &self.historical_source {
            self.validate_evolution().map_err(evolution_error)?;
            metadata_hash(b"polylogue-query-run-begin-v1\0", &source.record)
        } else {
            metadata_hash(b"polylogue-query-run-begin-v1\0", self)
        }
    }
}

impl QueryObject {
    /// Upgrade an explicit immutable structured format-1 input.
    pub fn from_structured_v1(source: serde_json::Value) -> Result<Self, String> {
        let current = evolution::query_object(&source)?;
        let mut object: Self =
            serde_json::from_value(current).map_err(|error| error.to_string())?;
        for (node, old) in object.definitions.iter_mut().zip(
            source["definitions"]
                .as_array()
                .ok_or("historical definitions are not an array")?,
        ) {
            node.historical_source = Some(Box::new(HistoricalQuerySource {
                format: QuerySourceFormat::StructuredQueryV1,
                record: old.clone(),
            }));
        }
        object.historical_source = Some(Box::new(HistoricalQuerySource {
            format: QuerySourceFormat::StructuredQueryV1,
            record: source,
        }));
        object.validate_evolution()?;
        Ok(object)
    }
    /// A migrated object must be exactly the named conversion of its source.
    pub fn validate_evolution(&self) -> Result<(), String> {
        if matches!(self.body, QuestionBody::Unavailable { .. }) && self.historical_source.is_none()
        {
            return Err(
                "unavailable questions require their exact historical source evidence".into(),
            );
        }
        if self.definitions.iter().any(|node| {
            matches!(node.body, QuestionBody::Unavailable { .. })
                && node.historical_source.is_none()
        }) {
            return Err(
                "unavailable dependencies require their exact historical source evidence".into(),
            );
        }
        if self
            .historical_source
            .as_ref()
            .is_some_and(|source| source.format != QuerySourceFormat::StructuredQueryV1)
            || self.definitions.iter().any(|node| {
                node.historical_source
                    .as_ref()
                    .is_some_and(|source| source.format != QuerySourceFormat::StructuredQueryV1)
            })
        {
            return Err("query objects require structured-query source evidence".into());
        }
        let closure: Vec<_> = self
            .definitions
            .iter()
            .filter_map(|node| {
                node.historical_source
                    .as_ref()
                    .map(|source| source.record.clone())
            })
            .collect();
        for node in &self.definitions {
            if let Some(source) = &node.historical_source {
                validate_evolution(node, Some(source), |old| {
                    evolution::frozen_definition(old, &closure)
                })?;
            }
        }
        let mut executable = self.clone();
        for node in &mut executable.definitions {
            node.historical_source = None;
        }
        validate_evolution(
            &executable,
            self.historical_source.as_deref(),
            evolution::query_object,
        )
    }
}
fn evolution_error(message: impl std::fmt::Display) -> serde_json::Error {
    <serde_json::Error as serde::ser::Error>::custom(message)
}
fn validate_evolution(
    current: &impl Serialize,
    source: Option<&HistoricalQuerySource>,
    convert: impl FnOnce(&serde_json::Value) -> Result<serde_json::Value, String>,
) -> Result<(), String> {
    let Some(source) = source else { return Ok(()) };
    if source.record.get("historical_source").is_some() {
        return Err("historical evidence cannot contain nested evidence".into());
    }
    let mut executable = serde_json::to_value(current).map_err(|error| error.to_string())?;
    executable
        .as_object_mut()
        .ok_or("expected query record")?
        .remove("historical_source");
    if convert(&source.record)? != executable {
        return Err("query representation differs from its numbered source conversion".into());
    }
    Ok(())
}

/// Completion never replaces the immutable execution inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryRunOutcome {
    pub run: crate::RunId,
    pub binding_hash: crate::BlobHash,
    pub run_hash: crate::BlobHash,
    pub at: Timestamp,
    pub outcome: RunOutcome,
    pub count: Option<u64>,
    pub exact: bool,
    /// Observed page result; this does not certify receipt by the client.
    pub next: Option<crate::read::ReadyCall>,
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionBindingError {
    #[error("exact object execution requires a request read")]
    ObjectTarget,
    #[error(transparent)]
    Read(#[from] crate::read::ReadError),
    #[error(transparent)]
    Span(#[from] TextFragmentError),
    #[error("paged reading requires its resolved logical budget; streaming has none")]
    PageBudget,
    #[error("span-source metadata differs from the accepted direct presentation")]
    SpanTarget,
}
impl ExecutionBinding {
    /// Shape only. Raw serialized bindings never establish custody or authority.
    pub fn validate_presentation(&self) -> Result<(), ExecutionBindingError> {
        validate_presentation(
            &self.presentation,
            self.query_options.direct_reference.as_ref(),
        )
    }
    /// Stable identity across JSON map insertion orders and feature unions.
    pub fn hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        metadata_hash(b"polylogue-execution-binding-v1\0", self)
    }
}
fn validate_presentation(
    presentation: &ExecutionPresentation,
    reference: Option<&crate::Reference>,
) -> Result<(), ExecutionBindingError> {
    if let ExecutionPresentation::Read {
        options,
        delivery,
        page_budget,
        direct_source,
        ..
    } = presentation
    {
        options.validate()?;
        if matches!(delivery, crate::read::ReadDelivery::Page) != page_budget.is_some() {
            return Err(ExecutionBindingError::PageBudget);
        }
        if let Some(source) = direct_source {
            source.validate()?;
            if reference != Some(&source.origin().reference) {
                return Err(ExecutionBindingError::SpanTarget);
            }
        }
    }
    Ok(())
}
fn metadata_hash(
    domain: &[u8],
    value: &impl Serialize,
) -> Result<crate::BlobHash, serde_json::Error> {
    fn ordered(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(fields) => {
                let fields: BTreeMap<_, _> = fields.into_iter().collect();
                serde_json::Value::Object(
                    fields
                        .into_iter()
                        .map(|(key, value)| (key, ordered(value)))
                        .collect(),
                )
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.into_iter().map(ordered).collect())
            }
            value => value,
        }
    }
    let mut bytes = domain.to_vec();
    bytes.extend(serde_json::to_vec(&ordered(serde_json::to_value(value)?))?);
    Ok(crate::BlobHash::of(&bytes))
}

/// Composition is recorded in semantic question coordinates, not mutable names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "how", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueryEdge {
    Saved { query: QuestionId },
    Result { result: String },
    Combine { query: QuestionId },
    In { query: QuestionId },
}

/// Queued means durable producer custody; it makes no canonical admission claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QueryCustody {
    Canonical,
    Queued,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryLookup<T> {
    pub value: T,
    pub custody: QueryCustody,
}

#[cfg(test)]
mod tests;
