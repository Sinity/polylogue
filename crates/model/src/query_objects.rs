//! Immutable questions and the exact bindings of their executions.
use crate::bundle::{
    TextCoverage, TextFragmentError, TextSourceVersion, TextSpanOrigin,
    fragments::seek::NativeSpanSeek,
};
use crate::{ChangePosition, Timestamp, authorship::Authorship};
use query_language::{
    ast::Query, catalog::SourceOutput, cursor::CursorContext, identity::QuestionId, time::ClockWire,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Semantic identity excludes display names and the clock of an execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QueryObject {
    pub id: QuestionId,
    pub normal_form: serde_json::Value,
    pub parameters: BTreeMap<String, String>,
    pub identity_ast: Query,
    pub source_outputs: Vec<SourceOutput>,
    pub definitions: Vec<FrozenQueryDefinition>,
    pub dependencies: Vec<QuestionId>,
    pub edges: Vec<QueryEdge>,
}

/// The complete semantic body of a dependency, addressed by its own hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrozenQueryDefinition {
    pub question: QuestionId,
    pub normal_form: serde_json::Value,
    pub parameters: BTreeMap<String, String>,
    pub identity_ast: Query,
    pub source_outputs: Vec<SourceOutput>,
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
    pub id: crate::RunId,
    pub question: QuestionId,
    pub at: Timestamp,
    pub read_at: ChangePosition,
    pub machine: String,
    pub authorship: Authorship,
    pub binding: RunBinding,
}

/// A continuation retains one bounded reference to the original execution.
/// Its own envelope is provenance of this invocation, never old authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RunBinding {
    Original {
        execution: Box<ExecutionBinding>,
    },
    Continuation {
        root: crate::RunId,
        binding_hash: crate::BlobHash,
        position: crate::read::ReadPosition,
        accepted_request: Option<serde_json::Value>,
    },
}
impl RunBinding {
    pub fn original(&self) -> Option<&ExecutionBinding> {
        match self {
            Self::Original { execution } => Some(execution),
            Self::Continuation { .. } => None,
        }
    }
    pub fn original_mut(&mut self) -> Option<&mut ExecutionBinding> {
        match self {
            Self::Original { execution } => Some(execution),
            Self::Continuation { .. } => None,
        }
    }
    /// Stable action identity belongs to the original execution, not the page.
    pub fn binding_hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        match self {
            Self::Original { execution } => execution.hash(),
            Self::Continuation { binding_hash, .. } => Ok(*binding_hash),
        }
    }
}
impl QueryRun {
    pub fn binding_hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        self.binding.binding_hash()
    }
    /// Integrity of this exact invocation, including its position and envelope.
    pub fn record_hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        metadata_hash(b"polylogue-query-run-begin-v1\0", self)
    }
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
        if let ExecutionPresentation::Read {
            options,
            delivery,
            page_budget,
            direct_source,
            ..
        } = &self.presentation
        {
            options.validate()?;
            if matches!(delivery, crate::read::ReadDelivery::Page) != page_budget.is_some() {
                return Err(ExecutionBindingError::PageBudget);
            }
            if let Some(source) = direct_source {
                source.validate()?;
                if self.query_options.direct_reference.as_ref() != Some(&source.origin().reference)
                {
                    return Err(ExecutionBindingError::SpanTarget);
                }
            }
        }
        Ok(())
    }
    /// Stable identity across JSON map insertion orders and feature unions.
    pub fn hash(&self) -> Result<crate::BlobHash, serde_json::Error> {
        metadata_hash(b"polylogue-execution-binding-v1\0", self)
    }
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
