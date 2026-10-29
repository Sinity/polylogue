//! Immutable questions and the exact bindings of their executions.
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
    pub id: String,
    pub question: QuestionId,
    pub at: Timestamp,
    pub read_at: ChangePosition,
    pub machine: String,
    pub authorship: Authorship,
    pub outcome: RunOutcome,
    pub count: Option<u64>,
    pub exact: bool,
    pub replay: ReplayBinding,
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
