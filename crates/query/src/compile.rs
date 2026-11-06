//! Pure typed query binding and output shape. Store lowers the validated Plan.
use crate::{QueryError, ast::*};
use query_language::time::Clock;
use serde::{Deserialize, Serialize};
mod kept;
pub use kept::unproven_match_policy;
pub(crate) mod saved;
pub(crate) mod schema;
pub(crate) mod source;
/// What compiling needs besides the text.
#[derive(Debug, Clone)]
pub struct Options {
    pub clock: Clock,
    /// Origin names (`claude-code`, `codex`, …): a word whose part before
    /// the first `:` is one of these is a session reference, not a field.
    pub origins: Vec<String>,
    /// Rows per page when the query sets none: 20 sessions, 50 otherwise.
    pub page_size: Option<u64>,
    /// One seed for otherwise unseeded random order across pages.
    pub random_seed: Option<u64>,
    /// Whether vectors exist to rank `similar:` by.
    pub embeddings: bool,
    /// The calling session, for `self`.
    pub self_session: Option<String>,
    /// Saved queries by name: the current revision of each `saved_query`
    /// assertion family with that name (`query:NAME`). Two families with one
    /// name are an ambiguity the query refuses.
    pub saved: std::collections::BTreeMap<String, Vec<SavedQuery>>,
    /// The columns to show instead of the unit's defaults (`--fields`):
    /// field names, `prefix.field` paths included.
    pub fields: Vec<String>,
    /// Kept results by ID (`result:<id>`): what their stored rows hold.
    pub kept: std::collections::BTreeMap<String, Kept>,
    /// Web chats whose conversation page URLs name sessions.
    pub chat_pages: Vec<ChatPage>,
}

/// One canonical saved-query assertion and its retained executable body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SavedQuery {
    pub text: String,
    /// Exact assertion revision supplying this name.
    pub revision: String,
    /// Canonical assertion family identity.
    pub root: String,
    /// The semantic identity declared by the current assertion.
    pub hash: query_language::identity::QuestionId,
    pub denominator: Option<query_language::identity::QuestionId>,
    /// Every current head when the owning family forks, across all names.
    pub conflicts: Vec<polylogue_model::Reference>,
    pub body: SavedQueryBody,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SavedQueryBody {
    Frozen {
        syntax: String,
        #[serde(
            skip_serializing_if = "Option::is_none",
            serialize_with = "serialize_retained_object"
        )]
        object: Option<std::sync::Arc<polylogue_model::query_objects::QueryObject>>,
    },
    Unavailable {
        reason: String,
    },
}
fn serialize_retained_object<S: serde::Serializer>(
    object: &Option<std::sync::Arc<polylogue_model::query_objects::QueryObject>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    object.as_deref().serialize(serializer)
}
impl SavedQuery {
    pub fn executable_text(&self) -> Result<&str, QueryError> {
        match &self.body {
            SavedQueryBody::Frozen { syntax, .. } => Ok(syntax),
            SavedQueryBody::Unavailable { reason } => Err(QueryError::Unavailable {
                what: reason.clone(),
            }),
        }
    }
}

/// A saved query a program ran, by the revision its name resolved to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Definition {
    pub name: String,
    pub revision: String,
}

/// A web chat's conversation pages, `https://<host>/<path>/<id>`, whose
/// conversations are sessions of `origin`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatPage {
    pub origin: String,
    pub hosts: Vec<String>,
    pub path: String,
}

/// A kept result's stored shape: the unit
/// its rows are, their columns, and the columns of its match rows when it
/// stored them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kept {
    #[serde(default)]
    pub output_metadata: polylogue_model::kept::OutputMetadata,
    pub unit: String,
    pub columns: Vec<Column>,
    #[serde(default)]
    pub match_columns: Vec<Column>,
    #[serde(default)]
    pub shape: Option<Shape>,
    #[serde(default)]
    pub native_identity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    /// One row per unit row.
    Rows,
    /// One row per group: the keys, then the aggregates.
    Groups,
    /// One summary row of aggregates.
    Summary,
    /// One row per match of an `exists` or `seq` (`| matches`): the grain
    /// is the (parent row, matching row) pair or the matched step tuple.
    Matches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ColKind {
    Text,
    /// Arrays of IDs stored as text, decoded at the result boundary.
    Ids,
    /// A structured value stored as text, decoded at the result boundary.
    Json,
    Number,
    /// Dollars.
    Money,
    /// Unix ms.
    Time,
    /// Milliseconds.
    Duration,
    /// Seconds at the SQL/output boundary; expressions normalize to milliseconds.
    DurationSeconds,
    Bool,
}

/// The entity named by a scalar column, proven by its owning field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ColumnReference {
    Session,
    Document,
    Message,
    Assertion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Column {
    pub name: String,
    pub kind: ColKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<ColumnReference>,
}

fn parse_query(text: &str, opts: &Bindings) -> Result<Query, QueryError> {
    crate::parse::with_catalog(text, &source::catalog(opts))
}
fn not_built(what: &str) -> QueryError {
    QueryError::NotBuilt { what: what.into() }
}
fn value_text(value: &Value) -> Result<String, QueryError> {
    match value {
        Value::Scalar { text, .. } => Ok(text.clone()),
        Value::Unknown => Ok("unknown".into()),
        Value::Known => Ok("known".into()),
        // Forwarding keeps the symbolic marker until its caller supplies a value.
        Value::Parameter { name } => Ok(format!("${name}")),
        _ => Err(not_built("set in scalar position")),
    }
}
pub use saved::{Parameters, saved_definition};
pub fn definition_parameters(
    text: &str,
    catalog: &query_language::catalog::Catalog,
) -> Result<Vec<String>, QueryError> {
    let mut query = crate::parse::with_catalog(text, catalog)?;
    Parameters::names(&mut query)
}
pub fn columns_schema(unit: &str, columns: &[Column]) -> query_language::catalog::OutputSchema {
    kept::physical_schema(unit, columns)
}
impl Options {
    pub fn definition_parameters(&self, text: &str) -> Result<Vec<String>, QueryError> {
        definition_parameters(text, &self.catalog())
    }
    pub fn catalog(&self) -> query_language::catalog::Catalog {
        source::catalog(&self.bindings())
    }
    pub fn output_schema(
        &self,
        text: &str,
    ) -> Result<query_language::catalog::OutputSchema, QueryError> {
        self.bindings().output_schema(text)
    }
}
pub fn compile(text: &str, options: &Options) -> Result<crate::plan::Plan, QueryError> {
    crate::plan::prepare(text, &options.bindings(), |_| Ok(()))
}

/// Immutable definition metadata, usable without a configured execution clock.
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    pub origins: Vec<String>,
    pub saved: std::collections::BTreeMap<String, Vec<SavedQuery>>,
    pub kept: std::collections::BTreeMap<String, Kept>,
}
impl Bindings {
    /// Add an acknowledged semantic object under a saved assertion's name.
    /// Executable text comes from its frozen AST, never current dependency names.
    pub fn add_saved(
        &mut self,
        name: &str,
        plan: &crate::plan::Plan,
        revision: &str,
        root: &str,
    ) -> Result<(), QueryError> {
        let object = plan.query_object()?;
        let frozen = crate::plan::saved_query_bindings(&object)?;
        let mut definition = frozen[&object.id.to_string()][0].clone();
        definition.revision = revision.into();
        definition.root = root.into();
        for (id, nodes) in frozen {
            self.saved.entry(id).or_insert(nodes);
        }
        self.saved.entry(name.into()).or_default().push(definition);
        Ok(())
    }
    pub fn catalog(&self) -> query_language::catalog::Catalog {
        source::catalog(self)
    }
    pub fn output_schema(
        &self,
        text: &str,
    ) -> Result<query_language::catalog::OutputSchema, QueryError> {
        self.output_metadata(text).map(|(produced, _)| produced)
    }
    /// Produced columns and queryable relation fields from one normalized bind.
    pub fn output_metadata(
        &self,
        text: &str,
    ) -> Result<
        (
            query_language::catalog::OutputSchema,
            query_language::catalog::OutputSchema,
        ),
        QueryError,
    > {
        for kept in self.kept.values() {
            kept.validate_output_metadata()?;
        }
        let (query, _) = source::expand(text, self)?;
        schema::output_metadata(&query, &self.catalog())
    }
    pub fn definition_parameters(&self, text: &str) -> Result<Vec<String>, QueryError> {
        definition_parameters(text, &self.catalog())
    }
}
pub trait Definitions {
    fn saved_queries(&self) -> &std::collections::BTreeMap<String, Vec<SavedQuery>>;
}
impl Definitions for Options {
    fn saved_queries(&self) -> &std::collections::BTreeMap<String, Vec<SavedQuery>> {
        &self.saved
    }
}
impl Definitions for Bindings {
    fn saved_queries(&self) -> &std::collections::BTreeMap<String, Vec<SavedQuery>> {
        &self.saved
    }
}
impl Options {
    pub fn bindings(&self) -> Bindings {
        Bindings {
            origins: self.origins.clone(),
            saved: self.saved.clone(),
            kept: self.kept.clone(),
        }
    }
}
