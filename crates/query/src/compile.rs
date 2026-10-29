//! Pure typed query binding and output shape. Store lowers the validated Plan.
use crate::{QueryError, ast::*};
use query_language::time::Clock;
use serde::{Deserialize, Serialize};
mod saved;
mod schema;
pub(crate) mod source;
/// What compiling needs besides the text.
#[derive(Debug, Clone)]
pub struct Options {
    pub clock: Clock,
    /// Origin names (`claude-code`, `codex`, …): a word whose part before
    /// the first `:` is one of these is a session reference, not a field.
    pub origins: Vec<String>,
    /// The CLI's command names: a query starting with one, or a near miss
    /// of one, is refused rather than searched for.
    pub command_words: Vec<String>,
    /// Rows per page when the query sets none: 20 sessions, 50 otherwise.
    pub page_size: Option<u64>,
    /// One seed for otherwise unseeded random order across pages.
    pub random_seed: Option<u64>,
    /// Whether vectors exist to rank `similar:` by.
    pub embeddings: bool,
    /// The calling session, for `self`.
    pub self_session: Option<String>,
    /// Saved queries by name: the current revision of each `saved_query`
    /// note family with that name (`query:NAME`). Two families with one
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

impl Options {
    /// Declared parameter names use the same AST visitor as execution binding.
    pub fn definition_parameters(&self, text: &str) -> Result<Vec<String>, QueryError> {
        definition_parameters(text, &self.catalog())
    }
    /// The execution registry plus immutable kept output types for this read.
    pub fn catalog(&self) -> query_language::catalog::Catalog {
        source::catalog(self)
    }

    /// Source field types for completion and binding, without executing SQL.
    /// A relation keeps its native fields; a shaped result exposes its columns.
    pub fn output_schema(
        &self,
        text: &str,
    ) -> Result<query_language::catalog::OutputSchema, QueryError> {
        let (query, _) = source::expand(text, self)?;
        let catalog = self.catalog();
        if query.stages.is_empty() && query.attach.is_empty() {
            if let Source::Unit { unit, .. } = &query.source {
                let declaration = catalog
                    .unit(unit)
                    .ok_or_else(|| not_built("this output unit"))?;
                return Ok(query_language::catalog::OutputSchema {
                    unit: Some(declaration.name.clone()),
                    fields: declaration.fields.clone(),
                    main_time: declaration.main_time.clone(),
                });
            }
            if let Some(schema) = catalog.source_output(&query.source) {
                return Ok(schema.clone());
            }
        }
        let program = compile(text, self)?;
        Ok(source::schema(
            &catalog,
            program.unit.name(),
            &program.columns,
        ))
    }
}

/// Parameter metadata shares execution's AST visitor and never scans DSL text.
pub fn definition_parameters(
    text: &str,
    catalog: &query_language::catalog::Catalog,
) -> Result<Vec<String>, QueryError> {
    let mut query = crate::parse::with_catalog(text, catalog)?;
    saved::Parameters::names(&mut query)
}
/// Output types for immutable kept columns, with no execution clock or I/O.
pub fn columns_schema(
    catalog: &query_language::catalog::Catalog,
    unit: &str,
    columns: &[Column],
) -> query_language::catalog::OutputSchema {
    source::schema(catalog, unit, columns)
}

/// A saved query's definition: the text of one revision of a
/// `saved_query` note.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SavedQuery {
    pub text: String,
    /// The note revision the text is from.
    pub revision: String,
    /// The first note of its revisions: the family's identity.
    pub first_note: String,
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
    pub unit: String,
    pub columns: Vec<Column>,
    #[serde(default)]
    pub match_columns: Vec<Column>,
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
    Bool,
}

/// The entity named by a scalar column, proven by its owning field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ColumnReference {
    Session,
    Document,
    Message,
    Note,
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
fn value_text(value: &Value) -> Result<&str, QueryError> {
    match value {
        Value::Scalar { text, .. } => Ok(text),
        Value::Unknown => Ok("unknown"),
        Value::Parameter { name } => Ok(name),
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
pub fn columns_schema(
    catalog: &query_language::catalog::Catalog,
    unit: &str,
    columns: &[Column],
) -> query_language::catalog::OutputSchema {
    source::schema(catalog, unit, columns)
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
    pub command_words: Vec<String>,
    pub saved: std::collections::BTreeMap<String, Vec<SavedQuery>>,
    pub kept: std::collections::BTreeMap<String, Kept>,
}
impl Bindings {
    pub fn catalog(&self) -> query_language::catalog::Catalog {
        source::catalog(self)
    }
    pub fn output_schema(
        &self,
        text: &str,
    ) -> Result<query_language::catalog::OutputSchema, QueryError> {
        let (query, _) = source::expand(text, self)?;
        schema::infer(&query, &self.catalog())
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
            command_words: self.command_words.clone(),
            saved: self.saved.clone(),
            kept: self.kept.clone(),
        }
    }
}
