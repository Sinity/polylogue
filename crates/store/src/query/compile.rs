//! Compiling a parsed query to the statements that read its answer, with
//! bound parameters ([`Statements`]). The answer has a total order; the
//! caller pages it by key and counts it without that order. There is no
//! row cap: a page is how much travels now, never how much is answered.

use crate::query::ast::*;
use crate::query::fields::{self, Aliases, Kind};
use crate::query::time::{self, Milliseconds, Time};
use crate::query::units::{self, Index, Search, Via};
use combine::join;
use polylogue_query::QueryError;
use polylogue_query::compile::Parameters;
pub use polylogue_query::search::SnippetTerms;
use polylogue_query::selection::{Order, Projection, Selection, contains_order, top_level};
pub use polylogue_query::{
    ChatPage, ColKind, Column, ColumnReference, Definition, Kept, Options, SavedQuery, Shape,
};
use serde::Serialize;

mod combine;
mod conditions;
pub(crate) mod custom_fields;
mod expression;
mod program;
mod relations;
mod saved;
mod search;
mod sequences;
mod stages;
mod statements;
use conditions::unknown_field;
use polylogue_query::filter::{alternatives, bind_unit, value_text};
pub use statements::{
    DerivedRows, KeyedRows, PageColumns, RecentSearch, ScopedSearch, Statements, TextSearch,
};

/// Reciprocal rank fusion gives a row `1 / (RANK_FUSION_OFFSET + place)` in
/// each ranked list it is in; the offset keeps one list's first places
/// from outweighing agreement between lists.
pub const RANK_FUSION_OFFSET: u32 = 60;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Param {
    Int(i64),
    Real(f64),
    Text(String),
}

/// A pasted ID-shaped word: it is a reference only if it names exactly one
/// session or message. The caller checks that before running the query.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IdCheck {
    pub word: String,
    /// Returns the session IDs the word may name (at most a few).
    pub sql: String,
    pub params: Vec<Param>,
}

/// Rows attached to each result row (`with messages(id, role)[last:1]`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Attach {
    pub name: String,
    /// The child rows of a page's parents, whose `_pk`s bind as one JSON
    /// array to the parameter after `params`: each row's parent key as
    /// `_parent`, then its columns, in each parent's order and inside the
    /// attachment's windows.
    pub sql: String,
    pub params: Vec<Param>,
    pub columns: Vec<Column>,
}

/// One top-level condition counted alone, for `--why`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WhyClause {
    pub clause: String,
    pub sql: String,
}

/// A `similar:` term to rank by: the text whose similarity to each stored
/// text the caller computes and supplies while the query runs
/// (`pl_similarity`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Near {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowBasis {
    Native { unit: Unit, columns: Vec<String> },
    Group { columns: Vec<String> },
    Summary,
    Distinct { columns: Vec<String> },
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub query: String,
    pub row_basis: RowBasis,
    pub unit: Unit,
    pub shape: Shape,
    /// The statements that select, page, count and read the answer's rows.
    pub statements: Statements,
    pub params: Vec<Param>,
    /// Every output column but `_pk` (each row unit's key).
    pub columns: Vec<Column>,
    /// Physical SQL slots for logical output names. Private identity and order
    /// slots never share the namespace of a user alias.
    pub column_slots: std::collections::BTreeMap<String, String>,
    #[serde(skip)]
    native_sources: Option<Vec<relations::NativeSource>>,
    #[serde(skip)]
    time_text_columns: std::collections::BTreeSet<String>,
    /// Ordered by relevance to text terms.
    pub ranked: bool,
    /// What a ranked row's snippet looks for: the positive top-level prose
    /// terms, all of them required.
    pub snippet_terms: Vec<SnippetTerms>,
    pub id_checks: Vec<IdCheck>,
    /// Required change-feed generations, validated before any result stages.
    pub change_generations: Vec<String>,
    pub assertion_cutoffs: Vec<i64>,
    pub attach: Vec<Attach>,
    pub notes: Vec<String>,
    /// Parts that could not run; the outcome is then `incomplete`.
    pub incomplete: Vec<String>,
    /// `--why`: each top-level condition counted alone, and `why_of`, the
    /// rows the defaults alone keep.
    pub why: Vec<WhyClause>,
    pub why_of: String,
    /// `--why`: each default the unit applied, with the count of rows
    /// without that one default, so an answer names what its defaults hid.
    pub why_defaults: Vec<WhyClause>,
    pub page_size: u64,
    /// `similar:`: the text to embed and rank by.
    pub near: Option<Near>,
    /// The answer rows have their own semantic similarities to count.
    pub counts_near: bool,
    /// The saved queries it ran, each at the revision its name resolved
    /// to when it was compiled.
    pub definitions: Vec<Definition>,
    /// The instant and zone the times were resolved against.
    pub clock: query_language::time::Clock,
    /// Shared expression resolutions carried unchanged by every continuation.
    pub resolved_times: Vec<query_language::time::ResolvedTime>,
    /// `| measure WHAT`: the measurement to run over the sessions `sql`
    /// selects, instead of returning its rows.
    pub measure: Option<Measure>,
    /// The answer's total order, as columns of its page statement; a page
    /// continues after the last row's values of these columns. Empty for a
    /// summary, which is one row.
    pub order: Vec<OrderKey>,
}

/// One term of an answer's order: a column of the answer and its
/// direction. SQLite sorts NULL first in ascending order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OrderKey {
    pub column: String,
    pub descending: bool,
}

impl OrderKey {
    fn ascending(column: &str) -> Self {
        Self {
            column: column.to_owned(),
            descending: false,
        }
    }
}

/// Splits an ORDER BY term into its expression and direction.
fn order_term(term: &str) -> (&str, bool) {
    match term.strip_suffix(" DESC") {
        Some(expr) => (expr, true),
        None => (term.strip_suffix(" ASC").unwrap_or(term), false),
    }
}

/// One hidden column `{prefix}{i}` per ORDER BY term, to select beside the
/// row, and the order keys over them.
fn hidden_order(order: &[String], prefix: &str) -> (Vec<String>, Vec<OrderKey>) {
    order
        .iter()
        .enumerate()
        .map(|(i, term)| {
            let (expr, descending) = order_term(term);
            let column = format!("{prefix}{i}");
            (
                format!("{expr} AS {column}"),
                OrderKey { column, descending },
            )
        })
        .unzip()
}

/// An ORDER BY list over order keys.
fn order_by(keys: &[OrderKey]) -> String {
    order_by_in_direction(keys, false)
}

fn order_by_in_direction(keys: &[OrderKey], reverse: bool) -> String {
    keys.iter()
        .map(|k| {
            let dir = if k.descending ^ reverse { " DESC" } else { "" };
            format!("\"{}\"{dir}", k.column)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measure {
    pub what: String,
    /// The same unit defaults before the query adds its conditions.
    pub population_sql: String,
    /// The selected sessions, before an application measure projects its rows.
    pub selection_sql: String,
    /// The query named no conditions: the measurement covers every session.
    pub everything: bool,
    /// Top-level `since:`/`until:` on the sessions, as Unix ms (`since`
    /// inclusive). A measure over messages counts only the messages inside.
    pub since: Option<i64>,
    pub until: Option<i64>,
}

impl Program {
    pub fn column_slot<'a>(&'a self, name: &'a str) -> &'a str {
        self.column_slots.get(name).map_or(name, String::as_str)
    }

    /// The rows of a stage's or combination's input, in order.
    fn rows(&self) -> &str {
        &self.statements.derived().rows
    }

    fn derived_mut(&mut self) -> &mut DerivedRows {
        match &mut self.statements {
            Statements::Derived(rows) => rows,
            Statements::Keyed(_) => {
                unreachable!("a relation's input is lowered to a derived statement")
            }
        }
    }
}

pub fn compile(text: &str, opts: &Options) -> Result<Program, QueryError> {
    query_language::depth::deeper(|| {
        let plan = polylogue_query::compile(text, opts)?;
        lower(&plan, opts)
    })
}
pub fn lower(plan: &polylogue_query::plan::Plan, opts: &Options) -> Result<Program, QueryError> {
    let mut captured = opts.clone();
    captured.saved = plan.bindings().saved.clone();
    captured.kept = plan.bindings().kept.clone();
    for (id, kept) in &mut captured.kept {
        let actual = opts.kept.get(id).ok_or_else(|| QueryError::Unavailable {
            what: format!("kept result {id:?} is absent from this read"),
        })?;
        actual.validate_output_metadata()?;
        if [ResultPart::Rows, ResultPart::Matches]
            .into_iter()
            .any(|part| kept.output_schema(part) != actual.output_schema(part))
        {
            return Err(QueryError::Unavailable {
                what: format!("kept result {id:?} differs from its bound columns"),
            });
        }
        *kept = actual.clone();
    }
    let opts = &captured;
    let bound = plan.expanded_query();
    let catalog = plan.catalog();
    let rendered = query_language::format::format_query(bound.query(), catalog)
        .map_err(|e| QueryError::language(e, ""))?;
    let mut compiler = Compiler::new(&rendered, opts);
    compiler.query_scope = bound.typing().query(&Default::default()).cloned();
    compiler.typing = Some(std::sync::Arc::new(bound.typing().clone()));
    compiler.catalog = Some(catalog.clone());
    compiler.answer = true;
    let mut program = compiler.program(bound.query())?;
    for definition in plan.definitions() {
        let used = Definition {
            name: definition.name.clone(),
            revision: definition.revision.clone(),
        };
        if !program.definitions.contains(&used) {
            program.definitions.push(used);
        }
    }
    program.query = plan.text().to_owned();
    Ok(program)
}

/// The unit a query's rows currently are, and the aliases they are read
/// through.
#[derive(Debug, Clone)]
pub(crate) struct RowContext {
    unit: Unit,
    a: Aliases,
}

impl RowContext {
    fn plain(unit: Unit) -> Self {
        Self {
            unit,
            a: Aliases::plain(),
        }
    }
}

/// The rows of `| matches`: an SQL body selecting `_pk` and `columns`,
/// ordered by `order` (columns it also selects), and whether it needs
/// `WITH RECURSIVE`.
struct MatchRel {
    sql: String,
    columns: Vec<Column>,
    order: Vec<OrderKey>,
    recursive: bool,
}

pub(crate) struct Compiler<'a> {
    resolved_times: std::cell::RefCell<Vec<query_language::time::ResolvedTime>>,
    text: &'a str,
    opts: &'a Options,
    params: Vec<Param>,
    ctes: Vec<String>,
    input_rows: Option<String>,
    query_scope: Option<query_language::scope::QueryScope>,
    typing: Option<std::sync::Arc<query_language::scope::Typing>>,
    catalog: Option<std::sync::Arc<query_language::catalog::Catalog>>,
    notes: Vec<String>,
    incomplete: Vec<String>,
    snippet_terms: Vec<SnippetTerms>,
    id_checks: Vec<IdCheck>,
    change_generations: Vec<String>,
    assertion_cutoffs: Vec<i64>,
    /// Fields shown beside each row because a condition or sort named them.
    shown: Vec<RowColumn>,
    aliases: usize,
    /// Saved queries being expanded, innermost last: a cycle is refused.
    saved_stack: Vec<String>,
    saved_relations: Vec<saved::BoundRelation>,
    /// The full-text ranking expression, when a top-level text term ranks
    /// the rows.
    rank: Option<String>,
    rank_source: Option<units::HitSource>,
    text_search: Option<TextSearch>,
    /// The semantic ranking expression of a `similar:` term, and the join
    /// that brings its similarities to the unit's rows.
    near_rank: Option<String>,
    near_join: Option<String>,
    /// Rows ranked by reciprocal rank fusion (words and `similar:` together,
    /// or a combining stage): the join bringing each row's fused rank.
    fusion: Option<Fusion>,
    near: Option<Near>,
    definitions: Vec<Definition>,
    /// A semantic ranking requested by the source conditions, built only
    /// when rows must be ordered.
    near_request: Option<(Near, RowContext)>,
    /// The index whose relevance ranks the rows, from the selection's
    /// search leg.
    rank_index: Option<polylogue_query::search::SnippetIndex>,
    source_near_allowed: bool,
    /// An unavailable semantic order affects the answer or an earlier selection.
    near_unavailable: bool,
    /// The query ends in `| measure`: a display preference such as hiding
    /// never changes a measured population.
    measuring: bool,
    /// Top-level `since:`/`until:` bounds on the sessions, for measures.
    session_window: (Option<i64>, Option<i64>),
    /// Lowering the answer itself, whose own rows are read key first;
    /// otherwise the input of a stage or combination, read whole.
    answer: bool,
    /// The last relation slot or step number given out, shared by every
    /// compiler of one query, so no two slots or steps share a name.
    slots: std::rc::Rc<std::cell::Cell<usize>>,
}

/// A fused ranking: the join that brings it to the rows, and the rank
/// column, lower first.
#[derive(Debug, Clone)]
struct Fusion {
    join: String,
    rank: String,
}

impl<'a> Compiler<'a> {
    fn new(text: &'a str, opts: &'a Options) -> Self {
        Self {
            text,
            opts,
            params: vec![],
            ctes: vec![],
            input_rows: None,
            query_scope: None,
            typing: None,
            catalog: None,
            notes: vec![],
            incomplete: vec![],
            snippet_terms: vec![],
            resolved_times: Default::default(),
            id_checks: vec![],
            change_generations: vec![],
            assertion_cutoffs: vec![],
            shown: vec![],
            aliases: 0,
            saved_stack: vec![],
            saved_relations: vec![],
            rank: None,
            rank_source: None,
            text_search: None,
            near_rank: None,
            near_join: None,
            fusion: None,
            near: None,
            definitions: vec![],
            near_request: None,
            rank_index: None,
            source_near_allowed: true,
            near_unavailable: false,
            measuring: false,
            session_window: (None, None),
            answer: false,
            slots: Default::default(),
        }
    }

    /// A compiler for a query inside this one, naming its slots and steps
    /// apart from this one's.
    fn nested<'b>(&self, text: &'b str) -> Compiler<'b>
    where
        'a: 'b,
    {
        let mut nested = Compiler::new(text, self.opts);
        nested.catalog = self.catalog.clone();
        nested.slots = self.slots.clone();
        nested
    }

    /// A fresh relation column name.
    fn slot(&self) -> String {
        format!("_rel_value{}", self.next_number())
    }

    fn next_number(&self) -> usize {
        self.slots.set(self.slots.get() + 1);
        self.slots.get()
    }

    fn bind(&mut self, p: Param) -> String {
        self.params.push(p);
        format!("?{}", self.params.len())
    }

    fn text_param(&mut self, s: &str) -> String {
        self.bind(Param::Text(s.to_owned()))
    }

    fn fresh(&mut self) -> Aliases {
        self.aliases += 1;
        Aliases::suffixed(self.aliases)
    }

    fn program(self, q: &Query) -> Result<Program, QueryError> {
        if matches!(q.source, Source::Result { .. }) {
            self.kept_program(q)
        } else {
            self.program_with(q, None)
        }
    }

    /// The stages of `q` over stored rows, its source ignored.
    fn program_over(mut self, q: &Query, mut program: Program) -> Result<Program, QueryError> {
        self.params = std::mem::take(&mut program.params);
        let stages = polylogue_query::stages::stages(&q.transforms, 0)?;
        self.finish_relations(q, &stages, relations::Relation::new(program), None)
    }
}

/// A column of `| matches` rows, by name.
fn column_sql(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}
fn not_built(what: &str) -> QueryError {
    QueryError::NotBuilt {
        what: what.to_owned(),
    }
}

fn and_all(parts: &[String]) -> String {
    if parts.is_empty() {
        "1".into()
    } else {
        parts.join(" AND ")
    }
}

/// Plain words search this unit.
fn text_unit(unit: Unit) -> bool {
    polylogue_query::units::spec(unit).words.plain()
}

fn like_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn like_prefix(s: &str) -> String {
    format!("{}%", like_escape(s))
}

/// The FROM clause of a unit over the given aliases.
fn base_from(unit: Unit, a: &Aliases) -> String {
    units::spec(unit).from(a)
}

/// The expression naming a row's session key.
fn child_session_column(unit: Unit, a: &Aliases) -> String {
    a.expand(units::spec(unit).session_pk)
}

type RowColumn = (String, String, ColKind, Option<ColumnReference>);

/// The key and default columns of each unit's rows.
fn row_columns(unit: Unit, a: &Aliases, now: i64) -> (String, Vec<RowColumn>) {
    let spec = units::spec(unit);
    let cols = spec
        .columns
        .iter()
        .map(|(name, e, k)| {
            (
                (*name).to_owned(),
                e.render(a, now),
                *k,
                if *k == ColKind::Text {
                    polylogue_query::fields::column_reference(unit, name)
                } else {
                    None
                },
            )
        })
        .collect();
    (a.expand(spec.key), cols)
}

/// A relation's template over the outer aliases `a` and the target's `n`,
/// with the target's FROM clause and the condition `cond` on it.
fn relation_sql(template: &str, a: &Aliases, n: &Aliases, target: Unit, cond: &str) -> String {
    units::expand_pair(template, a, n)
        .replace("{next}", &units::adjacent_call(a, false))
        .replace("{previous}", &units::adjacent_call(a, true))
        .replace("{from}", &base_from(target, n))
        .replace("{cond}", cond)
}

/// The session ID a web chat's conversation page URL names.
fn provider_url(pages: &[ChatPage], url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    let host = host.trim_start_matches("www.");
    let page = pages.iter().find(|p| p.hosts.iter().any(|h| h == host))?;
    let mut segments = path
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .split('/')
        .filter(|s| !s.is_empty());
    segments.find(|s| *s == page.path)?;
    Some(format!("{}:{}", page.origin, segments.next()?))
}

fn parse_pr(v: &str) -> Option<(Option<String>, i64)> {
    if let Ok(n) = v.parse::<i64>() {
        return Some((None, n));
    }
    if let Some((repo, n)) = v.rsplit_once('#') {
        return Some((Some(repo.to_owned()), n.parse().ok()?));
    }
    // https://github.com/owner/name/pull/42
    let rest = v.split_once("github.com/")?.1;
    let parts: Vec<&str> = rest.split('/').collect();
    if parts.len() >= 4 && parts[2] == "pull" {
        return Some((
            Some(format!("{}/{}", parts[0], parts[1])),
            parts[3].parse().ok()?,
        ));
    }
    None
}

#[cfg(test)]
mod tests;

fn scalar(kind: ColKind, field: &str, at: usize) -> Result<(), QueryError> {
    if matches!(kind, ColKind::Ids | ColKind::Json) {
        return Err(QueryError::Invalid {
            message: format!(
                "{field} is a {}, not a scalar field",
                if kind == ColKind::Ids {
                    "selection array"
                } else {
                    "structured value"
                }
            ),
            position: at,
        });
    }
    Ok(())
}

fn parse_query(text: &str, opts: &Options) -> Result<Query, QueryError> {
    crate::query::parse::with_catalog(text, &opts.catalog())
}
fn field_operand(operand: &Expression) -> Result<&str, QueryError> {
    match operand {
        Expression::Field { name } => Ok(name),
        _ => Err(not_built("this computed field")),
    }
}
fn literal_expression(expression: &Expression) -> Result<&Value, QueryError> {
    match expression {
        Expression::Value { value } => Ok(value),
        _ => Err(not_built("a computed value in this position")),
    }
}
fn parse_random_seed(operand: &Expression) -> Result<Option<u64>, QueryError> {
    let Expression::Function { name, arguments } = operand else {
        return Ok(None);
    };
    if name != "random" {
        return Err(not_built(&format!("sort key {name}")));
    }
    arguments
        .first()
        .map(|v| {
            literal_expression(v)?
                .text()
                .ok_or_else(|| not_built("this random seed"))?
                .parse()
                .map_err(|_| not_built("this random seed"))
        })
        .transpose()
}

pub(crate) struct FieldExpression {
    pub sql: String,
    pub kind: ColKind,
}
pub(crate) fn field_expression(
    unit: Unit,
    field: &str,
    now: jiff::Timestamp,
) -> Result<FieldExpression, QueryError> {
    let (sql, kind) = expression::expression(
        field,
        unit,
        Aliases::plain(),
        field,
        now.as_millisecond(),
        0,
        &mut 0,
    )?;
    Ok(FieldExpression { sql, kind })
}
