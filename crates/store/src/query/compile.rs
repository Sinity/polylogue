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
use polylogue_query::bound::Bound;
pub use polylogue_query::search::SnippetTerms;
use polylogue_query::selection::{FieldSelection, Order, Selection, contains_order};
pub use polylogue_query::{
    ColKind, Column, ColumnReference, Definition, Kept, Options, SavedQuery, Shape,
};
use serde::Serialize;

mod combine;
mod conditions;
pub(crate) mod custom_fields;
mod expression;
mod program;
mod ranking;
mod relations;
mod saved;
mod search;
mod sequences;
mod stages;
mod statements;
use conditions::unknown_field;
use polylogue_query::filter::{bind_unit, value_text};
use ranking::{Cx, Ranking, Reading};
pub use statements::{
    DerivedRows, KeyedRows, PageColumns, RecentSearch, ScopedSearch, Statements, TextSearch,
    page_of,
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

/// An ID-shaped word or a field condition on a session ID: it names at
/// most one session, and a reference names exactly one. The caller checks
/// that before running the query.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IdCheck {
    pub word: String,
    /// A reference that names no session is `not_found`; a field condition
    /// on an ID no session has matches no row.
    pub reference: bool,
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
    /// A population-changing default can have its own kept-row count.
    pub matches_sql: Option<String>,
    pub clause: String,
    pub sql: String,
}

/// A `similar:` term to rank by: the text whose similarity to each stored
/// text the caller computes and supplies while the query runs
/// (`pl_similarity`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Near {
    pub text: String,
    /// Every semantic selection also has a text relevance ranking.
    pub text_ranked: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowKind {
    UnitRows { unit: Unit, columns: Vec<String> },
    Group { columns: Vec<String> },
    Summary,
    Distinct { columns: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub query: String,
    pub row_kind: RowKind,
    pub unit: Unit,
    pub shape: Shape,
    /// The physical or logical meaning of a session row's native identity.
    #[serde(skip)]
    pub session_rows: SessionRows,
    /// The statements that select, page, count and read the answer's rows.
    pub statements: Statements,
    pub params: Vec<Param>,
    /// Every output column but `_pk` (each row unit's key).
    pub columns: Vec<Column>,
    /// Physical SQL slots for logical output names. Private identity and order
    /// slots never share the namespace of a user alias.
    pub column_slots: std::collections::BTreeMap<String, String>,
    #[serde(skip)]
    row_sources: Option<Vec<relations::RowSource>>,
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
    pub annotation_cutoffs: Vec<i64>,
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
    /// Saved-query revisions and result-name IDs resolved for this execution.
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

#[derive(Debug, Clone, PartialEq)]
pub enum SessionRows {
    Physical,
    Logical,
    /// A private SQL coordinate, selected with the winning operand's values.
    PerRow(String),
}

impl SessionRows {
    pub fn logical(&self, row: &serde_json::Value) -> Option<bool> {
        match self {
            Self::Physical => Some(false),
            Self::Logical => Some(true),
            Self::PerRow(column) => row.get(column)?.as_i64().map(|value| value != 0),
        }
    }

    fn sql(&self, alias: &str) -> String {
        match self {
            Self::Physical => "0".into(),
            Self::Logical => "1".into(),
            Self::PerRow(column) => format!("{alias}.{}", column_sql(column)),
        }
    }
}

/// One term of an answer's order: a column of the answer and its
/// direction. SQLite sorts NULL first in ascending order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OrderKey {
    pub column: String,
    pub descending: bool,
    /// The key is `<the next key> IS NULL`: it splits the rows into those
    /// with a value and those without, and a page reads each part in the
    /// order of the keys after it, so one index on the next key serves both.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub partition: bool,
    /// The key's values are bytes, which a page carries as hexadecimal text.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub bytes: bool,
}

impl OrderKey {
    fn ascending(column: &str) -> Self {
        Self {
            column: column.to_owned(),
            ..Self::default()
        }
    }

    /// The bound parameter `?{n}` as this key's stored value.
    pub(crate) fn bound(&self, n: usize) -> String {
        if self.bytes {
            format!("unhex(?{n})")
        } else {
            format!("?{n}")
        }
    }
}

/// Whether a row with these order `values` is in the second part of a
/// partitioned `order` ([`OrderKey::partition`]): the rows without a value.
pub fn in_second_part(order: &[OrderKey], values: &[serde_json::Value]) -> bool {
    order.first().is_some_and(|key| key.partition)
        && values
            .first()
            .is_some_and(|value| value.as_i64() == Some(1) || value.as_bool() == Some(true))
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
            let partition = i == 0
                && order
                    .get(1)
                    .is_some_and(|next| expr == format!("{} IS NULL", order_term(next).0));
            (
                format!("{expr} AS {column}"),
                OrderKey {
                    column,
                    descending,
                    partition,
                    bytes: expr.ends_with(".session_order"),
                },
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
    lower_bound(plan, None, opts)
}

/// Lowers `plan`, bound for this execution once: `bound` when the caller
/// already bound it with `opts`.
pub fn lower_bound(
    plan: &polylogue_query::plan::Plan,
    bound: Option<&Bound>,
    opts: &Options,
) -> Result<Program, QueryError> {
    let mut captured = opts.clone();
    captured.saved = plan.bindings().saved.clone();
    captured.kept = plan.bindings().kept.clone();
    for (id, kept) in &mut captured.kept {
        let current = opts.kept.get(id).ok_or_else(|| QueryError::Unavailable {
            reason: format!("kept result {id:?} is absent from this read"),
        })?;
        current.validate_output_metadata()?;
        if [ResultPart::Rows, ResultPart::Matches]
            .into_iter()
            .any(|part| kept.output_schema(part) != current.output_schema(part))
        {
            return Err(QueryError::Unavailable {
                reason: format!("kept result {id:?} differs from its bound columns"),
            });
        }
        *kept = current.clone();
    }
    let opts = &captured;
    let owned;
    let bound = match bound {
        Some(bound) => bound,
        None => {
            owned = plan.bound(opts)?;
            &owned
        }
    };
    let compiler = Compiler::new(plan.expanded_text(), opts);
    let mut program = compiler.program(bound, Reading::answer())?;
    let mut used: Vec<Definition> = plan
        .definitions()
        .iter()
        .map(|definition| Definition {
            name: definition.name.clone(),
            revision: definition.revision.clone(),
        })
        .collect();
    used.extend(
        plan.bindings()
            .kept_names
            .iter()
            .map(|(name, id)| Definition {
                name: format!("result:{name}"),
                revision: format!("result:{id}"),
            }),
    );
    join(&mut program.definitions, &mut used);
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
    notes: Vec<String>,
    incomplete: Vec<String>,
    snippet_terms: Vec<SnippetTerms>,
    id_checks: Vec<IdCheck>,
    change_generations: Vec<String>,
    annotation_cutoffs: Vec<i64>,
    /// Fields shown beside each row because a condition or sort named them.
    shown: Vec<RowColumn>,
    aliases: usize,
    /// The `similar:` text a statement of this query scores against,
    /// which execution embeds before reading.
    similar_text: Option<Near>,
    definitions: Vec<Definition>,
    /// Top-level `since:`/`until:` bounds on the sessions, for measures.
    session_window: (Option<i64>, Option<i64>),
    /// The programs of the `in (…)` queries this compiler lowered, by the
    /// bound query, so a query a filter and its `--why` clause share is
    /// lowered once.
    lowered_queries: std::collections::HashMap<*const Bound, Program>,
    /// The last relation slot or step number given out, shared by every
    /// compiler of one query, so no two slots or steps share a name.
    slots: std::rc::Rc<std::cell::Cell<usize>>,
}

impl<'a> Compiler<'a> {
    fn new(text: &'a str, opts: &'a Options) -> Self {
        Self {
            text,
            opts,
            params: vec![],
            ctes: vec![],
            notes: vec![],
            incomplete: vec![],
            snippet_terms: vec![],
            resolved_times: Default::default(),
            id_checks: vec![],
            change_generations: vec![],
            annotation_cutoffs: vec![],
            shown: vec![],
            aliases: 0,
            similar_text: None,
            definitions: vec![],
            session_window: (None, None),
            lowered_queries: Default::default(),
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
        nested.slots = self.slots.clone();
        nested
    }

    /// Takes over what a compiler of one part of this query recorded:
    /// notes, checks, definitions and resolved times.
    fn absorb(&mut self, mut part: Compiler<'_>) {
        join(&mut self.notes, &mut part.notes);
        join(&mut self.incomplete, &mut part.incomplete);
        join(&mut self.id_checks, &mut part.id_checks);
        join(&mut self.change_generations, &mut part.change_generations);
        join(&mut self.annotation_cutoffs, &mut part.annotation_cutoffs);
        join(&mut self.definitions, &mut part.definitions);
        let mut resolved = self.resolved_times.borrow_mut();
        for time in part.resolved_times.into_inner() {
            if !resolved.contains(&time) {
                resolved.push(time);
            }
        }
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

    fn program(self, bound: &Bound, reading: Reading) -> Result<Program, QueryError> {
        if matches!(bound.query.source, Source::Result { .. }) {
            self.kept_program(bound)
        } else {
            self.program_with(bound, None, reading)
        }
    }

    /// The stages of `bound` over stored rows, its source ignored.
    fn program_over(mut self, bound: &Bound, mut program: Program) -> Result<Program, QueryError> {
        self.params = std::mem::take(&mut program.params);
        let stages = polylogue_query::stages::stages(&bound.query.transforms, 0)?;
        self.finish_relations(bound, &stages, relations::Relation::new(program), None)
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

fn parse_pr(v: &str) -> Option<(Option<String>, i64)> {
    if let Ok(n) = v.parse::<i64>() {
        return Some((None, n));
    }
    // https://github.com/owner/name/pull/42, with any path, query or fragment after the number
    if let Some((_, rest)) = v.split_once("github.com/") {
        let path = rest.split(['?', '#']).next().unwrap_or(rest);
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 4 && parts[2] == "pull" {
            return Some((
                Some(format!("{}/{}", parts[0], parts[1])),
                parts[3].parse().ok()?,
            ));
        }
        return None;
    }
    let (repo, n) = v.rsplit_once('#')?;
    Some((Some(repo.to_owned()), n.parse().ok()?))
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

/// The page size when the caller gives none: sessions rows are heavier
/// than other rows.
fn default_page_size(unit: Unit, shape: Shape) -> u64 {
    if unit == Unit::Sessions && shape == Shape::Rows {
        20
    } else {
        50
    }
}

fn literal_expression(expression: &Expression) -> Result<&Value, QueryError> {
    match expression {
        Expression::Value { value } => Ok(value),
        _ => Err(not_built("a computed value in this position")),
    }
}
fn parse_random_seed(operand: &Expression, at: usize) -> Result<Option<u64>, QueryError> {
    let Expression::Function { name, arguments } = operand else {
        return Ok(None);
    };
    if name != "random" {
        return Err(not_built(&format!("sort key {name}")));
    }
    arguments
        .first()
        .map(|v| {
            let text = literal_expression(v)?
                .text()
                .ok_or_else(|| not_built("this random seed"))?;
            text.parse().map_err(|_| QueryError::BadValue {
                field: "random".into(),
                value: text.to_string(),
                position: at,
                message: format!("random seed {text:?} is not an unsigned integer"),
            })
        })
        .transpose()
}
