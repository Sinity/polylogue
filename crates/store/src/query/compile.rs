//! Compiling a parsed query to one SQL statement with bound parameters.
//! The statement is the whole answer in a total order; the caller pages it
//! by key with [`Program::page_sql`] and counts it with
//! [`Program::count_sql`]. There is no row cap: a page is how much travels
//! now, never how much is answered.

use crate::query::ast::*;
use crate::query::fields::{self, Aliases, Kind};
use crate::query::time::{self, Milliseconds, Time};
use crate::query::units::{self, Index, Search, Via};
use polylogue_query::QueryError;
use polylogue_query::compile::Parameters;
pub use polylogue_query::{
    ChatPage, ColKind, Column, ColumnReference, Definition, Kept, Options, SavedQuery, Shape,
};
use serde::Serialize;
use std::collections::HashSet;

mod combine;
mod conditions;
pub(crate) mod custom_fields;
mod expression;
mod program;
mod saved;
mod sequences;
mod stages;
use conditions::unknown_field;

use saved::compile_kept;

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
    /// Selects the child rows of one parent; the parent's `_pk` binds to
    /// the parameter after `params`.
    pub sql: String,
    pub params: Vec<Param>,
    pub columns: Vec<Column>,
    pub first: Option<u64>,
    pub last: Option<u64>,
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
    /// The whole answer, in order, without paging.
    pub sql: String,
    /// Exact cardinality and missing similarities, without presentation work
    /// that does not affect which rows the query selects.
    pub row_count_sql: Option<String>,
    /// Selected ordinary-row keys without presentation order. Combining
    /// queries uses these keys when only membership matters.
    pub row_membership_sql: Option<String>,
    pub params: Vec<Param>,
    /// Every output column but `_pk` (each row unit's key).
    pub columns: Vec<Column>,
    /// Ordered by relevance to text terms.
    pub ranked: bool,
    /// The full-text terms, for marking matches in snippets.
    pub text_terms: Vec<String>,
    pub id_checks: Vec<IdCheck>,
    /// Required change-feed generations, validated before any result stages.
    pub change_generations: Vec<String>,
    pub attach: Vec<Attach>,
    pub notes: Vec<String>,
    /// Parts that could not run; the outcome is then `incomplete`.
    pub incomplete: Vec<String>,
    /// `--why`: each top-level condition counted alone, and `why_of`, the
    /// rows the defaults alone keep.
    pub why: Vec<WhyClause>,
    pub why_of: String,
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
    /// The answer's total order, as columns of `sql`; a page continues
    /// after the last row's values of these columns. Empty for a summary,
    /// which is one row.
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
    keys.iter()
        .map(|k| {
            let dir = if k.descending { " DESC" } else { "" };
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
}

impl Program {
    /// One page of the answer. With `after`, it holds the rows that sort
    /// after a row whose order-key values are bound after `params`, one per
    /// key; the page size is bound last.
    pub fn page_sql(&self, after: bool) -> String {
        let n = self.params.len();
        let mut sql = format!("SELECT * FROM ({})", self.sql);
        if after && !self.order.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&self.after_condition(n));
        }
        if !self.order.is_empty() {
            sql.push_str(" ORDER BY ");
            sql.push_str(&order_by(&self.order));
        }
        let limit = if after {
            n + self.order.len() + 1
        } else {
            n + 1
        };
        sql.push_str(&format!(" LIMIT ?{limit}"));
        sql
    }

    /// Rows after the bound key values `?{first+1}`...: equal on every
    /// earlier key and after on this one, for some key. NULL sorts first.
    fn after_condition(&self, first: usize) -> String {
        let mut any = vec![];
        for (i, key) in self.order.iter().enumerate() {
            let mut all: Vec<String> = self.order[..i]
                .iter()
                .enumerate()
                .map(|(j, k)| format!("\"{}\" IS ?{}", k.column, first + j + 1))
                .collect();
            let (column, value) = (format!("\"{}\"", key.column), format!("?{}", first + i + 1));
            all.push(if key.descending {
                format!("({column} < {value} OR ({column} IS NULL AND {value} IS NOT NULL))")
            } else {
                format!("({column} > {value} OR ({value} IS NULL AND {column} IS NOT NULL))")
            });
            any.push(format!("({})", all.join(" AND ")));
        }
        any.join(" OR ")
    }

    pub fn count_sql(&self) -> String {
        self.row_count_sql.clone().unwrap_or_else(|| {
            format!(
                "SELECT count(*), {} FROM ({})",
                if self.counts_near {
                    "count(*) FILTER (WHERE _near IS NULL)"
                } else {
                    "0"
                },
                self.sql
            )
        })
    }
}

pub fn compile(text: &str, opts: &Options) -> Result<Program, QueryError> {
    let plan = polylogue_query::compile(text, opts)?;
    lower(&plan, opts)
}
pub fn lower(plan: &polylogue_query::plan::Plan, opts: &Options) -> Result<Program, QueryError> {
    let bound = plan.expanded_query();
    let catalog = plan.catalog();
    let rendered = query_language::format::format_query(bound.query(), catalog)
        .map_err(|e| QueryError::language(e, ""))?;
    let mut program = if let Some(program) = compile_kept(&rendered, bound.query(), opts)? {
        program
    } else {
        Compiler::new(&rendered, opts).program(bound.query())?
    };
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
    columns: Vec<(String, ColKind)>,
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
    text_terms: Vec<String>,
    id_checks: Vec<IdCheck>,
    change_generations: Vec<String>,
    /// Fields shown beside each row because a condition or sort named them.
    shown: Vec<RowColumn>,
    aliases: usize,
    /// Saved queries being expanded, innermost last: a cycle is refused.
    saved_stack: Vec<String>,
    saved_relations: Vec<saved::BoundRelation>,
    /// The full-text ranking expression, when a top-level text term ranks
    /// the rows.
    rank: Option<String>,
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
    /// An unavailable semantic order affects the answer or an earlier selection.
    near_unavailable: bool,
    /// Conditions at or after this position belong to a stage.
    stages_start: usize,
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
            notes: vec![],
            incomplete: vec![],
            text_terms: vec![],
            resolved_times: Default::default(),
            id_checks: vec![],
            change_generations: vec![],
            shown: vec![],
            aliases: 0,
            saved_stack: vec![],
            saved_relations: vec![],
            rank: None,
            near_rank: None,
            near_join: None,
            fusion: None,
            near: None,
            definitions: vec![],
            near_request: None,
            near_unavailable: false,
            stages_start: usize::MAX,
        }
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
        self.program_with(q, None)
    }

    /// The stages of `q` over stored rows, its source ignored.
    fn program_over(self, q: &Query, rel: MatchRel) -> Result<Program, QueryError> {
        self.program_with(q, Some(rel))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FieldOp<'q> {
    Match(&'q Value),
    Compare(Op, &'q str),
    Between(&'q str, &'q str),
}

/// A column of `| matches` rows, by name.
fn match_column(rel: &MatchRel, name: &str, at: usize) -> Result<(String, ColKind), QueryError> {
    let column = rel
        .columns
        .iter()
        .find(|(n, _)| n == name)
        .cloned()
        .ok_or_else(|| QueryError::Invalid {
            message: format!(
                "match rows have no field {name:?}; they have {}",
                rel.columns
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            position: at,
        })?;
    scalar(column.1, name, at)?;
    Ok(column)
}

/// An aggregate over `| matches` rows: computed over the pairs or tuples.
fn match_agg(
    f: &Aggregate,
    rel: &MatchRel,
    at: usize,
) -> Result<Vec<(String, String, ColKind)>, QueryError> {
    let col = f.column();
    let Some(field) = f.field() else {
        return Ok(vec![(col, "count(*)".into(), ColKind::Number)]);
    };
    let (name, kind) = match_column(rel, field, at)?;
    let unknown = (
        format!("unknown_{name}"),
        format!("count(*) - count({name})"),
        ColKind::Number,
    );
    Ok(match &f.function {
        AggregateFunction::Count => unreachable!("handled above"),
        AggregateFunction::Sum => vec![(col, format!("sum({name})"), kind), unknown],
        AggregateFunction::Avg => vec![(col, format!("avg({name})"), kind), unknown],
        AggregateFunction::Min => vec![(col, format!("min({name})"), kind), unknown],
        AggregateFunction::Max => vec![(col, format!("max({name})"), kind), unknown],
        AggregateFunction::Percentile(p) => {
            vec![(col, format!("pl_percentile({name}, {p})"), kind), unknown]
        }
        AggregateFunction::CountDistinct => {
            vec![(col, format!("count(DISTINCT {name})"), ColKind::Number)]
        }
        _ => return Err(not_built("this aggregate")),
    })
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

/// The conditions joined to the rest of the `where` by AND, with
/// parenthesized ANDs flattened.
fn top_level(c: &Condition) -> Vec<Condition> {
    match &c.kind {
        ConditionKind::And { conditions: all } => all.iter().flat_map(top_level).collect(),
        _ => vec![c.clone()],
    }
}

/// Field names the top-level conditions name, `session.` prefixes removed
/// for row units.
fn named_fields(top: &[Condition], unit: Unit) -> HashSet<String> {
    let mut out = HashSet::new();
    for c in top {
        let f = match &c.kind {
            ConditionKind::Match { field, .. } => field.as_str(),
            ConditionKind::Compare { operand, .. }
            | ConditionKind::Between { operand, .. }
            | ConditionKind::Since { operand, .. }
            | ConditionKind::Until { operand, .. } => match operand {
                Operand::Field { name } => name,
                _ => continue,
            },
            _ => continue,
        };
        let f = f.to_ascii_lowercase();
        let f = if unit != Unit::Sessions {
            f.strip_prefix("session.").map(str::to_owned).unwrap_or(f)
        } else {
            f
        };
        out.insert(f);
    }
    out
}

/// Plain words search this unit.
fn text_unit(unit: Unit) -> bool {
    match units::spec(unit).search {
        Search::Hits { words, .. } => words,
        Search::Substring(_) => true,
        Search::None => false,
    }
}

/// At least 8 characters of hex digits and dashes, containing a digit or
/// a dash: the shape of a pasted native ID.
fn id_shaped(word: &str) -> bool {
    word.len() >= 8
        && word.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
        && word.chars().any(|c| c.is_ascii_digit() || c == '-')
}

fn like_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn like_prefix(s: &str) -> String {
    format!("{}%", like_escape(s))
}

fn alternatives(v: &Value) -> Vec<String> {
    match v {
        Value::Scalar { text, .. } => vec![text.clone()],
        Value::Set { values } => values.iter().flat_map(alternatives).collect(),
        Value::Unknown => vec!["unknown".into()],
        Value::Parameter { name } => vec![format!("${name}")],
        Value::Any => vec![],
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Match | Kind::Prefix => "match",
        Kind::Substring => "substring",
        Kind::Subtree => "path",
        Kind::Number => "number",
        Kind::Money => "money",
        Kind::Duration => "duration",
        Kind::Time => "time",
        Kind::Bool => "true/false",
        Kind::Special => "special",
    }
}

fn bucket_name(b: Bucket) -> &'static str {
    match b {
        Bucket::Day => "day",
        Bucket::Week => "week",
        Bucket::Month => "month",
    }
}

/// The message refused for a value outside a closed set: the set, and the
/// units where the value does exist (`"error" is an outcome of tool_calls`).
fn closed_set_message(def: &fields::Field, value: &str) -> String {
    let set = def.values.unwrap_or(&[]).join(", ");
    let elsewhere: Vec<&str> = fields::FIELDS
        .iter()
        .filter(|f| f.name == def.name && f.unit != def.unit)
        .filter(|f| {
            f.values
                .is_some_and(|vs| vs.iter().any(|v| v.eq_ignore_ascii_case(value)))
        })
        .map(|f| f.unit.name())
        .collect();
    let mut m = format!("{} on {} is one of {set}", def.name, def.unit.name());
    if let Some(u) = elsewhere.first() {
        let article = if def.name.starts_with(['a', 'e', 'i', 'o', 'u']) {
            "an"
        } else {
            "a"
        };
        m.push_str(&format!("; \"{value}\" is {article} {} of {u}", def.name));
    }
    m
}

fn group_sort(
    key: &Operand,
    desc: bool,
    keys: Option<&[(String, String, ColKind)]>,
    aggs: Option<&[(String, String, ColKind)]>,
    random_seed: Option<u64>,
) -> Result<Vec<String>, QueryError> {
    let keys = keys.unwrap_or(&[]);
    let mut order = match key {
        Operand::Function { .. } => {
            let seed = parse_random_seed(key)?;
            if let Some(seed) = seed.or(random_seed) {
                let identity = format!(
                    "json_array({})",
                    keys.iter()
                        .map(|(n, _, _)| format!("g.{n}"))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                vec![format!("pl_random({identity}, '{seed}')")]
            } else {
                vec!["random()".into()]
            }
        }
        Operand::Field { name: field } => {
            let name = flat(field);
            if !keys
                .iter()
                .chain(aggs.unwrap_or(&[]).iter())
                .any(|(n, _, _)| *n == name)
            {
                return Err(QueryError::Invalid {
                    message: format!(
                        "after grouping, sort by one of the group's columns; {field:?} is not one"
                    ),
                    position: 0,
                });
            }
            let dir = if desc { " DESC" } else { "" };
            vec![format!("g.{name} IS NULL"), format!("g.{name}{dir}")]
        }
    };
    // A group's complete key is its stable identity, even when counts tie.
    order.extend(
        keys.iter()
            .map(|(name, _, _)| format!("g.{name} IS NULL, g.{name}")),
    );
    Ok(order)
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
        .map(|(name, e, k)| ((*name).to_owned(), e.render(a, now), *k, e.reference()))
        .collect();
    (a.expand(spec.key), cols)
}

/// A relation's template over the outer aliases `a` and the target's `n`,
/// with the target's FROM clause and the condition `cond` on it.
fn relation_sql(template: &str, a: &Aliases, n: &Aliases, target: Unit, cond: &str) -> String {
    units::expand_pair(template, a, n)
        .replace("{next}", &a.expand(units::NEXT_CALL))
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
fn bind_unit(name: &str) -> Result<Unit, QueryError> {
    Unit::from_name(name.strip_prefix("pl.").unwrap_or(name))
        .ok_or_else(|| not_built(&format!("unit {name}")))
}
fn field_operand(operand: &Operand) -> Result<&str, QueryError> {
    match operand {
        Operand::Field { name } => Ok(name),
        _ => Err(not_built("this computed field")),
    }
}
fn value_text(value: &Value) -> Result<&str, QueryError> {
    match value {
        Value::Scalar { text, .. } => Ok(text),
        Value::Unknown => Ok("unknown"),
        Value::Parameter { name } => Ok(name),
        _ => Err(not_built("a set in this scalar position")),
    }
}
trait AggregateField {
    fn field(&self) -> Option<&str>;
}
impl AggregateField for Aggregate {
    fn field(&self) -> Option<&str> {
        self.fields.first().map(String::as_str)
    }
}
#[derive(Clone, Copy)]
enum Bucket {
    Day,
    Week,
    Month,
}
fn group_operand(operand: &Operand) -> Result<(&str, Option<Bucket>), QueryError> {
    match operand {
        Operand::Field { name } => Ok((name, None)),
        Operand::Function { name, arguments } if arguments.len() == 1 => {
            let bucket = match name.as_str() {
                "day" => Bucket::Day,
                "week" => Bucket::Week,
                "month" => Bucket::Month,
                _ => return Err(not_built(&format!("group key {name}"))),
            };
            Ok((field_operand(&arguments[0])?, Some(bucket)))
        }
        _ => Err(not_built("this group key")),
    }
}
fn parse_random_seed(operand: &Operand) -> Result<Option<u64>, QueryError> {
    let Operand::Function { name, arguments } = operand else {
        return Ok(None);
    };
    if name != "random" {
        return Err(not_built(&format!("sort key {name}")));
    }
    arguments
        .first()
        .map(|v| {
            field_operand(v)?
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
