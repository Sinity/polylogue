//! Compiling a parsed query to one SQL statement with bound parameters
//! (spec query.md, principles 1, 4 and 5; Defaults; Stages). The statement
//! is the whole answer in order; the caller pages it with `LIMIT`/`OFFSET`
//! inside one read, and counts it with [`Program::count_sql`]. There is no
//! row cap: a page is how much travels now, never how much is answered.

use crate::ast::*;
use crate::fields::{self, Aliases, Kind};
use crate::time::{self, Clock, Time};
use crate::units::{self, Via};
use crate::{QueryError, parse};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

mod conditions;
mod custom_fields;
mod program;
mod saved;
mod sequences;
mod stages;

use saved::{bind_saved_source, compile_kept};

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
    /// Whether vectors exist to rank `near:` by.
    pub embeddings: bool,
    /// The calling session, for `self`.
    pub self_session: Option<String>,
    /// Saved queries by name (current `saved_query` notes): `query:NAME`.
    pub saved: std::collections::BTreeMap<String, String>,
    /// The columns to show instead of the unit's defaults (`--fields`):
    /// field names, `prefix.field` paths included.
    pub fields: Vec<String>,
    /// Kept results by ID (`result:<id>`): what their stored rows hold.
    pub kept: std::collections::BTreeMap<String, Kept>,
}

/// A kept result's stored shape (spec user-data.md, kept results): the unit
/// its rows are, their columns, and the columns of its match rows when it
/// stored them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kept {
    pub unit: String,
    pub columns: Vec<Column>,
    #[serde(default)]
    pub match_columns: Vec<Column>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Param {
    Int(i64),
    Real(f64),
    Text(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    /// One row per unit row.
    Rows,
    /// One row per group: the keys, then the aggregates.
    Groups,
    /// One summary row of aggregates.
    Summary,
    /// One row per match of an `exists` or `seq` (`| matches`): the grain
    /// is the (owner, matching row) pair or the matched step tuple.
    Matches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColKind {
    Text,
    Number,
    /// Dollars.
    Money,
    /// Unix ms.
    Time,
    /// Milliseconds.
    Duration,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub kind: ColKind,
}

/// A pasted ID-shaped word: it is a reference only if it names exactly one
/// session or message. The caller checks that before running the query.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IdCheck {
    pub word: String,
    /// Returns the candidate session IDs (at most a few).
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

/// A `near:` term to rank by: its text and the parameter holding the
/// nearest texts as a JSON array of `[text id, similarity]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Near {
    pub text: String,
    pub param: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub query: String,
    pub unit: Unit,
    pub shape: Shape,
    /// The whole answer, in order, without paging.
    pub sql: String,
    pub params: Vec<Param>,
    /// Every output column but `_pk` (each row unit's key).
    pub columns: Vec<Column>,
    /// Ordered by relevance to text terms.
    pub ranked: bool,
    /// The full-text terms, for marking matches in snippets.
    pub text_terms: Vec<String>,
    pub id_checks: Vec<IdCheck>,
    pub attach: Vec<Attach>,
    pub notes: Vec<String>,
    /// Parts that could not run; the outcome is then `degraded`.
    pub degraded: Vec<String>,
    /// `--why`: each top-level condition counted alone, and `why_of`, the
    /// rows the defaults alone keep.
    pub why: Vec<WhyClause>,
    pub why_of: String,
    pub page_size: u64,
    /// `near:`: the text to embed, and the 1-based parameter the caller
    /// fills with the nearest texts before running the query.
    pub near: Option<Near>,
    /// The instant and zone the times were resolved against.
    pub now: i64,
    pub zone: String,
    pub day_start_minutes: u16,
    /// `| measure WHAT`: the measurement to run over the sessions `sql`
    /// selects, instead of returning its rows.
    pub measure: Option<Measure>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measure {
    pub what: String,
    /// The query named no conditions: the measurement covers every session.
    pub everything: bool,
}

impl Program {
    /// The answer page by page: binds `LIMIT` and `OFFSET` after `params`.
    pub fn page_sql(&self) -> String {
        let n = self.params.len();
        format!("{} LIMIT ?{} OFFSET ?{}", self.sql, n + 1, n + 2)
    }

    pub fn count_sql(&self) -> String {
        format!("SELECT count(*) FROM ({})", self.sql)
    }
}

pub fn compile(text: &str, opts: &Options) -> Result<Program, QueryError> {
    if let Some(program) = compile_kept(text, opts)? {
        return Ok(program);
    }
    let bound = bind_saved_source(text, opts)?;
    let text = bound.as_deref().unwrap_or(text);
    let query = parse(text)?;
    refuse_commands(text, &query, opts)?;
    Compiler::new(text, opts).program(&query)
}

/// Words a query may start with only quoted: command names, and near
/// misses of unit or command names (spec query.md, principle 2).
fn refuse_commands(text: &str, query: &Query, opts: &Options) -> Result<(), QueryError> {
    let Source::Sessions { cond } = &query.source else {
        return Ok(());
    };
    let first = match &cond.kind {
        CondKind::And { all } => &all[0],
        _ => cond,
    };
    let CondKind::Text {
        text: word,
        phrase: false,
    } = &first.kind
    else {
        return Ok(());
    };
    if first.span.start != text.len() - text.trim_start().len() {
        return Ok(());
    }
    let lower = word.to_ascii_lowercase();
    if matches!(lower.as_str(), "last" | "latest" | "self") || opts.origins.contains(&lower) {
        return Ok(());
    }
    let rest = text[first.span.end..].trim_start();
    if opts.command_words.contains(&lower) {
        return Err(QueryError::NotAQuery {
            word: word.clone(),
            message: format!(
                "\"{word}\" is a command, not a query; to search for the word, quote it"
            ),
            suggestion: Some(format!("\"{word}\" {rest}").trim_end().to_owned()),
        });
    }
    let allowed = match lower.chars().count() {
        0..=3 => 0,
        4..=6 => 1,
        _ => 2,
    };
    if allowed == 0 {
        return Ok(());
    }
    let units = Unit::ALL.iter().map(|u| u.name());
    for candidate in units.chain(opts.command_words.iter().map(String::as_str)) {
        if candidate != lower && strsim::damerau_levenshtein(&lower, candidate) <= allowed {
            return Err(QueryError::NotAQuery {
                word: word.clone(),
                message: format!("did you mean \"{candidate}\"?"),
                suggestion: Some(format!("{candidate} {rest}").trim_end().to_owned()),
            });
        }
    }
    Ok(())
}

/// The unit a query's rows currently are, and the aliases they are read
/// through.
#[derive(Debug, Clone)]
struct Scope {
    unit: Unit,
    a: Aliases,
}

impl Scope {
    fn plain(unit: Unit) -> Self {
        Self {
            unit,
            a: Aliases::plain(),
        }
    }
}

/// Which full-text index a term searches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Index {
    /// `text` blocks written by a person or a model.
    Prose,
    /// `thinking` blocks.
    Thinking,
    /// Tool output.
    Output,
}

/// The rows of `| matches`: an SQL body selecting `_pk` and `columns`,
/// ordered, and whether it needs `WITH RECURSIVE`.
struct MatchRel {
    sql: String,
    columns: Vec<(String, ColKind)>,
    recursive: bool,
}

struct Compiler<'a> {
    text: &'a str,
    opts: &'a Options,
    params: Vec<Param>,
    ctes: Vec<String>,
    notes: Vec<String>,
    degraded: Vec<String>,
    text_terms: Vec<String>,
    id_checks: Vec<IdCheck>,
    /// Fields shown beside each row because a condition or sort named them.
    shown: Vec<(String, String, ColKind)>,
    aliases: usize,
    /// Saved queries being expanded, innermost last: a cycle is refused.
    saved_stack: Vec<String>,
    /// The ranking expression, when a top-level text term ranks the rows.
    rank: Option<String>,
    near: Option<Near>,
}

impl<'a> Compiler<'a> {
    fn new(text: &'a str, opts: &'a Options) -> Self {
        Self {
            text,
            opts,
            params: vec![],
            ctes: vec![],
            notes: vec![],
            degraded: vec![],
            text_terms: vec![],
            id_checks: vec![],
            shown: vec![],
            aliases: 0,
            saved_stack: vec![],
            rank: None,
            near: None,
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
enum FieldOp<'q> {
    Match(&'q Value),
    Compare(Op, &'q str),
    Between(&'q str, &'q str),
}

/// A column of `| matches` rows, by name.
fn match_column(rel: &MatchRel, name: &str, at: usize) -> Result<(String, ColKind), QueryError> {
    rel.columns
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
        })
}

/// An aggregate over `| matches` rows: computed over the pairs or tuples.
fn match_agg(
    f: &AggFn,
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
        format!("sum({name} IS NULL)"),
        ColKind::Number,
    );
    Ok(match f {
        AggFn::Count => unreachable!("handled above"),
        AggFn::Sum { .. } => vec![(col, format!("sum({name})"), kind), unknown],
        AggFn::Avg { .. } => vec![(col, format!("avg({name})"), kind), unknown],
        AggFn::Min { .. } => vec![(col, format!("min({name})"), kind), unknown],
        AggFn::Max { .. } => vec![(col, format!("max({name})"), kind), unknown],
        AggFn::Percentile { p, .. } => {
            vec![(col, format!("pl_percentile({name}, {p})"), kind), unknown]
        }
        AggFn::CountDistinct { .. } => {
            vec![(col, format!("count(DISTINCT {name})"), ColKind::Number)]
        }
    })
}

fn not_built(what: &str, section: &str) -> QueryError {
    QueryError::NotBuilt {
        what: what.to_owned(),
        section: section.to_owned(),
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
/// parenthesized ANDs flattened (spec query.md, Defaults, binding).
fn top_level(c: &Cond) -> Vec<Cond> {
    match &c.kind {
        CondKind::And { all } => all.iter().flat_map(top_level).collect(),
        _ => vec![c.clone()],
    }
}

/// Field names the top-level conditions name, `session.` prefixes removed
/// for row units.
fn named_fields(top: &[Cond], unit: Unit) -> HashSet<String> {
    let mut out = HashSet::new();
    for c in top {
        let f = match &c.kind {
            CondKind::Match { field, .. }
            | CondKind::Compare { field, .. }
            | CondKind::Between { field, .. } => field,
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

fn text_unit(unit: Unit) -> bool {
    units::spec(unit).text
}

/// At least 8 characters of hex digits and dashes, containing a digit or
/// a dash: the shape of a pasted native ID (spec query.md, Grammar).
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
        Value::One { text, .. } => vec![text.clone()],
        Value::Alternatives { any } => any.clone(),
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
    key: &SortKey,
    desc: bool,
    keys: Option<&[(String, String, ColKind)]>,
    aggs: Option<&[(String, String, ColKind)]>,
) -> Result<Vec<String>, QueryError> {
    let SortKey::Field { field } = key else {
        return Ok(vec!["random()".into()]);
    };
    let name = flat(field);
    let known = keys
        .unwrap_or(&[])
        .iter()
        .chain(aggs.unwrap_or(&[]).iter())
        .any(|(n, _, _)| *n == name);
    if !known {
        return Err(QueryError::Invalid {
            message: format!(
                "after grouping, sort by one of the group's columns; {field:?} is not one"
            ),
            position: 0,
        });
    }
    let dir = if desc { " DESC" } else { "" };
    Ok(vec![format!("g.{name} IS NULL"), format!("g.{name}{dir}")])
}

/// The FROM clause of a unit over the given aliases.
fn base_from(unit: Unit, a: &Aliases) -> String {
    units::spec(unit).from(a)
}

/// The expression naming a row's session key.
fn child_session_column(unit: Unit, a: &Aliases) -> String {
    a.expand(units::spec(unit).session_pk)
}

/// The key and default columns of each unit's rows.
fn row_columns(unit: Unit, a: &Aliases, now: i64) -> (String, Vec<(String, String, ColKind)>) {
    let spec = units::spec(unit);
    let cols = spec
        .columns
        .iter()
        .map(|(name, e, k)| ((*name).to_owned(), e.render(a, now), *k))
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

/// The session ID a provider conversation URL names.
fn provider_url(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    let host = host.trim_start_matches("www.");
    let segments: Vec<&str> = path
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let last = |after: &str| {
        segments
            .iter()
            .position(|s| *s == after)
            .and_then(|i| segments.get(i + 1))
            .map(|s| (*s).to_owned())
    };
    match host {
        "chatgpt.com" | "chat.openai.com" => last("c").map(|id| format!("chatgpt:{id}")),
        "claude.ai" => last("chat").map(|id| format!("claude-ai:{id}")),
        "gemini.google.com" => last("app").map(|id| format!("gemini-app:{id}")),
        "aistudio.google.com" => last("prompts").map(|id| format!("ai-studio:{id}")),
        "grok.com" => last("chat").map(|id| format!("grok:{id}")),
        _ => None,
    }
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
