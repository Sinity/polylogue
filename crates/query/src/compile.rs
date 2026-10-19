//! Compiling a parsed query to one SQL statement with bound parameters
//! (spec query.md, principles 1, 4 and 5; Defaults; Stages). The statement
//! is the whole answer in order; the caller pages it with `LIMIT`/`OFFSET`
//! inside one read, and counts it with [`Program::count_sql`]. There is no
//! row cap: a page is how much travels now, never how much is answered.

use crate::ast::*;
use crate::fields::{self, Aliases, Kind};
use crate::time::{self, Clock, Time};
use crate::{COMMAND_WORDS, QueryError, parse};
use serde::Serialize;
use std::collections::HashSet;

/// What compiling needs besides the text.
#[derive(Debug, Clone)]
pub struct Options {
    pub clock: Clock,
    /// Origin names (`claude-code`, `codex`, …): a word whose part before
    /// the first `:` is one of these is a session reference, not a field.
    pub origins: Vec<String>,
    /// Rows per page when the query sets none: 20 sessions, 50 otherwise.
    pub page_size: Option<u64>,
    /// Whether vectors exist to rank `near:` by.
    pub embeddings: bool,
    /// The calling session, for `self`.
    pub self_session: Option<String>,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize)]
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
    if COMMAND_WORDS.contains(&lower.as_str()) {
        return Err(QueryError::NotAQuery {
            word: word.clone(),
            message: format!(
                "\"{word}\" is a command, not a query; to search for the word, quote it or use find"
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
    for candidate in units.chain(COMMAND_WORDS.iter().copied()) {
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

    fn program(mut self, q: &Query) -> Result<Program, QueryError> {
        let (mut scope, wheres, top) = match &q.source {
            Source::Empty => (Scope::plain(Unit::Sessions), vec![], vec![]),
            Source::Sessions { cond } => {
                let top = top_level(cond);
                (Scope::plain(Unit::Sessions), vec![(cond.clone())], top)
            }
            Source::Unit { unit, cond } => {
                let top = cond.as_ref().map(top_level).unwrap_or_default();
                (Scope::plain(*unit), cond.iter().cloned().collect(), top)
            }
        };
        if scope.unit == Unit::Turns {
            return Err(not_built(
                "the turns unit",
                "architecture 7, query language",
            ));
        }
        let mut where_sql: Vec<String> = self.defaults(&scope, &top);
        let defaults_only = where_sql.clone();
        let mut why = vec![];
        for c in &wheres {
            let sql = self.cond(c, &scope, true)?;
            where_sql.push(sql);
        }
        for c in &top {
            // `near:` ranks rather than filters: it keeps every row.
            let ranking = matches!(&c.kind, CondKind::Match { field, .. } if field.eq_ignore_ascii_case("near"));
            if ranking {
                continue;
            }
            let rank = self.rank.clone();
            let alone = self.cond(c, &scope, false)?;
            self.rank = rank;
            why.push((self.text[c.span.start..c.span.end].to_owned(), alone));
        }
        let mut order: Option<Vec<String>> = None;
        let mut group: Option<Vec<(String, String, ColKind)>> = None;
        let mut aggs: Option<Vec<(String, String, ColKind)>> = None;
        let mut having: Vec<String> = vec![];
        let mut group_order: Option<Vec<String>> = None;
        let mut selection: Option<(Option<u64>, Option<u64>)> = None;
        let mut distinct = false;

        for stage in &q.stages {
            let grouped = group.is_some() || aggs.is_some();
            match &stage.kind {
                StageKind::Scope { unit, cond } => {
                    if grouped || scope.unit != Unit::Sessions {
                        return Err(QueryError::Invalid {
                            message: "a `| UNIT where …` stage follows a session query".into(),
                            position: stage.span.start,
                        });
                    }
                    if *unit == Unit::Turns {
                        return Err(not_built(
                            "the turns unit",
                            "architecture 7, query language",
                        ));
                    }
                    let sessions =
                        format!("SELECT s.pk FROM session s WHERE {}", and_all(&where_sql));
                    let next = Scope::plain(*unit);
                    let ntop = cond.as_ref().map(top_level).unwrap_or_default();
                    let mut w = self.defaults(&next, &ntop);
                    let link = if fields::has_session(*unit) {
                        format!("{}.pk IN ({sessions})", next.a.s)
                    } else {
                        self.session_link(&next, &sessions)?
                    };
                    w.push(link);
                    if let Some(c) = cond {
                        w.push(self.cond(c, &next, true)?);
                    }
                    scope = next;
                    where_sql = w;
                    why.clear();
                    order = None;
                    selection = None;
                }
                StageKind::Where { cond } if !grouped => {
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    where_sql.push(self.cond(cond, &scope, false)?);
                }
                StageKind::Where { cond } => {
                    having.push(self.group_cond(cond, group.as_deref(), aggs.as_deref())?);
                }
                StageKind::SortBy { key, descending } if !grouped => {
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    order = Some(self.sort(key, *descending, &scope, stage.span.start)?);
                }
                StageKind::SortBy { key, descending } => {
                    group_order = Some(group_sort(
                        key,
                        *descending,
                        group.as_deref(),
                        aggs.as_deref(),
                    )?);
                }
                StageKind::GroupBy { keys } => {
                    if grouped {
                        return Err(QueryError::Invalid {
                            message: "a query groups once".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let mut cols = vec![];
                    for k in keys {
                        cols.push(self.group_key(k, &scope, stage.span.start)?);
                    }
                    group = Some(cols);
                }
                StageKind::Count => {
                    if aggs.is_some() {
                        return Err(QueryError::Invalid {
                            message: "a query aggregates once".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() && group.is_none() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    aggs = Some(vec![("count".into(), "count(*)".into(), ColKind::Number)]);
                }
                StageKind::Agg { fns } => {
                    if aggs.is_some() {
                        return Err(QueryError::Invalid {
                            message: "a query aggregates once".into(),
                            position: stage.span.start,
                        });
                    }
                    if selection.is_some() && group.is_none() {
                        self.apply_selection(&scope, &mut where_sql, &order, selection.take());
                    }
                    let mut cols = vec![];
                    for f in fns {
                        cols.extend(self.agg(f, &scope, stage.span.start)?);
                    }
                    aggs = Some(cols);
                }
                StageKind::Limit { n } => {
                    let (_, off) = selection.unwrap_or((None, None));
                    selection = Some((Some(*n), off));
                }
                StageKind::Offset { n } => {
                    let (lim, _) = selection.unwrap_or((None, None));
                    selection = Some((lim, Some(*n)));
                }
                StageKind::Distinct => distinct = true,
                StageKind::Matches => {
                    return Err(not_built(
                        "the matches stage",
                        "architecture 7, query language",
                    ));
                }
            }
        }

        let ctes = if self.ctes.is_empty() {
            String::new()
        } else {
            format!("WITH {} ", self.ctes.join(", "))
        };
        let from = self.from(&scope)?;
        let wh = and_all(&where_sql);
        let unit = scope.unit;

        let (sql, shape, columns) = if group.is_some() || aggs.is_some() {
            let keys = group.clone().unwrap_or_default();
            let aggs =
                aggs.unwrap_or_else(|| vec![("count".into(), "count(*)".into(), ColKind::Number)]);
            let mut select: Vec<String> =
                keys.iter().map(|(n, e, _)| format!("{e} AS {n}")).collect();
            select.extend(aggs.iter().map(|(n, e, _)| format!("{e} AS {n}")));
            let mut columns: Vec<Column> = keys
                .iter()
                .chain(aggs.iter())
                .map(|(n, _, k)| Column {
                    name: n.clone(),
                    kind: *k,
                })
                .collect();
            if keys.is_empty() {
                let sql = format!("{ctes}SELECT {} FROM {from} WHERE {wh}", select.join(", "));
                (sql, Shape::Summary, std::mem::take(&mut columns))
            } else {
                let group_by = keys
                    .iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let inner = format!(
                    "SELECT {} FROM {from} WHERE {wh} GROUP BY {group_by}",
                    select.join(", ")
                );
                let order = group_order.unwrap_or_else(|| {
                    let mut o = vec![];
                    if aggs.iter().any(|(n, _, _)| n == "count") {
                        o.push("count DESC".to_owned());
                    }
                    o.extend(keys.iter().map(|(n, _, _)| format!("{n} IS NULL, {n}")));
                    o
                });
                let mut sql = format!(
                    "{ctes}SELECT * FROM ({inner}) g WHERE {} ORDER BY {}",
                    and_all(&having),
                    order.join(", ")
                );
                if let Some((lim, off)) = selection {
                    sql = format!(
                        "SELECT * FROM ({sql} LIMIT {} OFFSET {})",
                        lim.map_or(-1, |n| n.min(i64::MAX as u64) as i64),
                        off.unwrap_or(0)
                    );
                }
                (sql, Shape::Groups, columns)
            }
        } else {
            if selection.is_some() {
                self.apply_selection(&scope, &mut where_sql, &order, selection.take());
            }
            let wh = and_all(&where_sql);
            let order = order.unwrap_or_else(|| self.default_order(&scope));
            let (pk, mut cols) = row_columns(unit, &scope.a);
            let mut columns: Vec<Column> = cols
                .iter()
                .map(|(n, _, k)| Column {
                    name: n.clone(),
                    kind: *k,
                })
                .collect();
            for (n, e, k) in std::mem::take(&mut self.shown) {
                if !columns.iter().any(|c| c.name == n) {
                    columns.push(Column {
                        name: n.clone(),
                        kind: k,
                    });
                    cols.push((n, e, k));
                }
            }
            if let Some(rank) = &self.rank {
                cols.push(("rank".into(), rank.clone(), ColKind::Number));
                columns.push(Column {
                    name: "rank".into(),
                    kind: ColKind::Number,
                });
            }
            let mut select: Vec<String> = vec![format!("{pk} AS _pk")];
            select.extend(cols.iter().map(|(n, e, _)| format!("{e} AS {n}")));
            let sql = if distinct {
                let names = columns
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{ctes}SELECT min(_pk) AS _pk, {names} FROM (SELECT {} FROM {from} WHERE {wh} ORDER BY {}) GROUP BY {names}",
                    select.join(", "),
                    order.join(", ")
                )
            } else {
                format!(
                    "{ctes}SELECT {} FROM {from} WHERE {wh} ORDER BY {}",
                    select.join(", "),
                    order.join(", ")
                )
            };
            (sql, Shape::Rows, columns)
        };

        let mut attach = vec![];
        if !q.attach.is_empty() {
            if shape != Shape::Rows {
                return Err(QueryError::Invalid {
                    message: "attached rows apply to rows, not to groups or summaries".into(),
                    position: q.attach[0].span.start,
                });
            }
            for child in &q.attach {
                attach.push(self.attach(child, &scope)?);
            }
        }

        let why = why
            .into_iter()
            .map(|(clause, cond)| WhyClause {
                sql: format!(
                    "{ctes}SELECT count(*) FROM {from} WHERE {}",
                    and_all(&[defaults_only.clone(), vec![cond]].concat())
                ),
                clause,
            })
            .collect();
        let why_of = format!(
            "{ctes}SELECT count(*) FROM {from} WHERE {}",
            and_all(&defaults_only)
        );
        let page_size =
            self.opts
                .page_size
                .unwrap_or(if unit == Unit::Sessions && shape == Shape::Rows {
                    20
                } else {
                    50
                });
        Ok(Program {
            query: self.text.to_owned(),
            unit,
            shape,
            sql,
            params: self.params,
            columns,
            ranked: self.rank.is_some(),
            text_terms: self.text_terms,
            id_checks: self.id_checks,
            attach,
            notes: self.notes,
            degraded: self.degraded,
            why,
            why_of,
            page_size,
            near: self.near,
            now: self.opts.clock.now,
            zone: self.opts.clock.day.zone_name().unwrap_or("UTC").to_owned(),
            day_start_minutes: self.opts.clock.day.start_minutes,
        })
    }

    /// A selection (`| limit`, `| offset`) turned into a membership test,
    /// so later stages filter, group or count only the selected rows.
    fn apply_selection(
        &mut self,
        scope: &Scope,
        where_sql: &mut Vec<String>,
        order: &Option<Vec<String>>,
        selection: Option<(Option<u64>, Option<u64>)>,
    ) {
        let Some((lim, off)) = selection else {
            return;
        };
        let (pk, _) = row_columns(scope.unit, &scope.a);
        let order = order.clone().unwrap_or_else(|| self.default_order(scope));
        let from = self.from(scope).unwrap_or_default();
        let inner = format!(
            "SELECT {pk} FROM {from} WHERE {} ORDER BY {} LIMIT {} OFFSET {}",
            and_all(where_sql),
            order.join(", "),
            lim.map_or(-1, |n| n.min(i64::MAX as u64) as i64),
            off.unwrap_or(0)
        );
        where_sql.push(format!("{pk} IN ({inner})"));
    }

    /// The FROM clause of a unit, joining the rows its fields read.
    fn from(&self, scope: &Scope) -> Result<String, QueryError> {
        base_from(scope.unit, &scope.a)
    }

    /// The default filters a unit binds, dropped by a top-level condition
    /// naming their field (spec query.md, Defaults, binding).
    fn defaults(&mut self, scope: &Scope, top: &[Cond]) -> Vec<String> {
        let named = named_fields(top, scope.unit);
        let reference = top.iter().any(|c| self.is_reference(c));
        let has = |f: &str| named.contains(f);
        let a = &scope.a;
        let mut out = vec![];
        let session_defaults = |out: &mut Vec<String>, s: &str| {
            if !reference && !has("hidden") {
                out.push(format!("NOT {}", fields::hidden(s)));
            }
            out.push(format!("NOT {}", fields::excised(s)));
        };
        match scope.unit {
            Unit::Sessions => {
                let lineage = [
                    "kind",
                    "parent",
                    "root",
                    "depth",
                    "lineage",
                    "link",
                    "team",
                    "agent_name",
                ];
                if !reference && !lineage.iter().any(|f| has(f)) {
                    out.push(format!("{}.kind <> 'subagent'", a.s));
                }
                if !has("source") {
                    out.push(format!("{}.stub = 0", a.s));
                }
                session_defaults(&mut out, &a.s);
            }
            Unit::Messages | Unit::Blocks | Unit::ToolCalls => {
                if !reference && !has("inherited") {
                    out.push(format!("{}.inherited_from IS NULL", a.m));
                }
                if !reference && !has("active") {
                    out.push(format!("COALESCE({}.on_active_path, 1) = 1", a.m));
                }
                if scope.unit == Unit::ToolCalls && !has("action") {
                    out.push(fields::counted_call(&a.t));
                }
                session_defaults(&mut out, &a.s);
            }
            Unit::Files | Unit::Events => session_defaults(&mut out, &a.s),
            Unit::Deliveries => {
                out.push(format!(
                    "({s}.pk IS NULL OR (NOT {} AND NOT {}))",
                    fields::hidden(&a.s),
                    fields::excised(&a.s),
                    s = a.s
                ));
            }
            Unit::Notes => {
                if !has("current") && !has("root") && !has("withdrawn") {
                    out.push(fields::current_note(&a.row));
                }
                if !has("withdrawn") {
                    out.push(format!("NOT {}", fields::withdrawn_note(&a.row)));
                }
            }
            Unit::Links | Unit::Runs | Unit::Processes | Unit::Turns => {}
        }
        out
    }

    fn default_order(&self, scope: &Scope) -> Vec<String> {
        let a = &scope.a;
        if let Some(rank) = &self.rank {
            let time = match scope.unit {
                Unit::Sessions => format!("{}.last_activity_at", a.s),
                _ => format!("{}.at", a.m),
            };
            return vec![
                format!("{rank} IS NULL"),
                rank.clone(),
                format!("{time} IS NULL"),
                format!("{time} DESC"),
            ];
        }
        let (pk, _) = row_columns(scope.unit, a);
        let (time, desc) = match scope.unit {
            Unit::Sessions => (format!("{}.last_activity_at", a.s), true),
            Unit::Messages | Unit::Blocks => (format!("{}.at", a.m), false),
            Unit::ToolCalls => (format!("{}.at", a.t), false),
            Unit::Events | Unit::Deliveries => (format!("{}.at", a.row), false),
            Unit::Files => (format!("{}.first_at", a.row), false),
            Unit::Notes => (format!("{}.created_at", a.row), true),
            Unit::Links => (format!("{}p.last_activity_at", a.row), false),
            Unit::Runs => (format!("{}.started_at", a.row), false),
            Unit::Processes => (format!("{}.started_at", a.row), false),
            Unit::Turns => (pk.clone(), false),
        };
        let mut o = vec![format!("{time} IS NULL")];
        match scope.unit {
            // Messages and blocks in session order within the same time.
            Unit::Messages | Unit::Blocks => {
                o.push(time.clone());
                o.push(format!("{}.order_input_pk", a.m));
                o.push(format!("{}.seq", a.m));
            }
            _ => o.push(format!("{time}{}", if desc { " DESC" } else { "" })),
        }
        o.push(format!("{pk}{}", if desc { " DESC" } else { "" }));
        o
    }

    fn is_reference(&self, c: &Cond) -> bool {
        match &c.kind {
            CondKind::Url { .. } | CondKind::Position { .. } => true,
            CondKind::Match { field, .. } => {
                let f = field.to_ascii_lowercase();
                self.opts.origins.contains(&f) || matches!(f.as_str(), "session" | "message")
            }
            CondKind::Text {
                text,
                phrase: false,
            } => matches!(text.as_str(), "last" | "latest" | "self") || id_shaped(text),
            _ => false,
        }
    }

    /// A condition. `top` is true for the query's own `where`, where a
    /// top-level text term ranks the rows.
    fn cond(&mut self, c: &Cond, scope: &Scope, top: bool) -> Result<String, QueryError> {
        Ok(match &c.kind {
            CondKind::And { all } => {
                // Several words in one AND must occur in the same message.
                let mut parts = vec![];
                let mut words = vec![];
                for x in all {
                    match &x.kind {
                        CondKind::Text { text, phrase }
                            if !self.is_reference(x) && text_unit(scope.unit) =>
                        {
                            words.push((text.clone(), *phrase, x.span));
                        }
                        _ => parts.push(self.cond(x, scope, top)?),
                    }
                }
                if !words.is_empty() {
                    parts.push(self.text(&words, scope, Index::Prose, top)?);
                }
                format!("({})", parts.join(" AND "))
            }
            CondKind::Or { any } => {
                let parts = any
                    .iter()
                    .map(|x| self.cond(x, scope, false))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("({})", parts.join(" OR "))
            }
            CondKind::Not { cond } => format!("NOT ({})", self.cond(cond, scope, false)?),
            CondKind::Text { text, phrase } => {
                if self.is_reference(c) {
                    self.reference_word(text, scope, c.span)?
                } else {
                    self.text(&[(text.clone(), *phrase, c.span)], scope, Index::Prose, top)?
                }
            }
            CondKind::Url { url } => self.url(url, scope, c.span)?,
            CondKind::Position { .. } => {
                return Err(QueryError::Invalid {
                    message: "@N refers to a position in the last listing, which only an interactive surface keeps".into(),
                    position: c.span.start,
                });
            }
            CondKind::Exists { unit, cond } => self.exists(*unit, cond, scope, c.span)?,
            CondKind::Seq { steps, gaps } => self.seq(steps, gaps, scope, c.span)?,
            CondKind::Match { field, value } => {
                self.field_cond(field, FieldOp::Match(value), scope, c.span, top)?
            }
            CondKind::Compare { field, op, value } => {
                self.field_cond(field, FieldOp::Compare(*op, value), scope, c.span, top)?
            }
            CondKind::Between { field, from, to } => {
                self.field_cond(field, FieldOp::Between(from, to), scope, c.span, top)?
            }
        })
    }

    /// A full-text condition over one index. Terms ANDed together must all
    /// occur in the same message (or block).
    fn text(
        &mut self,
        words: &[(String, bool, Span)],
        scope: &Scope,
        index: Index,
        top: bool,
    ) -> Result<String, QueryError> {
        let mut phrases = vec![];
        // A quoted phrase and a bare word with punctuation both search
        // their tokens in order.
        for (word, _, span) in words {
            let p = polylogue_model::text::fts_phrase(word);
            let Some(p) = p else {
                return Err(QueryError::Invalid {
                    message: format!(
                        "{word:?} has no letter or digit to search for; use command:\"…\" or input:\"…\" for punctuation"
                    ),
                    position: span.start,
                });
            };
            self.text_terms.push(word.trim_end_matches('*').to_owned());
            phrases.push(p);
        }
        let match_text = phrases.join(" AND ");
        let table = match index {
            Index::Output => "output_fts",
            _ => "text_fts",
        };
        let param = self.text_param(&match_text);
        let cte = format!("h{}", self.ctes.len() + 1);
        self.ctes.push(format!(
            "{cte} AS MATERIALIZED (SELECT rowid AS text_id, bm25({table}) AS rank FROM {table} WHERE {table} MATCH {param})"
        ));
        let block_kind = match index {
            Index::Prose => "xb.kind = 'text'",
            Index::Thinking => "xb.kind = 'thinking'",
            Index::Output => "xb.kind = 'tool_result'",
        };
        let writer = match index {
            Index::Prose => " AND xm.written_by IN ('human', 'model')",
            _ => "",
        };
        let a = &scope.a;
        let (cond, rank) = match scope.unit {
            Unit::Sessions => {
                let scope_sql = format!(
                    "FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
                     JOIN message xm ON xm.pk = xb.message_pk \
                     WHERE {block_kind}{writer} AND xm.inherited_from IS NULL \
                     AND COALESCE(xm.on_active_path, 1) = 1 \
                     AND xm.session_pk IN ({})",
                    fields::subtree(&a.s)
                );
                (
                    format!("EXISTS (SELECT 1 {scope_sql})"),
                    format!("(SELECT min(h.rank) {scope_sql})"),
                )
            }
            Unit::Messages => {
                let scope_sql = format!(
                    "FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
                     WHERE xb.message_pk = {m}.pk AND {block_kind}",
                    m = a.m
                );
                let writer_m = match index {
                    Index::Prose => format!(" AND {}.written_by IN ('human', 'model')", a.m),
                    _ => String::new(),
                };
                (
                    format!("(EXISTS (SELECT 1 {scope_sql}){writer_m})"),
                    format!("(SELECT min(h.rank) {scope_sql})"),
                )
            }
            Unit::Blocks => (
                format!(
                    "({b}.text_id IN (SELECT text_id FROM {cte}) AND {})",
                    block_kind.replace("xb.", &format!("{}.", a.b)),
                    b = a.b
                ),
                format!(
                    "(SELECT h.rank FROM {cte} h WHERE h.text_id = {}.text_id)",
                    a.b
                ),
            ),
            Unit::ToolCalls if index == Index::Output => {
                let scope_sql = format!(
                    "FROM {cte} h JOIN block xb ON xb.text_id = h.text_id WHERE xb.pk = {t}.result_block_pk",
                    t = a.t
                );
                (
                    format!("EXISTS (SELECT 1 {scope_sql})"),
                    format!("(SELECT min(h.rank) {scope_sql})"),
                )
            }
            Unit::Notes => {
                // Notes are few and live in archive.db: a substring test.
                self.ctes.pop();
                self.params.pop();
                let mut parts = vec![];
                for (w, _, _) in words {
                    let p = self.text_param(w);
                    parts.push(format!("instr(pl_key({}.value), pl_key({p})) > 0", a.row));
                }
                return Ok(format!("({})", parts.join(" AND ")));
            }
            other => {
                self.ctes.pop();
                self.params.pop();
                return Err(QueryError::Invalid {
                    message: format!(
                        "a search word applies to sessions, messages, blocks and notes, not {}; for tool calls use output:\"…\", command:\"…\" or input:\"…\"",
                        other.name()
                    ),
                    position: words[0].2.start,
                });
            }
        };
        if top && index == Index::Prose && self.rank.is_none() {
            self.rank = Some(rank);
        }
        Ok(cond)
    }

    /// A word that names something: `last`, `latest`, `self`, or a pasted
    /// ID (8+ hex digits and dashes).
    fn reference_word(
        &mut self,
        word: &str,
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        Ok(match word {
            "last" | "latest" => format!(
                "{s}.pk = (SELECT pk FROM session WHERE kind <> 'subagent' AND stub = 0 \
                 ORDER BY last_activity_at IS NULL, last_activity_at DESC, pk DESC LIMIT 1)"
            ),
            "self" => {
                let Some(me) = self.opts.self_session.clone() else {
                    return Err(QueryError::Invalid {
                        message: "`self` names the calling session, and this caller is not one"
                            .into(),
                        position: span.start,
                    });
                };
                let p = self.text_param(&me);
                format!("{s}.id = {p}")
            }
            id => {
                let p = self.text_param(&like_prefix(id));
                self.id_checks.push(IdCheck {
                    word: id.to_owned(),
                    sql: "SELECT id FROM session WHERE native_id LIKE ?1 ESCAPE '\\' \
                          UNION SELECT s.id FROM message m JOIN session s ON s.pk = m.session_pk \
                          WHERE m.native_message_id LIKE ?1 ESCAPE '\\' LIMIT 5"
                        .into(),
                    params: vec![Param::Text(like_prefix(id))],
                });
                format!(
                    "({s}.native_id LIKE {p} ESCAPE '\\' OR {s}.pk IN (SELECT session_pk FROM message WHERE native_message_id LIKE {p} ESCAPE '\\'))"
                )
            }
        })
    }

    /// A provider URL pasted as a reference.
    fn url(&mut self, url: &str, scope: &Scope, span: Span) -> Result<String, QueryError> {
        let id = provider_url(url).ok_or_else(|| QueryError::Invalid {
            message: format!("{url} is not a conversation URL Polylogue recognizes"),
            position: span.start,
        })?;
        self.session_ref(&id, scope, span)
    }

    /// `origin:native` (or a unique prefix of it) as a session condition.
    fn session_ref(&mut self, id: &str, scope: &Scope, span: Span) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        let exact = self.text_param(id);
        let prefix = self.text_param(&like_prefix(id));
        Ok(format!(
            "({s}.id = {exact} OR {s}.id LIKE {prefix} ESCAPE '\\')"
        ))
    }

    /// The alias of the session a row of this unit belongs to.
    fn session_alias(&self, scope: &Scope, span: Span) -> Result<String, QueryError> {
        if scope.unit == Unit::Sessions || fields::has_session(scope.unit) {
            Ok(scope.a.s.clone())
        } else {
            Err(QueryError::Invalid {
                message: format!("{} rows do not belong to one session", scope.unit.name()),
                position: span.start,
            })
        }
    }

    /// The rows of a unit without a session alias that relate to the
    /// sessions `sessions` selects.
    fn session_link(&mut self, scope: &Scope, sessions: &str) -> Result<String, QueryError> {
        let r = &scope.a.row;
        Ok(match scope.unit {
            Unit::Links => format!("({r}.child_pk IN ({sessions}) OR {r}p.pk IN ({sessions}))"),
            Unit::Notes => format!(
                "EXISTS (SELECT 1 FROM a.note_target snt JOIN session ss ON 'session:' || ss.id = snt.ref \
                 WHERE snt.note_id = {r}.id AND ss.pk IN ({sessions}))"
            ),
            Unit::Runs => format!(
                "EXISTS (SELECT 1 FROM run_session rs WHERE rs.run_pk = {r}.pk AND rs.session_pk IN ({sessions}))"
            ),
            Unit::Processes => format!(
                "EXISTS (SELECT 1 FROM process_session ps WHERE ps.process_pk = {r}.pk AND ps.session_pk IN ({sessions}))"
            ),
            other => {
                return Err(not_built(
                    &format!("scoping {} to sessions", other.name()),
                    "architecture 7",
                ));
            }
        })
    }

    fn exists(
        &mut self,
        unit: Unit,
        cond: &Cond,
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        if unit == Unit::Turns {
            return Err(not_built(
                "the turns unit",
                "architecture 7, query language",
            ));
        }
        let child = Scope {
            unit,
            a: self.fresh(),
        };
        let link = match (scope.unit, unit) {
            (Unit::Messages, Unit::Blocks) => {
                format!("{}.message_pk = {}.pk", child.a.b, scope.a.m)
            }
            (Unit::Messages, Unit::ToolCalls) => {
                format!("{}.message_pk = {}.pk", child.a.cb, scope.a.m)
            }
            _ if fields::has_session(unit) => {
                let s = self.session_alias(scope, span)?;
                format!(
                    "{}.session_pk = {s}.pk",
                    child_session_column(unit, &child.a)
                )
            }
            _ => {
                let s = self.session_alias(scope, span)?;
                self.session_link(&child, &format!("SELECT {s}.pk"))?
            }
        };
        let top = top_level(cond);
        let mut w = self.defaults(&child, &top);
        w.push(link);
        w.push(self.cond(cond, &child, false)?);
        Ok(format!(
            "EXISTS (SELECT 1 FROM {} WHERE {})",
            base_from(unit, &child.a)?,
            and_all(&w)
        ))
    }

    /// `seq(a -> b -> c)`: tool calls of one session in session order,
    /// each step a condition on one call.
    fn seq(
        &mut self,
        steps: &[Cond],
        gaps: &[Gap],
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        let ord_cte = "tc_order";
        if !self.ctes.iter().any(|c| c.starts_with(ord_cte)) {
            self.ctes.push(format!(
                "{ord_cte} AS (SELECT t.pk AS pk, t.session_pk AS session_pk, t.at AS at, \
                 row_number() OVER (PARTITION BY t.session_pk ORDER BY m.order_input_pk, m.seq, cb.idx) AS n \
                 FROM tool_call t JOIN block cb ON cb.pk = t.call_block_pk JOIN message m ON m.pk = cb.message_pk \
                 WHERE m.inherited_from IS NULL AND COALESCE(m.on_active_path, 1) = 1 AND {})",
                fields::counted_call("t")
            ));
        }
        let mut aliases = vec![];
        let mut conds = vec![];
        for step in steps {
            let a = self.fresh();
            let sc = Scope {
                unit: Unit::ToolCalls,
                a: Aliases {
                    s: s.clone(),
                    ..a.clone()
                },
            };
            conds.push(self.cond(step, &sc, false)?);
            aliases.push(a);
        }
        // Build from the last step outwards: each step EXISTS inside the
        // previous one, ordered after it.
        let mut inner = String::new();
        for i in (0..steps.len()).rev() {
            let a = &aliases[i];
            let o = format!("o{}", a.t);
            let mut w = vec![format!("{o}.session_pk = {s}.pk"), conds[i].clone()];
            if i > 0 {
                let prev = format!("o{}", aliases[i - 1].t);
                match gaps[i - 1] {
                    Gap::Later => w.push(format!("{o}.n > {prev}.n")),
                    Gap::Next => w.push(format!("{o}.n = {prev}.n + 1")),
                    Gap::Within { ms } => {
                        w.push(format!("{o}.n > {prev}.n"));
                        w.push(format!("{o}.at - {prev}.at <= {ms}"));
                    }
                }
            }
            if !inner.is_empty() {
                w.push(inner.clone());
            }
            inner = format!(
                "EXISTS (SELECT 1 FROM {ord_cte} {o} JOIN tool_call {t} ON {t}.pk = {o}.pk \
                 JOIN block {cb} ON {cb}.pk = {t}.call_block_pk JOIN message {m} ON {m}.pk = {cb}.message_pk \
                 WHERE {})",
                and_all(&w),
                t = a.t,
                cb = a.cb,
                m = a.m
            );
        }
        Ok(inner)
    }

    /// A condition on a field: plain, related (`session.repo`), or special.
    fn field_cond(
        &mut self,
        field: &str,
        op: FieldOp<'_>,
        scope: &Scope,
        span: Span,
        top: bool,
    ) -> Result<String, QueryError> {
        let lower = field.to_ascii_lowercase();
        // A known origin or a reference prefix before `:` is a reference.
        if let FieldOp::Match(Value::One { text, .. }) = &op {
            if self.opts.origins.contains(&lower) {
                return self.session_ref(&format!("{lower}:{text}"), scope, span);
            }
            match lower.as_str() {
                "session" => return self.session_ref(text, scope, span),
                "message" => {
                    let s = self.session_alias(scope, span)?;
                    let p = self.text_param(text);
                    return Ok(match scope.unit {
                        Unit::Messages => format!("{}.id = {p}", scope.a.m),
                        _ => format!("{s}.pk IN (SELECT session_pk FROM message WHERE id = {p})"),
                    });
                }
                "query" => {
                    return Err(not_built(
                        "saved queries (query:NAME)",
                        "architecture 7, query language",
                    ));
                }
                "result" | "page" | "note" | "block" | "file" => {
                    return Err(not_built(
                        &format!("{lower}: references"),
                        "architecture 7, query language",
                    ));
                }
                _ => {}
            }
        }
        if let Some((prefix, rest)) = lower.split_once('.')
            && let Some(target) = fields::relation(scope.unit, prefix)
        {
            return self.related(prefix, rest, target, op, scope, span);
        }
        let (unit, a) = if fields::lookup(scope.unit, &lower).is_some() {
            (scope.unit, scope.a.clone())
        } else if (fields::has_session(scope.unit))
            && fields::lookup(Unit::Sessions, &lower).is_some()
        {
            (Unit::Sessions, scope.a.clone())
        } else if lower == "session" && scope.unit != Unit::Sessions {
            // `session:ID` handled above; `session` alone is a group key.
            return Err(self.unknown_field(field, scope.unit, span));
        } else {
            return Err(self.unknown_field(field, scope.unit, span));
        };
        let def = fields::lookup(unit, &lower).expect("checked");
        if def.kind == Kind::Special {
            return self.special(unit, def.name, op, &Scope { unit, a }, span, top);
        }
        let expr = fields::sql(unit, def.name, &a).expect("non-special fields have SQL");
        if top && matches!(op, FieldOp::Compare(..) | FieldOp::Between(..)) {
            self.show(def.name, &expr, col_kind(def.kind));
        }
        self.typed(def, &expr, op, span)
    }

    fn show(&mut self, name: &str, expr: &str, kind: ColKind) {
        let n = flat(name);
        if !self.shown.iter().any(|(x, _, _)| *x == n) {
            let e = match kind {
                ColKind::Money => format!("({expr}) / 1e9"),
                _ => expr.to_owned(),
            };
            self.shown.push((n, e, kind));
        }
    }

    fn unknown_field(&self, field: &str, unit: Unit, span: Span) -> QueryError {
        let lower = field.to_ascii_lowercase();
        let mut pool: Vec<&str> = fields::names(unit).collect();
        if fields::has_session(unit) {
            pool.extend(fields::names(Unit::Sessions));
        }
        let mut scored: Vec<(usize, &str)> = pool
            .into_iter()
            .map(|n| (strsim::damerau_levenshtein(&lower, n), n))
            .filter(|(d, n)| *d <= 2.max(n.len() / 4))
            .collect();
        scored.sort();
        scored.dedup_by(|a, b| a.1 == b.1);
        // Only the nearest names: one typing mistake beats two.
        let best = scored.first().map_or(0, |(d, _)| *d);
        let candidates: Vec<String> = scored
            .iter()
            .filter(|(d, _)| *d == best)
            .take(3)
            .map(|(_, n)| (*n).to_owned())
            .collect();
        let suggestion = match candidates.as_slice() {
            [one] => Some(format!(
                "{}{}{}",
                &self.text[..span.start],
                one,
                &self.text[span.start + field.len()..]
            )),
            _ => None,
        };
        QueryError::UnknownField {
            field: field.to_owned(),
            unit: unit.name().to_owned(),
            position: span.start,
            candidates,
            suggestion,
        }
    }

    /// A condition on a field of a related row.
    fn related(
        &mut self,
        prefix: &str,
        rest: &str,
        target: Unit,
        op: FieldOp<'_>,
        scope: &Scope,
        span: Span,
    ) -> Result<String, QueryError> {
        let sub = Span {
            start: span.start + prefix.len() + 1,
            end: span.end,
        };
        let a = &scope.a;
        match (scope.unit, prefix) {
            (_, "session") if fields::has_session(scope.unit) => {
                let sc = Scope {
                    unit: Unit::Sessions,
                    a: a.clone(),
                };
                self.field_cond(rest, op, &sc, sub, false)
            }
            (Unit::Blocks | Unit::ToolCalls, "message") => {
                let sc = Scope {
                    unit: Unit::Messages,
                    a: a.clone(),
                };
                self.field_cond(rest, op, &sc, sub, false)
            }
            (Unit::Links, "parent" | "child") => {
                let s = format!("{}{}", a.row, if prefix == "parent" { "p" } else { "c" });
                let sc = Scope {
                    unit: Unit::Sessions,
                    a: Aliases { s, ..a.clone() },
                };
                self.field_cond(rest, op, &sc, sub, false)
            }
            (Unit::Deliveries, "recipient") => {
                let sc = Scope {
                    unit: Unit::Sessions,
                    a: a.clone(),
                };
                self.field_cond(rest, op, &sc, sub, false)
            }
            _ => {
                let n = self.fresh();
                let inner = Scope {
                    unit: target,
                    a: n.clone(),
                };
                let cond = if (target, rest) == (Unit::Sessions, "agent_type") {
                    // child.agent_type: the agent type recorded on the link.
                    let FieldOp::Match(Value::One { text, .. }) = op else {
                        return Err(QueryError::Invalid {
                            message: "child.agent_type takes child.agent_type:VALUE".into(),
                            position: span.start,
                        });
                    };
                    let p = self.text_param(text);
                    format!("lower(json_extract(lc.attrs_json, '$.agent_type')) = lower({p})")
                } else {
                    self.field_cond(rest, op, &inner, sub, false)?
                };
                let from = base_from(target, &n)?;
                Ok(match (scope.unit, prefix) {
                    (Unit::Blocks, "call") => format!(
                        "EXISTS (SELECT 1 FROM {from} WHERE ({t}.call_block_pk = {b}.pk OR {t}.result_block_pk = {b}.pk) AND {cond})",
                        t = n.t,
                        b = a.b
                    ),
                    (Unit::ToolCalls, "child") => format!(
                        "EXISTS (SELECT 1 FROM link lc JOIN {from} ON {s}.pk = lc.child_pk \
                         WHERE lc.kind = 'subagent' AND lc.parent_id = {ps}.id \
                         AND (lc.spawning_call = {t}.call_id OR lc.spawned_by = {cb}.pk) AND {cond})",
                        s = n.s,
                        ps = a.s,
                        t = a.t,
                        cb = a.cb
                    ),
                    (Unit::ToolCalls, "next") => format!(
                        "EXISTS (SELECT 1 FROM {from} WHERE {nt}.pk = ({}) AND {cond})",
                        next_call(a),
                        nt = n.t
                    ),
                    (Unit::Runs | Unit::Processes, "session") => {
                        let (table, col) = if scope.unit == Unit::Runs {
                            ("run_session", "run_pk")
                        } else {
                            ("process_session", "process_pk")
                        };
                        format!(
                            "EXISTS (SELECT 1 FROM {table} rs JOIN {from} ON {s}.pk = rs.session_pk WHERE rs.{col} = {r}.pk AND {cond})",
                            s = n.s,
                            r = a.row
                        )
                    }
                    (Unit::Runs, "task") => format!(
                        "EXISTS (SELECT 1 FROM {from} WHERE {nr}.id = {r}.task_id AND {cond})",
                        nr = n.row,
                        r = a.row
                    ),
                    (Unit::Processes, "tool_call") => format!(
                        "EXISTS (SELECT 1 FROM {from} WHERE {t}.pk = {r}.tool_call_pk AND {cond})",
                        t = n.t,
                        r = a.row
                    ),
                    _ => {
                        return Err(QueryError::Invalid {
                            message: format!(
                                "{prefix}.{rest} does not apply to {}",
                                scope.unit.name()
                            ),
                            position: span.start,
                        });
                    }
                })
            }
        }
    }

    /// A condition on a plain field, by its type.
    fn typed(
        &mut self,
        def: &fields::Field,
        expr: &str,
        op: FieldOp<'_>,
        span: Span,
    ) -> Result<String, QueryError> {
        let name = def.name;
        let bad = |value: &str, message: String| QueryError::BadValue {
            field: name.to_owned(),
            value: value.to_owned(),
            position: span.start,
            message,
        };
        match (def.kind, op) {
            (_, FieldOp::Match(Value::Any)) => Ok("1".into()),
            (
                _,
                FieldOp::Match(Value::One {
                    text,
                    quoted: false,
                }),
            ) if text.eq_ignore_ascii_case("unknown") => Ok(format!("({expr}) IS NULL")),
            (Kind::Match, FieldOp::Match(v)) => {
                let values = match v {
                    Value::One { text, .. } => vec![text.clone()],
                    Value::Alternatives { any } => any.clone(),
                    Value::Any => unreachable!("handled above"),
                };
                if let Some(set) = def.values {
                    for x in &values {
                        if !set.iter().any(|s| s.eq_ignore_ascii_case(x)) {
                            return Err(bad(x, closed_set_message(def, x)));
                        }
                    }
                }
                let ps: Vec<String> = values
                    .iter()
                    .map(|x| self.text_param(&x.to_lowercase()))
                    .collect();
                Ok(format!("lower({expr}) IN ({})", ps.join(", ")))
            }
            (Kind::Match, FieldOp::Compare(op, value)) => {
                let p = self.text_param(&value.to_lowercase());
                Ok(format!("lower({expr}) {} {p}", op.sql()))
            }
            (Kind::Prefix, FieldOp::Match(v)) => {
                let values = alternatives(v);
                let mut parts = vec![];
                for x in values {
                    let exact = self.text_param(&x);
                    let prefix = self.text_param(&like_prefix(&x));
                    parts.push(format!(
                        "{expr} = {exact} OR {expr} LIKE {prefix} ESCAPE '\\'"
                    ));
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            (Kind::Substring, FieldOp::Match(v)) => {
                let mut parts = vec![];
                for x in alternatives(v) {
                    let p = self.text_param(&x);
                    parts.push(format!("instr(pl_key({expr}), pl_key({p})) > 0"));
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            (Kind::Subtree, FieldOp::Match(v)) => {
                let mut parts = vec![];
                for x in alternatives(v) {
                    parts.push(self.subtree(expr, &x));
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            (Kind::Bool, FieldOp::Match(Value::One { text, .. })) => {
                let b = match text.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => 1,
                    "false" | "no" | "0" => 0,
                    _ => return Err(bad(text, format!("{name} is true or false"))),
                };
                Ok(format!("({expr}) = {b}"))
            }
            (
                Kind::Number | Kind::Money | Kind::Duration,
                FieldOp::Match(Value::One { text, .. }),
            ) => {
                let v = self.scalar(def, text, span)?;
                Ok(format!("({expr}) = {v}"))
            }
            (Kind::Number | Kind::Money | Kind::Duration, FieldOp::Compare(op, value)) => {
                let v = self.scalar(def, value, span)?;
                Ok(format!("({expr}) {} {v}", op.sql()))
            }
            (Kind::Number | Kind::Money | Kind::Duration, FieldOp::Between(from, to)) => {
                let a = self.scalar(def, from, span)?;
                let b = self.scalar(def, to, span)?;
                Ok(format!("(({expr}) >= {a} AND ({expr}) <= {b})"))
            }
            (Kind::Time, FieldOp::Match(Value::One { text, .. })) => {
                let t = self.time(name, text, span)?;
                Ok(format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    t.start(),
                    t.end()
                ))
            }
            (Kind::Time, FieldOp::Compare(op, value)) => {
                let t = self.time(name, value, span)?;
                Ok(match op {
                    Op::Gt => format!("({expr}) >= {}", t.end()),
                    Op::Ge => format!("({expr}) >= {}", t.start()),
                    Op::Lt => format!("({expr}) < {}", t.start()),
                    Op::Le => format!("({expr}) < {}", t.end()),
                    Op::Eq => format!("(({expr}) >= {} AND ({expr}) < {})", t.start(), t.end()),
                    Op::Ne => format!("(({expr}) < {} OR ({expr}) >= {})", t.start(), t.end()),
                })
            }
            (Kind::Time, FieldOp::Between(from, to)) => {
                let a = self.time(name, from, span)?;
                let b = self.time(name, to, span)?;
                Ok(format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    a.start(),
                    b.end()
                ))
            }
            (kind, op) => Err(QueryError::Invalid {
                message: format!(
                    "{name} is a {} field; {}",
                    kind_name(kind),
                    match op {
                        FieldOp::Match(_) => "it takes a comparison such as `>`".to_owned(),
                        _ => format!("write {name}:VALUE"),
                    }
                ),
                position: span.start,
            }),
        }
    }

    fn scalar(
        &mut self,
        def: &fields::Field,
        text: &str,
        span: Span,
    ) -> Result<String, QueryError> {
        let bad = |message: String| QueryError::BadValue {
            field: def.name.to_owned(),
            value: text.to_owned(),
            position: span.start,
            message,
        };
        match def.kind {
            Kind::Money => time::money_nusd(text)
                .map(|n| n.to_string())
                .ok_or_else(|| bad(format!("{} is an amount in dollars such as 2.90", def.name))),
            Kind::Duration => time::duration_ms(text)
                .map(|n| n.to_string())
                .ok_or_else(|| bad(format!("{} is a duration such as 90s, 5m or 2h", def.name))),
            _ => {
                if let Ok(n) = text.parse::<i64>() {
                    Ok(n.to_string())
                } else if let Ok(f) = text.parse::<f64>()
                    && f.is_finite()
                {
                    Ok(format!("{f}"))
                } else {
                    Err(bad(format!("{} is a number", def.name)))
                }
            }
        }
    }

    fn time(&self, field: &str, text: &str, span: Span) -> Result<Time, QueryError> {
        time::resolve(text, &self.opts.clock).ok_or_else(|| QueryError::BadValue {
            field: field.to_owned(),
            value: text.to_owned(),
            position: span.start,
            message: format!(
                "{text:?} is not a time: use a date (2026-10-05), today, yesterday, this week, last month, 7d, 3h or 09:00"
            ),
        })
    }

    /// `path` and the directories below it, by whole components, exactly.
    fn subtree(&mut self, expr: &str, path: &str) -> String {
        let path = path.trim_end_matches('/');
        let path = if path.is_empty() { "/" } else { path };
        let exact = self.text_param(path);
        let below = self.text_param(&format!("{}/", path.trim_end_matches('/')));
        format!("({expr} = {exact} OR substr({expr}, 1, length({below})) = {below})")
    }

    /// Fields compiled by their own rule.
    fn special(
        &mut self,
        unit: Unit,
        name: &str,
        op: FieldOp<'_>,
        scope: &Scope,
        span: Span,
        top: bool,
    ) -> Result<String, QueryError> {
        let a = &scope.a;
        let s = &a.s;
        let one = |op: &FieldOp<'_>| -> Option<String> {
            match op {
                FieldOp::Match(Value::One { text, .. }) => Some(text.clone()),
                _ => None,
            }
        };
        let need_value = |name: &str| QueryError::Invalid {
            message: format!("{name} takes {name}:VALUE"),
            position: span.start,
        };
        let values = |op: &FieldOp<'_>| -> Vec<String> {
            match op {
                FieldOp::Match(v) => alternatives(v),
                _ => vec![],
            }
        };
        Ok(match (unit, name) {
            (_, "near") => {
                if !top {
                    return Err(QueryError::Invalid {
                        message: "near: ranks the whole query, so it appears once among the top-level conditions".into(),
                        position: span.start,
                    });
                }
                if !self.opts.embeddings {
                    self.degraded.push(
                        "semantic search is not enabled; rows are ordered by time and unranked"
                            .into(),
                    );
                    return Ok("1".into());
                }
                let text = one(&op).ok_or_else(|| need_value(name))?;
                // The caller binds the nearest texts as `[[text id, similarity], ...]`.
                let param = self.bind(Param::Text("[]".into()));
                let index = self.params.len();
                let cte = format!("n{}", self.ctes.len() + 1);
                self.ctes.push(format!(
                    "{cte} AS MATERIALIZED (SELECT CAST(json_extract(j.value, '$[0]') AS INTEGER) AS text_id, \
                     -json_extract(j.value, '$[1]') AS rank FROM json_each({param}) j)"
                ));
                let a = &scope.a;
                let rank = match unit {
                    Unit::Sessions => format!(
                        "(SELECT min(h.rank) FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
                         WHERE xb.session_pk IN ({}))",
                        fields::subtree(&a.s)
                    ),
                    Unit::Messages => format!(
                        "(SELECT min(h.rank) FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
                         WHERE xb.message_pk = {}.pk)",
                        a.m
                    ),
                    Unit::Blocks => {
                        format!(
                            "(SELECT h.rank FROM {cte} h WHERE h.text_id = {}.text_id)",
                            a.b
                        )
                    }
                    other => {
                        return Err(QueryError::Invalid {
                            message: format!(
                                "near: ranks sessions, messages and blocks, not {}",
                                other.name()
                            ),
                            position: span.start,
                        });
                    }
                };
                // Semantic similarity ranks the whole query, ahead of any
                // full-text rank.
                self.rank = Some(rank);
                self.near = Some(Near { text, param: index });
                "1".into()
            }
            (_, "output" | "thinking") => {
                let text = one(&op).ok_or_else(|| need_value(name))?;
                let quoted = matches!(op, FieldOp::Match(Value::One { quoted: true, .. }));
                let index = if name == "output" {
                    Index::Output
                } else {
                    Index::Thinking
                };
                self.text(&[(text, quoted, span)], scope, index, false)?
            }
            (_, "since") => {
                let t = self.time(name, &one(&op).ok_or_else(|| need_value(name))?, span)?;
                match unit {
                    Unit::Sessions => format!("{s}.last_activity_at >= {}", t.start()),
                    Unit::Messages | Unit::Blocks => format!("{}.at >= {}", a.m, t.start()),
                    Unit::ToolCalls => format!(
                        "(COALESCE({t}.at + {t}.duration_ms, {t}.at) >= {} OR {t}.result_block_pk IS NULL)",
                        t.start(),
                        t = a.t
                    ),
                    Unit::Events => format!("{}.at >= {}", a.row, t.start()),
                    Unit::Notes => format!("{}.created_at >= {}", a.row, t.start()),
                    _ => return Err(self.unknown_field(name, unit, span)),
                }
            }
            (_, "until") => {
                let t = self.time(name, &one(&op).ok_or_else(|| need_value(name))?, span)?;
                match unit {
                    Unit::Sessions => format!("{s}.started_at < {}", t.end()),
                    Unit::Messages | Unit::Blocks => format!("{}.at < {}", a.m, t.end()),
                    Unit::ToolCalls => format!("{}.at < {}", a.t, t.end()),
                    Unit::Events => format!("{}.at < {}", a.row, t.end()),
                    Unit::Notes => format!("{}.created_at < {}", a.row, t.end()),
                    _ => return Err(self.unknown_field(name, unit, span)),
                }
            }
            (Unit::Sessions, "repo")
                if matches!(&op, FieldOp::Match(Value::One { text, quoted: false })
                    if text.eq_ignore_ascii_case("unknown")) =>
            {
                format!("{s}.repo IS NULL")
            }
            (Unit::Sessions, "repo") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let v = v.to_lowercase().trim_end_matches('/').to_owned();
                    let p = self.text_param(&v);
                    let tail = self.text_param(&format!("%/{}", like_escape(&v)));
                    let tail_git = self.text_param(&format!("%/{}.git", like_escape(&v)));
                    parts.push(format!(
                        "(lower(rtrim({s}.repo, '/')) = {p} OR lower(rtrim({s}.repo, '/')) LIKE {tail} ESCAPE '\\' \
                         OR lower({s}.repo) LIKE {tail_git} ESCAPE '\\')"
                    ));
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "collection") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let p = self.text_param(&v.to_lowercase());
                    parts.push(format!(
                        "(lower({s}.collection_name) = {p} OR lower({s}.collection_id) = {p})"
                    ));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "commit") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                if v.len() < 7 || !v.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(QueryError::BadValue {
                        field: name.into(),
                        value: v,
                        position: span.start,
                        message: "commit takes a hash prefix of at least 7 hex digits".into(),
                    });
                }
                let p = self.text_param(&format!("{}%", v.to_lowercase()));
                format!(
                    "EXISTS (SELECT 1 FROM commit_link cl WHERE cl.session_pk = {s}.pk AND cl.commit_hash LIKE {p})"
                )
            }
            (Unit::Sessions, "pr") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (repo, number) = parse_pr(&v).ok_or_else(|| QueryError::BadValue {
                    field: name.into(),
                    value: v.clone(),
                    position: span.start,
                    message: "pr takes a number, owner/name#N, or a pull request URL".into(),
                })?;
                let n = self.bind(Param::Int(number));
                match repo {
                    Some(repo) => {
                        let p = self.text_param(&repo.to_lowercase());
                        format!(
                            "EXISTS (SELECT 1 FROM pr_link pl WHERE pl.session_pk = {s}.pk AND pl.number = {n} AND lower(pl.repo) = {p})"
                        )
                    }
                    None => format!(
                        "EXISTS (SELECT 1 FROM pr_link pl WHERE pl.session_pk = {s}.pk AND pl.number = {n})"
                    ),
                }
            }
            (Unit::Sessions, "cwd") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let a1 = self.subtree(&format!("{s}.cwd"), &v);
                    let a2 = self.subtree("cj.value", &v);
                    parts.push(format!(
                        "({a1} OR EXISTS (SELECT 1 FROM json_each({s}.cwd_json) cj WHERE {a2}))"
                    ));
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "model") => match op {
                FieldOp::Match(Value::One {
                    text,
                    quoted: false,
                }) if text.eq_ignore_ascii_case("unknown") => {
                    format!(
                        "NOT EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL)"
                    )
                }
                FieldOp::Match(Value::Any) => "1".into(),
                _ => {
                    let vs: Vec<String> = values(&op)
                        .iter()
                        .map(|v| self.text_param(&v.to_lowercase()))
                        .collect();
                    if vs.is_empty() {
                        return Err(need_value(name));
                    }
                    // Unknown when no message records a model, so neither
                    // the condition nor its NOT matches (three-valued, law 132).
                    format!(
                        "CASE WHEN EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL) \
                         THEN EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND lower(x.model) IN ({})) END",
                        vs.join(", ")
                    )
                }
            },
            (Unit::Sessions, "tag") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let v = v.to_lowercase();
                    let p = self.text_param(&v);
                    parts.push(format!(
                        "EXISTS (SELECT 1 FROM a.note tn JOIN a.note_target tt ON tt.note_id = tn.id \
                         WHERE tn.kind = 'tag' AND tt.ref = 'session:' || {s}.id AND {} \
                         AND ((tn.key IS NULL AND lower(tn.value) = {p}) \
                           OR lower(tn.key) || ':' || lower(tn.value) = {p} \
                           OR (tn.value IS NULL AND lower(tn.key) = {p})))",
                        fields::live_note("tn")
                    ));
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "has") => {
                let mut parts = vec![];
                for v in values(&op) {
                    parts.push(self.session_has(s, &v, span)?);
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Messages, "has") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let m = &a.m;
                    parts.push(match v.to_ascii_lowercase().as_str() {
                        "thinking" => format!("EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind = 'thinking')"),
                        "tool_calls" => format!("EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind = 'tool_call')"),
                        "attachments" => format!("EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind IN ('image', 'file'))"),
                        "code" => format!(
                            "EXISTS (SELECT 1 FROM block xb JOIN text tx ON tx.id = xb.text_id WHERE xb.message_pk = {m}.pk AND xb.kind = 'text' AND instr(pl_text(tx.body), char(96, 96, 96)) > 0)"
                        ),
                        other => {
                            return Err(QueryError::BadValue {
                                field: "has".into(),
                                value: other.into(),
                                position: span.start,
                                message: format!("has on messages is one of thinking, tool_calls, attachments, code; not {other:?}"),
                            });
                        }
                    });
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "parent") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                if v.eq_ignore_ascii_case("missing") {
                    format!(
                        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = l.parent_id))"
                    )
                } else {
                    let (e, p) = self.id_params(&v);
                    format!(
                        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND (l.parent_id = {e} OR l.parent_id LIKE {p} ESCAPE '\\'))"
                    )
                }
            }
            (Unit::Sessions, "root") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "{s}.pk IN (WITH RECURSIVE tr(pk) AS (SELECT pk FROM session WHERE id = {e} OR id LIKE {p} ESCAPE '\\' \
                     UNION SELECT l.child_pk FROM link l JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
                     WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr)"
                )
            }
            (Unit::Sessions, "depth") => {
                let expr = format!(
                    "(WITH RECURSIVE up(pk, d) AS (SELECT {s}.pk, 0 UNION SELECT lp.pk, up.d + 1 FROM up \
                     JOIN link l ON l.child_pk = up.pk AND l.kind = 'subagent' JOIN session lp ON lp.id = l.parent_id \
                     WHERE up.d < 1000) SELECT max(d) FROM up)"
                );
                let def = fields::Field {
                    unit: Unit::Sessions,
                    name: "depth",
                    kind: Kind::Number,
                    values: None,
                    help: "",
                };
                if top {
                    self.show("depth", &expr, ColKind::Number);
                }
                self.typed(&def, &expr, op, span)?
            }
            (Unit::Sessions, "link") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let cond = match v.to_ascii_lowercase().as_str() {
                        "fork" => "((l.kind = 'continuation' AND l.how = 'fork') OR (l.kind = 'subagent' AND l.how = 'copied'))".to_owned(),
                        "resume" => "(l.kind = 'continuation' AND l.how = 'resume')".to_owned(),
                        k @ ("subagent" | "continuation" | "reference") => format!("l.kind = '{k}'"),
                        other => {
                            return Err(QueryError::BadValue {
                                field: "link".into(),
                                value: other.into(),
                                position: span.start,
                                message: format!("link is one of subagent, continuation, reference, fork, resume; not {other:?}"),
                            });
                        }
                    };
                    parts.push(format!(
                        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND {cond})"
                    ));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "continues") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND l.kind = 'continuation' AND (l.parent_id = {e} OR l.parent_id LIKE {p} ESCAPE '\\'))"
                )
            }
            (Unit::Sessions, "continued_by") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "EXISTS (SELECT 1 FROM link l JOIN session lc ON lc.pk = l.child_pk WHERE l.parent_id = {s}.id \
                     AND l.kind = 'continuation' AND (lc.id = {e} OR lc.id LIKE {p} ESCAPE '\\'))"
                )
            }
            (Unit::Sessions, "lineage") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "{s}.id IN (WITH RECURSIVE comp(id) AS (SELECT id FROM session WHERE id = {e} OR id LIKE {p} ESCAPE '\\' \
                     UNION SELECT CASE WHEN l.parent_id = comp.id THEN lc.id ELSE l.parent_id END \
                     FROM link l JOIN session lc ON lc.pk = l.child_pk JOIN comp ON comp.id = l.parent_id OR comp.id = lc.id) \
                     SELECT id FROM comp)"
                )
            }
            (Unit::Sessions, "path" | "tool" | "action") => {
                let inner = Scope {
                    unit: Unit::ToolCalls,
                    a: Aliases {
                        t: "tq".into(),
                        ..a.clone()
                    },
                };
                let cond = self.field_cond(name, op, &inner, span, false)?;
                format!(
                    "EXISTS (SELECT 1 FROM tool_call tq WHERE tq.session_pk = {s}.pk AND {cond})"
                )
            }
            (Unit::ToolCalls, "command" | "path") => {
                let t = &a.t;
                let mut parts = vec![];
                for v in values(&op) {
                    let col = format!("{t}.{name}");
                    if v.chars().count() >= 3 {
                        let phrase = format!("\"{}\"", v.replace('"', "\"\""));
                        let p = self.text_param(&phrase);
                        parts.push(format!(
                            "{t}.pk IN (SELECT rowid FROM tool_call_trigram WHERE tool_call_trigram MATCH '{name}:' || {p})"
                        ));
                    } else {
                        let p = self.text_param(&v.to_lowercase());
                        parts.push(format!("instr(lower({col}), {p}) > 0"));
                    }
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::ToolCalls, "input") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let p = self.text_param(&v);
                format!(
                    "instr(pl_key(json_extract(pl_block({}.body), '$.input')), pl_key({p})) > 0",
                    a.cb
                )
            }
            (Unit::ToolCalls, "mcp_server") => {
                let vs: Vec<String> = values(&op)
                    .iter()
                    .map(|v| self.text_param(&v.to_lowercase()))
                    .collect();
                if vs.is_empty() {
                    return Err(need_value(name));
                }
                format!(
                    "({t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' AND lower(substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1)) IN ({}))",
                    vs.join(", "),
                    t = a.t
                )
            }
            (Unit::ToolCalls, "child" | "next")
            | (Unit::Links, "parent" | "child")
            | (Unit::Deliveries, "recipient") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                match (unit, name) {
                    (Unit::ToolCalls, "child") => format!(
                        "EXISTS (SELECT 1 FROM link lc JOIN session lcs ON lcs.pk = lc.child_pk WHERE lc.kind = 'subagent' \
                         AND lc.parent_id = {s}.id AND (lc.spawning_call = {t}.call_id OR lc.spawned_by = {cb}.pk) \
                         AND (lcs.id = {e} OR lcs.id LIKE {p} ESCAPE '\\'))",
                        t = a.t,
                        cb = a.cb
                    ),
                    (Unit::Links, "parent") => format!(
                        "({r}.parent_id = {e} OR {r}.parent_id LIKE {p} ESCAPE '\\')",
                        r = a.row
                    ),
                    (Unit::Links, "child") => {
                        format!("({r}c.id = {e} OR {r}c.id LIKE {p} ESCAPE '\\')", r = a.row)
                    }
                    (Unit::Deliveries, "recipient") => format!(
                        "({r}.recipient_id = {e} OR {r}.recipient_id LIKE {p} ESCAPE '\\')",
                        r = a.row
                    ),
                    _ => {
                        return Err(QueryError::Invalid {
                            message: "next names a relation; filter on next.<field>, such as next.action:shell".into(),
                            position: span.start,
                        });
                    }
                }
            }
            (Unit::Notes, "target") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let p = self.text_param(&v);
                let pp = self.text_param(&like_prefix(&v));
                format!(
                    "EXISTS (SELECT 1 FROM a.note_target nt WHERE nt.note_id = {r}.id AND (nt.ref = {p} OR nt.ref LIKE {pp} ESCAPE '\\' OR nt.ref = 'session:' || {p}))",
                    r = a.row
                )
            }
            (Unit::Deliveries, "refs" | "left_out") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let col = if name == "refs" {
                    "refs_json"
                } else {
                    "left_out_json"
                };
                let p = self.text_param(&v);
                format!(
                    "EXISTS (SELECT 1 FROM json_each({r}.{col}) dj WHERE instr(COALESCE(json_extract(dj.value, '$.ref'), dj.value), {p}) > 0)",
                    r = a.row
                )
            }
            _ => return Err(self.unknown_field(name, unit, span)),
        })
    }

    fn id_params(&mut self, v: &str) -> (String, String) {
        let e = self.text_param(v);
        let p = self.text_param(&like_prefix(v));
        (e, p)
    }

    fn session_has(&mut self, s: &str, v: &str, span: Span) -> Result<String, QueryError> {
        let own = fields::own_messages(s, "xm");
        Ok(match v.to_ascii_lowercase().as_str() {
            "thinking" => format!(
                "EXISTS (SELECT 1 FROM block xb JOIN message xm ON xm.pk = xb.message_pk WHERE {own} AND xb.kind = 'thinking')"
            ),
            "tool_calls" => {
                format!("EXISTS (SELECT 1 FROM tool_call xt WHERE xt.session_pk = {s}.pk)")
            }
            "errors" => format!(
                "EXISTS (SELECT 1 FROM tool_call xt WHERE xt.session_pk = {s}.pk AND xt.outcome = 'error')"
            ),
            "attachments" => format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.session_pk = {s}.pk AND xb.kind IN ('image', 'file'))"
            ),
            "compaction" => format!(
                "EXISTS (SELECT 1 FROM event xe WHERE xe.session_pk = {s}.pk AND xe.kind = 'compaction')"
            ),
            "branches" => format!(
                "EXISTS (SELECT 1 FROM message xm WHERE xm.session_pk = {s}.pk AND xm.on_active_path = 0)"
            ),
            "secret" => format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.session_pk = {s}.pk AND xb.looks_like_secret = 1)"
            ),
            "note" => format!(
                "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
                 WHERE (ht.ref = 'session:' || {s}.id OR ht.ref IN (SELECT 'message:' || xm.id FROM message xm WHERE xm.session_pk = {s}.pk)) \
                 AND {})",
                fields::live_note("hn")
            ),
            "conflict" => format!(
                "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
                 WHERE ht.ref = 'session:' || {s}.id AND {} \
                 GROUP BY hn.root HAVING count(*) > 1)",
                fields::current_note("hn")
            ),
            other => {
                return Err(QueryError::BadValue {
                    field: "has".into(),
                    value: other.into(),
                    position: span.start,
                    message: format!(
                        "has on sessions is one of {}; not {other:?}",
                        fields::SESSION_HAS.join(", ")
                    ),
                });
            }
        })
    }

    fn sort(
        &mut self,
        key: &SortKey,
        desc: bool,
        scope: &Scope,
        at: usize,
    ) -> Result<Vec<String>, QueryError> {
        let (pk, _) = row_columns(scope.unit, &scope.a);
        Ok(match key {
            SortKey::Random { seed: None } => vec!["random()".into()],
            SortKey::Random { seed: Some(seed) } => {
                vec![format!("(({pk} * 2654435761 + {seed}) % 4294967296)"), pk]
            }
            SortKey::Field { field } => {
                let (expr, kind) = self.expr(field, scope, at)?;
                self.show(field, &expr, kind);
                let dir = if desc { " DESC" } else { "" };
                vec![
                    format!("({expr}) IS NULL"),
                    format!("({expr}){dir}"),
                    format!("{pk}{dir}"),
                ]
            }
        })
    }

    /// The value of a field, for sorting and grouping.
    fn expr(
        &mut self,
        field: &str,
        scope: &Scope,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        let lower = field.to_ascii_lowercase();
        let span = Span {
            start: self.text[at..].find(field).map_or(at, |i| at + i),
            end: at,
        };
        let a = &scope.a;
        if lower == "session" && fields::has_session(scope.unit) {
            return Ok((format!("{}.id", a.s), ColKind::Text));
        }
        if let Some((prefix, rest)) = lower.split_once('.')
            && let Some(target) = fields::relation(scope.unit, prefix)
        {
            let sub = match (scope.unit, prefix) {
                (_, "session") if fields::has_session(scope.unit) => Scope {
                    unit: Unit::Sessions,
                    a: a.clone(),
                },
                (Unit::Blocks | Unit::ToolCalls, "message") => Scope {
                    unit: Unit::Messages,
                    a: a.clone(),
                },
                (Unit::Links, "parent" | "child") => Scope {
                    unit: Unit::Sessions,
                    a: Aliases {
                        s: format!("{}{}", a.row, if prefix == "parent" { "p" } else { "c" }),
                        ..a.clone()
                    },
                },
                (Unit::Deliveries, "recipient") => Scope {
                    unit: Unit::Sessions,
                    a: a.clone(),
                },
                (Unit::ToolCalls, "next") => {
                    let n = self.fresh();
                    let (e, k) = self.expr(
                        rest,
                        &Scope {
                            unit: target,
                            a: n.clone(),
                        },
                        at,
                    )?;
                    let from = base_from(target, &n)?;
                    return Ok((
                        format!(
                            "(SELECT {e} FROM {from} WHERE {}.pk = ({}))",
                            n.t,
                            next_call(a)
                        ),
                        k,
                    ));
                }
                _ => {
                    return Err(not_built(
                        &format!(
                            "grouping or sorting by {prefix}.{rest} on {}",
                            scope.unit.name()
                        ),
                        "architecture 7, query language",
                    ));
                }
            };
            return self.expr(rest, &sub, at);
        }
        let (unit, def) = if let Some(d) = fields::lookup(scope.unit, &lower) {
            (scope.unit, d)
        } else if fields::has_session(scope.unit)
            && let Some(d) = fields::lookup(Unit::Sessions, &lower)
        {
            (Unit::Sessions, d)
        } else {
            return Err(self.unknown_field(field, scope.unit, span));
        };
        if def.kind == Kind::Special {
            return match (unit, def.name) {
                (Unit::Sessions, "repo") => Ok((format!("{}.repo", a.s), ColKind::Text)),
                (Unit::Sessions, "cwd") => Ok((format!("{}.cwd", a.s), ColKind::Text)),
                (Unit::Sessions, "collection") => Ok((
                    format!("COALESCE({s}.collection_name, {s}.collection_id)", s = a.s),
                    ColKind::Text,
                )),
                (Unit::Sessions, "model") => Ok((
                    format!(
                        "(SELECT x.model FROM message x WHERE x.session_pk = {}.pk AND x.model IS NOT NULL GROUP BY x.model ORDER BY count(*) DESC LIMIT 1)",
                        a.s
                    ),
                    ColKind::Text,
                )),
                (Unit::ToolCalls, "command") => Ok((format!("{}.command", a.t), ColKind::Text)),
                (Unit::ToolCalls, "path") => Ok((format!("{}.path", a.t), ColKind::Text)),
                (Unit::ToolCalls, "mcp_server") => Ok((
                    format!(
                        "CASE WHEN {t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' THEN substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1) END",
                        t = a.t
                    ),
                    ColKind::Text,
                )),
                (Unit::Links, "parent") => Ok((format!("{}.parent_id", a.row), ColKind::Text)),
                (Unit::Links, "child") => Ok((format!("{}c.id", a.row), ColKind::Text)),
                (Unit::Deliveries, "recipient") => {
                    Ok((format!("{}.recipient_id", a.row), ColKind::Text))
                }
                _ => Err(QueryError::Invalid {
                    message: format!("{} cannot be grouped or sorted by", def.name),
                    position: span.start,
                }),
            };
        }
        let expr = fields::sql(unit, def.name, a).expect("non-special fields have SQL");
        Ok((expr, col_kind(def.kind)))
    }

    fn group_key(
        &mut self,
        key: &GroupKey,
        scope: &Scope,
        at: usize,
    ) -> Result<(String, String, ColKind), QueryError> {
        let (expr, kind) = self.expr(&key.field, scope, at)?;
        let name = flat(&key.field);
        Ok(match key.bucket {
            None => {
                let e = match kind {
                    ColKind::Money => format!("({expr}) / 1e9"),
                    _ => expr,
                };
                (name, e, kind)
            }
            Some(b) => {
                if kind != ColKind::Time {
                    return Err(QueryError::Invalid {
                        message: format!("{}({}) needs a time field", bucket_name(b), key.field),
                        position: at,
                    });
                }
                let zone = self.text_param(self.opts.clock.day.zone_name().unwrap_or("UTC"));
                let start = self.opts.clock.day.start_minutes;
                (
                    format!("{}_{name}", bucket_name(b)),
                    format!("pl_{}({expr}, {zone}, {start})", bucket_name(b)),
                    ColKind::Text,
                )
            }
        })
    }

    fn agg(
        &mut self,
        f: &AggFn,
        scope: &Scope,
        at: usize,
    ) -> Result<Vec<(String, String, ColKind)>, QueryError> {
        let col = f.column();
        let Some(field) = f.field() else {
            return Ok(vec![(col, "count(*)".into(), ColKind::Number)]);
        };
        let (expr, kind) = self.expr(field, scope, at)?;
        let value = |e: String| match kind {
            ColKind::Money => format!("{e} / 1e9"),
            _ => e,
        };
        let flat_field = flat(field);
        let unknown = (
            format!("unknown_{flat_field}"),
            format!("sum(({expr}) IS NULL)"),
            ColKind::Number,
        );
        let mut out = match f {
            AggFn::Count => unreachable!("handled above"),
            AggFn::Sum { .. } => vec![(col, value(format!("sum({expr})")), kind), unknown],
            AggFn::Avg { .. } => vec![(col, value(format!("avg({expr})")), kind), unknown],
            AggFn::Min { .. } => vec![(col, value(format!("min({expr})")), kind), unknown],
            AggFn::Max { .. } => vec![(col, value(format!("max({expr})")), kind), unknown],
            AggFn::Percentile { p, .. } => {
                vec![
                    (col, value(format!("pl_percentile({expr}, {p})")), kind),
                    unknown,
                ]
            }
            AggFn::CountDistinct { .. } => {
                vec![(col, format!("count(DISTINCT {expr})"), ColKind::Number)]
            }
        };
        // A session cost known only in part (some request-bearing messages
        // unpriced) is a lower bound, and the total says how many.
        if scope.unit == Unit::Sessions
            && matches!(field, "cost" | "tree_cost")
            && matches!(f, AggFn::Sum { .. } | AggFn::Avg { .. })
        {
            out.push((
                format!("partial_{flat_field}"),
                format!(
                    "sum(({expr}) IS NOT NULL AND EXISTS (SELECT 1 FROM message xp WHERE {} AND xp.role = 'assistant' AND xp.cost_nusd IS NULL))",
                    fields::own_messages(&scope.a.s, "xp")
                ),
                ColKind::Number,
            ));
        }
        Ok(out)
    }

    /// `| where …` after grouping: conditions on the group's columns.
    fn group_cond(
        &mut self,
        c: &Cond,
        keys: Option<&[(String, String, ColKind)]>,
        aggs: Option<&[(String, String, ColKind)]>,
    ) -> Result<String, QueryError> {
        let cols: Vec<&(String, String, ColKind)> = keys
            .unwrap_or(&[])
            .iter()
            .chain(aggs.unwrap_or(&[]).iter())
            .collect();
        let find = |field: &str| {
            cols.iter()
                .find(|(n, _, _)| *n == flat(field))
                .map(|(n, _, k)| (n.clone(), *k))
        };
        let missing = |field: &str, at: usize| QueryError::Invalid {
            message: format!(
                "after grouping, conditions name the group's columns ({}); {field:?} is not one",
                cols.iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            position: at,
        };
        Ok(match &c.kind {
            CondKind::And { all } => format!(
                "({})",
                all.iter()
                    .map(|x| self.group_cond(x, keys, aggs))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" AND ")
            ),
            CondKind::Or { any } => format!(
                "({})",
                any.iter()
                    .map(|x| self.group_cond(x, keys, aggs))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" OR ")
            ),
            CondKind::Not { cond } => format!("NOT ({})", self.group_cond(cond, keys, aggs)?),
            CondKind::Compare { field, op, value } => {
                let (n, _) = find(field).ok_or_else(|| missing(field, c.span.start))?;
                let v = match value.parse::<f64>() {
                    Ok(x) if x.is_finite() => format!("{x}"),
                    _ => self.text_param(value),
                };
                format!("g.{n} {} {v}", op.sql())
            }
            CondKind::Match {
                field,
                value: Value::One { text, .. },
            } => {
                let (n, _) = find(field).ok_or_else(|| missing(field, c.span.start))?;
                let p = self.text_param(&text.to_lowercase());
                format!("lower(g.{n}) = {p}")
            }
            _ => {
                return Err(QueryError::Invalid {
                    message:
                        "after grouping, `| where` compares the group's columns, such as count >= 3"
                            .into(),
                    position: c.span.start,
                });
            }
        })
    }

    fn attach(&mut self, child: &Child, parent: &Scope) -> Result<Attach, QueryError> {
        if child.unit == Unit::Turns {
            return Err(not_built(
                "the turns unit",
                "architecture 7, query language",
            ));
        }
        let saved = std::mem::take(&mut self.params);
        let saved_ctes = std::mem::take(&mut self.ctes);
        let saved_shown = std::mem::take(&mut self.shown);
        let saved_rank = self.rank.take();
        let sc = Scope::plain(child.unit);
        let conds: Vec<Cond> = child
            .windows
            .iter()
            .filter_map(|w| match w {
                Window::Cond { cond } => Some(cond.clone()),
                _ => None,
            })
            .collect();
        let top: Vec<Cond> = conds.iter().flat_map(top_level).collect();
        let mut w = self.defaults(&sc, &top);
        for c in &conds {
            w.push(self.cond(c, &sc, false)?);
        }
        let mut select = vec![];
        let mut columns = vec![];
        for f in &child.fields {
            let lower = f.to_ascii_lowercase();
            let (expr, kind) = if child.unit == Unit::Messages && lower == "text" {
                (
                    format!(
                        "(SELECT group_concat(pl_text(tx.body), char(10)) FROM (SELECT xb.text_id FROM block xb WHERE xb.message_pk = {}.pk AND xb.kind = 'text' ORDER BY xb.idx) xbs JOIN text tx ON tx.id = xbs.text_id)",
                        sc.a.m
                    ),
                    ColKind::Text,
                )
            } else {
                self.expr(f, &sc, child.span.start)?
            };
            let e = match kind {
                ColKind::Money => format!("({expr}) / 1e9"),
                _ => expr,
            };
            select.push(format!("{e} AS {}", flat(f)));
            columns.push(Column {
                name: flat(f),
                kind,
            });
        }
        let parent_link = match (parent.unit, child.unit) {
            (Unit::Sessions, u) if fields::has_session(u) => {
                format!("{}.session_pk = ?PARENT", child_session_column(u, &sc.a))
            }
            (Unit::Messages, Unit::Blocks) => format!("{}.message_pk = ?PARENT", sc.a.b),
            (Unit::Messages, Unit::ToolCalls) => format!("{}.message_pk = ?PARENT", sc.a.cb),
            (Unit::Sessions, _) => self.session_link(&sc, "SELECT ?PARENT")?,
            (p, c) => {
                return Err(not_built(
                    &format!("attaching {} to {}", c.name(), p.name()),
                    "architecture 7, query language",
                ));
            }
        };
        w.push(parent_link);
        let ctes = if self.ctes.is_empty() {
            String::new()
        } else {
            format!("WITH {} ", self.ctes.join(", "))
        };
        let order = self.default_order(&sc);
        let n = self.params.len() + 1;
        let sql = format!(
            "{ctes}SELECT {} FROM {} WHERE {} ORDER BY {}",
            if select.is_empty() {
                "*".into()
            } else {
                select.join(", ")
            },
            base_from(child.unit, &sc.a)?,
            and_all(&w),
            order.join(", ")
        )
        .replace("?PARENT", &format!("?{n}"));
        let params = std::mem::replace(&mut self.params, saved);
        self.ctes = saved_ctes;
        self.shown = saved_shown;
        self.rank = saved_rank;
        let (mut first, mut last) = (None, None);
        for w in &child.windows {
            match w {
                Window::First { n } => first = Some(*n),
                Window::Last { n } => last = Some(*n),
                Window::Cond { .. } => {}
            }
        }
        Ok(Attach {
            name: child.unit.name().to_owned(),
            sql,
            params,
            columns,
            first,
            last,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum FieldOp<'q> {
    Match(&'q Value),
    Compare(Op, &'q str),
    Between(&'q str, &'q str),
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
    matches!(
        unit,
        Unit::Sessions | Unit::Messages | Unit::Blocks | Unit::Notes
    )
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

fn col_kind(k: Kind) -> ColKind {
    match k {
        Kind::Number => ColKind::Number,
        Kind::Money => ColKind::Money,
        Kind::Duration => ColKind::Duration,
        Kind::Time => ColKind::Time,
        Kind::Bool => ColKind::Bool,
        _ => ColKind::Text,
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
fn base_from(unit: Unit, a: &Aliases) -> Result<String, QueryError> {
    let (s, m, b, t, cb, r) = (&a.s, &a.m, &a.b, &a.t, &a.cb, &a.row);
    Ok(match unit {
        Unit::Sessions => format!("session {s}"),
        Unit::Messages => format!("message {m} JOIN session {s} ON {s}.pk = {m}.session_pk"),
        Unit::Blocks => format!(
            "block {b} JOIN message {m} ON {m}.pk = {b}.message_pk JOIN session {s} ON {s}.pk = {b}.session_pk"
        ),
        Unit::ToolCalls => format!(
            "tool_call {t} JOIN block {cb} ON {cb}.pk = {t}.call_block_pk \
             JOIN message {m} ON {m}.pk = {cb}.message_pk JOIN session {s} ON {s}.pk = {t}.session_pk"
        ),
        Unit::Files => format!(
            "(SELECT ft.session_pk AS session_pk, ft.path AS path, min(ft.pk) AS pk, \
             sum(ft.action = 'file_read') AS reads, sum(ft.action = 'file_write') AS writes, \
             sum(ft.action = 'file_edit') AS edits, min(ft.at) AS first_at, max(ft.at) AS last_at \
             FROM tool_call ft WHERE ft.path IS NOT NULL \
             AND ft.action IN ('file_read', 'file_write', 'file_edit') GROUP BY ft.session_pk, ft.path) {r} \
             JOIN session {s} ON {s}.pk = {r}.session_pk"
        ),
        Unit::Events => format!("event {r} JOIN session {s} ON {s}.pk = {r}.session_pk"),
        Unit::Links => format!(
            "link {r} LEFT JOIN session {r}p ON {r}p.id = {r}.parent_id LEFT JOIN session {r}c ON {r}c.pk = {r}.child_pk"
        ),
        Unit::Notes => format!("a.note {r}"),
        Unit::Runs => format!("run {r}"),
        Unit::Processes => format!("process {r}"),
        Unit::Deliveries => {
            format!("a.delivery {r} LEFT JOIN session {s} ON {s}.id = {r}.recipient_id")
        }
        Unit::Turns => {
            return Err(not_built(
                "the turns unit",
                "architecture 7, query language",
            ));
        }
    })
}

/// The alias of the table whose `session_pk` names a row's session.
fn child_session_column(unit: Unit, a: &Aliases) -> String {
    match unit {
        Unit::Messages => a.m.clone(),
        Unit::Blocks => a.b.clone(),
        Unit::ToolCalls => a.t.clone(),
        _ => a.row.clone(),
    }
}

/// The session's next tool call after `a.t`, in session order.
fn next_call(a: &Aliases) -> String {
    format!(
        "SELECT tq.pk FROM tool_call tq JOIN block qb ON qb.pk = tq.call_block_pk \
         JOIN message qm ON qm.pk = qb.message_pk WHERE tq.session_pk = {t}.session_pk \
         AND (qm.order_input_pk, qm.seq, qb.idx) > ({m}.order_input_pk, {m}.seq, {cb}.idx) \
         ORDER BY qm.order_input_pk, qm.seq, qb.idx LIMIT 1",
        t = a.t,
        m = a.m,
        cb = a.cb
    )
}

/// The key and default columns of each unit's rows.
fn row_columns(unit: Unit, a: &Aliases) -> (String, Vec<(String, String, ColKind)>) {
    use ColKind::*;
    let c = |n: &str, e: String, k: ColKind| (n.to_owned(), e, k);
    let (s, m, b, t, r) = (&a.s, &a.m, &a.b, &a.t, &a.row);
    match unit {
        Unit::Sessions => (
            format!("{s}.pk"),
            vec![
                c("id", format!("{s}.id"), Text),
                c("origin", format!("{s}.origin"), Text),
                c("kind", format!("{s}.kind"), Text),
                c("title", format!("{s}.title"), Text),
                c("started", format!("{s}.started_at"), Time),
                c("last_active", format!("{s}.last_activity_at"), Time),
                c("cwd", format!("{s}.cwd"), Text),
                c("repo", format!("{s}.repo"), Text),
            ],
        ),
        Unit::Messages => (
            format!("{m}.pk"),
            vec![
                c("id", format!("{m}.id"), Text),
                c("session", format!("{s}.id"), Text),
                c("role", format!("{m}.role"), Text),
                c("written_by", format!("{m}.written_by"), Text),
                c("at", format!("{m}.at"), Time),
                c("model", format!("{m}.model"), Text),
                c(
                    "inherited",
                    format!("({m}.inherited_from IS NOT NULL)"),
                    Bool,
                ),
            ],
        ),
        Unit::Blocks => (
            format!("{b}.pk"),
            vec![
                c("session", format!("{s}.id"), Text),
                c("message", format!("{m}.id"), Text),
                c("kind", format!("{b}.kind"), Text),
                c("idx", format!("{b}.idx"), Number),
                c("at", format!("{m}.at"), Time),
            ],
        ),
        Unit::ToolCalls => (
            format!("{t}.pk"),
            vec![
                c("session", format!("{s}.id"), Text),
                c("message", format!("{m}.id"), Text),
                c("tool", format!("{t}.tool"), Text),
                c("action", format!("{t}.action"), Text),
                c("command", format!("{t}.command"), Text),
                c("path", format!("{t}.path"), Text),
                c("outcome", format!("{t}.outcome"), Text),
                c("exit_code", format!("{t}.exit_code"), Number),
                c("at", format!("{t}.at"), Time),
                c("duration", format!("{t}.duration_ms"), Duration),
            ],
        ),
        Unit::Files => (
            format!("{r}.pk"),
            vec![
                c("session", format!("{s}.id"), Text),
                c("path", format!("{r}.path"), Text),
                c("reads", format!("{r}.reads"), Number),
                c("writes", format!("{r}.writes"), Number),
                c("edits", format!("{r}.edits"), Number),
                c("first_at", format!("{r}.first_at"), Time),
                c("last_at", format!("{r}.last_at"), Time),
            ],
        ),
        Unit::Events => (
            format!("{r}.pk"),
            vec![
                c("id", format!("{r}.id"), Text),
                c("session", format!("{s}.id"), Text),
                c("kind", format!("{r}.kind"), Text),
                c("at", format!("{r}.at"), Time),
            ],
        ),
        Unit::Links => (
            format!("{r}.pk"),
            vec![
                c("parent", format!("{r}.parent_id"), Text),
                c("child", format!("{r}.child_id"), Text),
                c("kind", format!("{r}.kind"), Text),
                c("how", format!("{r}.how"), Text),
                c("evidence", format!("{r}.evidence"), Text),
                c("pending", format!("({r}p.pk IS NULL)"), Bool),
            ],
        ),
        Unit::Notes => (
            format!("{r}.rowid"),
            vec![
                c("id", format!("{r}.id"), Text),
                c("kind", format!("{r}.kind"), Text),
                c("key", format!("{r}.key"), Text),
                c("value", format!("{r}.value"), Text),
                c("author", format!("{r}.author"), Text),
                c("created_at", format!("{r}.created_at"), Time),
                c("root", format!("{r}.root"), Text),
                c(
                    "target",
                    format!(
                        "(SELECT group_concat(nt.ref, ' ') FROM a.note_target nt WHERE nt.note_id = {r}.id)"
                    ),
                    Text,
                ),
            ],
        ),
        Unit::Runs => (
            format!("{r}.pk"),
            vec![
                c("id", format!("{r}.id"), Text),
                c("kind", format!("{r}.kind"), Text),
                c("state", format!("{r}.state"), Text),
                c("instruction", format!("{r}.instruction"), Text),
                c("started", format!("{r}.started_at"), Time),
                c("ended", format!("{r}.ended_at"), Time),
            ],
        ),
        Unit::Processes => (
            format!("{r}.pk"),
            vec![
                c("role", format!("{r}.role"), Text),
                c("comm", format!("{r}.comm"), Text),
                c("pid", format!("{r}.pid"), Number),
                c("started_at", format!("{r}.started_at"), Time),
                c("ended_at", format!("{r}.ended_at"), Time),
            ],
        ),
        Unit::Deliveries => (
            format!("{r}.pk"),
            vec![
                c("id", format!("{r}.id"), Text),
                c("recipient", format!("{r}.recipient_id"), Text),
                c("event", format!("{r}.event"), Text),
                c("outcome", format!("{r}.outcome"), Text),
                c("at", format!("{r}.at"), Time),
                c("bytes", format!("{r}.bytes"), Number),
            ],
        ),
        Unit::Turns => ("0".into(), vec![]),
    }
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
