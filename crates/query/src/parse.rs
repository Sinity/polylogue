//! From text to the AST. A syntax error carries the position and, when one
//! can be formed, a corrected query that parses: unbalanced quotes or
//! parentheses closed, or the offending word quoted so it is searched as
//! text.

use crate::ast::*;
use crate::{Grammar, QueryError, Rule};
use pest::Parser;
use pest::iterators::Pair;

/// Parentheses nest at most this deep: the parser is recursive, and deeper
/// nesting would exhaust its stack (spec query.md, "No limit on what may be
/// asked, one physical one").
pub const MAX_DEPTH: usize = 64;

pub fn parse(text: &str) -> Result<Query, QueryError> {
    if depth(text) > MAX_DEPTH {
        return Err(QueryError::TooDeep { max: MAX_DEPTH });
    }
    let mut pairs = Grammar::parse(Rule::query, text).map_err(|e| syntax(text, &e))?;
    let mut query = pairs.next().expect("query rule");
    if let Some(bad) = unknown_unit(query.clone()) {
        if !lone_word(query.clone(), &bad) {
            return Err(not_a_unit(text, &bad));
        }
        query = Grammar::parse(Rule::conditions_query, text)
            .map_err(|e| syntax(text, &e))?
            .next()
            .expect("conditions_query rule");
        if let Some(bad) = unknown_unit(query.clone()) {
            return Err(not_a_unit(text, &bad));
        }
    }
    let Some(shape) = query.kids().find(|p| p.as_rule() != Rule::EOI) else {
        return Ok(Query {
            source: Source::Empty,
            attach: vec![],
            stages: vec![],
        });
    };
    match shape.as_rule() {
        Rule::unit_query => {
            let mut inner = shape.kids();
            let unit = unit(inner.next().expect("unit"));
            let mut cond = None;
            let mut attach = vec![];
            let mut stages = vec![];
            for p in inner {
                match p.as_rule() {
                    Rule::expr => cond = Some(expr(p)),
                    Rule::attach => attach = children(p),
                    Rule::stages => stages = stage_list(p),
                    _ => {}
                }
            }
            Ok(Query {
                source: Source::Unit { unit, cond },
                attach,
                stages,
            })
        }
        Rule::expr_query => {
            if let Some((start, end)) = shape.clone().kids().next().and_then(bare_where) {
                // `kind:* where exists …` would search for the word "where"
                // and answer wrongly; refuse it with the reading meant.
                let joined = format!("{} {}", text[..start].trim_end(), text[end..].trim_start());
                return Err(QueryError::syntax(
                    text,
                    start,
                    "`where` follows a unit (`sessions where …`); conditions on sessions are \
                     joined directly or with AND, and \"where\" in quotes searches the word"
                        .to_owned(),
                    Some(joined.trim().to_owned()),
                ));
            }
            let mut inner = shape.kids();
            let cond = expr(inner.next().expect("expr"));
            let mut attach = vec![];
            let mut stages = vec![];
            for p in inner {
                match p.as_rule() {
                    Rule::attach => attach = children(p),
                    Rule::stages => stages = stage_list(p),
                    _ => {}
                }
            }
            Ok(Query {
                source: Source::Sessions { cond },
                attach,
                stages,
            })
        }
        other => unreachable!("query shape {other:?}"),
    }
}

/// The deepest parenthesis nesting outside quotes.
fn depth(text: &str) -> usize {
    let (mut depth, mut max, mut quoted, mut escaped) = (0usize, 0usize, false, false);
    for c in text.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            '(' if !quoted => {
                depth += 1;
                max = max.max(depth);
            }
            ')' if !quoted => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    max
}

/// A pair's children without the keyword tokens. Keywords are atomic
/// rules so their word boundary sees no implicit whitespace, which puts
/// them in the tree; nothing reads them.
type KidsIter<'i> =
    std::iter::Filter<pest::iterators::Pairs<'i, Rule>, fn(&Pair<'i, Rule>) -> bool>;

trait Kids<'i> {
    fn kids(self) -> KidsIter<'i>;
}

impl<'i> Kids<'i> for Pair<'i, Rule> {
    fn kids(self) -> KidsIter<'i> {
        fn keep(p: &Pair<'_, Rule>) -> bool {
            !matches!(
                p.as_rule(),
                Rule::kw_where
                    | Rule::kw_between
                    | Rule::kw_and_lc
                    | Rule::kw_with
                    | Rule::kw_AND
                    | Rule::kw_OR
                    | Rule::kw_NOT
                    | Rule::kw_exists
                    | Rule::kw_seq
            )
        }
        self.into_inner()
            .filter(keep as fn(&Pair<'i, Rule>) -> bool)
    }
}

fn span(p: &Pair<'_, Rule>) -> Span {
    let s = p.as_span();
    Span {
        start: s.start(),
        end: s.end(),
    }
}

fn unit(p: Pair<'_, Rule>) -> Unit {
    Unit::from_name(p.as_str().trim()).expect("unit names are resolved before the tree is read")
}

/// The first unit name in the tree that names no unit.
fn unknown_unit(p: Pair<'_, Rule>) -> Option<Pair<'_, Rule>> {
    p.into_inner()
        .flatten()
        .find(|q| q.as_rule() == Rule::unit && Unit::from_name(q.as_str().trim()).is_none())
}

/// Whether `bad` is the first word of a query with no `where` after it, so
/// the word is meant as text (`crash`, `crash | count`).
fn lone_word(query: Pair<'_, Rule>, bad: &Pair<'_, Rule>) -> bool {
    let Some(shape) = query.kids().find(|p| p.as_rule() == Rule::unit_query) else {
        return false;
    };
    let mut kids = shape.kids();
    kids.next().is_some_and(|u| u.as_span() == bad.as_span())
        && kids.all(|k| k.as_rule() != Rule::expr)
}

fn not_a_unit(text: &str, bad: &Pair<'_, Rule>) -> QueryError {
    let word = bad.as_str().trim();
    let lower = word.to_ascii_lowercase();
    let near = Unit::all()
        .flat_map(|u| [u.name(), u.singular()])
        .map(|n| (strsim::damerau_levenshtein(&lower, n), n))
        .filter(|(d, n)| *d <= 2.max(n.len() / 4))
        .min_by_key(|(d, _)| *d);
    let start = bad.as_span().start();
    let names: Vec<&str> = Unit::all().map(Unit::name).collect();
    QueryError::syntax(
        text,
        start,
        format!("{word:?} is not a unit; the units are {}", names.join(", ")),
        near.map(|(_, n)| format!("{}{n}{}", &text[..start], &text[bad.as_span().end()..])),
    )
}

fn number(p: Pair<'_, Rule>) -> u64 {
    // The grammar admits only digits; a value too large for u64 saturates,
    // which is still "more rows than exist".
    p.as_str().parse().unwrap_or(u64::MAX)
}

fn unquote(p: Pair<'_, Rule>) -> String {
    let inner = p.kids().next().map(|i| i.as_str()).unwrap_or("");
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(n);
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn expr(p: Pair<'_, Rule>) -> Cond {
    debug_assert_eq!(p.as_rule(), Rule::expr);
    or_expr(p.kids().next().expect("or_expr"))
}

fn or_expr(p: Pair<'_, Rule>) -> Cond {
    let sp = span(&p);
    let mut any: Vec<Cond> = p.kids().map(and_expr).collect();
    if any.len() == 1 {
        return any.pop().expect("one");
    }
    Cond {
        span: sp,
        kind: CondKind::Or { any },
    }
}

fn and_expr(p: Pair<'_, Rule>) -> Cond {
    let sp = span(&p);
    let mut all: Vec<Cond> = p.kids().map(not_expr).collect();
    if all.len() == 1 {
        return all.pop().expect("one");
    }
    Cond {
        span: sp,
        kind: CondKind::And { all },
    }
}

fn not_expr(p: Pair<'_, Rule>) -> Cond {
    let sp = span(&p);
    let mut inner = p.kids();
    let first = inner.next().expect("not_expr");
    match first.as_rule() {
        Rule::negate => Cond {
            span: sp,
            kind: CondKind::Not {
                cond: Box::new(not_expr(inner.next().expect("negated"))),
            },
        },
        Rule::atom => atom(first),
        other => unreachable!("not_expr {other:?}"),
    }
}

fn atom(p: Pair<'_, Rule>) -> Cond {
    let inner = p.kids().next().expect("atom");
    leaf(inner)
}

fn leaf(p: Pair<'_, Rule>) -> Cond {
    let sp = span(&p);
    let kind = match p.as_rule() {
        Rule::expr => return expr(p),
        Rule::exists => {
            let mut i = p.kids();
            let unit = unit(i.next().expect("row unit"));
            let cond = expr(i.next().expect("exists body"));
            CondKind::Exists {
                unit,
                cond: Box::new(cond),
            }
        }
        Rule::seq => {
            let mut steps = vec![];
            let mut gaps = vec![];
            for q in p.kids() {
                match q.as_rule() {
                    Rule::step => steps.push(step(q)),
                    Rule::arrow => gaps.push(gap(q)),
                    _ => {}
                }
            }
            CondKind::Seq { steps, gaps }
        }
        Rule::between => {
            let mut i = p.kids();
            CondKind::Between {
                field: i.next().expect("field").as_str().to_owned(),
                from: scalar(i.next().expect("from")),
                to: scalar(i.next().expect("to")),
            }
        }
        Rule::compare => {
            let mut i = p.kids();
            let field = i.next().expect("field").as_str().to_owned();
            let op = Op::parse(i.next().expect("op").as_str()).expect("grammar op");
            CondKind::Compare {
                field,
                op,
                value: scalar(i.next().expect("value")),
            }
        }
        Rule::match_ => {
            let mut i = p.kids();
            let field = i.next().expect("field").as_str().to_owned();
            let v = i.next().expect("mvalue").kids().next().expect("value");
            match v.as_rule() {
                Rule::quoted => CondKind::Match {
                    field,
                    value: Value::One {
                        text: unquote(v),
                        quoted: true,
                    },
                },
                Rule::alternatives => CondKind::Match {
                    field,
                    value: Value::Alternatives {
                        any: v.kids().map(|b| b.as_str().to_owned()).collect(),
                    },
                },
                Rule::any_value => CondKind::Match {
                    field,
                    value: Value::Any,
                },
                Rule::bare => {
                    let text = v.as_str();
                    // The compact comparison `messages:>=10`.
                    match compact_compare(text) {
                        Some((op, value)) => CondKind::Compare {
                            field,
                            op,
                            value: value.to_owned(),
                        },
                        None => CondKind::Match {
                            field,
                            value: Value::One {
                                text: text.to_owned(),
                                quoted: false,
                            },
                        },
                    }
                }
                other => unreachable!("mvalue {other:?}"),
            }
        }
        Rule::url => CondKind::Url {
            url: p.as_str().to_owned(),
        },
        Rule::position_ref => CondKind::Position {
            text: p.as_str().to_owned(),
        },
        Rule::text => {
            let t = p.kids().next().expect("text");
            match t.as_rule() {
                Rule::quoted => CondKind::Text {
                    text: unquote(t),
                    phrase: true,
                },
                _ => CondKind::Text {
                    text: t.as_str().to_owned(),
                    phrase: false,
                },
            }
        }
        Rule::step_atom | Rule::atom => return leaf(p.kids().next().expect("inner")),
        other => unreachable!("leaf {other:?}"),
    };
    Cond { span: sp, kind }
}

fn compact_compare(text: &str) -> Option<(Op, &str)> {
    for op in [">=", "<=", "!=", ">", "<", "="] {
        if let Some(rest) = text.strip_prefix(op)
            && !rest.is_empty()
        {
            return Some((Op::parse(op).expect("listed"), rest));
        }
    }
    None
}

fn scalar(p: Pair<'_, Rule>) -> String {
    let inner = p.kids().next().expect("scalar");
    match inner.as_rule() {
        Rule::quoted => unquote(inner),
        Rule::time_phrase => inner
            .as_str()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase(),
        _ => inner.as_str().to_owned(),
    }
}

fn step(p: Pair<'_, Rule>) -> Cond {
    let sp = span(&p);
    let mut all: Vec<Cond> = p.kids().map(leaf).collect();
    if all.len() == 1 {
        return all.pop().expect("one");
    }
    Cond {
        span: sp,
        kind: CondKind::And { all },
    }
}

fn gap(p: Pair<'_, Rule>) -> Gap {
    let Some(g) = p.kids().next() else {
        return Gap::Later;
    };
    let g = g.kids().next().expect("gap kind");
    match g.as_rule() {
        Rule::gap_next => Gap::Next,
        Rule::gap_within => {
            let d = g.kids().next().expect("duration").as_str();
            Gap::Within {
                ms: crate::time::duration_ms(d).expect("grammar duration"),
            }
        }
        other => unreachable!("gap {other:?}"),
    }
}

fn children(p: Pair<'_, Rule>) -> Vec<Child> {
    p.kids()
        .filter(|c| c.as_rule() == Rule::child)
        .map(|c| {
            let sp = span(&c);
            let mut i = c.kids();
            let unit = unit(i.next().expect("unit"));
            let mut fields = vec![];
            let mut windows = vec![];
            for q in i {
                match q.as_rule() {
                    Rule::field => fields.push(q.as_str().to_owned()),
                    Rule::window => {
                        let w = q.kids().next().expect("window");
                        windows.push(match w.as_rule() {
                            Rule::window_n => {
                                let first = w.as_str().to_ascii_lowercase().starts_with("first");
                                let n = number(w.kids().next().expect("n"));
                                if first {
                                    Window::First { n }
                                } else {
                                    Window::Last { n }
                                }
                            }
                            _ => Window::Cond { cond: expr(w) },
                        });
                    }
                    _ => {}
                }
            }
            Child {
                span: sp,
                unit,
                fields,
                windows,
            }
        })
        .collect()
}

fn stage_list(p: Pair<'_, Rule>) -> Vec<Stage> {
    p.kids().map(stage).collect()
}

fn stage(p: Pair<'_, Rule>) -> Stage {
    let sp = span(&p);
    let s = p.kids().next().expect("stage");
    let kind = match s.as_rule() {
        Rule::sort_by => {
            let mut descending = false;
            let mut key = None;
            let mut then = vec![];
            for q in s.kids() {
                match q.as_rule() {
                    Rule::sort_then => {
                        let mut i = q.kids();
                        let field = i.next().expect("field").as_str().to_owned();
                        let desc = i
                            .next()
                            .is_some_and(|d| d.as_str().eq_ignore_ascii_case("desc"));
                        then.push((field, desc));
                    }
                    Rule::sort_key => {
                        let k = q.kids().next().expect("sort key");
                        key = Some(match k.as_rule() {
                            Rule::random => SortKey::Random {
                                seed: k.kids().next().map(number),
                            },
                            _ => SortKey::Field {
                                field: k.as_str().to_owned(),
                            },
                        });
                    }
                    Rule::direction => {
                        descending = q.as_str().eq_ignore_ascii_case("desc");
                    }
                    _ => {}
                }
            }
            StageKind::SortBy {
                key: key.expect("sort key"),
                descending,
                then,
            }
        }
        Rule::group_by => StageKind::GroupBy {
            keys: s
                .kids()
                .map(|k| {
                    let mut i = k.kids();
                    let first = i.next().expect("group key");
                    match first.as_rule() {
                        Rule::bucket => GroupKey {
                            bucket: Some(match first.as_str().to_ascii_lowercase().as_str() {
                                "day" => Bucket::Day,
                                "week" => Bucket::Week,
                                _ => Bucket::Month,
                            }),
                            field: i.next().expect("field").as_str().to_owned(),
                        },
                        _ => GroupKey {
                            field: first.as_str().to_owned(),
                            bucket: None,
                        },
                    }
                })
                .collect(),
        },
        Rule::agg => StageKind::Agg {
            fns: s.kids().map(agg_fn).collect(),
        },
        Rule::count => StageKind::Count,
        Rule::limit => StageKind::Limit {
            n: number(s.kids().next().expect("n")),
        },
        Rule::offset => StageKind::Offset {
            n: number(s.kids().next().expect("n")),
        },
        Rule::where_stage => StageKind::Where {
            cond: expr(s.kids().next().expect("expr")),
        },
        Rule::distinct => StageKind::Distinct,
        Rule::matches => StageKind::Matches,
        Rule::measure => StageKind::Measure {
            what: s
                .kids()
                .next()
                .expect("measure_what")
                .as_str()
                .to_ascii_lowercase(),
        },
        Rule::scope_stage => {
            let mut i = s.kids();
            let unit = unit(i.next().expect("unit"));
            StageKind::Scope {
                unit,
                cond: i.next().map(expr),
            }
        }
        other => unreachable!("stage {other:?}"),
    };
    Stage { span: sp, kind }
}

fn agg_fn(p: Pair<'_, Rule>) -> AggFn {
    let mut i = p.kids();
    let Some(name) = i.next() else {
        return AggFn::Count;
    };
    let field = i.next().expect("agg field").as_str().to_owned();
    let n = name.as_str().to_ascii_lowercase();
    match n.as_str() {
        "sum" => AggFn::Sum { field },
        "avg" => AggFn::Avg { field },
        "min" => AggFn::Min { field },
        "max" => AggFn::Max { field },
        "count_distinct" => AggFn::CountDistinct { field },
        p => AggFn::Percentile {
            p: p[1..].parse::<u8>().unwrap_or(100).min(100),
            field,
        },
    }
}

/// The syntax error, with a corrected query when a simple repair parses.
/// The span of an unquoted `where` read as a search word among session
/// conditions.
fn bare_where(p: Pair<'_, Rule>) -> Option<(usize, usize)> {
    // Row conditions inside `exists`/`seq` are another scope.
    if matches!(p.as_rule(), Rule::exists | Rule::seq) {
        return None;
    }
    if p.as_rule() == Rule::text {
        let t = p.clone().into_inner().next()?;
        if t.as_rule() != Rule::quoted && t.as_str().eq_ignore_ascii_case("where") {
            let s = t.as_span();
            return Some((s.start(), s.end()));
        }
        return None;
    }
    p.into_inner().find_map(bare_where)
}

fn syntax(text: &str, e: &pest::error::Error<Rule>) -> QueryError {
    let at = match e.location {
        pest::error::InputLocation::Pos(p) => p,
        pest::error::InputLocation::Span((s, _)) => s,
    };
    let suggestion = repair(text, at);
    QueryError::syntax(text, at, expected(e), suggestion)
}

fn expected(e: &pest::error::Error<Rule>) -> String {
    match &e.variant {
        pest::error::ErrorVariant::ParsingError { positives, .. } => {
            let mut names: Vec<&str> = positives.iter().filter_map(|r| rule_name(*r)).collect();
            names.sort_unstable();
            names.dedup();
            match names.as_slice() {
                [] => "unexpected text".into(),
                [one] => format!("expected {one}"),
                many => format!("expected one of: {}", many.join(", ")),
            }
        }
        pest::error::ErrorVariant::CustomError { message } => message.clone(),
    }
}

fn rule_name(r: Rule) -> Option<&'static str> {
    Some(match r {
        Rule::expr | Rule::atom | Rule::not_expr => "a condition",
        Rule::field => "a field name",
        Rule::mvalue | Rule::bare | Rule::scalar => "a value",
        Rule::quoted | Rule::inner => "a closing quote",
        Rule::number => "a number",
        Rule::unit => "a unit name",
        Rule::stage => "a stage",
        Rule::step | Rule::step_atom => "a step",
        Rule::duration => "a duration such as 5m",
        Rule::EOI => "the end of the query",
        _ => return None,
    })
}

/// Simple repairs, tried in order; the first that parses is suggested.
fn repair(text: &str, at: usize) -> Option<String> {
    let mut candidates = vec![];
    if unbalanced_quotes(text) {
        candidates.push(format!("{text}\""));
    }
    let (open, close) = parens(text);
    if open > close {
        candidates.push(format!("{text}{}", ")".repeat(open - close)));
    }
    if let Some(quoted) = quote_word_at(text, at) {
        candidates.push(quoted);
    }
    candidates
        .into_iter()
        .find(|c| Grammar::parse(Rule::query, c).is_ok())
}

fn unbalanced_quotes(text: &str) -> bool {
    let mut n = 0;
    let mut escaped = false;
    for c in text.chars() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            n += 1;
        }
    }
    n % 2 == 1
}

fn parens(text: &str) -> (usize, usize) {
    (text.matches('(').count(), text.matches(')').count())
}

/// The query with the whitespace-delimited word around `at` quoted.
fn quote_word_at(text: &str, at: usize) -> Option<String> {
    let at = at.min(text.len());
    let start = text[..at].rfind(char::is_whitespace).map_or(0, |i| i + 1);
    let end = text[at..]
        .find(char::is_whitespace)
        .map_or(text.len(), |i| at + i);
    let word = &text[start..end];
    // Quoting a keyword would turn a broken query into a search for it.
    let keyword = ["where", "with", "between", "and"]
        .iter()
        .any(|k| word.eq_ignore_ascii_case(k))
        || ["AND", "OR", "NOT"].contains(&word);
    if word.is_empty() || word.contains('"') || keyword {
        return None;
    }
    Some(format!("{}\"{}\"{}", &text[..start], word, &text[end..]))
}

#[cfg(test)]
mod tests;
