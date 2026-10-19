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
    let query = pairs.next().expect("query rule");
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
    Unit::from_name(p.as_str().trim()).expect("grammar lists only units")
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
            for q in s.kids() {
                match q.as_rule() {
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
        Rule::unit | Rule::row_unit => "a unit name",
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
mod tests {
    use super::*;

    fn p(text: &str) -> Query {
        parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
    }

    fn cond(text: &str) -> CondKind {
        match p(text).source {
            Source::Sessions { cond } => cond.kind,
            other => panic!("{text:?}: {other:?}"),
        }
    }

    #[test]
    fn every_spec_example_parses() {
        // spec query.md, "Examples against the Morrow archive".
        for text in [
            "origin:claude-code",
            "origin:claude-code kind:subagent",
            "kind:* | agg count, sum:cost",
            "sessions | group by origin | count",
            "repo:morrow",
            "tag:project:morrow",
            "sessions where cost > 1",
            "cost:unknown",
            "sessions where human_prompts >= 3",
            "sessions where messages >= 10",
            "sessions where tool_calls > 5",
            "messages where written_by:human \"chrono\"",
            "messages where written_by:human inherited:* \"chrono\"",
            "tool_calls where outcome:error",
            "tool_calls where outcome:error | group by session | count",
            "tool_calls | group by action | count",
            "tool_calls where tool:Bash | agg count",
            "tool_calls where session.tag:project:morrow | agg count",
            "sessions where exists tool_call(action:shell AND outcome:error)",
            "sessions where seq(command:\"cargo test\" AND outcome:error -> tool:Edit -> command:\"cargo test\" AND outcome:ok)",
            "messages where written_by:human | group by session.origin | count",
            "messages where active:false",
            "parent:claude-code:7f3a2c10-0002-4000-8000-000000000002",
            "lineage:codex:01a20000-0005-7000-8000-000000000005",
            "since:2026-10-05",
            "started between 2026-10-02 and 2026-10-03",
            "has:compaction",
            "model:claude-opus-5-5",
            "origin:(chatgpt|claude-ai)",
            "notes where kind:decision",
            "origin:codex with messages(id, role)[role:assistant, last:1]",
            "near:\"ambiguous date formats\"",
            "repo:other",
            "reop:morrow",
            "links where kind:continuation",
            "link:fork",
            "tree_cost > 2.90",
        ] {
            p(text);
        }
    }

    #[test]
    fn juxtaposition_is_and_and_capitals_are_operators() {
        let CondKind::And { all } = cond("repo:x \"a b\" refactor*") else {
            panic!()
        };
        assert_eq!(all.len(), 3);
        assert!(matches!(&all[1].kind, CondKind::Text { phrase: true, text } if text == "a b"));
        assert!(
            matches!(&all[2].kind, CondKind::Text { phrase: false, text } if text == "refactor*")
        );
        // Lower-case `and`/`or` are words to search for.
        let CondKind::And { all } = cond("cats and dogs") else {
            panic!()
        };
        assert_eq!(all.len(), 3);
        assert!(matches!(cond("a OR b"), CondKind::Or { .. }));
        assert!(matches!(cond("NOT tag:x"), CondKind::Not { .. }));
        assert!(matches!(cond("-tag:x"), CondKind::Not { .. }));
    }

    #[test]
    fn values_and_compact_comparisons() {
        assert!(matches!(
            cond("origin:(a|b|c)"),
            CondKind::Match { value: Value::Alternatives { any }, .. } if any.len() == 3
        ));
        assert!(matches!(
            cond("kind:*"),
            CondKind::Match {
                value: Value::Any,
                ..
            }
        ));
        assert!(matches!(
            cond("messages:>=10"),
            CondKind::Compare { op: Op::Ge, value, .. } if value == "10"
        ));
        assert!(matches!(
            cond("tag:project:morrow"),
            CondKind::Match { value: Value::One { text, .. }, .. } if text == "project:morrow"
        ));
        assert!(matches!(
            cond("started > last week"),
            CondKind::Compare { value, .. } if value == "last week"
        ));
    }

    #[test]
    fn references_parse_as_their_own_conditions() {
        assert!(matches!(
            cond("https://chatgpt.com/c/abc"),
            CondKind::Url { .. }
        ));
        assert!(matches!(cond("@3.2"), CondKind::Position { .. }));
    }

    #[test]
    fn a_pasted_call_is_one_search_word() {
        assert!(matches!(
            cond("weekday_offset()"),
            CondKind::Text { text, phrase: false } if text == "weekday_offset()"
        ));
        assert!(matches!(cond("parse(x) fails"), CondKind::And { all } if all.len() == 2));
        // Parentheses with spaces inside still group.
        assert!(matches!(cond("a (b OR c)"), CondKind::And { all } if all.len() == 2));
    }

    #[test]
    fn seq_records_its_gaps() {
        let q = p("tool_calls where seq(tool:Bash -> [next] tool:Edit -> [within:5m] tool:Bash)");
        let Source::Unit {
            cond:
                Some(Cond {
                    kind: CondKind::Seq { steps, gaps },
                    ..
                }),
            ..
        } = q.source
        else {
            panic!()
        };
        assert_eq!(steps.len(), 3);
        assert_eq!(gaps, vec![Gap::Next, Gap::Within { ms: 300_000 }]);
    }

    #[test]
    fn a_session_query_can_scope_a_row_query() {
        let q = p("tag:project:morrow | tool_calls where action:shell | group by tool | count");
        assert!(matches!(q.source, Source::Sessions { .. }));
        assert!(matches!(
            q.stages[0].kind,
            StageKind::Scope {
                unit: Unit::ToolCalls,
                ..
            }
        ));
        assert_eq!(q.stages.len(), 3);
    }

    #[test]
    fn a_unit_name_that_is_also_a_field_still_compares() {
        // `messages` is a unit and a session field.
        assert!(matches!(cond("messages >= 10"), CondKind::Compare { .. }));
        assert!(matches!(
            p("messages").source,
            Source::Unit {
                unit: Unit::Messages,
                cond: None
            }
        ));
    }

    #[test]
    fn stages_parse() {
        let q = p(
            "tool_calls | group by day(at), tool | agg count, sum:duration, p90:duration, count_distinct:session | sort by count desc | where count >= 3 | limit 5 | offset 10",
        );
        assert_eq!(q.stages.len(), 6);
        let StageKind::Agg { fns } = &q.stages[1].kind else {
            panic!()
        };
        assert_eq!(fns[2].column(), "p90_duration");
        assert!(matches!(
            p("sessions | sort by random(7)").stages[0].kind,
            StageKind::SortBy {
                key: SortKey::Random { seed: Some(7) },
                ..
            }
        ));
    }

    #[test]
    fn errors_point_at_the_problem_and_offer_a_repair() {
        let e = parse("repo:(a|").unwrap_err();
        let QueryError::Syntax { position, .. } = &e else {
            panic!("{e:?}")
        };
        assert!(*position >= 5);
        let QueryError::Syntax { suggestion, .. } = parse("\"unclosed phrase").unwrap_err() else {
            panic!()
        };
        assert_eq!(suggestion.as_deref(), Some("\"unclosed phrase\""));
        let QueryError::Syntax { suggestion, .. } = parse("messages where").unwrap_err() else {
            panic!()
        };
        assert_eq!(suggestion, None);
    }

    #[test]
    fn nesting_past_the_stack_bound_is_refused_by_name() {
        let deep = format!("{}a{}", "(".repeat(65), ")".repeat(65));
        assert!(matches!(parse(&deep), Err(QueryError::TooDeep { .. })));
        let ok = format!("{}a{}", "(".repeat(64), ")".repeat(64));
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn a_bare_where_after_session_conditions_is_refused_not_searched() {
        // It used to search the word "where" and answer wrongly (tuu.7).
        let QueryError::Syntax {
            position,
            suggestion,
            ..
        } = parse("kind:* where exists tool_call(outcome:error)").unwrap_err()
        else {
            panic!()
        };
        assert_eq!(position, 7);
        assert_eq!(
            suggestion.as_deref(),
            Some("kind:* exists tool_call(outcome:error)")
        );
        assert!(parse("kind:* NOT where").is_err());
        // The quoted word is searched; `where` inside a row condition, and
        // `sessions where …`, keep their meaning.
        assert!(parse("kind:* \"where\"").is_ok());
        assert!(parse("exists tool_call(where)").is_ok());
        assert!(parse("sessions where exists tool_call(outcome:error)").is_ok());
    }

    #[test]
    fn the_empty_query_is_every_session() {
        assert!(matches!(p("").source, Source::Empty));
        assert!(matches!(p("   ").source, Source::Empty));
    }
}
