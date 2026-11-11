//! The conditions that select a unit's rows, bound to the catalog: each
//! field condition names its declared field ([`FieldId`]) and holds its
//! value in the form the store keeps (nano-dollars, milliseconds, times
//! resolved against the query's clock, closed sets in their declared
//! spelling), so lowering is a match on the field and never reads the
//! value's spelling again (docs/query.md: Fields, Times). Conditions over
//! other rows (exists, count, membership, saved queries, sequences) and
//! computed comparisons stay as written; the store binds their inner
//! conditions when it lowers them.
use crate::{
    QueryError,
    ast::*,
    fields::{self, Field, FieldId, Kind},
    time::{Clock, ResolvedTime},
};

/// What binding needs from the query: its text (for diagnostics), the
/// origin names a reference may start with, and the clock times resolve
/// against.
#[derive(Clone, Copy)]
pub struct Binder<'a> {
    pub text: &'a str,
    pub origins: &'a [String],
    pub clock: &'a Clock,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Every condition holds. Search words directly inside one `All`
    /// occur in the same message.
    All(Vec<Filter>),
    Any(Vec<Filter>),
    Not(Box<Filter>),
    /// A search word, a quoted phrase or a prefix (`retr*`).
    Word(Word),
    /// A word naming a session: `last`, `latest`, `self`, or a pasted ID.
    SessionWord {
        word: String,
        span: Span,
    },
    /// `origin:native` or `session:ID`: one session, or the unique one
    /// whose ID starts with it.
    Session {
        id: String,
        span: Span,
    },
    /// A provider conversation URL.
    Url {
        url: String,
        span: Span,
    },
    /// A reference to rows by their natural key (`block:…`, `message:a..b`).
    Rows {
        reference: String,
        span: Span,
    },
    /// A condition on a field of a related unit (`session.repo:x`).
    Related {
        prefix: String,
        target: Unit,
        filter: Box<Filter>,
        span: Span,
    },
    Field(FieldTest),
    /// A condition the store lowers as written: one over other rows or
    /// another query, or a computed comparison.
    Written(Condition),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// The searched text: a prefix keeps its trailing `*`.
    pub text: String,
    pub phrase: bool,
    pub span: Span,
}

/// One condition on one declared field.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldTest {
    pub field: FieldId,
    pub test: Test,
    pub span: Span,
}

/// What a field condition asks of the field's value, in stored form.
#[derive(Debug, Clone, PartialEq)]
pub enum Test {
    /// `F:*`: names the field without restricting it.
    Any,
    Unknown,
    Known,
    /// One of these whole values: a closed set's in its declared spelling,
    /// others in lowercase. `unknown` also matches an unknown value, for a
    /// field that stores a word for it.
    OneOf {
        values: Vec<String>,
        unknown: bool,
    },
    /// Text compared in lowercase.
    TextCompare(Op, String),
    /// Each an ID or a unique prefix of one.
    Prefix(Vec<String>),
    /// An ID equal or not equal to this one.
    PrefixCompare(Op, String),
    Substring(Vec<String>),
    /// Each a path and the directories below it.
    Subtree(Vec<String>),
    Bool(bool),
    /// A number in stored form: nano-dollars for amounts, milliseconds for
    /// durations.
    NumberEquals(String),
    NumberCompare(Op, String),
    NumberBetween(String, String),
    Time(TimeTest),
    /// A field with its own rule, which reads the value as written.
    Written(Comparison),
}

impl Test {
    /// Written as a comparison (`>`, `between`) rather than `F:VALUE`.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Test::TextCompare(..)
                | Test::PrefixCompare(..)
                | Test::NumberCompare(..)
                | Test::NumberBetween(..)
                | Test::Time(TimeTest::Compare(..) | TimeTest::Between(..))
                | Test::Written(Comparison::Compare(..) | Comparison::Between(..))
        )
    }
}

/// A time condition, its times resolved against the query's clock.
#[derive(Debug, Clone, PartialEq)]
pub enum TimeTest {
    /// `F:T`: within T.
    Within(ResolvedTime),
    Compare(Op, ResolvedTime),
    Between(ResolvedTime, ResolvedTime),
}

/// How a condition compares a field with its value, as written.
#[derive(Debug, Clone, PartialEq)]
pub enum Comparison {
    Match(Value),
    Compare(Op, String),
    Between(String, String),
}

impl Binder<'_> {
    /// `condition` over the rows of `unit`.
    pub fn bind(&self, c: &Condition, unit: Unit) -> Result<Filter, QueryError> {
        query_language::depth::deeper(|| {
            Ok(match &c.kind {
                ConditionKind::And { conditions } => Filter::All(
                    conditions
                        .iter()
                        .map(|c| self.bind(c, unit))
                        .collect::<Result<_, _>>()?,
                ),
                ConditionKind::Or { conditions } => Filter::Any(
                    conditions
                        .iter()
                        .map(|c| self.bind(c, unit))
                        .collect::<Result<_, _>>()?,
                ),
                ConditionKind::Not { condition } => {
                    Filter::Not(Box::new(self.bind(condition, unit)?))
                }
                ConditionKind::Text { text, mode } => {
                    if crate::defaults::is_reference(c, unit, self.origins) {
                        Filter::SessionWord {
                            word: text.clone(),
                            span: c.span,
                        }
                    } else {
                        Filter::Word(Word {
                            text: search_word(text, *mode)?,
                            phrase: *mode == TextMode::Phrase,
                            span: c.span,
                        })
                    }
                }
                ConditionKind::Reference { reference } => {
                    self.reference(reference, unit, c.span)?
                }
                ConditionKind::Match { field, value } => {
                    self.field(unit, field, Comparison::Match(value.clone()), c.span)?
                }
                ConditionKind::Compare {
                    left: Expression::Field { name },
                    op,
                    right:
                        Expression::Value {
                            value: Value::Scalar { text, .. },
                        },
                } if !matches!(op, Op::Contains | Op::Prefix) && compares_as_field(unit, name) => {
                    self.field(unit, name, Comparison::Compare(*op, text.clone()), c.span)?
                }
                ConditionKind::Since { value, .. } => {
                    self.field(unit, "since", Comparison::Match(value.clone()), c.span)?
                }
                ConditionKind::Until { value, .. } => {
                    self.field(unit, "until", Comparison::Match(value.clone()), c.span)?
                }
                ConditionKind::Between {
                    left: Expression::Field { name },
                    from,
                    to,
                } => self.field(
                    unit,
                    name,
                    Comparison::Between(value_text(from)?.into(), value_text(to)?.into()),
                    c.span,
                )?,
                ConditionKind::Rank {
                    list: RankList::Similar,
                    value,
                } => self.field(unit, "similar", Comparison::Match(value.clone()), c.span)?,
                ConditionKind::Sequence {
                    unit: rows,
                    all: false,
                    ..
                } if bind_unit(rows)? != Unit::ToolCalls => {
                    return Err(not_built("this shared condition"));
                }
                ConditionKind::Exists { .. }
                | ConditionKind::Count { .. }
                | ConditionKind::In { .. }
                | ConditionKind::Saved { .. }
                | ConditionKind::Sequence { all: false, .. }
                | ConditionKind::Compare { .. }
                | ConditionKind::Between { .. }
                | ConditionKind::Order { .. } => Filter::Written(c.clone()),
                _ => return Err(not_built("this shared condition")),
            })
        })
    }

    fn reference(&self, reference: &str, unit: Unit, span: Span) -> Result<Filter, QueryError> {
        let kind = reference.split_once(':').map(|(kind, _)| kind);
        if matches!(
            kind,
            Some("block" | "document" | "event" | "span" | "assertion" | "delivery")
        ) || (reference.starts_with("message:") && reference.contains(".."))
        {
            return Ok(Filter::Rows {
                reference: reference.into(),
                span,
            });
        }
        // Membership in a kept result is designed and has no producer yet.
        if reference.starts_with("in:result:") {
            return Err(not_built("in:result: conditions"));
        }
        // Pages and commits are designed reference forms with no producer yet.
        if let Some(kind @ ("page" | "file")) = kind {
            return Err(not_built(&format!("{kind} references")));
        }
        if reference.starts_with("http") {
            Ok(Filter::Url {
                url: reference.into(),
                span,
            })
        } else if reference.starts_with('@') {
            Err(QueryError::Invalid {
                message: "@N requires an interactive listing context".into(),
                position: span.start,
            })
        } else if let Some((field, text)) = reference.split_once(':') {
            self.field(unit, field, Comparison::Match(Value::scalar(text)), span)
        } else {
            Ok(Filter::SessionWord {
                word: reference.into(),
                span,
            })
        }
    }

    /// A condition on a field: plain, related (`session.repo`), a field of
    /// the row's session, or a reference written as `kind:value`.
    pub fn field(
        &self,
        unit: Unit,
        field: &str,
        comparison: Comparison,
        span: Span,
    ) -> Result<Filter, QueryError> {
        let lower = field.to_ascii_lowercase();
        if let Comparison::Match(Value::Scalar { text, .. }) = &comparison {
            let rows = |reference: String| Filter::Rows { reference, span };
            if (lower == "message" && text.contains(".."))
                || (lower == "event" && unit != Unit::Deliveries)
                || matches!(
                    lower.as_str(),
                    "block" | "document" | "span" | "assertion" | "delivery"
                )
            {
                return Ok(rows(format!("{lower}:{text}")));
            }
            let invalid = |error: String| QueryError::Invalid {
                message: error,
                position: span.start,
            };
            if lower == "request" {
                polylogue_model::Reference::parse(&format!("request:{text}"))
                    .map_err(|error| invalid(error.to_string()))?;
                return Err(not_built(
                    "accepted request objects have no query row-domain producer",
                ));
            }
            if lower == "run" {
                polylogue_model::RunId::parse(text).map_err(|error| invalid(error.to_string()))?;
                return Err(not_built(
                    "run references require their acknowledged execution binding",
                ));
            }
            if self.origins.contains(&lower) {
                return Ok(Filter::Session {
                    id: format!("{lower}:{text}"),
                    span,
                });
            }
            match lower.as_str() {
                "session" => {
                    return Ok(Filter::Session {
                        id: text.clone(),
                        span,
                    });
                }
                "message" => return Ok(rows(format!("message:{text}"))),
                "result" => {
                    return Err(invalid(
                        "result:<id> is a source: start the query with it \
                         (`result:<id> | count`); it cannot be a condition"
                            .into(),
                    ));
                }
                "page" => return Err(not_built("page: references (pages do not exist yet)")),
                "file" => {
                    return Err(not_built(
                        "file:<path>@<commit> references (commits are not a unit yet)",
                    ));
                }
                _ => {}
            }
        }
        if let Some((prefix, rest)) = lower.split_once('.')
            && let Some(target) = fields::relation(unit, prefix)
        {
            let inner = Span {
                start: span.start + prefix.len() + 1,
                end: span.end,
            };
            return Ok(Filter::Related {
                prefix: prefix.into(),
                target,
                filter: Box::new(self.field(target, rest, comparison, inner)?),
                span,
            });
        }
        let Some(id) = fields::lookup(unit, &lower).or_else(|| {
            fields::has_session(unit)
                .then(|| fields::lookup(Unit::Sessions, &lower))
                .flatten()
        }) else {
            return Err(unknown_field(self.text, field, unit, span));
        };
        let def = id.declaration();
        let test = match comparison {
            Comparison::Match(Value::Any) => Test::Any,
            comparison if def.kind != Kind::Special => {
                self.test(def, def.kind, &comparison, span)?
            }
            comparison if matches!(def.name, "since" | "until" | "evidence_until") => {
                let Comparison::Match(Value::Scalar { text, .. }) = &comparison else {
                    return Err(QueryError::Invalid {
                        message: format!("{0} takes {0}:VALUE", def.name),
                        position: span.start,
                    });
                };
                Test::Time(TimeTest::Within(self.time(text, span)?))
            }
            comparison => Test::Written(comparison),
        };
        Ok(Filter::Field(FieldTest {
            field: id,
            test,
            span,
        }))
    }

    /// `comparison` on a value of `def` that compares as `kind`.
    pub fn test(
        &self,
        def: &Field,
        kind: Kind,
        comparison: &Comparison,
        span: Span,
    ) -> Result<Test, QueryError> {
        let name = def.name;
        let bad = |value: &str, message: String| QueryError::BadValue {
            field: name.to_owned(),
            value: value.to_owned(),
            position: span.start,
            message,
        };
        let scalar = |text: &str| {
            def.kind
                .scalar(name, text)
                .map_err(|message| bad(text, message))
        };
        use Comparison::{Between, Compare, Match};
        Ok(match (kind, comparison) {
            (_, Match(Value::Any)) => Test::Any,
            (_, Match(Value::Unknown)) => Test::Unknown,
            (_, Match(Value::Known)) => Test::Known,
            (
                _,
                Match(Value::Scalar {
                    text,
                    quoted: false,
                }),
            ) if text.eq_ignore_ascii_case("unknown") => Test::Unknown,
            (Kind::Match, Match(value)) => {
                let mut values = alternatives(value);
                if let Some(set) = def.values {
                    for value in &mut values {
                        let Some(declared) = set.iter().find(|s| s.eq_ignore_ascii_case(value))
                        else {
                            return Err(bad(value, closed_set_message(def, value)));
                        };
                        *value = (*declared).to_owned();
                    }
                } else {
                    for value in &mut values {
                        *value = value.to_lowercase();
                    }
                }
                let stores_unknown = def.stored_unknown.is_some();
                let before = values.len();
                values.retain(|value| !(stores_unknown && value.eq_ignore_ascii_case("unknown")));
                Test::OneOf {
                    unknown: values.len() != before,
                    values,
                }
            }
            (Kind::Match, Compare(op, value)) => Test::TextCompare(*op, value.to_lowercase()),
            (Kind::Prefix, Compare(op @ (Op::Eq | Op::Ne), value)) => {
                Test::PrefixCompare(*op, value.clone())
            }
            (Kind::Prefix, Match(value)) => Test::Prefix(alternatives(value)),
            (Kind::Substring, Match(value)) => Test::Substring(alternatives(value)),
            (Kind::Subtree, Match(value)) => Test::Subtree(alternatives(value)),
            (Kind::Bool, Match(Value::Scalar { text, .. })) => {
                Test::Bool(match text.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => true,
                    "false" | "no" | "0" => false,
                    _ => return Err(bad(text, format!("{name} is true or false"))),
                })
            }
            (Kind::Number | Kind::Money | Kind::Duration, Match(Value::Scalar { text, .. })) => {
                Test::NumberEquals(scalar(text)?)
            }
            (Kind::Number | Kind::Money | Kind::Duration, Compare(op, value)) => {
                Test::NumberCompare(*op, scalar(value)?)
            }
            (Kind::Number | Kind::Money | Kind::Duration, Between(from, to)) => {
                Test::NumberBetween(scalar(from)?, scalar(to)?)
            }
            (Kind::Time, Match(Value::Scalar { text, .. })) => {
                Test::Time(TimeTest::Within(self.time(text, span)?))
            }
            (Kind::Time, Compare(op, value)) => {
                let at = self.time(value, span)?;
                if matches!(op, Op::Contains | Op::Prefix) {
                    return Err(not_built("this time comparison operator"));
                }
                Test::Time(TimeTest::Compare(*op, at))
            }
            (Kind::Time, Between(from, to)) => Test::Time(TimeTest::Between(
                self.time(from, span)?,
                self.time(to, span)?,
            )),
            (kind, comparison) => {
                return Err(QueryError::Invalid {
                    message: format!(
                        "{name} is a {} field; {}",
                        kind_name(kind),
                        match comparison {
                            Match(_) => "it takes a comparison such as `>`".to_owned(),
                            _ => format!("write {name}:VALUE"),
                        }
                    ),
                    position: span.start,
                });
            }
        })
    }

    /// A time expression resolved against the query's clock.
    pub fn time(&self, text: &str, span: Span) -> Result<ResolvedTime, QueryError> {
        crate::time::resolve(text, self.clock).map_err(|mut error| {
            error.position = span.start;
            QueryError::language(error, self.text)
        })
    }
}

/// A comparison of a field with a literal compares as that field when the
/// field is a number, amount, duration or time, or names a time through a
/// relation; otherwise it is a computed comparison.
fn compares_as_field(unit: Unit, name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let numeric = fields::lookup(unit, &lower)
        .or_else(|| {
            fields::has_session(unit)
                .then(|| fields::lookup(Unit::Sessions, &lower))
                .flatten()
        })
        .is_some_and(|id| {
            matches!(
                id.declaration().kind,
                Kind::Number | Kind::Money | Kind::Duration | Kind::Time
            )
        });
    numeric || names_time(unit, &lower)
}

/// The field `name` reaches from `unit` is a time column.
fn names_time(unit: Unit, name: &str) -> bool {
    if name == "session" && fields::has_session(unit) {
        return false;
    }
    if let Some((prefix, rest)) = name.split_once('.')
        && let Some(target) = fields::relation(unit, prefix)
    {
        return names_time(target, rest);
    }
    fields::lookup(unit, name)
        .or_else(|| {
            fields::has_session(unit)
                .then(|| fields::lookup(Unit::Sessions, name))
                .flatten()
        })
        .is_some_and(|id| {
            let def = id.declaration();
            def.kind != Kind::Special && def.column_kind() == crate::ColKind::Time
        })
}

fn search_word(text: &str, mode: TextMode) -> Result<String, QueryError> {
    match mode {
        TextMode::Prefix => Ok(format!("{text}*")),
        TextMode::Word | TextMode::Phrase => Ok(text.to_owned()),
        TextMode::Approximate => Err(not_built("approximate text search")),
    }
}

/// The values a condition lists: a set's members, the shared language's
/// `unknown` and `known` by name, a parameter by its `$name`.
pub fn alternatives(v: &Value) -> Vec<String> {
    match v {
        Value::Scalar { text, .. } => vec![text.clone()],
        Value::Set { values } => values.iter().flat_map(alternatives).collect(),
        Value::Unknown => vec!["unknown".into()],
        Value::Known => vec!["known".into()],
        Value::Parameter { name } => vec![format!("${name}")],
        Value::Any => vec![],
    }
}

/// The one value in a scalar position.
pub fn value_text(value: &Value) -> Result<&str, QueryError> {
    match value {
        Value::Scalar { text, .. } => Ok(text),
        Value::Unknown => Ok("unknown"),
        Value::Parameter { name } => Ok(name),
        _ => Err(not_built("a set in this scalar position")),
    }
}

pub fn bind_unit(name: &str) -> Result<Unit, QueryError> {
    Unit::from_name(name.strip_prefix("pl.").unwrap_or(name))
        .ok_or_else(|| not_built(&format!("unit {name}")))
}

fn not_built(what: &str) -> QueryError {
    QueryError::NotBuilt {
        what: what.to_owned(),
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

/// The message refused for a value outside a closed set: the set, and the
/// units where the value does exist (`"error" is an outcome of tool_calls`).
pub fn closed_set_message(def: &Field, value: &str) -> String {
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

/// A field `unit` does not have, with the nearest names it does.
pub fn unknown_field(text: &str, field: &str, unit: Unit, span: Span) -> QueryError {
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
    let similar: Vec<String> = scored
        .iter()
        .filter(|(d, _)| *d == best)
        .take(3)
        .map(|(_, n)| (*n).to_owned())
        .collect();
    let suggestion = match similar.as_slice() {
        [one] => Some(format!(
            "{}{}{}",
            &text[..span.start],
            one,
            &text[span.start + field.len()..]
        )),
        _ => None,
    };
    QueryError::UnknownField {
        field: field.to_owned(),
        unit: unit.name().to_owned(),
        position: span.start,
        similar,
        suggestion,
    }
}

#[cfg(test)]
mod tests;
