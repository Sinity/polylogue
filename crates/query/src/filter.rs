//! The conditions that select a unit's rows, bound to the catalog: each
//! field condition names its declared field ([`FieldId`]) and holds its
//! value in the form the store keeps (nano-dollars, milliseconds, times
//! resolved against the query's clock, closed sets in their declared
//! spelling), so lowering is a match on the field and never reads the
//! value's spelling again (docs/query.md: Units and fields, Times). A saved
//! query where a condition stands binds to the relation its frozen body
//! selects. A condition over other rows holds those rows bound to their own
//! unit, and a query inside a condition is bound whole ([`Bound`]), so the
//! store lowers one bound tree and binds nothing itself. A computed
//! comparison keeps its expressions, whose operand types the shared binder
//! already fixed.
use crate::{
    QueryError,
    ast::*,
    bound::Bound,
    defaults::Defaults,
    fields::{self, Field, FieldId, Kind},
    selection::top_level_refs,
    time::{Clock, ResolvedTime},
};

/// What binding needs from the query: its text (for diagnostics), the
/// origin names a reference may start with, the clock times resolve
/// against, the saved queries it may name, the columns every selection
/// shows, and the kept results a source may read.
#[derive(Clone, Copy)]
pub struct Binder<'a> {
    pub text: &'a str,
    pub origins: &'a [String],
    pub clock: &'a Clock,
    pub provider_catalog: Option<&'a references::Catalog>,
    /// The saved definitions this question resolved, by name and by hash.
    pub saved: &'a std::collections::BTreeMap<String, Vec<crate::SavedQuery>>,
    pub catalog: &'a query_language::catalog::Catalog,
    /// The saved queries whose bodies are being bound around this
    /// binding, outermost first.
    pub expanding: &'a [String],
    /// The columns asked for (`--fields`), which every selection of the
    /// query shows.
    pub fields: &'a [String],
    /// Kept results by ID: the unit a kept source's rows are.
    pub kept: &'a std::collections::BTreeMap<String, crate::Kept>,
    /// The shared binder's typing of the question whose text this binds,
    /// which types the queries nested in it.
    pub typing: Option<&'a crate::bound::QuestionTyping<'a>>,
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
    /// A word naming the caller repository, `latest`, `self`, or a pasted ID.
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
    /// `query:NAME` where a condition stands: the saved relation over the
    /// same unit, from the body its identity hashed, its parameters bound.
    /// It binds the unit's defaults by its own top-level conditions.
    Saved(Box<SavedRelation>),
    /// `exists UNIT(…)`: some related row matches.
    Exists {
        rows: ChildRows,
        span: Span,
    },
    /// `count(UNIT where …) OP N`: how many related rows match.
    Count {
        rows: ChildRows,
        op: Op,
        count: u64,
        span: Span,
    },
    /// `seq(a -> b)`: tool calls of one session in session order, each step
    /// bound to tool calls.
    Sequence {
        steps: Vec<Filter>,
        gaps: Vec<Gap>,
        span: Span,
    },
    /// `FIELD in (QUERY)`: the field's value is one the query groups by.
    /// Shared, so copying a filter (a `--why` clause is one) never copies
    /// the query.
    In {
        field: String,
        query: std::sync::Arc<Bound>,
        span: Span,
    },
    /// A comparison of computed values.
    Compare {
        left: Expression,
        op: Op,
        right: Expression,
        span: Span,
    },
    Between {
        left: Expression,
        from: Value,
        to: Value,
        span: Span,
    },
}

/// The rows of another unit a condition reads: the unit, the condition on
/// its rows bound to it, and the defaults the condition's own top-level
/// conditions leave in place.
#[derive(Debug, Clone, PartialEq)]
pub struct ChildRows {
    pub unit: Unit,
    pub filter: Option<Box<Filter>>,
    pub defaults: Defaults,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavedRelation {
    pub name: String,
    /// The saved annotation revision the name resolved to.
    pub revision: String,
    pub selection: crate::selection::Selection,
    pub span: Span,
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
    /// A field with its own rule, which reads this value.
    Special(Special),
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
                | Test::Special(Special::Compare(..) | Special::Between(..))
        )
    }
}

/// The value of a condition on a special field, in the form its rule reads:
/// values flattened from sets, a comparison's operator and operand, or the
/// condition a session's tool-call field puts on its tool calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Special {
    /// `F:VALUE`, and whether the value was quoted.
    One {
        text: String,
        quoted: bool,
    },
    /// `F:(A, B)`: the set's members, `unknown`, `known` and parameters by
    /// name.
    Set(Vec<String>),
    /// A parameter the caller has not supplied, by name.
    Parameter(String),
    Unknown,
    Known,
    Compare(Op, String),
    Between(String, String),
    /// `since:seen`: what changed since the caller last marked it seen.
    SinceSeen,
    /// `path:`, `tool:` or `action:` on sessions: the session made a tool
    /// call matching this condition, bound to tool calls.
    ToolCalls(Box<Filter>),
}

impl Special {
    /// The values a match lists, as [`alternatives`] gives them; none for
    /// a comparison.
    pub fn listed(&self) -> Vec<String> {
        match self {
            Special::One { text, .. } => vec![text.clone()],
            Special::Set(values) => values.clone(),
            Special::Parameter(name) => vec![format!("${name}")],
            Special::Unknown => vec!["unknown".into()],
            Special::Known => vec!["known".into()],
            Special::Compare(..)
            | Special::Between(..)
            | Special::SinceSeen
            | Special::ToolCalls(_) => vec![],
        }
    }

    /// The one value a match names, as [`value_text`] reads it.
    pub fn text(&self) -> Option<&str> {
        match self {
            Special::One { text, .. } => Some(text),
            Special::Unknown => Some("unknown"),
            Special::Parameter(name) => Some(name),
            _ => None,
        }
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
                ConditionKind::Saved { name, arguments } => {
                    self.saved_relation(name, arguments, unit, c.span)?
                }
                ConditionKind::Exists {
                    unit: rows,
                    condition,
                } => Filter::Exists {
                    rows: self.child_rows(rows, condition.as_deref())?,
                    span: c.span,
                },
                ConditionKind::Count {
                    unit: rows,
                    condition,
                    op,
                    count,
                } => Filter::Count {
                    rows: self.child_rows(rows, condition.as_deref())?,
                    op: *op,
                    count: *count,
                    span: c.span,
                },
                ConditionKind::Sequence {
                    steps,
                    gaps,
                    all: false,
                    ..
                } => Filter::Sequence {
                    steps: steps
                        .iter()
                        .map(|step| self.bind(step, Unit::ToolCalls))
                        .collect::<Result<_, _>>()?,
                    gaps: gaps.clone(),
                    span: c.span,
                },
                ConditionKind::In { left, query } => self.membership(left, query, c.span)?,
                ConditionKind::Compare { left, op, right } => Filter::Compare {
                    left: left.clone(),
                    op: *op,
                    right: right.clone(),
                    span: c.span,
                },
                ConditionKind::Between { left, from, to } => Filter::Between {
                    left: left.clone(),
                    from: from.clone(),
                    to: to.clone(),
                    span: c.span,
                },
                // An order condition's matching row is read by the relation
                // holding it, never inside one unit's rows.
                _ => return Err(not_built("this shared condition")),
            })
        })
    }

    /// The rows of `unit_name` a condition reads, `condition` bound to them.
    fn child_rows(
        &self,
        unit_name: &str,
        condition: Option<&Condition>,
    ) -> Result<ChildRows, QueryError> {
        let unit = bind_unit(unit_name)?;
        let filter = condition
            .map(|condition| self.bind(condition, unit).map(Box::new))
            .transpose()?;
        let top = condition.map(top_level_refs).unwrap_or_default();
        Ok(ChildRows {
            unit,
            filter,
            defaults: Defaults::bind(unit, &top, self.origins),
        })
    }

    /// `FIELD in (QUERY)`: the query must group by one field, whose values
    /// the field is compared with.
    fn membership(
        &self,
        left: &Expression,
        query: &Query,
        span: Span,
    ) -> Result<Filter, QueryError> {
        let Expression::Field { name: field } = left else {
            return Err(not_built("this computed field"));
        };
        let grouped_by_one = query.transforms.iter().rev().find_map(|s| match &s.kind {
            TransformKind::Group { keys } => Some(keys.len() == 1),
            _ => None,
        });
        if grouped_by_one != Some(true) {
            return Err(QueryError::Invalid {
                message: format!(
                    "`{field} in (…)` compares {field} with the values a query groups by: \
                     end it with `| group by FIELD` (one field)"
                ),
                position: span.start,
            });
        }
        Ok(Filter::In {
            field: field.clone(),
            query: std::sync::Arc::new(
                self.nested_query(query, crate::bound::Nested::condition(span))?,
            ),
            span,
        })
    }

    /// A query inside this one, standing at `at`, after its inherited unit
    /// was bound. The question's typing types it; a query outside that
    /// typing (in a saved body) is bound on its own, and so are the
    /// queries inside it.
    pub fn nested_query(
        &self,
        query: &Query,
        at: crate::bound::Nested,
    ) -> Result<Bound, QueryError> {
        let mut query = query.clone();
        if let Source::Unit { implicit, .. } = &mut query.source {
            *implicit = false;
        }
        if let Some(typing) = self.typing.and_then(|typing| typing.at(at)) {
            return Bound::of(query, Some(typing), self);
        }
        let bound = query_language::bind::bind(&query, self.catalog)
            .map_err(|error| QueryError::language(error, self.text))?;
        let typing = crate::bound::TypingAt {
            typing: bound.typing(),
            base: Default::default(),
        };
        Bound::of(
            bound.query().clone(),
            Some(typing),
            &Binder {
                typing: None,
                ..*self
            },
        )
    }

    /// `query:NAME` over `unit`'s rows: a relation (a unit and conditions,
    /// no transforms) over the same unit. A saved relation over another
    /// stays a reference to it, so the inner one keeps the defaults it
    /// bound.
    fn saved_relation(
        &self,
        name: &str,
        arguments: &[Argument],
        unit: Unit,
        span: Span,
    ) -> Result<Filter, QueryError> {
        let (body, definition) = crate::compile::source::saved_body(
            name,
            arguments,
            self.text,
            span.start,
            self.saved,
            self.catalog,
            self.expanding,
        )?;
        let shaped = || QueryError::Invalid {
            message: format!(
                "query:{name} is a shaped result (it has stages), so it cannot be a condition; \
                 read it as `query:{name} | where …`"
            ),
            position: span.start,
        };
        if !body.transforms.is_empty() || !body.attach.is_empty() {
            return Err(shaped());
        }
        let mut expanding = self.expanding.to_vec();
        expanding.push(name.to_owned());
        // The body's spans are positions in its written-out syntax.
        let syntax = definition.executable_text()?;
        let inner = Binder {
            text: syntax,
            expanding: &expanding,
            fields: &[],
            typing: None,
            ..*self
        };
        let (selected, condition) = match body.source {
            Source::Unit {
                unit, condition, ..
            } => (bind_unit(&unit)?, condition),
            Source::Saved {
                name: nested,
                arguments,
                condition,
                ..
            } => {
                let selected = inner.relation_unit(&nested, &arguments)?;
                let reference = Condition {
                    span: Span::default(),
                    kind: ConditionKind::Saved {
                        name: nested,
                        arguments,
                    },
                };
                let condition = match condition {
                    Some(condition) => Condition {
                        span: condition.span,
                        kind: ConditionKind::And {
                            conditions: vec![reference, condition],
                        },
                    },
                    None => reference,
                };
                (selected, Some(condition))
            }
            Source::Result { .. } => {
                return Err(not_built("a saved source of this kind as a condition"));
            }
        };
        if selected != unit {
            return Err(QueryError::Invalid {
                message: format!(
                    "query:{name} selects {}, not {}",
                    selected.name(),
                    unit.name()
                ),
                position: span.start,
            });
        }
        let query = Query {
            source: Source::Unit {
                unit: selected.name().into(),
                implicit: false,
                condition,
            },
            transforms: vec![],
            attach: vec![],
        };
        let selection = crate::selection::Selection::bind(&query, &inner, false)
            .map_err(|error| error.inside_saved(name, syntax, span.start))?
            .expect("a unit source");
        if selection.semantic.is_some() {
            return Err(shaped());
        }
        Ok(Filter::Saved(Box::new(SavedRelation {
            name: name.to_owned(),
            revision: definition.revision.clone(),
            selection,
            span,
        })))
    }

    /// The unit a saved relation selects, through the saved relations it
    /// is defined over.
    fn relation_unit(&self, name: &str, arguments: &[Argument]) -> Result<Unit, QueryError> {
        let (body, _) = crate::compile::source::saved_body(
            name,
            arguments,
            self.text,
            0,
            self.saved,
            self.catalog,
            self.expanding,
        )?;
        match &body.source {
            Source::Unit { unit, .. } => bind_unit(unit),
            Source::Saved {
                name: nested,
                arguments,
                ..
            } => {
                let mut expanding = self.expanding.to_vec();
                expanding.push(name.to_owned());
                Binder {
                    expanding: &expanding,
                    ..*self
                }
                .relation_unit(nested, arguments)
            }
            Source::Result { .. } => Err(not_built("a saved source of this kind as a condition")),
        }
    }

    fn reference(&self, reference: &str, unit: Unit, span: Span) -> Result<Filter, QueryError> {
        let kind = reference.split_once(':').map(|(kind, _)| kind);
        if matches!(
            kind,
            Some("block" | "document" | "event" | "span" | "annotation" | "delivery")
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
            let invalid = || QueryError::Invalid {
                message: format!("{reference} is not a conversation URL Polylogue recognizes"),
                position: span.start,
            };
            let parsed = self
                .provider_catalog
                .ok_or_else(invalid)?
                .read_url(reference)
                .map_err(|_| invalid())?;
            let references::Target::Object(object) = parsed.target() else {
                return Err(invalid());
            };
            if object.product() != references::Product::Polylogue || object.kind() != "session" {
                return Err(invalid());
            }
            let session = polylogue_model::SessionId::parse(object.id()).map_err(|_| invalid())?;
            // A conversation page fragment selects its browser view; other
            // session selection keeps naming the complete physical session.
            Ok(Filter::Session {
                id: session.to_string(),
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
            // `tool:<session>#<call id>` names one call; a tool's name has no `#`.
            if lower == "tool"
                && matches!(
                    polylogue_model::Reference::parse(&format!("tool:{text}")),
                    Ok(polylogue_model::Reference::ToolCall(..))
                )
            {
                return Ok(rows(format!("tool:{text}")));
            }
            if (lower == "message" && text.contains(".."))
                || (lower == "event" && unit != Unit::Deliveries)
                || matches!(
                    lower.as_str(),
                    "block" | "document" | "span" | "annotation" | "delivery"
                )
            {
                return Ok(rows(format!("{lower}:{text}")));
            }
            let invalid = |error: String| QueryError::Invalid {
                message: error,
                position: span.start,
            };
            if lower == "request" && fields::lookup(unit, &lower).is_none() {
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
            comparison
                if def.name == "since"
                    && matches!(&comparison, Comparison::Match(Value::Scalar {text,..}) if text == "seen") =>
            {
                if !matches!(unit, Unit::Sessions | Unit::Annotations) {
                    return Err(QueryError::BadValue {
                        field: "since".into(),
                        value: "seen".into(),
                        position: span.start,
                        message: "since:seen applies to sessions and annotations".into(),
                    });
                }
                Test::Special(Special::SinceSeen)
            }
            comparison if matches!(def.name, "since" | "until" | "seen_by") => {
                let Comparison::Match(Value::Scalar { text, .. }) = &comparison else {
                    return Err(QueryError::Invalid {
                        message: format!("{0} takes {0}:VALUE", def.name),
                        position: span.start,
                    });
                };
                Test::Time(TimeTest::Within(self.time(text, span)?))
            }
            comparison
                if matches!(
                    id,
                    FieldId::SessionsPath | FieldId::SessionsTool | FieldId::SessionsAction
                ) =>
            {
                Test::Special(Special::ToolCalls(Box::new(self.field(
                    Unit::ToolCalls,
                    def.name,
                    comparison,
                    span,
                )?)))
            }
            Comparison::Match(value) => Test::Special(match value {
                Value::Scalar { text, quoted } => Special::One { text, quoted },
                Value::Unknown => Special::Unknown,
                Value::Known => Special::Known,
                Value::Parameter { name } => Special::Parameter(name),
                set => Special::Set(alternatives(&set)),
            }),
            Comparison::Compare(op, value) => Test::Special(Special::Compare(op, value)),
            Comparison::Between(from, to) => Test::Special(Special::Between(from, to)),
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
                // A field that stores its own unknown word answers both that
                // word and `unknown` with a missing-value test.
                let stored_unknown = def.stored_unknown;
                let before = values.len();
                values.retain(|value| {
                    !stored_unknown.is_some_and(|word| {
                        value.eq_ignore_ascii_case("unknown") || value.eq_ignore_ascii_case(word)
                    })
                });
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
