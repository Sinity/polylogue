//! A query bound whole, before lowering: its selection, and what each
//! transform read as a relation stage needs bound (a query it reads, the
//! condition it puts on the relation's rows, the unit rows it selects).
//! Binding happens once here; the store lowers the result and binds nothing
//! itself (docs/query.md: Units and fields, Stages). Where a part cannot be
//! bound, the binding keeps the error, which the store refuses with when it
//! reaches that part, so a query is refused for the first part it lowers.
use crate::{
    QueryError,
    ast::*,
    filter::{Binder, Filter, bind_unit},
    selection::{Selection, contains_order, top_level_refs},
};
use query_language::{
    catalog::NaturalMatch,
    scope::{NodePath, Scope, Typing},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Bound {
    /// The query as the shared binder bound it, read by position for its
    /// source, stages, fields, groups and aggregates.
    pub query: Query,
    /// What a unit source selects; `None` for a kept result.
    pub selection: Option<Selection>,
    /// A source condition holding an order condition, read over the
    /// relation the selection's rows form.
    pub source_condition: Option<RelationCondition>,
    /// What each transform needs bound when it is lowered as a relation
    /// stage, by its index.
    pub stages: Vec<StageBinding>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StageBinding {
    /// Nothing beyond the transform as written.
    None,
    /// `| where` over a relation.
    Where(Box<RelationCondition>),
    /// The query a combination, join or diff reads, or the query over the
    /// related unit of `| UNIT where …`.
    Query(Result<Box<Bound>, QueryError>),
    /// `| matches`, `| measure` or another unit transform over a relation:
    /// the relation's unit selected by the patterns it carries (each
    /// top-level `exists` or `seq` of the source and later `| where`
    /// stages), then the transform.
    UnitRows(Result<Box<Selection>, QueryError>),
}

/// A condition over a relation's rows, mirroring its written form. Each
/// node can also be read over the relation's unit rows, from the leaves
/// bound to that unit, when it holds no order condition.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationCondition {
    pub span: Span,
    /// An order condition is somewhere in this condition.
    pub has_order: bool,
    pub kind: RelationTest,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationTest {
    All(Vec<RelationCondition>),
    Any(Vec<RelationCondition>),
    Not(Box<RelationCondition>),
    /// `before(…)`, `after(…)` and the other order relations, with the
    /// query whose rows they match.
    Order {
        relation: OrderRelation,
        within: Option<Distance>,
        at: TimePoint,
        scope: Option<String>,
        pick: MatchChoice,
        query: Result<Box<Bound>, QueryError>,
        /// This order condition exposes its matching row (`match.`) after
        /// the whole condition.
        exposed: bool,
    },
    /// A leaf, and the leaf bound to the relation's unit.
    Leaf {
        leaf: RelationLeaf,
        unit: Result<Filter, QueryError>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationLeaf {
    /// A comparison of computed values; `subtree` when it is a path prefix
    /// over a field matched by directory.
    Compare {
        left: Expression,
        op: Op,
        right: Expression,
        subtree: bool,
    },
    Between {
        left: Expression,
        from: Value,
        to: Value,
    },
    /// `F:known` or `F:unknown`.
    Presence { field: String, known: bool },
    /// `F:VALUE`, matched as the relation's field declares; `list` when the
    /// field holds a list.
    Match {
        field: String,
        value: Value,
        natural: NaturalMatch,
        list: bool,
    },
    /// Another condition, which only the unit's rows can answer.
    Other(Condition),
}

impl RelationCondition {
    /// This condition over the relation's unit rows, composed from its
    /// leaves bound to that unit, as binding the whole condition gives it.
    pub fn unit_filter(&self) -> Result<Filter, QueryError> {
        query_language::depth::deeper(|| match &self.kind {
            RelationTest::All(parts) => Ok(Filter::All(
                parts
                    .iter()
                    .map(RelationCondition::unit_filter)
                    .collect::<Result<_, _>>()?,
            )),
            RelationTest::Any(parts) => Ok(Filter::Any(
                parts
                    .iter()
                    .map(RelationCondition::unit_filter)
                    .collect::<Result<_, _>>()?,
            )),
            RelationTest::Not(inner) => Ok(Filter::Not(Box::new(inner.unit_filter()?))),
            RelationTest::Order { .. } => Err(QueryError::NotBuilt {
                what: "this shared condition".into(),
            }),
            RelationTest::Leaf { unit, .. } => unit.clone(),
        })
    }
}

/// What binding a relation condition reads: the unit of the relation's
/// rows, the fields in scope before it, and the operand types.
struct RelationScope<'s> {
    unit: Option<Unit>,
    fields: Option<&'s Scope>,
    typing: Option<&'s TypingAt<'s>>,
}

/// The shared binder's scopes and operand types for a whole question, and
/// where in it each query nested in a condition or a stage sits, so a
/// nested query is typed from them rather than bound again.
pub struct QuestionTyping<'t> {
    typing: &'t Typing,
    nested: std::collections::HashMap<Nested, NodePath>,
}

/// A query nested in the question, by the position of the condition or
/// stage holding it: positions in one text tell its nodes apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nested {
    Condition(usize, usize),
    Stage(usize, usize),
}

impl Nested {
    pub fn condition(span: Span) -> Self {
        Self::Condition(span.start, span.end)
    }
    fn stage(span: Span) -> Self {
        Self::Stage(span.start, span.end)
    }
}

/// The shared binder's typing for one query: the question's typing read
/// under the path of the query's root.
pub struct TypingAt<'t> {
    pub typing: &'t Typing,
    pub base: NodePath,
}

impl TypingAt<'_> {
    fn query(&self) -> Option<&query_language::scope::QueryScope> {
        self.typing.query(&self.base)
    }

    fn operand(&self, path: &NodePath) -> Option<&query_language::scope::TypedOperand> {
        let mut full = self.base.clone();
        full.0.extend(path.0.iter().cloned());
        self.typing.operand(&full)
    }
}

impl<'t> QuestionTyping<'t> {
    /// The typing of `query`, which the shared binder bound as `typing`.
    pub fn new(query: &Query, typing: &'t Typing) -> Self {
        let mut nested = std::collections::HashMap::new();
        nested_queries(query, &NodePath::default(), &mut nested);
        Self { typing, nested }
    }

    /// The typing of the query nested at `at`, if this question holds one.
    pub fn at(&self, at: Nested) -> Option<TypingAt<'t>> {
        self.nested.get(&at).map(|base| TypingAt {
            typing: self.typing,
            base: base.clone(),
        })
    }
}

/// Where each nested query sits, by the paths the shared binder's typing
/// uses (query-language parse/typing.rs).
fn nested_queries(
    query: &Query,
    path: &NodePath,
    out: &mut std::collections::HashMap<Nested, NodePath>,
) {
    query_language::depth::deeper(|| {
        match &query.source {
            Source::Unit {
                condition: Some(condition),
                ..
            }
            | Source::Saved {
                condition: Some(condition),
                ..
            }
            | Source::Result {
                condition: Some(condition),
                ..
            } => nested_in_condition(condition, &path.field("source").field("condition"), out),
            _ => {}
        }
        for (index, attachment) in query.attach.iter().enumerate() {
            for (window_index, window) in attachment.windows.iter().enumerate() {
                if let Window::Where(condition) = window {
                    let at = path
                        .field("attach")
                        .index(index)
                        .field("windows")
                        .index(window_index)
                        .field("where");
                    nested_in_condition(condition, &at, out);
                }
            }
        }
        for (index, transform) in query.transforms.iter().enumerate() {
            let at = path.field("transforms").index(index).field("kind");
            match &transform.kind {
                TransformKind::Where { condition }
                | TransformKind::Scope {
                    condition: Some(condition),
                    ..
                } => nested_in_condition(condition, &at.field("condition"), out),
                TransformKind::Combine { query, .. }
                | TransformKind::Join { query, .. }
                | TransformKind::Diff { query } => {
                    let at = at.field("query");
                    nested_queries(query, &at, out);
                    out.insert(Nested::stage(transform.span), at);
                }
                _ => {}
            }
        }
    })
}

fn nested_in_condition(
    condition: &Condition,
    path: &NodePath,
    out: &mut std::collections::HashMap<Nested, NodePath>,
) {
    query_language::depth::deeper(|| {
        let at = path.field("kind");
        match &condition.kind {
            ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
                for (index, part) in conditions.iter().enumerate() {
                    nested_in_condition(part, &at.field("conditions").index(index), out);
                }
            }
            ConditionKind::Not { condition } => {
                nested_in_condition(condition, &at.field("condition"), out)
            }
            ConditionKind::In { query, .. } | ConditionKind::Order { query, .. } => {
                let at = at.field("query");
                nested_queries(query, &at, out);
                out.insert(Nested::condition(condition.span), at);
            }
            ConditionKind::Exists {
                condition: Some(inner),
                ..
            }
            | ConditionKind::Count {
                condition: Some(inner),
                ..
            } => nested_in_condition(inner, &at.field("condition"), out),
            ConditionKind::Sequence { steps, .. } => {
                for (index, step) in steps.iter().enumerate() {
                    nested_in_condition(step, &at.field("steps").index(index), out);
                }
            }
            _ => {}
        }
    })
}

impl Bound {
    /// `query` bound to Polylogue's units, fields and stages; `typing` is
    /// the shared binder's scopes and operand types for it, when the shared
    /// binder bound it.
    pub fn of(
        query: Query,
        typing: Option<TypingAt<'_>>,
        binder: &Binder<'_>,
    ) -> Result<Bound, QueryError> {
        let selection = Selection::of(&query, binder)?;
        let typing = typing.as_ref();
        let scope = typing.and_then(TypingAt::query);
        let (first, mut unit) = match (&selection, &query.source) {
            (Some(selection), _) => (selection.relations_from, Some(selection.unit)),
            (None, Source::Result { reference, .. }) => {
                let id = reference.strip_prefix("result:").unwrap_or(reference);
                (
                    Some(0),
                    binder
                        .kept
                        .get(id)
                        .and_then(|kept| Unit::from_bound_name(&kept.unit)),
                )
            }
            (None, _) => (None, None),
        };
        let source_condition = match &query.source {
            Source::Unit {
                condition: Some(condition),
                ..
            } if contains_order(condition) => Some(relation_condition(
                condition,
                &RelationScope {
                    unit,
                    fields: scope.map(|scope| &scope.source),
                    typing,
                },
                NodePath::default().field("source").field("condition"),
                condition.exposed_order(),
                binder,
            )),
            _ => None,
        };
        let mut patterns: Vec<&Condition> = match &query.source {
            Source::Unit {
                condition: Some(condition),
                ..
            } => match_patterns(condition),
            _ => vec![],
        };
        let mut stages = vec![StageBinding::None; query.transforms.len()];
        for (index, transform) in query.transforms.iter().enumerate() {
            if first.is_none_or(|first| index < first) {
                continue;
            }
            stages[index] = match &transform.kind {
                TransformKind::Where { condition } => {
                    let binding = StageBinding::Where(Box::new(relation_condition(
                        condition,
                        &RelationScope {
                            unit,
                            fields: scope
                                .and_then(|scope| scope.transforms.get(index))
                                .map(|transform| &transform.before),
                            typing,
                        },
                        NodePath::default()
                            .field("transforms")
                            .index(index)
                            .field("kind")
                            .field("condition"),
                        condition.exposed_order(),
                        binder,
                    )));
                    patterns.extend(match_patterns(condition));
                    binding
                }
                TransformKind::Combine { query, .. }
                | TransformKind::Join { query, .. }
                | TransformKind::Diff { query } => StageBinding::Query(
                    binder
                        .nested_query(query, Nested::stage(transform.span))
                        .map(Box::new),
                ),
                TransformKind::Scope {
                    unit: target,
                    condition,
                } => {
                    let related = bind_unit(target).and_then(|target| {
                        let query = Query {
                            source: Source::Unit {
                                unit: target.name().into(),
                                implicit: false,
                                condition: condition.clone(),
                            },
                            transforms: vec![],
                            attach: vec![],
                        };
                        Bound::of(query, None, binder).map(Box::new)
                    });
                    unit = bind_unit(target).ok();
                    patterns = condition.as_ref().map(match_patterns).unwrap_or_default();
                    StageBinding::Query(related)
                }
                TransformKind::Fields { .. }
                | TransformKind::Sort { .. }
                | TransformKind::Limit { .. }
                | TransformKind::First { .. }
                | TransformKind::Offset { .. }
                | TransformKind::Distinct
                | TransformKind::Group { .. }
                | TransformKind::Aggregate { .. }
                | TransformKind::AsOf { .. } => StageBinding::None,
                _ => {
                    let binding =
                        StageBinding::UnitRows(unit_rows(unit, &patterns, transform, binder));
                    if matches!(transform.kind, TransformKind::Matches) {
                        patterns.clear();
                    }
                    binding
                }
            };
        }
        Ok(Bound {
            query,
            selection,
            source_condition,
            stages,
        })
    }
}

/// The selection a unit transform over a relation reads: the relation's
/// unit under `patterns`, then the transform.
fn unit_rows(
    unit: Option<Unit>,
    patterns: &[&Condition],
    transform: &Transform,
    binder: &Binder<'_>,
) -> Result<Box<Selection>, QueryError> {
    let unit = unit.ok_or_else(|| QueryError::NotBuilt {
        what: "this unit transform after a change of row grain".into(),
    })?;
    let condition = (matches!(transform.kind, TransformKind::Matches) && !patterns.is_empty())
        .then(|| Condition {
            kind: ConditionKind::And {
                conditions: patterns.iter().map(|&pattern| pattern.clone()).collect(),
            },
            span: transform.span,
        });
    let query = Query {
        source: Source::Unit {
            unit: unit.name().into(),
            implicit: false,
            condition,
        },
        transforms: vec![transform.clone()],
        attach: vec![],
    };
    Ok(Box::new(
        Selection::of(&query, binder)?.expect("a unit source"),
    ))
}

/// The top-level `exists` and `seq` conditions, which `| matches` reads.
fn match_patterns(condition: &Condition) -> Vec<&Condition> {
    top_level_refs(condition)
        .into_iter()
        .filter(|condition| {
            matches!(
                condition.kind,
                ConditionKind::Exists { .. } | ConditionKind::Sequence { .. }
            )
        })
        .collect()
}

fn relation_condition(
    condition: &Condition,
    scope: &RelationScope<'_>,
    path: NodePath,
    exposed: Option<&Condition>,
    binder: &Binder<'_>,
) -> RelationCondition {
    query_language::depth::deeper(|| {
        let leaf = |leaf: RelationLeaf| RelationTest::Leaf {
            leaf,
            unit: match scope.unit {
                Some(unit) => binder.bind(condition, unit),
                None => Err(QueryError::NotBuilt {
                    what: "conditions on these rows".into(),
                }),
            },
        };
        let parts = |conditions: &[Condition]| {
            conditions
                .iter()
                .enumerate()
                .map(|(index, part)| {
                    relation_condition(
                        part,
                        scope,
                        path.field("kind").field("conditions").index(index),
                        exposed,
                        binder,
                    )
                })
                .collect::<Vec<_>>()
        };
        let kind = match &condition.kind {
            ConditionKind::Order {
                relation,
                query,
                within,
                at,
                scope: order_scope,
                pick,
            } => RelationTest::Order {
                relation: *relation,
                within: within.clone(),
                at: *at,
                scope: order_scope.clone(),
                pick: pick.clone(),
                query: binder
                    .nested_query(query, Nested::condition(condition.span))
                    .map(Box::new),
                exposed: exposed.is_some_and(|exposed| std::ptr::eq(exposed, condition)),
            },
            ConditionKind::And { conditions } => RelationTest::All(parts(conditions)),
            ConditionKind::Or { conditions } => RelationTest::Any(parts(conditions)),
            ConditionKind::Not { condition: inner } => {
                RelationTest::Not(Box::new(relation_condition(
                    inner,
                    scope,
                    path.field("kind").field("condition"),
                    exposed,
                    binder,
                )))
            }
            ConditionKind::Compare { left, op, right } => {
                let natural = scope
                    .typing
                    .and_then(|typing| typing.operand(&path.field("kind").field("left")))
                    .map(|operand| operand.result.field.natural_match());
                leaf(RelationLeaf::Compare {
                    left: left.clone(),
                    op: *op,
                    right: right.clone(),
                    subtree: *op == Op::Prefix && natural == Some(NaturalMatch::Subtree),
                })
            }
            ConditionKind::Between { left, from, to } => leaf(RelationLeaf::Between {
                left: left.clone(),
                from: from.clone(),
                to: to.clone(),
            }),
            ConditionKind::Match {
                field,
                value: value @ (Value::Unknown | Value::Known),
            } => leaf(RelationLeaf::Presence {
                field: field.clone(),
                known: matches!(value, Value::Known),
            }),
            ConditionKind::Match { field, value } => {
                let info = scope.fields.and_then(|fields| fields.field(field));
                leaf(RelationLeaf::Match {
                    field: field.clone(),
                    value: value.clone(),
                    natural: info
                        .map(|info| info.field.natural_match())
                        .unwrap_or(NaturalMatch::Equal),
                    list: info.is_some_and(|info| {
                        matches!(info.field.kind, query_language::catalog::FieldKind::List(_))
                    }),
                })
            }
            _ => leaf(RelationLeaf::Other(condition.clone())),
        };
        let has_order = match &kind {
            RelationTest::Order { .. } => true,
            RelationTest::All(parts) | RelationTest::Any(parts) => {
                parts.iter().any(|part| part.has_order)
            }
            RelationTest::Not(inner) => inner.has_order,
            RelationTest::Leaf { .. } => false,
        };
        RelationCondition {
            span: condition.span,
            has_order,
            kind,
        }
    })
}

#[cfg(test)]
mod tests;
