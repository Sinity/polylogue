//! What a query over a unit selects, in one form however it was written:
//! the unit, the defaults its own conditions leave in place, the conditions
//! that filter its rows, the text that ranks them, the stages that follow,
//! the attached rows and the columns shown (docs/query.md: Defaults,
//! Stages, Attached rows; docs/search.md). `A B`, `A | where B` and
//! `(A) AND (B)` select alike. Projection and attachment sit beside the
//! selection and never decide which rows it holds.
use crate::{
    QueryError,
    ast::*,
    defaults::{self, DefaultId},
    filter::{Binder, Filter},
    search::Search,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub unit: Unit,
    /// The defaults the source's own top-level conditions leave in place.
    pub defaults: Vec<DefaultId>,
    /// The conditions as written, joined by AND: the source's condition
    /// without its `similar:` term, then each `| where` stage that directly
    /// follows the source and means the same as its conjunction.
    pub written: Vec<Condition>,
    /// Those conditions bound to the unit's fields: what selects the rows,
    /// or `None` for every row.
    pub filter: Option<Filter>,
    /// What the words search for and which index ranks by them.
    pub search: Search,
    /// The `similar:` text the rows rank by.
    pub semantic: Option<Semantic>,
    pub order: Order,
    /// The index of the first transform that stays a stage; the filter
    /// holds the `| where` stages before it.
    pub stages_from: usize,
    /// The index of the first stage that reads the selection's rows as a
    /// relation, or `None` when every stage reads the selection itself.
    pub relations_from: Option<usize>,
    /// A `| measure` stage measures the selection, whose population then
    /// includes hidden sessions.
    pub measured: bool,
    pub attach: Vec<Attachment>,
    pub projection: Projection,
}

/// A `similar:` ranking: the text whose embedding the rows are compared
/// with, and the unit whose stored texts are compared (a row unit without
/// its own similarity ranks by its session's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Semantic {
    pub text: String,
    pub unit: Unit,
    pub span: Span,
}

/// How the selection's rows are ordered (docs/search.md: Hits and
/// ranking, Fusion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    /// The unit's own order: time, then identity.
    Default,
    /// By one ranking, lower first.
    Ranked(Ranking),
    /// By the words' relevance and the similarity together, fused by
    /// reciprocal rank.
    Fused,
    /// A later stage sorts, groups, aggregates, rescopes or measures the
    /// rows before anything reads their order: no similarity is computed
    /// for it.
    Sorted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ranking {
    /// The search leg's bm25 relevance.
    Relevance,
    /// Similarity to the semantic leg's text.
    Similarity,
}

/// The columns each row shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Projection {
    /// The unit's own columns and the fields the conditions name.
    Default,
    /// Exactly these fields (`--fields`), `prefix.field` paths included.
    Fields(Vec<String>),
}

impl Selection {
    /// The selection of a query whose source is a unit, or `None` for a
    /// saved query or kept result as the source; `fields` are the columns
    /// asked for.
    pub fn of(
        query: &Query,
        binder: &Binder<'_>,
        fields: &[String],
    ) -> Result<Option<Selection>, QueryError> {
        let origins = binder.origins;
        let Source::Unit {
            unit, condition, ..
        } = &query.source
        else {
            return Ok(None);
        };
        let Some(unit) = Unit::from_name(unit.strip_prefix("pl.").unwrap_or(unit)) else {
            return Err(QueryError::NotBuilt {
                what: format!("unit {unit}"),
            });
        };
        let (condition, semantic) = match condition {
            Some(condition) => lift_similar(condition, unit, origins)?,
            None => (None, None),
        };
        let top = condition.as_ref().map(top_level).unwrap_or_default();
        let folded = stages_from(query);
        let relations_from = if condition.as_ref().is_some_and(contains_order) {
            Some(0)
        } else {
            query.transforms[folded..]
                .iter()
                .position(|stage| relational(&stage.kind))
                .map(|at| at + folded)
        };
        let measured = query.transforms[folded..relations_from.unwrap_or(query.transforms.len())]
            .iter()
            .any(|stage| {
                matches!(&stage.kind, TransformKind::Extension { name, .. }
                if name == crate::catalog::MEASURE)
            });
        let defaults = defaults::bind(unit, &top, origins, measured);
        let mut written: Vec<Condition> = condition.into_iter().collect();
        written.extend(
            query.transforms[..folded]
                .iter()
                .filter_map(|stage| match &stage.kind {
                    TransformKind::Where { condition } => Some(condition.clone()),
                    _ => None,
                }),
        );
        // `A | where B` is the conjunction `(A) (B)`: each part keeps its
        // own grouping of search words.
        let filter = match written.as_slice() {
            [] => None,
            [one] => Some(binder.bind(one, unit)?),
            all => Some(Filter::All(
                all.iter()
                    .map(|c| binder.bind(c, unit))
                    .collect::<Result<_, _>>()?,
            )),
        };
        // An order condition's witness is read by the relation holding it.
        let search = if written.iter().any(contains_order) {
            Search::default()
        } else {
            Search::of(filter.as_ref(), unit)
        };
        let order = if !order_read(&query.transforms[folded..]) {
            Order::Sorted
        } else {
            match (search.rank.is_some(), semantic.is_some()) {
                (true, true) => Order::Fused,
                (true, false) => Order::Ranked(Ranking::Relevance),
                (false, true) => Order::Ranked(Ranking::Similarity),
                (false, false) => Order::Default,
            }
        };
        Ok(Some(Selection {
            unit,
            defaults,
            written,
            filter,
            search,
            semantic,
            order,
            stages_from: folded,
            relations_from,
            measured,
            attach: query.attach.clone(),
            projection: if fields.is_empty() {
                Projection::Default
            } else {
                Projection::Fields(fields.to_vec())
            },
        }))
    }

    /// The written conditions at the top level, parenthesized ANDs
    /// flattened.
    pub fn top_level(&self) -> Vec<Condition> {
        self.written.iter().flat_map(top_level).collect()
    }
}

/// How many `| where` stages directly after a unit source join its
/// selection: the index of the first transform that stays a stage. A stage stays a stage when the conjunction would mean
/// something else: an order condition (whose witness the stage replaces),
/// `last`/`latest` (the newest session meeting every condition beside it)
/// and `| matches` (which reads its pattern from the stage).
pub fn stages_from(query: &Query) -> usize {
    let Source::Unit { condition, .. } = &query.source else {
        return 0;
    };
    let newest = |c: &Condition| top_level(c).iter().any(newest_word);
    if condition
        .as_ref()
        .is_some_and(|c| contains_order(c) || newest(c))
        || query
            .transforms
            .iter()
            .any(|stage| matches!(stage.kind, TransformKind::Matches))
    {
        return 0;
    }
    query
        .transforms
        .iter()
        .take_while(|stage| match &stage.kind {
            TransformKind::Where { condition } => !contains_order(condition) && !newest(condition),
            _ => false,
        })
        .count()
}

/// Whether anything reads the order of rows entering `stages`: a row
/// selection (`limit`, `first`, `offset`, `| group by K | first N`), a
/// diff, or the answer itself, unless a sort, grouping, aggregate, rescope,
/// `| matches` or `| measure` replaces it first.
fn order_read(stages: &[Transform]) -> bool {
    let mut inherited = true;
    let mut group_input = None;
    for stage in stages {
        match &stage.kind {
            TransformKind::Group { .. } => {
                group_input = Some(inherited);
                inherited = false;
                continue;
            }
            TransformKind::First { .. } if group_input.is_some() => {
                inherited = group_input.take().expect("group input order");
                if inherited {
                    return true;
                }
            }
            TransformKind::Limit { .. }
            | TransformKind::Offset { .. }
            | TransformKind::First { .. } => {
                if inherited {
                    return true;
                }
            }
            TransformKind::Sort { .. }
            | TransformKind::Aggregate { .. }
            | TransformKind::Scope { .. }
            | TransformKind::Matches => inherited = false,
            TransformKind::Extension { name, .. } if name == crate::catalog::MEASURE => {
                inherited = false
            }
            TransformKind::Diff { .. } => return true,
            TransformKind::Combine {
                how: Combination::Union | Combination::Fuse,
                ..
            } => inherited = true,
            _ => {}
        }
        group_input = None;
    }
    inherited
}

/// A stage that reads the rows entering it as a relation.
fn relational(kind: &TransformKind) -> bool {
    matches!(
        kind,
        TransformKind::Where { .. }
            | TransformKind::Sort { .. }
            | TransformKind::Fields { .. }
            | TransformKind::Group { .. }
            | TransformKind::Aggregate { .. }
            | TransformKind::First { .. }
            | TransformKind::Limit { .. }
            | TransformKind::Offset { .. }
            | TransformKind::Distinct
            | TransformKind::Join { .. }
            | TransformKind::Diff { .. }
            | TransformKind::Scope { .. }
            | TransformKind::Matches
            | TransformKind::Combine { .. }
    )
}

/// Takes the one `similar:` term out of the source's top-level ANDs. A
/// second one, or one under OR or NOT, stays in the condition, where
/// lowering refuses it. A field the unit does not have stays too, for the
/// unknown-field answer.
fn lift_similar(
    condition: &Condition,
    unit: Unit,
    origins: &[String],
) -> Result<(Option<Condition>, Option<Semantic>), QueryError> {
    let mut semantic = None;
    let condition = lift(condition, unit, origins, &mut semantic)?;
    Ok((condition, semantic))
}

fn lift(
    condition: &Condition,
    unit: Unit,
    origins: &[String],
    semantic: &mut Option<Semantic>,
) -> Result<Option<Condition>, QueryError> {
    query_language::depth::deeper(|| match &condition.kind {
        ConditionKind::And { conditions } => {
            let mut kept = Vec::with_capacity(conditions.len());
            for c in conditions {
                kept.extend(lift(c, unit, origins, semantic)?);
            }
            Ok((!kept.is_empty()).then(|| Condition {
                span: condition.span,
                kind: ConditionKind::And { conditions: kept },
            }))
        }
        ConditionKind::Rank {
            list: RankList::Similar,
            value,
        } if semantic.is_none() => match similar_unit(unit) {
            Some(ranked) => {
                *semantic = Some(Semantic {
                    text: similar_text(value, origins, condition.span)?,
                    unit: ranked,
                    span: condition.span,
                });
                Ok(None)
            }
            None => Ok(Some(condition.clone())),
        },
        _ => Ok(Some(condition.clone())),
    })
}

/// The unit whose similarity ranks `unit`'s rows: its own when it declares
/// `similar`, else its session's.
fn similar_unit(unit: Unit) -> Option<Unit> {
    if crate::fields::lookup(unit, "similar").is_some() {
        Some(unit)
    } else if crate::fields::has_session(unit)
        && crate::fields::lookup(Unit::Sessions, "similar").is_some()
    {
        Some(Unit::Sessions)
    } else {
        None
    }
}

fn similar_text(value: &Value, origins: &[String], span: Span) -> Result<String, QueryError> {
    let Value::Scalar { text, .. } = value else {
        return Err(QueryError::Invalid {
            message: "similar takes similar:VALUE".into(),
            position: span.start,
        });
    };
    // Ranking by a reference's own stored vectors is designed and not built;
    // embedding the reference's spelling as text would be a wrong, paid answer.
    let origin_reference = text.split_once(':').is_some_and(|(origin, rest)| {
        !rest.is_empty() && origins.iter().any(|known| known == origin)
    });
    if origin_reference || polylogue_model::Reference::parse(text).is_ok() {
        return Err(QueryError::NotBuilt {
            what: "similar: by a reference".into(),
        });
    }
    Ok(text.clone())
}

/// The conditions joined to the rest of the `where` by AND, with
/// parenthesized ANDs flattened.
pub fn top_level(c: &Condition) -> Vec<Condition> {
    match &c.kind {
        ConditionKind::And { conditions: all } => all.iter().flat_map(top_level).collect(),
        _ => vec![c.clone()],
    }
}

/// An order condition anywhere outside a nested query: its witness makes
/// the condition a relation boundary.
pub fn contains_order(condition: &Condition) -> bool {
    query_language::depth::deeper(|| match &condition.kind {
        ConditionKind::Order { .. } => true,
        ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
            conditions.iter().any(contains_order)
        }
        ConditionKind::Not { condition } => contains_order(condition),
        _ => false,
    })
}

/// `last` or `latest`: the newest session, resolved inside the query.
pub fn newest_word(condition: &Condition) -> bool {
    match &condition.kind {
        ConditionKind::Reference { reference } => matches!(reference.as_str(), "last" | "latest"),
        ConditionKind::Text {
            text,
            mode: TextMode::Word,
        } => matches!(text.as_str(), "last" | "latest"),
        _ => false,
    }
}

#[cfg(test)]
mod tests;
