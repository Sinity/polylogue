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
};

#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub unit: Unit,
    /// The defaults the source's own top-level conditions leave in place.
    pub defaults: Vec<DefaultId>,
    /// Conditions joined by AND: the source's condition without its
    /// `similar:` term, then each `| where` stage that directly follows the
    /// source and means the same as its conjunction.
    pub filter: Vec<Condition>,
    /// The `similar:` text the rows rank by.
    pub semantic: Option<Semantic>,
    /// The index of the first transform that stays a stage; the filter
    /// holds the `| where` stages before it.
    pub stages_from: usize,
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
    /// saved query or kept result as the source. `origins` are the origin
    /// names a reference may start with; `fields` the columns asked for.
    pub fn of(
        query: &Query,
        origins: &[String],
        fields: &[String],
    ) -> Result<Option<Selection>, QueryError> {
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
        let measured = query.transforms.iter().any(|stage| {
            matches!(&stage.kind, TransformKind::Extension { name, .. }
                if name == crate::catalog::MEASURE)
        });
        let defaults = defaults::bind(unit, &top, origins, measured);
        let folded = folded_filters(query);
        let mut filter: Vec<Condition> = condition.into_iter().collect();
        filter.extend(
            query.transforms[..folded]
                .iter()
                .filter_map(|stage| match &stage.kind {
                    TransformKind::Where { condition } => Some(condition.clone()),
                    _ => None,
                }),
        );
        Ok(Some(Selection {
            unit,
            defaults,
            filter,
            semantic,
            stages_from: folded,
            measured,
            attach: query.attach.clone(),
            projection: if fields.is_empty() {
                Projection::Default
            } else {
                Projection::Fields(fields.to_vec())
            },
        }))
    }

    /// The filter's own top-level conditions, parenthesized ANDs flattened.
    pub fn top_level(&self) -> Vec<Condition> {
        self.filter.iter().flat_map(top_level).collect()
    }
}

/// How many `| where` stages directly after a unit source join its
/// selection. A stage stays a stage when the conjunction would mean
/// something else: an order condition (whose witness the stage replaces),
/// `last`/`latest` (the newest session meeting every condition beside it)
/// and `| matches` (which reads its pattern from the stage).
fn folded_filters(query: &Query) -> usize {
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
