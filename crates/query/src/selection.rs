//! What a query over a unit selects, in one form however it was written:
//! the unit, the defaults its own conditions leave in place, the conditions
//! that filter its rows, the text that ranks them, the stages that follow,
//! the attached rows and the columns shown (docs/query.md: Defaults,
//! Stages, Attached rows; docs/search.md). `A B`, `A | where B` and
//! `(A) AND (B)` select alike. Field selection and attachment sit beside the
//! selection and never decide which rows it holds.
use crate::{
    QueryError,
    ast::*,
    defaults::Defaults,
    filter::{Binder, Filter, bind_unit},
    search::Search,
    stages::Stage,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub unit: Unit,
    /// The defaults the source's own top-level conditions leave in place.
    pub defaults: Defaults,
    /// The conditions as written, joined by AND: the source's condition
    /// without its `similar:` term, then each `| where` stage that directly
    /// follows the source and means the same as its conjunction.
    pub written: Vec<Condition>,
    /// Those conditions bound to the unit's fields: what selects the rows,
    /// or `None` for every row. An order condition among them makes the
    /// source a relation boundary, which reads them instead.
    pub filter: Option<Filter>,
    /// Each top-level condition outside an order condition, bound, for
    /// `--why` to count alone.
    pub clauses: Vec<Clause>,
    /// What the words search for and which index ranks by them.
    pub search: Search,
    /// The `similar:` text the rows rank by.
    pub semantic: Option<Semantic>,
    pub order: Order,
    /// The index of the first transform that stays a stage; the filter
    /// holds the `| where` stages before it.
    pub stages_from: usize,
    /// The stages from `stages_from` on.
    pub stages: Vec<Stage>,
    /// The index of the first stage that reads the selection's rows as a
    /// relation, or `None` when every stage reads the selection itself.
    pub relations_from: Option<usize>,
    /// A `| measure` stage measures the selection, whose population then
    /// includes hidden sessions.
    pub measured: bool,
    pub attach: Vec<Attached>,
    pub field_selection: FieldSelection,
}

/// Rows attached to each result row (`with messages(id)[last:1]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Attached {
    pub span: Span,
    /// What binding made of the attachment, refused where it is read.
    pub rows: Result<AttachedRows, QueryError>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttachedRows {
    pub unit: Unit,
    pub fields: Vec<String>,
    /// Each window's condition, bound to the unit.
    pub filters: Vec<Filter>,
    /// The defaults all the windows' top-level conditions leave in place.
    pub defaults: Defaults,
    pub first: Option<u64>,
    pub last: Option<u64>,
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
pub enum FieldSelection {
    /// The unit's own columns and the fields the conditions name.
    Default,
    /// Exactly these fields (`--fields`), `prefix.field` paths included.
    Fields(Vec<String>),
}

impl Selection {
    /// The selection of a query whose source is a unit, or `None` for a
    /// saved query or kept result as the source; its columns are the ones
    /// the binder was asked for.
    pub fn of(query: &Query, binder: &Binder<'_>) -> Result<Option<Selection>, QueryError> {
        Self::bind(query, binder, true)
    }

    /// [`Selection::of`], counting each top-level condition alone only when
    /// `counts_alone`: a selection that stands as one condition has no
    /// clauses.
    pub(crate) fn bind(
        query: &Query,
        binder: &Binder<'_>,
        counts_alone: bool,
    ) -> Result<Option<Selection>, QueryError> {
        let origins = binder.origins;
        let fields = binder.fields;
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
        let top = condition.as_ref().map(top_level_refs).unwrap_or_default();
        let stage_start = stages_from(query);
        let relations_from = if condition.as_ref().is_some_and(contains_order) {
            Some(0)
        } else {
            query.transforms[stage_start..]
                .iter()
                .position(|stage| relational(&stage.kind))
                .map(|at| at + stage_start)
        };
        let measured = query.transforms
            [stage_start..relations_from.unwrap_or(query.transforms.len())]
            .iter()
            .any(crate::stages::measures);
        let defaults = Defaults::bind(unit, &top, origins);
        let mut written: Vec<Condition> = condition.into_iter().collect();
        written.extend(query.transforms[..stage_start].iter().filter_map(
            |stage| match &stage.kind {
                TransformKind::Where { condition } => Some(condition.clone()),
                _ => None,
            },
        ));
        let top: Vec<&Condition> = written.iter().flat_map(top_level_refs).collect();
        let (filter, clauses) = if written.iter().any(contains_order) {
            // An order condition's matching row is read by the relation
            // holding it; the conditions beside it still count alone.
            let clauses = top
                .iter()
                .filter(|condition| counts_alone && !contains_order(condition))
                .map(|condition| {
                    Ok(Clause {
                        text: clause_text(condition, unit, binder)?,
                        filter: binder.bind(condition, unit)?,
                    })
                })
                .collect::<Result<_, QueryError>>()?;
            (None, clauses)
        } else {
            // `A | where B` is the conjunction `(A) (B)`: each part keeps
            // its own grouping of search words.
            let filter = match written.as_slice() {
                [] => None,
                [one] => Some(binder.bind(one, unit)?),
                all => Some(Filter::All(
                    all.iter()
                        .map(|c| binder.bind(c, unit))
                        .collect::<Result<_, _>>()?,
                )),
            };
            let mut leaves = vec![];
            if let Some(filter) = &filter {
                conjuncts(filter, &mut leaves);
            }
            debug_assert_eq!(leaves.len(), top.len());
            let clauses = top
                .iter()
                .zip(leaves)
                .filter(|_| counts_alone)
                .map(|(condition, filter)| {
                    Ok(Clause {
                        text: clause_text(condition, unit, binder)?,
                        filter: filter.clone(),
                    })
                })
                .collect::<Result<_, QueryError>>()?;
            (filter, clauses)
        };
        let search = Search::of(filter.as_ref(), unit);
        let order = if !order_read(&query.transforms[stage_start..]) {
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
            clauses,
            search,
            semantic,
            order,
            stages_from: stage_start,
            stages: crate::stages::stages(&query.transforms, stage_start)?,
            relations_from,
            measured,
            attach: query
                .attach
                .iter()
                .map(|attachment| Attached {
                    span: attachment.span,
                    rows: attached_rows(attachment, binder),
                })
                .collect(),
            field_selection: if fields.is_empty() {
                FieldSelection::Default
            } else {
                FieldSelection::Fields(fields.to_vec())
            },
        }))
    }

    /// The written conditions at the top level, parenthesized ANDs
    /// flattened.
    pub fn top_level(&self) -> Vec<Condition> {
        self.written.iter().flat_map(top_level).collect()
    }

    /// The bound conditions joined to the whole filter by AND.
    pub fn conjuncts(&self) -> Vec<&Filter> {
        let mut leaves = vec![];
        if let Some(filter) = &self.filter {
            conjuncts(filter, &mut leaves);
        }
        leaves
    }
}

/// The parts of a bound filter joined by AND, nested conjunctions
/// flattened: binding gives `Filter::All` for exactly the written ANDs, so
/// these are the written top-level conditions, bound.
fn conjuncts<'f>(filter: &'f Filter, out: &mut Vec<&'f Filter>) {
    query_language::depth::deeper(|| match filter {
        Filter::All(all) => all.iter().for_each(|part| conjuncts(part, out)),
        leaf => out.push(leaf),
    })
}

/// A top-level condition written as the query reads it back.
fn clause_text(
    condition: &Condition,
    unit: Unit,
    binder: &Binder<'_>,
) -> Result<String, QueryError> {
    let query = Query {
        source: Source::Unit {
            unit: unit.name().into(),
            implicit: false,
            condition: Some(condition.clone()),
        },
        transforms: vec![],
        attach: vec![],
    };
    let text = query_language::format::format_query(&query, binder.catalog)
        .map_err(|error| QueryError::language(error, binder.text))?;
    let prefix = format!("{} where ", unit.name());
    Ok(text.strip_prefix(&prefix).unwrap_or(&text).to_owned())
}

/// An attachment's unit, fields and windows, its window conditions bound.
fn attached_rows(attachment: &Attachment, binder: &Binder<'_>) -> Result<AttachedRows, QueryError> {
    let unit = bind_unit(&attachment.unit)?;
    let mut filters = vec![];
    let mut top = vec![];
    let (mut first, mut last) = (None, None);
    for window in &attachment.windows {
        match window {
            Window::Where(condition) => {
                top.extend(top_level_refs(condition));
                filters.push(binder.bind(condition, unit)?);
            }
            Window::First(n) => first = Some(*n),
            Window::Last(n) => last = Some(*n),
        }
    }
    Ok(AttachedRows {
        unit,
        fields: attachment.fields.clone(),
        filters,
        defaults: Defaults::bind(unit, &top, binder.origins),
        first,
        last,
    })
}

/// One top-level condition: its text and its bound form.
#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    pub text: String,
    pub filter: Filter,
}

/// How many `| where` stages directly after a unit source join its
/// selection: the index of the first transform that stays a stage. A stage stays a stage when the conjunction would mean
/// something else: an order condition (whose matching row the stage replaces),
/// `latest` (the newest session meeting every condition beside it)
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
    let Value::Scalar { text, quoted } = value else {
        return Err(QueryError::Invalid {
            message: "similar takes similar:VALUE".into(),
            position: span.start,
        });
    };
    // Ranking by a reference's own stored vectors is designed and not built;
    // embedding the reference's spelling as text would be a wrong, paid answer.
    // Quoted words with spaces are prose, even when they begin like an
    // origin's reference (`"codex: rate limit exceeded"`).
    if *quoted && text.contains(char::is_whitespace) {
        return Ok(text.clone());
    }
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
    top_level_refs(c).into_iter().cloned().collect()
}

/// [`top_level`] without copying the conditions.
pub fn top_level_refs(c: &Condition) -> Vec<&Condition> {
    query_language::depth::deeper(|| match &c.kind {
        ConditionKind::And { conditions: all } => all.iter().flat_map(top_level_refs).collect(),
        _ => vec![c],
    })
}

/// An order condition anywhere outside a nested query: its matching row makes
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

/// `latest`: the newest session, resolved inside the query.
pub fn newest_word(condition: &Condition) -> bool {
    match &condition.kind {
        ConditionKind::Reference { reference } => matches!(reference.as_str(), "latest"),
        ConditionKind::Text {
            text,
            mode: TextMode::Word,
        } => matches!(text.as_str(), "latest"),
        _ => false,
    }
}

#[cfg(test)]
mod tests;
