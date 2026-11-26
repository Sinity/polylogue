//! The statements that read one answer (docs/query.md: Paging). A unit's
//! own rows are read key first: the selection's keys and order values,
//! paged by key, then only the page's rows. A derived relation (stages,
//! matches, a measure) is read whole in its order. Either is counted
//! without its order, and its members are its keys without presentation.

use super::OrderKey;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Statements {
    Keyed(KeyedRows),
    Derived(DerivedRows),
}

/// A unit's rows read in two statements: every selected row's key with its
/// order values, of which a page takes its rows' keys, then the row columns
/// of only those keys.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KeyedRows {
    /// `_pk`, `_rank` when ranked, `_near` and the order columns of every
    /// selected row, unordered.
    pub keys: String,
    /// The row columns read from the unit's rows, without the condition
    /// that names which: [`Statements::hydrate`] adds it.
    pub rows: String,
    /// The count of selected rows and of those without a similarity.
    pub count: String,
    /// Text-selection alternatives before row hydration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<TextSearch>,
}

/// The other statements a ranked text search can be read by
/// (docs/search.md: The search leg). Each selects the same rows as
/// [`KeyedRows::keys`] in the same order; the store picks one by the size
/// of the hit set. Every statement is complete as built: its inputs are
/// bound after the query's parameters, never written into them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextSearch {
    pub index: polylogue_query::search::SnippetIndex,
    /// The match expression, for the planning probe and the rank boundary.
    pub expression: String,
    /// The index drives: the keys of rows reached by the best-ranked
    /// indexed texts, how many bound after the query's parameters.
    pub by_index: String,
    /// The non-text scope can drive instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<ScopedSearch>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScopedSearch {
    /// Streams in-scope block occurrences for the bounded planning probe.
    pub blocks: String,
    /// The scope drives: the keys of rows reached through the scope's
    /// blocks.
    pub by_scope: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recent: Option<RecentSearch>,
}

/// Recency drives: candidate rows newest first, tested batch by batch.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecentSearch {
    /// Keys of the scope rows and their final order values.
    pub candidates: String,
    /// The keys among one batch of candidates, bound as a JSON array of
    /// row keys after the query's parameters.
    pub batch: String,
}

/// A derived relation's rows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DerivedRows {
    /// Every row with its columns, in the answer's order.
    pub rows: String,
    /// The count and missing similarities, when a statement cheaper than
    /// counting `rows` gives them.
    pub count: Option<String>,
    /// The selected `_pk` and `_near`, when a statement without the
    /// presentation work gives them.
    pub members: Option<String>,
}

/// What a page statement reads of each row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageColumns {
    /// Every column: the keys of a keyed answer, the rows of a derived one.
    All,
    /// The order values alone, so ordering sorts no row bodies.
    Order,
}

impl Statements {
    /// The rows a relation over this answer reads in order. Only an
    /// answer's own rows are read key first; a stage or combination reads
    /// the derived statement its inner query was lowered to.
    pub(super) fn derived(&self) -> &DerivedRows {
        match self {
            Statements::Derived(rows) => rows,
            Statements::Keyed(_) => {
                unreachable!("a relation's input is lowered to a derived statement")
            }
        }
    }

    /// `count(*)` and the count of rows without a similarity.
    pub fn count(&self, counts_near: bool) -> String {
        match self {
            Statements::Keyed(keyed) => keyed.count.clone(),
            Statements::Derived(DerivedRows {
                count: Some(count), ..
            }) => count.clone(),
            Statements::Derived(DerivedRows { rows, .. }) => format!(
                "SELECT count(*), {} FROM ({rows})",
                if counts_near {
                    "count(*) FILTER (WHERE _near IS NULL)"
                } else {
                    "0"
                }
            ),
        }
    }

    /// The selected rows' `_pk` and `_near` (similarity), unordered.
    pub fn members(&self, counts_near: bool) -> String {
        let near = if counts_near { "_near" } else { "1" };
        match self {
            Statements::Keyed(keyed) => format!("SELECT _pk, _near FROM ({})", keyed.keys),
            Statements::Derived(DerivedRows {
                members: Some(members),
                ..
            }) => members.clone(),
            Statements::Derived(DerivedRows { rows, .. }) => {
                format!("SELECT _pk, {near} AS _near FROM ({rows})")
            }
        }
    }

    /// One page in `order`, `reverse`d or not. With `after`, it holds the
    /// rows that sort after a row whose order values are bound after the
    /// query's `params` parameters, one per key, and `after` says whether
    /// that row is in the second part of a partitioned order
    /// ([`OrderKey::partition`]); the page size is bound last. A keyed answer
    /// pages its keys, which [`Statements::hydrate`] turns into rows.
    pub fn page(
        &self,
        order: &[OrderKey],
        params: usize,
        after: Option<bool>,
        reverse: bool,
        columns: PageColumns,
    ) -> String {
        let source = match self {
            Statements::Keyed(keyed) => &keyed.keys,
            Statements::Derived(derived) => &derived.rows,
        };
        page_of(source, order, params, after, reverse, columns)
    }

    /// The row columns of the keys bound as one JSON array after the
    /// query's `params` parameters; `None` for a derived relation, whose
    /// page holds its rows.
    // Drive from the page keys. With IN, ANALYZE can choose to scan an
    // entire unit table even though only a few rows need hydration.
    pub fn hydrate(&self, params: usize) -> Option<String> {
        match self {
            Statements::Keyed(keyed) => Some(format!(
                "SELECT found.* FROM json_each(?{}) selected CROSS JOIN ({}) found ON found._pk=selected.value",
                params + 1,
                keyed.rows
            )),
            Statements::Derived(_) => None,
        }
    }

    /// The one derived row at the order values bound after the query's
    /// `params` parameters. Reading a derived relation backwards sorts its
    /// order values alone and reads each row at its own, so no row before
    /// the last one visited is decoded.
    pub fn row_at(&self, order: &[OrderKey], params: usize) -> Option<String> {
        let Statements::Derived(derived) = self else {
            return None;
        };
        let predicate = order
            .iter()
            .enumerate()
            .map(|(i, key)| format!("\"{}\" IS {}", key.column, key.bound(params + i + 1)))
            .collect::<Vec<_>>()
            .join(" AND ");
        Some(format!(
            "SELECT * FROM ({}) WHERE {predicate} LIMIT 1",
            derived.rows
        ))
    }
}

/// One page of `source` in `order`, `reverse`d or not. With `after`, it
/// holds the rows that sort after a row whose order values are bound after
/// the source's `params` parameters, one per key, and `after` says whether
/// that row is in the second part of a partitioned order; the page size is
/// bound last.
///
/// An order led by a partition (`t IS NULL, t, ...`) is read as two ordered
/// parts, the rows with a value and then those without, each in the order
/// of the keys after the partition, so an index on `(t, ...)` serves both
/// and a time bound or a continuation seeks in it.
pub fn page_of(
    source: &str,
    order: &[OrderKey],
    params: usize,
    after: Option<bool>,
    reverse: bool,
    columns: PageColumns,
) -> String {
    let select_list = match columns {
        PageColumns::All => "*".to_owned(),
        PageColumns::Order => order
            .iter()
            .map(|key| format!("\"{}\"", key.column))
            .collect::<Vec<_>>()
            .join(", "),
    };
    let limit = if after.is_some() {
        params + order.len() + 1
    } else {
        params + 1
    };
    if order.len() >= 2 && order[0].partition {
        let value = &order[1];
        // The part without a value sorts second in ascending order.
        let mut parts = [false, true];
        if order[0].descending ^ reverse {
            parts.reverse();
        }
        let start = after.map_or(0, |second| usize::from(second == parts[1]));
        let arms: Vec<String> = parts[start..]
            .iter()
            .enumerate()
            .map(|(n, &without)| {
                // Rows without a value order by the keys after it.
                let (keys, first) = if without {
                    (&order[2..], 2)
                } else {
                    (&order[1..], 1)
                };
                let mut condition = format!(
                    "\"{}\" IS {}NULL",
                    value.column,
                    if without { "" } else { "NOT " }
                );
                if n == 0 && after.is_some() && !keys.is_empty() {
                    condition.push_str(" AND ");
                    if !without {
                        // The value is never NULL here, so the first key's
                        // bound is a range the index seeks.
                        let op = if value.descending ^ reverse {
                            "<="
                        } else {
                            ">="
                        };
                        condition.push_str(&format!(
                            "\"{}\" {op} {} AND ",
                            value.column,
                            value.bound(params + 2)
                        ));
                    }
                    condition.push_str(&format!(
                        "({})",
                        after_condition(keys, params + first, reverse)
                    ));
                }
                let mut arm = format!("SELECT {select_list} FROM ({source}) WHERE {condition}");
                if !keys.is_empty() {
                    arm.push_str(" ORDER BY ");
                    arm.push_str(&super::order_by_in_direction(keys, reverse));
                }
                format!("SELECT * FROM ({arm} LIMIT ?{limit})")
            })
            .collect();
        return format!(
            "SELECT * FROM ({}) LIMIT ?{limit}",
            arms.join(" UNION ALL ")
        );
    }
    let mut sql = format!("SELECT {select_list} FROM ({source})");
    if after.is_some() && !order.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&after_condition(order, params, reverse));
    }
    if !order.is_empty() {
        sql.push_str(" ORDER BY ");
        sql.push_str(&super::order_by_in_direction(order, reverse));
    }
    sql.push_str(&format!(" LIMIT ?{limit}"));
    sql
}

/// Rows after the key values bound from `?{params+1}`: equal on every
/// earlier key and after on this one, for some key. Reversing the order
/// reverses NULL placement together with each value comparison.
fn after_condition(order: &[OrderKey], params: usize, reverse: bool) -> String {
    let mut any = vec![];
    for (i, key) in order.iter().enumerate() {
        let mut all: Vec<String> = order[..i]
            .iter()
            .enumerate()
            .map(|(j, k)| format!("\"{}\" IS {}", k.column, k.bound(params + j + 1)))
            .collect();
        let (column, value) = (format!("\"{}\"", key.column), key.bound(params + i + 1));
        all.push(if key.descending ^ reverse {
            format!("({column} < {value} OR ({column} IS NULL AND {value} IS NOT NULL))")
        } else {
            format!("({column} > {value} OR ({value} IS NULL AND {column} IS NOT NULL))")
        });
        any.push(format!("({})", all.join(" AND ")));
    }
    any.join(" OR ")
}
