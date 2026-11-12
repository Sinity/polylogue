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
    /// The ranked text prefix that can fill a page before row hydration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<RankedSearch>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankedSearch {
    pub index: polylogue_query::search::SnippetIndex,
    pub expression: String,
    /// One-based parameter holding the number of ranked indexed texts.
    /// Its ordinary value is -1, so counts and whole selections stay exact.
    pub limit: usize,
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
    /// query's `params` parameters, one per key; the page size is bound
    /// last. A keyed answer pages its keys, which [`Statements::hydrate`]
    /// turns into rows.
    pub fn page(
        &self,
        order: &[OrderKey],
        params: usize,
        after: bool,
        reverse: bool,
        columns: PageColumns,
    ) -> String {
        let source = match self {
            Statements::Keyed(keyed) => &keyed.keys,
            Statements::Derived(derived) => &derived.rows,
        };
        let projection = match columns {
            PageColumns::All => "*".to_owned(),
            PageColumns::Order => order
                .iter()
                .map(|key| format!("\"{}\"", key.column))
                .collect::<Vec<_>>()
                .join(", "),
        };
        let mut sql = format!("SELECT {projection} FROM ({source})");
        if after && !order.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&after_condition(order, params, reverse));
        }
        if !order.is_empty() {
            sql.push_str(" ORDER BY ");
            sql.push_str(&super::order_by_in_direction(order, reverse));
        }
        let limit = if after {
            params + order.len() + 1
        } else {
            params + 1
        };
        sql.push_str(&format!(" LIMIT ?{limit}"));
        sql
    }

    /// The row columns of the keys bound as one JSON array after the
    /// query's `params` parameters; `None` for a derived relation, whose
    /// page holds its rows.
    // Drive from the page keys. With IN, ANALYZE can choose to scan an
    // entire native unit even though only a few rows need hydration.
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
            .map(|(i, key)| format!("\"{}\" IS ?{}", key.column, params + i + 1))
            .collect::<Vec<_>>()
            .join(" AND ");
        Some(format!(
            "SELECT * FROM ({}) WHERE {predicate} LIMIT 1",
            derived.rows
        ))
    }

    /// `columns` of every selected row, in key order.
    pub fn member_columns(&self, columns: &[String]) -> String {
        let projection = columns
            .iter()
            .map(|c| format!("q.\"{c}\""))
            .collect::<Vec<_>>()
            .join(",");
        match self {
            Statements::Keyed(keyed) => format!(
                "SELECT {projection} FROM ({}) q WHERE q._pk IN (SELECT _pk FROM ({})) ORDER BY q._pk",
                keyed.rows, keyed.keys
            ),
            Statements::Derived(derived) => {
                format!(
                    "SELECT {projection} FROM ({}) q ORDER BY q._pk",
                    derived.rows
                )
            }
        }
    }
}

/// Rows after the bound key values `?{first+1}`...: equal on every earlier
/// key and after on this one, for some key. Reversing the order reverses
/// NULL placement together with each value comparison.
fn after_condition(order: &[OrderKey], first: usize, reverse: bool) -> String {
    let mut any = vec![];
    for (i, key) in order.iter().enumerate() {
        let mut all: Vec<String> = order[..i]
            .iter()
            .enumerate()
            .map(|(j, k)| format!("\"{}\" IS ?{}", k.column, first + j + 1))
            .collect();
        let (column, value) = (format!("\"{}\"", key.column), format!("?{}", first + i + 1));
        all.push(if key.descending ^ reverse {
            format!("({column} < {value} OR ({column} IS NULL AND {value} IS NOT NULL))")
        } else {
            format!("({column} > {value} OR ({value} IS NULL AND {column} IS NOT NULL))")
        });
        any.push(format!("({})", all.join(" AND ")));
    }
    any.join(" OR ")
}
