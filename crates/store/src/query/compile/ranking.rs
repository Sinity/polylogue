//! How a selection's rows rank, as a value lowering returns
//! (docs/search.md: Hits and ranking, Fusion), and the lowering context
//! that says where a condition stands.

use super::*;
use polylogue_query::search::SnippetIndex;
use std::cell::RefCell;

/// What the rows a selection lowers to are for.
#[derive(Debug, Clone, Default)]
pub(super) struct Reading {
    /// The answer itself, whose rows are read key first and whose ranked
    /// text search may run by one of its plans; otherwise the input of a
    /// stage or combination, read whole.
    pub answer: bool,
    /// The keys of the rows entering a stage: the selection keeps only
    /// these.
    pub within: Option<String>,
    /// A `similar:` term may rank this selection.
    pub similar: bool,
}

impl Reading {
    pub fn answer() -> Self {
        Self {
            answer: true,
            within: None,
            similar: true,
        }
    }

    /// A selection read whole, as a stage, combination or sub-query reads it.
    pub fn whole() -> Self {
        Self {
            answer: false,
            within: None,
            similar: true,
        }
    }
}

/// Where a condition is lowered: inside a measured selection or not, and
/// at the selection's own top level or below it.
#[derive(Clone, Copy)]
pub(crate) struct Cx<'l> {
    /// A measured population keeps the sessions a default would hide.
    pub measured: bool,
    /// At the top level, where the first term over the search leg's index
    /// ranks the rows.
    pub top: Option<&'l TopLevel>,
    /// The `similar:` text the selection ranks by: a query in its
    /// conditions may rank by no other.
    pub similar: Option<&'l str>,
    /// A top-level alternate or language condition of a messages selection
    /// admits these alternate texts (all languages when empty) to its
    /// prose searches.
    pub alternates: Option<&'l [String]>,
}

impl<'l> Cx<'l> {
    pub fn plain(measured: bool) -> Self {
        Self {
            measured,
            top: None,
            similar: None,
            alternates: None,
        }
    }

    /// The same selection below its top level: under OR, NOT, a relation
    /// or another unit's rows.
    pub fn inner(self) -> Self {
        Self { top: None, ..self }
    }

    pub fn is_top(&self) -> bool {
        self.top.is_some()
    }
}

/// The top level of one selection's conditions. Lowering puts the ranking
/// of the first term over the search leg's index here.
pub(crate) struct TopLevel {
    /// The index whose relevance ranks the rows, from the search leg.
    pub index: Option<SnippetIndex>,
    /// The answer's ranked text term may be read by search plans: the rows
    /// are the answer's own, not measured and not ranked by `similar:`.
    pub plans: bool,
    pub ranking: RefCell<Option<TextRank>>,
}

impl TopLevel {
    pub fn take(self) -> Option<TextRank> {
        self.ranking.into_inner()
    }
}

/// Full-text relevance: the rank expression, lower first, and the hit
/// that supplies a row's time and snippet source.
#[derive(Debug, Clone)]
pub(crate) struct TextRank {
    pub expression: String,
    pub source: Option<units::HitSource>,
    pub leg: Option<SearchLeg>,
}

/// A ranked text term that search plans can read: the statements that
/// read it differ only in the CTEs from `ctes.start` to `ctes.end` of the
/// compiler's list, which select the hits.
#[derive(Debug, Clone)]
pub(crate) struct SearchLeg {
    pub index: SnippetIndex,
    /// The match expression and its parameter.
    pub expression: String,
    pub parameter: String,
    /// The CTE of `(text_id, rank)`; its hit CTEs are named after it.
    pub name: String,
    pub ctes: std::ops::Range<usize>,
    /// The hit CTEs reading the scope's candidate blocks, when the scope
    /// may drive.
    pub scoped: Option<Vec<String>>,
    /// The words this term searches, removed from the scope's conditions.
    pub spans: Vec<Span>,
}

impl SearchLeg {
    pub fn table(&self) -> &'static str {
        match self.index {
            SnippetIndex::Prose => "text_fts",
            SnippetIndex::Output => "output_fts",
        }
    }

    /// The `(text_id, rank)` CTE over the whole index, or its best `limit`
    /// texts.
    pub fn indexed(&self, limit: Option<&str>) -> String {
        let table = self.table();
        let prefix = limit.map_or_else(String::new, |limit| {
            format!(" ORDER BY {table}.rank LIMIT {limit}")
        });
        format!(
            "{} AS MATERIALIZED (SELECT rowid AS text_id, bm25({table}) AS rank FROM {table} WHERE {table} MATCH {}{prefix})",
            self.name, self.parameter
        )
    }

    pub fn candidates(&self) -> String {
        format!("{}_scope_candidates", self.name)
    }
}

/// Similarity to a `similar:` text: its expression, lower first, and the
/// join bringing it to the rows.
#[derive(Debug, Clone)]
pub(super) struct Similarity {
    pub expression: String,
    pub join: Option<String>,
}

/// How a selection's rows rank, lower first.
#[derive(Debug, Clone)]
pub(super) enum Ranking {
    Text(TextRank),
    Similar(Similarity),
    /// Words and `similar:` together, by reciprocal rank fusion over the
    /// rows the conditions keep: `join` brings each row's fused rank.
    Fused {
        join: String,
        expression: String,
        similarity: Similarity,
    },
}

impl Ranking {
    pub fn expression(&self) -> &str {
        match self {
            Ranking::Text(text) => &text.expression,
            Ranking::Similar(similar) => &similar.expression,
            Ranking::Fused { expression, .. } => expression,
        }
    }

    /// Each row's similarity, when `similar:` ranks the rows.
    pub fn similarity(&self) -> Option<&str> {
        match self {
            Ranking::Text(_) => None,
            Ranking::Similar(similarity) | Ranking::Fused { similarity, .. } => {
                Some(&similarity.expression)
            }
        }
    }

    /// The hit a text ranking chose for each row.
    pub fn hit(&self) -> Option<&units::HitSource> {
        match self {
            Ranking::Text(text) => text.source.as_ref(),
            _ => None,
        }
    }

    pub fn leg(&self) -> Option<&SearchLeg> {
        match self {
            Ranking::Text(text) => text.leg.as_ref(),
            _ => None,
        }
    }

    /// The joins a row reads its ranking through.
    pub fn joins(&self) -> Vec<&str> {
        match self {
            Ranking::Text(_) => vec![],
            Ranking::Similar(similarity) => similarity.join.iter().map(String::as_str).collect(),
            Ranking::Fused {
                join, similarity, ..
            } => similarity
                .join
                .iter()
                .map(String::as_str)
                .chain([join.as_str()])
                .collect(),
        }
    }
}
