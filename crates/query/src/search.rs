//! The search leg of a selection: the positive words a ranked row's snippet
//! looks for, and the full-text index whose relevance ranks the rows
//! (docs/search.md: Hits and ranking, The search leg).
use crate::{
    ast::*,
    filter::{Comparison, FieldTest, Filter, Test, value_text},
};
use serde::Serialize;

/// The index a snippet's terms search: prose (`text_fts`) or tool output
/// (`output_fts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SnippetIndex {
    Prose,
    Output,
}

/// The positive terms of a top-level condition, which a ranked row's snippet
/// looks for: bare words and phrases search prose, `output:` values tool
/// output. Negated terms and reference words are not among them; an `OR`
/// keeps its branches as alternatives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SnippetTerms {
    /// A word or phrase as written, a trailing `*` marking a prefix.
    Word {
        text: String,
        index: SnippetIndex,
    },
    All(Vec<SnippetTerms>),
    Any(Vec<SnippetTerms>),
}

impl SnippetTerms {
    /// The FTS5 expression of the terms searching `index`.
    pub fn fts(&self, index: SnippetIndex) -> Option<String> {
        let group = |parts: &[SnippetTerms], joiner: &str| {
            let mut parts: Vec<String> = parts.iter().filter_map(|p| p.fts(index)).collect();
            match parts.len() {
                0 => None,
                1 => parts.pop(),
                _ => Some(format!("({})", parts.join(joiner))),
            }
        };
        match self {
            SnippetTerms::Word { text, index: own } if *own == index => {
                polylogue_model::text::fts_phrase(text)
            }
            SnippetTerms::Word { .. } => None,
            SnippetTerms::All(parts) => group(parts, " AND "),
            SnippetTerms::Any(parts) => group(parts, " OR "),
        }
    }

    /// The words and phrases searching `index`, for marking the cut text.
    pub fn words<'t>(&'t self, index: SnippetIndex, out: &mut Vec<&'t str>) {
        match self {
            SnippetTerms::Word { text, index: own } if *own == index => out.push(text),
            SnippetTerms::Word { .. } => {}
            SnippetTerms::All(parts) | SnippetTerms::Any(parts) => {
                for part in parts {
                    part.words(index, out);
                }
            }
        }
    }
}

/// What the words of a selection search for and how they rank it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Search {
    /// What a ranked row's snippet looks for.
    pub terms: Option<SnippetTerms>,
    /// The index whose bm25 relevance ranks the rows: prose when a
    /// top-level word searches the unit's prose, else tool output when a
    /// top-level `output:` term does. `thinking:` and substring terms
    /// filter without ranking.
    pub rank: Option<SnippetIndex>,
}

impl Search {
    /// The search leg of `filter` over `unit`'s rows.
    pub fn of(filter: Option<&Filter>, unit: Unit) -> Search {
        let Some(filter) = filter else {
            return Search::default();
        };
        let words = crate::units::spec(unit).words;
        let mut top = vec![];
        top_level(filter, &mut top);
        let ranks = |index: SnippetIndex| {
            words.ranks(index)
                && top.iter().any(|leaf| match index {
                    SnippetIndex::Prose => matches!(leaf, Filter::Word(_)),
                    SnippetIndex::Output => output_term(leaf).is_some(),
                })
        };
        Search {
            terms: positive_terms(filter, unit),
            rank: [SnippetIndex::Prose, SnippetIndex::Output]
                .into_iter()
                .find(|&index| ranks(index)),
        }
    }
}

/// The leaves joined to the whole filter by AND.
fn top_level<'f>(filter: &'f Filter, out: &mut Vec<&'f Filter>) {
    query_language::depth::deeper(|| match filter {
        Filter::All(all) => all.iter().for_each(|x| top_level(x, out)),
        leaf => out.push(leaf),
    })
}

/// The searched text of an `output:` condition.
fn output_term(filter: &Filter) -> Option<&str> {
    match filter {
        Filter::Field(FieldTest {
            field,
            test: Test::Written(Comparison::Match(value)),
            ..
        }) if field.declaration().name == "output" => value_text(value).ok(),
        _ => None,
    }
}

/// The positive terms of a condition, for the snippet ([`SnippetTerms`]):
/// words under `NOT` and reference words are left out, and an `OR` keeps
/// its branches.
pub fn positive_terms(filter: &Filter, unit: Unit) -> Option<SnippetTerms> {
    let collect = |all: &[Filter]| -> Vec<SnippetTerms> {
        all.iter().filter_map(|x| positive_terms(x, unit)).collect()
    };
    let one =
        |mut parts: Vec<SnippetTerms>, many: fn(Vec<SnippetTerms>) -> SnippetTerms| match parts
            .len()
        {
            0 => None,
            1 => parts.pop(),
            _ => Some(many(parts)),
        };
    query_language::depth::deeper(|| match filter {
        Filter::Word(word) if crate::units::spec(unit).words.plain() => Some(SnippetTerms::Word {
            text: word.text.clone(),
            index: SnippetIndex::Prose,
        }),
        Filter::All(all) => one(collect(all), SnippetTerms::All),
        Filter::Any(any) => one(collect(any), SnippetTerms::Any),
        leaf => output_term(leaf).map(|text| SnippetTerms::Word {
            text: text.to_owned(),
            index: SnippetIndex::Output,
        }),
    })
}
