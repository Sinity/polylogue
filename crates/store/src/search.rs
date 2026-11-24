//! Full-text search over prose (`text_fts`) and tool output (`output_fts`),
//! and trigram search over commands and paths. Every search runs through
//! the query language (`polylogue_query`), which applies the units'
//! visibility rules; this module splits words as the index does, writes the
//! statements that search the full-text files, and cuts a query result's
//! snippet. The entries are held in several files beside the index
//! ([`files`]) and ranked with the statistics of all of them ([`rank`]). A
//! first build fills the files while it composes; afterwards the search
//! thread computes entries shortly after their text commits
//! (`search_pending`) and the writer applies them, and a flush writes every
//! waiting entry before it returns.
//!
//! The indexes are contentless: a hit is a `text.id`, and its snippet is
//! selected and cut from stored text.
//! Text is folded the same way when
//! indexed and when searched ([`polylogue_model::text`]): the tokenizer
//! folds case and accents, and letters with a stroke are folded first, so
//! `lodz` finds `Łódź`.

use crate::{Reader, Result, StoreError};
use polylogue_model::text::{key, tokens};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchDriver {
    Index,
    Scope,
    Recency,
}

/// The words of `text` as the index splits them: runs of letters and
/// digits, lowercased, letters with a stroke folded as the index folds
/// them. Punctuation in pasted paths and identifiers splits words instead
/// of failing the query.
pub fn words(text: &str) -> Vec<String> {
    polylogue_model::text::fold(text)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Words around the first match in a query result's snippet.
const WINDOW: usize = 12;

/// The most characters a snippet's text carries, whatever the words' length.
const MAX_CHARS: usize = 400;

/// A query result's snippet: about 12 words around the first match, with
/// the matches' character ranges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Snippet {
    /// The cut text, with `…` where it was cut.
    pub text: String,
    /// Character (code point) ranges of the matches within `text`.
    pub marks: Vec<(usize, usize)>,
    /// The session and message the text came from (a subagent's, when a
    /// session matched through one).
    pub session: String,
    pub message: String,
}

/// Where a query result's hit is looked for.
#[derive(Debug, Clone, Copy)]
pub enum Within {
    /// A session and its subagent tree: own, active-path prose.
    Session(i64),
    Message(i64),
}

pub use polylogue_query::search::SnippetIndex;

/// The positive bound search terms whose first matching index supplies a snippet.
#[derive(Debug, Clone)]
pub struct SnippetQuery<'a> {
    pub index: SnippetIndex,
    pub terms: &'a [polylogue_query::search::SnippetTerms],
}

impl SnippetQuery<'_> {
    pub(crate) fn fts(&self) -> Option<String> {
        let expressions = self
            .terms
            .iter()
            .filter_map(|term| term.fts(self.index))
            .collect::<Vec<_>>();
        (!expressions.is_empty()).then(|| expressions.join(" AND "))
    }

    fn words(&self) -> Vec<&str> {
        let mut words = Vec::new();
        for term in self.terms {
            term.words(self.index, &mut words);
        }
        words
    }
}

pub(crate) mod files;
mod hits;
mod rank;

/// The full-text table `index` names in each file.
pub(crate) fn table(index: SnippetIndex) -> &'static str {
    match index {
        SnippetIndex::Prose => "text_fts",
        SnippetIndex::Output => "output_fts",
    }
}

/// The summed statistics of every file for the expression bound at
/// `expression`, as the CTE `search_stats` ([`rank`]).
fn statistics(table: &str, expression: &str) -> String {
    let files = (0..files::FILES)
        .map(|n| {
            format!(
                "SELECT pl_search_stats(f.{table},t.entries,t.tokens) AS s \
                 FROM {file}.{table} f CROSS JOIN {file}.totals t ON t.name='{table}' \
                 WHERE f.{table} MATCH '(' || {expression} || ') OR \"' || char(57344) || '\"' AND f.rowid=0",
                file = files::schema(n)
            )
        })
        .collect::<Vec<_>>()
        .join(" UNION ALL ");
    format!("search_stats AS MATERIALIZED (SELECT pl_search_total(s) AS s FROM ({files}))")
}

/// A statement of `(text_id, rank)` for the texts in `index` matching the
/// expression bound at `expression`, ranked over every file as over one;
/// lower ranks first. `texts` names a relation of `text_id`s that bounds
/// the texts looked up; `limit` keeps the best that many, ties by key.
pub(crate) fn ranked_texts(
    index: SnippetIndex,
    expression: &str,
    texts: Option<&str>,
    limit: Option<&str>,
) -> String {
    let table = table(index);
    let arms = (0..files::FILES)
        .map(|n| {
            let file = files::schema(n);
            let rank = format!("pl_bm25(f.{table},(SELECT s FROM search_stats))");
            match texts {
                Some(texts) => format!(
                    "SELECT t.text_id,{rank} AS rank FROM {texts} t \
                     CROSS JOIN {file}.{table} f ON f.rowid=t.text_id WHERE f.{table} MATCH {expression}"
                ),
                None => format!(
                    "SELECT f.rowid AS text_id,{rank} AS rank FROM {file}.{table} f \
                     WHERE f.{table} MATCH {expression} AND f.rowid>0"
                ),
            }
        })
        .collect::<Vec<_>>()
        .join(" UNION ALL ");
    let limit = limit.map_or_else(String::new, |limit| {
        format!(" ORDER BY rank,text_id LIMIT {limit}")
    });
    format!(
        "WITH {} SELECT text_id,rank FROM ({arms}){limit}",
        statistics(table, expression)
    )
}

/// A statement of the `text_id`s in `index` matching the expression bound at
/// `expression`, unranked, in no order.
pub(crate) fn matching_texts(index: SnippetIndex, expression: &str) -> String {
    let table = table(index);
    (0..files::FILES)
        .map(|n| {
            format!(
                "SELECT f.rowid AS text_id FROM {}.{table} f WHERE f.{table} MATCH {expression} AND f.rowid>0",
                files::schema(n)
            )
        })
        .collect::<Vec<_>>()
        .join(" UNION ALL ")
}

/// The same source wins query ranking and snippet selection.
pub(crate) fn best_source(source: &str, group: &str) -> String {
    format!(
        "SELECT * FROM (SELECT *,row_number() OVER(PARTITION BY {group} ORDER BY rank,at DESC,message,block) AS choice FROM {source}) WHERE choice=1"
    )
}

#[derive(Clone)]
struct Hit {
    /// Which of the queries found it.
    query: usize,
    block: i64,
    session_pk: i64,
    session: String,
    message: String,
}

impl Reader<'_> {
    /// Cut the chosen source blocks around the raw FTS matches. The queries
    /// are tried in order for each scope; the first that finds text cuts it.
    pub fn snippets(
        &self,
        queries: &[SnippetQuery<'_>],
        scopes: &[Within],
    ) -> Result<Vec<Option<Snippet>>> {
        let hits = self.snippet_hits(queries, scopes)?;
        self.cut_snippets(queries, hits)
    }

    /// Cut sources already chosen by the query's identical ranked expression.
    pub(crate) fn snippets_from_blocks(
        &self,
        query: &SnippetQuery<'_>,
        blocks: &[i64],
    ) -> Result<Vec<Option<Snippet>>> {
        let wanted = serde_json::to_string(blocks)?;
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(
            "SELECT b.pk,m.session_pk,s.id,m.id FROM json_each(?1) wanted \
             CROSS JOIN block b ON b.pk=wanted.value CROSS JOIN message m ON m.pk=b.message_pk \
             CROSS JOIN session s ON s.pk=m.session_pk",
        )?;
        let selected = statement
            .query_map([wanted], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    Hit {
                        query: 0,
                        block: row.get(0)?,
                        session_pk: row.get(1)?,
                        session: row.get(2)?,
                        message: row.get(3)?,
                    },
                ))
            })?
            .collect::<std::result::Result<HashMap<_, _>, _>>()?;
        let hits = blocks
            .iter()
            .map(|block| {
                selected
                    .get(block)
                    .cloned()
                    .map(Some)
                    .ok_or_else(|| StoreError::internal("selected hit block is unavailable"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.cut_snippets(std::slice::from_ref(query), hits)
    }

    fn cut_snippets(
        &self,
        queries: &[SnippetQuery<'_>],
        hits: Vec<Option<Hit>>,
    ) -> Result<Vec<Option<Snippet>>> {
        let words = queries.iter().map(SnippetQuery::words).collect::<Vec<_>>();
        let mut sessions: BTreeMap<i64, HashMap<(i64, usize), Vec<usize>>> = BTreeMap::new();
        for (index, hit) in hits.iter().enumerate() {
            if let Some(hit) = hit {
                sessions
                    .entry(hit.session_pk)
                    .or_default()
                    .entry((hit.block, hit.query))
                    .or_default()
                    .push(index);
            }
        }
        let mut snippets = vec![None; hits.len()];
        for (_, pending) in sessions {
            for ((block_pk, query), indices) in pending {
                let block = crate::ordered_blocks::one(self.conn()?, block_pk)?;
                let text = match &block {
                    polylogue_model::Block::Text { text } => text.as_str(),
                    polylogue_model::Block::ToolResult {
                        output: Some(text), ..
                    } => text.as_str(),
                    polylogue_model::Block::ToolCall {
                        written_text: Some(text),
                        ..
                    } => text.as_str(),
                    _ => {
                        return Err(StoreError::internal(
                            "selected snippet block has no matching source text",
                        ));
                    }
                };
                let (text, marks) = cut(text, &words[query]);
                for index in indices {
                    let hit = hits[index].as_ref().expect("chosen hit index");
                    snippets[index] = Some(Snippet {
                        text: text.clone(),
                        marks: marks.clone(),
                        session: hit.session.clone(),
                        message: hit.message.clone(),
                    });
                }
            }
        }
        Ok(snippets)
    }
}

/// `text` cut to [`WINDOW`] words around the first word matching one of
/// `terms`, and the character ranges of every matching word in the cut. A
/// term ending in `*` matches its last word as a prefix.
pub fn cut(text: &str, terms: &[impl AsRef<str>]) -> (String, Vec<(usize, usize)>) {
    // Each term's words, as the index compares them.
    let mut wanted: Vec<(String, bool)> = vec![];
    for term in terms {
        let term = term.as_ref();
        let prefix = term.ends_with('*');
        let term = term.trim_end_matches('*');
        let words: Vec<(usize, usize)> = tokens(term);
        let n = words.len();
        for (i, (s, e)) in words.into_iter().enumerate() {
            wanted.push((key(&term[s..e]), prefix && i + 1 == n));
        }
    }
    let words = tokens(text);
    let matches = |w: &str| {
        let k = key(w);
        wanted.iter().any(|(t, prefix)| {
            if *prefix {
                k.starts_with(t.as_str())
            } else {
                k == *t
            }
        })
    };
    if words.is_empty() {
        return (text.chars().take(80).collect(), vec![]);
    }
    let first = words
        .iter()
        .position(|(s, e)| matches(&text[*s..*e]))
        .unwrap_or(0);
    let mut start = first.saturating_sub(WINDOW / 3);
    let mut end = (start + WINDOW).min(words.len());
    // A word is a run of letters and digits, so one of tool output can be a
    // whole encoded file: drop the words farthest from the match until the
    // cut fits, and cut a single over-long word itself.
    let length = |start: usize, end: usize| text[words[start].0..words[end - 1].1].chars().count();
    while end - start > 1 && length(start, end) > MAX_CHARS {
        if end - 1 > first {
            end -= 1;
        } else {
            start += 1;
        }
    }
    let from = words[start].0;
    let mut to = words[end - 1].1;
    let mut clipped = false;
    if let Some((bytes, _)) = text[from..to].char_indices().nth(MAX_CHARS) {
        to = from + bytes;
        clipped = true;
    }
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let offset = out.chars().count();
    out.push_str(&text[from..to]);
    if clipped || end < words.len() {
        out.push('…');
    }
    let cut = &text[from..to];
    let chars = |bytes: usize| cut[..bytes].chars().count() + offset;
    let marks = words[start..end]
        .iter()
        .filter(|(s, e)| *s < to && matches(&text[*s..*e]))
        .map(|(s, e)| (chars(s - from), chars((*e).min(to) - from)))
        .collect();
    (out, marks)
}

#[cfg(test)]
mod tests;
