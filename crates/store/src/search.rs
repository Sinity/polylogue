//! Full-text search over prose (`text_fts`) and tool output (`output_fts`),
//! and trigram search over commands and paths. Every search runs through
//! the query language (`polylogue_query`), which applies the units'
//! visibility rules; this module splits words as the index does and cuts
//! a query result's snippet. A first build writes the entries in bulk when
//! it ends; afterwards the search thread writes them shortly after their
//! text commits (`search_pending`), and a flush writes
//! every waiting entry before it returns.
//!
//! The indexes are contentless: a hit is a `text.id`, and its snippet is
//! selected and cut from stored text.
//! Text is folded the same way when
//! indexed and when searched ([`polylogue_model::text`]): the tokenizer
//! folds case and accents, and letters with a stroke are folded first, so
//! `lodz` finds `Łódź`.

use crate::{Reader, Result, StoreError};
use polylogue_model::text::{key, tokens};
use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

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

/// A query result's snippet: about 12 words around the first match, with
/// the matches' byte ranges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Snippet {
    /// The cut text, with `…` where it was cut.
    pub text: String,
    /// Byte ranges of the matches within `text`.
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

struct Hit {
    block: i64,
    session_pk: i64,
    session: String,
    message: String,
}

impl Reader<'_> {
    /// Cut the chosen source blocks around the raw FTS matches.
    pub fn snippets(
        &self,
        fts: &str,
        scopes: &[Within],
        terms: &[String],
    ) -> Result<Vec<Option<Snippet>>> {
        let hits = scopes
            .iter()
            .map(|scope| self.snippet_hit(fts, *scope))
            .collect::<Result<Vec<_>>>()?;
        let mut sessions: BTreeMap<i64, HashMap<i64, Vec<usize>>> = BTreeMap::new();
        for (index, hit) in hits.iter().enumerate() {
            if let Some(hit) = hit {
                sessions
                    .entry(hit.session_pk)
                    .or_default()
                    .entry(hit.block)
                    .or_default()
                    .push(index);
            }
        }
        let mut snippets = vec![None; scopes.len()];
        let cut_from = |block: &polylogue_model::Block,
                        indices: Vec<usize>,
                        snippets: &mut Vec<Option<Snippet>>|
         -> Result<()> {
            let text = match block {
                polylogue_model::Block::Text { text } => text.as_str(),
                polylogue_model::Block::ToolResult {
                    output: Some(text), ..
                } => text.as_str(),
                _ => {
                    return Err(StoreError::Invalid(
                        "selected snippet block has no matching source text".into(),
                    ));
                }
            };
            let (text, marks) = cut(text, terms);
            for index in indices {
                let hit = hits[index].as_ref().expect("chosen hit index");
                snippets[index] = Some(Snippet {
                    text: text.clone(),
                    marks: marks.clone(),
                    session: hit.session.clone(),
                    message: hit.message.clone(),
                });
            }
            Ok(())
        };
        for (_, pending) in sessions {
            for (block_pk, indices) in pending {
                let block = crate::ordered_blocks::one(self.conn()?, block_pk)?;
                cut_from(&block, indices, &mut snippets)?;
            }
        }

        Ok(snippets)
    }

    /// The best-ranked text matching `fts` (an FTS5 expression) within
    /// `within`: prose first, then tool output when no prose matches.
    /// Selection stays raw; chosen blocks are decoded after the batch.
    fn snippet_hit(&self, fts: &str, within: Within) -> Result<Option<Hit>> {
        let (scope, pk) = match within {
            Within::Session(pk) => (
                "m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1 \
                 AND m.session_pk IN \
                 (WITH RECURSIVE tr(pk) AS (SELECT ?2 UNION SELECT l.child_pk FROM link l \
                  JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
                  WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr)",
                pk,
            ),
            Within::Message(pk) => ("m.pk = ?2", pk),
        };
        for (table, block, writer) in [
            (
                "text_fts",
                "text",
                " AND m.written_by IN ('human', 'model')",
            ),
            ("output_fts", "tool_result", ""),
        ] {
            let sql = format!(
                "SELECT b.pk, s.pk, s.id, m.id FROM {table} JOIN text tx ON tx.id = {table}.rowid \
                 JOIN block b ON b.text_id = tx.id AND b.kind = '{block}' \
                 JOIN message m ON m.pk = b.message_pk JOIN session s ON s.pk = m.session_pk \
                 WHERE {table} MATCH ?1 AND {scope}{writer} \
                 ORDER BY bm25({table}), m.at DESC LIMIT 1"
            );
            let hit: Option<Hit> = self
                .conn()?
                .prepare_cached(&sql)?
                .query_row(params![fts, pk], |r| {
                    Ok(Hit {
                        block: r.get(0)?,
                        session_pk: r.get(1)?,
                        session: r.get(2)?,
                        message: r.get(3)?,
                    })
                })
                .optional()?;
            if hit.is_some() {
                return Ok(hit);
            }
        }
        Ok(None)
    }
}

/// `text` cut to [`WINDOW`] words around the first word matching one of
/// `terms`, and the byte ranges of every matching word in the cut.
pub fn cut(text: &str, terms: &[String]) -> (String, Vec<(usize, usize)>) {
    // Each term's words, as the index compares them; a trailing `*` makes
    // the term's last word a prefix.
    let mut wanted: Vec<(String, bool)> = vec![];
    for term in terms {
        let prefix = term.ends_with('*');
        let words: Vec<(usize, usize)> = tokens(term.trim_end_matches('*'));
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
    let start = first.saturating_sub(WINDOW / 3);
    let end = (start + WINDOW).min(words.len());
    let from = words[start].0;
    let to = words[end - 1].1;
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let offset = out.len();
    out.push_str(&text[from..to]);
    if end < words.len() {
        out.push('…');
    }
    let marks = words[start..end]
        .iter()
        .filter(|(s, e)| matches(&text[*s..*e]))
        .map(|(s, e)| (s - from + offset, e - from + offset))
        .collect();
    (out, marks)
}

#[cfg(test)]
mod tests;
