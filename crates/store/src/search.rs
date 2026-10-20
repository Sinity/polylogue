//! Full-text search over prose (`text_fts`) and tool output (`output_fts`),
//! and trigram search over commands and paths. Every search runs through
//! the query language (`polylogue_query`), which applies the units'
//! visibility rules; this module splits words as the index does and cuts
//! a query result's snippet. A first build writes the entries in bulk when
//! it ends; afterwards the search thread writes them shortly after their
//! text commits (`search_pending`, architecture 5.5c), and a flush writes
//! every waiting entry before it returns.
//!
//! The indexes are contentless: a hit is a `text.id`, and its snippet is
//! cut from the stored (compressed) text. Text is folded the same way when
//! indexed and when searched ([`polylogue_model::text`]): the tokenizer
//! folds case and accents, and letters with a stroke are folded first, so
//! `lodz` finds `Łódź`.

use crate::encoding::unpack_text;
use crate::{Reader, Result};
use polylogue_model::text::{key, tokens};
use rusqlite::{OptionalExtension, params};
use serde::Serialize;

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
/// the matches' byte ranges (spec query.md, Full-text search, Snippets).
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

impl Reader<'_> {
    /// The best-ranked text matching `fts` (an FTS5 expression) within
    /// `within`, cut around the first of `terms`: prose first, then tool
    /// output when no prose matches.
    pub fn snippet(&self, fts: &str, within: Within, terms: &[String]) -> Result<Option<Snippet>> {
        let (scope, pk) = match within {
            Within::Session(pk) => (
                "m.inherited_from IS NULL AND COALESCE(m.on_active_path, 1) = 1 \
                 AND m.session_pk IN \
                 (WITH RECURSIVE tr(pk) AS (SELECT ?2 UNION SELECT l.child_pk FROM current_link l \
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
                "SELECT tx.body, s.id, m.id FROM {table} JOIN text tx ON tx.id = {table}.rowid \
                 JOIN block b ON b.text_id = tx.id AND b.kind = '{block}' \
                 JOIN current_message m ON m.pk = b.message_pk JOIN session s ON s.pk = m.session_pk \
                 WHERE {table} MATCH ?1 AND {scope}{writer} \
                 ORDER BY bm25({table}), m.at DESC LIMIT 1"
            );
            let hit: Option<(Vec<u8>, String, String)> = self
                .conn()
                .prepare_cached(&sql)?
                .query_row(params![fts, pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .optional()?;
            if let Some((body, session, message)) = hit {
                let text = unpack_text(&body)?;
                let (text, marks) = cut(&text, terms);
                return Ok(Some(Snippet {
                    text,
                    marks,
                    session,
                    message,
                }));
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
