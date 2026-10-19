//! Full-text search over prose (`text_fts`) and tool output (`output_fts`),
//! and trigram search over commands and paths. A first build writes the
//! entries in bulk when it ends; afterwards the search thread writes them
//! shortly after their text commits (`search_pending`, architecture 5.5c),
//! and a flush writes every waiting entry before it returns.
//!
//! The indexes are contentless: a hit is a `text.id`, and its snippet is
//! cut from the stored (compressed) text. Text is folded the same way when
//! indexed and when searched ([`polylogue_model::text`]): the tokenizer
//! folds case and accents, and letters with a stroke are folded first, so
//! `lodz` finds `Łódź`.

use crate::writer::unpack_text;
use crate::{Reader, Result};
use polylogue_model::text::{key, tokens};
use rusqlite::{OptionalExtension, params};
use serde::Serialize;

/// Text folded for building queries: lowercase words. The index itself
/// folds case and diacritics in its tokenizer (`unicode61
/// remove_diacritics 2`), so queries only need the same word boundaries.
pub fn fold(text: &str) -> String {
    words(text).join(" ")
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

/// An FTS5 query matching every word, each quoted so no word is read as
/// query syntax. `None` when the text has no words.
pub fn match_all(text: &str) -> Option<String> {
    let w = words(text);
    if w.is_empty() {
        return None;
    }
    Some(
        w.iter()
            .map(|w| format!("\"{w}\""))
            .collect::<Vec<_>>()
            .join(" AND "),
    )
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    pub message_pk: i64,
    pub message_id: String,
    pub session_id: String,
    /// BM25 rank: lower is better.
    pub rank: f64,
    /// Where the hit was: `prose` or `output`.
    pub lane: &'static str,
    pub snippet: String,
}

/// Ranked hits for every word of `query`, prose first then tool output,
/// one hit per message. `after` continues after the previous page's last
/// rank-ordered position (an offset).
pub fn search(
    reader: &Reader<'_>,
    query: &str,
    limit: usize,
    after: Option<i64>,
) -> Result<Vec<Hit>> {
    let Some(fts) = match_all(query) else {
        return Ok(Vec::new());
    };
    let words = words(query);
    let offset = after.unwrap_or(0);
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let conn = reader.conn();
    let mut stmt = conn.prepare_cached(
        "WITH hits AS (
             SELECT rowid AS text_id, bm25(text_fts) AS rank, 'prose' AS lane
               FROM text_fts WHERE text_fts MATCH ?1
             UNION ALL
             SELECT rowid, bm25(output_fts) + 1000.0, 'output'
               FROM output_fts WHERE output_fts MATCH ?1
         ),
         per_message AS (
             SELECT b.message_pk, min(h.rank) AS rank, h.lane, h.text_id
               FROM hits h JOIN block b ON b.text_id = h.text_id
              GROUP BY b.message_pk
         )
         SELECT p.message_pk, m.id, s.id, p.rank, p.lane, t.body
           FROM per_message p
           JOIN message m ON m.pk = p.message_pk
           JOIN session s ON s.pk = m.session_pk
           JOIN text t ON t.id = p.text_id
          ORDER BY p.rank, p.message_pk
          LIMIT ?2 OFFSET ?3",
    )?;
    let rows = stmt.query_map(params![fts, limit, offset], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, f64>(3)?,
            r.get::<_, String>(4)?,
            r.get::<_, Vec<u8>>(5)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (message_pk, message_id, session_id, rank, lane, body) = row?;
        let text = unpack_text(&body)?;
        out.push(Hit {
            message_pk,
            message_id,
            session_id,
            rank: if lane == "output" {
                rank - 1000.0
            } else {
                rank
            },
            lane: if lane == "output" { "output" } else { "prose" },
            snippet: snippet(&text, &words, 160),
        });
    }
    Ok(out)
}

/// About `width` characters of `text` around the first matching word, on
/// one line.
pub fn snippet(text: &str, words: &[String], width: usize) -> String {
    let lower = text.to_lowercase();
    let at = words
        .iter()
        .filter_map(|w| lower.find(w.as_str()))
        .min()
        .unwrap_or(0);
    // `at` is a byte offset in the lowercased text; lowercasing can change
    // lengths, so map it back by characters.
    let char_at = lower[..at.min(lower.len())].chars().count();
    let start = char_at.saturating_sub(width / 3);
    let piece: String = text.chars().skip(start).take(width).collect();
    let mut one_line = piece.split_whitespace().collect::<Vec<_>>().join(" ");
    if start > 0 {
        one_line.insert(0, '…');
    }
    if text.chars().count() > start + width {
        one_line.push('…');
    }
    one_line
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
    /// The best-ranked prose matching `fts` (an FTS5 expression) within
    /// `within`, cut around the first of `terms`.
    pub fn snippet(&self, fts: &str, within: Within, terms: &[String]) -> Result<Option<Snippet>> {
        let (scope, pk) = match within {
            Within::Session(pk) => (
                "m.inherited_from IS NULL AND COALESCE(m.on_active_path, 1) = 1 \
                 AND m.written_by IN ('human', 'model') AND m.session_pk IN \
                 (WITH RECURSIVE tr(pk) AS (SELECT ?2 UNION SELECT l.child_pk FROM link l \
                  JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
                  WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr)",
                pk,
            ),
            Within::Message(pk) => ("m.pk = ?2", pk),
        };
        let sql = format!(
            "SELECT tx.body, s.id, m.id FROM text_fts JOIN text tx ON tx.id = text_fts.rowid \
             JOIN block b ON b.text_id = tx.id AND b.kind = 'text' \
             JOIN message m ON m.pk = b.message_pk JOIN session s ON s.pk = m.session_pk \
             WHERE text_fts MATCH ?1 AND {scope} \
             ORDER BY bm25(text_fts), m.at DESC LIMIT 1"
        );
        let hit: Option<(Vec<u8>, String, String)> = self
            .conn()
            .prepare_cached(&sql)?
            .query_row(params![fts, pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .optional()?;
        let Some((body, session, message)) = hit else {
            return Ok(None);
        };
        let text = unpack_text(&body)?;
        let (text, marks) = cut(&text, terms);
        Ok(Some(Snippet {
            text,
            marks,
            session,
            message,
        }))
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
