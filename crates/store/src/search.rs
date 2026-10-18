//! Full-text search over prose (`text_fts`) and tool output (`output_fts`),
//! and trigram search over commands and paths. The entries are written in
//! the message transaction, so a record is searchable as soon as it is
//! stored.
//!
//! The indexes are contentless: a hit is a `text.id`, and its snippet is
//! cut from the stored (compressed) text.

use crate::{Reader, Result};
use rusqlite::params;
use serde::Serialize;

/// Text folded for building queries: lowercase words. The index itself
/// folds case and diacritics in its tokenizer (`unicode61
/// remove_diacritics 2`), so queries only need the same word boundaries.
pub fn fold(text: &str) -> String {
    words(text).join(" ")
}

/// The words of `text` as the index splits them: runs of letters and
/// digits, lowercased. Punctuation in pasted paths and identifiers splits
/// words instead of failing the query.
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
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
        let text = crate::writer::unpack_text(&body)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_splits_words_instead_of_failing() {
        assert_eq!(
            words("/realm/project/polylogue: cargo-nextest!"),
            ["realm", "project", "polylogue", "cargo", "nextest"]
        );
        assert_eq!(match_all("a.b").as_deref(), Some("\"a\" AND \"b\""));
        assert_eq!(match_all("--- !!"), None);
    }

    #[test]
    fn snippets_centre_on_the_first_match() {
        let text = format!(
            "{} the needle is here {}",
            "x ".repeat(200),
            "y ".repeat(200)
        );
        let s = snippet(&text, &["needle".into()], 60);
        assert!(s.contains("needle"));
        assert!(s.starts_with('…') && s.ends_with('…'));
    }
}
