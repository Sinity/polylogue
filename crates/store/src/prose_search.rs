//! Full-text search over human and model Text blocks, built per query in an
//! in-memory SQLite FTS5 database from a pinned read. It matches but does not
//! rank. Every term must match the same Message; a phrase must match within
//! one Text block.
use crate::{Error, Result, contributions::MessageView};
use polylogue_model::{WrittenBy, query::TextTerm};
use rusqlite::{Connection, params};
use std::time::Instant;
const MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const CORPUS_BYTES: usize = 32 * 1024 * 1024;
const MAX_MESSAGES: u64 = 20000;
const FTS_SCHEMA: &str = "
    CREATE TABLE docs(rowid INTEGER PRIMARY KEY, message TEXT, session TEXT, body TEXT);
    CREATE VIRTUAL TABLE prose USING fts5(
        body, content='docs', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2'
    );
    CREATE TABLE parts(rowid INTEGER PRIMARY KEY, doc INTEGER, body TEXT);
    CREATE INDEX parts_doc ON parts(doc);
    CREATE VIRTUAL TABLE prose_parts USING fts5(
        body, content='parts', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2'
    );
";
pub struct ProseSearch {
    connection: Connection,
    bytes: usize,
    documents: u64,
}
pub struct ProseHit {
    pub message: String,
    pub session: String,
}
// Folds letters with strokes, which unicode61 does not treat as diacritics.
fn strokes(s: &str) -> String {
    let mut folded = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'ß' | 'ẞ' => folded.push_str("ss"),
            'ł' => folded.push('l'),
            'Ł' => folded.push('L'),
            'ø' => folded.push('o'),
            'Ø' => folded.push('O'),
            'đ' => folded.push('d'),
            'Đ' => folded.push('D'),
            'ħ' => folded.push('h'),
            'Ħ' => folded.push('H'),
            'ı' => folded.push('i'),
            _ => folded.push(c),
        }
    }
    folded
}
fn expression(term: &TextTerm) -> String {
    let (s, prefix) = match term {
        TextTerm::Word(s) | TextTerm::Phrase(s) => (s, false),
        TextTerm::Prefix(s) => (s, true),
    };
    format!(
        "\"{}\"{}",
        strokes(s).replace('"', "\"\""),
        if prefix { "*" } else { "" }
    )
}
impl ProseSearch {
    pub fn new() -> Result<Self> {
        let connection = Connection::open_in_memory().map_err(failure)?;
        connection.execute_batch(FTS_SCHEMA).map_err(failure)?;
        // One open transaction for the whole in-memory build; never committed.
        connection.execute_batch("BEGIN").map_err(failure)?;
        Ok(Self {
            connection,
            bytes: 0,
            documents: 0,
        })
    }
    /// Adds the Text blocks of one human or model Message. A Message or corpus
    /// over the size limits fails the query rather than being searched in part.
    pub fn add(
        &mut self,
        message: &MessageView,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<()> {
        if !matches!(
            message.header()?.written_by,
            WrittenBy::Human | WrittenBy::Model
        ) || !super::contributions::reads::search_membership(message)?
        {
            return Ok(());
        }
        self.connection
            .execute(
                "INSERT INTO docs(message, session) VALUES(?1, ?2)",
                params![message.id().as_str(), message.id().session().as_str()],
            )
            .map_err(failure)?;
        let doc = self.connection.last_insert_rowid();
        let mut body = Vec::new();
        let mut index = 0;
        while let Some(block) = message.block_at(index)? {
            check(until, cancelled)?;
            index += 1;
            if block.kind != "text" {
                continue;
            }
            let mut part = Vec::new();
            let mut offset = 0;
            loop {
                check(until, cancelled)?;
                let Some(range) = block.text_range(offset, 4096)? else {
                    break;
                };
                check_message_size(
                    body.len()
                        .saturating_add(part.len())
                        .saturating_add(range.bytes.len()),
                )?;
                if !range.eof && range.next_offset <= offset {
                    return Err(Error::Invariant(
                        "query prose range made no progress".into(),
                    ));
                }
                part.extend_from_slice(&range.bytes);
                offset = range.next_offset;
                if range.eof {
                    break;
                }
            }
            if part.is_empty() {
                continue;
            }
            if !body.is_empty() {
                body.push(b'\n');
            }
            check_message_size(body.len().saturating_add(part.len()))?;
            body.extend_from_slice(&part);
            let part = String::from_utf8(part).map_err(failure)?;
            self.connection
                .execute(
                    "INSERT INTO parts(doc, body) VALUES(?1, ?2)",
                    params![doc, part],
                )
                .map_err(failure)?;
            self.connection
                .execute(
                    "INSERT INTO prose_parts(rowid, body) VALUES(?1, ?2)",
                    params![self.connection.last_insert_rowid(), strokes(&part)],
                )
                .map_err(failure)?;
        }
        if body.is_empty() {
            return Ok(());
        }
        self.bytes = self.bytes.checked_add(body.len()).ok_or(Error::Range)?;
        if self.bytes > CORPUS_BYTES {
            return Err(Error::ParserAdmission {
                resource: "query selected prose corpus bytes",
                required: self.bytes as u64,
            });
        }
        let body = String::from_utf8(body).map_err(failure)?;
        self.connection
            .execute(
                "UPDATE docs SET body = ?2 WHERE rowid = ?1",
                params![doc, body],
            )
            .map_err(failure)?;
        self.connection
            .execute(
                "INSERT INTO prose(rowid, body) VALUES(?1, ?2)",
                params![doc, strokes(&body)],
            )
            .map_err(failure)?;
        self.documents += 1;
        if self.documents > MAX_MESSAGES {
            return Err(Error::ParserAdmission {
                resource: "query prose Message count",
                required: self.documents,
            });
        }
        Ok(())
    }
    pub fn documents(&self) -> u64 {
        self.documents
    }
    pub fn hits(
        &self,
        terms: &[TextTerm],
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<ProseHit>> {
        let all = terms
            .iter()
            .map(expression)
            .collect::<Vec<_>>()
            .join(" AND ");
        let phrases = terms
            .iter()
            .filter(|term| matches!(term, TextTerm::Phrase(_)))
            .map(expression)
            .collect::<Vec<_>>();
        let mut statement = self
            .connection
            .prepare(
                "SELECT d.rowid, d.message, d.session FROM prose \
                 JOIN docs d ON d.rowid = prose.rowid \
                 WHERE prose MATCH ?1 ORDER BY d.message DESC",
            )
            .map_err(failure)?;
        let rows = statement
            .query_map([all], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    ProseHit {
                        message: r.get(1)?,
                        session: r.get(2)?,
                    },
                ))
            })
            .map_err(failure)?;
        let mut hits = vec![];
        for row in rows {
            check(until, cancelled)?;
            let (doc, hit) = row.map_err(failure)?;
            // Each phrase must match within a single Text block of the Message.
            let mut matched = true;
            for phrase in &phrases {
                check(until, cancelled)?;
                matched = self
                    .connection
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM prose_parts \
                         JOIN parts p ON p.rowid = prose_parts.rowid \
                         WHERE prose_parts MATCH ?1 AND p.doc = ?2)",
                        params![phrase, doc],
                        |r| r.get(0),
                    )
                    .map_err(failure)?;
                if !matched {
                    break;
                }
            }
            if matched {
                hits.push(hit);
            }
        }
        Ok(hits)
    }
}
fn failure(error: impl ToString) -> Error {
    Error::Invariant(error.to_string())
}
fn check_message_size(bytes: usize) -> Result<()> {
    if bytes > MESSAGE_BYTES {
        Err(Error::ParserAdmission {
            resource: "query prose Message bytes",
            required: bytes as u64,
        })
    } else {
        Ok(())
    }
}
fn check(until: Instant, cancelled: &mut dyn FnMut() -> bool) -> Result<()> {
    if Instant::now() >= until || cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
