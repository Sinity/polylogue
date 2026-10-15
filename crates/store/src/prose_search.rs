//! Query-local FTS corpus built ONLY from a pinned selected-body view. This
//! does not publish an archive index or give staged text ranking authority.
use crate::{Error, Result, contributions::MessageView};
use polylogue_model::{WrittenBy, query::TextTerm};
use rusqlite::{Connection, params};
use std::time::Instant;
const MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const CORPUS_BYTES: usize = 32 * 1024 * 1024;
pub struct ProseSearch {
    connection: Connection,
    bytes: usize,
    documents: u64,
}
pub struct ProseHit {
    pub message: String,
    pub session: String,
    pub rank: f64,
    pub at: Option<i64>,
}
// unicode61 owns case/diacritic tokenization. Stroke letters are explicitly
// folded before insertion/query. No snippet is cut from unmasked context.
fn strokes(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'ł' => 'l',
            'Ł' => 'L',
            'ø' => 'o',
            'Ø' => 'O',
            'đ' => 'd',
            'Đ' => 'D',
            'ð' => 'd',
            'Ð' => 'D',
            'ħ' => 'h',
            'Ħ' => 'H',
            'ŧ' => 't',
            'Ŧ' => 'T',
            _ => c,
        })
        .collect()
}
impl ProseSearch {
    pub fn new() -> Result<Self> {
        let connection =
            Connection::open_in_memory().map_err(|e| Error::Invariant(e.to_string()))?;
        connection.execute_batch("CREATE TABLE docs(rowid INTEGER PRIMARY KEY, message TEXT, session TEXT, body TEXT, at INTEGER); CREATE VIRTUAL TABLE prose USING fts5(body,content='docs',content_rowid='rowid',tokenize='unicode61 remove_diacritics 2');").map_err(|e|Error::Invariant(e.to_string()))?;
        Ok(Self {
            connection,
            bytes: 0,
            documents: 0,
        })
    }
    /// Whole Message text has explicit admission. Rejection fails the query;
    /// it never reports a bounded prefix as the whole searched population.
    pub fn add(
        &mut self,
        message: &MessageView,
        until: Instant,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<()> {
        let header = message.header()?;
        if !matches!(header.written_by, WrittenBy::Human | WrittenBy::Model) {
            return Ok(());
        }
        let read = super::contributions::reads::search_membership(message)?;
        if !read {
            return Ok(());
        }
        let mut body = Vec::new();
        let mut index = 0;
        while let Some(block) = message.block_at(index)? {
            check(until, cancelled)?;
            index += 1;
            if block.kind != "text" {
                continue;
            }
            if !body.is_empty() {
                body.push(b'\n');
                if body.len() > MESSAGE_BYTES {
                    return Err(Error::ParserAdmission {
                        resource: "query prose Message bytes",
                        required: body.len() as u64,
                    });
                }
            }
            let mut offset = 0;
            loop {
                check(until, cancelled)?;
                let Some(range) = block.text_range(offset, 4096)? else {
                    break;
                };
                if body.len().saturating_add(range.bytes.len()) > MESSAGE_BYTES {
                    return Err(Error::ParserAdmission {
                        resource: "query prose Message bytes",
                        required: (body.len() + range.bytes.len()) as u64,
                    });
                }
                if !range.eof && range.next_offset <= offset {
                    return Err(Error::Invariant(
                        "query prose range made no progress".into(),
                    ));
                }
                body.extend_from_slice(&range.bytes);
                offset = range.next_offset;
                if range.eof {
                    break;
                }
            }
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
        let body = String::from_utf8(body).map_err(|e| Error::Invariant(e.to_string()))?;
        let at = super::contributions::reads::search_time(message)?;
        self.connection
            .execute(
                "INSERT INTO docs(message,session,body,at) VALUES(?1,?2,?3,?4)",
                params![
                    message.id().as_str(),
                    message.id().session().as_str(),
                    body,
                    at
                ],
            )
            .map_err(|e| Error::Invariant(e.to_string()))?;
        self.connection
            .execute(
                "INSERT INTO prose(rowid,body) VALUES(?1,?2)",
                params![self.connection.last_insert_rowid(), strokes(&body)],
            )
            .map_err(|e| Error::Invariant(e.to_string()))?;
        self.documents += 1;
        if self.documents > 20000 {
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
        let expression = terms
            .iter()
            .map(|t| {
                let (s, prefix) = match t {
                    TextTerm::Word(s) | TextTerm::Phrase(s) => (s, false),
                    TextTerm::Prefix(s) => (s, true),
                };
                format!(
                    "\"{}\"{}",
                    strokes(s).replace('"', "\"\""),
                    if prefix { "*" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join(" AND ");
        let mut statement=self.connection.prepare("SELECT d.message,d.session,bm25(prose),d.at FROM prose JOIN docs d ON d.rowid=prose.rowid WHERE prose MATCH ?1 ORDER BY bm25(prose),d.at IS NULL,d.at DESC,d.message DESC").map_err(|e|Error::Invariant(e.to_string()))?;
        let rows = statement
            .query_map([expression], |r| {
                Ok(ProseHit {
                    message: r.get(0)?,
                    session: r.get(1)?,
                    rank: r.get(2)?,
                    at: r.get(3)?,
                })
            })
            .map_err(|e| Error::Invariant(e.to_string()))?;
        let mut hits = vec![];
        for row in rows {
            check(until, cancelled)?;
            hits.push(row.map_err(|e| Error::Invariant(e.to_string()))?);
        }
        Ok(hits)
    }
}
fn check(until: Instant, cancelled: &mut dyn FnMut() -> bool) -> Result<()> {
    if Instant::now() >= until || cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
