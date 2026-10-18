//! Full-text search over prose (`text_fts`) and tool output (`output_fts`),
//! and trigram search over commands and paths. The entries are written in
//! the message transaction, so a record is searchable as soon as it is
//! stored.

use crate::Result;

/// Text folded for indexing and for queries the same way.
pub fn fold(_text: &str) -> String {
    // Architecture 5.6: ported text folding from the current rewrite.
    todo!()
}

pub struct Hit {
    pub message_pk: i64,
    pub rank: f64,
    pub snippet: String,
}

pub fn search(
    _reader: &crate::Reader<'_>,
    _query: &str,
    _limit: usize,
    _after: Option<i64>,
) -> Result<Vec<Hit>> {
    // Architecture 7: ranked search with snippets; empty results explained.
    todo!()
}
