//! Read connections. The daemon keeps a pool; every read in one request
//! sees one consistent snapshot of both databases.

use crate::Result;
use polylogue_model::{Message, Session, SessionId};

pub struct ReadPool {
    _size: usize,
}

/// One consistent read: a transaction on `index.db` with `archive.db`
/// attached.
pub struct Reader<'a> {
    _pool: &'a ReadPool,
}

impl ReadPool {
    pub fn reader(&self) -> Result<Reader<'_>> {
        // Architecture 6: pool of read connections.
        todo!()
    }
}

impl Reader<'_> {
    pub fn session(&self, _id: &SessionId) -> Result<Option<Session>> {
        // Architecture 4.4: session row.
        todo!()
    }

    /// The session's messages in order, the preferred version of each,
    /// with inherited ones when `logical` is set.
    pub fn messages(&self, _id: &SessionId, _logical: bool) -> Result<Vec<Message>> {
        // Architecture 5.4, 5.5: lineage and preferred version.
        todo!()
    }

    /// Runs a compiled query's SQL.
    pub fn query(
        &self,
        _sql: &str,
        _params: &[rusqlite::types::Value],
    ) -> Result<Vec<serde_json::Value>> {
        // Architecture 7: query execution with pages bound to this read.
        todo!()
    }
}
