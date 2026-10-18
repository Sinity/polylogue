//! The one writer. It receives batches from the parse pool and writes each
//! in one large transaction on a long-lived connection: rows, their
//! full-text entries and the parse position they cover, together.

use crate::Result;
use polylogue_model::{BlobHash, Event, Link, Message, Session, SessionId, ToolOutcome};
use rusqlite::Connection;

/// One input's parse step, as the writer stores it.
pub struct Batch {
    pub input: i64,
    /// Where the next step starts and the parser's carried state, committed
    /// with the rows they cover.
    pub position: u64,
    pub carry: serde_json::Value,
    /// Facts to merge into sessions: titles by precedence, working
    /// directories appended, earliest start.
    pub sessions: Vec<Session>,
    /// `(offset, message)`: a row at the same input and offset replaces the
    /// earlier one; another offset is another version of the message.
    pub messages: Vec<(u64, Message)>,
    pub events: Vec<Event>,
    /// Upserted by (kind, parent, child), filling fields one side lacks.
    pub links: Vec<Link>,
    pub background: Vec<(SessionId, String, ToolOutcome)>,
    pub blobs: Vec<(BlobHash, Vec<u8>)>,
    /// The step's accounting, refused by the parser crate if it did not add
    /// up; stored for `status` and `unrecognized`.
    pub accounting: serde_json::Value,
}

pub struct Writer {
    _archive: Connection,
    _index: Connection,
}

impl Writer {
    /// Writes one transaction. Batches are grouped until the transaction
    /// holds a few thousand rows or a live append waits.
    pub fn write(&mut self, _batches: Vec<Batch>) -> Result<()> {
        // Architecture 5.3, 5.5, 5.6: rows, message_source, preferred version,
        // FTS in the same transaction, parse_state.
        todo!()
    }
}
