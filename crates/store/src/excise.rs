//! Excision: permanent removal from every file and derived product. The
//! `excision` row keeps the content from coming back by any route.

use crate::Result;
use polylogue_model::{MessageId, SessionId};

pub enum ExciseTarget {
    Session(SessionId),
    Message(MessageId),
    /// Every occurrence of a byte string, for a leaked secret.
    Text(String),
}

pub fn excise(_target: &ExciseTarget) -> Result<u64> {
    // Architecture 5.6: blobs, rows, FTS, embeddings; excision row.
    todo!()
}
