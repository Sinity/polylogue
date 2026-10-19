//! Helpers shared by the parsers of whole documents (exports, captures):
//! a running position for the rows they emit, and small JSON readers.

use crate::parse::Output;
use crate::{Accounting, Disposition};
use polylogue_model::{Json, MessageId, NativeId, Origin, SessionId, Timestamp};

/// What a document parser writes into. A document has no byte offsets
/// for its items, so rows take their order from `next`: one number per
/// message in document order, unique in the input (architecture 5.5a,
/// the order key).
pub struct Sink<'a> {
    pub out: &'a mut Output,
    pub acc: &'a mut Accounting,
    next: u64,
}

impl<'a> Sink<'a> {
    pub fn new(out: &'a mut Output, acc: &'a mut Accounting) -> Self {
        Self { out, acc, next: 0 }
    }

    /// The next position in the document.
    pub fn position(&mut self) -> u64 {
        let at = self.next;
        self.next += 1;
        at
    }

    /// Frames one record of the document and records how it ended.
    pub fn record(&mut self, kind: &str, disposition: &Disposition) {
        self.acc.framed += 1;
        self.acc.add(kind, disposition);
    }
}

pub fn str_of<'a>(v: &'a Json, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Json::as_str).filter(|s| !s.is_empty())
}

pub fn string_of(v: &Json, key: &str) -> Option<String> {
    str_of(v, key).map(str::to_owned)
}

pub fn session_id(origin: &str, native: &str) -> Option<SessionId> {
    Some(SessionId::new(
        Origin::new(origin).ok()?,
        NativeId::new(native).ok()?,
    ))
}

pub fn message_id(origin: &str, native: &str) -> Option<MessageId> {
    Some(MessageId::new(
        Origin::new(origin).ok()?,
        NativeId::new(native).ok()?,
    ))
}

/// The first key of the first object in a document's first bytes: in
/// `[{"uuid": …` it is `uuid`, in `{"source": …` it is `source`. Exports
/// that share a file name (`conversations.json`) are told apart by it.
pub fn first_key(head: &[u8]) -> Option<&str> {
    let mut rest = head;
    let skip = |r: &mut &[u8]| {
        while let [b, tail @ ..] = *r
            && b.is_ascii_whitespace()
        {
            *r = tail;
        }
    };
    skip(&mut rest);
    if let [b'[', tail @ ..] = rest {
        rest = tail;
        skip(&mut rest);
    }
    let [b'{', tail @ ..] = rest else {
        return None;
    };
    rest = tail;
    skip(&mut rest);
    let [b'"', tail @ ..] = rest else {
        return None;
    };
    let end = tail.iter().position(|&b| b == b'"')?;
    std::str::from_utf8(&tail[..end]).ok()
}

/// A time written either as epoch seconds or as an RFC 3339 string.
pub fn time_of(v: &Json) -> Option<Timestamp> {
    crate::time::epoch_seconds(v).or_else(|| crate::time::rfc3339(v))
}
