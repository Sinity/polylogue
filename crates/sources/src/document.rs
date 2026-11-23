//! Helpers shared by the parsers of whole documents (exports, captures):
//! a running position for the rows they emit, and small JSON readers.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::Records;
use crate::parse::Output;
use polylogue_model::{Json, JsonExt, MessageId, NativeId, Origin, SessionId};

/// What a document or conversation parser writes into. `next` supplies
/// message positions in reading order. The code that splits an input into
/// records replaces these temporary positions with the record's byte range
/// and sends the reading order to the writer as the record position.
pub struct Sink<'a> {
    pub out: &'a mut Output,
    pub records: &'a mut Records,
    next: u64,
}

impl<'a> Sink<'a> {
    pub fn new(out: &'a mut Output, records: &'a mut Records) -> Self {
        Self {
            out,
            records,
            next: 0,
        }
    }

    /// The next position in the document.
    pub fn position(&mut self) -> u64 {
        let at = self.next;
        self.next += 1;
        at
    }
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

/// The first non-whitespace byte of a document.
pub(crate) fn first_byte(head: &[u8]) -> Option<u8> {
    head.iter().copied().find(|b| !b.is_ascii_whitespace())
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
pub fn time_of(v: &Json) -> Option<crate::time::SourceTime> {
    crate::time::epoch_seconds(v).or_else(|| crate::time::rfc3339(v))
}

/// Text of `content`: a non-empty string, or the non-empty `text` of its
/// parts joined by newlines.
pub fn text_of(content: Option<&Json>) -> Option<String> {
    match content? {
        Json::String(s) if !s.is_empty() => Some(s.clone()),
        Json::Array(parts) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| p.str("text").filter(|s| !s.is_empty()))
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
}
