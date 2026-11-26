//! Helpers shared by the parsers of whole documents (exports, captures):
//! a running position for the rows they emit, and small JSON readers.

/// Parse version of these shared helpers. Every origin's parse version
/// includes it, so raise it when what any helper returns changes.
pub const VERSION: u32 = 1;

parser_modules! {}

use crate::Records;
use crate::parse::Output;
use polylogue_model::{Json, JsonExt, MessageId, NativeId, Origin, SessionId};
use std::io::{self, Read, Seek};

/// The UTF-8 byte order mark some tools write at the start of a text file.
/// It is part of the stored input and of every recorded byte offset, and is
/// not part of the document: every reader starts after it, through the
/// functions below.
pub const BYTE_ORDER_MARK: &[u8] = b"\xEF\xBB\xBF";

/// How many bytes of a document's start a recognizer sees, after the mark.
pub const HEAD_LEN: usize = 64;

/// `bytes` without a leading byte order mark.
pub fn after_byte_order_mark(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(BYTE_ORDER_MARK).unwrap_or(bytes)
}

/// The first [`HEAD_LEN`] bytes of a document, after a byte order mark.
pub fn read_head(reader: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut head = Vec::new();
    reader
        .by_ref()
        .take((HEAD_LEN + BYTE_ORDER_MARK.len()) as u64)
        .read_to_end(&mut head)?;
    let mark = head.len() - after_byte_order_mark(&head).len();
    head.drain(..mark);
    head.truncate(HEAD_LEN);
    Ok(head)
}

/// Seeks to the start of a document: past a byte order mark when there is one.
pub fn rewind_past_byte_order_mark(file: &mut (impl Read + Seek)) -> io::Result<()> {
    file.rewind()?;
    let mut mark = Vec::new();
    file.by_ref()
        .take(BYTE_ORDER_MARK.len() as u64)
        .read_to_end(&mut mark)?;
    if mark != BYTE_ORDER_MARK {
        file.rewind()?;
    }
    Ok(())
}

/// What a document or conversation parser writes into. `next` supplies
/// placeholder message positions; the code that splits an input into records
/// replaces each message's offset and end with the record's byte range. The
/// reading order of messages is their order in the step, not these positions.
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

/// Whether a file's first bytes can begin a stream of JSON object lines: no
/// content yet, or an object.
pub(crate) fn starts_object_lines(head: &[u8]) -> bool {
    first_byte(head).is_none_or(|byte| byte == b'{')
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

#[cfg(test)]
mod tests;
