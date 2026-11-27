//! Helpers shared by the parsers of whole documents (exports, captures):
//! a running position for the rows they emit, and small JSON readers.

/// Parse version of these shared helpers. Every origin's parse version
/// includes it, so raise it when what any helper returns changes.
pub const VERSION: u32 = 1;

parser_modules! {}

use crate::Records;
use crate::parse::Output;
use polylogue_model::{Json, JsonExt, MessageId, NativeId, Origin, SessionId};
use std::collections::BTreeMap;
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

/// What a recognizer sees of a file: its first bytes and, when the whole
/// document was read for it (an import), its top-level structure. A file
/// found at an origin's location is recognized from its first bytes alone.
#[derive(Debug, Clone, Copy)]
pub struct Head<'a> {
    /// The first [`HEAD_LEN`] bytes, after a byte order mark.
    pub bytes: &'a [u8],
    pub shape: Option<&'a Shape>,
}

impl<'a> Head<'a> {
    /// The first bytes alone.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, shape: None }
    }

    /// The kind of the top-level value: from the whole document when it was
    /// read, else from the first byte.
    pub fn top(&self) -> Option<Kind> {
        match self.shape {
            Some(shape) => shape.top,
            None => first_byte(self.bytes).map(|byte| match byte {
                b'{' => Kind::Object,
                b'[' => Kind::Array,
                b'"' => Kind::String,
                _ => Kind::Other,
            }),
        }
    }

    /// Whether the top-level object, or an array's first element, has
    /// `key`. Without the whole document only the first key is known.
    pub(crate) fn has(&self, key: ShapeKey) -> bool {
        match self.shape {
            Some(shape) => shape.keys.contains_key(&key),
            None => first_key(self.bytes).map(ShapeKey::from_str) == Some(key),
        }
    }

    /// Whether the top-level object, or an array's first element, holds
    /// `key` with a value of `kind`. Field order and a long first value do
    /// not matter. Without the whole document only the first key is known,
    /// and its value is assumed to be of `kind`.
    pub(crate) fn holds(&self, key: ShapeKey, kind: Kind) -> bool {
        match self.shape {
            Some(shape) => shape.keys.get(&key) == Some(&kind),
            None => first_key(self.bytes).map(ShapeKey::from_str) == Some(key),
        }
    }
}

/// The kind of a JSON value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Object,
    Array,
    String,
    Other,
}

/// External field names that an origin can ask about during recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ShapeKey {
    Characters,
    ChatMessages,
    #[serde(rename = "chunkedPrompt")]
    ChunkedPrompt,
    ConversationId,
    Conversations,
    CurrentNode,
    Docs,
    Id,
    Mapping,
    Messages,
    PromptTemplate,
    #[serde(rename = "resourceSpans")]
    ResourceSpans,
    Turns,
    Uuid,
    Version,
    #[serde(other)]
    Unknown,
}

impl ShapeKey {
    fn from_str(key: &str) -> Self {
        use serde::de::value::{Error, StrDeserializer};
        serde::Deserialize::deserialize(StrDeserializer::<Error>::new(key)).unwrap_or(Self::Unknown)
    }
}

/// A JSON document's top-level structure: what it is, and the keys of its
/// top-level object (or of an array's first element) used by recognition.
/// Read through the whole stream, retaining only the kinds of known fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Shape {
    /// The kind of the top-level value; `None` for an empty document.
    top: Option<Kind>,
    keys: BTreeMap<ShapeKey, Kind>,
}

impl Shape {
    /// The shape of the JSON document `reader` holds, after a byte order
    /// mark; `None` when it is not one complete JSON value.
    pub fn read(reader: impl Read) -> Option<Shape> {
        let mut reader = io::BufReader::new(reader);
        let mut mark = [0u8; 3];
        let n = read_up_to(&mut reader, &mut mark).ok()?;
        let start: &[u8] = if mark[..n] == *BYTE_ORDER_MARK {
            &[]
        } else {
            &mark[..n]
        };
        serde_json::from_reader::<_, TopLevel>(start.chain(reader))
            .ok()
            .map(|top| top.0)
    }
}

fn read_up_to(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

struct TopLevel(Shape);

impl<'de> serde::Deserialize<'de> for TopLevel {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
        struct Fields;
        impl<'de> Visitor<'de> for Fields {
            type Value = BTreeMap<ShapeKey, Kind>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut keys = BTreeMap::new();
                while let Some(key) = map.next_key::<ShapeKey>()? {
                    if key == ShapeKey::Unknown {
                        map.next_value::<IgnoredAny>()?;
                    } else {
                        keys.insert(key, map.next_value::<ValueKind>()?.0);
                    }
                }
                Ok(keys)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, _: S) -> Result<Self::Value, S::Error> {
                Ok(Default::default())
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
                Ok(Default::default())
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
                Ok(Default::default())
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
                Ok(Default::default())
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
                Ok(Default::default())
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
                Ok(Default::default())
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(Default::default())
            }
        }
        struct Top;
        impl<'de> Visitor<'de> for Top {
            type Value = Shape;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Shape, M::Error> {
                Ok(Shape {
                    top: Some(Kind::Object),
                    keys: Fields.visit_map(map)?,
                })
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Shape, S::Error> {
                let keys = seq
                    .next_element::<Element>()?
                    .map(|e| e.0)
                    .unwrap_or_default();
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Shape {
                    top: Some(Kind::Array),
                    keys,
                })
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Shape, E> {
                Ok(Shape {
                    top: Some(Kind::String),
                    keys: Default::default(),
                })
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Shape, E> {
                Ok(other())
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Shape, E> {
                Ok(other())
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Shape, E> {
                Ok(other())
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Shape, E> {
                Ok(other())
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Shape, E> {
                Ok(other())
            }
        }
        fn other() -> Shape {
            Shape {
                top: Some(Kind::Other),
                keys: Default::default(),
            }
        }
        struct Element(BTreeMap<ShapeKey, Kind>);
        impl<'de> serde::Deserialize<'de> for Element {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                d.deserialize_any(Fields).map(Element)
            }
        }
        d.deserialize_any(Top).map(TopLevel)
    }
}

struct ValueKind(Kind);

impl<'de> serde::Deserialize<'de> for ValueKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
        struct Of;
        impl<'de> Visitor<'de> for Of {
            type Value = Kind;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Kind, M::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Kind::Object)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Kind, S::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Kind::Array)
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Kind, E> {
                Ok(Kind::String)
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
        }
        d.deserialize_any(Of).map(ValueKind)
    }
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
