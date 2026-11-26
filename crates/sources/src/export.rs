//! Export records and conversation scan state, with a Serde reader that splits
//! JSON arrays into records.
/// Version of the rows this module produces.
pub const VERSION: u32 = 2;

parser_modules! {
    mod objects;
    mod state;
    mod path;
    mod keyed;
    mod nested;
}

use crate::document::{BYTE_ORDER_MARK, after_byte_order_mark};
use serde::Deserialize;
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde_json::value::RawValue;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{self, BufReader, Read};
use std::ops::Range;
use std::rc::Rc;

#[derive(Default)]
struct ScanState {
    position: u64,
    last: Option<(u64, u8)>,
    copied: Option<(u64, Vec<u8>)>,
    stats: ScanStats,
    failure: Option<ScanError>,
}

/// Reads one byte at a time and records each byte's position in the input,
/// which counts a leading byte order mark that the reader does not deliver.
struct PositionReader<R> {
    reader: BufReader<R>,
    scan_state: Rc<RefCell<ScanState>>,
    started: bool,
    /// Bytes read ahead while looking for a byte order mark that was not one.
    pending: VecDeque<u8>,
}
impl<R: Read> PositionReader<R> {
    fn new(reader: R, scan_state: &Rc<RefCell<ScanState>>) -> Self {
        Self {
            reader: BufReader::new(reader),
            scan_state: Rc::clone(scan_state),
            started: false,
            pending: VecDeque::new(),
        }
    }

    fn skip_byte_order_mark(&mut self) -> io::Result<()> {
        let mut head = Vec::new();
        (&mut self.reader)
            .take(BYTE_ORDER_MARK.len() as u64)
            .read_to_end(&mut head)?;
        if head == BYTE_ORDER_MARK {
            self.scan_state.borrow_mut().position = head.len() as u64;
        } else {
            self.pending.extend(head);
        }
        Ok(())
    }
}
impl<R: Read> Read for PositionReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if !self.started {
            self.started = true;
            self.skip_byte_order_mark()?;
        }
        if let Some(byte) = self.pending.pop_front() {
            out[0] = byte;
        } else if self.reader.read(&mut out[..1])? == 0 {
            return Ok(0);
        }
        let mut scan_state = self.scan_state.borrow_mut();
        let at = scan_state.position;
        scan_state.position += 1;
        scan_state.last = Some((at, out[0]));
        if let Some((start, bytes)) = &mut scan_state.copied {
            // Copying a map value starts after its colon and whitespace; copying
            // an array element already contains the value's first byte.
            if bytes.is_empty() && (out[0].is_ascii_whitespace() || out[0] == b':') {
                *start = at + 1;
            } else {
                bytes.push(out[0]);
            }
        }
        Ok(1)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("malformed export: {0}")]
    Malformed(serde_json::Error),
    #[error("reading export: {0}")]
    Input(serde_json::Error),
    #[error("export callback: {0}")]
    Callback(String),
    #[error("Serde element bytes differ from observed source bytes")]
    Span,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct ScanStats {
    pub elements: u64,
    pub peak_element_bytes: usize,
}

pub use keyed::scan_map;
pub use nested::{scan_array_or_nested, scan_nested, scan_nested_unique};
pub(crate) use objects::{FieldShape, scan_objects};
pub use path::scan_path;
pub use state::MemoryState;

/// Calls `item` once per complete element. Only the current raw element is
/// buffered. A callback error stops decoding; the outer document is valid only
/// after this returns successfully, including the trailing EOF check.
pub fn scan(
    reader: impl Read,
    mut item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let scan_state = Rc::new(RefCell::new(ScanState::default()));
    let input = PositionReader::new(reader, &scan_state);
    let mut de = serde_json::Deserializer::from_reader(input);
    let result = (|| {
        ArraySeed {
            scan_state: &scan_state,
            item: &mut item,
        }
        .deserialize(&mut de)?;
        de.end()
    })();
    finish_scan(result, &scan_state)
}

/// Reads up to and including the first non-whitespace byte after a byte order
/// mark, which the caller chains back in front of the reader together with the
/// mark and the whitespace. `None` when the input is all whitespace.
pub(crate) fn peek_first_byte<R: std::io::BufRead + ?Sized>(
    reader: &mut R,
) -> Result<(Option<u8>, Vec<u8>), ScanError> {
    let mut prefix = Vec::new();
    loop {
        let buffer = reader
            .fill_buf()
            .map_err(|error| ScanError::Input(serde_json::Error::io(error)))?;
        if buffer.is_empty() {
            return Ok((None, prefix));
        }
        let byte = buffer[0];
        reader.consume(1);
        prefix.push(byte);
        if BYTE_ORDER_MARK.starts_with(&prefix) {
            continue;
        }
        if !byte.is_ascii_whitespace() {
            return Ok((Some(byte), prefix));
        }
    }
}

/// Single-conversation exports use the same item pipeline as an export array.
/// The array path buffers one conversation, and the object path is one conversation.
pub fn scan_array_or_object(
    reader: &mut dyn Input,
    mut item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let (first, prefix) = peek_first_byte(reader)?;
    if first != Some(b'{') {
        return scan(std::io::Cursor::new(prefix).chain(reader), item);
    }
    let start = prefix.len() as u64 - 1;
    let input = std::io::Cursor::new(after_byte_order_mark(&prefix).to_vec()).chain(reader);
    let mut de = serde_json::Deserializer::from_reader(input);
    let raw = Box::<RawValue>::deserialize(&mut de).map_err(classify_error)?;
    de.end().map_err(classify_error)?;
    let length = raw.get().len();
    item(start..start + length as u64, &raw).map_err(ScanError::Callback)?;
    Ok(ScanStats {
        elements: 1,
        peak_element_bytes: length,
    })
}

fn finish_scan(
    result: Result<(), serde_json::Error>,
    scan_state: &Rc<RefCell<ScanState>>,
) -> Result<ScanStats, ScanError> {
    result.map_err(|error| scan_error(error, scan_state))?;
    Ok(scan_state.borrow().stats)
}

fn scan_error(error: serde_json::Error, scan_state: &Rc<RefCell<ScanState>>) -> ScanError {
    scan_state
        .borrow_mut()
        .failure
        .take()
        .unwrap_or_else(|| classify_error(error))
}

/// A reader failure is an input error; anything else is malformed JSON.
fn classify_error(error: serde_json::Error) -> ScanError {
    if error.is_io() {
        ScanError::Input(error)
    } else {
        ScanError::Malformed(error)
    }
}

/// Records a failed callback so `scan_error` reports it instead of Serde's
/// generic message.
fn callback_failed<E: serde::de::Error>(scan_state: &Rc<RefCell<ScanState>>, error: String) -> E {
    scan_state.borrow_mut().failure = Some(ScanError::Callback(error));
    E::custom("export callback failed")
}

/// Checks that Serde's value is the bytes observed at `start`, counts the
/// element and hands it to the callback.
fn emit_element<E: serde::de::Error>(
    scan_state: &Rc<RefCell<ScanState>>,
    start: u64,
    copied: Vec<u8>,
    raw: &RawValue,
    mismatch: &'static str,
    item: impl FnOnce(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<(), E> {
    if !copied.starts_with(raw.get().as_bytes()) {
        scan_state.borrow_mut().failure = Some(ScanError::Span);
        return Err(E::custom(mismatch));
    }
    {
        let mut scan_state = scan_state.borrow_mut();
        scan_state.stats.elements += 1;
        scan_state.stats.peak_element_bytes = scan_state
            .stats
            .peak_element_bytes
            .max(copied.len() + raw.get().len());
    }
    drop(copied);
    item(start..start + raw.get().len() as u64, raw)
        .map_err(|error| callback_failed(scan_state, error))
}

fn map_value<
    'de,
    A: serde::de::MapAccess<'de>,
    F: FnMut(Range<u64>, &RawValue) -> Result<(), String>,
>(
    map: &mut A,
    scan_state: &Rc<RefCell<ScanState>>,
    item: &mut F,
) -> Result<(), A::Error> {
    let start = scan_state.borrow().position;
    scan_state.borrow_mut().copied = Some((start, Vec::new()));
    let raw = map.next_value::<Box<RawValue>>()?;
    let (start, copied) = scan_state.borrow_mut().copied.take().expect("copy started");
    emit_element(
        scan_state,
        start,
        copied,
        &raw,
        "Serde map value differs from observed source bytes",
        item,
    )
}

struct ArraySeed<'a, F> {
    scan_state: &'a Rc<RefCell<ScanState>>,
    item: &'a mut F,
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> DeserializeSeed<'de>
    for ArraySeed<'_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_seq(self)
    }
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> Visitor<'de> for ArraySeed<'_, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an export array")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while let Some(()) = seq.next_element_seed(ItemSeed {
            scan_state: self.scan_state,
            item: self.item,
        })? {}
        Ok(())
    }
}
struct ItemSeed<'a, F> {
    scan_state: &'a Rc<RefCell<ScanState>>,
    item: &'a mut F,
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> DeserializeSeed<'de>
    for ItemSeed<'_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        use serde::de::Error;
        // Serde's IoRead presents the element's first byte to SeqAccess before
        // invoking the seed. Verify that byte and every subsequent byte against
        // RawValue; incompatible read-ahead fails instead of inventing a span.
        let last = self.scan_state.borrow().last;
        let Some((start, first)) = last else {
            self.scan_state.borrow_mut().failure = Some(ScanError::Span);
            return Err(D::Error::custom("missing array element byte"));
        };
        self.scan_state.borrow_mut().copied = Some((start, vec![first]));
        let raw = Box::<RawValue>::deserialize(de)?;
        let (_, copied) = self
            .scan_state
            .borrow_mut()
            .copied
            .take()
            .expect("copy started");
        emit_element(
            self.scan_state,
            start,
            copied,
            &raw,
            "Serde element bytes differ from observed source bytes",
            self.item,
        )
    }
}

/// Numeric export times sort in total-float order; unknowns follow
/// known values. Formats with integer or compound keys supply their own bytes.
pub fn number_order(value: Option<f64>) -> Vec<u8> {
    let Some(value) = value else {
        return vec![1];
    };
    let bits = value.to_bits();
    let bits = if bits >> 63 != 0 {
        !bits
    } else {
        bits ^ (1 << 63)
    };
    let mut key = vec![0];
    key.extend(bits.to_be_bytes());
    key
}

/// Stages original record bytes and their byte spans. Order keys compare
/// lexicographically. Reusing a source key replaces a prior value only for
/// formats whose object entries have last-value semantics.
pub trait ItemSink {
    fn push(
        &mut self,
        range: Range<u64>,
        order: Vec<u8>,
        source_key: Option<&str>,
        raw: &[u8],
    ) -> Result<(), String>;
    fn begin(&mut self) -> Result<(), String> {
        Err("this export sink cannot begin a scan group".into())
    }
    fn commit(&mut self) -> Result<(), String> {
        Err("this export sink cannot commit a scan group".into())
    }
    fn rollback(&mut self) -> Result<(), String> {
        Err("this export sink cannot roll back a scan group".into())
    }
}

impl<F> ItemSink for F
where
    F: FnMut(Range<u64>, Vec<u8>, Option<&str>, &[u8]) -> Result<(), String>,
{
    fn push(
        &mut self,
        range: Range<u64>,
        order: Vec<u8>,
        source_key: Option<&str>,
        raw: &[u8],
    ) -> Result<(), String> {
        self(range, order, source_key, raw)
    }
}

/// A stored input that can reopen from its beginning. Rewinding does not
/// read the provider's file again or change which bytes are parsed.
pub trait Input: std::io::BufRead {
    fn rewind(&mut self) -> io::Result<()>;
}
impl<T: AsRef<[u8]>> Input for std::io::Cursor<T> {
    fn rewind(&mut self) -> io::Result<()> {
        self.set_position(0);
        Ok(())
    }
}
impl<R: Read + std::io::Seek> Input for BufReader<R> {
    fn rewind(&mut self) -> io::Result<()> {
        std::io::Seek::rewind(self)
    }
}

/// Splits an export into records, then parses each in staged order. The
/// default scanner reads JSON array elements; other origins split their own.
pub trait ExportParser: Send {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        scan(reader, |range, raw| {
            item.push(range, self.order_key(raw), None, raw.get().as_bytes())
        })
    }
    /// File-level output after validation and ordering, before its items. `first`
    /// is the selected first record after any keyed replacements. A step with
    /// `repeats_whole_conversations` set makes every later item repeat whole records; the owning
    /// session's facts precede message-only items, whose ranges stay unchanged.
    fn begin(
        &mut self,
        _length: u64,
        _first: Option<&[u8]>,
    ) -> Result<Option<crate::parse::Step>, String> {
        Ok(None)
    }
    /// File order unless the format defines another ordering.
    fn order_key(&self, _raw: &RawValue) -> Vec<u8> {
        Vec::new()
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut crate::document::Sink<'_>,
        first_sessions: &mut dyn ExportState,
    ) -> Result<(), String>;

    /// A message outside its chunk, such as a document's system header.
    /// Other messages use their record's range.
    fn message_range(&self, _message: &polylogue_model::Message) -> Option<Range<u64>> {
        None
    }
}

/// Conversation scan state within one export operation. A native ID seen
/// for the first time returns None; a repeated one returns the session that
/// first used it. Previous-message links are
/// stored separately so chronological records can interleave conversations.
pub trait ExportState {
    fn claim_native_id(
        &mut self,
        native: &str,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::SessionId>, String>;
    fn last_message(
        &mut self,
        session: &polylogue_model::SessionId,
    ) -> Result<Option<polylogue_model::MessageId>, String>;
    fn set_last_message(
        &mut self,
        session: &polylogue_model::SessionId,
        message: &polylogue_model::MessageId,
    ) -> Result<(), String>;
}

pub fn parse_item(
    parser: &mut dyn ExportParser,
    bytes: &[u8],
    range: Range<u64>,
    first_sessions: &mut dyn ExportState,
) -> Result<crate::parse::Step, String> {
    let mut output = crate::parse::Output::default();
    let mut records = crate::Records::default();
    parser.item(
        bytes,
        &mut crate::document::Sink::new(&mut output, &mut records),
        first_sessions,
    )?;
    if let Some(path) = first_repeated_key(bytes) {
        records.repeated_key("export item", &path);
    }
    for row in &mut output.messages {
        let source = parser
            .message_range(&row.message)
            .unwrap_or_else(|| range.clone());
        row.offset = source.start;
        row.end = source.end;
    }
    for member in &mut output.prefix_members {
        if member.source.is_none() {
            member.position[0] = range.start;
        }
    }
    for row in &mut output.documents {
        row.offset = range.start;
        row.end = range.end;
    }
    for fragment in &mut output.fragments {
        fragment.offset = range.start;
        fragment.end = range.end;
    }
    for event in &mut output.events {
        event.range.get_or_insert((range.start, range.end));
    }
    Ok(crate::parse::Step {
        output,
        counts: records.finish().map_err(|e| e.to_string())?,
        start: range.start,
        consumed: range.end,
        repeats_whole_conversations: false,
    })
}

/// The path of the first object key an element repeats, if any. Readers of
/// the element see only the key's last value.
fn first_repeated_key(bytes: &[u8]) -> Option<String> {
    struct Walk<'a> {
        path: &'a mut Vec<String>,
        found: &'a mut Option<String>,
    }
    impl<'de> DeserializeSeed<'de> for Walk<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
            de.deserialize_any(self)
        }
    }
    impl<'de> Visitor<'de> for Walk<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a JSON value")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            let mut keys = std::collections::HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if !keys.insert(key.clone()) && self.found.is_none() {
                    self.path.push(key.clone());
                    *self.found = Some(self.path.join("."));
                    self.path.pop();
                }
                self.path.push(key);
                map.next_value_seed(Walk {
                    path: self.path,
                    found: self.found,
                })?;
                self.path.pop();
            }
            Ok(())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
            self.path.push("[]".into());
            while seq
                .next_element_seed(Walk {
                    path: self.path,
                    found: self.found,
                })?
                .is_some()
            {}
            self.path.pop();
            Ok(())
        }
        fn visit_bool<E>(self, _: bool) -> Result<(), E> {
            Ok(())
        }
        fn visit_i64<E>(self, _: i64) -> Result<(), E> {
            Ok(())
        }
        fn visit_u64<E>(self, _: u64) -> Result<(), E> {
            Ok(())
        }
        fn visit_f64<E>(self, _: f64) -> Result<(), E> {
            Ok(())
        }
        fn visit_str<E>(self, _: &str) -> Result<(), E> {
            Ok(())
        }
        fn visit_unit<E>(self) -> Result<(), E> {
            Ok(())
        }
    }
    let (mut path, mut found) = (Vec::new(), None);
    let mut de = serde_json::Deserializer::from_slice(bytes);
    Walk {
        path: &mut path,
        found: &mut found,
    }
    .deserialize(&mut de)
    .ok()?;
    found
}

/// One export element the parser could not read: counted as unreadable,
/// with no rows, over that element's byte range.
pub fn unreadable_item(error: String, range: Range<u64>) -> crate::parse::Step {
    let mut records = crate::Records::default();
    records.record(
        "export item",
        &polylogue_model::RecordOutcome::unreadable(error),
    );
    crate::parse::Step {
        output: Default::default(),
        counts: records.finish().expect("unreadable item counted"),
        start: range.start,
        consumed: range.end,
        repeats_whole_conversations: false,
    }
}

/// The whole export is refused as malformed JSON: counted as one unreadable
/// record over `length` bytes, with no rows.
pub fn refused(error: String, length: u64) -> crate::parse::Step {
    let mut records = crate::Records::default();
    records.record(
        "export",
        &polylogue_model::RecordOutcome::unreadable(format!("malformed JSON: {error}")),
    );
    crate::parse::Step {
        output: Default::default(),
        counts: records.finish().expect("refusal counted"),
        start: 0,
        consumed: length,
        repeats_whole_conversations: false,
    }
}

#[cfg(test)]
#[path = "export/fixture.rs"]
pub(crate) mod fixture;

#[cfg(test)]
mod tests;
