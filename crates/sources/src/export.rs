//! Export records and conversation state, with Serde framing for JSON arrays.
use serde::Deserialize;
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde_json::value::RawValue;
use std::cell::RefCell;
use std::io::{self, BufReader, Read};
use std::ops::Range;
use std::rc::Rc;

#[derive(Default)]
struct Tap {
    position: u64,
    last: Option<(u64, u8)>,
    captured: Option<(u64, Vec<u8>)>,
    stats: ScanStats,
    failure: Option<ScanError>,
}

struct Tapped<R> {
    reader: R,
    tap: Rc<RefCell<Tap>>,
}
impl<R: Read> Read for Tapped<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let n = self.reader.read(&mut out[..1])?;
        if n == 1 {
            let mut tap = self.tap.borrow_mut();
            let at = tap.position;
            tap.position += 1;
            tap.last = Some((at, out[0]));
            if let Some((start, bytes)) = &mut tap.captured {
                // Map capture starts before its colon and whitespace; array
                // capture already contains the value's first byte.
                if bytes.is_empty() && (out[0].is_ascii_whitespace() || out[0] == b':') {
                    *start = at + 1;
                } else {
                    bytes.push(out[0]);
                }
            }
        }
        Ok(n)
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

mod state;
pub use state::MemoryState;
mod path;
pub use path::scan_path;
mod keyed;
pub use keyed::scan_map;
mod nested;
pub use nested::{scan_array_or_nested, scan_nested, scan_nested_unique};

/// Calls `item` once per complete element. Only the current raw element is
/// buffered. A callback error stops decoding; the outer document is valid only
/// after this returns successfully, including the trailing EOF check.
pub fn scan(
    reader: impl Read,
    mut item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let tap = Rc::new(RefCell::new(Tap::default()));
    let input = Tapped {
        reader: BufReader::new(reader),
        tap: Rc::clone(&tap),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    let result = (|| {
        ArraySeed {
            tap: &tap,
            item: &mut item,
        }
        .deserialize(&mut de)?;
        de.end()
    })();
    finish_scan(result, &tap)
}

fn finish_scan(
    result: Result<(), serde_json::Error>,
    tap: &Rc<RefCell<Tap>>,
) -> Result<ScanStats, ScanError> {
    result.map_err(|error| scan_error(error, tap))?;
    Ok(tap.borrow().stats)
}

fn scan_error(error: serde_json::Error, tap: &Rc<RefCell<Tap>>) -> ScanError {
    tap.borrow_mut().failure.take().unwrap_or_else(|| {
        if error.is_io() {
            ScanError::Input(error)
        } else {
            ScanError::Malformed(error)
        }
    })
}

struct ArraySeed<'a, F> {
    tap: &'a Rc<RefCell<Tap>>,
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
            tap: self.tap,
            item: self.item,
        })? {}
        Ok(())
    }
}
struct ItemSeed<'a, F> {
    tap: &'a Rc<RefCell<Tap>>,
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
        let last = self.tap.borrow().last;
        let Some((start, first)) = last else {
            self.tap.borrow_mut().failure = Some(ScanError::Span);
            return Err(D::Error::custom("missing array element byte"));
        };
        self.tap.borrow_mut().captured = Some((start, vec![first]));
        let raw = Box::<RawValue>::deserialize(de)?;
        let (_, captured) = self
            .tap
            .borrow_mut()
            .captured
            .take()
            .expect("capture started");
        if !captured.starts_with(raw.get().as_bytes()) {
            self.tap.borrow_mut().failure = Some(ScanError::Span);
            return Err(D::Error::custom(
                "Serde element bytes differ from observed source bytes",
            ));
        }
        {
            let mut tap = self.tap.borrow_mut();
            tap.stats.elements += 1;
            tap.stats.peak_element_bytes = tap
                .stats
                .peak_element_bytes
                .max(captured.len() + raw.get().len());
        }
        drop(captured);
        (self.item)(start..start + raw.get().len() as u64, &raw).map_err(|error| {
            self.tap.borrow_mut().failure = Some(ScanError::Callback(error));
            D::Error::custom("export callback failed")
        })
    }
}

/// Numeric export times keep the previous total-float order; unknowns follow
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

/// Stages original record bytes and their physical spans. Order keys compare
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

/// A stable retained view that can reopen from its beginning. Rewinding does
/// not re-observe the provider's file or change which bytes are interpreted.
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

/// Frames original records, then interprets each in staged order. The default
/// scanner reads JSON array elements; other origins supply their own framing.
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
    /// is the selected first record after any keyed replacements.
    fn begin(
        &mut self,
        _length: u64,
        _first: Option<&[u8]>,
    ) -> Result<Option<crate::parse::Step>, String> {
        Ok(None)
    }
    /// Physical record order unless the format defines another ordering.
    fn order_key(&self, _raw: &RawValue) -> Vec<u8> {
        Vec::new()
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut crate::document::Sink<'_>,
        owners: &mut dyn ExportState,
    ) -> Result<(), String>;
}

/// Conversation state within one export operation. A new ownership key returns
/// None; an existing key returns its first owner. Previous-message links are
/// stored separately so chronological records can interleave conversations.
pub trait ExportState {
    fn claim(
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
    owners: &mut dyn ExportState,
) -> Result<crate::parse::Step, String> {
    let mut output = crate::parse::Output::default();
    let mut records = crate::Records::default();
    parser.item(
        bytes,
        &mut crate::document::Sink::new(&mut output, &mut records),
        owners,
    )?;
    for row in &mut output.messages {
        row.offset = range.start;
        row.end = range.end;
    }
    for row in &mut output.documents {
        row.offset = range.start;
        row.end = range.end;
    }
    for event in &mut output.events {
        event.range = Some((range.start, range.end));
    }
    Ok(crate::parse::Step {
        output,
        accounting: records.finish().map_err(|e| e.to_string())?,
        start: range.start,
        consumed: range.end,
        snapshot: None,
    })
}

pub fn refused(error: String, length: u64) -> crate::parse::Step {
    let mut records = crate::Records::default();
    records.record(
        "export",
        &polylogue_model::Disposition::refused(format!("malformed JSON: {error}")),
    );
    crate::parse::Step {
        output: Default::default(),
        accounting: records.finish().expect("refusal accounted"),
        start: 0,
        consumed: length,
        snapshot: None,
    }
}

#[cfg(test)]
#[path = "export/fixture.rs"]
pub(crate) mod fixture;

#[cfg(test)]
mod tests;
