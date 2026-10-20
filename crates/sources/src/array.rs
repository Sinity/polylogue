//! Serde framing of export array elements, preserving their original lexemes.
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
            if let Some((_, bytes)) = &mut tap.captured {
                bytes.push(out[0]);
            }
        }
        Ok(n)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct ScanStats {
    pub elements: u64,
    pub peak_element_bytes: usize,
}

/// Calls `item` once per complete element. Only the current raw element is
/// buffered. A callback error stops decoding; the outer document is valid only
/// after this returns successfully, including the trailing EOF check.
pub fn scan(
    reader: impl Read,
    mut item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> serde_json::Result<ScanStats> {
    let tap = Rc::new(RefCell::new(Tap::default()));
    let input = Tapped {
        reader: BufReader::new(reader),
        tap: Rc::clone(&tap),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    ArraySeed {
        tap: &tap,
        item: &mut item,
    }
    .deserialize(&mut de)?;
    de.end()?;
    let stats = tap.borrow().stats;
    Ok(stats)
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
        let (start, first) = self
            .tap
            .borrow()
            .last
            .ok_or_else(|| D::Error::custom("missing array element byte"))?;
        self.tap.borrow_mut().captured = Some((start, vec![first]));
        let raw = Box::<RawValue>::deserialize(de)?;
        let (_, captured) = self
            .tap
            .borrow_mut()
            .captured
            .take()
            .expect("capture started");
        if !captured.starts_with(raw.get().as_bytes()) {
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
        (self.item)(start..start + raw.get().len() as u64, &raw).map_err(D::Error::custom)
    }
}

/// An export item parser. Ordering is separate from the physical record span.
pub trait ArrayParser: Send {
    fn order_key(&self, raw: &RawValue) -> Option<f64>;
    fn item(
        &self,
        value: &polylogue_model::Json,
        sink: &mut crate::document::Sink<'_>,
        owners: &mut dyn Owners,
    ) -> Result<(), String>;
}

/// Message ownership across the conversations in one export operation.
pub trait Owners {
    fn claim(
        &mut self,
        native: &str,
        session: &polylogue_model::SessionId,
    ) -> Result<polylogue_model::SessionId, String>;
}

pub fn parse_item(
    parser: &dyn ArrayParser,
    value: &polylogue_model::Json,
    range: Range<u64>,
    owners: &mut dyn Owners,
) -> Result<crate::parse::Step, String> {
    let mut output = crate::parse::Output::default();
    let mut records = crate::Records::default();
    parser.item(
        value,
        &mut crate::document::Sink::new(&mut output, &mut records),
        owners,
    )?;
    for row in &mut output.messages {
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
#[path = "array/fixture.rs"]
pub(crate) mod fixture;

#[cfg(test)]
mod tests;
