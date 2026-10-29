//! Current canonical bundle records, framed one record at a time.
pub const VERSION: u32 = 1;
pub const ORIGIN: &str = "polylogue-bundle";
use crate::RecordOutcome;
use crate::document::Sink;
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::parse::{Parse, Step};
use polylogue_model::bundle::{Manifest, Record};
use std::ops::Range;
use std::path::Path;

/// Stream canonical record values, or an inventory-named array of record values.
/// Only one record is buffered; the manifest and inventory are validated by ingress.
pub fn visit(
    reader: &mut dyn std::io::BufRead,
    mut accept: impl FnMut(Range<u64>, Record, &[u8]) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let mut prefix = Vec::new();
    let first = loop {
        let bytes = reader
            .fill_buf()
            .map_err(|error| ScanError::Callback(error.to_string()))?;
        if bytes.is_empty() {
            break None;
        }
        let byte = bytes[0];
        reader.consume(1);
        prefix.push(byte);
        if !byte.is_ascii_whitespace() {
            break Some(byte);
        }
    };
    let input = std::io::Read::chain(std::io::Cursor::new(prefix), reader);
    let mut one = |range, raw: &serde_json::value::RawValue| {
        let record: Record = serde_json::from_str(raw.get()).map_err(|error| error.to_string())?;
        record.validate().map_err(|error| error.to_string())?;
        accept(range, record, raw.get().as_bytes())
    };
    if first == Some(b'[') {
        return crate::export::scan(input, one);
    }
    let mut stream = serde_json::Deserializer::from_reader(input)
        .into_iter::<Box<serde_json::value::RawValue>>();
    let mut stats = ScanStats::default();
    while let Some(raw) = stream.next() {
        let raw = raw.map_err(ScanError::Malformed)?;
        let end = stream.byte_offset() as u64;
        one(end - raw.get().len() as u64..end, &raw).map_err(ScanError::Callback)?;
        stats.elements += 1;
        stats.peak_element_bytes = stats.peak_element_bytes.max(raw.get().len());
    }
    Ok(stats)
}

pub fn manifest(reader: impl std::io::Read) -> Result<Manifest, String> {
    let mut stream = serde_json::Deserializer::from_reader(std::io::BufReader::new(reader))
        .into_iter::<Record>();
    match stream.next() {
        Some(Ok(Record::Manifest(manifest))) => {
            manifest.validate().map_err(|error| error.to_string())?;
            Ok(manifest)
        }
        Some(Ok(_)) | None => Err("a Polylogue bundle starts with its manifest".into()),
        Some(Err(error)) => Err(error.to_string()),
    }
}
pub fn recognize(_: &Path, head: &[u8]) -> bool {
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    struct Header<'a>(&'a mut bool, &'a mut bool);
    struct Data<'a>(&'a mut bool);
    impl<'de> Visitor<'de> for Data<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a bundle manifest discriminator")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
            while let Some(key) = map.next_key::<String>()? {
                if key == "format" {
                    *self.0 = map.next_value::<String>()? == polylogue_model::bundle::FORMAT;
                    return Ok(());
                }
                map.next_value::<IgnoredAny>()?;
            }
            Ok(())
        }
    }
    impl<'de> serde::de::DeserializeSeed<'de> for Data<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            d.deserialize_map(self)
        }
    }
    impl<'de> Visitor<'de> for Header<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a bundle manifest record")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
            while let Some(key) = map.next_key::<String>()? {
                if key == "record" {
                    *self.0 = map.next_value::<String>()? == "manifest";
                } else if key == "data" {
                    map.next_value_seed(Data(&mut *self.1))?;
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
                if *self.0 && *self.1 {
                    return Ok(());
                }
            }
            Ok(())
        }
    }
    let (mut tag, mut format) = (false, false);
    let _ = serde::Deserializer::deserialize_map(
        &mut serde_json::Deserializer::from_slice(head),
        Header(&mut tag, &mut format),
    );
    tag && format
}
pub fn open(_: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Bundle)))
}
struct Bundle;
impl ExportParser for Bundle {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        visit(reader, |range, record, raw| {
            let mut order = Vec::new();
            if matches!(record, Record::Manifest(_)) {
                order.push(0);
            } else {
                order.push(1);
                order.extend_from_slice(&record.commit().unwrap_or(0).to_be_bytes());
                if let Some(reference) = record.reference() {
                    order.extend_from_slice(reference.as_bytes());
                }
            }
            item.push(range, order, None, raw)
        })
    }
    fn begin(&mut self, _: u64, _: Option<&[u8]>) -> Result<Option<Step>, String> {
        Ok(None)
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let record: Record = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        let at = sink.position();
        let kind = match record {
            Record::Manifest(_) => "manifest",
            Record::Session(record) => {
                sink.out.sessions.push(record.session);
                "session"
            }
            Record::Message(record) => {
                let index = sink.out.messages.len();
                sink.out.message_order.insert(index, record.order);
                sink.out.message_projections.insert(
                    index,
                    polylogue_model::bundle::MessageProjection {
                        source_version: record.source_version,
                        block_count: record.block_count,
                        block_ordinals: record.block_ordinals,
                    },
                );
                sink.out.messages.push(polylogue_model::MessageRow {
                    offset: at,
                    end: at,
                    message: record.message,
                });
                "message"
            }
            Record::Event(record) => {
                let index = sink.out.events.len();
                sink.out.imported_events.insert(index, record.identity);
                let mut event = record.event;
                event.range = Some((at, at));
                sink.out.events.push(event);
                "event"
            }
            Record::Document(record) => {
                sink.out.documents.push(polylogue_model::DocumentRow {
                    offset: at,
                    end: at,
                    document: record.document,
                });
                "document"
            }
            Record::Link(record) => {
                sink.out.links.push(record.link);
                "link"
            }
            Record::Authored(_) => {
                sink.records.record(
                    "authored",
                    &RecordOutcome::skipped("historical state restored by import"),
                );
                return Ok(());
            }
            Record::Variant(_) => {
                sink.records.record(
                    "variant",
                    &RecordOutcome::skipped("historical state restored by import"),
                );
                return Ok(());
            }
            Record::VariantPart(_) => {
                sink.records.record(
                    "variant_part",
                    &RecordOutcome::skipped("historical state restored by import"),
                );
                return Ok(());
            }
            Record::Blob(_) => "blob",
        };
        sink.records.record(kind, &RecordOutcome::Used);
        Ok(())
    }
}
