//! Read Polylogue exports, checking the whole file before emitting anything.
//! Session and document identities are kept. Observations that belong to the
//! input cite the local export record; sender identity and source claims stay
//! in the raw bytes.

use crate::document::Sink;
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::parse::{MessageRow, Output, Parse, Step};
use crate::{RecordOutcome, Records};
use polylogue_model::export::{Document, EventExport, FORMAT, SessionExport};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-export";

/// Recognize the root format discriminator without depending on key order.
/// The supplied head may end after that field; the full scan validates the file.
pub fn recognize(_path: &Path, head: &[u8]) -> bool {
    recognize_reader(head).unwrap_or(false)
}

/// Check a stable input, skipping preceding values without keeping their bodies.
pub fn recognize_reader(reader: impl std::io::Read) -> std::io::Result<bool> {
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    struct Header<'a>(&'a mut bool);
    impl<'de> Visitor<'de> for Header<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a Polylogue export header")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
            while let Some(key) = map.next_key::<String>()? {
                if key == "format" {
                    *self.0 = map.next_value::<String>()? == FORMAT;
                    return Ok(());
                }
                map.next_value::<IgnoredAny>()?;
            }
            Ok(())
        }
    }
    let mut recognized = false;
    // Ending the visitor early may leave a truncated or unconsumed map. Only
    // the observed root discriminator matters here; scan validates the rest.
    let result = serde::Deserializer::deserialize_map(
        &mut serde_json::Deserializer::from_reader(reader),
        Header(&mut recognized),
    );
    if let Err(error) = result
        && error.is_io()
    {
        return Err(std::io::Error::other(error));
    }
    Ok(recognized)
}

pub fn open(_path: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Export::default())))
}

#[derive(Default)]
pub struct Export {
    valid: bool,
    refusal: Option<String>,
    kinds: Vec<Kind>,
    at: usize,
}

#[derive(Clone, Copy)]
enum Kind {
    Session,
    Document,
    Event,
}

impl ExportParser for Export {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.valid = false;
        self.refusal = None;
        self.kinds.clear();
        self.at = 0;
        let mut invalid = None;
        let mut required_version = 1;
        let (mut metadata, stats) = crate::export::scan_objects(
            reader,
            false,
            &[
                ("sessions", crate::export::FieldShape::Array),
                ("documents", crate::export::FieldShape::Array),
                ("events", crate::export::FieldShape::Array),
            ],
            &["format", "version"],
            |_, field, _, range, raw| {
                let kind = match field {
                    "documents" => Kind::Document,
                    "events" => Kind::Event,
                    _ => Kind::Session,
                };
                let valid: Result<u32, String> = (|| match kind {
                    Kind::Document => {
                        let doc = serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
                        Ok(Document::new(vec![])
                            .with_documents(vec![doc])
                            .required_version())
                    }
                    Kind::Event => {
                        let event: EventExport =
                            serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
                        event.validate().map_err(|e| e.to_string())?;
                        Ok(3)
                    }
                    Kind::Session => {
                        let session: SessionExport =
                            serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
                        session.validate().map_err(|e| e.to_string())?;
                        Ok(Document::new(vec![session]).required_version())
                    }
                })();
                match valid {
                    Ok(version) => required_version = required_version.max(version),
                    Err(error) if invalid.is_none() => invalid = Some(error),
                    Err(_) => {}
                }
                self.kinds.push(kind);
                item.push(range, Vec::new(), None, raw.get().as_bytes())
            },
        )?;
        let mut metadata = metadata.pop().expect("root export object").fields;
        // User state is not parsed into sessions: import replays it through
        // the writer. The scan marks it present without keeping it.
        if metadata
            .as_object_mut()
            .and_then(|fields| fields.remove("user_state"))
            .is_some()
        {
            required_version = required_version.max(4);
        }
        self.refusal = Document::from_json(&metadata.to_string())
            .err()
            .map(|e| e.to_string())
            .or(invalid);
        if metadata["version"]
            .as_u64()
            .is_some_and(|version| version < required_version as u64)
        {
            self.refusal = Some(format!(
                "content requires Polylogue export version {required_version}"
            ));
        }
        self.valid = self.refusal.is_none();
        Ok(stats)
    }

    fn begin(&mut self, length: u64, _first: Option<&[u8]>) -> Result<Option<Step>, String> {
        let Some(reason) = self.refusal.take() else {
            return Ok(None);
        };
        let mut records = Records::default();
        records.record("export", &RecordOutcome::unreadable(reason));
        Ok(Some(Step {
            output: Output::default(),
            counts: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            repeats_whole_conversations: false,
        }))
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let value: polylogue_model::Json =
            serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let value = &value;
        if !self.valid {
            return Ok(());
        }
        let kind = self
            .kinds
            .get(self.at)
            .copied()
            .ok_or("export record is outside validated scan")?;
        self.at += 1;
        match kind {
            Kind::Document => {
                let document = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
                let at = sink.position();
                sink.out.documents.push(polylogue_model::DocumentRow {
                    offset: at,
                    end: at,
                    document,
                });
                sink.records.record("document", &RecordOutcome::Used);
                return Ok(());
            }
            Kind::Event => {
                let mut exported: EventExport =
                    serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
                // Sender IDs and coordinates stay claims in the stored export bytes.
                // This observation belongs to this local export record.
                exported.event.range = None;
                sink.out.events.push(exported.event);
                sink.records.record("event", &RecordOutcome::Used);
                return Ok(());
            }
            Kind::Session => {}
        }
        let session: SessionExport =
            serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        sink.out.sessions.push(session.session);
        for message in session.messages {
            let at = sink.position();
            sink.out.messages.push(MessageRow {
                offset: at,
                end: at,
                message,
            });
        }
        for mut event in session.events {
            // An imported event cites this export record, not its former input.
            event.range = None;
            sink.out.events.push(event);
        }
        sink.out.links.extend(session.links);
        sink.records.record("session", &RecordOutcome::Used);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
