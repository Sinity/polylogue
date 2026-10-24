//! Polylogue's own export format read back (architecture 7, Export and
//! import): importing an export into a fresh archive gives the same
//! sessions and IDs. Sessions, messages, events and links are written as
//! the export holds them. Sessions retain document order; their messages
//! and events cite the exact exported-session record in the new input.

use crate::document::Sink;
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::parse::{MessageRow, Output, Parse, Step};
use crate::{Disposition, Records};
use polylogue_model::export::{Document, FORMAT, SessionExport};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-export";

/// A JSON document whose first key is `format`, naming this format.
pub fn recognize(_path: &Path, head: &[u8]) -> bool {
    crate::document::first_key(head) == Some("format")
        && head
            .windows(FORMAT.len() + 2)
            .any(|w| w[0] == b'"' && &w[1..=FORMAT.len()] == FORMAT.as_bytes())
}

pub fn open(_path: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Export::default())))
}

#[derive(Default)]
pub struct Export {
    valid: bool,
    refusal: Option<String>,
    kinds: Vec<bool>,
    at: usize,
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
        let (mut metadata, stats) = crate::export::scan_objects(
            reader,
            false,
            &[
                ("sessions", crate::export::FieldShape::Array),
                ("documents", crate::export::FieldShape::Array),
            ],
            &["format", "version"],
            |_, field, _, range, raw| {
                let document = field == "documents";
                let valid = if document {
                    serde_json::from_str::<polylogue_model::Document>(raw.get()).map(|_| ())
                } else {
                    serde_json::from_str::<SessionExport>(raw.get()).map(|_| ())
                };
                if invalid.is_none() {
                    invalid = valid.err().map(|e| e.to_string());
                }
                self.kinds.push(document);
                item.push(range, Vec::new(), None, raw.get().as_bytes())
            },
        )?;
        let metadata = metadata.pop().expect("root export object").fields;
        self.refusal = Document::from_json(&metadata.to_string())
            .err()
            .map(|e| e.to_string())
            .or(invalid);
        if self.kinds.contains(&true) && metadata["version"].as_u64() != Some(2) {
            self.refusal = Some("documents require Polylogue export version 2".into());
        }
        self.valid = self.refusal.is_none();
        Ok(stats)
    }

    fn begin(&mut self, length: u64, _first: Option<&[u8]>) -> Result<Option<Step>, String> {
        let Some(reason) = self.refusal.take() else {
            return Ok(None);
        };
        let mut records = Records::default();
        records.record("export", &Disposition::refused(reason));
        Ok(Some(Step {
            output: Output::default(),
            accounting: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            snapshot: None,
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
        let document = self
            .kinds
            .get(self.at)
            .copied()
            .ok_or("export record is outside validated scan")?;
        self.at += 1;
        if document {
            let document = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
            let at = sink.position();
            sink.out.documents.push(polylogue_model::DocumentRow {
                offset: at,
                end: at,
                document,
            });
            sink.records.record("document", &Disposition::Materialized);
            return Ok(());
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
        sink.records.record("session", &Disposition::Materialized);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
