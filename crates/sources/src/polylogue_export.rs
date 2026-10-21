//! Polylogue's own export format read back (architecture 7, Export and
//! import): importing an export into a fresh archive gives the same
//! sessions and IDs. Sessions, messages, events and links are written as
//! the export holds them. Sessions retain document order; their messages
//! and events cite the exact exported-session record in the new input.

use crate::document::Sink;
use crate::export::{ExportParser, ItemSink, Owners, ScanError, ScanStats};
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
}

impl ExportParser for Export {
    fn scan(
        &mut self,
        reader: &mut dyn std::io::Read,
        item: &mut ItemSink<'_>,
    ) -> Result<ScanStats, ScanError> {
        self.valid = false;
        self.refusal = None;
        let mut invalid_session = None;
        let (metadata, stats) =
            crate::export::scan_nested_unique(reader, &["sessions"], |range, raw| {
                // Validate every typed session before any of them may be published.
                if invalid_session.is_none()
                    && let Err(error) = serde_json::from_str::<SessionExport>(raw.get())
                {
                    invalid_session = Some(error.to_string());
                }
                item(range, Vec::new(), None, raw)
            })?;
        self.refusal = Document::from_json(&metadata.to_string())
            .err()
            .map(|e| e.to_string())
            .or(invalid_session);
        self.valid = self.refusal.is_none();
        Ok(stats)
    }

    fn begin(
        &mut self,
        length: u64,
        _first: Option<&polylogue_model::Json>,
    ) -> Result<Option<Step>, String> {
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
        value: &polylogue_model::Json,
        sink: &mut Sink<'_>,
        _: &mut dyn Owners,
    ) -> Result<(), String> {
        if !self.valid {
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
