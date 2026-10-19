//! Polylogue's own export format read back (architecture 7, Export and
//! import): importing an export into a fresh archive gives the same
//! sessions and IDs. Sessions, messages, events and links are written as
//! the export holds them; only positions are new, taken from the
//! document's order, which is the order the export was read in.

use crate::document::Sink;
use crate::parse::{DocumentParser, MessageRow, Output, StepParser, Whole};
use crate::{Accounting, Disposition};
use polylogue_model::export::{Document, FORMAT};
use std::path::Path;

pub const ORIGIN: &str = "polylogue-export";

/// A JSON document whose first key is `format`, naming this format.
pub fn recognize(_path: &Path, head: &[u8]) -> bool {
    crate::document::first_key(head) == Some("format")
        && head
            .windows(FORMAT.len() + 2)
            .any(|w| w[0] == b'"' && &w[1..=FORMAT.len()] == FORMAT.as_bytes())
}

pub fn open(_path: &Path) -> Option<Box<dyn StepParser>> {
    Some(Box::new(Whole(Export)))
}

pub struct Export;

impl DocumentParser for Export {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let text = match std::str::from_utf8(bytes) {
            Ok(t) => t,
            Err(e) => {
                sink.record("export", &Disposition::refused(format!("not UTF-8: {e}")));
                return;
            }
        };
        let doc = match Document::from_json(text) {
            Ok(d) => d,
            Err(e) => {
                sink.record("export", &Disposition::refused(e.to_string()));
                return;
            }
        };
        for s in doc.sessions {
            sink.out.sessions.push(s.session);
            for message in s.messages {
                let at = sink.position();
                sink.out.messages.push(MessageRow {
                    offset: at,
                    end: at,
                    message,
                });
            }
            for mut event in s.events {
                // Its byte range named the original input; here the event
                // is placed after the message it follows.
                event.range = None;
                sink.out.events.push(event);
            }
            sink.out.links.extend(s.links);
            sink.record("session", &Disposition::Materialized);
        }
    }
}

#[cfg(test)]
mod tests;
