//! Polylogue's own export format read back (architecture 7, Export and
//! import): importing an export into a fresh archive gives the same
//! sessions and IDs. Sessions, messages, events and links are written as
//! the export holds them; only positions are new, taken from the
//! document's order, which is the order the export was read in.

use crate::account::{Accounting, Disposition};
use crate::document::Sink;
use crate::parse::{DocumentParser, MessageRow, Output, StepParser, Whole};
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
mod tests {
    use super::*;
    use crate::parse::run_document;
    use polylogue_model::export::SessionExport;
    use polylogue_model::*;

    fn doc() -> Document {
        let sid = SessionId::parse("claude-code:s1").unwrap();
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        session.offer_title("Fix the build", TitleSource::Custom);
        let message = |id: &str, text: &str, role| Message {
            id: MessageId::parse(id).unwrap(),
            session: sid.clone(),
            parent: None,
            role,
            at: Timestamp::from_unix_ms(1_700_000_000_000),
            model: None,
            prompt_kind: None,
            blocks: vec![Block::Text { text: text.into() }],
            usage: Usage::default(),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
            production: Default::default(),
        };
        Document::new(vec![SessionExport {
            session,
            messages: vec![
                message("claude-code:m1", "hello", Role::User),
                message("claude-code:m2", "hi", Role::Assistant),
            ],
            events: vec![Event {
                session: sid.clone(),
                kind: "compaction".into(),
                at: None,
                after: Some(MessageId::parse("claude-code:m1").unwrap()),
                range: Some((10, 20)),
                data: serde_json::json!({"x": 1}),
            }],
            links: vec![],
        }])
    }

    #[test]
    fn recognizes_only_its_own_format() {
        let text = doc().to_json().unwrap();
        assert!(recognize(Path::new("x.json"), text.as_bytes()));
        assert!(!recognize(Path::new("x.json"), br#"{"format": "other"}"#));
        assert!(!recognize(Path::new("x.json"), br#"[{"uuid": "1"}]"#));
    }

    #[test]
    fn reads_back_the_same_ids_in_document_order() {
        let text = doc().to_json().unwrap();
        let step = run_document(&Export, text.as_bytes()).unwrap();
        let ids: Vec<String> = step
            .output
            .messages
            .iter()
            .map(|m| m.message.id.to_string())
            .collect();
        assert_eq!(ids, ["claude-code:m1", "claude-code:m2"]);
        assert!(step.output.messages[0].offset < step.output.messages[1].offset);
        assert_eq!(step.output.sessions[0].id.to_string(), "claude-code:s1");
        assert_eq!(step.output.events[0].range, None);
        assert_eq!(step.accounting.framed, 1);
    }

    #[test]
    fn refuses_a_newer_version_as_a_record() {
        let text = r#"{"format":"polylogue-export","version":99,"sessions":[]}"#;
        let step = run_document(&Export, text.as_bytes()).unwrap();
        assert!(step.output.sessions.is_empty());
        assert_eq!(step.accounting.framed, 1);
    }
}
