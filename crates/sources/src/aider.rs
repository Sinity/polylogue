//! Aider's role-prefixed LLM log and its companion Markdown. Request
//! snapshots repeat the sent context; equal prefixes are the same messages.
use crate::document::{Sink, message_id, session_id};
use crate::export::{ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use crate::parse::{DocumentParser, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Block, Document, DocumentId, DocumentPurpose, DocumentRow, Event, Json, Message, NativeId,
    Origin, Role, Session, SessionKind, Usage,
};
use std::path::Path;

pub const VERSION: u32 = 1;
pub const ORIGIN: &str = "aider";
pub const HISTORY: &str = ".aider.llm.history";
pub const COMPANION: &str = ".aider.chat.history.md";

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|name| name == HISTORY) && head.starts_with(b"TO LLM ")
}
pub fn open(path: &Path) -> Option<Parse> {
    match path.file_name()?.to_str()? {
        HISTORY => Some(Parse::Export(Box::new(History::default()))),
        COMPANION => Some(Parse::Document(Box::new(Companion))),
        _ => None,
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum LogRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Default)]
pub struct History {
    native: Option<String>,
    context: Vec<(LogRole, String)>,
    next: u64,
    last: Option<polylogue_model::MessageId>,
}

impl ExportParser for History {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn Input,
        items: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        *self = Self::default();
        let mut line = Vec::new();
        let mut bytes = Vec::new();
        let mut offset = 0;
        let mut start = 0;
        let mut stats = ScanStats::default();
        loop {
            line.clear();
            let n = reader
                .read_until(b'\n', &mut line)
                .map_err(|error| ScanError::Input(serde_json::Error::io(error)))?;
            if n == 0 {
                break;
            }
            if line.starts_with(b"TO LLM ") && !bytes.is_empty() {
                items
                    .push(start..offset, Vec::new(), None, &bytes)
                    .map_err(ScanError::Callback)?;
                stats.elements += 1;
                stats.peak_element_bytes = stats.peak_element_bytes.max(bytes.len());
                bytes.clear();
                start = offset;
            }
            if bytes.is_empty() && !line.starts_with(b"TO LLM ") {
                return Err(ScanError::Callback("not an Aider LLM history".into()));
            }
            bytes.extend_from_slice(&line);
            offset += n as u64;
        }
        if !bytes.is_empty() {
            stats.elements += 1;
            stats.peak_element_bytes = stats.peak_element_bytes.max(bytes.len());
            items
                .push(start..offset, Vec::new(), None, &bytes)
                .map_err(ScanError::Callback)?;
        }
        Ok(stats)
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
        let (header, body) = text
            .split_once('\n')
            .ok_or("Aider invocation has no request body")?;
        if !header.starts_with("TO LLM ") {
            return Err("not an Aider invocation".into());
        }
        let (request, response) = body
            .split_once("\nLLM RESPONSE ")
            .map_or((body, None), |(request, response)| {
                (request, Some(response))
            });
        let native = self
            .native
            .get_or_insert_with(|| {
                format!(
                    "history-{}",
                    polylogue_model::BlobHash::of(format!("{header}\n{request}").as_bytes())
                )
            })
            .clone();
        let sid = session_id(ORIGIN, &native).ok_or("unusable Aider history identity")?;
        if self.next == 0 {
            sink.out
                .sessions
                .push(Session::new(sid.clone(), SessionKind::Main));
        }
        let mut messages = parse_messages(request)?;
        let copied = messages
            .iter()
            .zip(&self.context)
            .take_while(|(a, b)| a == b)
            .count();
        for (role, text) in &messages[copied..] {
            self.emit(&sid, *role, text.clone(), header, sink)?;
        }
        if let Some(response) = response {
            let (at, body) = response.split_once('\n').unwrap_or((response, ""));
            let replies = parse_messages(body)?;
            for (role, text) in replies {
                self.emit(&sid, role, text.clone(), at, sink)?;
                messages.push((role, text));
            }
        }
        self.context = messages;
        sink.out.events.push(Event {session:Some(sid),kind:"aider_request".into(),at:None,after:self.last.clone(),range:None,data:serde_json::json!({"logged_at":header.strip_prefix("TO LLM "),"copied_prefix_messages":copied,"companion_name":COMPANION})});
        sink.records.record("invocation", &RecordOutcome::Used);
        Ok(())
    }
}
impl History {
    fn emit(
        &mut self,
        sid: &polylogue_model::SessionId,
        role: LogRole,
        text: String,
        logged_at: &str,
        sink: &mut Sink<'_>,
    ) -> Result<(), String> {
        let id = message_id(ORIGIN, &format!("{}#{}", sid.native(), self.next))
            .ok_or("unusable Aider message identity")?;
        self.next += 1;
        let (role, prompt_kind, blocks) = match role {
            LogRole::Tool => (
                Role::User,
                Some(polylogue_model::PromptKind::Harness),
                vec![Block::ToolResult {
                    reported_duration_ms: None,
                    call_id: None,
                    outcome: polylogue_model::ToolOutcome::Unknown {
                        reason: polylogue_model::UnknownReason::StatusFieldUnreadable,
                    },
                    output: Some(text),
                    data: None,
                    attachments: Vec::new(),
                }],
            ),
            role => (
                match role {
                    LogRole::System => Role::System,
                    LogRole::User => Role::User,
                    _ => Role::Assistant,
                },
                None,
                vec![Block::Text { text }],
            ),
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message: Message {
                id: id.clone(),
                session: sid.clone(),
                parent: self.last.clone(),
                role,
                at: None,
                model: None,
                prompt_kind,
                blocks,
                usage: Usage::default(),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                production: Default::default(),
                attrs: serde_json::Map::from_iter([(
                    "aider_logged_at".into(),
                    Json::String(logged_at.into()),
                )]),
            },
        });
        self.last = Some(id);
        Ok(())
    }
}

fn parse_messages(text: &str) -> Result<Vec<(LogRole, String)>, String> {
    let mut messages = Vec::new();
    let mut current: Option<(LogRole, Vec<&str>)> = None;
    for line in text.lines() {
        if line == "-------" {
            if let Some((role, lines)) = current.take() {
                messages.push((role, lines.join("\n")));
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let (name, body) = line
            .split_once(' ')
            .ok_or("Aider content has no role prefix")?;
        let role = match name {
            "SYSTEM" => LogRole::System,
            "USER" => LogRole::User,
            "ASSISTANT" => LogRole::Assistant,
            "TOOL" | "FUNCTION" => LogRole::Tool,
            _ => return Err(format!("unrecognized Aider role {name}")),
        };
        if current
            .as_ref()
            .is_some_and(|(previous, _)| *previous != role)
            && let Some((role, lines)) = current.take()
        {
            messages.push((role, lines.join("\n")));
        }
        current.get_or_insert((role, Vec::new())).1.push(body);
    }
    if let Some((role, lines)) = current {
        messages.push((role, lines.join("\n")));
    }
    Ok(messages)
}
struct Companion;
impl DocumentParser for Companion {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        let Ok(text) = std::str::from_utf8(bytes) else {
            records.record(
                "companion",
                &RecordOutcome::unreadable("Aider Markdown is not UTF-8"),
            );
            return;
        };
        let id = DocumentId::new(
            Origin::new(ORIGIN).expect("declared origin"),
            NativeId::new(&format!(
                "chat-history-{}",
                polylogue_model::BlobHash::of(bytes)
            ))
            .expect("content identity"),
        );
        let mut document = Document::new(id, DocumentPurpose::Transcript);
        document.title = Some(COMPANION.into());
        document.blocks.push(Block::Text { text: text.into() });
        document.metadata = serde_json::json!({"companion":HISTORY,"package_role":"chat-history"});
        out.documents.push(DocumentRow {
            offset: 0,
            end: bytes.len() as u64,
            document,
        });
        records.record("companion", &RecordOutcome::Used);
    }
}

#[cfg(test)]
mod tests;
