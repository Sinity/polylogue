//! Grok CLI sessions: `~/.grok/sessions/<cwd>/<session>/chat_history.jsonl`,
//! where `<cwd>` is the working directory percent-encoded (`%2Frealm%2F…`).
//!
//! The history is the model's message list, appended a line at a time:
//! `system`, `user`, `reasoning`, `assistant` (with tool calls) and
//! `tool_result`. Lines hold no IDs and no times, so a message is named by
//! its byte offset in the file, a stable place in an append-only file;
//! unknown times stay unknown. A `reasoning` line belongs to the assistant
//! message after it.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::document::{first_key, message_id, session_id, str_of, string_of};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::lines::Record;
use crate::parse::{MessageRow, Output, Parse, Parser};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Block, Event, Json, Message, MessageId, PromptKind, Role, Session, SessionKind, ToolAction,
    ToolOutcome,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ORIGIN: &str = "grok-cli";
pub const LOCATIONS: &[&str] = &[".grok/sessions/*/*/chat_history.jsonl"];

/// The session and working directory a history file belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    pub session: String,
    pub cwd: Option<String>,
}

impl Place {
    pub fn from_path(path: &Path) -> Option<Self> {
        if path.file_name()? != "chat_history.jsonl" {
            return None;
        }
        let dir = path.parent()?;
        let session = dir.file_name()?.to_str()?.to_owned();
        let cwd = dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .map(percent_decode);
        Some(Self { session, cwd })
    }
}

/// `%2Frealm%2Fproject` → `/realm/project`.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let Some(hex) = bytes.get(i + 1..i + 3)
            && hex.iter().all(u8::is_ascii_hexdigit)
            && let Ok(b) = u8::from_str_radix(std::str::from_utf8(hex).unwrap_or(""), 16)
        {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    Place::from_path(path).is_some() && first_key(head) == Some("type")
}

pub fn open(path: &Path) -> Option<Parse> {
    Place::from_path(path).map(|place| Parse::Lines(Box::new(GrokCli { place })))
}

pub struct GrokCli {
    pub place: Place,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {
    /// The previous message, the next one's parent.
    last: Option<MessageId>,
    /// Reasoning waiting for the assistant message it belongs to.
    thinking: Vec<Block>,
    /// The session's facts were sent.
    announced: bool,
}

static FIELDS: Fields = Fields::new(&[
    ("tool_calls[].id", Rule::Read(T::String)),
    ("tool_calls[].name", Rule::Read(T::String)),
    (
        "tool_calls[].arguments",
        Rule::Kept("complete parsed tool input"),
    ),
    ("content[].type", Rule::Read(T::String)),
    ("content[].text", Rule::Read(T::String)),
    ("type", Rule::Read(T::String)),
    ("content", Rule::Read(T::StringOrArray)),
    ("id", Rule::Read(T::String)),
    ("summary", Rule::Read(T::String)),
    (
        "encrypted_content",
        Rule::Kept("reasoning only the provider can read"),
    ),
    ("status", Rule::Excluded("completion of the reasoning step")),
    ("model_id", Rule::Read(T::String)),
    (
        "model_fingerprint",
        Rule::Kept("the model build's fingerprint"),
    ),
    ("reasoning_effort", Rule::Read(T::String)),
    ("tool_calls", Rule::Read(T::Array)),
    ("tool_call_id", Rule::Read(T::String)),
    ("synthetic_reason", Rule::Read(T::String)),
    (
        "prompt_index",
        Rule::Excluded("the CLI's count of typed prompts"),
    ),
    ("kind", Rule::Read(T::String)),
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &["user", "assistant", "tool_result", "system", "reasoning"],
    }],
);

/// What a Grok CLI tool does, by its name.
fn action(tool: &str) -> ToolAction {
    match tool {
        "run_terminal_command" | "monitor" => ToolAction::Shell,
        "read_file" => ToolAction::FileRead,
        "write" => ToolAction::FileWrite,
        "search_replace" => ToolAction::FileEdit,
        "grep" | "list_dir" | "search_tool" => ToolAction::Search,
        "web_fetch" | "web_search" => ToolAction::Web,
        _ => ToolAction::Other,
    }
}

fn text_of(content: Option<&Json>) -> Option<String> {
    match content? {
        Json::String(s) if !s.is_empty() => Some(s.clone()),
        Json::Array(parts) => {
            let texts: Vec<&str> = parts.iter().filter_map(|p| str_of(p, "text")).collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
}

impl Parser for GrokCli {
    type ParserState = ParserState;

    fn kind(&self, value: &Json) -> String {
        str_of(value, "type").unwrap_or("?").to_owned()
    }

    fn parse_record(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let kind = self.kind(value);
        records.fields(&kind, value, &FIELDS);
        let Some(sid) = session_id(ORIGIN, &self.place.session) else {
            return RecordOutcome::unreadable("unusable session directory name");
        };
        if !parser_state.announced {
            let mut session = Session::new(sid.clone(), SessionKind::Main);
            if let Some(cwd) = &self.place.cwd {
                session.offer_cwd(cwd);
            }
            out.sessions.push(session);
            parser_state.announced = true;
        }
        let end = record.offset + record.bytes.len() as u64;
        if kind == "reasoning" {
            let summaries: Vec<&str> = value
                .get("summary")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
                .filter_map(|s| str_of(s, "text"))
                .collect();
            let text = (!summaries.is_empty()).then(|| summaries.join("\n"));
            parser_state.thinking.push(Block::Thinking {
                redacted: text.is_none() && value.get("encrypted_content").is_some(),
                text,
                signature: None,
            });
            return RecordOutcome::Used;
        }
        if kind == "backend_tool_call" {
            // A tool the provider ran itself (web search).
            out.events.push(Event {
                session: Some(sid),
                kind: "backend_tool_call".into(),
                at: None,
                after: parser_state.last.clone(),
                range: Some((record.offset, end)),
                data: value.get("kind").cloned().unwrap_or(Json::Null),
            });
            return RecordOutcome::Used;
        }
        let (role, prompt_kind, blocks) = match kind.as_str() {
            "system" => (
                Role::System,
                None,
                text_of(value.get("content"))
                    .map(|text| vec![Block::Text { text }])
                    .unwrap_or_default(),
            ),
            "user" => {
                let synthetic = value.get("synthetic_reason").is_some_and(|r| !r.is_null());
                (
                    Role::User,
                    Some(if synthetic {
                        PromptKind::Harness
                    } else {
                        PromptKind::Prose
                    }),
                    text_of(value.get("content"))
                        .map(|text| vec![Block::Text { text }])
                        .unwrap_or_default(),
                )
            }
            "assistant" => {
                let mut blocks = std::mem::take(&mut parser_state.thinking);
                if let Some(text) = text_of(value.get("content")) {
                    blocks.push(Block::Text { text });
                }
                for call in value
                    .get("tool_calls")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                {
                    let raw = call.get("arguments");
                    // Arguments are a JSON string; kept as written when not JSON.
                    let input = match raw.and_then(Json::as_str) {
                        Some(s) => serde_json::from_str(s).unwrap_or(Json::String(s.to_owned())),
                        None => raw.cloned().unwrap_or(Json::Null),
                    };
                    blocks.push(Block::ToolCall {
                        run_by_script: false,
                        call_id: string_of(call, "id"),
                        tool: string_of(call, "name"),
                        action: action(str_of(call, "name").unwrap_or("")),
                        command: input
                            .get("command")
                            .and_then(Json::as_str)
                            .map(str::to_owned),
                        path: input
                            .get("target_file")
                            .or_else(|| input.get("path"))
                            .or_else(|| input.get("file_path"))
                            .or_else(|| input.get("target_directory"))
                            .and_then(Json::as_str)
                            .map(str::to_owned),
                        input,
                    });
                }
                (Role::Assistant, None, blocks)
            }
            "tool_result" => (
                Role::User,
                Some(PromptKind::Harness),
                vec![Block::ToolResult {
                    reported_duration_ms: None,
                    call_id: string_of(value, "tool_call_id"),
                    outcome: ToolOutcome::Ok,
                    output: text_of(value.get("content")),
                    data: None,
                    attachments: Vec::new(),
                }],
            ),
            other => return RecordOutcome::skipped(format!("record type {other}")),
        };
        let Some(id) = message_id(ORIGIN, &format!("{}@{}", self.place.session, record.offset))
        else {
            return RecordOutcome::unreadable("unusable message ID");
        };
        let production = polylogue_model::Production {
            effort: string_of(value, "reasoning_effort"),
            ..Default::default()
        };
        out.messages.push(MessageRow {
            offset: record.offset,
            end,
            message: Message {
                id: id.clone(),
                session: sid,
                parent: parser_state.last.replace(id),
                role,
                at: None,
                model: string_of(value, "model_id"),
                prompt_kind,
                blocks,
                usage: Default::default(),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                attrs: crate::fields::attributes(value, &FIELDS),
                production,
            },
        });
        RecordOutcome::Used
    }

    fn flush(&self, _carry: &mut ParserState, _out: &mut Output) {}
}

#[cfg(test)]
mod tests;
