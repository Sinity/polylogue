//! Gemini CLI sessions: `~/.gemini/tmp/<project>/chats/session-*.json`,
//! one JSON document per session that the CLI rewrites as it grows.
//!
//! Messages are `user` and `gemini` turns; `info` and `error` entries are
//! the CLI's own notices (a cancelled request, an API error) and become
//! events. A tool call carries its result in the same turn, so call and
//! result sit together in the assistant message.

use crate::array::{ArrayParser, ItemSink, Owners, ScanError, ScanStats};
use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Output, Parse};
use crate::{Disposition, Records};
use polylogue_model::{
    Block, ErrorReason, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource,
    ToolAction, ToolOutcome, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "gemini-cli";
pub const LOCATIONS: &[&str] = &[".gemini/tmp/*/chats/*.json"];

fn in_chats_dir(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "json")
        && path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            == Some("chats")
}

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    in_chats_dir(path) && first_byte(head) == Some(b'{') && first_key(head) == Some("sessionId")
}

/// `gemini --resume <id>`.
pub fn resume(native: &str) -> Option<Vec<String>> {
    Some(vec!["gemini".into(), "--resume".into(), native.into()])
}

pub fn open(path: &Path) -> Option<Parse> {
    in_chats_dir(path).then(|| Parse::Array(Box::new(GeminiCli::default())))
}

#[derive(Default)]
pub struct GeminiCli {
    state: SessionState,
}

#[derive(Default)]
struct SessionState {
    metadata: Option<Json>,
    session: Option<polylogue_model::SessionId>,
    previous: Option<polylogue_model::MessageId>,
}

const MESSAGE_FIELDS: Fields = &[
    ("id", Rule::Read),
    ("timestamp", Rule::Read),
    ("type", Rule::Read),
    ("content", Rule::Read),
    (
        "displayContent",
        Rule::Excluded("the prompt as shown, with file references collapsed"),
    ),
    ("thoughts", Rule::Read),
    ("model", Rule::Read),
    ("tokens", Rule::Read),
    ("toolCalls", Rule::Read),
    ("toolCalls[].id", Rule::Read),
    ("toolCalls[].name", Rule::Read),
    ("toolCalls[].args", Rule::Read),
    ("toolCalls[].result", Rule::Read),
    ("toolCalls[].status", Rule::Read),
    (
        "toolCalls[].timestamp",
        Rule::Excluded("the turn's time places the call"),
    ),
    (
        "toolCalls[].resultDisplay",
        Rule::Excluded("the result as rendered; the result is read"),
    ),
    (
        "toolCalls[].displayName",
        Rule::Excluded("label for the tool name"),
    ),
    (
        "toolCalls[].description",
        Rule::Excluded("the tool's own description"),
    ),
    (
        "toolCalls[].renderOutputAsMarkdown",
        Rule::Excluded("display setting"),
    ),
];

impl ArrayParser for GeminiCli {
    fn scan(
        &mut self,
        reader: &mut dyn std::io::Read,
        item: &mut ItemSink<'_>,
    ) -> Result<ScanStats, ScanError> {
        self.state = SessionState::default();
        let (metadata, stats) =
            crate::array::scan_nested(reader, &["messages"], |range, raw| item(range, None, raw))?;
        self.state.metadata = Some(metadata);
        Ok(stats)
    }
    fn begin(&mut self, length: u64) -> Result<Option<crate::parse::Step>, String> {
        let doc = self
            .state
            .metadata
            .take()
            .ok_or("session metadata not scanned")?;
        let mut output = Output::default();
        let mut records = Records::default();
        self.start_session(&doc, &mut Sink::new(&mut output, &mut records));
        for event in &mut output.events {
            event.range = Some((0, length));
        }
        Ok(Some(crate::parse::Step {
            output,
            accounting: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            snapshot: None,
        }))
    }
    fn item(&mut self, m: &Json, sink: &mut Sink<'_>, _: &mut dyn Owners) -> Result<(), String> {
        let Some(sid) = self.state.session.clone() else {
            return Ok(());
        };
        let kind = str_of(m, "type").unwrap_or("?").to_owned();
        sink.records.fields(&kind, m, MESSAGE_FIELDS);
        let role = match kind.as_str() {
            "user" => Role::User,
            "gemini" => Role::Assistant,
            "info" | "error" => {
                sink.out.events.push(Event {
                    session: sid.clone(),
                    kind: if kind == "error" {
                        "api_error"
                    } else {
                        "notice"
                    }
                    .into(),
                    at: m.get("timestamp").and_then(time_of),
                    after: self.state.previous.clone(),
                    range: None,
                    data: m.get("content").cloned().unwrap_or(Json::Null),
                });
                sink.records.record(&kind, &Disposition::Materialized);
                return Ok(());
            }
            other => {
                sink.records
                    .record(other, &Disposition::unknown(format!("type {other}")));
                return Ok(());
            }
        };
        let Some(id) = str_of(m, "id").and_then(|i| message_id(ORIGIN, i)) else {
            sink.records.record(&kind, &Disposition::refused("no id"));
            return Ok(());
        };
        let message = Message {
            id: id.clone(),
            session: sid.clone(),
            parent: self.state.previous.replace(id),
            role,
            at: m.get("timestamp").and_then(time_of),
            model: string_of(m, "model"),
            prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
            blocks: blocks(m),
            usage: usage(m.get("tokens")),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
            production: Default::default(),
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message,
        });
        sink.records.record(&kind, &Disposition::Materialized);
        Ok(())
    }
}

impl GeminiCli {
    fn start_session(&mut self, doc: &Json, sink: &mut Sink<'_>) {
        if !doc.is_object() {
            return sink
                .records
                .record("session", &Disposition::refused("not an object"));
        }
        let Some(sid) = str_of(doc, "sessionId").and_then(|s| session_id(ORIGIN, s)) else {
            return sink
                .records
                .record("session", &Disposition::refused("no sessionId"));
        };
        let kind = match str_of(doc, "kind") {
            Some("subagent") => SessionKind::Subagent,
            _ => SessionKind::Main,
        };
        let mut session = Session::new(sid.clone(), kind);
        if let Some(summary) = str_of(doc, "summary") {
            session.offer_title(summary, TitleSource::Summary);
        }
        if let Some(at) = doc.get("startTime").and_then(time_of) {
            session.offer_start(at);
        }
        for dir in doc
            .get("directories")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
            .filter_map(Json::as_str)
        {
            session.offer_cwd(dir);
        }
        sink.out.sessions.push(session);
        sink.records.record("session", &Disposition::Materialized);
        if let Some(hash) = str_of(doc, "projectHash") {
            sink.out.events.push(Event {
                session: sid.clone(),
                kind: "project".into(),
                at: None,
                after: None,
                range: None,
                data: serde_json::json!({"project_hash": hash}),
            });
        }

        self.state.session = Some(sid);
    }
}

/// Text of `content`: a string, or parts with `text`.
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

/// What a Gemini CLI tool does, by its name. MCP tools are named
/// `mcp_<server>_<tool>`.
fn action(tool: &str) -> ToolAction {
    match tool {
        "run_shell_command" => ToolAction::Shell,
        "read_file" | "read_many_files" => ToolAction::FileRead,
        "write_file" => ToolAction::FileWrite,
        "replace" | "edit" => ToolAction::FileEdit,
        "glob" | "grep_search" | "search_file_content" | "list_directory" => ToolAction::Search,
        "google_web_search" | "web_fetch" => ToolAction::Web,
        "codebase_investigator" => ToolAction::Subagent,
        t if t.starts_with("mcp_") => ToolAction::Mcp,
        _ => ToolAction::Other,
    }
}

fn blocks(m: &Json) -> Vec<Block> {
    let mut out = Vec::new();
    for t in m
        .get("thoughts")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let text = match (str_of(t, "subject"), str_of(t, "description")) {
            (Some(s), Some(d)) => Some(format!("{s}\n\n{d}")),
            (s, d) => s.or(d).map(str::to_owned),
        };
        out.push(Block::Thinking {
            text,
            signature: None,
            redacted: false,
        });
    }
    if let Some(text) = text_of(m.get("content")) {
        out.push(Block::Text { text });
    }
    for call in m
        .get("toolCalls")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let args = call.get("args").cloned().unwrap_or(Json::Null);
        out.push(Block::ToolCall {
            call_id: string_of(call, "id"),
            tool: string_of(call, "name"),
            action: action(str_of(call, "name").unwrap_or("")),
            command: args
                .get("command")
                .and_then(Json::as_str)
                .map(str::to_owned),
            path: args
                .get("file_path")
                .or_else(|| args.get("absolute_path"))
                .or_else(|| args.get("path"))
                .and_then(Json::as_str)
                .map(str::to_owned),
            input: args,
        });
        let response = call
            .get("result")
            .and_then(Json::as_array)
            .and_then(|r| r.first())
            .and_then(|r| r.pointer("/functionResponse/response"));
        let output = response
            .and_then(|r| r.get("output").or_else(|| r.get("error")))
            .and_then(Json::as_str)
            .map(str::to_owned);
        out.push(Block::ToolResult {
            call_id: string_of(call, "id"),
            outcome: match str_of(call, "status") {
                Some("success") => ToolOutcome::Ok,
                Some("cancelled") => ToolOutcome::Error {
                    reason: Some(ErrorReason::Interrupted),
                },
                Some(_) => ToolOutcome::Error { reason: None },
                None => ToolOutcome::NoResult,
            },
            output,
            data: None,
            attachments: Vec::new(),
        });
    }
    out
}

/// Gemini reports input including the cached part; the counts split it.
fn usage(tokens: Option<&Json>) -> Usage {
    let Some(t) = tokens else {
        return Usage::default();
    };
    let n = |k: &str| t.get(k).and_then(Json::as_u64);
    let cached = n("cached");
    Usage {
        input: n("input").map(|i| i.saturating_sub(cached.unwrap_or(0))),
        cached_input: cached,
        cache_write: None,
        output: n("output"),
        reasoning: n("thoughts"),
        ..Usage::default()
    }
}

#[cfg(test)]
mod tests;
