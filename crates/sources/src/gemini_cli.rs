//! Gemini CLI sessions: `~/.gemini/tmp/<project>/chats/session-*.json`,
//! one JSON document per session that the CLI rewrites as it grows.
//!
//! Messages are `user` and `gemini` turns; `info` and `error` entries are
//! the CLI's own notices (a cancelled request, an API error) and become
//! events. A tool call carries its result in the same turn, so call and
//! result sit together in the assistant message.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource, ToolAction,
    Usage,
};
use std::path::Path;

mod tool_output;

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
    in_chats_dir(path).then(|| Parse::Export(Box::new(GeminiCli::default())))
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

static MESSAGE_FIELDS: Fields = Fields::new(&[
    ("content[].text", Rule::Read(T::String)),
    ("thoughts[].subject", Rule::Read(T::String)),
    ("thoughts[].description", Rule::Read(T::String)),
    (
        "thoughts[].timestamp",
        Rule::Excluded("native thought time not interpreted as message time"),
    ),
    ("tokens.input", Rule::Read(T::Number)),
    ("tokens.output", Rule::Read(T::Number)),
    ("tokens.cached", Rule::Read(T::Number)),
    ("tokens.thoughts", Rule::Read(T::Number)),
    (
        "tokens.tool",
        Rule::Excluded("native tool-token lane has no domain usage lane"),
    ),
    (
        "tokens.total",
        Rule::Excluded("reported aggregate; individual lanes are read"),
    ),
    ("id", Rule::Read(T::String)),
    ("timestamp", Rule::Read(T::StringOrNumber)),
    ("type", Rule::Read(T::String)),
    ("content", Rule::Read(T::StringOrArray)),
    (
        "displayContent",
        Rule::Excluded("the prompt as shown, with file references collapsed"),
    ),
    ("thoughts", Rule::Read(T::Array)),
    ("model", Rule::Read(T::String)),
    ("tokens", Rule::Read(T::Object)),
    ("toolCalls", Rule::Read(T::Array)),
    ("toolCalls[].id", Rule::Read(T::String)),
    ("toolCalls[].name", Rule::Read(T::String)),
    ("toolCalls[].args", Rule::Kept("complete call input")),
    (
        "toolCalls[].result",
        Rule::Kept("complete native result data"),
    ),
    ("toolCalls[].status", Rule::Read(T::String)),
    (
        "toolCalls[].timestamp",
        Rule::Excluded("the turn's time places the call"),
    ),
    (
        "toolCalls[].resultDisplay",
        Rule::Kept("native result display data"),
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
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &["user", "gemini", "info", "error"],
    }],
);

impl ExportParser for GeminiCli {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.state = SessionState::default();
        let (metadata, stats) = crate::export::scan_nested(reader, &["messages"], |range, raw| {
            item.push(range, Vec::new(), None, raw.get().as_bytes())
        })?;
        self.state.metadata = Some(metadata);
        Ok(stats)
    }
    fn begin(
        &mut self,
        length: u64,
        _first: Option<&[u8]>,
    ) -> Result<Option<crate::parse::Step>, String> {
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
        let m: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let m = &m;
        let Some(sid) = self.state.session.clone() else {
            return Ok(());
        };
        let kind = str_of(m, "type").unwrap_or("?").to_owned();
        sink.records.fields(&kind, m, &MESSAGE_FIELDS);
        let role = match kind.as_str() {
            "user" => Role::User,
            "gemini" => Role::Assistant,
            "info" | "error" => {
                sink.out.events.push({
                    let (at, time_evidence) =
                        crate::time::parts(m.get("timestamp").and_then(time_of));
                    Event {
                        session: Some(sid.clone()),
                        kind: if kind == "error" {
                            "api_error"
                        } else {
                            "notice"
                        }
                        .into(),
                        at,
                        at_quality: time_evidence.quality,
                        at_precision: time_evidence.precision,
                        at_until: time_evidence.until,
                        after: self.state.previous.clone(),
                        range: None,
                        data: m.get("content").cloned().unwrap_or(Json::Null),
                    }
                });
                sink.records.record(&kind, &RecordOutcome::Used);
                return Ok(());
            }
            other => {
                sink.records
                    .record(other, &RecordOutcome::skipped(format!("type {other}")));
                return Ok(());
            }
        };
        let Some(id) = str_of(m, "id").and_then(|i| message_id(ORIGIN, i)) else {
            sink.records
                .record(&kind, &RecordOutcome::unreadable("no id"));
            return Ok(());
        };
        let message = {
            let (at, time_evidence) = crate::time::parts(m.get("timestamp").and_then(time_of));
            Message {
                id: id.clone(),
                session: sid.clone(),
                parent: self.state.previous.replace(id),
                role,
                at,
                at_quality: time_evidence.quality,
                at_precision: time_evidence.precision,
                at_until: time_evidence.until,
                model: string_of(m, "model"),
                prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
                blocks: blocks(m),
                usage: usage(m.get("tokens")),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                attrs: Default::default(),
                production: Default::default(),
            }
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message,
        });
        sink.records.record(&kind, &RecordOutcome::Used);
        Ok(())
    }
}

impl GeminiCli {
    fn start_session(&mut self, doc: &Json, sink: &mut Sink<'_>) {
        if !doc.is_object() {
            return sink
                .records
                .record("session", &RecordOutcome::unreadable("not an object"));
        }
        let Some(sid) = str_of(doc, "sessionId").and_then(|s| session_id(ORIGIN, s)) else {
            return sink
                .records
                .record("session", &RecordOutcome::unreadable("no sessionId"));
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
            session
                .offer_start(at.timestamp(), at.evidence())
                .expect("decoded source time is valid");
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
        sink.records.record("session", &RecordOutcome::Used);
        if let Some(hash) = str_of(doc, "projectHash") {
            sink.out.events.push({
                let (at, time_evidence) = crate::time::parts(None);
                Event {
                    session: Some(sid.clone()),
                    kind: "project".into(),
                    at,
                    at_quality: time_evidence.quality,
                    at_precision: time_evidence.precision,
                    at_until: time_evidence.until,
                    after: None,
                    range: None,
                    data: serde_json::json!({"project_hash": hash}),
                }
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
            run_by_script: false,
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
        out.push(tool_output::result(call));
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
