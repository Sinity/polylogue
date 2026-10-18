//! Gemini CLI sessions: `~/.gemini/tmp/<project>/chats/session-*.json`,
//! one JSON document per session that the CLI rewrites as it grows.
//!
//! Messages are `user` and `gemini` turns; `info` and `error` entries are
//! the CLI's own notices (a cancelled request, an API error) and become
//! events. A tool call carries its result in the same turn, so call and
//! result sit together in the assistant message.

#[cfg(test)]
mod tests;

use crate::account::{Accounting, Disposition};
use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::fields::{Fields, Rule, unaccounted};
use crate::parse::{DocumentParser, MessageRow, Output, StepParser, Whole};
use polylogue_model::{
    Block, ErrorReason, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource,
    ToolOutcome, Usage,
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

pub fn open(path: &Path) -> Option<Box<dyn StepParser>> {
    in_chats_dir(path).then(|| Box::new(Whole(GeminiCli)) as Box<dyn StepParser>)
}

pub struct GeminiCli;

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

impl DocumentParser for GeminiCli {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let doc = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) if v.is_object() => v,
            Ok(_) => return sink.record("session", &Disposition::refused("not an object")),
            Err(e) => {
                return sink.record(
                    "session",
                    &Disposition::refused(format!("malformed JSON: {e}")),
                );
            }
        };
        let Some(sid) = str_of(&doc, "sessionId").and_then(|s| session_id(ORIGIN, s)) else {
            return sink.record("session", &Disposition::refused("no sessionId"));
        };
        let kind = match str_of(&doc, "kind") {
            Some("subagent") => SessionKind::Subagent,
            _ => SessionKind::Main,
        };
        let mut session = Session::new(sid.clone(), kind);
        if let Some(summary) = str_of(&doc, "summary") {
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
        sink.record("session", &Disposition::Materialized);
        if let Some(hash) = str_of(&doc, "projectHash") {
            sink.out.events.push(Event {
                session: sid.clone(),
                kind: "project".into(),
                at: None,
                after: None,
                range: None,
                data: serde_json::json!({"project_hash": hash}),
            });
        }

        let mut previous = None;
        for m in doc
            .get("messages")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            let kind = str_of(m, "type").unwrap_or("?").to_owned();
            for path in unaccounted(m, MESSAGE_FIELDS) {
                sink.acc.unread(&kind, &path);
            }
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
                        after: previous.clone(),
                        range: None,
                        data: m.get("content").cloned().unwrap_or(Json::Null),
                    });
                    sink.record(&kind, &Disposition::Materialized);
                    continue;
                }
                other => {
                    sink.record(other, &Disposition::unknown(format!("type {other}")));
                    continue;
                }
            };
            let Some(id) = str_of(m, "id").and_then(|i| message_id(ORIGIN, i)) else {
                sink.record(&kind, &Disposition::refused("no id"));
                continue;
            };
            let message = Message {
                id: id.clone(),
                session: sid.clone(),
                parent: previous.replace(id),
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
            sink.record(&kind, &Disposition::Materialized);
        }
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

/// Gemini reports input including the cached part; the lanes split it.
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
        service_tier: None,
    }
}
