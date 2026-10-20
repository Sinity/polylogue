//! OpenTelemetry GenAI spans, as OTLP JSON trace exports
//! (`{"resourceSpans": …}`, one document or one per line as the file
//! exporter writes them), following the GenAI semantic conventions.
//!
//! A conversation is `gen_ai.conversation.id`, else the trace. Each model
//! call's `gen_ai.input.messages` repeats the history so far, so a message
//! keeps its identity by its place in the conversation (architecture 5.5)
//! and is written the first time that place appears; the call's
//! `gen_ai.output.messages` follow it, with the call's token usage. Spans
//! that are not GenAI spans are counted and set aside.

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id};
use crate::parse::{DocumentParser, MessageRow, Output, Parse};
use crate::{Accounting, Disposition};
use polylogue_model::{
    Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, Timestamp, ToolAction,
    ToolOutcome, Usage,
};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub const ORIGIN: &str = "otel-genai";

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    let ext = path.extension().and_then(|e| e.to_str());
    matches!(ext, Some("json" | "jsonl"))
        && first_byte(head) == Some(b'{')
        && first_key(head) == Some("resourceSpans")
}

pub fn open(_path: &Path) -> Option<Parse> {
    Some(Parse::Document(Box::new(OtelGenAi)))
}

pub struct OtelGenAi;

/// One span with its attributes decoded from OTLP `AnyValue`s.
struct Span {
    trace: String,
    span: String,
    start: Option<Timestamp>,
    end_ns: Option<u64>,
    start_ns: Option<u64>,
    attrs: BTreeMap<String, Json>,
    status_error: bool,
    raw: Json,
}

impl DocumentParser for OtelGenAi {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let documents: Vec<Json> = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) => vec![v],
            Err(_) => {
                let mut docs = Vec::new();
                for line in bytes.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
                    match serde_json::from_slice::<Json>(line) {
                        Ok(v) => docs.push(v),
                        Err(e) => sink.record(
                            "line",
                            &Disposition::refused(format!("malformed JSON: {e}")),
                        ),
                    }
                }
                docs
            }
        };
        let mut spans = Vec::new();
        for doc in &documents {
            collect_spans(doc, &mut spans);
        }
        spans.sort_by(|a, b| (a.start_ns, &a.span).cmp(&(b.start_ns, &b.span)));

        // Places already written, per conversation.
        let mut written: BTreeMap<String, HashSet<usize>> = BTreeMap::new();
        let mut sessions_made: HashSet<String> = HashSet::new();
        for span in spans {
            let op = span
                .attrs
                .get("gen_ai.operation.name")
                .and_then(Json::as_str)
                .map(str::to_owned);
            let is_genai = op.is_some() || span.attrs.keys().any(|k| k.starts_with("gen_ai."));
            if !is_genai {
                sink.record("span", &Disposition::unknown("not a GenAI span"));
                continue;
            }
            let conversation = span
                .attrs
                .get("gen_ai.conversation.id")
                .and_then(Json::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| span.trace.clone());
            let Some(sid) = session_id(ORIGIN, &conversation) else {
                sink.record("span", &Disposition::refused("unusable conversation ID"));
                continue;
            };
            if sessions_made.insert(conversation.clone()) {
                let mut s = Session::new(sid.clone(), SessionKind::Main);
                if let Some(at) = span.start {
                    s.offer_start(at);
                }
                sink.out.sessions.push(s);
            }
            sink.out.events.push(Event {
                session: sid.clone(),
                kind: "otel_span".into(),
                at: span.start,
                after: None,
                range: None,
                data: span.raw.clone(),
            });
            let seen = written.entry(conversation.clone()).or_default();
            match op.as_deref() {
                Some("execute_tool") => tool_span(&span, &conversation, &sid, seen, &mut sink),
                _ => model_span(&span, &conversation, &sid, seen, &mut sink),
            }
            sink.record(
                op.as_deref().unwrap_or("gen_ai"),
                &Disposition::Materialized,
            );
        }
    }
}

fn collect_spans(doc: &Json, out: &mut Vec<Span>) {
    for rs in doc
        .get("resourceSpans")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        for ss in rs
            .get("scopeSpans")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            for s in ss
                .get("spans")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
            {
                let nanos = |k: &str| {
                    s.get(k).and_then(|v| match v {
                        Json::String(t) => t.parse::<u64>().ok(),
                        other => other.as_u64(),
                    })
                };
                let start_ns = nanos("startTimeUnixNano");
                out.push(Span {
                    trace: s
                        .get("traceId")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    span: s
                        .get("spanId")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    start: start_ns.and_then(|n| Timestamp::from_unix_ms((n / 1_000_000) as i64)),
                    start_ns,
                    end_ns: nanos("endTimeUnixNano"),
                    attrs: attributes(s.get("attributes")),
                    status_error: s.pointer("/status/code").is_some_and(|c| {
                        c.as_i64() == Some(2) || c.as_str() == Some("STATUS_CODE_ERROR")
                    }),
                    raw: s.clone(),
                });
            }
        }
    }
}

/// OTLP attributes (`[{"key", "value": AnyValue}]`) as plain JSON.
fn attributes(value: Option<&Json>) -> BTreeMap<String, Json> {
    let mut map = BTreeMap::new();
    for kv in value.and_then(Json::as_array).into_iter().flatten() {
        if let (Some(k), Some(v)) = (kv.get("key").and_then(Json::as_str), kv.get("value")) {
            map.insert(k.to_owned(), any_value(v));
        }
    }
    map
}

fn any_value(v: &Json) -> Json {
    for key in ["stringValue", "boolValue", "doubleValue", "bytesValue"] {
        if let Some(x) = v.get(key) {
            return x.clone();
        }
    }
    if let Some(i) = v.get("intValue") {
        return match i {
            Json::String(s) => s
                .parse::<i64>()
                .map(Json::from)
                .unwrap_or_else(|_| i.clone()),
            other => other.clone(),
        };
    }
    if let Some(items) = v.pointer("/arrayValue/values").and_then(Json::as_array) {
        return Json::Array(items.iter().map(any_value).collect());
    }
    if let Some(kv) = v.pointer("/kvlistValue/values") {
        return Json::Object(attributes(Some(kv)).into_iter().collect());
    }
    v.clone()
}

/// A messages attribute: a JSON string (the conventions' recommended
/// form) or an already structured array.
fn messages_attr(span: &Span, key: &str) -> Vec<Json> {
    match span.attrs.get(key) {
        Some(Json::String(s)) => serde_json::from_str::<Json>(s)
            .ok()
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default(),
        Some(Json::Array(items)) => items.clone(),
        _ => Vec::new(),
    }
}

fn model_span(
    span: &Span,
    conversation: &str,
    sid: &polylogue_model::SessionId,
    seen: &mut HashSet<usize>,
    sink: &mut Sink<'_>,
) {
    let inputs = messages_attr(span, "gen_ai.input.messages");
    let outputs = messages_attr(span, "gen_ai.output.messages");
    let model = span
        .attrs
        .get("gen_ai.response.model")
        .or_else(|| span.attrs.get("gen_ai.request.model"))
        .and_then(Json::as_str)
        .map(str::to_owned);
    let tokens = |k: &str| span.attrs.get(k).and_then(Json::as_u64);
    let cached = tokens("gen_ai.usage.cache_read.input_tokens");
    let usage = Usage {
        input: tokens("gen_ai.usage.input_tokens").map(|i| i.saturating_sub(cached.unwrap_or(0))),
        cached_input: cached,
        output: tokens("gen_ai.usage.output_tokens"),
        ..Usage::default()
    };
    let n = inputs.len();
    for (place, m) in inputs.iter().chain(outputs.iter()).enumerate() {
        if !seen.insert(place) {
            continue;
        }
        let is_output = place >= n;
        let Some(id) = message_id(ORIGIN, &format!("{conversation}#{place}")) else {
            continue;
        };
        let role = match m.get("role").and_then(Json::as_str) {
            Some("user") => Role::User,
            Some("system") => Role::System,
            Some("tool") => Role::User,
            _ => Role::Assistant,
        };
        let blocks = parts(m);
        if blocks.is_empty() {
            continue;
        }
        let parent = place
            .checked_sub(1)
            .and_then(|p| message_id(ORIGIN, &format!("{conversation}#{p}")));
        let message = Message {
            id,
            session: sid.clone(),
            parent,
            role,
            at: if is_output { span.start } else { None },
            model: is_output.then(|| model.clone()).flatten(),
            prompt_kind: (role == Role::User).then_some(
                if m.get("role").and_then(Json::as_str) == Some("tool") {
                    PromptKind::Harness
                } else {
                    PromptKind::Prose
                },
            ),
            blocks,
            usage: if is_output && place == n {
                usage.clone()
            } else {
                Usage::default()
            },
            stop_reason: m
                .get("finish_reason")
                .and_then(Json::as_str)
                .map(str::to_owned),
            reported_cost_usd: None,
            duration_ms: match (is_output, span.start_ns, span.end_ns) {
                (true, Some(a), Some(b)) if b >= a => Some((b - a) / 1_000_000),
                _ => None,
            },
            production: Default::default(),
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message,
        });
    }
}

/// An `execute_tool` span: its call and result, when no model message
/// carried them, as an event on the conversation.
fn tool_span(
    span: &Span,
    _conversation: &str,
    sid: &polylogue_model::SessionId,
    _seen: &mut HashSet<usize>,
    sink: &mut Sink<'_>,
) {
    sink.out.events.push(Event {
        session: sid.clone(),
        kind: "tool_execution".into(),
        at: span.start,
        after: None,
        range: None,
        data: serde_json::json!({
            "call_id": span.attrs.get("gen_ai.tool.call.id"),
            "tool": span.attrs.get("gen_ai.tool.name"),
            "arguments": span.attrs.get("gen_ai.tool.call.arguments"),
            "result": span.attrs.get("gen_ai.tool.call.result"),
            "error": span.status_error,
        }),
    });
}

/// The conventions' message parts: text, reasoning, tool calls and tool
/// call responses; others kept whole.
fn parts(m: &Json) -> Vec<Block> {
    let mut blocks = Vec::new();
    for p in m
        .get("parts")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let text = || p.get("content").and_then(Json::as_str).map(str::to_owned);
        match p.get("type").and_then(Json::as_str) {
            Some("text") => {
                if let Some(t) = text() {
                    blocks.push(Block::Text { text: t });
                }
            }
            Some("reasoning") => blocks.push(Block::Thinking {
                text: text(),
                signature: None,
                redacted: false,
            }),
            Some("tool_call") => blocks.push(Block::ToolCall {
                call_id: p.get("id").and_then(Json::as_str).map(str::to_owned),
                tool: p.get("name").and_then(Json::as_str).map(str::to_owned),
                action: ToolAction::Other,
                input: p.get("arguments").cloned().unwrap_or(Json::Null),
                command: None,
                path: None,
            }),
            Some("tool_call_response") => blocks.push(Block::ToolResult {
                call_id: p.get("id").and_then(Json::as_str).map(str::to_owned),
                outcome: ToolOutcome::Ok,
                output: match p.get("response") {
                    Some(Json::String(s)) => Some(s.clone()),
                    Some(other) => Some(other.to_string()),
                    None => None,
                },
                data: None,
                attachments: Vec::new(),
            }),
            Some(other) => blocks.push(Block::Other {
                type_name: other.to_owned(),
                data: p.clone(),
            }),
            None => {}
        }
    }
    blocks
}

#[cfg(test)]
mod tests;
