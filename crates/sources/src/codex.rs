//! Codex: rollouts `<codex>/sessions/YYYY/MM/DD/rollout-<time>-<thread>.jsonl`
//! (and the same under `archived_sessions/`), the prompt history
//! `history.jsonl` and the thread-name index `session_index.jsonl`, where
//! `<codex>` is `$CODEX_HOME`, by default `~/.codex`.
//!
//! A rollout is one thread. Each line is an envelope
//! `{timestamp, type, payload}`; files from mid-2025 write the items bare
//! after a header line. Items arrive complete, so a parser never holds a
//! whole turn: each response item is one addressable contribution. Rollouts
//! do not identify complete response membership across stream retries.
//! Usage and quota are independent observations, not item attributes.
//!
//! A subagent's rollout starts with its parent's history copied in
//! (`subagent_history_start_ordinal` says where its own begins). The copy
//! is stored once, in the parent: copied items that carry an ID keep it
//! and their parent session, and the rest is accounted as copied.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

pub mod database;
mod fields;
mod index;
mod items;
mod usage;

use crate::lines::Record;
use crate::origin::Root;
use crate::parse::{BackgroundUpdate, MessageRow, Output, Parse, Parser};
use crate::time;
use crate::{RecordOutcome, Records};
use items::{Completed, str_field};
use polylogue_model::{
    Block, Event, Json, Link, LinkKind, Message, MessageId, NativeId, Origin, Production,
    PromptKind, QuotaLimit, QuotaReading, QuotaWindow, Role, Session, SessionId, SessionKind,
    TitleSource, ToolAction, ToolOutcome, Usage,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

pub const ROOT: Root = Root {
    env: Some("CODEX_HOME"),
    default: ".codex",
};

pub const LOCATIONS: &[&str] = &[
    "sessions/**/rollout-*.jsonl",
    "archived_sessions/**/rollout-*.jsonl",
    "history.jsonl",
    "session_index.jsonl",
];

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    File::from_path(path).is_some()
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

/// A token count's `rate_limits`: the plan, credits, and a `primary` and
/// `secondary` window of the limit `limit_id`. A window's reset is epoch
/// seconds. Relative resets lack the provider observation time needed to
/// place them: token records can be emitted after draining tools.
fn quota(q: &Json) -> Option<QuotaReading> {
    let limit = str_field(q, "limit_id").unwrap_or("codex");
    let windows = ["primary", "secondary"]
        .iter()
        .filter_map(|slot| {
            let w = q.get(*slot)?;
            let minutes = w.get("window_minutes").and_then(Json::as_i64)?;
            let resets_at = w
                .get("resets_at")
                .and_then(Json::as_i64)
                .and_then(|s| s.checked_mul(1000))
                .and_then(polylogue_model::Timestamp::from_unix_ms);
            let used = w.get("used_percent").and_then(|u| {
                u.as_f64()
                    .or_else(|| u.as_str().and_then(|s| s.parse().ok()))
            });
            Some(QuotaWindow {
                limit: QuotaLimit {
                    limit: limit.to_owned().into(),
                    minutes,
                    from_first_response: false,
                },
                used_percent: used,
                resets_at,
                status: None,
            })
        })
        .collect();
    let reading = QuotaReading {
        plan: str_field(q, "plan_type").map(str::to_owned),
        credits: q
            .get("credits")
            .and_then(|c| str_field(c, "balance"))
            .map(str::to_owned),
        windows,
    };
    (reading != QuotaReading::default()).then_some(reading)
}

/// `codex resume <id>`.
pub fn resume(native: &str) -> Option<Vec<String>> {
    Some(vec!["codex".into(), "resume".into(), native.into()])
}

pub fn open(path: &Path) -> Option<Parse> {
    Some(Parse::Lines(match File::from_path(path)? {
        File::Rollout { thread } => Box::new(Rollout { thread }),
        File::History => Box::new(index::History),
        File::SessionIndex => Box::new(index::SessionIndex),
    }))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum File {
    /// A thread's rollout; the thread ID from the file name, for files
    /// whose first record does not name it.
    Rollout {
        thread: Option<String>,
    },
    History,
    SessionIndex,
}

impl File {
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        match name {
            "history.jsonl" => Some(Self::History),
            "session_index.jsonl" => Some(Self::SessionIndex),
            _ => {
                let stem = name.strip_prefix("rollout-")?.strip_suffix(".jsonl")?;
                // `rollout-<time>-<uuid>`: the UUID is the last 36 characters.
                let thread = stem
                    .len()
                    .checked_sub(36)
                    .map(|at| &stem[at..])
                    .filter(|t| t.chars().filter(|&c| c == '-').count() == 4)
                    .map(str::to_owned);
                Some(Self::Rollout { thread })
            }
        }
    }
}

fn origin() -> Origin {
    Origin::new("codex").expect("constant origin")
}

pub fn session_id(native: &str) -> Option<SessionId> {
    Some(SessionId::new(origin(), NativeId::new(native).ok()?))
}

fn message_id(native: &str) -> Option<MessageId> {
    Some(MessageId::new(origin(), NativeId::new(native).ok()?))
}

fn digest(v: &Json) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.to_string().hash(&mut h);
    h.finish()
}

pub struct Rollout {
    pub thread: Option<String>,
}

/// A tool call seen in the current turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Call {
    tool: String,
    /// The handle a `write_stdin` or `wait` call continues.
    continues: Option<String>,
    answered: bool,
    /// How the harness recorded the command ending, when its
    /// `item_completed` arrived before the output.
    completed: Option<Completed>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {
    session: Option<Session>,
    /// The thread a subagent's copied history came from, and the envelope
    /// ordinal where the subagent's own records begin.
    copied_from: Option<String>,
    copied_until: Option<u64>,
    /// The last copied item, where the child branched.
    branch_point: Option<MessageId>,
    model: Option<String>,
    #[serde(default)]
    usage: usage::Observer,
    #[serde(default)]
    scope_recorded: bool,
    calls: BTreeMap<String, Call>,
    /// Commands left running, by handle (`process 123`, `cell 4`), to the
    /// call that started them.
    running: BTreeMap<String, String>,
    links: BTreeSet<String>,
    last: Option<MessageId>,
    /// Digests of the last stored turn settings and rate limits; a repeat
    /// is not stored again.
    settings: Option<u64>,
    limits: Option<String>,
    /// Reasoning effort the current turn asked for.
    effort: Option<String>,
    /// The service tier the thread runs at (`default`, `flex`, `priority`),
    /// from its settings; prices follow it.
    #[serde(default)]
    tier: Option<String>,
    /// The byte range of the record being read, for events.
    #[serde(skip)]
    current: Option<(u64, u64)>,
    #[serde(skip)]
    copied_session: Option<SessionId>,
    /// The previous record, when it was a stored user prompt: an item or a
    /// `user_message` event. Codex can write one prompt both ways in
    /// adjacent records; the second of the pair is the same prompt.
    #[serde(default)]
    prompt: Option<Prompt>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Prompt {
    text: String,
    from_event: bool,
}

/// One record, unwrapped from its envelope.
struct Rec<'a> {
    /// `session_meta`, `response_item`, `event_msg`, `turn_context`, ...
    ty: &'a str,
    payload: &'a Json,
    at: Option<polylogue_model::Timestamp>,
    ordinal: Option<u64>,
}

const BARE_ITEMS: &[&str] = &[
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "local_shell_call",
    "web_search_call",
];

fn unwrap(v: &Json) -> Rec<'_> {
    let at = v.get("timestamp").and_then(time::rfc3339);
    let ordinal = v.get("ordinal").and_then(Json::as_u64);
    if let Some(payload) = v.get("payload").filter(|p| p.is_object()) {
        return Rec {
            ty: str_field(v, "type").unwrap_or("?"),
            payload,
            // Some writers put the time on the item, not the envelope.
            at: at.or_else(|| payload.get("timestamp").and_then(time::rfc3339)),
            ordinal,
        };
    }
    // Mid-2025 rollouts: a header, then bare items and state markers.
    let ty = match str_field(v, "type") {
        Some(t) if BARE_ITEMS.contains(&t) => "response_item",
        Some(_) => "?",
        None if v.get("record_type").is_some() => "state",
        None if v.get("id").is_some() => "older_header",
        None => "?",
    };
    Rec {
        ty,
        payload: v,
        at,
        ordinal,
    }
}

impl Rollout {
    fn unknown_boundaries(session: SessionId, out: &mut Output) {
        out.events.push(Event {
            session: Some(session),
            kind: "response_boundaries_unknown".into(),
            at: None,
            after: None,
            range: None,
            data: serde_json::json!({}),
        });
    }

    fn own(&self, parser_state: &mut ParserState) -> Option<SessionId> {
        if parser_state.session.is_none() {
            let id = session_id(self.thread.as_deref()?)?;
            parser_state.session = Some(Session::new(id, SessionKind::Main));
        }
        parser_state.session.as_ref().map(|s| s.id.clone())
    }

    fn session_meta(
        &self,
        parser_state: &mut ParserState,
        p: &Json,
        out: &mut Output,
    ) -> RecordOutcome {
        let Some(thread) = str_field(p, "id").or(self.thread.as_deref()) else {
            return RecordOutcome::unreadable("session_meta without a thread ID");
        };
        let Some(id) = session_id(thread) else {
            return RecordOutcome::unreadable("thread ID is empty");
        };
        if parser_state.session.as_ref().is_some_and(|s| s.id != id) {
            // A resumed rollout names the thread it came from; this file's
            // thread was already set by an earlier header.
            return self.event(parser_state, p, "session_meta", None, out);
        }
        let subagent = str_field(p, "thread_source") == Some("subagent")
            || p.get("source").and_then(|s| s.get("subagent")).is_some();
        let kind = if subagent {
            SessionKind::Subagent
        } else {
            SessionKind::Main
        };
        let s = parser_state
            .session
            .get_or_insert_with(|| Session::new(id.clone(), kind));
        if let Some(at) = p.get("timestamp").and_then(time::rfc3339) {
            s.offer_start(at);
        }
        if let Some(cwd) = str_field(p, "cwd") {
            s.offer_cwd(cwd);
        }
        if let Some(v) = str_field(p, "cli_version") {
            s.tool_version = Some(v.to_owned());
        }
        if let Some(branch) = p.get("git").and_then(|g| str_field(g, "branch")) {
            s.git_branch = Some(branch.to_owned());
        }
        if let Some(url) = p.get("git").and_then(|g| str_field(g, "repository_url")) {
            s.repo = Some(url.to_owned());
        }
        if let Some(name) = str_field(p, "agent_nickname") {
            s.offer_title(name, TitleSource::AgentName);
        }
        let parent = str_field(p, "parent_thread_id").or_else(|| {
            p.get("source")
                .and_then(|s| s.get("subagent"))
                .and_then(|s| s.get("thread_spawn"))
                .and_then(|t| str_field(t, "parent_thread_id"))
        });
        let forked = str_field(p, "forked_from_id");
        if let Some(parent_id) = parent.and_then(session_id) {
            link(
                parser_state,
                out,
                LinkKind::Spawn,
                parent_id,
                id.clone(),
                None,
                None,
                "parent_thread_id",
            );
        }
        if let Some(fork_id) = forked.filter(|f| Some(*f) != parent).and_then(session_id) {
            link(
                parser_state,
                out,
                LinkKind::Fork,
                fork_id,
                id.clone(),
                None,
                None,
                "forked_from_id",
            );
        }
        if let Some(start) = p
            .get("subagent_history_start_ordinal")
            .and_then(Json::as_u64)
        {
            parser_state.copied_until = Some(start);
            parser_state.copied_from = forked.or(parent).map(str::to_owned);
            if let Some(parent) = parser_state.copied_from.as_deref().and_then(session_id) {
                Self::unknown_boundaries(parent, out);
            }
        }
        RecordOutcome::Used
    }

    fn older_header(&self, parser_state: &mut ParserState, v: &Json) -> RecordOutcome {
        let Some(id) = str_field(v, "id")
            .or(self.thread.as_deref())
            .and_then(session_id)
        else {
            return RecordOutcome::unreadable("header without a thread ID");
        };
        let s = parser_state
            .session
            .get_or_insert_with(|| Session::new(id, SessionKind::Main));
        if let Some(at) = v.get("timestamp").and_then(time::rfc3339) {
            s.offer_start(at);
        }
        if let Some(branch) = v.get("git").and_then(|g| str_field(g, "branch")) {
            s.git_branch = Some(branch.to_owned());
        }
        RecordOutcome::Used
    }

    fn event(
        &self,
        parser_state: &mut ParserState,
        data: &Json,
        kind: &str,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> RecordOutcome {
        let Some(session) = self.own(parser_state) else {
            return RecordOutcome::unreadable("no thread ID in the file name or its header");
        };
        out.events.push(Event {
            session: Some(session),
            kind: kind.to_owned(),
            at,
            after: parser_state.last.clone(),
            range: parser_state.current,
            data: data.clone(),
        });
        RecordOutcome::Used
    }

    fn unanchored_event(
        &self,
        parser_state: &mut ParserState,
        data: &Json,
        kind: &str,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> RecordOutcome {
        let outcome = self.event(parser_state, data, kind, at, out);
        if outcome == RecordOutcome::Used {
            let event = out.events.last_mut().expect("new event");
            event.after = None;
            if let Some(parent) = &parser_state.copied_session {
                event.session = Some(parent.clone());
            }
        }
        outcome
    }

    /// Whether a record belongs to history a subagent copied from its
    /// parent.
    fn copied(parser_state: &ParserState, ordinal: Option<u64>) -> bool {
        matches!((parser_state.copied_until, ordinal), (Some(until), Some(o)) if o > 1 && o < until)
    }

    #[allow(clippy::too_many_arguments)]
    fn response_item(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        item: &Json,
        at: Option<polylogue_model::Timestamp>,
        ordinal: Option<u64>,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let ty = str_field(item, "type").unwrap_or("?");
        records.fields(&format!("response_item/{ty}"), item, fields::for_item(ty));
        let copied = Self::copied(parser_state, ordinal);
        let session = if copied {
            // Stored once, in the parent; only items with an ID can be
            // recognized there.
            let (Some(parent), Some(_)) = (parser_state.copied_from.clone(), str_field(item, "id"))
            else {
                return RecordOutcome::skipped("copied from the parent thread, which stores it");
            };
            match session_id(&parent) {
                Some(s) => s,
                None => return RecordOutcome::unreadable("parent thread ID is empty"),
            }
        } else {
            match self.own(parser_state) {
                Some(s) => s,
                None => {
                    return RecordOutcome::unreadable(
                        "no thread ID in the file name or its header",
                    );
                }
            }
        };
        let native = str_field(item, "id")
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}@{}", session.native(), record.offset));
        let Some(id) = message_id(&native) else {
            return RecordOutcome::unreadable("item ID is empty");
        };
        let Some((role, prompt_kind, blocks)) =
            self.item_blocks(parser_state, &session, ty, item, out)
        else {
            return RecordOutcome::skipped("item with no content");
        };
        let message = Message {
            id: id.clone(),
            session,
            parent: None,
            role,
            at,
            model: (role == Role::Assistant)
                .then(|| parser_state.model.clone())
                .flatten(),
            prompt_kind,
            blocks,
            usage: Usage::default(),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
            production: Production {
                response_boundary_unknown: (role == Role::Assistant).then_some(true),
                effort: (role == Role::Assistant && !copied)
                    .then(|| parser_state.effort.clone())
                    .flatten(),
                ..Production::default()
            },
        };
        let row = MessageRow {
            offset: record.offset,
            end: record.offset + record.bytes.len() as u64,
            message,
        };
        if copied {
            parser_state.branch_point = Some(id);
            out.messages.push(row);
            return RecordOutcome::Used;
        }
        parser_state.last = Some(id);
        out.messages.push(row);
        RecordOutcome::Used
    }

    /// An item's role, prompt kind and blocks; `None` for an item with
    /// nothing in it.
    fn item_blocks(
        &self,
        parser_state: &mut ParserState,
        session: &SessionId,
        ty: &str,
        item: &Json,
        out: &mut Output,
    ) -> Option<(Role, Option<PromptKind>, Vec<Block>)> {
        let call_id = || {
            str_field(item, "call_id")
                .or(str_field(item, "id"))
                .map(str::to_owned)
        };
        Some(match ty {
            "message" => {
                let content = item.get("content").unwrap_or(&Json::Null);
                let blocks = items::message_blocks(content, &mut out.blobs);
                match str_field(item, "role") {
                    Some("user") => (Role::User, Some(items::prompt_kind(content)), blocks),
                    Some("assistant") => (Role::Assistant, None, blocks),
                    _ => (Role::System, None, blocks),
                }
            }
            "agent_message" => {
                let content = item.get("content").unwrap_or(&Json::Null);
                let mut blocks = items::message_blocks(content, &mut out.blobs);
                blocks.insert(
                    0,
                    Block::Other {
                        type_name: "agent_message".into(),
                        data: serde_json::json!({
                            "author": item.get("author"),
                            "recipient": item.get("recipient"),
                        }),
                    },
                );
                (Role::User, Some(PromptKind::Harness), blocks)
            }
            "reasoning" => (Role::Assistant, None, vec![items::reasoning_block(item)]),
            "function_call" | "custom_tool_call" | "local_shell_call" | "tool_search_call" => {
                let tool = items::tool_name(item)
                    .unwrap_or_else(|| ty.trim_end_matches("_call").to_owned());
                let input = items::call_input(item);
                let continues = match tool.as_str() {
                    "write_stdin" => input
                        .get("session_id")
                        .map(|s| format!("process {}", unquote(s))),
                    "wait" => input.get("cell_id").map(|s| format!("cell {}", unquote(s))),
                    _ => None,
                };
                let id = call_id();
                if let Some(id) = &id {
                    parser_state.calls.insert(
                        id.clone(),
                        Call {
                            tool: tool.clone(),
                            continues,
                            ..Call::default()
                        },
                    );
                }
                let block = Block::ToolCall {
                    run_by_script: false,
                    call_id: id,
                    action: items::action(&tool),
                    command: items::call_command(&tool, &input),
                    path: items::call_path(&tool, &input),
                    tool: Some(tool),
                    input,
                };
                (Role::Assistant, None, vec![block])
            }
            "function_call_output" | "custom_tool_call_output" | "tool_search_output" => {
                let id = call_id();
                let raw = item
                    .get("output")
                    .or_else(|| item.get("tools"))
                    .unwrap_or(&Json::Null);
                let (text, attachments) = items::output_parts(raw, &mut out.blobs);
                let (outcome, duration) =
                    self.answer(parser_state, session, id.as_deref(), &text, out);
                let block = Block::ToolResult {
                    call_id: id,
                    outcome,
                    output: Some(text),
                    data: duration.map(|ms| serde_json::json!({ "duration_ms": ms })),
                    attachments,
                };
                (Role::User, Some(PromptKind::Harness), vec![block])
            }
            "web_search_call" => {
                // One item holds both the search and that it ran.
                let id = call_id();
                let outcome = match str_field(item, "status") {
                    Some("completed") => ToolOutcome::Ok,
                    Some("failed") => ToolOutcome::Error { reason: None },
                    _ => ToolOutcome::Unknown {
                        reason: polylogue_model::UnknownReason::StatusFieldUnreadable,
                    },
                };
                let blocks = vec![
                    Block::ToolCall {
                        run_by_script: false,
                        call_id: id.clone(),
                        tool: Some("web_search".into()),
                        action: ToolAction::Web,
                        input: item.get("action").cloned().unwrap_or(Json::Null),
                        command: None,
                        path: None,
                    },
                    Block::ToolResult {
                        call_id: id,
                        outcome,
                        output: None,
                        data: None,
                        attachments: Vec::new(),
                    },
                ];
                (Role::Assistant, None, blocks)
            }
            other => (
                Role::Assistant,
                None,
                vec![Block::Other {
                    type_name: other.to_owned(),
                    data: item.clone(),
                }],
            ),
        })
    }

    /// The outcome of a call's output, and its duration when the harness
    /// recorded one. Updates the call a `write_stdin` or `wait` continues,
    /// and links an agent a `spawn_agent` call started.
    fn answer(
        &self,
        parser_state: &mut ParserState,
        session: &SessionId,
        call_id: Option<&str>,
        text: &str,
        out: &mut Output,
    ) -> (ToolOutcome, Option<u64>) {
        let call = call_id
            .and_then(|id| parser_state.calls.get(id))
            .cloned()
            .unwrap_or_default();
        let tool = if call.tool.is_empty() {
            "?"
        } else {
            call.tool.as_str()
        };
        let (mut outcome, running) = items::output_outcome(tool, text);
        let mut duration = None;
        if let Some(done) = &call.completed {
            outcome = done.outcome.clone();
            duration = done.duration_ms;
        }
        if let (Some(handle), Some(id)) = (running, call_id) {
            parser_state.running.insert(handle.key(), id.to_owned());
        }
        // A later read of a running command reports how it ended.
        if let Some(handle) = &call.continues
            && !matches!(
                outcome,
                ToolOutcome::Unknown {
                    reason: polylogue_model::UnknownReason::RunningInBackground
                }
            )
            && let Some(original) = parser_state.running.remove(handle)
        {
            out.background.push(BackgroundUpdate {
                session: session.clone(),
                task_id: handle.clone(),
                call_id: Some(original),
                outcome: outcome.clone(),
                output_file: None,
            });
        }
        if tool.ends_with("spawn_agent")
            && let Ok(Json::Object(reply)) = serde_json::from_str::<Json>(text.trim())
            && let Some(child) = reply
                .get("agent_id")
                .and_then(Json::as_str)
                .and_then(session_id)
        {
            link(
                parser_state,
                out,
                LinkKind::Spawn,
                session.clone(),
                child,
                None,
                call_id.map(str::to_owned),
                "spawn_agent",
            );
        }
        if let Some(id) = call_id
            && let Some(c) = parser_state.calls.get_mut(id)
        {
            c.answered = true;
        }
        (outcome, duration)
    }

    /// A thread item the harness recorded when it finished.
    fn item_completed(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        p: &Json,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> RecordOutcome {
        let item = p.get("item").unwrap_or(&Json::Null);
        let ty = str_field(item, "type").unwrap_or("?");
        match ty {
            "AgentMessage" | "UserMessage" | "Reasoning" => {
                RecordOutcome::skipped("mirrors the response item, which is stored")
            }
            "ContextCompaction" => {
                RecordOutcome::skipped("mirrors the compacted record, which is stored")
            }
            "CommandExecution" | "FileChange" | "McpToolCall" | "WebSearch" | "ImageView" => {
                let Some(session) = self.own(parser_state) else {
                    return RecordOutcome::unreadable(
                        "no thread ID in the file name or its header",
                    );
                };
                let id = str_field(item, "id").unwrap_or("");
                if let Some(call) = parser_state.calls.get_mut(id) {
                    let Some(done) = items::completed(item) else {
                        return RecordOutcome::skipped("call item without a status");
                    };
                    if call.answered {
                        out.background.push(BackgroundUpdate {
                            session,
                            task_id: id.to_owned(),
                            call_id: Some(id.to_owned()),
                            outcome: done.outcome,
                            output_file: None,
                        });
                    } else {
                        call.completed = Some(done);
                    }
                    return RecordOutcome::Used;
                }
                // A command a code-mode script ran: no call item of its own.
                let Some(blocks) = items::recovered_call(item, &mut out.blobs) else {
                    return RecordOutcome::skipped("tool item without an ID");
                };
                let Some(mid) = message_id(id) else {
                    return RecordOutcome::unreadable("item ID is empty");
                };
                let duration = items::completed(item).and_then(|c| c.duration_ms);
                out.messages.push(MessageRow {
                    offset: record.offset,
                    end: record.offset + record.bytes.len() as u64,
                    message: Message {
                        id: mid.clone(),
                        session,
                        parent: None,
                        role: Role::Assistant,
                        at,
                        model: None,
                        prompt_kind: Some(PromptKind::Harness),
                        blocks,
                        usage: Usage::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: duration,
                        production: Production::default(),
                    },
                });
                parser_state.last = Some(mid);
                RecordOutcome::Used
            }
            "SubAgentActivity" => {
                if let (Some(parent), Some(child)) = (
                    self.own(parser_state),
                    str_field(item, "agent_thread_id").and_then(session_id),
                ) {
                    let call = str_field(item, "id").map(str::to_owned);
                    link(
                        parser_state,
                        out,
                        LinkKind::Spawn,
                        parent,
                        child,
                        None,
                        call,
                        "SubAgentActivity",
                    );
                }
                self.event(parser_state, p, "subagent_activity", at, out)
            }
            other => {
                let kind = format!("item_{}", snake(other));
                self.event(parser_state, p, &kind, at, out)
            }
        }
    }

    fn event_msg(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        p: &Json,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> RecordOutcome {
        let sub = str_field(p, "type").unwrap_or("?");
        match sub {
            "token_count" => {
                // Keep the native observation as reported, including malformed,
                // rate-only and context-size reports. Its time is emission time.
                self.unanchored_event(parser_state, p, "token_count", at, out);
                let Some(session) = self.own(parser_state) else {
                    return RecordOutcome::unreadable(
                        "no thread ID in the file name or its header",
                    );
                };
                let session = parser_state.copied_session.clone().unwrap_or(session);
                let report_id = usage::identity(&(session, at, p));
                if let Some(info) = p.get("info").filter(|info| !info.is_null())
                    && let Some(reading) = parser_state.usage.read(
                        info,
                        report_id,
                        parser_state.model.clone(),
                        parser_state.tier.clone(),
                    )
                {
                    self.unanchored_event(
                        parser_state,
                        &serde_json::to_value(reading).expect("usage reading"),
                        polylogue_model::reports::USAGE_REPORT,
                        at,
                        out,
                    );
                }
                if let Some(reading) = p.get("rate_limits").and_then(quota) {
                    let identity = usage::identity(&reading);
                    if parser_state.limits.as_ref() != Some(&identity) {
                        parser_state.limits = Some(identity.clone());
                        let reading = polylogue_model::QuotaReport {
                            report_id: identity,
                            reading,
                            measured_at: None,
                        };
                        self.unanchored_event(
                            parser_state,
                            &serde_json::to_value(reading).expect("quota reading"),
                            polylogue_model::reports::QUOTA_REPORT,
                            at,
                            out,
                        );
                    }
                }
                RecordOutcome::Used
            }
            "item_completed" => self.item_completed(parser_state, record, p, at, out),
            "agent_message"
            | "agent_reasoning"
            | "agent_reasoning_raw_content"
            | "agent_reasoning_section_break" => {
                RecordOutcome::skipped("mirrors a response item, which is stored")
            }
            "exec_command_end" | "patch_apply_end" | "mcp_tool_call_end" => {
                // Older rollouts: how a call ended, before item_completed.
                let id = str_field(p, "call_id").unwrap_or("");
                if let (Some(call), Some(done)) =
                    (parser_state.calls.get_mut(id), items::completed(p))
                {
                    if call.answered
                        && let Some(session) = parser_state.session.as_ref().map(|s| s.id.clone())
                    {
                        out.background.push(BackgroundUpdate {
                            session,
                            task_id: id.to_owned(),
                            call_id: Some(id.to_owned()),
                            outcome: done.outcome,
                            output_file: None,
                        });
                    } else {
                        call.completed = Some(done);
                    }
                }
                self.event(parser_state, p, sub, at, out)
            }
            "task_complete" | "turn_aborted" => {
                // Calls do not outlive their turn; commands still running do.
                parser_state.calls.clear();
                self.event(parser_state, p, sub, at, out)
            }
            "thread_settings" => {
                if let Some(tier) = p
                    .get("thread_settings")
                    .and_then(|s| str_field(s, "service_tier"))
                {
                    parser_state.tier = Some(tier.to_owned());
                }
                self.event(parser_state, p, sub, at, out)
            }
            "thread_name_updated" => {
                if let (Some(name), Some(s)) =
                    (str_field(p, "thread_name"), parser_state.session.as_mut())
                {
                    s.offer_title(name, TitleSource::Generated);
                }
                self.event(parser_state, p, sub, at, out)
            }
            _ => self.event(parser_state, p, sub, at, out),
        }
    }
}

/// The text of a user message item, for pairing it with a `user_message`
/// event that records the same prompt.
fn user_text(item: &Json) -> Option<String> {
    if str_field(item, "type") != Some("message") || str_field(item, "role") != Some("user") {
        return None;
    }
    let Json::Array(parts) = item.get("content")? else {
        return None;
    };
    Some(
        parts
            .iter()
            .filter_map(|c| str_field(c, "text"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn unquote(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `CollabAgentToolCall` as `collab_agent_tool_call`.
fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn link(
    parser_state: &mut ParserState,
    out: &mut Output,
    kind: LinkKind,
    parent: SessionId,
    child: SessionId,
    at: Option<MessageId>,
    call_id: Option<String>,
    evidence: &str,
) {
    if parent == child {
        return;
    }
    let key = format!(
        "{kind:?} {parent} {child} {}",
        call_id.as_deref().unwrap_or("")
    );
    if parser_state.links.insert(key) {
        out.links.push(Link {
            kind,
            parent,
            child,
            at,
            call_id,
            evidence: evidence.to_owned(),
        });
    }
}

impl Parser for Rollout {
    type ParserState = ParserState;

    fn kind(&self, v: &Json) -> String {
        let rec = unwrap(v);
        match rec.ty {
            "response_item" | "event_msg" => {
                format!(
                    "{}/{}",
                    rec.ty,
                    str_field(rec.payload, "type").unwrap_or("?")
                )
            }
            ty => ty.to_owned(),
        }
    }

    fn parse_record(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        if !v.is_object() {
            return RecordOutcome::unreadable("record is not a JSON object");
        }
        let rec = unwrap(v);
        let p = rec.payload;
        parser_state.current = Some((record.offset, record.offset + record.bytes.len() as u64));
        if record.offset == 0 {
            parser_state.usage.start(0, false);
        }
        parser_state.copied_session = Self::copied(parser_state, rec.ordinal)
            .then(|| parser_state.copied_from.as_deref().and_then(session_id))
            .flatten();
        let token_observation =
            rec.ty == "event_msg" && str_field(p, "type") == Some("token_count");
        if Self::copied(parser_state, rec.ordinal)
            && token_observation
            && parser_state.copied_from.is_none()
        {
            if let Some(info) = p.get("info").filter(|info| !info.is_null()) {
                // Advance the observed counters without charging an unknown
                // original owner as usage of the child.
                parser_state
                    .usage
                    .read(info, usage::identity(p), None, None);
            }
            return self.unanchored_event(parser_state, p, "copied_token_count", rec.at, out);
        }
        if Self::copied(parser_state, rec.ordinal)
            && rec.ty != "response_item"
            && !token_observation
        {
            return RecordOutcome::skipped("copied from the parent thread, which stores it");
        }
        let prev = parser_state.prompt.take();
        if rec.ty == "response_item" && str_field(p, "type") == Some("token_count") {
            return self.unanchored_event(parser_state, p, "token_count_item", rec.at, out);
        }
        let same = |text: &str, from_event: bool| {
            prev.as_ref()
                .is_some_and(|p| p.from_event == from_event && p.text == text)
        };
        if rec.ty == "response_item"
            && let Some(text) = user_text(p)
        {
            if same(&text, true) {
                return RecordOutcome::skipped(
                    "same prompt as the user_message event before it, which is stored",
                );
            }
            let d = self.response_item(parser_state, record, p, rec.at, rec.ordinal, out, records);
            if d == RecordOutcome::Used {
                parser_state.prompt = Some(Prompt {
                    text,
                    from_event: false,
                });
            }
            return d;
        }
        if rec.ty == "event_msg" && str_field(p, "type") == Some("user_message") {
            let text = str_field(p, "message").unwrap_or_default().to_owned();
            if same(&text, false) {
                return RecordOutcome::skipped(
                    "same prompt as the user message item before it, which is stored",
                );
            }
            // A prompt only the event recorded: it is the prompt.
            let mut content = vec![serde_json::json!({"type": "input_text", "text": text})];
            if let Some(Json::Array(images)) = p.get("images") {
                content.extend(
                    images
                        .iter()
                        .filter_map(Json::as_str)
                        .map(|url| serde_json::json!({"type": "input_image", "image_url": url})),
                );
            }
            let mut item =
                serde_json::json!({"type": "message", "role": "user", "content": content});
            if let Some(id) = str_field(p, "client_id") {
                item["id"] = Json::String(id.to_owned());
            }
            let d = self.response_item(
                parser_state,
                record,
                &item,
                rec.at,
                rec.ordinal,
                out,
                records,
            );
            if d == RecordOutcome::Used {
                parser_state.prompt = Some(Prompt {
                    text,
                    from_event: true,
                });
            }
            return d;
        }
        match rec.ty {
            "session_meta" => self.session_meta(parser_state, p, out),
            "older_header" => self.older_header(parser_state, p),
            "state" => {
                RecordOutcome::skipped("turn-state marker of 2025 rollouts; holds no content")
            }
            "response_item" => {
                self.response_item(parser_state, record, p, rec.at, rec.ordinal, out, records)
            }
            "event_msg" => self.event_msg(parser_state, record, p, rec.at, out),
            "turn_context" => {
                if let Some(model) = str_field(p, "model") {
                    parser_state.model = Some(model.to_owned());
                }
                parser_state.effort = str_field(p, "effort").map(str::to_owned);
                if let Some(tier) = str_field(p, "service_tier") {
                    parser_state.tier = Some(tier.to_owned());
                }
                if let (Some(cwd), Some(s)) = (str_field(p, "cwd"), parser_state.session.as_mut()) {
                    s.offer_cwd(cwd);
                }
                // Settings repeat every turn; only a change is stored.
                let mut settings = p.clone();
                if let Some(m) = settings.as_object_mut() {
                    m.remove("turn_id");
                    m.remove("root_turn_id");
                }
                let d = digest(&settings);
                if parser_state.settings == Some(d) {
                    return RecordOutcome::Used;
                }
                parser_state.settings = Some(d);
                self.event(parser_state, p, "turn_context", rec.at, out)
            }
            "token_usage_record" => {
                self.unanchored_event(parser_state, p, "token_usage_record", rec.at, out)
            }
            "compacted" => {
                // A fork starts from its parent's history: the last
                // identified item is where it branched. Keep that history
                // as native evidence, without creating authored messages.
                if parser_state.last.is_none()
                    && let Some(history) = p.get("replacement_history").and_then(Json::as_array)
                    && let Some(at) = history.iter().rev().find_map(|i| str_field(i, "id"))
                {
                    parser_state.branch_point = message_id(at);
                }
                self.event(parser_state, p, "compaction", rec.at, out)
            }
            "world_state" | "inter_agent_communication_metadata" => {
                self.event(parser_state, p, rec.ty, rec.at, out)
            }
            _ => RecordOutcome::skipped("unrecognized record type"),
        }
    }

    fn flush(&self, parser_state: &mut ParserState, out: &mut Output) {
        if !parser_state.scope_recorded
            && let Some(session) = self.own(parser_state)
        {
            Self::unknown_boundaries(session, out);
            parser_state.scope_recorded = true;
        }
        // The branch point is known once the copied history has been read.
        if let (Some(at), Some(s)) = (&parser_state.branch_point, &parser_state.session) {
            for l in &mut out.links {
                if l.child == s.id && l.at.is_none() && l.kind != LinkKind::Copy {
                    l.at = Some(at.clone());
                }
            }
        }
        out.sessions = parser_state.session.iter().cloned().collect();
    }
}

#[cfg(test)]
mod tests;
